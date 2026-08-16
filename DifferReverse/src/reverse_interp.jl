# Reverse-mode AD: branches and loops, Mooncake style.
#
# Two separately-compiled carriers, wired through the same `build_contextual_ir` override as forward
# mode (`ContextualInterpreter{Reverse}`):
#
#   * `reverse_fwds_impl(codualargs::CoDual...) -> (result::CoDual, tape::Tape)` — replays the primal
#     computation forward (ordinary value recomputation, no shadow/tangent) and, wherever control
#     flow is ambiguous, instruments it: pushes the current block number onto a shared `Stack{Int32}`
#     (the block stack) so the pullback can replay control flow in exact reverse order, and pushes
#     any forward-computed operand values a rule in that block will need onto a per-block comms
#     `Stack`.
#   * `reverse_pullback_impl(tape::Tape, seed) -> rdata_tuple` — walks the primal's blocks in reverse
#     (over a fresh CFG built via the `ID`/`CFGBlock` layer in `cfg_ir.jl`, since the pullback
#     inserts phi-routing blocks and lowers multi-way dispatches into `GotoIfNot` chains), popping
#     the block stack to know which predecessor fired and popping each block's comms stack to
#     recover that visit's forward values, accumulating rdata into per-SSA/per-argument `Ref`s.
#
# Unlike Mooncake (two `OpaqueClosure`s sharing captured state), the two passes here are ordinary
# `CodeInstance`s and the shared state is an explicit `Tape` value: `reverse_fwds_impl` returns it,
# `reverse_pullback_impl` takes it as an argument.
#
# `rrule`/`rev_gradient` (bottom of this file) are plain, uncompiled Julia: they hold the original
# argument fdata (from `zero_fcodual`) and combine it with the rdata the pullback carrier returns via
# `tangent(fdata, rdata)` — the pullback carrier itself only ever returns rdata.
#
# Scope: branches and loops (any number of back-edges, any number of reachable exit blocks — Julia's
# optimizer commonly keeps a separate `return` per branch arm rather than merging via a phi, so
# multi-exit is the common case, not a corner case). A block pushes to (forwards) / pops from
# (pullback) the block stack only when its predecessor identity is actually ambiguous (see
# `_unique_predecessor_info`), so straight-line code emits zero block-stack traffic. Data-wise:
# scalar float arithmetic intrinsics, immutable/mutable struct fields and array mutation (via the
# shadow-chain comms scheme in `builtins_reverse.jl`), statically-resolvable recursive calls
# (concrete callee + concrete, trivial-fdata args — see `_static_recursible_call`; a hand-written
# rule always takes priority over raw recursion), direct self-recursion (see
# `reverse_fwds_recursive_ci`), and array indexing via a provenance chain traceable to a function
# argument (see `_fdata_tracked`). Still out of scope, bails cleanly: dynamic-dispatch recursion,
# mutual recursion (needs a tape-type pre-pass across the whole SCC), try/catch.

_codual_primal_type(@nospecialize P) = fieldtype(P, 1)
_codual_fdata_type(@nospecialize P) = fieldtype(P, 2)

# ===========================================================================
# The rule interface.
#
# Everything is called on `CoDual`s and returns `(ycd, pullback)`; the pullback is called on an rdata
# seed and returns the tuple of argument rdatas. The pullback closure *is* the tape — there is no
# separate tape value threaded between two free functions. For a derived (compiler-generated) rule
# that closure is a `Tape` (below), holding the block stack and the per-block comms stacks; for a
# hand-written rule it's whatever's cheapest to remember (`src/rrules.jl`).
#
# `rrule!!` is a single stateless generic function under one uniform convention:
#
#     rrule!!(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) -> (ycd, pullback)
#
# Both flavours of rule are methods of it: hand-written primitives (`src/rrules.jl`, a method for a
# specific `fcd`/`argcds` shape) and the derived path (a single `@generated` fallback that transforms
# a composite primal's IR). The tape lives in the `ctx` argument, not the rule, so `rrule!!` itself is
# stateless and shareable — reentrancy is "one `Ctx` per task", not "one rule per task".
#
# The fallback is ambiguity-free against hand rules because `ctx` is dispatch-neutral: every method
# declares it `::AbstractCtx`, never a concrete subtype, so specificity is decided purely by the fcd +
# arg slots and the fallback (strictly least specific) never wins. Consequence: "does `rrule!!`
# resolve?" no longer means "is there a hand rule?" (the fallback always resolves), so
# `hand_reverse_rule_match` must recognize the fallback by signature and report "no hand rule" when
# that's what matched.
#
# RULE-AUTHORING CONSTRAINT: a hand rule's `ctx` slot must stay `::AbstractCtx` — a concrete ctx
# subtype would bring the ambiguity back.
# ===========================================================================

"""
    rrule!!(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) -> (ycd, pullback)

Reverse-mode rule for `primal(fcd)(primal.(argcds)...)`, returning the result as a `CoDual` plus a
pullback. Call the pullback with an rdata seed for the result to get the tuple of rdatas for
`(f, args...)`; fdata-carried gradients (arrays, mutable structs) are accumulated in place into the
`CoDual`s' own shadows instead.

Hand-written primitives (see `src/rrules.jl` for the `sin`/`cos` rules and the shape to follow) are
methods with a specific `fcd`/`argcds` shape; a composite function is handled by an `@generated`
fallback that derives the rule from `f`'s IR. `ctx::`[`AbstractCtx`](@ref) carries the tape (build a
reusable one with [`build_ctx`](@ref)); a hand rule that needs no tape ignores it. A hand rule
**must** declare its `ctx` slot as `::AbstractCtx` (never a concrete subtype) — that's what keeps
dispatch against the fallback unambiguous.

Contract for an fdata-carrying result (an array or mutable struct): the returned `CoDual`'s shadow
must be the exact object the rule's own pullback reads back, never a detached copy — a caller may
accumulate into it in place before the pullback runs. Every hand-written array rule already
follows this (`rules_broadcast.jl`, `rules_reductions.jl`, `rules_indexing.jl`, `rules_linalg.jl`).
"""
function rrule!! end

"""
    AbstractCtx

Supertype of reverse-mode differentiation contexts — the argument [`rrule!!`](@ref) threads its
per-call/per-task state through, chiefly the tape. [`Ctx`](@ref) is the default. Every `rrule!!`
method dispatches this slot as `::AbstractCtx` (never a concrete subtype), which keeps the
derived-fallback-vs-hand-rule dispatch ambiguity-free.
"""
abstract type AbstractCtx end

"""
    Ctx{P} <: AbstractCtx

The default differentiation context, carrying a pre-allocated tape in `tape::P`.

`P === Nothing` means "allocate a fresh tape on every call" — the simple, stateless mode. A
hand-written rule's callee always gets this (its pullback isn't a `Tape`, so there's nothing to
pre-allocate for it). A derived (non-hand-ruled) nested or recursive inner call instead gets a
*recycled* tape: its pullback is a value pushed onto the outer block's comms `Stack` once per
execution, and a `Stack` never deallocates, so after the first execution the slot the next push lands
in already holds a structurally identical tape — the emission site hands that one to the callee (via
`Ctx{P}` with a concrete `P`, not `Ctx{Nothing}`) instead of allocating fresh (`_inner_ctx`,
`src/stack.jl`). Any `P` other than `Nothing` is this shape: a tape whose stacks are reset and reused
per call — either the caller's own top-level tape (`build_ctx(...; prealloc=true)`) or a callee's
recycled one. Build a top-level one with [`build_ctx`](@ref).
"""
struct Ctx{P} <: AbstractCtx
    tape::P
end
Ctx() = Ctx(nothing)

# ===========================================================================
# The tape — also the pullback closure for the generated fallback.
# `ArgsTT` is a `Tuple` of the primal's `CoDual` argument types, carried so a `Tape`'s own call
# specialization can recover which primal method it belongs to (mirroring how the forwards carrier's
# `specTypes` names it directly). `CS` is a `Tuple` of per-primal-block comms-stack types (`Stack{T}`
# for a block with something to communicate, `SingletonStack{Tuple{}}` for one with nothing —
# mirroring Mooncake's `SharedDataPairs`/singleton-type optimization).
# ===========================================================================
mutable struct Tape{ArgsTT<:Tuple,CS<:Tuple}
    const block_stack::Stack{Int32}
    const comms::CS
    # The primal's `(fcd, argcds...)` coduals, stored by the forwards pass so the pullback can reach
    # argument values: `reverse_pullback_impl`'s signature is just `(tape, seed)`, so an `Argument(k)`
    # of the primal would otherwise be unavailable to it.
    #
    # Non-`const` because `build_ctx` allocates the tape knowing only the argument types — the values
    # arrive later, on each call. The other two fields stay `const`: the pullback reads them once and
    # hoists, which a mutable field would inhibit.
    # Reusable buffers for bulk primal save/restore, one slot per bulk-saved argument
    # (`_bulk_save_args`). `const`: the `Vector` object is fixed for the tape's life, only its
    # contents change, letting a pre-allocated context reuse the buffers instead of allocating a copy
    # per call. `_NO_BULK_BUFS` (a shared empty) when this primal saves nothing.
    const bufs::Vector{Any}
    # Storage for a direct self-recursive call's own inner tape — always exactly `Tape{ArgsTT,CS}`
    # itself, since a self-recursive callee's tape type is by construction identical to the caller's.
    # Ordinary recursive-struct self-reference (same mechanism as `next::Union{Nothing,Node{T}}`
    # inside `Node{T}`), not an attempt to solve `CS`'s own fixed point. Kept outside `CS` so `CS`
    # only ever enumerates genuinely non-self-recursive per-block comms.
    #
    # One shared stack serves every direct self-recursive call site in this primal — they all target
    # this identical type, so there's no per-site heterogeneity to partition around. Pushed in the
    # fwds pass's program order and popped in the pullback's exact reverse order, so the stack stays
    # LIFO-consistent regardless of which site or block each entry came from.
    #
    # Declared before `args` so the 3-argument constructor can leave only `args` undef. Always
    # constructed, even with no self-recursive call (a fresh-tape-construction cost, not a
    # steady-state one).
    const subtapes::Stack{Tape{ArgsTT,CS}}
    args::ArgsTT
    Tape{ArgsTT,CS}(block_stack, comms, bufs=_NO_BULK_BUFS,
                    subtapes=Stack{Tape{ArgsTT,CS}}()) where {ArgsTT<:Tuple,CS<:Tuple} =
        new{ArgsTT,CS}(block_stack, comms, bufs, subtapes)    # `args` deliberately left undef
    Tape{ArgsTT,CS}(block_stack, comms, bufs, subtapes, args) where {ArgsTT<:Tuple,CS<:Tuple} =
        new{ArgsTT,CS}(block_stack, comms, bufs, subtapes, args)
end

# `Tape` needs a hand-written `tangent_type` under forward-over-reverse: it's where the forward
# pass's saved primal values live, so a `Tape` flowing through an outer `D` needs a real shadow, not
# `NoTangent` (which would silently zero the second derivative). The generic per-field derivation
# can't be used: `subtapes::Stack{Tape{ArgsTT,CS}}` makes `Tape` self-referential, so re-deriving
# `Tape`'s tangent by re-deriving its own fields doesn't terminate.
#
# Map the type parameters directly instead: the shadow of `Tape{ArgsTT,CS}` is `Tape{ArgsTT',CS'}`,
# each mapped element-wise through `tangent_type`. This terminates the same way `Tape`'s own struct
# definition does — the self-reference is untouched by the mapping.
#
# Payoff: the shadow of a `Tape` *is* a `Tape` and the shadow of a `Stack` *is* a `Stack`, so a fresh
# shadow tape builds via the same `_alloc_tape`/`_fresh_tape_expr` code as a fresh primal one.
#
# `_tuple_tangent_types` (unlike the ordinary tuple-of-values `tangent_type`, which collapses an
# all-`NoTangent` tuple to a bare `NoTangent`) maps both `ArgsTT` and `CS` element-wise
# unconditionally — needed since an all-non-differentiable-argument primal (e.g.
# `rev_gradient(f, 3)` for `f(::Int)`) is exactly the all-`NoTangent` case the ordinary collapse
# would break.
_tuple_tangent_types(::Type{T}) where {T<:Tuple} = Tuple{(tangent_type(t) for t in T.parameters)...}

@foldable tangent_type(::Type{Tape{ArgsTT,CS}}) where {ArgsTT<:Tuple,CS<:Tuple} =
    Tape{_tuple_tangent_types(ArgsTT),_tuple_tangent_types(CS)}

# A fresh shadow tape is exactly what `_alloc_tape` already builds from a `Tape` type alone —
# nothing in `x`'s current state is worth copying, since every stack in a `Tape` starts empty
# regardless of what the primal's stacks currently hold. Cached (like `MutableTangent`'s
# `zero_tangent_internal`) so the same primal `Tape` object reached twice gets the same shadow
# instance both times.
function zero_tangent_internal(x::Tape, d::MaybeCache)
    TapeT = tangent_type(typeof(x))
    haskey(d, x) && return d[x]::TapeT
    t = _alloc_tape(TapeT)
    d[x] = t
    return t
end

# Resets every stack's `position` to 0 in place, mirroring the primal-tape reset
# `reverse_fwds_to_ircode` emits for reuse, not reallocating.
function set_to_zero_internal!!(c::SetToZeroCache, x::Tape)
    x.block_stack.position = 0
    tuple_map(s -> set_to_zero_internal!!(c, s), x.comms)
    x.subtapes.position = 0
    return x
end

# Two layers, as in forward mode (`frule!!` entry / `dualized_impl` carrier):
#
#   * The *entry* is the public surface — the `@generated` fallback method of `rrule!!` and the
#     `@generated` pullback callable `(t::Tape)(seed)`. Both compile the corresponding carrier under
#     `ContextualInterpreter{Reverse}` and emit a static `:invoke` to the result. The fwds entry is a
#     straight pass-through, invoking the carrier with exactly its own `(fcd, ctx, argcds...)`.
#   * The *carrier* is the hidden function whose specializations `build_contextual_ir` actually
#     transforms. Its body below is the stub that runs only if the transform bailed and the more
#     specific `reverse_error_ircode` wasn't installed.
#
# `ctx` is a real argument of the fwds carrier — that's how a `Ctx` hands its pre-allocated stacks to
# the generated body.

reverse_fwds_impl(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) =
    error("Differ.reverse_fwds_impl ran directly: ContextualInterpreter could not build the reverse " *
          "forwards pass (likely control flow Differ doesn't support yet, a mutable/undef-field " *
          "struct, a surviving high-level call, or an intrinsic with no registered reverse rule).")

reverse_pullback_impl(tape, seed) =
    error("Differ.reverse_pullback_impl ran directly: ContextualInterpreter could not build the reverse " *
          "pullback pass.")

# Is `mi` a *carrier* specialization — the thing `build_contextual_ir` transforms? The carrier is
# `reverse_fwds_impl(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...)`, so `ctx` is at `params[3]`.
# The `isa(mi.specTypes, DataType)` guard is load-bearing, not defensive: this runs on every
# MethodInstance the interpreter infers, and a `specTypes` with free typevars is a `UnionAll`, whose
# `.parameters` throws a `FieldError` straight out of `finishinfer!` (see `is_dualized_impl`,
# `forward_interp.jl`, for the same guard and a concrete offender). A carrier is always concrete.
is_reverse_fwds_impl(mi) = isa(mi.def, Method) && isa(mi.specTypes, DataType) &&
                          length(mi.specTypes.parameters) >= 3 &&
                          mi.specTypes.parameters[1] === typeof(reverse_fwds_impl) &&
                          mi.specTypes.parameters[2] <: CoDual &&
                          mi.specTypes.parameters[3] <: AbstractCtx
is_reverse_pullback_impl(mi) = isa(mi.def, Method) && isa(mi.specTypes, DataType) &&
                              length(mi.specTypes.parameters) >= 2 &&
                              mi.specTypes.parameters[1] === typeof(reverse_pullback_impl) &&
                              mi.specTypes.parameters[2] <: Tape

# Two unsupported compositions, both non-goals: reverse-mode differentiation of a forward-mode call
# (`frule!!`/`dualized_impl`) or of a reverse-mode call (`rrule!!`/`reverse_fwds_impl`/
# `reverse_pullback_impl`). Recognized as soon as a callee's primal function type is known, at the
# two chokepoints reverse mode resolves a callee's primal from (`has_hand_reverse_rule` and
# `resolve_reverse_primal` below) — before a `Dual` argument reaches `fcodual_type`/`fdata_type`
# (an unlocated "Unhandled type" error otherwise) and before the derived path tries to differentiate
# through `rrule!!`'s own `(ycd, pullback)` return.
#
# Coupling-point hook: `frule!!`/`dualized_impl` are DifferForwards-owned names, so DifferReverse
# can't reference them directly. Names the forward-mode entry point `ftype` is, if any, purely for a
# more helpful bail message. Default `nothing` — in a standalone `DifferReverse` this composition
# can't arise, so the generic `_is_foreign_forward_carrier`-triggered message still fires. Overridden
# in `DifferForwardsOverReverseExt` once both packages are loaded.
_forward_entry_name(@nospecialize(ftype)) = nothing

# `tangent_type`/`fcodual_type` at a given inference world. Every transform-time query goes through
# one of these rather than calling the function directly: dispatch inside a `@generated` generator is
# pinned to the generator's own definition world, so a `tangent_type` method owned by a later-loaded
# package (including this package's own, seen from a nested forward-over-reverse build) would be
# invisible. See Contextual's world-age contract.
_tt(world::UInt, @nospecialize P) = at_world(world, tangent_type, P)
_fcdtype(world::UInt, @nospecialize P) = at_world(world, fcodual_type, P)
_reverse_entry_name(@nospecialize(ftype)) =
    ftype === typeof(rrule!!)              ? "rrule!!" :
    ftype === typeof(reverse_fwds_impl)     ? "reverse_fwds_impl" :
    ftype === typeof(reverse_pullback_impl) ? "reverse_pullback_impl" : nothing

function _composition_bail_message(world::UInt, @nospecialize(ftype))
    fname = at_world(world, _forward_entry_name, ftype)
    fname !== nothing && return "reverse-over-forward is not supported: encountered a call to " *
        "`Differ.$(fname)` (forward-mode differentiation) while building a reverse-mode pass"
    rname = _reverse_entry_name(ftype)
    rname !== nothing && return "reverse-over-reverse is not supported: encountered a call to " *
        "`Differ.$(rname)` (reverse-mode differentiation) while building a reverse-mode pass"
    return nothing
end

# The `@generated` derived fallback — the least-specific `rrule!!` method. Recognized by its exact
# signature so `hand_reverse_rule_match` can tell "matched a hand rule" from "matched the fallback":
# a `findsup` on a concrete query always resolves some method now that the fallback exists.
is_generated_reverse_fwds_fallback(m::Method) =
    m.sig === Tuple{typeof(rrule!!),CoDual,AbstractCtx,Vararg{CoDual}}

# The `rrule!!` signature a *hypothetical* reverse-mode differentiation of `callee_mi` would resolve
# against, built from `callee_mi.specTypes` rather than an actual differentiation call site. Returns
# `nothing` for anything the shape doesn't apply to, rather than throwing: `callee_mi` is an
# arbitrary callee discovered via `frame.edges`, not something a call site validated.
function implicit_rrule_tt(world::UInt, callee_mi::MethodInstance)
    isa(callee_mi.def, Method) || return nothing
    isa(callee_mi.specTypes, DataType) || return nothing   # `UnionAll` sig — see `is_reverse_fwds_impl`
    params = callee_mi.specTypes.parameters
    isempty(params) && return nothing
    ftype = params[1]
    (ftype isa Type) || return nothing
    try
        # `fcodual_type(P)` reaches `tangent_type(P)`, which throws for non-`Type` `P` and can
        # `MethodError` on a `P` this tangent system has no case for. Best-effort: skip this one
        # callee's implicit backedge rather than abort the whole build over it.
        argcodualtys = Any[_fcdtype(world, P) for P in params[2:end]]
        return Tuple{typeof(rrule!!),CoDual{ftype,NoFData},Ctx{Nothing},argcodualtys...}
    catch
        return nothing
    end
end

# An mt-backedge on the `rrule!!` resolution a hypothetical differentiation of `callee_mi` would use,
# registered even though `callee_mi`'s call was (or may have been) inlined away and never actually
# reached `reverse_fwds_recursive_ci`'s resolution. So a user later hand-writing `rrule!!` for a
# callee that was inlined away before a rule existed still invalidates a derivative built earlier.
function register_implicit_rrule_backedge!(world::UInt, edges::Vector{Any}, callee_mi::MethodInstance)
    rrule_tt = implicit_rrule_tt(world, callee_mi)
    rrule_tt === nothing || push!(edges, rrule_tt, Core.methodtable)
    return nothing
end

# Resolve the hand-written `rrule!!` for a call, or `nothing` if only the derived fallback applies.
# The query's `ctx` slot is `Ctx{Nothing}` (the fresh-tape mode a recursive inner call uses); every
# method — hand rule or fallback — declares that slot `::AbstractCtx`, so it never affects which
# method wins, only the fcd + arg slots do.
#
# `tt` calls tangent_type on every argtype, unsafe for arbitrary callees `src_inlining_policy`
# probes: the generic-struct fallback isn't guaranteed to terminate on a self-referential type
# (`DifferCore/src/tangents.jl`). `has_hand_reverse_rule` guards via `_hand_rule_ftype_candidate` first.
function hand_reverse_rule_match(interp::ContextualInterpreter, @nospecialize(ftype), argtypes)
    tt = Tuple{typeof(rrule!!),CoDual{ftype,NoFData},Ctx{Nothing},(_fcdtype(interp.world, P) for P in argtypes)...}
    m, _ = CC.findsup(tt, CC.method_table(interp))
    (m === nothing || !isa(m.method, Method)) && return nothing
    is_generated_reverse_fwds_fallback(m.method) && return nothing
    return tt, m
end

# Cheap pre-filter for `has_hand_reverse_rule`: abstract `Vararg{Any}` probe for whether any
# hand-written `rrule!!` could match this `ftype`, without calling tangent_type. Its signature is a
# superset of every concrete `tt` `hand_reverse_rule_match` builds, so skipping here is safe when
# only the fallback matches.
function _hand_rule_ftype_candidate(interp::ContextualInterpreter, @nospecialize(ftype))
    loose_tt = Tuple{typeof(rrule!!),CoDual{ftype,NoFData},AbstractCtx,Vararg{Any}}
    matches = CC.findall(loose_tt, CC.method_table(interp))
    matches === nothing && return true   # unbounded lookup returned "too many" — be conservative
    for m in matches
        is_generated_reverse_fwds_fallback(m.method) || return true
    end
    return false
end

# Does a hand-written `rrule!!` apply to a hypothetical reverse-mode differentiation of `callee_mi`?
# Used by `src_inlining_policy` below to keep such a call from being inlined away before it reaches
# `_static_recursible_call`'s recursion dispatch, regardless of how cheap the callee looks to
# Julia's ordinary cost heuristic.
#
# Composition check here too: called from `src_inlining_policy`, which has no `reason::Ref` to bail
# through, so an unsupported composition throws directly instead of returning `false` — a `Dual`
# argument would otherwise reach `fcodual_type` and crash with an unrelated "Unhandled type" error.
function has_hand_reverse_rule(interp::ContextualInterpreter, callee_mi::MethodInstance)
    isa(callee_mi.def, Method) || return false
    isa(callee_mi.specTypes, DataType) || return false     # `UnionAll` sig — see `is_reverse_fwds_impl`
    params = callee_mi.specTypes.parameters
    isempty(params) && return false
    ftype = params[1]
    (ftype isa Type && isconcretetype(ftype)) || return false
    msg = _composition_bail_message(interp.world, ftype)
    msg === nothing || error(msg)
    argtypes = params[2:end]
    all(P -> P isa Type && isconcretetype(P), argtypes) || return false
    # A `Dual` argument (not just a `Dual` callee) is reverse-over-forward too, and the more common
    # shape in practice: `frule!!`'s hand rules are small enough that ordinary inlining absorbs the
    # `frule!!` call itself before this function ever sees it as a callee, leaving a surviving call
    # (e.g. `getproperty(::Dual{Float64,Float64}, :x)`, pulling `.x` back out) whose *argument* is the
    # `Dual`. Caught here, before `hand_reverse_rule_match` reaches `fcodual_type` on it below.
    any(P -> P isa Type && at_world(interp.world, _is_foreign_forward_carrier, P), argtypes) &&
        error("reverse-over-forward is not supported: encountered a `Dual` (forward-mode carrier) " *
              "argument while building a reverse-mode pass")
    # Skip callees no hand rule could apply to before touching argtypes' tangent_type.
    _hand_rule_ftype_candidate(interp, ftype) || return false
    return hand_reverse_rule_match(interp, ftype, argtypes) !== nothing
end

# Is `mi` itself a `reverse_fwds_impl`/`reverse_pullback_impl` specialization — the target of one of
# Part 1's recursive `:invoke`s (`reverse_fwds_recursive_ci`/`reverse_pullback_recursive_ci`), hand-
# written rule or generated fallback alike? A hand rule's body is typically small (e.g. one `sin(x)`
# call) — small enough that Julia's ordinary cost heuristic would happily inline it, unlike the
# generated fallback (always sizable) or forward mode's `frule!!` hand rules for the same functions
# (which call two transcendentals — `Dual(sin(x), cos(x)*dx)` — comfortably over the inlining
# threshold, so this hazard never arose there). Inlining one back into its recursive caller is never
# correct here: the inlined statements carry GlobalRefs resolved relative to the callee's own defining
# module (confirmed empirically — a `sin(x)` call inside a hand rule in `src/rrules.jl` inlines as
# `GlobalRef(Differ, :sin)`, not `GlobalRef(Base, :sin)`), which `Core.Compiler.verify_ir` rejects as
# an "unbound or partitioned GlobalRef... in value position" once re-embedded in the caller's own
# compiled unit. So every recursive carrier invoke must stay a genuine `:invoke`, never inlined,
# regardless of apparent cost.
#
# Covers the two carriers and the hand-written `rrule!!` primitives. A hand-written pullback has no
# common supertype to test (it's a method on the closure's own compiler-generated type), so those
# are blocked at the call site instead: the emitted pullback-recursion `:invoke` carries
# `CC.IR_FLAG_NOINLINE`, which `resolve_todo` honours regardless of the callee's type.
_is_reverse_carrier_mi(mi::MethodInstance) = isa(mi.def, Method) && isa(mi.specTypes, DataType) &&
    !isempty(mi.specTypes.parameters) &&
    (mi.specTypes.parameters[1] === typeof(reverse_fwds_impl) ||
     mi.specTypes.parameters[1] === typeof(reverse_pullback_impl) ||
     mi.specTypes.parameters[1] === typeof(rrule!!))

# ---------------------------------------------------------------------------
# Coupling-point hooks: reverse-over-forward rejection and nested-forward frule protection
# (DifferForwards.jl is not a dependency of DifferReverse.jl). Default-inert here; overridden in
# `DifferForwardsOverReverseExt` (loaded only when both `DifferForwards` and `DifferReverse` are
# present).
# ---------------------------------------------------------------------------

# Coupling point 6: is `P` a carrier type owned by a different (forward-mode) AD-mode package
# (e.g. DifferForwards' `Dual`)? Encountering one while building a reverse-mode pass means
# reverse-over-forward, which is unsupported — `has_hand_reverse_rule`/
# `reverse_fwds_recursive_ci` use this to reject cleanly instead of crashing inside
# `fcodual_type`. Default `false` (no other mode is loaded).
_is_foreign_forward_carrier(@nospecialize(P)) = false

# Coupling point: does protecting hand-written forward-mode rules from inlining apply to this
# build? Only relevant when this reverse-mode interpreter was itself built to compile a carrier on
# behalf of an outer forward-over-reverse dualization — `nested_forward` is always available on
# `Reverse`, but the actual protection predicate (`has_hand_frule`) can only be consulted when
# DifferForwards is loaded.
function _nested_forward_protects_frule(interp::ContextualInterpreter{Reverse}, mi::MethodInstance)
    interp.owner.nested_forward || return false
    return at_world(interp.world, _foreign_has_hand_frule, interp, mi)
end
# `mi` deliberately untyped (not `::MethodInstance`): an override living in a package extension
# must be a strictly more specific method than the default, or Julia treats it as an illegal
# same-signature overwrite across modules ("Method overwriting is not permitted during Module
# precompilation").
_foreign_has_hand_frule(interp, @nospecialize(mi)) = false

# Never inline a call whose callee has a hand-written reverse-mode rule (so it survives into the
# primal IR for `_static_recursible_call`'s recursion dispatch), and never inline a recursive
# reverse-mode carrier invoke once emitted (`_is_reverse_carrier_mi` above).
#
# `_nested_forward_protects_frule` (above) adds a third condition, active only when this interp was
# built to compile a reverse-mode carrier on behalf of an outer forward-over-reverse dualization:
# also never inline a call whose callee has a hand-written forward-mode `frule!!` — reverse mode
# itself has no reason to protect it, but the outer forward-mode dualizer needs it to still be a
# surviving call. Always `false` for the ordinary (non-nested) reverse path.
function CC.src_inlining_policy(interp::ContextualInterpreter{Reverse}, mi::MethodInstance,
                                @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::UInt32)
    (_is_reverse_carrier_mi(mi) || has_hand_reverse_rule(interp, mi)) && return false
    _nested_forward_protects_frule(interp, mi) && return false
    return @invoke CC.src_inlining_policy(interp::CC.AbstractInterpreter, mi::MethodInstance,
                                          src::Any, info::CC.CallInfo, stmt_flag::UInt32)
end

# Build a minimal IRCode whose only effect is to `error(msg)` when invoked, installed via the same
# `finishinfer!`/`optimize` path as a real reverse-mode body. Works for either carrier's argument
# shape (`_impl_argtypes` below).
function reverse_error_ircode(impl_mi::MethodInstance, msg::String)
    stream = CC.InstructionStream(2)
    stream.stmt[1] = Expr(:call, error, msg); stream.type[1] = Union{}; stream.flag[1] = CC.IR_FLAG_NULL
    stream.stmt[2] = Core.ReturnNode();       stream.type[2] = Union{}; stream.flag[2] = CC.IR_FLAG_NULL
    cfg = CC.CFG(CC.BasicBlock[CC.BasicBlock(CC.StmtRange(1, 2), Int[], Int[])], Int[3])
    di = CC.DebugInfoStream(stream.line)
    di.def = impl_mi
    ir = CC.IRCode(stream, cfg, di, _impl_argtypes(impl_mi), Expr[], CC.VarState[])
    CC.verify_ir(ir)
    return ir
end

# `reverse_fwds_impl(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...)` is vararg after two fixed
# arguments, so its IR sees four slots: `#self#` (`Argument(1)`), `fcd` (`Argument(2)`), `ctx`
# (`Argument(3)`), and one packed tuple of the *argument* coduals (`Argument(4)`).
# `reverse_pullback_impl(tape, seed)` is ordinary — two flat slots after `#self#`, so tape is
# `Argument(2)` and seed `Argument(3)`.
function _impl_argtypes(mi::MethodInstance)
    params = mi.specTypes.parameters
    m = mi.def::Method
    m.isva || return Any[params...]
    nfixed = Int(m.nargs) - 1          # declared slots before the vararg, `#self#` included
    return Any[params[1:nfixed]..., Tuple{params[(nfixed + 1):end]...}]
end

# Mode hook — the only thing `contextual.jl` needs from reverse mode.
function build_contextual_ir(interp::ContextualInterpreter{Reverse}, mi::MethodInstance)
    if is_reverse_fwds_impl(mi)
        reason = Ref("Differ could not build the reverse forwards pass (no specific reason recorded).")
        edges = Any[]
        ir = build_reverse_fwds_ir(interp, mi, reason, edges)
        interp.transformed_edges[mi] = edges
        if ir === nothing
            interp.custom_state.bail_reasons[mi] = reason[]   # so a recursing caller can report *this* reason
            return reverse_error_ircode(mi, reason[])
        end
        return ir
    elseif is_reverse_pullback_impl(mi)
        reason = Ref("Differ could not build the reverse pullback pass (no specific reason recorded).")
        edges = Any[]
        ir = build_reverse_pullback_ir(interp, mi, reason, edges)
        interp.transformed_edges[mi] = edges
        if ir === nothing
            interp.custom_state.bail_reasons[mi] = reason[]
            return reverse_error_ircode(mi, reason[])
        end
        return ir
    end
    return nothing
end

# ===========================================================================
# Shared primal resolution: both carriers eventually need the same (primal_mi, n) pair, obtained from
# the tuple of `CoDual` argument types — directly for the fwds carrier, recovered from the `Tape`'s
# `ArgsTT` parameter for the pullback carrier (see `build_reverse_pullback_ir`).
# ===========================================================================
function resolve_reverse_primal(interp::ContextualInterpreter, codualparams::Vector{Any},
                                reason::Ref{String}, edges::Vector{Any})
    if !all(P -> P isa Type && P <: CoDual, codualparams)
        reason[] = "not every codual argument type is a `CoDual` (a vararg call?)"
        return nothing
    end
    # Every path that resolves what's actually being reverse-differentiated funnels through here (the
    # top-level primal, every recursive call via `reverse_fwds_recursive_ci`), so the composition check
    # belongs here too — reverse-over-reverse reaches this with `ftype === typeof(rrule!!)`.
    msg = _composition_bail_message(interp.world, _codual_primal_type(codualparams[1]))
    if msg !== nothing
        reason[] = msg
        return nothing
    end
    primal_tt = Base.to_tuple_type(Any[_codual_primal_type(P) for P in codualparams])
    push!(edges, primal_tt, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
    pmatch, _ = CC.findsup(primal_tt, CC.method_table(interp))
    if pmatch === nothing
        reason[] = "no unique primal method resolves for argument types " *
                   "$(Tuple(_codual_primal_type(P) for P in codualparams))"
        return nothing
    end
    if !isa(pmatch.method, Method)
        reason[] = "the resolved primal match is not a concrete Method"
        return nothing
    end
    primal_mi = specialize_method(pmatch.method, pmatch.spec_types, pmatch.sparams)::MethodInstance
    CC.add_inlining_edge!(edges, primal_mi)
    # `nfixed`: declared primal slots before the vararg tail (matches `_impl_argtypes`'s formula).
    # `-1` for non-vararg — not `nfixed == length(codualparams)`, since a vararg primal called with
    # zero trailing arguments has `nfixed == length(codualparams)` too (its IR still has a real,
    # empty, packed tail slot).
    nfixed = pmatch.method.isva ? Int(pmatch.method.nargs) - 1 : -1
    return (primal_mi, length(codualparams), nfixed)
end

# The primal IR indexes its own arguments in *packed* space: Julia's own lowering of a vararg
# method already binds the trailing parameters to one tuple slot, `Core.Argument(nfixed+1)`. Every
# reverse-mode helper that resolves a `Core.Argument` found *in the primal IR* needs the codual
# argument types in that packed shape, not the flat shape the carrier's own call arguments come in.
# `nfixed < 0` (see `resolve_reverse_primal`) means non-vararg: identity. Not `nfixed == n`: a
# *vararg* primal called with zero trailing arguments also has `nfixed == n`, but the primal IR
# still has a real (empty-tuple) packed tail slot in that case, so it must still pack.
function _packed_codualparams(iworld::UInt, codualparams::Vector{Any}, nfixed::Int)
    nfixed < 0 && return codualparams
    tailP = Tuple{(_codual_primal_type(P) for P in codualparams[(nfixed + 1):end])...}
    return Any[codualparams[1:nfixed]..., _fcdtype(iworld, tailP)]
end

# Packs a prologue's own flat `parg`/`farg` (positions `nfixed+1:end`) into one slot at
# `nfixed+1` and truncates both to packed length. Shared between the fwds and pullback prologues.
function _pack_vararg_args!(emit!::F, ctuple, parg::Vector{Any}, farg::Vector{Any},
                            codualparams::Vector{Any}, iworld::UInt, nfixed::Int) where {F}
    nfixed < 0 && return nothing
    n = length(codualparams)
    tailP = Tuple{(_codual_primal_type(codualparams[k]) for k in (nfixed + 1):n)...}
    # Read the flat tail before resizing: with zero trailing arguments, `nfixed+1` is one past
    # `parg`/`farg`'s current length, so the slot doesn't exist to assign into until after `resize!`.
    packed_primal = emit!(Expr(:call, ctuple, (parg[k] for k in (nfixed + 1):n)...), tailP)
    tailFT = fdtype(iworld, tailP)
    packed_fdata = if tailFT === NoFData
        # `tangent_type` collapses an all-`NoTangent`/`Tuple{}` tail to `NoTangent`, so this needs
        # the literal `NoFData()` — an emitted `Core.tuple(NoFData(), ...)` has type
        # `Tuple{NoFData,...}`, not `NoFData`.
        NoFData()
    else
        emit!(Expr(:call, ctuple,
            ((isassigned(farg, k) ? farg[k] : NoFData()) for k in (nfixed + 1):n)...), tailFT)
    end
    resize!(parg, nfixed + 1)
    resize!(farg, nfixed + 1)
    parg[nfixed + 1] = packed_primal
    farg[nfixed + 1] = packed_fdata
    return nothing
end

# Splits the packed tail's one rdata accumulator (`arg_ref_id[nfixed+1]`) back across a flat
# position `j` at the pullback's return, since the return arity is flat. `acc` is a real per-field
# tuple, or a collapsed `NoRData`/`ZeroRData` when the tail carries no real rdata.
@noinline function _pb_vararg_tail_rdata(acc, ::Val{j}, ::Type{Pj}) where {j,Pj}
    acc isa NoRData && return NoRData()
    acc isa ZeroRData && return zero_like_rdata_from_type(Pj)
    return getfield(acc, j)
end

# The primal's optimized `IRCode`, mirroring `Core.Compiler.typeinf_ircode`'s own body so
# `frame.edges` is available too.
function _optimized_primal_ir(interp::ContextualInterpreter, primal_mi::MethodInstance,
                              reason::Ref{String}, edges::Vector{Any})
    frame = CC.typeinf_frame(interp, primal_mi, false)
    if frame === nothing
        reason[] = "inference failed to produce optimized IR for the primal method $(primal_mi)"
        return nothing
    end
    opt = CC.OptimizationState(frame, interp)
    pir = CC.run_passes_ipo_safe(opt.src, opt, nothing)
    append!(edges, frame.edges)
    # For every concrete callee discovered above, whether its call survived or was inlined away,
    # also register the mt-backedge a hand-written `rrule!!` for it would need
    # (`register_implicit_rrule_backedge!`). Covers both carriers, since both `build_reverse_fwds_ir`
    # and `build_reverse_pullback_ir` call this function.
    for (_, item) in CC.ForwardToBackedgeIterator(Core.svec(frame.edges...))
        isa(item, MethodInstance) && register_implicit_rrule_backedge!(interp.world, edges, item)
    end
    return pir
end

# Recursion cycle guard (`interp.custom_state.in_progress`), keyed by the *carrier* mi. This guard's
# job is purely "don't recompile a carrier that's already being compiled higher up the call stack"
# (mutual recursion, A->B->A); it has nothing to do with whether an edge is cyclic (same primal),
# which `reverse_fwds_recursive_ci`/`reverse_pullback_recursive_ci` answer separately from an
# explicitly-passed `primal_mi`.
#
# Carrier-mi keying matters because the fwds carrier has two independent specializations per primal:
# `Ctx{Nothing}` (fresh-tape, plain `rev_gradient`) and `Ctx{<:Tape}` (pre-allocated, what a
# self-recursive edge always targets). Building the `Ctx{Nothing}` variant of a self-recursive
# primal genuinely requires also compiling the `Ctx{<:Tape}` sibling — a real, bounded, one-off
# nested compile, not a cycle. A primal-mi-keyed guard would wrongly flag that nested compile as
# "already in progress"; keying by carrier mi keeps the two independent builds from colliding.
function build_reverse_fwds_ir(interp::ContextualInterpreter, impl_mi::MethodInstance,
                               reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    if haskey(interp.custom_state.in_progress, impl_mi)
        reason[] = "recursive reverse-mode forwards-pass build for $(impl_mi) detected (a self- or " *
                   "mutually-recursive primal) — not yet supported; bailing instead of recursing forever"
        return nothing
    end
    interp.custom_state.in_progress[impl_mi] = nothing
    try
        # Carrier is `reverse_fwds_impl(fcd, ctx, argcds...)`: `params[2]` is `fcd`, `params[3]` the
        # `ctx`, `params[4:end]` the argument coduals. The full codual list is `(fcd, args...)`.
        p = impl_mi.specTypes.parameters
        codualparams = Any[p[2], p[4:end]...]
        info = resolve_reverse_primal(interp, codualparams, reason, edges)
        info === nothing && return nothing
        primal_mi, n, nfixed = info
        pir = _optimized_primal_ir(interp, primal_mi, reason, edges)
        pir === nothing && return nothing
        return reverse_fwds_to_ircode(interp, impl_mi, pir, n, nfixed, primal_mi; reason, edges)
    finally
        delete!(interp.custom_state.in_progress, impl_mi)
    end
end

function build_reverse_pullback_ir(interp::ContextualInterpreter, impl_mi::MethodInstance,
                                   reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    if haskey(interp.custom_state.in_progress, impl_mi)
        reason[] = "recursive reverse-mode pullback-pass build for $(impl_mi) detected (a self- or " *
                   "mutually-recursive primal) — not yet supported; bailing instead of recursing forever"
        return nothing
    end
    interp.custom_state.in_progress[impl_mi] = nothing
    try
        params = impl_mi.specTypes.parameters
        TapeT = length(params) >= 2 ? params[2] : Any
        if !(TapeT isa Type && TapeT <: Tape)
            reason[] = "the pullback carrier's tape argument is not a `Tape` (malformed specialization)"
            return nothing
        end
        ArgsTT = TapeT.parameters[1]
        codualparams = Any[ArgsTT.parameters...]
        info = resolve_reverse_primal(interp, codualparams, reason, edges)
        info === nothing && return nothing
        primal_mi, n, nfixed = info
        pir = _optimized_primal_ir(interp, primal_mi, reason, edges)
        pir === nothing && return nothing
        return reverse_pullback_to_ircode(interp, impl_mi, pir, n, nfixed, primal_mi; reason, edges)
    finally
        delete!(interp.custom_state.in_progress, impl_mi)
    end
end

# The optimized IR for a carrier: exactly what `CC.optimize` installs. Used both by
# `code_reverse_fwds_ircode`/`code_reverse_pullback_ircode` (reflection.jl) and available for
# future higher-order composition.
function optimized_reverse_fwds_ir(interp::ContextualInterpreter, impl_mi::MethodInstance,
                                   reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    ir = build_reverse_fwds_ir(interp, impl_mi, reason, edges)
    ir === nothing && return nothing
    world = CC.get_inference_world(interp)
    opt = CC.OptimizationState(impl_mi, CC.retrieve_code_info(impl_mi, world), interp)
    return run_ipo_passes!(ir, opt)
end
function optimized_reverse_pullback_ir(interp::ContextualInterpreter, impl_mi::MethodInstance,
                                       reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    ir = build_reverse_pullback_ir(interp, impl_mi, reason, edges)
    ir === nothing && return nothing
    world = CC.get_inference_world(interp)
    opt = CC.OptimizationState(impl_mi, CC.retrieve_code_info(impl_mi, world), interp)
    return run_ipo_passes!(ir, opt)
end

# ===========================================================================
# Shared static analysis: both the forwards and pullback builders need to agree, byte-for-byte, on
# which primal blocks are throw-only/unreachable, which blocks are reachable exits, and which
# runtime operand values each block's statements need communicated from forwards to pullback.
# Since both builders derive `pir` identically, computing this twice always agrees.
# ===========================================================================

function _unreachable_blocks(pir)
    nblocks = length(pir.cfg.blocks)
    unreachable = falses(nblocks)
    for b in 1:nblocks
        term = pir.stmts[pir.cfg.blocks[b].stmts.stop][:stmt]
        unreachable[b] = isa(term, Core.ReturnNode) && !isdefined(term, :val)
    end
    return unreachable
end

# Every reachable-return block, in block-number order. A branch with a `return` in each arm is the
# normal shape Julia's optimizer produces (it doesn't generally merge arms via a phi + single
# return), so multi-exit is the common case: every exit needs its own routing, like a `PhiNode`'s
# per-predecessor routing (see `reverse_pullback_to_ircode`).
function _exit_blocks(pir, unreachable)
    exits = Int[]
    for b in eachindex(pir.cfg.blocks)
        unreachable[b] && continue
        term = pir.stmts[pir.cfg.blocks[b].stmts.stop][:stmt]
        isa(term, Core.ReturnNode) && isdefined(term, :val) && push!(exits, b)
    end
    return exits
end

# Bulk save/restore analysis: which arguments have their primal contents saved once per call,
# rather than one overwritten element at a time.
#
# A `memoryrefset!`'s pullback restores the element it overwrote, so the primal is left exactly as
# the call found it. Per-element, that costs two tape slots and a load+store per store executed —
# O(iterations) in a loop. It can instead be done once: no pullback rule ever reads primal memory
# (only this restore writes it), so nothing observes the primal mid-pullback, and one `copyto!` in
# and one out at the boundary reproduces the same net effect.
#
# Restricted to arguments whose element type is `isbits`: the root has to be reachable from the
# pullback (arguments are, via `Tape.args`), and a non-bits element is a reference whose contents
# the copy wouldn't capture.

# Blocks that can execute more than once per call: the union of the natural loops of every back edge
# (an edge `b -> s` whose target dominates its source). Over-approximating this is harmless — it only
# moves a store from the per-element scheme to the bulk one, a cost decision, not a correctness one.
function _loop_blocks(pir)
    cfg = pir.cfg
    nb = length(cfg.blocks)
    inloop = falses(nb)
    dt = CC.construct_domtree(cfg.blocks)
    for b in 1:nb, s in cfg.blocks[b].succs
        (1 <= s <= nb) || continue
        CC.dominates(dt, s, b) || continue        # `b -> s` is a back edge
        # Natural loop body: header `s` plus everything reaching `b` without going through `s`.
        inloop[s] = true
        seen = falses(nb)
        seen[s] = true
        worklist = Int[b]
        while !isempty(worklist)
            x = pop!(worklist)
            (1 <= x <= nb) || continue
            seen[x] && continue
            seen[x] = true
            inloop[x] = true
            append!(worklist, cfg.blocks[x].preds)
        end
    end
    return inloop
end

# Walk a `MemoryRef` chain back to the object it indexes into, through exactly the shapes Julia 1.13
# lowers array indexing to (already tracked by `_fdata_tracked`): `PiNode` aliases, `memoryrefnew`
# (both the 1-arg `Memory` form and the 3-arg offsetting form), and an `Array`'s `.ref` field. Returns
# the terminal node, or `nothing` if the chain runs into anything else.
function _provenance_root(pir, iworld, @nospecialize(node))
    for _ in 1:length(pir.stmts)      # bounded: each step moves strictly up the chain
        isa(node, Core.Argument) && return node
        isa(node, Core.SSAValue) || return nothing
        s = pir.stmts[node.id][:stmt]
        if isa(s, Core.PiNode)
            node = s.val
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos, actual = _call_parts(s)
            f = _calleeval(fpos, iworld)
            isempty(actual) && return nothing
            if f === Base.memoryrefnew
                node = actual[1]
            elseif f === Core.getfield && length(actual) >= 2 && _bi_fieldname(actual[2]) === :ref
                node = actual[1]
            else
                return node        # a root in its own right (`memorynew`, a call, ...)
            end
        else
            return node            # `%new`, a phi, ...
        end
    end
    return nothing
end

# Does `ref_node`'s provenance root resolve to a bulk-saved argument? Shared by the comms scan and
# both IR builders (`_scan_block_comms`, `reverse_fwds_to_ircode`, `reverse_pullback_to_ircode`),
# which must all agree on this predicate.
_is_bulk_saved(pir, iworld, bulk_args::Set{Int}, @nospecialize(ref_node)) = begin
    root = _provenance_root(pir, iworld, ref_node)
    isa(root, Core.Argument) && root.n in bulk_args
end

# `block_of[i]`: which block statement `i` belongs to. Statement indices are monotonic within a
# block, so a single forward scan suffices. Shared by every static-analysis pass that needs a
# stmt->block lookup but isn't otherwise threading per-block state through the same scan (compare
# `reverse_fwds_to_ircode`'s main loop, which fuses this with live code-emission bookkeeping and keeps
# its own inline version).
function _stmt_block_map(pir)
    nb = length(pir.cfg.blocks)
    block_of = Vector{Int}(undef, length(pir.stmts))
    bidx = 1
    for i in eachindex(block_of)
        while bidx < nb && i > pir.cfg.blocks[bidx].stmts.stop
            bidx += 1
        end
        block_of[i] = bidx
    end
    return block_of
end

# Which argument positions are bulk-saved. `arg_primal_types[k]` is argument `k`'s primal type.
function _bulk_save_args(pir, iworld, arg_primal_types::Vector{Any})
    inloop = _loop_blocks(pir)
    bulk = Set{Int}()
    block_of = _stmt_block_map(pir)
    for i in 1:length(pir.stmts)
        inloop[block_of[i]] || continue   # a store executed once per call is not worth a whole copy
        s = pir.stmts[i][:stmt]
        (isa(s, Expr) && (s.head === :call || s.head === :invoke)) || continue
        fpos, actual = _call_parts(s)
        _calleeval(fpos, iworld) === Base.memoryrefset! || continue
        isempty(actual) && continue
        root = _provenance_root(pir, iworld, actual[1])
        isa(root, Core.Argument) || continue
        k = root.n                        # primal `Argument(j)` is `codualparams[j]` (`#self#` is 1)
        (1 <= k <= length(arg_primal_types)) || continue
        P = _widen(arg_primal_types[k])
        (P isa DataType && (P <: Memory || P <: Array) && isbitstype(eltype(P))) || continue
        push!(bulk, k)
    end
    return bulk
end

# ===========================================================================
# Statically re-derivable `MemoryRef` handles.
#
# A `memoryrefget`/`memoryrefset!` over a `MemoryRef` built by
# `memoryrefnew(getfield(arr, :ref), idx, boundscheck)` with `arr` an `Argument(k)` can be rebuilt in
# the pullback from `tape.args[k]` + `idx`, so its handle need not be pushed on the comms tuple (which
# would make it GC-tracked). `idx` may be a literal `Int` (pushed nowhere, baked into the pullback) or
# an `Int`-typed `SSAValue` (pushed as a plain `Int` comms item instead of the 16-byte, GC-scanned
# `MemoryRef` it replaces).
#
# Returns `(k::Int, idx::Union{Int,Core.SSAValue}, bc::Bool)` when `node` is such a `MemoryRef` SSA,
# else `nothing` (1-arg form over a fresh allocation, non-`Int` index, or non-argument root).
# ===========================================================================
function _static_ref_derivation(pir, iworld, @nospecialize(node))
    isa(node, Core.SSAValue) || return nothing
    s = pir.stmts[node.id][:stmt]
    (isa(s, Expr) && (s.head === :call || s.head === :invoke)) || return nothing
    fpos, actual = _call_parts(s)
    _calleeval(fpos, iworld) === Base.memoryrefnew || return nothing
    length(actual) >= 3 || return nothing          # need the 3-arg offsetting form
    idx = _calleeval(actual[2], iworld)
    if idx === nothing && isa(actual[2], Core.SSAValue) && _optype(pir, actual[2]) === Int
        idx = actual[2]                             # dynamic index, re-derived via the tape
    end
    (idx isa Int || idx isa Core.SSAValue) || return nothing
    bc = _calleeval(actual[3], iworld)
    (bc isa Bool) || return nothing                 # literal Bool boundscheck
    root = _provenance_root(pir, iworld, actual[1])
    isa(root, Core.Argument) || return nothing
    return (root.n, idx, bc)
end

# ===========================================================================
# Unique-predecessor analysis (mirrors Mooncake's
# `_characterise_unique_predecessor_blocks`), worked directly over primal block numbers rather than
# the `ID`s `cfg_ir.jl`'s copy of that algorithm uses — the forwards pass never leaves block-number
# space, and the pullback already tracks predecessors by primal block number too
# (`pir.cfg.blocks[b].preds`).
#
# `is_unique_pred[b]`: is block `b` the only predecessor of every one of its successors? If so, no
# successor of `b` can ever be ambiguous about where it came from, so `b` need not push its own
# number onto the block stack — this also covers a single reachable exit, a de facto unique
# predecessor of "the pullback's own entry".
#
# `pred_is_unique_pred[b]`: current formula is `length(preds[b]) <= 1` (see the per-edge note below)
# — governs whether `b` needs to pop the block stack.
# ===========================================================================
function _unique_predecessor_info(pir, exit_blocks::Vector{Int}, unreachable::AbstractVector{Bool},
                                  regions::Dict{Int,Int}, quiet::Set{Int})
    nblocks = length(pir.cfg.blocks)
    preds = [filter(!=(0), pir.cfg.blocks[b].preds) for b in 1:nblocks]
    succs = [pir.cfg.blocks[b].succs for b in 1:nblocks]

    is_unique_pred = falses(nblocks)
    for b in 1:nblocks
        ss = succs[b]
        is_unique_pred[b] = !isempty(ss) && all(s -> length(preds[s]) == 1, ss)
    end
    # A lone reachable exit is the only way control can leave the function, so it's a de facto
    # unique predecessor of "the pullback's entry routing" even though `succs[b]` is empty for it.
    length(exit_blocks) == 1 && (is_unique_pred[only(exit_blocks)] = true)

    # Collapsible regions (the `@boundscheck` diamond — see `_collapsible_regions`): none of a
    # region's blocks (entry `br`, interior `chk`/`pass`) need to push, exactly as if their ambiguous
    # successor were unique — nothing downstream ever needs to know whether the direct or the
    # checked edge into `merge` ran.
    #
    # `regions`/`quiet` come from `_scan_block_comms` rather than being recomputed here: comms
    # fusion there mutates the per-block comms lists `_collapsible_regions` reads, so they must be
    # computed before that mutation happens.
    for b in quiet
        is_unique_pred[b] = true
    end

    # Per-edge pop (fixes ISSUES #52, a real gradient-corruption bug — see the warning at
    # `_split_ambiguous_block_pushes` below before changing either side of this scheme): `b` pops
    # iff it has multiple predecessors, each of which pushed on its edge into `b`. A
    # single-predecessor block is never itself ambiguous, so its sole predecessor never pushes on
    # the edge into it, and `b` must not pop either — hence `length(preds[b]) <= 1`, which also
    # covers the entry block's 0-predecessor case. The push/pop granularity moved from per-block to
    # per-edge; this pop formula and `_split_ambiguous_block_pushes`'s push are two halves of one
    # change and must not be modified independently.
    pred_is_unique_pred = falses(nblocks)
    for b in 1:nblocks
        pred_is_unique_pred[b] = length(preds[b]) <= 1
    end
    # A collapsible region's `merge` block has two real predecessors, so the generic formula above
    # never marks it — force it directly: `reverse_pullback_to_ircode` routes it through the canonical
    # `br` alone (see `regions`), so nothing is ever popped on its behalf either.
    for merge in keys(regions)
        pred_is_unique_pred[merge] = true
    end

    return is_unique_pred, pred_is_unique_pred, regions
end

# Collapsible regions: extends the unique-predecessor optimization above from a single edge to a
# whole comms-free sub-region. The fixed shape Julia's `@boundscheck` lowering produces around every
# `getindex`/`setindex!` is exactly this: entry block `br` branches to `{merge, chk}` (skip-check vs
# run-check); `chk` is a single-predecessor, comms-free block branching to `{thrw, onward}`
# (checked-fail vs checked-pass), where `onward` reaches `merge` directly or via a linear chain of
# further single-pred/single-succ comms-free pass-through blocks; `thrw` is an unreachable dead end.
# Since neither `chk`/chain push any comms and `merge` has no leading `PhiNode` (both checked, never
# assumed), the pullback never needs to know whether the forwards pass took the direct edge or the
# checked detour: replaying `merge` in reverse and always routing back through `br` directly reaches
# the correct upstream state either way.
#
# Deliberately narrow pattern match (not a general dominance/SESE analysis) — anything that doesn't
# match byte-for-byte falls through to the ordinary unique-predecessor handling above; never guess.
# Returns `(merges, quiet)`: `merges` maps a collapsible `merge` block to its canonical entry `br`;
# `quiet` is every block in every matched region that must stop pushing onto the block stack.
function _collapsible_regions(pir, unreachable::AbstractVector{Bool},
                              block_comms_nodes::Vector{Vector{Any}},
                              block_has_subtape::AbstractVector{Bool})
    nblocks = length(pir.cfg.blocks)
    preds = [filter(!=(0), pir.cfg.blocks[b].preds) for b in 1:nblocks]
    succs = [pir.cfg.blocks[b].succs for b in 1:nblocks]
    # `block_has_subtape[b]`: does `b` contain a direct self-recursive call? Its inner tape is
    # stored in `Tape.subtapes`, not in `block_comms_nodes[b]` — so a block whose only communicated
    # state was that call must still count as not comms-free.
    comms_free(b) = isempty(block_comms_nodes[b]) && !block_has_subtape[b]
    dead_end(b) = unreachable[b] && isempty(succs[b])
    solo_pred(b, from) = length(preds[b]) == 1 && only(preds[b]) == from
    no_leading_phi(b) = !isa(pir.stmts[pir.cfg.blocks[b].stmts.start][:stmt], Core.PhiNode)
    # `merge`'s full predecessor set must reduce to exactly `{br, exitb}` — no third, unrelated edge
    # feeding it from elsewhere — and it must be a genuine reachable block, not itself a dead end.
    closes_at(merge, br, exitb) = !unreachable[merge] && no_leading_phi(merge) &&
                                  Set(preds[merge]) == Set((br, exitb))

    # Does `chk` (the "run the check" arm out of `br`) lead only to `merge` (directly, or via a
    # linear chain of further comms-free pass-through blocks) and a throw dead end? Returns the
    # interior block list on a match, `nothing` otherwise.
    function matches_check(br, chk, merge)
        (solo_pred(chk, br) && comms_free(chk) && length(succs[chk]) == 2) || return nothing
        for (thrw, onward) in ((succs[chk][1], succs[chk][2]), (succs[chk][2], succs[chk][1]))
            dead_end(thrw) || continue
            interior = [chk]
            cur = onward
            # Bounded by `nblocks` as a hard stop against a shape this analysis didn't anticipate —
            # a comms-free chain can't legitimately revisit a block.
            while cur != merge && length(interior) <= nblocks && solo_pred(cur, interior[end]) &&
                  comms_free(cur) && length(succs[cur]) == 1
                push!(interior, cur)
                cur = only(succs[cur])
            end
            cur == merge && closes_at(merge, br, interior[end]) && return interior
        end
        return nothing
    end

    # `merges`: merge block -> canonical entry `br`, consumed by the pullback builder to route
    # `merge`'s reverse-replay through `br` alone. `quiet`: every block in every matched region (`br`
    # plus interior `chk`/`pass`) that must stop pushing onto the block stack.
    merges = Dict{Int,Int}()
    quiet = Set{Int}()
    for br in 1:nblocks
        length(succs[br]) == 2 || continue
        s1, s2 = succs[br]
        s1 == s2 && continue
        interior = matches_check(br, s1, s2)
        merge = s2
        if interior === nothing
            interior = matches_check(br, s2, s1)
            merge = s1
        end
        interior === nothing && continue
        merges[merge] = br
        push!(quiet, br)
        union!(quiet, interior)
    end
    return merges, quiet
end

# ===========================================================================
# ISSUES #52: split a block's stack push per edge rather than per block.
#
# `is_unique_pred[b] == false` makes `emit_epilogue!` push block `b`'s number unconditionally, but a
# `GotoIfNot` with one unambiguous arm (unique predecessor) and one ambiguous arm (multiple
# predecessors) only needs the push on the ambiguous arm — the unambiguous arm's target never pops
# it. This is a post-processing pass over the already-built fwds-carrier `ir`: relocate the push into
# a new relay block reached only via the ambiguous arm, using the `ID`/`CFGBlock` working-IR layer
# (`cfg_ir.jl`) the pullback pass already uses for its own extra routing blocks.
#
# Redirecting one arm through a relay changes the ambiguous target's real predecessor from `b` to the
# relay — any `PhiNode` there still names `b`, so its `edges` must be patched to match, or the result
# miscompiles (a stale edge reference isn't something `verify_ir`/codegen catch on their own: the
# block number the edge resolves to still exists, it just now names the wrong predecessor).
# ===========================================================================

"""
    _is_expected_block_push(inst::CC.NewInstruction, b::Int)::Bool

`true` iff `inst` is exactly the block-stack push `emit_epilogue!` emits for block `b`
(`push!(block_stack, Int32(b))`, as a `:call` or a resolved `:invoke`) — the shape
`_split_ambiguous_block_pushes` expects at `blks[b].insts[end-1]`.
"""
function _is_expected_block_push(inst::CC.NewInstruction, b::Int)::Bool
    s = inst.stmt
    isa(s, Expr) || return false
    if s.head === :invoke
        length(s.args) == 4 || return false
        callee, args = s.args[2], s.args[3:4]
    elseif s.head === :call
        length(s.args) == 3 || return false
        callee, args = s.args[1], s.args[2:3]
    else
        return false
    end
    return callee === Base.push! && args[2] === Int32(b)
end

"""
    _split_ambiguous_block_pushes(ir::CC.IRCode, pir, is_unique_pred::AbstractVector{Bool})::CC.IRCode

Post-processes the already-built forwards-carrier `ir` (still 1:1 in block topology with the primal
`pir`) so a block with a `GotoIfNot` terminator where exactly one arm is ambiguous no longer pushes
its block number on the unambiguous arm. Three stages: classify candidates in primal block-number
space (Stage 0), splice a relay block per candidate plus fix up any `PhiNode` edge the redirect
disturbs (Stage 1), reassemble and lower back to a real `IRCode` (Stage 2). Returns `ir` unchanged
(no round trip through the `cfg_ir.jl` layer) when there is nothing to split.
"""
function _split_ambiguous_block_pushes(ir::CC.IRCode, pir, is_unique_pred::AbstractVector{Bool})::CC.IRCode
    nblocks = length(pir.cfg.blocks)

    # Stage 0: classify, in primal block-number space — no IR construction yet.
    candidates = Tuple{Int,Symbol,Int}[]   # (b, :dest|:fallthrough, ambiguous target block number)
    for b in 1:(nblocks - 1)
        is_unique_pred[b] && continue
        term = pir.stmts[pir.cfg.blocks[b].stmts.stop][:stmt]
        isa(term, Core.GotoIfNot) || continue    # the only 2-successor terminator kind in primal IR
        dest, fall = Int(term.dest), b + 1
        npd = length(filter(!=(0), pir.cfg.blocks[dest].preds))
        npf = length(filter(!=(0), pir.cfg.blocks[fall].preds))
        if npd > 1 && npf > 1
            continue                              # both already ambiguous — nothing to split off
        elseif npd > 1
            push!(candidates, (b, :dest, dest))
        elseif npf > 1
            push!(candidates, (b, :fallthrough, fall))
        end
        # else: neither ambiguous — contradicts `!is_unique_pred[b]`; skip defensively.
    end
    isempty(candidates) && return ir

    # Stage 1: surgery + PhiNode fixup.
    blks = _ircode_to_cfg_blocks(ir)          # blks[b] is primal block b — order-preserving

    rewritten = Dict{Int,CFGBlock}()          # b -> its push-stripped (maybe re-terminated) block
    append_relays = CFGBlock[]                # `:dest` relays — appended at the very end
    insert_after = Dict{Int,CFGBlock}()       # b -> its `:fallthrough` relay, inserted right after b
    phi_fixups = Tuple{Int,ID,ID}[]           # (target block#, old pred ID, relay ID) — pre-mutation

    for (b, side, target) in candidates
        blk = blks[b]
        length(blk) >= 2 ||
            error("Differ internal error: block $b has no room for the block-stack push expected " *
                  "by _split_ambiguous_block_pushes")
        push_id, push_inst = blk.inst_ids[end - 1], blk.insts[end - 1]
        _is_expected_block_push(push_inst, b) ||
            error("Differ internal error: expected a block-stack push as the second-to-last " *
                  "instruction of block $b, found `$(push_inst.stmt)`")

        relay_id = ID()
        relay = CFGBlock(relay_id, [push_id, ID()],
                         CC.NewInstruction[push_inst, new_inst(IDGotoNode(blks[target].id), Any)])

        term_id, term_inst = blk.inst_ids[end], blk.insts[end]
        if side === :dest
            old_term = term_inst.stmt::IDGotoIfNot
            new_term = CC.NewInstruction(term_inst; stmt=IDGotoIfNot(old_term.cond, relay_id))
            push!(append_relays, relay)
        else
            new_term = term_inst              # dest untouched — only the implicit fallthrough moves
            insert_after[b] = relay
        end
        rewritten[b] = CFGBlock(blk.id, vcat(blk.inst_ids[1:(end - 2)], term_id),
                                vcat(blk.insts[1:(end - 2)], new_term))
        push!(phi_fixups, (target, blk.id, relay_id))
    end

    # Applied in one dedicated pass, over the pre-mutation candidate list, so a target block number
    # being before or after `b` (a loop back-edge target has a *lower* number) can't cause ordering
    # or aliasing bugs.
    for (target, old_id, relay_id) in phi_fixups
        _, phis = phi_nodes(blks[target])
        for phi_inst in phis
            phi = phi_inst.stmt::IDPhiNode
            for j in eachindex(phi.edges)
                phi.edges[j] == old_id && (phi.edges[j] = relay_id)
            end
        end
    end

    # Stage 2: reassemble in primal block order, then lower back to a real IRCode.
    final = CFGBlock[]
    for b in 1:nblocks
        push!(final, get(rewritten, b, blks[b]))
        haskey(insert_after, b) && push!(final, insert_after[b])
    end
    append!(final, append_relays)
    return lower_cfg_blocks_to_ir(final, ir)
end

# Static provenance analysis: which SSA statements' values have a statically-known fdata (shadow),
# traceable back to a function argument whose own fdata is non-trivial. Two chains:
#
#  * array-index chain: exactly the shape Julia 1.13 lowers `x[i]` to, back to an `Array` argument —
#    `Base.getfield(x, :ref)::MemoryRef{T}` then `Base.memoryrefnew(ref, i, false)`, with an optional
#    `PiNode` alias in between.
#  * general-struct chain: `Core.getfield(x, fld)` off a tracked, non-`Array` object, whenever the
#    result itself has non-trivial fdata (a nested array or mutable-struct field) —
#    `_get_fdata_field` covers `FData`/`MutableTangent`/`Tuple`/`NamedTuple` uniformly, so an
#    immutable closure holding a mutable ref and a mutable struct holding another mutable struct
#    both work with the same emission path.
#
# Bounds the feature precisely: an array/struct reachable any other way (nested inside a returned
# value, freshly allocated, read out of a container via ordinary indexing) is untracked, and
# untracked-but-differentiable is a real, located bail at the point of use, never silently
# mishandled.
#
# Which top-level (fwds-carrier) arguments carry non-trivial fdata. Factored out of
# `_fdata_tracked` so `_static_recursible_call`'s array-argument-recursion guard can check a bare
# `Core.Argument` operand directly without recomputing this.
function _arg_fdata_tracked(iworld::UInt, n::Int, codualparams::Vector{Any})
    arg_tracked = falses(n)
    for k in 1:n
        arg_tracked[k] = fdtype(iworld, _codual_primal_type(codualparams[k])) !== NoFData
    end
    return arg_tracked
end

# Returns a `BitVector` of length `length(pir.stmts)`, indexed by SSA id (`tracked[i]`). Argument
# provenance (`Core.Argument(k)`) is checked inline via `arg_tracked` rather than returned, since
# every caller that needs it only ever looks the chain up starting from an `SSAValue`, never a bare
# `Argument` directly (array-argument recursion in `_static_recursible_call` needs `arg_tracked`
# itself, exposed separately via `_arg_fdata_tracked` above).
function _fdata_tracked(pir, iworld, n::Int, codualparams::Vector{Any})
    N = length(pir.stmts)
    tracked = falses(N)
    arg_tracked = _arg_fdata_tracked(iworld, n, codualparams)
    provenance_tracked(@nospecialize node) =
        isa(node, Core.SSAValue) ? tracked[node.id] :
        isa(node, Core.Argument) ? (node.n <= n && arg_tracked[node.n]) : false
    # `tracked` is monotone (false -> true only), so a loop-carried `PhiNode` reading a
    # not-yet-computed back-edge is handled by rescanning to a fixpoint rather than a separate
    # pre-pass. Least fixpoint = "provably traceable"; anything left untracked bails at point of use.
    changed = true
    while changed
        changed = false
        for i in 1:N
            was = tracked[i]
            s = pir.stmts[i][:stmt]
            if isa(s, Core.PiNode)
                tracked[i] = provenance_tracked(s.val)
            elseif isa(s, Core.PhiNode)
                Ti = pir.stmts[i][:type]
                if fdtype(iworld, Ti) !== NoFData
                    ok = true
                    for j in 1:length(s.values)
                        if !isassigned(s.values, j) || !provenance_tracked(s.values[j])
                            ok = false
                            break
                        end
                    end
                    tracked[i] = ok
                end
            elseif isa(s, Expr) && s.head === :new
                # A locally-created mutable struct is a fresh provenance root: the fwds pass always
                # builds it a real `MutableTangent` shadow, so downstream `getfield`/`setfield!` can
                # resolve a shadow with no function-argument ancestor. Only when `T` has differentiable
                # content — otherwise `tangent_type(T) === NoTangent`. `_calleeval` resolves a
                # `GlobalRef` type argument (the common shape after optimization, `%new(Main.MPoint,
                # ...)`, not a literal `DataType`) at the inference world.
                T = _calleeval(s.args[1], iworld)
                nargs = length(s.args) - 1
                if T isa DataType && ismutabletype(T) && fdtype(iworld, T) !== NoFData
                    tracked[i] = true
                elseif T isa DataType && !ismutabletype(T) && isconcretetype(T) &&
                       fdtype(iworld, T) !== NoFData && fieldcount(T) == nargs
                    # Immutable struct isn't a provenance root itself: fdata is the tuple of field
                    # fdata, tracked iff every fdata-carrying field is (same gate as `Core.tuple` below).
                    tracked[i] = all(j -> fdtype(iworld, fieldtype(T, j)) === NoFData ||
                                          provenance_tracked(s.args[j + 1]),
                                     1:nargs)
                end
            elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
                fpos, actual = _call_parts(s)
                f = _calleeval(fpos, iworld)
                if f === Core.getfield && length(actual) >= 2 && provenance_tracked(actual[1])
                    fk = actual[2]
                    fname = isa(fk, QuoteNode) ? fk.value : fk
                    Ti = pir.stmts[i][:type]
                    Pobj = _optype(pir, actual[1])
                    if fname === :ref && _widen(Ti) <: MemoryRef
                        tracked[i] = true
                    elseif fdtype(iworld, Ti) !== NoFData &&
                           !(Pobj isa DataType && (Pobj <: Array || Pobj <: MemoryRef || Pobj <: Memory))
                        # Struct field with non-trivial fdata (nested array/mutable struct); excludes
                        # Array/MemoryRef/Memory, whose raw `.ref`-chain shadow `_get_fdata_field`
                        # can't serve — needed since `fdata_type(tangent_type(Ptr{Nothing}))` is non-trivial.
                        tracked[i] = true
                    end
                elseif f === Core.memorynew && !isempty(actual)
                    # Array allocation step 1: a fresh `Memory{P}` is itself a provenance root, exactly
                    # like a locally-`%new`'d mutable struct above.
                    MT = _widen(pir.stmts[i][:type])
                    tracked[i] = MT isa DataType && MT <: Memory && fdtype(iworld, MT) !== NoFData
                elseif f === Base.memoryrefnew && !isempty(actual) && provenance_tracked(actual[1])
                    tracked[i] = true
                elseif f === Base.memoryrefget && !isempty(actual) && provenance_tracked(actual[1])
                    # A read off a tracked ref whose result is itself array/mutable-struct-valued (a
                    # nested array) is a provenance root in turn — its shadow is the corresponding
                    # element of the shadow array. A scalar (bits) result carries no fdata of its own, so
                    # is deliberately left untracked — its gradient flows via rdata routing, not shadow
                    # aliasing.
                    fdtype(iworld, pir.stmts[i][:type]) !== NoFData && (tracked[i] = true)
                elseif f === Core.tuple && !isempty(actual)
                    # Mirrors `builtin_rrule_comms(::Val{Core.tuple}, ...)`'s scope gate: concrete
                    # non-vararg Tuple, one field per operand, tracked iff every fdata-carrying operand is.
                    Ti = pir.stmts[i][:type]
                    T = _widen(Ti)
                    if fdtype(iworld, Ti) !== NoFData && T isa DataType && T <: Tuple && isconcretetype(T) &&
                       !(!isempty(T.parameters) && isa(last(T.parameters), Core.TypeofVararg)) &&
                       fieldcount(T) == length(actual)
                        tracked[i] = all(j -> fdtype(iworld, fieldtype(T, j)) === NoFData || provenance_tracked(actual[j]),
                                         eachindex(actual))
                    end
                elseif !(f isa Core.Builtin) && !(f isa Core.IntrinsicFunction) &&
                       fdtype(iworld, pir.stmts[i][:type]) !== NoFData
                    # A derived recursive call's array/mutable-struct result now gets a real shadow
                    # (`shadow_map`). Over-approximates (a callee that fails to resolve bails the
                    # whole build anyway), so can't produce a wrong gradient.
                    tracked[i] = true
                end
            end
            tracked[i] && !was && (changed = true)
        end
    end
    return tracked
end

# `_widen` (lattice-element widener — a statement's result type can be a `Core.PartialStruct`/
# `Const` rather than a plain `Type`) lives in `DifferCore/src/shared_ir_helpers.jl`, shared with
# `DifferForwards`.
#
# `rdtype`/`fdtype` route the whole `rdata_type(tangent_type(...))`/`fdata_type(tangent_type(...))`
# composition through `at_world`, not just the inner `tangent_type` call: dispatch inside a
# `@generated` generator is pinned to the generator's definition world, so a later-loaded package's
# `tangent_type` method (this package's own `Stack`/`Tape` methods, seen from a nested
# forward-over-reverse build) would otherwise be invisible.
_rdtype_impl(@nospecialize W) = rdata_type(tangent_type(W))
_fdtype_impl(@nospecialize W) = fdata_type(tangent_type(W))
rdtype(world::UInt, @nospecialize P) = at_world(world, _rdtype_impl, _widen(P))
fdtype(world::UInt, @nospecialize P) = at_world(world, _fdtype_impl, _widen(P))

# Extract the callee-position node and actual-argument nodes from a `:call`/`:invoke` statement —
# the same split used at every call-statement site in this file, factored out since the recursion
# path needs it independently from the main dispatch loops.
_call_parts(s::Expr) = s.head === :invoke ? (s.args[2], @view s.args[3:end]) : (s.args[1], @view s.args[2:end])

# Is call statement `i` (`s`, already known to be a surviving, non-intrinsic, non-`getfield`,
# non-array-builtin `:call`/`:invoke`) a candidate for Part 1's recursive `rrule` support? Purely
# static (no compilation): resolves the callee value and argument/result types only. Returns
# `(fval, ftype, argtypes)` on success or `nothing` (with `reason[]` set) otherwise.
#
# Two guards, in order:
#  1. Callee must be statically resolvable to a concrete, non-tangent-carrying value (an ordinary
#     top-level function/singleton, never a closure with differentiable captures or a
#     dynamically-dispatched callee) — what lets the recursive invoke pass `CoDual{ftype,NoFData}`
#     for the callee slot with no fdata to thread through.
#  2. Every argument type must be concrete, and its fdata must either be trivial or a real `Array`/
#     mutable struct whose identity is traceable back to a tracked function argument. A correctness
#     guard, not just missing-feature: passing a freshly-zeroed fdata with no link to the real
#     shadow would be silently wrong, not just unsupported.
#
# The call's own result may carry fdata (an array/mutable-struct return) — the emission side routes
# its shadow into `shadow_map` for the caller to accumulate into.
function _static_recursible_call(pir, iworld, i::Int, s::Expr, reason::Ref{String},
                                 arg_tracked::BitVector, fdata_tracked::BitVector)
    fpos, actual = _call_parts(s)
    fval = _calleeval(fpos, iworld)
    # `_calleeval` returns `nothing` for a callee in argument position (an `Argument`/`SSAValue`,
    # e.g. `mapreduce_impl(f, op, A, ...)`'s `f`) — no compile-time value, but the recursion
    # machinery never needs one: `reverse_fwds_recursive_ci` takes `ftype`, not `fval`; `fval` is
    # used only once, to build the callee's `CoDual` at emission, where `fval === nothing` means
    # "resolve the operand there instead" (`presolve(fpos)`). So fall back to the operand's type
    # (`_optype_w`) rather than bailing — what lets `sum(sin, v)` recurse even though `sin`'s value
    # isn't statically known inside `mapreduce_impl`'s body.
    ftype = fval === nothing ? _optype_w(pir, iworld, fpos) : _typeof(fval)
    if !(ftype isa DataType && isconcretetype(ftype))
        reason[] = "callee type $(ftype) is not a concrete DataType at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    # A composite inner function survives dualization as a genuine recursive call to `rrule!!`, not
    # caught by `has_hand_reverse_rule` during inlining — so it reaches here. Checked before every
    # other rejection so reverse-over-reverse gets this message regardless of whether the inner call
    # is a hand rule or composite.
    msg = _composition_bail_message(iworld, ftype)
    if msg !== nothing
        reason[] = "$(msg) at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    if ftype === DataType
        # "Some Type value, identity erased" — mirrors the argument-side rejection below. A
        # genuinely concrete singleton type value (`Type{Float64}`) is handled fine by
        # `fcodual_type`; a bare `DataType` isn't, and the `%new(CoDual{...}, ...)` this guard
        # protects would be illegal IR.
        reason[] = "recursive call with a Type-valued callee of erased identity is not supported " *
                   "yet at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    if _tt(iworld, ftype) !== NoTangent
        reason[] = "recursive calls into a callee with differentiable captures ($(ftype)) are not " *
                   "supported yet at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    # `_optype_w`, not `_optype`: an operand can be a `GlobalRef`/`QuoteNode` naming a value (e.g.
    # `sum(sin, v)`'s `sin`), and the types derived here decide which `rrule!!` the recursion
    # resolves and annotate the `%new(CoDual{...}, ...)` the emission side builds — a node-shaped
    # answer would both misresolve the rule and emit IR whose declared type doesn't match the value.
    argtypes = Any[_optype_w(pir, iworld, a) for a in actual]
    for (j, P) in enumerate(argtypes)
        if !(P isa DataType && isconcretetype(P))
            reason[] = "recursive call has a non-concrete argument type $(P) at %$i: `$(_stmt_str(s))`"
            return nothing
        end
        # `P === DataType` means "some Type value, identity erased" — `fcodual_type(DataType)`
        # special-cases to a bare abstract `CoDual`, which would make the `%new` this guard protects
        # illegal IR. Reject here instead.
        if P === DataType
            reason[] = "recursive call with a Type-valued argument of erased identity is not " *
                       "supported yet at %$i: `$(_stmt_str(s))`"
            return nothing
        end
        if fdtype(iworld, P) !== NoFData
            # Any fdata shape is allowed through: fdata is the identity-carrying half of a tangent,
            # so an immutable aggregate's is a value wrapper whose leaves are the caller's own shared
            # shadow arrays/`MutableTangent`s — a callee accumulates into the caller's real buffers
            # either way, and the value half comes back as the call's returned rdata. What must be
            # guarded is that the fdata is the caller's real shadow and not a detached zero, which is
            # the provenance check below.
            a = actual[j]
            tracked_here = isa(a, Core.Argument) ? (a.n <= length(arg_tracked) && arg_tracked[a.n]) :
                           isa(a, Core.SSAValue) ? fdata_tracked[a.id] : false
            if !tracked_here
                reason[] = "recursive call with an fdata-carrying argument ($(P)) whose " *
                           "provenance is not traceable to a function argument is not supported " *
                           "yet at %$i: `$(_stmt_str(s))`"
                return nothing
            end
        end
    end
    # No result-fdata guard here; a mismatch is caught at the emission site instead.
    return (fval, ftype, argtypes)
end

# Resolve (and compile, under the caller's own `interp`) the `CodeInstance` for the callee's
# `reverse_fwds_impl(CoDual{ftype,NoFData}(fval,NoFData()), argcoduals...)` specialization, so
# recursion support can emit a static `:invoke` to it. Must reuse the caller's own `interp`, not a
# fresh `NativeInterpreter`: `reverse_fwds_impl` specializations only get transformed via
# `build_contextual_ir` when compiled under a `ContextualInterpreter`. Reusing `interp` is also what
# lets the `in_progress` cycle guard (`build_reverse_fwds_ir` above) see a genuine cycle.
#
# An inner call resolves one of two ways, both sharing the argument layout `(fcd, ctx, argcds...)`,
# so the only thing the caller needs is which value goes in the `:invoke`'s callee slot:
#
#   * a hand-written `rrule!!` for the callee, if one exists — invoked as `rrule!!`;
#   * otherwise the derived path: the fwds carrier `reverse_fwds_impl`, invoked directly.
#
# The ctx slot the emitted `:invoke` passes differs between the two. A hand rule's pullback isn't a
# `Tape` — nothing to pre-allocate for it — so it always gets a fresh `Ctx()`. A derived callee's
# pullback is a `Tape`, pushed as a `(:subtape, SSAValue(i))` value onto the outer block's comms
# `Stack` once per execution — and since a `Stack` never deallocates, after the first execution the
# next push's slot already holds a structurally identical tape from a previous call. The emission
# site hands the callee that recycled tape (`_inner_ctx`, `stack.jl`) instead of allocating fresh,
# which is why the `ci` this function returns for a derived non-self-recursive callee targets the
# `Ctx{InnerTapeT}` pre-allocated sibling, not the `Ctx{Nothing}` variant resolved first only to
# learn `InnerTapeT`. Self-recursion (below) follows the identical recycling idea but needs its own
# resolution, since the sibling it recycles from is itself, not a separately-compiled callee.
#
# Returns `(ci, callee_val, InnerFCoDualT, InnerPullbackT)` on success or `nothing` (with `reason[]`
# set); the emit sites build `invoke(callee_val, ci, fcd, ctx_val, argcds...)`, where `ctx_val` is
# `Ctx()` for a hand rule and the `_inner_ctx`-recycled value otherwise.
#
# `impl_mi` is the carrier mi of the build this call is part of; `current_primal_mi` is that build's
# own primal mi. `R` is this call statement's own (widened) primal result type, read straight off
# the primal IR.
#
# Direct self-recursion (the callee's own primal mi is exactly `current_primal_mi`) never needs a
# fixed point solved for its `Tape` type: the comms slot is declared as the bare `Tape` UnionAll
# (`InnerPullbackT` below), which makes the comms-tuple type finite — unconditional whenever the
# primal mi matches, since every ctx-type variant of the fwds carrier must agree on it or the
# concrete `Tape` type each separately computes would disagree.
#
# Whether compiling is needed is a narrower question, decided by comparing the resolved
# recursive-call target `callee_impl_mi` against `impl_mi` itself. The self-edge always targets
# `Ctx{own_TapeT}` (this build's own tape type, passed by the emission loop once known):
#   * `callee_impl_mi === impl_mi` — a literal self-edge, holding whenever this build's own ctx type
#     already is `Ctx{own_TapeT}` (a `Ctx{<:Tape}` pre-allocated build recursing into itself) — no
#     compile needed: codegen's `mi == ctx.linfo` self-recursion fast path triggers directly.
#   * otherwise — same primal, different carrier (this build is the `Ctx{Nothing}` tape-allocating
#     variant, whose self-edge targets the `Ctx{<:Tape}` sibling). That sibling has never been
#     compiled, so it must be — but this always terminates in exactly one bounded nested compile:
#     building it hits its own self-edge, whose ctx is `Ctx{own_TapeT}` (computed identically, since
#     tape shape depends only on comms structure, not ctx), which takes the literal-identity branch
#     above and stops. (This is also why `interp.custom_state.in_progress` stays keyed by carrier
#     mi, not primal mi: a primal-keyed guard would mistake this legitimate nested compile for the
#     primal already being in progress.)
function reverse_fwds_recursive_ci(interp, impl_mi::MethodInstance, current_primal_mi::MethodInstance,
                                   @nospecialize(ftype), argtypes::Vector{Any}, @nospecialize(R),
                                   edges::Vector{Any}, reason::Ref{String};
                                   @nospecialize(own_TapeT=nothing))
    # `has_hand_reverse_rule` already rejects a surviving call with a `Dual` callee/argument during
    # inlining, so this shouldn't be reachable — kept as a `reason[]`-based fallback in case some path
    # reaches recursion without going through that check, rather than crashing in `fcodual_type` below.
    msg = _composition_bail_message(interp.world, ftype)
    if msg === nothing && any(P -> P isa Type && at_world(interp.world, _is_foreign_forward_carrier, P), argtypes)
        msg = "reverse-over-forward is not supported: encountered a `Dual` (forward-mode carrier) " *
              "argument while building a reverse-mode pass"
    end
    if msg !== nothing
        reason[] = msg
        return nothing
    end
    argcodualtys = Any[_fcdtype(interp.world, P) for P in argtypes]
    hand = hand_reverse_rule_match(interp, ftype, argtypes)
    if hand !== nothing
        tt, fm = hand
        callee_val = rrule!!
    else
        tt = Tuple{typeof(reverse_fwds_impl),CoDual{ftype,NoFData},Ctx{Nothing},argcodualtys...}
        primal_tt = Base.to_tuple_type(Any[ftype, argtypes...])
        push!(edges, primal_tt, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
        pmatch, _ = CC.findsup(primal_tt, CC.method_table(interp))
        if pmatch !== nothing && isa(pmatch.method, Method)
            callee_primal_mi =
                specialize_method(pmatch.method, pmatch.spec_types, pmatch.sparams)::MethodInstance
            if callee_primal_mi === current_primal_mi
                # The self-edge targets `Ctx{own_TapeT}` — this build's own tape type, passed by the
                # emission loop once known. At scan time `own_TapeT` isn't known yet (this scan is
                # what determines it), but scan only ever reads the `Tape` marker below, never this
                # call's own `ci` — nothing to resolve yet, just report the marker.
                #
                # Not only an optimization: resolving something here regardless (e.g. a fixed
                # `Ctx{Nothing}` placeholder) would break this — the emission side genuinely depends
                # on compiling the `Ctx{TapeT}` sibling as a nested step of building `Ctx{Nothing}`,
                # so if that sibling's own scan eagerly tried to compile its apparent self-edge
                # target too, it would recurse straight back into the still-mid-construction
                # `Ctx{Nothing}` build, and `in_progress` would (correctly) flag that as a cycle.
                own_TapeT === nothing && return nothing, reverse_fwds_impl, CoDual{R,NoFData}, Tape
                tt_self = Tuple{typeof(reverse_fwds_impl),CoDual{ftype,NoFData},Ctx{own_TapeT},
                                argcodualtys...}
                push!(edges, tt_self, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
                # `CC.findall`, not `findsup` — see the ABI note above `helper_ci`: `findall` gives a
                # `spec_types` already intersected with `tt_self`, so `specialize_method` yields a
                # `MethodInstance` whose `specTypes` is exactly `tt_self`.
                matches = CC.findall(tt_self, CC.method_table(interp))
                if matches === nothing || length(matches) != 1 || !matches[1].fully_covers
                    reason[] = "no reverse-mode rule resolves for recursive (self-cyclic) call " *
                               "signature $(tt_self)"
                    return nothing
                end
                callee_impl_mi = specialize_method(matches[1])::MethodInstance
                if callee_impl_mi === impl_mi
                    CC.add_inlining_edge!(edges, callee_impl_mi)
                    return callee_impl_mi, reverse_fwds_impl, CoDual{R,NoFData}, Tape
                else
                    ci = CC.typeinf_ext_toplevel(interp, callee_impl_mi, CC.SOURCE_MODE_ABI)::CodeInstance
                    CC.add_invoke_edge!(edges, tt_self, ci)
                    return ci, reverse_fwds_impl, CoDual{R,NoFData}, Tape
                end
            end
        end
        fm, _ = CC.findsup(tt, CC.method_table(interp))
        if fm === nothing || !isa(fm.method, Method)
            reason[] = "no reverse-mode rule resolves for recursive call signature $(tt)"
            return nothing
        end
        callee_val = reverse_fwds_impl
    end
    push!(edges, tt, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
    callee_impl_mi = specialize_method(fm.method, fm.spec_types, fm.sparams)::MethodInstance
    ci = CC.typeinf_ext_toplevel(interp, callee_impl_mi, CC.SOURCE_MODE_ABI)::CodeInstance
    InnerRT = ci.rettype
    # The pullback half (`InnerRT.parameters[2]`) is left unconstrained beyond "some concrete type":
    # the derived path always returns a `Tape{...}`, but a hand-written rule is free to use whatever's
    # cheapest to remember — this glue never inspects a pullback's internals.
    if !(InnerRT isa DataType && InnerRT <: Tuple && length(InnerRT.parameters) == 2 &&
         InnerRT.parameters[1] isa DataType && InnerRT.parameters[1] <: CoDual &&
         isconcretetype(InnerRT.parameters[2]))
        # `Union{}` means the callee's own carrier bailed and only its error stub compiled; anything
        # else means a rule resolved but its return type isn't the concrete `(CoDual, pullback)`
        # pair the emission below needs.
        inner = get(interp.custom_state.bail_reasons, callee_impl_mi, nothing)
        reason[] = inner !== nothing ? "the reverse-mode build for `$(tt)` bailed: $(inner)" :
            InnerRT === Union{} ?
                "the reverse rule resolved for `$(tt)` never returns — either its own derived " *
                "build bailed, or it cannot run on those argument types" :
                "the reverse rule resolved for `$(tt)` returned `$(InnerRT)`, which is not a " *
                "concrete `(CoDual, pullback)` pair"
        return nothing
    end
    CC.add_invoke_edge!(edges, tt, ci)
    InnerFCoDualT = InnerRT.parameters[1]
    InnerTapeT = InnerRT.parameters[2]
    callee_val === reverse_fwds_impl || return ci, callee_val, InnerFCoDualT, InnerTapeT
    # The emission site always hands an ordinary (non-hand-ruled, non-self-recursive) derived callee
    # a recycled tape (`_inner_ctx`), never a fresh `Ctx()` — so the `:invoke` actually needed
    # targets the `Ctx{InnerTapeT}` pre-allocated sibling, not the `Ctx{Nothing}` variant `ci` above
    # (which exists only to learn `InnerTapeT`, the callee's own tape type). Resolving the sibling is
    # the same `findall` + `typeinf_ext_toplevel` two-step, against a different `tt`; both variants
    # are verified to return an identical `(CoDual, Tape)` pair, so a mismatch here is an
    # internal-error bail, not a recoverable one.
    tt2 = Tuple{typeof(reverse_fwds_impl),CoDual{ftype,NoFData},Ctx{InnerTapeT},argcodualtys...}
    push!(edges, tt2, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
    matches2 = CC.findall(tt2, CC.method_table(interp))
    if matches2 === nothing || length(matches2) != 1 || !matches2[1].fully_covers
        reason[] = "no reverse-mode rule resolves for the pre-allocated recursive call signature $(tt2)"
        return nothing
    end
    callee_impl_mi2 = specialize_method(matches2[1])::MethodInstance
    ci2 = CC.typeinf_ext_toplevel(interp, callee_impl_mi2, CC.SOURCE_MODE_ABI)::CodeInstance
    if ci2.rettype !== InnerRT
        reason[] = "the pre-allocated (`Ctx{<:Tape}`) recursive build for `$(tt2)` returned " *
                   "$(ci2.rettype), which does not match the tape-allocating variant's $(InnerRT)"
        return nothing
    end
    CC.add_invoke_edge!(edges, tt2, ci2)
    # (ci, callee_val, InnerFCoDualT, InnerPullbackT)
    return ci2, callee_val, InnerFCoDualT, InnerTapeT
end

# Mirrors `reverse_fwds_recursive_ci` for the pullback carrier: resolves
# `reverse_pullback_impl(tape::InnerTapeT, seed::InnerSeedT)`. Simpler than the fwds case — by the
# time the pullback builder needs this, `InnerTapeT` is already a known-concrete type (resolved
# once already, while building the fwds pass; both builders derive it identically), so no "did the
# callee bail" recovery logic is needed beyond a defensive rettype-shape check.
#
# Cyclic edge (self-recursion): `InnerTapeT` arrives as the bare `Tape` UnionAll — the marker
# `reverse_fwds_recursive_ci` leaves in a cyclic block's comms type — never a concrete type there,
# so it can't be resolved into a `tt` directly. The concrete type is `own_TapeT`, the current
# pullback build's own tape type: a self-recursive callee's tape is by construction the same type as
# the caller's own. The callee's rettype is likewise closed-form regardless of whether compiling
# turns out to be needed below, computed straight from `own_codualparams`.
#
# Whether compiling is needed mirrors the fwds side's reasoning, but the source of the mismatch
# differs: the pullback carrier has no ctx-type variance, but the recursive call's own seed type
# (`InnerSeedT`) need not equal the current build's own incoming seed type even for a literally
# self-recursive primal. `pb_mi === impl_mi` catches exactly the case where it does — codegen's
# self-recursion fast path, no compile. Otherwise `pb_mi` is a genuine, not-yet-compiled sibling
# (same tape, different seed type); that nested compile's own recursive edge targets its own seed
# type, so it resolves via the literal-identity branch and terminates.
function reverse_pullback_recursive_ci(interp, impl_mi::MethodInstance, @nospecialize(own_TapeT),
                                       own_codualparams::Vector{Any}, @nospecialize(InnerTapeT),
                                       @nospecialize(InnerSeedT), edges::Vector{Any}, reason::Ref{String})
    if InnerTapeT === Tape
        tt = Tuple{typeof(reverse_pullback_impl),own_TapeT,InnerSeedT}
        push!(edges, tt, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
        # `CC.findall`, not `findsup` — same ABI reasoning as `reverse_fwds_recursive_ci`.
        matches = CC.findall(tt, CC.method_table(interp))
        if matches === nothing || length(matches) != 1 || !matches[1].fully_covers
            reason[] = "no pullback method resolves for recursive (self-cyclic) call signature $(tt)"
            return nothing
        end
        pb_mi = specialize_method(matches[1])::MethodInstance
        InnerRdatasT = Tuple{(zero_like_rdata_type(_codual_primal_type(c)) for c in own_codualparams)...}
        if pb_mi === impl_mi
            CC.add_inlining_edge!(edges, pb_mi)
            return pb_mi, true, InnerRdatasT
        else
            ci = CC.typeinf_ext_toplevel(interp, pb_mi, CC.SOURCE_MODE_ABI)::CodeInstance
            CC.add_invoke_edge!(edges, tt, ci)
            return ci, true, InnerRdatasT
        end
    end
    # Mirrors the fwds side's two shapes. A derived inner pullback is a `Tape`, so we target its
    # carrier directly rather than the generated `(t::Tape)(seed)` entry — the entry is only a
    # one-line `:invoke` wrapper, and skipping it keeps this a single static call. A hand-written
    # pullback is a method on its own compiler-generated closure type, so the object is its own callee.
    derived = InnerTapeT isa Type && InnerTapeT <: Tape
    tt = derived ? Tuple{typeof(reverse_pullback_impl),InnerTapeT,InnerSeedT} :
                   Tuple{InnerTapeT,InnerSeedT}
    push!(edges, tt, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
    fm, _ = CC.findsup(tt, CC.method_table(interp))
    if fm === nothing || !isa(fm.method, Method)
        reason[] = "no pullback method resolves for recursive call signature $(tt)"
        return nothing
    end
    callee_impl_mi = specialize_method(fm.method, fm.spec_types, fm.sparams)::MethodInstance
    ci = CC.typeinf_ext_toplevel(interp, callee_impl_mi, CC.SOURCE_MODE_ABI)::CodeInstance
    if !(ci.rettype isa DataType && ci.rettype <: Tuple)
        inner = get(interp.custom_state.bail_reasons, callee_impl_mi, nothing)
        reason[] = inner !== nothing ? "the reverse-mode pullback build for `$(tt)` bailed: $(inner)" :
            ci.rettype === Union{} ?
                "the pullback resolved for `$(tt)` never returns — either its own derived build " *
                "bailed, or it cannot run on that tape/seed" :
                "the pullback resolved for `$(tt)` returned `$(ci.rettype)`, not a tuple of rdatas"
        return nothing
    end
    CC.add_invoke_edge!(edges, tt, ci)
    return ci, derived, ci.rettype
end

# ===========================================================================
# Static resolution of the runtime helpers this engine emits (`push!`/`pop!`/`increment!!`/`Stack`
# construction/…), so they can be emitted as `Expr(:invoke, ci, f, args...)` rather than
# `Expr(:call, f, args...)`.
#
# Why this is mandatory, not an optimization: both carriers build their IR by hand, so every
# statement carries `CC.NoCallInfo()` (`CC.InstructionStream(len)` in `reverse_fwds_to_ircode`;
# `new_inst` in `cfg_ir.jl`). Inference never ran over this IR, so nothing populates `info`, and
# `assemble_inline_todo!` bails on exactly that ("Inference determined this couldn't be analyzed.
# Don't question it.", `Compiler/src/ssair/inlining.jl`). A surviving `:call` to a non-builtin
# therefore compiles to a full `jl_apply_generic`: boxed arguments, method lookup, boxed return — so a
# single `increment!!(::Float64, ::Float64)`, which is one `add_float`, costs three heap allocations
# and a dynamic dispatch on every accumulation of every iteration. `:invoke` sidesteps the problem:
# `assemble_inline_todo!` dispatches `handle_invoke_expr!` before the `NoCallInfo` check, so an
# `:invoke` gets the unboxed specsig ABI and stays eligible for inlining. Same fix forward mode already
# uses for surviving high-level calls (`frule_codeinstance` + the `Expr(:invoke, ci, ...)` emission in
# `forward_interp.jl`).
#
# Two details that are load-bearing:
#
#  * `CC.findall`, not `CC.findsup`. `findall` returns matches whose `spec_types` are already
#    intersected with `tt`, so `specialize_method` yields a concrete `MethodInstance` and the
#    `:invoke` uses the specialized (unboxed) ABI. `findsup` can hand back the method's own widened
#    signature, which would still box every argument — a static call, but not a fast one.
#
#  * Compile under `interp`, not a fresh `CC.NativeInterpreter`. `resolve_todo` looks the callee up
#    in `code_cache(state.interp)`, keyed by `cache_owner`; a `CodeInstance` compiled by a
#    `NativeInterpreter` lands in a different cache, so it would be found by codegen but never by
#    the inliner. Routing through `interp` is safe: `build_contextual_ir` returns `nothing` for any
#    non-carrier MethodInstance, so `push!`/`increment!!`/... infer and inline as they would natively.
#
# Returns `nothing` if the signature doesn't resolve to a single method (the caller falls back to a
# plain `:call` — slow, but correct). `cache` memoizes per builder invocation, since the same
# handful of signatures recur at many sites.
function helper_ci(interp, @nospecialize(tt::Type), edges::Vector{Any}, cache::IdDict{Any,Any})
    cached = get(cache, tt, nothing)
    cached === nothing || return cached::CodeInstance
    matches = CC.findall(tt, CC.method_table(interp))
    (matches === nothing || length(matches) != 1) && return nothing
    match = matches[1]
    match.fully_covers || return nothing
    mi = specialize_method(match)::MethodInstance
    ci = CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_ABI)::CodeInstance
    CC.add_invoke_edge!(edges, tt, ci)
    cache[tt] = ci
    return ci
end

# For each primal block, the list of tagged `(kind, node)` comms items — genuine `SSAValue`/
# `Argument` operands (or, for recursion, the call's own `SSAValue`) that must be communicated from
# forwards to pullback. `getfield`/`memoryrefnew` need no runtime value (their reverse rules route
# by static type + node identity only, or are rebuilt deterministically by both passes
# independently), so only intrinsic operands (`:primal`), recursive-call inner tapes (`:subtape`),
# and tracked array reads (`:shadow_ref`) ever need comms.
#
# NOT side-effect-free once recursion is present: resolving a `:subtape` candidate compiles that
# callee's own `reverse_fwds_impl` (via `reverse_fwds_recursive_ci`) — a deliberate departure from
# "pure static scan". Both builders still derive it identically because `cache_owner` is a
# mode-level singleton: a `CodeInstance` compiled while scanning from one builder's `interp` is
# found, not recompiled, when the other builder's separate `interp` resolves the same callsite.
function _scan_block_comms(interp, scan_impl_mi::MethodInstance, primal_mi::MethodInstance, pir, iworld,
                           unreachable, codualparams::Vector{Any}, reason::Ref{String}, edges::Vector{Any})
    nblocks = length(pir.cfg.blocks)
    nodes = [Any[] for _ in 1:nblocks]
    types = [Any[] for _ in 1:nblocks]
    # A direct self-recursive call's inner tape is routed through `Tape.subtapes` (a single,
    # global, `Tape`-wide stack), not a per-block comms item: every such call in this primal targets
    # the identical concrete type. `self_recursive_ssas` records which statement indices resolved
    # this way; `block_has_subtape` records which blocks contain at least one, needed by
    # `_collapsible_regions` so it doesn't mistake such a block for comms-free.
    self_recursive_ssas = BitSet()
    block_has_subtape = falses(nblocks)
    n = length(codualparams)
    fdata_tracked = _fdata_tracked(pir, iworld, n, codualparams)
    arg_tracked = _arg_fdata_tracked(iworld, n, codualparams)
    # A `Core.Argument`'s primal value never needs a comms slot: `Tape.args` already holds every
    # argument codual, and the pullback's own `pb_presolve` already falls back to `parg_pb[a.n]`
    # whenever no comms item was declared for it. Guarded on type: a declared item's type can be
    # narrower than `parg_pb[a.n]`'s when the primal's own const-prop narrowed `pir.argtypes[a.n]`
    # past the argument's nominal type — eliding then would hand a rule a too-wide value.
    elide_argument_primal(item, @nospecialize(ty)) =
        item[1] === :primal && isa(item[2], Core.Argument) &&
        _widen(ty) === _codual_primal_type(codualparams[item[2].n])
    # Bulk-saved arguments, decided before any item is declared: a store into one of them needs
    # neither its old value nor the primal `MemoryRef` to put it back through, so this changes what
    # `builtin_rrule_comms` declares (via `ctx.bulk_saved` below).
    arg_primal_types = Any[_codual_primal_type(c) for c in codualparams]
    bulk_args = _bulk_save_args(pir, iworld, arg_primal_types)
    bulk_saved(@nospecialize ref_node) = _is_bulk_saved(pir, iworld, bulk_args, ref_node)
    block_of = _stmt_block_map(pir)
    for i in 1:length(pir.stmts)
        b = block_of[i]
        unreachable[b] && continue
        s = pir.stmts[i][:stmt]
        if isa(s, Expr) && s.head === :new
            # A tracked mutable `%new` needs its fresh shadow communicated forward to the pullback,
            # like any other object's `(:fshadow, obj)` item — except here the object is this
            # statement's own result, keyed by its own `SSAValue` rather than an existing operand.
            T = _calleeval(s.args[1], iworld)
            # `T <: Array` excluded: an array's shadow is a plain `Array`, never referenced via a
            # `(:fshadow, obj)` comms item (only produced by `getfield`/`setfield!` on a genuinely
            # mutable struct). Declaring one here would just be dead weight on every array allocation.
            if T isa DataType && !(T <: Array) && ismutabletype(T) && fdtype(iworld, T) !== NoFData
                push!(nodes[b], (:fshadow, Core.SSAValue(i)))
                push!(types[b], fdtype(iworld, T))
            end
            continue
        elseif isa(s, Expr) && s.head === :foreigncall
            # `foreigncalls_reverse.jl` decides scope, mirroring `Core.Builtin` below. Unlike a
            # builtin, an unregistered target always bails (native code can write through any pointer,
            # so no primal-only replay fallback).
            fc = _fc_parse(s)
            ctx = (pstmt=(x::Core.SSAValue) -> pir.stmts[x.id][:stmt],
                  calleeval=(@nospecialize x) -> _calleeval(x, iworld),
                  optype=(@nospecialize x) -> _optype(pir, x), reason=reason,
                  tt=(@nospecialize(T) -> _tt(iworld, _widen(T))),
                  tracked=fdata_tracked, arg_tracked=arg_tracked, ssa=Core.SSAValue(i))
            if fc === nothing
                reason[] = "reverse mode does not support a `:foreigncall` with a runtime function " *
                           "pointer target at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            result = foreigncall_rrule_comms(Val(fc.name), fc, pir.stmts[i][:type], ctx)
            result === false && return nothing
            if result === nothing
                reason[] = "reverse mode does not support foreigncall target `:$(fc.name)` at %$i: " *
                           "`$(_stmt_str(s))`"
                return nothing
            end
            for (item, ty) in result
                any(nd -> nd == item, nodes[b]) && continue
                elide_argument_primal(item, ty) && continue
                push!(nodes[b], item)
                push!(types[b], ty)
            end
            continue
        end
        (isa(s, Expr) && (s.head === :call || s.head === :invoke)) || continue
        fpos, actual = _call_parts(s)
        f = _calleeval(fpos, iworld)
        if isa(f, Core.IntrinsicFunction)
            # Two reasons an intrinsic operand needs no comms slot, both decided purely from the
            # primal IR so fwds and pullback builders derive identical tuple types: the statement
            # carries no gradient at all (`NoRData` result), or the rule for this specific intrinsic
            # doesn't read that operand (see `intrinsic_rrule_operands` — a linear op like
            # `add_float` reads neither addend).
            rdtype(iworld, pir.stmts[i][:type]) === NoRData && continue
            needed = intrinsic_rrule_operands(Val(f))
            for (k, a) in enumerate(actual)
                (isa(a, Core.SSAValue) || isa(a, Core.Argument)) || continue
                (needed === nothing || k in needed) || continue
                item = (:primal, a)
                any(nd -> nd == item, nodes[b]) && continue
                ty = _optype(pir, a)
                elide_argument_primal(item, ty) && continue
                push!(nodes[b], item)
                push!(types[b], ty)
            end
        elseif isa(f, Core.Builtin)
            # The dispatch layer (`builtins_reverse.jl`) decides everything: whether this call needs
            # comms items, and whether it's even in scope. `tt`/`rdtype`/`fdtype` are
            # world-parameterized funnels: the builtin rules must not call `tangent_type` directly,
            # or their dispatch is pinned to the generator's own world.
            ctx = (optype=(@nospecialize x) -> _optype(pir, x), ssa=Core.SSAValue(i),
                  tracked=fdata_tracked, arg_tracked=arg_tracked, reason=reason,
                  bulk_saved=bulk_saved,
                  tt=(@nospecialize(T) -> _tt(iworld, _widen(T))),
                  rdtype=(@nospecialize(T) -> rdtype(iworld, T)),
                  fdtype=(@nospecialize(T) -> fdtype(iworld, T)),
                  # A `MemoryRef` statically re-derivable from an argument + literal index need not
                  # be pushed (see `_static_ref_derivation`). Consulted by `builtins_reverse.jl`.
                  static_ref=(@nospecialize x) -> _static_ref_derivation(pir, iworld, x))
            result = builtin_rrule_comms(Val(f), actual, pir.stmts[i][:type], ctx)
            result === false && return nothing
            if result !== nothing
                for (item, ty) in result
                    any(nd -> nd == item, nodes[b]) && continue
                    elide_argument_primal(item, ty) && continue
                    push!(nodes[b], item)
                    push!(types[b], ty)
                end
            end
        else
            # A surviving high-level call: attempt static recursion. Not qualifying is not an error
            # at scan time (mirrors how an unregistered intrinsic isn't flagged here either); only a
            # genuine attempted-and-failed resolution propagates as a real bail here.
            info = _static_recursible_call(pir, iworld, i, s, Ref(""), arg_tracked, fdata_tracked)
            info === nothing && continue
            _, ftype, argtypes = info
            R = _widen(pir.stmts[i][:type])
            resolved = reverse_fwds_recursive_ci(interp, scan_impl_mi, primal_mi, ftype, argtypes, R,
                                                 edges, reason)
            if resolved === nothing
                reason[] *= " — at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            # For a cyclic edge this is the bare `Tape` UnionAll — the marker meaning "route through
            # `Tape.subtapes`, not a comms item". A concrete `TapeT` isn't known yet at scan time
            # anyway (it depends on this very comms scan).
            InnerTapeT = resolved[4]
            if InnerTapeT === Tape
                push!(self_recursive_ssas, i)
                block_has_subtape[b] = true
            else
                push!(nodes[b], (:subtape, Core.SSAValue(i)))
                push!(types[b], InnerTapeT)
            end
        end
    end

    # Hoist a loop-invariant `(:primal, %v)` item out of a looped consumer block `b` into `%v`'s own
    # defining block `d`'s comms slot, instead of re-declaring it on `b`'s repeatedly-pushed stack.
    # All three must hold: (1) `inloop[b]` — the push actually repeats; (2) `!inloop[d]` — `d` runs
    # at most once per call, so `%v` is invariant w.r.t. every loop (SSA already guarantees `d`
    # dominates `b`); (3) `%v`'s type, and every item already in `d`, are `isbits` — what keeps `d`'s
    # slot a `CommsCell` rather than a `Stack`. A `Stack` would break this: in reverse order `d` is
    # replayed after `b`, so a stack pop by `d`'s own reverse code would come too late for `b`'s
    # (earlier) reverse code to have already read it — a `CommsCell` is a plain `getfield`, safe to
    # read from any block, any number of times, in any order.
    # `block_hoisted_refs[b]` records, per hoisted item, which block's slot the pullback should read
    # it from instead of popping it out of `b`'s own comms stack.
    inloop = _loop_blocks(pir)
    block_hoisted_refs = [Tuple{Any,Int,Int}[] for _ in 1:nblocks]
    d_all_isbits = Bool[all(isbitstype, types[d]) for d in 1:nblocks]
    for b in 1:nblocks
        (inloop[b] && !isempty(nodes[b])) || continue
        keep = trues(length(nodes[b]))
        for (idx, item) in enumerate(nodes[b])
            item[1] === :primal || continue
            node = item[2]
            isa(node, Core.SSAValue) || continue
            d = block_of[node.id]
            (d == b || inloop[d]) && continue
            ty = types[b][idx]
            (isbitstype(ty) && d_all_isbits[d]) || continue
            # Reuse `%v`'s own slot in `d` if `d` already declares it natively (e.g. `%v` is also
            # used by a later statement inside `d` itself); otherwise give it a fresh one.
            slot = findfirst(==(item), nodes[d])
            if slot === nothing
                push!(nodes[d], item)
                push!(types[d], ty)
                slot = length(nodes[d])
            end
            push!(block_hoisted_refs[b], (item, d, slot))
            keep[idx] = false
        end
        all(keep) && continue
        nodes[b] = nodes[b][keep]
        types[b] = types[b][keep]
    end

    # Must run before the fusion loop below, which empties `nodes[b]` for fused-away blocks.
    # `_collapsible_regions` decides "comms-free" by reading `nodes`; a block emptied by fusion would
    # then wrongly match as a region interior (`chk`/`pass`) on a recompute, and the pullback skips a
    # region interior's reverse processing entirely, silently dropping that block's rdata routing.
    regions, quiet = _collapsible_regions(pir, unreachable, nodes, block_has_subtape)

    # Comms fusion: if `b`'s only successor is `c` and `c`'s only predecessor is `b`, the two run the
    # same number of times in a fixed order, so `b`'s items can ride along on `c`'s stack instead of
    # pushing their own — one push per visit instead of two. Canonical shape: one block pushes an
    # array index, its successor pushes the loaded element.
    #
    # `c`'s pop is legal to read from `b`'s reverse block because the pullback walks backwards: `c`'s
    # reverse block runs first, and since `b` has exactly one successor, `b`'s reverse block has
    # exactly one predecessor — `c`'s — which therefore dominates it.
    #
    # Guards beyond the CFG shape: both blocks must already have items (fusing into an empty block
    # is no win), and both tuples must agree on `isbits`-ness (mixing a GC-tracked value into an
    # isbits tuple would add a write barrier that wasn't there before).
    #
    # `block_fused_refs[b]` records, per moved item, which block's popped tuple (and which slot in
    # it) the pullback should read instead of popping `b`'s own stack.
    succs = [pir.cfg.blocks[b].succs for b in 1:nblocks]
    preds = [filter(!=(0), pir.cfg.blocks[b].preds) for b in 1:nblocks]
    # `nextb[b]`: `b`'s control-equivalent successor, or 0. Pure CFG, independent of what is declared.
    nextb = zeros(Int, nblocks)
    for b in 1:nblocks
        unreachable[b] && continue
        length(succs[b]) == 1 || continue
        c = only(succs[b])
        (c == b || unreachable[c]) && continue
        (length(preds[c]) == 1 && only(preds[c]) == b) || continue
        nextb[b] = c
    end
    block_fused_refs = [Tuple{Any,Int,Int}[] for _ in 1:nblocks]
    # `host[b]`: which block's stack `b`'s items ended up on (itself, if it kept them). Resolved
    # tail-first so a chain `a -> b -> c` puts all three blocks' items on `c` rather than stalling at
    # `b`. `host` doubles as the memo (0 = not yet resolved) and, via the provisional
    # self-assignment, as a guard against a cycle in `nextb`.
    host = zeros(Int, nblocks)
    resolve_host!(b) = begin
        host[b] != 0 && return host[b]
        host[b] = b
        c = nextb[b]
        c == 0 && return b
        t = resolve_host!(c)
        (t == b || isempty(nodes[b]) || isempty(nodes[t])) && return b
        isbitstype(Tuple{types[b]...}) == isbitstype(Tuple{types[t]...}) || return b
        for (idx, item) in enumerate(nodes[b])
            # Reuse `t`'s slot if it already declares the identical item.
            slot = findfirst(==(item), nodes[t])
            if slot === nothing
                push!(nodes[t], item)
                push!(types[t], types[b][idx])
                slot = length(nodes[t])
            end
            push!(block_fused_refs[b], (item, t, slot))
        end
        empty!(nodes[b])
        empty!(types[b])
        return host[b] = t
    end
    for b in 1:nblocks
        resolve_host!(b)
    end

    return nodes, types, bulk_args, self_recursive_ssas, block_has_subtape, block_hoisted_refs,
           block_fused_refs, regions, quiet
end

# ===========================================================================
# Forwards pass: 1:1 block-topology-preserving replay of the primal (built the same way
# `dualize_to_ircode` is, minus any shadow/tangent — see this file's header), instrumented with the
# block-stack/comms pushes described above.
# ===========================================================================
function reverse_fwds_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int, nfixed::Int,
                                primal_mi::MethodInstance;
                                reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    pstmts = pir.stmts
    N = length(pstmts)
    nblocks = length(pir.cfg.blocks)
    iworld = CC.get_inference_world(interp)

    if isa(pstmts[1][:stmt], Core.PhiNode)
        reason[] = "primal IR has a leading PhiNode in block 1 (unsupported shape)"
        return nothing
    end
    unreachable_block = _unreachable_blocks(pir)
    exit_blocks = _exit_blocks(pir, unreachable_block)
    if isempty(exit_blocks)
        reason[] = "primal has no reachable `return` (every path throws) — reverse mode cannot " *
                   "differentiate a function that never returns"
        return nothing
    end

    # Carrier is `reverse_fwds_impl(fcd, ctx, argcds...)`: `params[2]` is `fcd`, `params[3]` is the
    # `ctx`, `params[4:end]` are the argument coduals. `codualparams`/`ArgsTT` are the full codual list
    # `(fcd, args...)` — the shape the rest of this builder and the pullback side both expect.
    # `vararg_tt` is only the argument coduals: it types the packed vararg slot `Argument(4)`.
    codualparams = Any[impl_mi.specTypes.parameters[2], impl_mi.specTypes.parameters[4:end]...]
    vararg_tt = Tuple{impl_mi.specTypes.parameters[4:end]...}
    ArgsTT = Tuple{codualparams...}
    packed_codualparams = _packed_codualparams(iworld, codualparams, nfixed)

    scan = _scan_block_comms(interp, impl_mi, primal_mi, pir, iworld, unreachable_block, packed_codualparams, reason, edges)
    scan === nothing && return nothing
    # `bulk_args`: argument positions whose primal contents are saved/restored once per call rather
    # than per overwritten element. Derived inside the scan so both builders get it identically —
    # they must agree exactly, since it decides which comms items exist.
    # `block_hoisted_refs` needs no handling here: a hoisted `:primal` item is emitted from whichever
    # block now owns it, same as any other comms item. `block_fused_refs` is bound (not discarded):
    # the `:subtape` inner-tape recycling lookup below reads a block's comms stack directly rather
    # than through `presolve`, so it needs to know where a fused item actually landed.
    block_comms_nodes, block_comms_types, bulk_args, self_recursive_ssas, _, _, block_fused_refs,
        regions, quiet = scan
    bulk_slot = Dict(k => j for (j, k) in enumerate(sort!(collect(bulk_args))))
    is_unique_pred, _, _ = _unique_predecessor_info(pir, exit_blocks, unreachable_block, regions, quiet)
    # Same predicate `_scan_block_comms` used to decide the comms items, recomputed here so the
    # emission sides agree with the declaration side by construction.
    pb_bulk_saved(@nospecialize ref_node) = _is_bulk_saved(pir, iworld, bulk_args, ref_node)
    fdata_tracked = _fdata_tracked(pir, iworld, length(packed_codualparams), packed_codualparams)
    arg_tracked = _arg_fdata_tracked(iworld, length(packed_codualparams), packed_codualparams)

    getf = GlobalRef(Core, :getfield)
    setf = GlobalRef(Core, :setfield!)
    ctuple = GlobalRef(Core, :tuple)
    push_g = Base.push!
    zerofcodual_g = zero_fcodual

    code = Any[]; types = Any[]
    emit!(ex, @nospecialize(ty)) = (push!(code, ex); push!(types, ty); Core.SSAValue(length(code)))
    opf(name, ty, args...) = emit!(Expr(:call, GlobalRef(Core.Intrinsics, name), args...), ty)

    # Emit a call to a runtime helper as a static `:invoke` (see `helper_ci` for why a plain `:call`
    # here would be a full dynamic dispatch). `argtys` is the helper's declared argument types —
    # already known at every call site, since every emitted SSA value carries its type. `f` goes in the
    # `:invoke`'s callee value position as the function/type object, never a `GlobalRef` (`verify_ir`
    # rejects a `GlobalRef` there — see the same note in `dualize_to_ircode`).
    hcache = IdDict{Any,Any}()
    icall!(@nospecialize(f), @nospecialize(ty), argtys::Tuple, args...) = begin
        ci = helper_ci(interp, Tuple{Core.Typeof(f),argtys...}, edges, hcache)
        ci === nothing ? emit!(Expr(:call, f, args...), ty) :
                         emit!(Expr(:invoke, ci, f, args...), ty)
    end

    # --- Argument-unpacking prologue: primal always; fdata (shadow) too, for any argument whose
    # fdata is non-trivial — consumed only by Part 2's array shadow chain below, but extracted
    # generically here for any argument type it happens to apply to. ---
    parg = Vector{Any}(undef, n)
    farg = Vector{Any}(undef, n)
    carg = Vector{Any}(undef, n)   # the coduals themselves — stored on the tape for the pullback
    for i in 1:n
        Ci = codualparams[i]
        Pi = _codual_primal_type(Ci)
        Fi = _codual_fdata_type(Ci)
        # codual 1 is `fcd` — `Argument(2)` directly; coduals 2..n are the argument coduals, packed in
        # the vararg tuple `Argument(4)` at position `i-1`. `Argument(3)` is the `ctx`.
        ci = i == 1 ? Core.Argument(2) : emit!(Expr(:call, getf, Core.Argument(4), i - 1), Ci)
        carg[i] = ci
        parg[i] = emit!(Expr(:call, getf, ci, 1), Pi)
        Fi !== NoFData && (farg[i] = emit!(Expr(:call, getf, ci, 2), Fi))
    end
    # `carg`/`args_tup_ssa`/`ArgsTT` stay flat: the tape and the pullback's return arity are keyed
    # by the caller's actual argument list.
    _pack_vararg_args!(emit!, ctuple, parg, farg, codualparams, iworld, nfixed)
    # Packed once here rather than at each use: both tape shapes below need it, and the
    # pre-allocated shape stores it in the prologue (not at the return) so an early bail can't sink
    # the store past a point where the tape is already visible to the caller.
    args_tup_ssa = emit!(Expr(:call, ctuple, carg...), ArgsTT)

    # --- Tape prologue. Two shapes, chosen by the `ctx` type in `Argument(3)`. ---
    # Select `CommsCell{T}` (single-slot inline holder) for a non-loop block whose comms tuple is
    # `isbits` (pushed once per call): the carrier emits a direct `setfield!`/`getfield`, no
    # `push!`/`pop!`/`position`/boundscheck. A loop block or non-`isbits` tuple keeps `Stack{T}`; an
    # empty tuple uses `SingletonStack`.
    comms_stack_ty = Vector{Any}(undef, nblocks)
    inloop = _loop_blocks(pir)
    for b in 1:nblocks
        CommsT = Tuple{block_comms_types[b]...}
        if Base.issingletontype(CommsT)
            comms_stack_ty[b] = SingletonStack{CommsT}
        else
            static_ok = !inloop[b] && isbitstype(CommsT)
            comms_stack_ty[b] = static_ok ? CommsCell{CommsT} : Stack{CommsT}
        end
    end
    TapeT = Tape{ArgsTT,Tuple{comms_stack_ty...}}
    comms_stack_ssa = Vector{Any}(undef, nblocks)

    CtxT = impl_mi.specTypes.parameters[3]
    PreTapeT = (CtxT isa DataType && CtxT <: Ctx) ? CtxT.parameters[1] : Nothing
    tape_ssa = nothing
    if PreTapeT === Nothing
        # Tape-allocating mode: build the block stack, one comms stack per block, and the
        # self-recursion `subtapes` stack (empty — a primal with no self-recursive call never pushes to
        # it, and `_alloc_tape`'s cold path never needs one either since it's rebuilt fresh here).
        block_stack_ssa = icall!(Stack{Int32}, Stack{Int32}, ())
        for b in 1:nblocks
            ST = comms_stack_ty[b]
            # `SingletonStack` has no fields, so `%new`. `CommsCell{T}()` zero-fills `val`.
            # `Stack{T}` takes a capacity (pre-size to 1 for a single push).
            comms_stack_ssa[b] = ST <: SingletonStack ? emit!(Expr(:new, ST), ST) : icall!(ST, ST, ())
        end
        subtapes_ssa = icall!(Stack{TapeT}, Stack{TapeT}, ())
    else
        # Pre-allocated mode: read the caller's tape out of the `ctx` and reuse its stacks — the
        # whole point of `build_ctx(...; prealloc=true)`, since a `Stack` is three heap objects.
        if PreTapeT !== TapeT
            reason[] = "the pre-allocated tape has type $(PreTapeT), but this primal's tape shape is " *
                       "$(TapeT) — rebuild the context with `build_ctx` for these exact argument types"
            return nothing
        end
        tape_ssa = emit!(Expr(:call, getf, Core.Argument(3), 1), TapeT)
        block_stack_ssa = emit!(Expr(:call, getf, tape_ssa, 1), Stack{Int32})
        comms_tuple_in = emit!(Expr(:call, getf, tape_ssa, 2), Tuple{comms_stack_ty...})
        for b in 1:nblocks
            comms_stack_ssa[b] = emit!(Expr(:call, getf, comms_tuple_in, b), comms_stack_ty[b])
        end
        subtapes_ssa = emit!(Expr(:call, getf, tape_ssa, 4), Stack{TapeT})
        # Reset every reusable stack to empty. Balanced push/pop already leaves them at 0 after a
        # completed round trip, but resetting here makes reuse correct unconditionally — a caller
        # can run the forwards pass and never call the pullback, or bail out partway through.
        emit!(Expr(:call, setf, block_stack_ssa, 2, 0), Any)
        for b in 1:nblocks
            ST = comms_stack_ty[b]
            (ST <: SingletonStack || ST <: CommsCell) && continue
            # `Stack` only: reset `position` (field 2) to 0. `CommsCell` is overwritten in place;
            # `SingletonStack` is empty.
            emit!(Expr(:call, setf, comms_stack_ssa[b], 2, 0), Any)
        end
        emit!(Expr(:call, setf, subtapes_ssa, 2, 0), Any)
        # This call's coduals replace the previous call's. A re-used context therefore keeps the
        # previous call's arguments (and their shadows) alive until the next call overwrites them —
        # noted in `build_ctx`'s docstring; the field is concretely typed, so there's nothing to null
        # it to in between. `args` is field 5 now that `subtapes` (field 4) sits before it.
        emit!(Expr(:call, setf, tape_ssa, 5, args_tup_ssa), Any)
    end

    # --- Bulk primal save. Still in the prologue, before any primal statement has run, so it captures
    # the arguments exactly as the call found them. The buffers live on the tape, so a pre-allocated
    # context reuses them across calls and a steady-state call allocates nothing; a `Ctx()` call
    # allocates each buffer once, on its tape's first (and only) use. ---
    bufs_ssa = if isempty(bulk_args)
        # Shared empty sentinel: nothing will ever index it, and this way a primal that bulk-saves
        # nothing pays no allocation for the field.
        emit!(GlobalRef(@__MODULE__(), :_NO_BULK_BUFS), Vector{Any})
    elseif tape_ssa === nothing
        icall!(Vector{Any}, Vector{Any}, ())
    else
        emit!(Expr(:call, getf, tape_ssa, 3), Vector{Any})
    end
    for k in sort!(collect(bulk_args))
        # `bulk_args` is packed-space; never the tail slot (`_bulk_save_args` only bulk-saves an
        # `Array`/`Memory`, never a `Tuple`).
        Pk = _widen(_codual_primal_type(packed_codualparams[k]))
        icall!(_bulk_save!, Nothing, (Vector{Any}, Int, Pk), bufs_ssa, bulk_slot[k], parg[k])
    end

    primal_map = Vector{Any}(undef, N)
    shadow_map = Vector{Any}(undef, N)   # Part 2: array shadow (MemoryRef) chain, sparse — only
                                          # `fdata_tracked[i]` entries are ever assigned or read.
    # A bare `GlobalRef` operand (e.g. a `ReturnNode`'s `.val` for a function ending `return nothing`)
    # must be resolved to its actual value here, not passed through: an unresolved `GlobalRef` outside
    # `Core`/`Base` (a `Main` binding, say) is illegal in value position (`verify_ir`) — the same
    # hazard `_calleeval` exists to handle for callees (`forward_interp.jl`).
    presolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? primal_map[x.id] : isa(x, Core.Argument) ? parg[x.n] :
        isa(x, GlobalRef) ? _calleeval(x, iworld) : x
    sresolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? shadow_map[x.id] : isa(x, Core.Argument) ? farg[x.n] : x

    # Part 1: which block's own recursive-call `:subtape` comms item already has its inner `Tape`
    # SSA value computed, so `emit_epilogue!` can find it (sparse, parallel to `primal_map`).
    inner_tape_map = Dict{Int,Any}()

    # Part 3: `:old_primal`/`:old_tangent` comms items (`builtins_reverse.jl`) are computed by a
    # rule's own fwds emission rather than resolved from an existing node — each such rule returns a
    # `saved::Dict{Any,Any}` mapping its declared item to the SSA value holding it, merged in here per
    # block for `emit_epilogue!` to look up.
    block_saved = [Dict{Any,Any}() for _ in 1:nblocks]

    # Forward-reference patches for a `PhiNode` operand not yet resolved when the phi is processed (a
    # loop back-edge: the operand is defined later in linear statement order). Keyed by the referenced
    # original SSA index; each entry is (target values-vector, slot, want_primal) — mirrors
    # `dualize_to_ircode`'s `pending`; `want_primal` picks `primal_map` vs `shadow_map` once SSA `i`
    # is resolved.
    pending = Dict{Int,Vector{Tuple{Vector{Any},Int,Bool}}}()

    block_start_new = Vector{Int}(undef, nblocks)
    block_start_new[1] = 1
    bidx = 1

    emit_epilogue!(b) = begin
        nodes = block_comms_nodes[b]
        ST = comms_stack_ty[b]
        # Nothing to communicate: the stack is a `SingletonStack` whose `push!` is a no-op, so the
        # tuple construction and the push are both pure overhead. (The pullback already skips the
        # matching `pop!` for this case.)
        if !isempty(nodes)
            vals = (nd[1] === :primal ? presolve(nd[2]) :
                    nd[1] === :subtape ?
                        get(() -> error("Differ internal error: comms item $(nd) was declared but " *
                                        "its inner tape was never recorded"),
                            inner_tape_map, nd[2].id) :
                    (nd[1] === :old_primal || nd[1] === :old_tangent) ?
                        get(() -> error("Differ internal error: comms item $(nd) was declared but " *
                                        "never saved by its rule's fwds emission (builtin_rrule_comms/" *
                                        "apply_builtin_rrule_fwds! disagree)"),
                            block_saved[b], nd) :
                    sresolve(nd[2]) for nd in nodes)
            CommsT = Tuple{block_comms_types[b]...}
            tup = emit!(Expr(:call, ctuple, vals...), CommsT)
            if ST <: CommsCell
                # Unrolled single-slot store: `setfield!` to `val` (field 1) — no `push!`/`position`/boundscheck
                emit!(Expr(:call, setf, comms_stack_ssa[b], 1, tup), Any)
            else
                icall!(push_g, Any, (ST, CommsT), comms_stack_ssa[b], tup)
            end
        end
        # Skip the block-stack push when `b` is a unique predecessor of whatever runs next (an
        # ordinary successor, or — for a lone exit — the pullback's own entry routing): nothing
        # downstream can ever be ambiguous about having come from `b`, so there's nothing to record.
        if !is_unique_pred[b]
            icall!(push_g, Any, (Stack{Int32}, Int32), block_stack_ssa, Int32(b))
        end
        return nothing
    end

    for i in 1:N
        while bidx < nblocks && i > pir.cfg.blocks[bidx].stmts.stop
            if length(code) < block_start_new[bidx]
                emit!(nothing, Nothing)
            end
            bidx += 1
            block_start_new[bidx] = length(code) + 1
        end
        s = pstmts[i][:stmt]; Ti = pstmts[i][:type]
        is_terminator = i == pir.cfg.blocks[bidx].stmts.stop
        # Every reachable, non-throw block pushes its own comms + block number before whatever comes
        # next — an ordinary successor block, or the pullback's own entry, which pops this same
        # stack to learn which of possibly several reachable exits actually ran (a branch with a
        # `return` in each arm — the common case) and routes accordingly. Not conditioned on "the
        # terminator is an explicit GotoNode/GotoIfNot": some fallthrough blocks have no explicit
        # terminator, yet still have a real successor.
        #
        # Defer past any non-control-transfer terminator: a `PhiNode` must lead its block
        # (`verify_ir`), and a value-producing terminator can own this block's own comms item
        # (`:subtape`/`:fshadow`/`:old_primal`/`:old_tangent`), unresolvable until it's emitted.
        # `_split_ambiguous_block_pushes` later relocates the `GotoIfNot` case per-edge.
        is_ctrl_transfer = isa(s, Core.GotoNode) || isa(s, Core.GotoIfNot) || isa(s, Core.ReturnNode)
        defer_epilogue = is_terminator && !unreachable_block[bidx] && !is_ctrl_transfer
        if is_terminator && !unreachable_block[bidx] && !defer_epilogue
            emit_epilogue!(bidx)
        end
        if unreachable_block[bidx]
            if isa(s, Core.ReturnNode)
                emit!(Core.ReturnNode(), Union{})
            elseif isa(s, Expr) && s.head === :invoke
                fv = _calleeval(s.args[2], iworld)
                ex = Expr(:invoke, s.args[1], fv === nothing ? presolve(s.args[2]) : fv,
                          (presolve(a) for a in s.args[3:end])...)
                primal_map[i] = emit!(ex, Ti)
            elseif isa(s, Expr) && s.head === :call
                fv = _calleeval(s.args[1], iworld)
                ex = Expr(:call, fv === nothing ? presolve(s.args[1]) : fv,
                          (presolve(a) for a in s.args[2:end])...)
                primal_map[i] = emit!(ex, Ti)
            elseif isa(s, Expr) && s.head === :new
                primal_map[i] = emit!(Expr(:new, s.args[1], (presolve(a) for a in s.args[2:end])...), Ti)
            elseif isa(s, Expr) && s.head === :boundscheck
                primal_map[i] = emit!(Expr(:boundscheck, s.args...), Ti)
            elseif isa(s, Expr) && s.head === :throw_undef_if_not
                # Pure control marker (undef-var/boxed-capture guard) — see the matching arm in the main
                # (reachable-block) loop below for the full rationale. `args[1]` (name) is copied
                # verbatim; `args[2]` (condition) is resolved like any other operand.
                primal_map[i] = emit!(Expr(:throw_undef_if_not, s.args[1], presolve(s.args[2])), Ti)
            elseif isa(s, Expr) && s.head === :loopinfo
                # Same treatment (and same `julia.ivdep` filtering) as the main loop's `:loopinfo` arm
                # below — see there for the full rationale.
                args = filter(a -> a !== Symbol("julia.ivdep"), s.args)
                primal_map[i] = emit!(Expr(:loopinfo, args...), Ti)
            elseif isa(s, Expr) && s.head === :gc_preserve_begin
                # Primal-only: nothing in a throw-only block carries a tangent, so no shadow to root.
                primal_map[i] = emit!(Expr(:gc_preserve_begin, (presolve(a) for a in s.args)...), Any)
            elseif isa(s, Expr) && s.head === :gc_preserve_end
                primal_map[i] = emit!(Expr(:gc_preserve_end, presolve(s.args[1])), Ti)
            elseif isa(s, Expr) && s.head === :foreigncall
                # Primal-only, mirroring `dualize_to_ircode`'s unreachable-block arm (nothing to
                # route). `args[2:5]` are literals copied verbatim; `args[1]` too, except the
                # runtime-function-pointer form, which needs resolving like any operand.
                nm = s.args[1]
                pnm = isa(nm, Expr) ? Expr(nm.head, nm.args...) : presolve(nm)
                primal_map[i] = emit!(Expr(:foreigncall, pnm, s.args[2], s.args[3], s.args[4], s.args[5],
                                           (presolve(a) for a in s.args[6:end])...), Ti)
            elseif isa(s, Core.PiNode)
                primal_map[i] = presolve(s.val)
            elseif isa(s, GlobalRef)
                primal_map[i] = emit!(s, Ti)
            elseif !isa(s, Expr)
                primal_map[i] = presolve(s)
            else
                reason[] = "unexpected statement kind $(typeof(s)) in an unreachable (throw-only) " *
                           "block at %$i: `$(_stmt_str(s))`"
                return nothing
            end
        elseif isa(s, Core.ReturnNode)
            if !isdefined(s, :val)
                reason[] = "internal error — unreachable ReturnNode in a reachable block at %$i"
                return nothing
            end
            ret_val = presolve(s.val)
            # `_optype` has no world to resolve a bare `GlobalRef` with (unlike `presolve` above), so a
            # `GlobalRef`-valued return (`return nothing` at `Main` scope, say) needs its type read off
            # the already-resolved `ret_val` instead of the unresolved node. `_widen`: `_optype` is
            # lattice-faithful and can return a `Core.PartialStruct` (e.g. a tuple return narrowed by
            # const-prop), but `fcodual_type` below needs a bare `Type`.
            R = isa(s.val, GlobalRef) ? Core.Typeof(ret_val) : _widen(_optype(pir, s.val))
            FR = _fcdtype(iworld, R)
            has_shadow = isa(s.val, Core.SSAValue) ? isassigned(shadow_map, s.val.id) :
                         isa(s.val, Core.Argument) ? isassigned(farg, s.val.n) : false
            result_cd = if fdtype(iworld, R) === NoFData
                icall!(zerofcodual_g, FR, (R,), ret_val)
            elseif has_shadow
                emit!(Expr(:new, FR, ret_val, sresolve(s.val)), FR)
            else
                reason[] = "reverse mode cannot return a value carrying fdata ($(R)) whose shadow " *
                           "is not traceable to a function argument at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            # Pre-allocated mode returns the caller's own tape object; otherwise `%new` one around the
            # stacks the prologue just built.
            tape = tape_ssa
            if tape === nothing
                comms_tuple = emit!(Expr(:call, ctuple, comms_stack_ssa...), Tuple{comms_stack_ty...})
                tape = emit!(Expr(:new, TapeT, block_stack_ssa, comms_tuple, bufs_ssa, subtapes_ssa,
                                  args_tup_ssa), TapeT)
            end
            final = emit!(Expr(:call, ctuple, result_cd, tape), Tuple{FR,TapeT})
            emit!(Core.ReturnNode(final), Any)
        elseif isa(s, Core.PiNode)
            primal_map[i] = presolve(s.val)
            fdata_tracked[i] && (shadow_map[i] = sresolve(s.val))
        elseif isa(s, Expr) && s.head === :new
            # `_calleeval` resolves a `GlobalRef` type argument (the common post-optimization shape,
            # e.g. `%new(Main.MPoint, ...)`) at the inference world; a literal `DataType` passes
            # through unchanged.
            T = _calleeval(s.args[1], iworld)
            if !(T isa DataType)
                reason[] = "reverse mode `%new` needs a statically-resolvable type at %$i: " *
                           "`$(_stmt_str(s))`"
                return nothing
            end
            if !is_always_fully_initialised(T) && length(s.args) - 1 != fieldcount(T)
                reason[] = "reverse mode does not support a partially-initialised `%new` of a " *
                           "struct with possibly-undef fields ($(T)) at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            args = @view s.args[2:end]
            primal_map[i] = emit!(Expr(:new, T, (presolve(a) for a in args)...), Ti)
            if T <: Array
                # Array allocation step 4 of 4 (`%new(Vector{P}, ref, size)`): the shadow is a real
                # same-shape `Array{tangent_type(P),N}`, never a `MutableTangent` — must run before
                # the generic mutable-struct branch below (`ismutabletype(Vector) === true`). Exactly
                # 2 fields: `:ref` (differentiable, `sresolve`d) and `:size` (structural, `presolve`d).
                TT = _tt(iworld, T)
                shadow_map[i] = emit!(Expr(:new, TT, sresolve(args[1]), presolve(args[2])), TT)
            elseif ismutabletype(T) && fdtype(iworld, T) !== NoFData
                # Fresh shadow `MutableTangent`, mirroring Mooncake's `_new_` rrule: each field's
                # initial tangent is `zero_tangent(field_primal, field_fdata)`, which aliases the
                # assigned value's own shadow when that field carries fdata instead of fabricating a
                # detached zero — the same aliasing `Core.setfield!`'s rule relies on. This SSA is a
                # tracked provenance root (`_fdata_tracked` marks it), so downstream `getfield`/
                # `setfield!` on it resolve to this shadow exactly like a tracked function argument.
                field_tangents = Any[]
                for (j, a) in enumerate(args)
                    Fty = fieldtype(T, j)
                    FTj = fdtype(iworld, Fty)
                    if FTj !== NoFData
                        tracked_here = isa(a, Core.SSAValue) ? (a.id <= length(fdata_tracked) && fdata_tracked[a.id]) :
                                       isa(a, Core.Argument) ? (a.n <= length(arg_tracked) && arg_tracked[a.n]) : false
                        if !tracked_here
                            reason[] = "reverse mode `%new` of a mutable struct with a field " *
                                       "($(Fty)) whose assigned value's fdata is not traceable to a " *
                                       "function argument is not supported at %$i: `$(_stmt_str(s))`"
                            return nothing
                        end
                        fdata_val = sresolve(a)
                    else
                        fdata_val = NoFData()
                    end
                    push!(field_tangents, icall!(_rr_zero_tangent2, _tt(iworld, Fty), (Fty, FTj), presolve(a), fdata_val))
                end
                argtys = (Type{T}, Tuple(_tt(iworld, fieldtype(T, j)) for j in eachindex(args))...)
                shadow_map[i] = icall!(_rr_build_tangent, _tt(iworld, T), argtys, T, field_tangents...)
            elseif !ismutabletype(T) && fdata_tracked[i]
                # Tracked immutable `%new`: no `MutableTangent` of its own — fdata is `FData` wrapping
                # the tangent tree built from each field's tangent (as the mutable branch above builds
                # a `MutableTangent`), then wrapped via `_rr_fdata`. `_fdata_tracked` already required
                # every fdata-carrying field to be tracked, so no per-field check needed here.
                field_tangents = Any[]
                for (j, a) in enumerate(args)
                    Fty = fieldtype(T, j)
                    FTj = fdtype(iworld, Fty)
                    fdata_val = FTj !== NoFData ? sresolve(a) : NoFData()
                    push!(field_tangents, icall!(_rr_zero_tangent2, _tt(iworld, Fty), (Fty, FTj), presolve(a), fdata_val))
                end
                argtys = (Type{T}, Tuple(_tt(iworld, fieldtype(T, j)) for j in eachindex(args))...)
                tangent_ssa = icall!(_rr_build_tangent, _tt(iworld, T), argtys, T, field_tangents...)
                shadow_map[i] = icall!(_rr_fdata, fdtype(iworld, T), (_tt(iworld, T),), tangent_ssa)
            end
        elseif isa(s, Expr) && s.head === :boundscheck
            primal_map[i] = emit!(Expr(:boundscheck, s.args...), Ti)
        elseif isa(s, Expr) && s.head === :throw_undef_if_not
            # Pure control marker: raises `UndefVarError`/`UndefRefError` for an unassigned slot or
            # boxed-capture field (the guard around a captured, reassigned variable's read). `args[1]`
            # is a bare Symbol/GlobalRef name, never a value to resolve; `args[2]` is the Bool
            # condition, a genuine operand here (a literal in a throw-only block, handled in the
            # unreachable-block arm above, or an SSA reference on a live path). No fdata: its result is
            # never consumed.
            primal_map[i] = emit!(Expr(:throw_undef_if_not, s.args[1], presolve(s.args[2])), Ti)
        elseif isa(s, Expr) && s.head === :loopinfo
            # `@simd`'s loop marker. Pure metadata, copied through like `:boundscheck` above (no
            # `shadow_map` entry): its operands are `Symbol`s/`nothing` and never traversed by
            # `userefs`, so `presolve`ing them would be wrong, not just unnecessary.
            #
            # One difference from forward mode: `julia.ivdep` is dropped, `julia.simdloop` and
            # anything else pass through. `ivdep` asserts no loop-carried memory dependence, which
            # forward mode's shadow honestly preserves but the reverse carrier does not —
            # `emit_epilogue!` pushes onto the tape's stacks on every iteration, which *is* a
            # loop-carried dependence. Keeping the marker would be a silent miscompile under
            # vectorization; dropping it only costs the vectorization itself.
            args = filter(a -> a !== Symbol("julia.ivdep"), s.args)
            primal_map[i] = emit!(Expr(:loopinfo, args...), Ti)
        elseif isa(s, Expr) && s.head === :gc_preserve_begin
            # `GC.@preserve`: roots operands until `:gc_preserve_end`. Carrier holds interior pointers
            # into both primal and shadow, so root both; an untracked operand (`fdata_tracked`/
            # `arg_tracked`) isn't ours to root.
            pargs = Any[]
            for a in s.args
                push!(pargs, presolve(a))
                tracked_a = isa(a, Core.SSAValue) ? (a.id <= length(fdata_tracked) && fdata_tracked[a.id]) :
                            isa(a, Core.Argument) ? (a.n <= length(arg_tracked) && arg_tracked[a.n]) : false
                tracked_a && push!(pargs, sresolve(a))
            end
            primal_map[i] = emit!(Expr(:gc_preserve_begin, pargs...), Any)
        elseif isa(s, Expr) && s.head === :gc_preserve_end
            # References the `:gc_preserve_begin` token by SSAValue to end the region.
            primal_map[i] = emit!(Expr(:gc_preserve_end, presolve(s.args[1])), Ti)
        elseif isa(s, Expr) && s.head === :foreigncall
            # Dispatch layer (`foreigncalls_reverse.jl`) decides everything, mirroring `Core.Builtin`
            # below; `_scan_block_comms` already bailed on an unregistered/out-of-scope target, so a
            # rule applies here.
            fc = _fc_parse(s)
            fcctx = (; emit!, icall!, presolve, sresolve,
                    optype=(@nospecialize x) -> _optype(pir, x),
                    pstmt=(x::Core.SSAValue) -> pir.stmts[x.id][:stmt],
                    calleeval=(@nospecialize x) -> _calleeval(x, iworld),
                    tt=(@nospecialize(T) -> _tt(iworld, _widen(T))),
                    fdtype=(@nospecialize(T) -> fdtype(iworld, T)),
                    tracked=fdata_tracked, arg_tracked=arg_tracked, ssa=Core.SSAValue(i))
            result = fc === nothing ? nothing : apply_foreigncall_rrule_fwds!(Val(fc.name), fc, Ti, fcctx)
            if result === nothing
                reason[] = "reverse mode does not support foreigncall target " *
                           "`$(fc === nothing ? "(runtime function pointer)" : ":$(fc.name)")` at " *
                           "%$i: `$(_stmt_str(s))`"
                return nothing
            end
            p, shadow, saved = result
            primal_map[i] = p
            shadow !== nothing && (shadow_map[i] = shadow)
            isempty(saved) || merge!(block_saved[bidx], saved)
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos, actual = _call_parts(s)
            f = _calleeval(fpos, iworld)
            if isa(f, Core.IntrinsicFunction)
                primal_map[i] = emit!(Expr(:call, f, (presolve(a) for a in actual)...), Ti)
            elseif isa(f, Core.Builtin)
                bctx = (; emit!, icall!, presolve, sresolve,
                       optype=(@nospecialize x) -> _optype(pir, x), tracked=fdata_tracked,
                       ssa=Core.SSAValue(i), bulk_saved=pb_bulk_saved,
                       tt=(@nospecialize(T) -> _tt(iworld, _widen(T))),
                       rdtype=(@nospecialize(T) -> rdtype(iworld, T)),
                       fdtype=(@nospecialize(T) -> fdtype(iworld, T)))
                result = apply_builtin_rrule_fwds!(Val(f), actual, Ti, bctx)
                if result !== nothing
                    p, shadow, saved = result
                    primal_map[i] = p
                    shadow !== nothing && (shadow_map[i] = shadow)
                    isempty(saved) || merge!(block_saved[bidx], saved)
                elseif _tt(iworld, _widen(Ti)) === NoTangent
                    # A non-differentiable builtin result (`===`, `isa`, comparisons, ... — e.g. the
                    # `Base.iterate`-state check a `for i in 1:length(x)` loop's own lowering embeds) has
                    # no gradient to route, so no rule/comms is needed — just replay it primally, the
                    # same treatment `GlobalRef`/literal statements already get.
                    primal_map[i] = emit!(Expr(:call, f, (presolve(a) for a in actual)...), Ti)
                else
                    reason[] = "reverse mode does not support builtin `$(f)` with a differentiable " *
                               "result ($(Ti)) and no reverse rule at %$i: `$(_stmt_str(s))`"
                    return nothing
                end
            else
                info = _static_recursible_call(pir, iworld, i, s, reason, arg_tracked, fdata_tracked)
                info === nothing && return nothing
                fval, ftype, argtypes = info
                R = _widen(Ti)
                resolved = reverse_fwds_recursive_ci(interp, impl_mi, primal_mi, ftype, argtypes, R,
                                                     edges, reason; own_TapeT=TapeT)
                if resolved === nothing
                    reason[] *= " — at %$i: `$(_stmt_str(s))`"
                    return nothing
                end
                ci, callee_val, InnerFCoDualT, InnerTapeT0 = resolved
                # Cyclic edge: `reverse_fwds_recursive_ci` leaves this the bare `Tape` UnionAll (see its
                # docstring); the concrete type is this build's own `TapeT`, already computed above — a
                # self-recursive callee's tape is, by construction, the same type as the caller's own
                # (which is also `own_TapeT`, passed into the very call above that produced `resolved`).
                InnerTapeT = InnerTapeT0 === Tape ? TapeT : InnerTapeT0
                FCT = CoDual{ftype,NoFData}
                # `fval === nothing` means `_static_recursible_call` resolved `ftype` from the operand's
                # type, not its value (an argument-position callee) — `presolve(fpos)` resolves the
                # operand itself uniformly, exactly as it already does for the argument coduals below.
                fcodual = emit!(Expr(:new, FCT, fval === nothing ? presolve(fpos) : fval, NoFData()), FCT)
                argcoduals = Any[]
                for (j, a) in enumerate(actual)
                    Cj = _fcdtype(iworld, argtypes[j])
                    # Thread the argument's real shadow through when its fdata is non-trivial (an
                    # array whose identity `_static_recursible_call` already confirmed is traceable
                    # to a function argument) — `sresolve` resolves both a bare `Core.Argument` and a
                    # tracked `SSAValue` uniformly. A hardcoded `NoFData()` here would silently detach
                    # the callee's accumulation from the caller's real buffer.
                    fdata_val = fdtype(iworld, argtypes[j]) === NoFData ? NoFData() : sresolve(a)
                    push!(argcoduals, emit!(Expr(:new, Cj, presolve(a), fdata_val), Cj))
                end
                # Uniform layout `(fcd, ctx, argcds...)` for both callees. Recycle a tape instead of
                # allocating a fresh one every call for any derived (non-hand-ruled) callee — only a
                # hand rule's pullback (`callee_val !== reverse_fwds_impl`) has no `Tape` to
                # pre-allocate, and keeps the fresh-tape `Ctx()`. A self-recursive edge
                # (`InnerTapeT0 === Tape`) recycles from the tape's own dedicated `subtapes` stack via
                # `_inner_self_ctx`. An ordinary nested callee still recycles from the comms slot its
                # own `:subtape` push will land in — a `Stack` never deallocates, so
                # after the first execution of this block the slot already holds a structurally
                # identical inner tape.
                is_self_edge = InnerTapeT0 === Tape
                ctx_val = if callee_val !== reverse_fwds_impl
                    Ctx()
                elseif is_self_edge
                    icall!(_inner_self_ctx, Ctx{TapeT}, (Stack{TapeT},), subtapes_ssa)
                else
                    # Comms fusion may have moved this `(:subtape, %i)` item off `bidx`'s own comms
                    # stack onto a control-equivalent successor's; `block_fused_refs[bidx]` records
                    # where it landed. Fall back to `bidx` itself when it wasn't fused.
                    item = (:subtape, Core.SSAValue(i))
                    fused = findfirst(fr -> fr[1] == item, block_fused_refs[bidx])
                    host_b, k = fused === nothing ? (bidx, nothing) : block_fused_refs[bidx][fused][2:3]
                    ST = comms_stack_ty[host_b]
                    @assert ST <: Stack "a `:subtape` comms item must force a real `Stack` (never " *
                                        "isbits/singleton — a `Tape` is always mutable), got $(ST)"
                    if k === nothing
                        k = findfirst(nd -> nd == item, block_comms_nodes[host_b])
                    end
                    icall!(_inner_ctx, Ctx{InnerTapeT}, (ST, Val{k}, Type{InnerTapeT}),
                           comms_stack_ssa[host_b], Val(k), InnerTapeT)
                end
                pair = emit!(Expr(:invoke, ci, callee_val, fcodual, ctx_val, argcoduals...),
                            Tuple{InnerFCoDualT,InnerTapeT})
                result_cd = emit!(Expr(:call, getf, pair, 1), InnerFCoDualT)
                primal_map[i] = emit!(Expr(:call, getf, result_cd, 1), Ti)
                if fdtype(iworld, Ti) !== NoFData
                    # Route the callee's returned shadow so a caller can accumulate into it; bail
                    # if its declared fdata type doesn't match this call's inferred one.
                    InnerFT = InnerFCoDualT.parameters[2]
                    if InnerFT !== fdtype(iworld, Ti)
                        reason[] = "recursive call's resolved result fdata type ($(InnerFT)) does " *
                                   "not match this call's own inferred fdata type " *
                                   "($(fdtype(iworld, Ti))) at %$i: `$(_stmt_str(s))`"
                        return nothing
                    end
                    shadow_map[i] = emit!(Expr(:call, getf, result_cd, 2), InnerFT)
                end
                inner_tape_ssa = emit!(Expr(:call, getf, pair, 2), InnerTapeT)
                if is_self_edge
                    # Pushed immediately at the call site rather than deferred to `emit_epilogue!` like
                    # ordinary comms: `_scan_block_comms` never declared a `(:subtape, ...)` comms node
                    # for a self-recursive statement, so there's nothing for `emit_epilogue!` to look up
                    # for it. Safe regardless of where in the block this statement sits:
                    # `_split_ambiguous_block_pushes`/`_is_expected_block_push` only ever classifies a
                    # block whose primal terminator is a `GotoIfNot`, which a call statement never is, so
                    # this push can't be mistaken for the block-stack push.
                    icall!(push_g, Any, (Stack{TapeT}, TapeT), subtapes_ssa, inner_tape_ssa)
                else
                    inner_tape_map[i] = inner_tape_ssa
                end
            end
        elseif isa(s, Core.GotoNode)
            emit!(Core.GotoNode(s.label), Any)
        elseif isa(s, Core.GotoIfNot)
            emit!(Core.GotoIfNot(presolve(s.cond), s.dest), Any)
        elseif isa(s, Core.PhiNode)
            k = length(s.values)
            pvals = Vector{Any}(undef, k)
            want_shadow = fdata_tracked[i]
            svals = want_shadow ? Vector{Any}(undef, k) : nothing
            for j in 1:k
                isassigned(s.values, j) || continue
                v = s.values[j]
                if isa(v, Core.SSAValue) && !isassigned(primal_map, v.id)
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (pvals, j, true))
                    want_shadow && push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (svals, j, false))
                else
                    pvals[j] = presolve(v)
                    want_shadow && (svals[j] = sresolve(v))
                end
            end
            primal_map[i] = emit!(Core.PhiNode(s.edges, pvals), Ti)
            # Shadow phi, emitted right after the primal one to stay in the block's leading phi run
            # (`verify_ir`). `rdata_type` of an fdata-carrying value is always `NoRData`, so no
            # pullback comms needed; `getfield`/`memoryrefget` off this phi resolves via `sresolve`
            # like any tracked node.
            want_shadow && (shadow_map[i] = emit!(Core.PhiNode(s.edges, svals), fdtype(iworld, Ti)))
        elseif isa(s, GlobalRef)
            primal_map[i] = emit!(s, Ti)
        elseif !isa(s, Expr)
            primal_map[i] = presolve(s)
        else
            reason[] = "unsupported statement kind $(typeof(s)) at %$i: `$(_stmt_str(s))`"
            return nothing
        end
        if haskey(pending, i)
            for (arr, slot, wantp) in pending[i]
                arr[slot] = wantp ? primal_map[i] : shadow_map[i]
            end
            delete!(pending, i)
        end
        # Deferred (see above); still within this block's range.
        if defer_epilogue
            emit_epilogue!(bidx)
        end
    end
    if length(code) < block_start_new[nblocks]
        emit!(nothing, Nothing)
    end
    if !isempty(pending)   # unreachable on well-formed IR; bail, don't emit invalid IR
        reason[] = "internal error — an unresolved forward-reference remained (a bug in this " *
                   "transform, not unsupported input)"
        return nothing
    end

    len = length(code)
    stream = CC.InstructionStream(len)
    for i in 1:len
        stream.stmt[i] = code[i]
        stream.type[i] = types[i]
        stream.flag[i] = CC.IR_FLAG_NULL
    end
    new_blocks = Vector{CC.BasicBlock}(undef, nblocks)
    for b in 1:nblocks
        lo = block_start_new[b]
        hi = b == nblocks ? len : block_start_new[b + 1] - 1
        ob = pir.cfg.blocks[b]
        new_blocks[b] = CC.BasicBlock(CC.StmtRange(lo, hi), copy(ob.preds), copy(ob.succs))
    end
    cfg = CC.CFG(new_blocks, Int[bb.stmts.stop + 1 for bb in new_blocks])
    di = CC.DebugInfoStream(stream.line)
    di.def = impl_mi
    # `#self#`, `fcd`, `ctx`, then the packed vararg tuple of *argument* coduals — matching
    # `_impl_argtypes(impl_mi)`, which is what the bail path (`reverse_error_ircode`) uses.
    argtypes = Any[impl_mi.specTypes.parameters[1], impl_mi.specTypes.parameters[2],
                   impl_mi.specTypes.parameters[3], vararg_tt]
    # `pir.valid_worlds`, not the constructor's unbounded default — see the matching comment in
    # `forward_interp.jl`'s `dualize_to_ircode` and `cfg_ir.jl`'s `lower_cfg_blocks_to_ir`.
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[], pir.valid_worlds)
    # `_split_ambiguous_block_pushes` (fixes ISSUES #52): relocate the per-block block-stack push
    # onto only the ambiguous arm of a mixed `GotoIfNot`, making the forwards push per-edge — the
    # companion to the per-edge `pred_is_unique_pred` formula above (`length(preds[b]) <= 1`). The
    # two changes are coupled; neither is correct alone (a first fwds-only attempt at this shipped
    # verify_ir-clean IR and passed its own tests but silently corrupted gradients for any loop
    # iterating >= 2 times — see `_emit_switch!` below for the root cause).
    ir = _split_ambiguous_block_pushes(ir, pir, is_unique_pred)
    CC.verify_ir(ir)
    return ir
end

# Pullback pass: walks the primal's blocks in reverse, over a freshly built CFG (not 1:1 with the
# primal — extra phi-routing blocks are inserted, and multi-way predecessor dispatch is lowered from
# a `Switch` into a `GotoIfNot` chain), using the `ID`/`CFGBlock` layer from `cfg_ir.jl`. rdata
# accumulators are real mutable `Ref`s, one per primal SSA + one per argument.

# `Base.pop!`, not a bare `pop!`: inlined into synthetic carrier IR, where a bare call would
# resolve as `GlobalRef(Differ, :pop!)` and trip `verify_ir`.
@inline __pop_blk_stack!(block_stack) = Base.pop!(block_stack)::Int32
# Fully qualified for the same reason as `__pop_blk_stack!` above.
@inline __switch_case(id::Int32, prev::Int32) = Base.:!(Core.:(===)(id, prev))

# Materializes a real zero rdata from `acc` when it's the `ZeroRData` placeholder. Needed wherever a
# `deref_and_zero!`-derived accumulator must be treated as an actual `RDataT` value (e.g. decomposed
# field-by-field for an immutable `%new`) rather than merely `increment!!`-ed into. `@noinline`: gets
# threaded through `icall` into hand-built carrier IR; without it, inlining the tiny body would
# re-embed a bare call as `GlobalRef(Differ, ...)`, which `verify_ir` rejects.
@noinline _rr_realize_rdata(acc, ::Type{RDataT}) where {RDataT} =
    (acc isa ZeroRData ? zero_rdata_from_type(RDataT) : acc)::RDataT

function reverse_pullback_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int, nfixed::Int,
                                    primal_mi::MethodInstance;
                                    reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    pstmts = pir.stmts
    N = length(pstmts)
    nblocks = length(pir.cfg.blocks)
    iworld = CC.get_inference_world(interp)

    unreachable_block = _unreachable_blocks(pir)
    exit_blocks = _exit_blocks(pir, unreachable_block)
    if isempty(exit_blocks)
        reason[] = "primal has no reachable `return` (every path throws) — reverse mode cannot " *
                   "differentiate a function that never returns"
        return nothing
    end

    params = impl_mi.specTypes.parameters
    TapeT = params[2]
    PullbackSeedT = params[3]
    ArgsTT = TapeT.parameters[1]
    codualparams = Any[ArgsTT.parameters...]
    CS = TapeT.parameters[2]
    comms_stack_ty = Any[CS.parameters...]
    packed_codualparams = _packed_codualparams(iworld, codualparams, nfixed)

    scan = _scan_block_comms(interp, impl_mi, primal_mi, pir, iworld, unreachable_block, packed_codualparams, reason, edges)
    scan === nothing && return nothing
    # `bulk_args`: argument positions whose primal contents are saved/restored once per call rather
    # than per overwritten element — derived inside the scan so both builders agree.
    # `block_hoisted_refs[b]`: cross-block reads — for each `(item, d, slot)`, block `b` reads `item`
    # out of block `d`'s own comms slot instead of popping it locally. `block_fused_refs[b]`: same
    # `(item, host, slot)` shape, but read out of the tuple the *host* block's reverse code popped.
    block_comms_nodes, block_comms_types, bulk_args, self_recursive_ssas, _,
        block_hoisted_refs, block_fused_refs, regions, quiet = scan
    _, pred_is_unique_pred, _ = _unique_predecessor_info(pir, exit_blocks, unreachable_block, regions, quiet)
    bulk_slot = Dict(k => j for (j, k) in enumerate(sort!(collect(bulk_args))))
    # Same predicate `_scan_block_comms` used to decide the comms items, recomputed here so the
    # emission sides agree with the declaration side by construction.
    pb_bulk_saved(@nospecialize ref_node) = _is_bulk_saved(pir, iworld, bulk_args, ref_node)
    fdata_tracked = _fdata_tracked(pir, iworld, length(packed_codualparams), packed_codualparams)
    arg_tracked = _arg_fdata_tracked(iworld, length(packed_codualparams), packed_codualparams)

    getf = GlobalRef(Core, :getfield)
    setf = GlobalRef(Core, :setfield!)
    ctuple = GlobalRef(Core, :tuple)
    pop_g = Base.pop!
    increment_g = increment!!

    # Build the statement for a call to a runtime helper: a static `:invoke` when the callee resolves,
    # else a plain `:call` (see `helper_ci` for why the difference matters). Returns the `Expr` rather
    # than emitting it, because this builder has several distinct emit closures (`eemit!` for the entry
    # block, `remit!` per exit route, `emit!` per reverse block) and every one of them needs it.
    # `argtys` is the helper's declared argument types, known at each site from the types of the values
    # being passed.
    hcache = IdDict{Any,Any}()
    icall(@nospecialize(f), argtys::Tuple, args...) = begin
        ci = helper_ci(interp, Tuple{Core.Typeof(f),argtys...}, edges, hcache)
        ci === nothing ? Expr(:call, f, args...) : Expr(:invoke, ci, f, args...)
    end

    # Which block each statement belongs to (reused throughout).
    stmt_block = _stmt_block_map(pir)

    # Every statement except a pure control marker (or one living in a throw-only block) gets a `Ref`
    # to accumulate rdata into; literal/GlobalRef/`:boundscheck`/`:loopinfo` operands never do (both
    # always have `NoRData`, so skipping them here just avoids a useless allocation — not load-bearing).
    needs_ref(i) = !unreachable_block[stmt_block[i]] &&
                   !isa(pstmts[i][:stmt], Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode}) &&
                   !(isa(pstmts[i][:stmt], Expr) && (pstmts[i][:stmt]::Expr).head in (:boundscheck, :loopinfo))

    entry_id = ID()
    block_id = [ID() for _ in 1:nblocks]
    # The tuple block `b`'s reverse code pops. Minted up front rather than by `emit!`, because a block
    # whose items were fused onto a later-numbered host must name the host's popped tuple before the
    # host itself is emitted. Forward references are fine — `_ids_to_line_numbers` resolves every `ID`
    # against one map built over all blocks at once — the only real obligation is dominance, which the
    # fusion criterion already establishes.
    comms_popped_id = [ID() for _ in 1:nblocks]

    # The node each exit block returns — potentially a different one per exit (e.g. each arm of an
    # if/else returning its own value; see `_exit_blocks`).
    exit_ret_node = Dict(b => pstmts[pir.cfg.blocks[b].stmts.stop][:stmt].val for b in exit_blocks)

    # --- Entry block: unpack the tape and allocate every rdata `Ref`. Which exit actually ran isn't
    # known statically (there may be several — an ordinary branch returning early in each arm is the
    # common case, not a corner case), so which one gets seeded from the incoming `seed` is decided by
    # a runtime switch below, exactly like a `PhiNode`'s per-predecessor routing. ---
    entry_stmts = IDInstPair[]
    eemit!(ex, @nospecialize(ty)) = begin
        id = ID()
        push!(entry_stmts, (id, new_inst(ex, ty)))
        id
    end

    tape_id = eemit!(Core.Argument(2), TapeT)
    seed_id = Core.Argument(3)
    block_stack_id = eemit!(Expr(:call, getf, tape_id, 1), Stack{Int32})
    comms_tuple_id = eemit!(Expr(:call, getf, tape_id, 2), CS)
    # Direct self-recursion's dedicated storage (`Tape.subtapes`, field 4). Unconditionally unpacked
    # (unused/DCE'd by `run_ipo_passes!` when this primal has no self-recursive call).
    subtapes_id = eemit!(Expr(:call, getf, tape_id, 4), Stack{TapeT})

    comms_obj_id = Vector{Any}(undef, nblocks)
    for b in 1:nblocks
        isempty(block_comms_types[b]) && continue
        comms_obj_id[b] = eemit!(Expr(:call, getf, comms_tuple_id, b), comms_stack_ty[b])
    end

    # --- The primal's own coduals, stored on the tape by the forwards pass (`Tape.args`) — the
    # pullback's only route to an `Argument(k)` of the primal, since its own signature is just
    # `(tape, seed)`. Everything is unpacked eagerly (unused entries DCE'd later); emitting lazily
    # isn't an option because `_emit_switch!` below appends the entry block's terminator into this
    # same `entry_stmts` vector, and anything appended after that would land past a terminator. ---
    parg_pb = Vector{Any}(undef, n)
    farg_pb = Vector{Any}(undef, n)
    let args_tup_id = eemit!(Expr(:call, getf, tape_id, 5), ArgsTT)
        for k in 1:n
            Ck = codualparams[k]
            Pk = _codual_primal_type(Ck)
            Fk = _codual_fdata_type(Ck)
            cd = eemit!(Expr(:call, getf, args_tup_id, k), Ck)
            parg_pb[k] = eemit!(Expr(:call, getf, cd, 1), Pk)
            Fk !== NoFData && (farg_pb[k] = eemit!(Expr(:call, getf, cd, 2), Fk))
        end
    end
    _pack_vararg_args!(eemit!, ctuple, parg_pb, farg_pb, codualparams, iworld, nfixed)

    npacked = length(packed_codualparams)
    arg_ref_id = Vector{Any}(undef, npacked)
    for k in 1:npacked
        Pk = _codual_primal_type(packed_codualparams[k])
        # `zero_like_rdata_type`/`zero_like_rdata_from_type`, not `rdtype`/`zero_rdata_from_type`: when
        # `Pk` isn't concrete enough to produce a real zero rdata from its type alone (e.g. an
        # abstractly-typed argument slot), the ref's element type must include `ZeroRData` and the zero
        # literal must be `ZeroRData()` instead of crashing (`zero_rdata_from_type` returns the
        # `CannotProduceZeroRDataFromType` sentinel in that case, which `:new`'s field-type check below
        # rejects). Both collapse to the old behavior exactly when `Pk` is concrete. The packed
        # vararg-tail slot's accumulator is one combined `Tuple`-shaped rdata, split at the
        # pullback's return (`_pb_vararg_tail_rdata`).
        RT = zero_like_rdata_type(Pk)
        arg_ref_id[k] = eemit!(Expr(:new, Base.RefValue{RT}, zero_like_rdata_from_type(Pk)), Base.RefValue{RT})
    end

    ssa_ref_id = Vector{Any}(undef, N)
    for i in 1:N
        needs_ref(i) || continue
        # `_widen`: a statement's inferred type can be a lattice element (`Core.PartialStruct`), not a
        # bare `Type` — e.g. a `Core.memorynew` call with a literal length — and `zero_rdata_from_type`
        # (like `tangent_type`) is only ever defined on `Type`s.
        Ti = _widen(pstmts[i][:type])
        # See the `arg_ref_id` prologue above for why `zero_like_rdata_type`/`zero_like_rdata_from_type`.
        RT = zero_like_rdata_type(Ti)
        ssa_ref_id[i] = eemit!(Expr(:new, Base.RefValue{RT}, zero_like_rdata_from_type(Ti)), Base.RefValue{RT})
    end

    ref_for(@nospecialize node) =
        isa(node, Core.SSAValue) ? ssa_ref_id[node.id] :
        isa(node, Core.Argument) ? arg_ref_id[node.n] : nothing

    # One small routing block per exit: seed *that* exit's own return-value `Ref` from `seed`, then
    # jump to its reverse code. Which one runs is chosen by the switch below, popping the block
    # stack the forwards pass pushed to right before returning (see `reverse_fwds_to_ircode`).
    exit_route_blocks = CFGBlock[]
    exit_route_ids = ID[]
    for b in exit_blocks
        rstmts = IDInstPair[]
        remit!(ex, @nospecialize(ty)) = begin
            id = ID()
            push!(rstmts, (id, new_inst(ex, ty)))
            id
        end
        target = ref_for(exit_ret_node[b])
        if target !== nothing
            # `zero_like_rdata_type`, not `rdtype`: `target`'s actual declared element type (set in the
            # `arg_ref_id`/`ssa_ref_id` prologue above) may be `Union{R,ZeroRData}`, and `cur`'s declared
            # type here must agree with that or the `icall` below could resolve to (and statically
            # `:invoke`) a method compiled for the too-narrow `R` alone. `_widen`: `_optype` is
            # lattice-faithful and can return a `Core.PartialStruct`.
            RT = zero_like_rdata_type(_widen(_optype(pir, exit_ret_node[b])))
            cur = remit!(Expr(:call, getf, target, 1), RT)
            new = remit!(icall(increment_g, (RT, PullbackSeedT), cur, seed_id), RT)
            remit!(Expr(:call, setf, target, 1, new), Any)
        end
        rid = ID()
        remit!(IDGotoNode(block_id[b]), Any)
        push!(exit_route_blocks, CFGBlock(rid, rstmts))
        push!(exit_route_ids, rid)
    end
    _emit_switch!(eemit!, icall, block_stack_id, exit_blocks, exit_route_ids;
                 skip_pop=length(exit_blocks) == 1)

    blocks = vcat(CFGBlock(entry_id, entry_stmts), exit_route_blocks)

    # --- One reverse block per primal block. ---
    for b in 1:nblocks
        if unreachable_block[b]
            push!(blocks, CFGBlock(block_id[b], [ID()], [new_inst(nothing, Nothing)]))
            continue
        end

        stmts = IDInstPair[]
        emit!(ex, @nospecialize(ty), flag=CC.IR_FLAG_REFINED) = begin
            id = ID()
            push!(stmts, (id, new_inst(ex, ty, flag)))
            id
        end
        # `Pi` is the *primal* type of the statement whose rdata `ref` accumulates — needed (not just
        # its rdata type) because the zero-reset literal is computed via `zero_rdata_from_type(Pi)`.
        # `_widen` first: callers pass `pstmts[i][:type]` directly, which can be a lattice element
        # (see the `ssa_ref_id` prologue above) — `zero_rdata_from_type` only accepts a bare `Type`.
        deref_and_zero!(ref, @nospecialize(Pi)) = begin
            Pi = _widen(Pi)
            # `zero_like_rdata_type`/`zero_like_rdata_from_type` — see the `arg_ref_id`/`ssa_ref_id`
            # prologue above; `ref`'s actual declared element type is whichever of the two this produces,
            # so reading it back out (and re-zeroing it) must agree exactly.
            RT = zero_like_rdata_type(Pi)
            cur = emit!(Expr(:call, getf, ref, 1), RT)
            emit!(Expr(:call, setf, ref, 1, zero_like_rdata_from_type(Pi)), Any)
            cur
        end
        route!(@nospecialize(node), contrib, @nospecialize(ty)) = begin
            target = ref_for(node)
            if target !== nothing
                # `target`'s actual declared element type is `zero_like_rdata_type` of `node`'s own
                # primal type (set in the `arg_ref_id`/`ssa_ref_id` prologue above) — not necessarily
                # the caller-supplied `ty` (which describes `contrib`, computed from whatever type the
                # contribution happens to come from, e.g. a field type rather than `node`'s own type).
                # Deriving `cur`'s type independently here, rather than trusting `ty`, keeps this correct
                # regardless of what each call site passes for `ty` — a too-narrow declared type here
                # would let `icall` resolve to (and statically `:invoke`) a method compiled for a type
                # narrower than what the ref can actually hold.
                rty = zero_like_rdata_type(_widen(_optype(pir, node)))
                cur = emit!(Expr(:call, getf, target, 1), rty)
                new = emit!(icall(increment_g, (rty, ty), cur, contrib), rty)
                emit!(Expr(:call, setf, target, 1, new), Any)
            end
            nothing
        end

        # (a) Recover this visit's forwards-computed operand values, if this block has any. Keyed by the
        # same tagged `(kind, node)` items `_scan_block_comms` produced. `comms_type_id` records each
        # item's static type alongside its SSA value — needed by the `:subtape` recursion case below,
        # which must pass the inner tape's concrete `InnerTapeT` to `reverse_pullback_recursive_ci`, not
        # just the runtime SSA reference to its value.
        comms_val_id = Dict{Any,Any}()
        comms_type_id = Dict{Any,Any}()
        if !isempty(block_comms_types[b])
            ST = comms_stack_ty[b]
            # Unrolled single-slot read for a `CommsCell`: `getfield` of `val` (field 1) — no
            # `pop!`/`position`. Emitted under the pre-minted `comms_popped_id[b]` so any block whose
            # items were fused onto `b` can name this tuple (see `comms_popped_id`).
            pop_ex = ST <: CommsCell ? Expr(:call, getf, comms_obj_id[b], 1) :
                                       icall(pop_g, (ST,), comms_obj_id[b])
            popped = comms_popped_id[b]
            push!(stmts, (popped, new_inst(pop_ex, Tuple{block_comms_types[b]...}, CC.IR_FLAG_REFINED)))
            for (j, nd) in enumerate(block_comms_nodes[b])
                comms_val_id[nd] = emit!(Expr(:call, getf, popped, j), block_comms_types[b][j])
                comms_type_id[nd] = block_comms_types[b][j]
            end
        end
        # Fused items: not on `b`'s own stack — they rode along on control-equivalent successor
        # `host`'s single push, so they come out of the tuple `host`'s reverse code popped. Legal because
        # `host`'s reverse block is this one's sole predecessor in the pullback CFG.
        for (item, hostb, slot) in block_fused_refs[b]
            comms_val_id[item] = emit!(Expr(:call, getf, comms_popped_id[hostb], slot),
                                       block_comms_types[hostb][slot])
            comms_type_id[item] = block_comms_types[hostb][slot]
        end
        # Hoisted items: not on `b`'s own comms stack at all — read straight out of the defining block
        # `d`'s `CommsCell` (`comms_obj_id[d]`, already unpacked in the entry block). A plain double
        # `getfield` — no `pop!`, and safe regardless of visit order since the forwards pass has already
        # finished and `d`'s single-slot cell was written exactly once.
        for (item, d, slot) in block_hoisted_refs[b]
            dslot = emit!(Expr(:call, getf, comms_obj_id[d], 1), Tuple{block_comms_types[d]...})
            comms_val_id[item] = emit!(Expr(:call, getf, dslot, slot), block_comms_types[d][slot])
            comms_type_id[item] = block_comms_types[d][slot]
        end
        # --- Comms resolvers. Every pullback-side read of a forwards-recorded value goes through one
        # of these three, rather than indexing `comms_val_id` directly — the single place that knows
        # how a value can be obtained, letting that set be extended without touching any rule.
        #
        # `fetch_shadow` accepts either spelling of "this node's fdata handle": `:shadow_ref` (what
        # `memoryrefget`/`memoryrefset!` declare for a `MemoryRef`) and `:fshadow` (what
        # `getfield`/`setfield!`/a tracked `%new` declare for a struct object) — the two node
        # populations are disjoint, so trying both is unambiguous.
        #
        # When neither is on the tape, a statically-derivable `MemoryRef` is re-derived from
        # `tape.args` + the literal index (`pb_rederive_ref`) — the comms rule skipped it for that.
        pb_rederive_ref(@nospecialize(a), shadow::Bool) = begin
            d = _static_ref_derivation(pir, iworld, a)
            d === nothing && return nothing
            k, idx, bc = d
            arr = shadow ? farg_pb[k] : parg_pb[k]
            arr === nothing && return nothing        # argument carries no fdata
            reft = _optype(pir, a)
            rt = shadow ? fdtype(iworld, reft) : reft
            base = emit!(Expr(:call, getf, arr, QuoteNode(:ref)), rt)
            # A literal index is baked in directly; a dynamic one was pushed as a plain `Int` comms
            # item and comes back through `pb_presolve` (never re-enters here — an `Int`-typed node
            # never matches `_static_ref_derivation`'s `MemoryRef` shape).
            idx_val = isa(idx, Core.SSAValue) ? pb_presolve(idx) : idx
            # Shadow ref forces boundscheck `true`; primal ref keeps the primal's own.
            return emit!(Expr(:call, Base.memoryrefnew, base, idx_val, shadow ? true : bc), rt)
        end

        pb_fetch_shadow(@nospecialize a) =
            haskey(comms_val_id, (:shadow_ref, a)) ? comms_val_id[(:shadow_ref, a)] :
            haskey(comms_val_id, (:fshadow, a)) ? comms_val_id[(:fshadow, a)] :
            begin
                rd = pb_rederive_ref(a, true)
                rd === nothing ? error("Differ internal error: no shadow comms item for $(a) in " *
                                       "block $(b) — its rule's `builtin_rrule_comms` and its " *
                                       "pullback disagree about what was recorded") : rd
            end

        # A literal/`GlobalRef` operand needs no comms item at all (`_calleeval` resolves it); an
        # `SSAValue`/`Argument` does, and `_calleeval` returns `nothing` for those. Erroring on that
        # `nothing` is the point: before, an operand a rule read but the scan never recorded would emit
        # `nothing` into the IR and fail far away with no indication of where.
        pb_presolve(@nospecialize a) = begin
            haskey(comms_val_id, (:primal, a)) && return comms_val_id[(:primal, a)]
            isa(a, Core.Argument) && return parg_pb[a.n]
            # A statically-derivable primal `MemoryRef` handle is re-derived, not pushed.
            rd = pb_rederive_ref(a, false)
            rd === nothing || return rd
            v = _calleeval(a, iworld)
            (v === nothing && (isa(a, Core.SSAValue) || isa(a, Core.Argument))) &&
                error("Differ internal error: no primal comms item for $(a) in block $(b) — a rule " *
                      "reads it, but the forwards pass never recorded it")
            return v
        end

        # Values a rule's own fwds emission stashed (`(:old_primal, ssa)`, `(:old_tangent, ssa)`),
        # keyed by the full tagged item rather than by node.
        pb_fetch_saved(@nospecialize item) =
            haskey(comms_val_id, item) ? comms_val_id[item] :
            error("Differ internal error: comms item $(item) was declared but never saved by its " *
                  "rule's fwds emission (builtin_rrule_comms/apply_builtin_rrule_fwds! disagree)")

        # (b) This block's own (non-phi) statements, in reverse order.
        lo, hi = pir.cfg.blocks[b].stmts.start, pir.cfg.blocks[b].stmts.stop
        phi_end = lo - 1
        for i in lo:hi
            isa(pstmts[i][:stmt], Core.PhiNode) || break
            phi_end = i
        end
        for i in reverse((phi_end + 1):hi)
            s = pstmts[i][:stmt]; Ti = pstmts[i][:type]
            if isa(s, Core.GotoNode) || isa(s, Core.GotoIfNot) || isa(s, Core.ReturnNode)
                continue
            elseif isa(s, Expr) && s.head === :boundscheck
                continue   # pure control marker, always `NoRData` — nothing to route
            elseif isa(s, Expr) && s.head === :throw_undef_if_not
                continue   # pure control marker, always `NoRData` — nothing to route
            elseif isa(s, Expr) && s.head === :loopinfo
                continue   # pure control marker, always `NoRData` — nothing to route
            elseif isa(s, Expr) && s.head === :gc_preserve_begin
                continue   # pure control marker, always `NoRData` — nothing to route
            elseif isa(s, Expr) && s.head === :gc_preserve_end
                continue   # pure control marker, always `NoRData` — nothing to route
            elseif isa(s, Expr) && s.head === :foreigncall
                # Dispatch layer decides everything, mirroring `Core.Builtin` below. Unlike a
                # builtin's `contribs` (routed via `s`'s own call operands), a foreigncall rule's
                # contribs route through `_fc_parse(s).args`, not raw `s.args`.
                fc = _fc_parse(s)
                fcctx = (; emit!, icall, pb_presolve, bulk_saved=pb_bulk_saved,
                        fetch_shadow=pb_fetch_shadow, fetch_primal=pb_presolve,
                        fetch_saved=pb_fetch_saved,
                        deref_and_zero! = (@nospecialize Pi) -> deref_and_zero!(ssa_ref_id[i], Pi),
                        optype=(@nospecialize x) -> _optype(pir, x),
                        pstmt=(x::Core.SSAValue) -> pstmts[x.id][:stmt],
                        calleeval=(@nospecialize x) -> _calleeval(x, iworld),
                        ssa=Core.SSAValue(i), ref_for,
                        tt=(@nospecialize(T) -> _tt(iworld, _widen(T))),
                        rdtype=(@nospecialize(T) -> rdtype(iworld, T)),
                        fdtype=(@nospecialize(T) -> fdtype(iworld, T)))
                contribs = fc === nothing ? nothing : apply_foreigncall_rrule!(Val(fc.name), fc, Ti, fcctx)
                if contribs === nothing
                    reason[] = "reverse mode does not support foreigncall target " *
                               "`$(fc === nothing ? "(runtime function pointer)" : ":$(fc.name)")` at " *
                               "%$i: `$(_stmt_str(s))`"
                    return nothing
                end
                for (a, c) in zip(fc.args, contribs)
                    c === nothing && continue
                    route!(a, c, zero_like_rdata_type(_widen(_optype(pir, a))))
                end
            elseif isa(s, Core.PiNode)
                acc = deref_and_zero!(ssa_ref_id[i], Ti)
                route!(s.val, acc, zero_like_rdata_type(_widen(_optype(pir, s.val))))
            elseif isa(s, Expr) && s.head === :new
                T = _calleeval(s.args[1], iworld)
                args = @view s.args[2:end]
                if T <: Array
                    # Nothing to route: both fields (`:ref`, `:size`) always have `NoRData`, and no
                    # `(:fshadow, ...)` comms item exists for an array's own `%new` — everything
                    # accumulated during the reverse walk already landed directly in the shadow
                    # `Array` the fwds pass built. Must be checked before `ismutabletype(T)` below,
                    # which is also true for `Array` but assumes a `MutableTangent` shadow.
                elseif ismutabletype(T)
                    # A mutable struct carries no rdata of its own — everything accumulated during
                    # the reverse walk landed directly in the shadow `MutableTangent` built by the
                    # fwds pass. Since the pullback walks in reverse, this `%new`'s own turn runs
                    # last, after every use of the object, so that accumulation is complete and each
                    # field's gradient is handed back as this `%new` argument's own rdata — mirroring
                    # Mooncake's `_mutable_new_pullback!!`.
                    FT = fdtype(iworld, T)
                    if FT !== NoFData
                        shadow = pb_fetch_shadow(Core.SSAValue(i))
                        for j in eachindex(args)
                            Fty = fieldtype(T, j)
                            RFty = rdtype(iworld, Fty)
                            RFty === NoRData && continue
                            TFty = _tt(iworld, Fty)
                            field_tangent = emit!(icall(_rr_get_tangent_field, (FT, Int), shadow, j), TFty)
                            contrib = emit!(icall(_rr_rdata, (TFty,), field_tangent), RFty)
                            route!(args[j], contrib, RFty)
                        end
                    end
                else
                    acc = deref_and_zero!(ssa_ref_id[i], Ti)
                    RDataT = rdtype(iworld, T)
                    if RDataT !== NoRData
                        # `acc`'s own declared type can include `ZeroRData` when `Ti` isn't concrete
                        # enough on its own to produce a real zero (e.g. this `%new` sits behind a
                        # later `Union` merge). `T` is always concrete here, so materialize a real
                        # zero of type `RDataT` before treating `acc` as the real `NamedTuple`
                        # wrapper below (a raw `getfield` on the fieldless `ZeroRData()` singleton
                        # would otherwise throw).
                        real_acc = emit!(
                            icall(_rr_realize_rdata, (zero_like_rdata_type(_widen(Ti)), Type{RDataT}),
                                  acc, RDataT),
                            RDataT)
                        # `rdata_type` wraps a general struct's rdata as `RData{NamedTuple}`, but for
                        # `T <: Union{Tuple,NamedTuple}` it's already the bare field container.
                        if RDataT <: Union{Tuple,NamedTuple}
                            NT = RDataT
                            data_id = real_acc
                        else
                            NT = fields_type(RDataT)
                            data_id = emit!(Expr(:call, getf, real_acc, 1), NT)
                        end
                        for j in eachindex(args)
                            Fty = rdtype(iworld, fieldtype(T, j))
                            Fty === NoRData && continue
                            contrib = emit!(Expr(:call, getf, data_id, j), Fty)
                            route!(args[j], contrib, Fty)
                        end
                    end
                end
            elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
                fpos, actual = _call_parts(s)
                f = _calleeval(fpos, iworld)
                if isa(f, Core.IntrinsicFunction)
                    # Non-differentiable results (comparisons -> Bool, integer ops, ...) have `NoRData`;
                    # skip entirely rather than asking `apply_intrinsic_rrule!` for a rule that doesn't
                    # exist and shouldn't — nothing flows backward through them.
                    if rdtype(iworld, Ti) !== NoRData
                        acc = deref_and_zero!(ssa_ref_id[i], Ti)
                        # Operands the forwards pass deliberately didn't record (see
                        # `intrinsic_rrule_operands`, consulted identically in `_scan_block_comms`) come
                        # through as `UnrecordedOperand`, and `opf` refuses to emit anything referencing
                        # one. So a rule whose declaration understates what it reads fails here, loudly
                        # and located, instead of silently emitting IR against a value that was never put
                        # on the tape.
                        needed = intrinsic_rrule_operands(Val(f))
                        pvals = Tuple((needed === nothing || k in needed) ? pb_presolve(a) :
                                      UnrecordedOperand(k) for (k, a) in enumerate(actual))
                        ctx = (opf=(name, ty, args...) -> begin
                                   for a in args
                                       isa(a, UnrecordedOperand) || continue
                                       error("Differ internal error: the reverse rule for intrinsic " *
                                             "`$(nameof(f))` reads operand $(a.position), but " *
                                             "`intrinsic_rrule_operands(Val($(nameof(f))))` does not " *
                                             "list it, so the forwards pass never recorded it. Fix " *
                                             "that declaration in src/intrinsics_reverse.jl.")
                                   end
                                   emit!(Expr(:call, GlobalRef(Core.Intrinsics, name), args...), ty)
                               end,
                              # An operand's own declared primal type, read straight from the primal IR —
                              # not derivable from `pvals` (a resolved value, not a type) or `Ti` (the
                              # statement's own result type, not an operand's). Needed by
                              # `fpext`/`fptrunc`'s reverse rule, which converts `dz` back to the
                              # operand's own float width.
                              optype=(k::Int) -> _widen(_optype(pir, actual[k])),)
                        contribs = apply_intrinsic_rrule!(Val(f), pvals, acc, Ti, ctx)
                        if contribs === nothing
                            reason[] = "no reverse rule for intrinsic `$(nameof(f))` at %$i: " *
                                       "`$(_stmt_str(s))` (no rule registered; add one in " *
                                       "src/intrinsics_reverse.jl via `apply_intrinsic_rrule!`)"
                            return nothing
                        end
                        for (a, c) in zip(actual, contribs)
                            route!(a, c, zero_like_rdata_type(_widen(_optype(pir, a))))
                        end
                    end
                elseif isa(f, Core.Builtin)
                    cctx = (; emit!, icall, pb_presolve, bulk_saved=pb_bulk_saved,
                           fetch_shadow=pb_fetch_shadow, fetch_primal=pb_presolve,
                           fetch_saved=pb_fetch_saved,
                           deref_and_zero! = (@nospecialize Pi) -> deref_and_zero!(ssa_ref_id[i], Pi),
                           optype=(@nospecialize x) -> _optype(pir, x), ssa=Core.SSAValue(i), ref_for,
                           tt=(@nospecialize(T) -> _tt(iworld, _widen(T))),
                           rdtype=(@nospecialize(T) -> rdtype(iworld, T)),
                           fdtype=(@nospecialize(T) -> fdtype(iworld, T)))
                    contribs = apply_builtin_rrule!(Val(f), actual, Ti, cctx)
                    if contribs !== nothing
                        for (a, c) in zip(actual, contribs)
                            c === nothing && continue
                            route!(a, c, zero_like_rdata_type(_widen(_optype(pir, a))))
                        end
                    elseif _tt(iworld, _widen(Ti)) === NoTangent
                        # Mirrors the fwds pass's own treatment: a non-differentiable builtin result
                        # has nothing to route backward, so this is a genuine no-op here (unlike the
                        # fwds pass, the pullback never needs to *replay* the statement at all).
                    else
                        reason[] = "reverse mode does not support builtin `$(f)` with a differentiable " *
                                   "result ($(Ti)) and no reverse rule at %$i: `$(_stmt_str(s))`"
                        return nothing
                    end
                else
                    info = _static_recursible_call(pir, iworld, i, s, reason, arg_tracked, fdata_tracked)
                    info === nothing && return nothing
                    _, ftype, argtypes = info
                    acc = deref_and_zero!(ssa_ref_id[i], Ti)   # this call's own seed for the inner pullback
                    # A direct self-recursive call's inner tape was pushed onto `Tape.subtapes` (fwds
                    # side), not a per-block comms item. Pop it back off `subtapes_id` directly
                    # instead; since this visits statements in descending SSA order while the fwds
                    # pass pushed in ascending order, multiple self-recursive calls sharing one block
                    # pop in the correct (reverse) order automatically.
                    is_self = i in self_recursive_ssas
                    InnerTapeT = is_self ? Tape : comms_type_id[(:subtape, Core.SSAValue(i))]
                    inner_tape = is_self ? emit!(icall(pop_g, (Stack{TapeT},), subtapes_id), TapeT) :
                                           comms_val_id[(:subtape, Core.SSAValue(i))]
                    # `zero_like_rdata_type`, not `rdtype`: `acc`'s actual type (see `deref_and_zero!`
                    # above) may include `ZeroRData` when `Ti` isn't concrete enough on its own, so the
                    # inner pullback must be resolved to accept exactly that (possibly wider) seed type —
                    # its own exit-route `increment!!` already tolerates `ZeroRData` generically.
                    SeedT = zero_like_rdata_type(_widen(Ti))
                    pb_resolved = reverse_pullback_recursive_ci(interp, impl_mi, TapeT, codualparams,
                                                                 InnerTapeT, SeedT, edges, reason)
                    if pb_resolved === nothing
                        reason[] *= " — at %$i: `$(_stmt_str(s))`"
                        return nothing
                    end
                    pb_ci, pb_derived, InnerRdatasT = pb_resolved
                    # A derived inner pullback goes through its carrier (`reverse_pullback_impl` the
                    # callee, tape an argument); a hand-written one is the callee. The hand-written
                    # case is flagged `IR_FLAG_NOINLINE`: inlining a rule author's body back in here
                    # would re-embed its `GlobalRef`s relative to its defining module (a bare `cos`
                    # becomes `GlobalRef(Differ, :cos)`), which `verify_ir` rejects. `_is_reverse_carrier_mi`
                    # can't cover this case by type — a hand pullback has no common supertype.
                    pb_stmt = pb_derived ? Expr(:invoke, pb_ci, reverse_pullback_impl, inner_tape, acc) :
                                           Expr(:invoke, pb_ci, inner_tape, acc)
                    inner_rdatas = emit!(pb_stmt, InnerRdatasT,
                                         pb_derived ? CC.IR_FLAG_REFINED : CC.IR_FLAG_NOINLINE)
                    # Slot 1 is the callee's own rdata — guaranteed `NoRData` by
                    # `_static_recursible_call`'s callee guard, discarded exactly like `ref_for` already
                    # discards any literal operand's contribution — so routing starts at 2.
                    for (j, a) in enumerate(actual)
                        # `zero_like_rdata_type`: the callee's own `argtypes[j]`-th argument rdata (this
                        # same function, recursively, for the callee) can likewise be `ZeroRData` when
                        # that argument's type isn't concrete enough.
                        Fty = zero_like_rdata_type(_widen(argtypes[j]))
                        Fty === NoRData && continue
                        contrib = emit!(Expr(:call, getf, inner_rdatas, j + 1), Fty)
                        route!(a, contrib, Fty)
                    end
                end
            end
        end

        # (c) Leading PhiNodes: dereference+zero each accumulated rdata, then route per-predecessor.
        preds = filter(!=(0), pir.cfg.blocks[b].preds)
        # Collapsible region (`_collapsible_regions`): `b` is a `merge` block whose real, ambiguous
        # predecessor set is a comms-free `@boundscheck` diamond around one canonical entry. Route
        # through that entry alone — `pred_is_unique_pred[b]` is forced `true` for this case
        # (`_unique_predecessor_info`), so `_emit_switch!` below never pops for it either.
        haskey(regions, b) && (preds = [regions[b]])
        phi_acc = Any[]
        for i in lo:phi_end
            Ti = pstmts[i][:type]
            push!(phi_acc, deref_and_zero!(ssa_ref_id[i], Ti))
        end

        if b == 1
            # No predecessors: this is the pullback's own final block — the last thing that runs.
            # Restore every bulk-saved argument's primal contents here, which is what makes the whole
            # scheme equivalent to restoring each overwritten element as the sweep passes it: nothing
            # reads primal memory in between, so only the state at this boundary is visible.
            if !isempty(bulk_args)
                bufs_id = emit!(Expr(:call, getf, Core.Argument(2), 3), Vector{Any})
                for k in sort!(collect(bulk_args))
                    Pk = _widen(_codual_primal_type(packed_codualparams[k]))
                    emit!(icall(_bulk_restore!, (Vector{Any}, Int, Pk),
                                bufs_id, bulk_slot[k], parg_pb[k]), Nothing)
                end
            end
            # Read out every argument's accumulated rdata and return them as a tuple, flat arity.
            # `nfixed_eff` is `n` when non-vararg, so the tail branch below never fires.
            is_vararg = nfixed >= 0
            nfixed_eff = is_vararg ? nfixed : n
            result_ids = Vector{Any}(undef, n)
            for k in 1:nfixed_eff
                Pk = _codual_primal_type(codualparams[k])
                result_ids[k] = emit!(Expr(:call, getf, arg_ref_id[k], 1), zero_like_rdata_type(Pk))
            end
            if is_vararg
                tailPk = _codual_primal_type(packed_codualparams[nfixed_eff + 1])
                tailRT = zero_like_rdata_type(tailPk)
                tail_acc = emit!(Expr(:call, getf, arg_ref_id[nfixed_eff + 1], 1), tailRT)
                for k in (nfixed_eff + 1):n
                    Pk = _codual_primal_type(codualparams[k])
                    RTk = zero_like_rdata_type(Pk)
                    j = k - nfixed_eff
                    result_ids[k] = emit!(icall(_pb_vararg_tail_rdata, (tailRT, Val{j}, Type{Pk}),
                                                tail_acc, Val(j), Pk), RTk)
                end
            end
            res = emit!(Expr(:call, ctuple, result_ids...),
                       Tuple{(zero_like_rdata_type(_codual_primal_type(c)) for c in codualparams)...})
            emit!(Core.ReturnNode(res), Any)
        elseif phi_end < lo
            # No PhiNodes at the top of this block: switch straight to each predecessor's own block.
            _emit_switch!(emit!, icall, block_stack_id, preds, ID[block_id[p] for p in preds];
                         skip_pop=pred_is_unique_pred[b])
        else
            # PhiNodes: route each predecessor's own edge value, in a small dedicated block per pred.
            phi_ids = lo:phi_end
            new_blocks = CFGBlock[]
            target_ids = ID[]
            for p in preds
                rstmts = IDInstPair[]
                remit!(ex, @nospecialize(ty)) = begin
                    id = ID()
                    push!(rstmts, (id, new_inst(ex, ty)))
                    id
                end
                for (j, i) in enumerate(phi_ids)
                    phi = pstmts[i][:stmt]::Core.PhiNode
                    eidx = findfirst(==(Int32(p)), phi.edges)
                    (eidx === nothing || !isassigned(phi.values, eidx)) && continue
                    v = phi.values[eidx]
                    tgt = ref_for(v)
                    tgt === nothing && continue
                    Ti = pstmts[i][:type]
                    # `tgt`'s actual declared element type is `zero_like_rdata_type` of `v`'s own primal
                    # type (the edge value) — generally not the same as the phi node's own (merged,
                    # typically wider) type `Ti`. `phi_acc[j]` is `zero_like_rdata_type` of `Ti` instead,
                    # since that's what `deref_and_zero!` actually produced for the phi itself. Deriving
                    # these independently (rather than a single `RT` from `Ti` used for both, as before
                    # `ZeroRData` support) is what makes this correct on a loop back-edge, where the
                    # phi's accumulator is genuinely re-zeroed every visit.
                    RTcur = zero_like_rdata_type(_widen(_optype(pir, v)))
                    RTacc = zero_like_rdata_type(_widen(Ti))
                    cur = remit!(Expr(:call, getf, tgt, 1), RTcur)
                    new = remit!(icall(increment_g, (RTcur, RTacc), cur, phi_acc[j]), RTcur)
                    remit!(Expr(:call, setf, tgt, 1, new), Any)
                end
                rid = ID()
                remit!(IDGotoNode(block_id[p]), Any)
                push!(new_blocks, CFGBlock(rid, rstmts))
                push!(target_ids, rid)
            end
            _emit_switch!(emit!, icall, block_stack_id, preds, target_ids; skip_pop=pred_is_unique_pred[b])
            append!(blocks, new_blocks)
        end

        push!(blocks, CFGBlock(block_id[b], stmts))
    end

    # `blocks` is currently ordered by ascending primal block number, which is not a valid
    # topological/dominance order for the pullback's own control flow — the pullback runs in the
    # opposite direction, so primal block 1 sits first in the vector despite being reached last.
    # `_sort_cfg_blocks!` reorders by BFS distance from the pullback's own entry: without it,
    # `sroa_pass!`/`adce_pass!` crash with an `UndefRefError` inside `is_union_phi` on any primal
    # with a loop, because SROA's phi-insertion when scalar-replacing the rdata `Ref`s assumes this
    # order. Safe to reorder even though blocks still hold `Switch` terminators here — the implicit
    # fallthrough-to-next-block semantics that would make reordering unsound only apply once
    # `_cfg_lower_switch_statements` lowers those to `IDGotoIfNot`, after this sort.
    # `_remove_unreachable_cfg_blocks!` drops the throw-only-primal-block stubs: nothing in the
    # pullback ever pushes/routes to them, so they're genuinely unreachable here.
    blocks = _remove_unreachable_cfg_blocks!(_sort_cfg_blocks!(blocks))
    ir2 = lower_cfg_blocks_to_ir(blocks, pir; argtypes=Any[impl_mi.specTypes.parameters...], def=impl_mi)
    CC.verify_ir(ir2)
    return ir2
end

# Emits a plain goto (`skip_pop`, the only path a single-predecessor block takes under the per-edge
# formula) or `pop!(block_stack)` followed by a `Switch` comparing the popped id against each
# candidate (`preds[1:end-1]`), falling through to `preds[end]`.
#
# The `length(preds) == 1` branch below is dead code under the current `pred_is_unique_pred[b] =
# length(preds[b]) <= 1` formula (every single-predecessor block returns early via `skip_pop`
# above). Kept as a defensive no-op, not removed — the old per-block push/pop scheme needed a
# balance-pop here (ISSUES #52); the two schemes must not be mixed.
function _emit_switch!(emit!, icall, block_stack_id, preds::Vector{Int}, targets::Vector{ID};
                       skip_pop::Bool=false)
    if skip_pop
        @assert length(targets) == 1 "skip_pop requires an unambiguous (single) target"
        emit!(IDGotoNode(targets[1]), Any)
        return nothing
    end
    prev_id = emit!(icall(__pop_blk_stack!, (Stack{Int32},), block_stack_id), Int32)
    if length(preds) == 1
        # Dead under the per-edge formula (see above) — defensive balance-pop, kept for safety.
        emit!(IDGotoNode(targets[1]), Any)
        return nothing
    end
    conds = ID[]
    for p in preds[1:(end - 1)]
        push!(conds, emit!(icall(__switch_case, (Int32, Int32), Int32(p), prev_id), Bool))
    end
    emit!(Switch(Any[c for c in conds], targets[1:(end - 1)], targets[end]), Any)
    return nothing
end

# ===========================================================================
# Generated entry points + public API.
# ===========================================================================

# The `@generated` derived fallback for `rrule!!(fcd, ctx, argcds...)` — the least-specific method
# (see the ambiguity note above the type definitions). Generated, because compiling the carrier is
# what calls `build_contextual_ir` (see the two-layer note near the carrier stubs). The carrier
# mirrors this signature exactly, so the body is a straight pass-through invoke, no reordering. `ctx`
# reaches the generated carrier as an ordinary argument, which is how a `Ctx` carrying a pre-allocated
# tape hands its stacks in.
function rrule_entry_body(world::UInt, source, self, fcd, ctx, argcds)
    argnames = Any[Symbol("#self#"), :fcd, :ctx, :argcds]
    impl_tt = Tuple{typeof(reverse_fwds_impl),fcd,ctx,argcds...}
    interp = build_reverse_interp(; world)
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                                :(error("Differ: no reverse_fwds_impl match")), true)
    end
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    cinst = CC.typeinf_ext_toplevel(interp, impl_mi, CC.SOURCE_MODE_ABI)
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(reverse_fwds_impl, $cinst, fcd, ctx, argcds...)), true)
    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

function refresh_rrule_entry()
    @eval function rrule!!(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, rrule_entry_body))
    end
end
refresh_rrule_entry()

# The pullback callable: `(t::Tape)(seed)`. `self` is the concrete `Tape` type, which is what keys the
# pullback carrier (and, via its `ArgsTT` parameter, names the primal it belongs to).
function pullback_entry_body(world::UInt, source, self, seedtype)
    argnames = Any[Symbol("#self#"), :seed]
    impl_tt = Tuple{typeof(reverse_pullback_impl),self,seedtype}
    interp = build_reverse_interp(; world)
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                                :(error("Differ: no reverse_pullback_impl match")), false)
    end
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    cinst = CC.typeinf_ext_toplevel(interp, impl_mi, CC.SOURCE_MODE_ABI)
    selfref = Symbol("#self#")
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(reverse_pullback_impl, $cinst, $selfref, seed)), false)
    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

function refresh_pullback_entry()
    @eval function (t::Tape)(seed)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, pullback_entry_body))
    end
end
refresh_pullback_entry()

# ===========================================================================
# Public API.
# ===========================================================================

"""
    build_ctx(f, argtypes::Tuple; prealloc=true) -> Ctx

Build a reusable differentiation context for `f` applied to arguments of types `argtypes` — a
[`Ctx`](@ref) wrapping a tape sized for `f`'s derived rule (obtained by transforming `f`'s optimized
IR). Pass it to [`rrule!!`](@ref) / [`value_and_gradient!`](@ref) / [`rev_gradient!`](@ref).

With `prealloc=true` (the default) the tape is allocated once, and its stacks are reset and reused on
every call — the whole point of holding onto a context rather than differentiating afresh. That makes
the context **single-use at a time**: it is not reentrant and not thread-safe, so give each task its
own. `prealloc=false` returns `Ctx()` — a context that allocates a fresh tape per call instead.

A reused tape also holds onto the *previous* call's argument coduals (the pullback reaches primal
argument values through them) — and so keeps their shadows alive — until the next call overwrites
them. Drop the context to release them.

```julia
ctx = build_ctx(f, (Vector{Float64},))
y, pb = rrule!!(zero_fcodual(f), ctx, CoDual(x, dx))
_, gx = pb(1.0)
```
"""
function build_ctx(@nospecialize(f), @nospecialize(argtypes::Tuple); prealloc::Bool=true)
    prealloc || return Ctx()
    # `Base.to_tuple_type` moves the argument types into a type parameter, which is the only way
    # `_build_tape`'s generator can see them: a runtime tuple of types has type `Tuple{DataType,…}`,
    # which says nothing about which types they were.
    return Ctx(_build_tape(f, Base.to_tuple_type(argtypes)))
end

"""
    _build_tape(f, ::Type{ArgsT}) -> Tape

Allocate a fresh `Tape` of exactly the shape `f`'s derived rule will use — same block stack, same
per-block comms stacks. Generated: the tape type is read off the tape-allocating carrier's return type
at generation time, so at run time this is just the allocations.
"""
function _build_tape_body(world::UInt, source, self, ftype, argst)
    argnames = Any[Symbol("#self#"), :f, :ArgsT]
    bail(msg) = expr_to_codeinfo(@__MODULE__(), argnames, [], (), :(error($msg)), false)
    (argst isa DataType && argst <: Type) ||
        return bail("Differ.build_ctx: argtypes must be a tuple of types")
    ArgsT = argst.parameters[1]
    (ArgsT isa DataType && ArgsT <: Tuple) ||
        return bail("Differ.build_ctx: argtypes must be a tuple of types")
    interp = build_reverse_interp(; world)
    codualtys = Any[_fcdtype(world, ftype)]
    for T in ArgsT.parameters
        (T isa Type) || return bail("Differ.build_ctx: argtypes must be a tuple of types")
        push!(codualtys, _fcdtype(world, T))
    end
    # Carrier layout is `reverse_fwds_impl(fcd, ctx, argcds...)`: fcd first, then `Ctx{Nothing}`
    # (tape-allocating mode — this only reads its return type), then the argument coduals.
    primal_tt = Tuple{ftype,ArgsT.parameters...}
    impl_tt = Tuple{typeof(reverse_fwds_impl),codualtys[1],Ctx{Nothing},codualtys[2:end]...}
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    match === nothing &&
        return bail("Differ.build_ctx: no reverse_fwds_impl match for `$(primal_tt)`")
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    cinst = CC.typeinf_ext_toplevel(interp, impl_mi, CC.SOURCE_MODE_ABI)
    RT = cinst.rettype
    if !(RT isa DataType && RT <: Tuple && length(RT.parameters) == 2 &&
         RT.parameters[2] isa DataType && RT.parameters[2] <: Tape)
        why = get(REVERSE_BAIL_REASONS, impl_mi, nothing)
        return bail(why !== nothing ?
                    "Differ.build_ctx: could not derive a reverse rule for `$(primal_tt)`: $(why)" :
                    "Differ.build_ctx: could not derive a reverse rule for `$(primal_tt)`: the " *
                    "reverse forwards pass returned `$(RT)` rather than a `(CoDual, Tape)` pair, " *
                    "and no bail reason was recorded")
    end
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return $(_fresh_tape_expr(RT.parameters[2]))), false)
    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

# Expression constructing a fresh `Tape{ArgsTT,CS}`: an empty `Stack` per comms slot that stores
# anything, the singleton instance for the rest (a `SingletonStack` has no fields and stores nothing).
function _fresh_tape_expr(@nospecialize(TapeT))
    CS = TapeT.parameters[2]
    # Construct each comms stack slot: `SingletonStack`/`CommsCell` take no args (`%new`);
    # `Stack{T}` takes a capacity (pre-size to 1 so a single push is always in-bounds).
    slots = Any[S <: SingletonStack ? :($S()) :
                S <: CommsCell ? :($S()) :
                :($S(1)) for S in CS.parameters]
    return :($TapeT($(Stack{Int32})(1), ($(slots...),), $(Vector{Any})(), $(Stack{TapeT})()))
end

# Nested-tape recycling (`_inner_ctx`, `stack.jl`): allocate a fresh tape of exactly the shape
# `TapeT` names, for when the recycled comms slot is empty or holds the wrong type (the first
# execution of a block). Plain `@generated` (unlike `_build_tape`, which needs a specific
# `interp`/world) since `_fresh_tape_expr` builds its expression from the type parameter alone.
# `@noinline`: this is the cold path, and keeping it a real call means its `Stack`/comms-tuple
# construction isn't re-embedded inside whatever carrier IR `_inner_ctx` gets inlined into.
@noinline @generated function _alloc_tape(::Type{TapeT}) where {TapeT<:Tape}
    return _fresh_tape_expr(TapeT)
end

function refresh_build_tape()
    @eval function _build_tape(f, ArgsT)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, _build_tape_body))
    end
end
refresh_build_tape()

"""
    rev_gradient(f, args...) -> (df, dx1, dx2, ...)

Reverse-mode gradient of `f(args...)` for scalar output. Allocates everything it needs (zero shadows
for each argument, and a tape); see [`rev_gradient!`](@ref) for the pre-allocated form and
[`build_ctx`](@ref) to hold onto a reusable context. `public`, not exported — DifferentiationInterface
is the primary user-facing entry point, this is the direct one.
"""
function rev_gradient(f, args...)
    y, grads = value_and_gradient!(Ctx(), zero_fcodual(f), map(zero_fcodual, args)...)
    return grads
end

# ---------------------------------------------------------------------------
# Pre-allocated entry points.
#
# `rev_gradient` above calls `zero_fcodual` on `f` and every argument, allocating a fresh shadow per
# call (for an `Array` argument: the shadow array itself). These variants instead take the caller's
# own `CoDual`s, so the shadow buffers are owned and reused by the caller. Pair them with a
# `build_ctx(...; prealloc=true)` context and a steady-state call allocates essentially nothing.
#
# The gradient w.r.t. an argument arrives one of two ways, decided by the argument's type, not by
# this API:
#
#   * fdata-carried (an `Array`, a mutable struct): accumulated in place into the shadow the caller
#     supplied, so `dx` holds the answer when the call returns (and is also returned, as the same
#     object — no copy).
#   * rdata-carried (a scalar): nothing to pre-allocate; returned by value.
#
# The supplied fdata is zeroed on entry (`set_to_zero!!`, allocation-free for arrays), so one of these
# calls means the same thing as the equivalent `rev_gradient` call — the caller owns the buffer, not
# the accumulation history. Pass `zero_fcodual(f)` for the function slot unless differentiating w.r.t.
# a closure's captures.
# ---------------------------------------------------------------------------

"""
    value_and_gradient!(ctx::AbstractCtx, fcd::CoDual, argcds::CoDual...) -> (y, (df, dx1, dx2, ...))

Pre-allocated reverse mode: the caller supplies the context (from [`build_ctx`](@ref)) and each
argument's shadow. Returns the primal value `y` alongside the tangents.

Each `CoDual(x, dx)` pairs an argument with the buffer its gradient is accumulated into; `dx` is
zeroed on entry. For an argument whose tangent lives in fdata (an `Array`, a mutable struct) `dx`
holds that argument's gradient when the call returns. For a scalar there is nothing to pre-allocate
— pass `zero_fcodual(x)` and read the gradient out of the returned tuple.

```julia
x, dx = [1.0, 2.0], zeros(2)
ctx = build_ctx(f, (Vector{Float64},))
y, (_, gx) = value_and_gradient!(ctx, zero_fcodual(f), CoDual(x, dx))
gx === dx   # true — accumulated in place
```

See also [`rev_gradient!`](@ref) and [`rev_gradient`](@ref).
"""
function value_and_gradient!(ctx::AbstractCtx, fcd::CoDual, argcds::CoDual...)
    set_to_zero!!(tangent(fcd))
    map(cd -> set_to_zero!!(tangent(cd)), argcds)
    result_cd, pb = rrule!!(fcd, ctx, argcds...)
    y = primal(result_cd)
    all_cds = (fcd, argcds...)
    # A derived pullback can hand back `ZeroRData` for an argument whose concrete type has an
    # abstractly-typed field. This is the one place `rev_gradient`/`rev_gradient!` funnel through, so
    # instantiate a real zero here rather than ever handing a `ZeroRData` back to the user.
    rdatas = map((cd, r) -> r isa ZeroRData ? zero_rdata(primal(cd)) : r, all_cds, pb(one(y)))
    fdatas = (tangent(fcd), map(tangent, argcds)...)
    return y, map(tangent, fdatas, rdatas)
end

"""
    rev_gradient!(ctx::AbstractCtx, fcd::CoDual, argcds::CoDual...) -> (df, dx1, dx2, ...)

Pre-allocated form of [`rev_gradient`](@ref). See [`value_and_gradient!`](@ref) for the full
description; this drops the primal value.
"""
rev_gradient!(ctx::AbstractCtx, fcd::CoDual, argcds::CoDual...) =
    value_and_gradient!(ctx, fcd, argcds...)[2]
