# Reverse-mode AD: branches and loops, Mooncake style.
#
# Two separately-compiled carriers, wired through the same `build_contextual_ir` override as forward
# mode (`ADInterpreter{Reverse}`, `contextual.jl`, unchanged):
#
#   * `reverse_fwds_impl(codualargs::CoDual...) -> (result::CoDual, tape::Tape)` — forward-replays
#     the primal computation (no shadow/tangent — this isn't dualization, just ordinary value
#     recomputation) and, at every place control flow is ambiguous, instruments it: pushes the
#     current block number onto a shared `Stack{Int32}` ("block stack") so the pullback can replay
#     control flow in exact reverse order, and pushes whatever forward-computed operand values a
#     differentiable rule inside that block will need back onto a per-block "comms" `Stack`.
#   * `reverse_pullback_impl(tape::Tape, seed) -> rdata_tuple` — walks the primal's blocks in
#     reverse (a fresh CFG built via the `ID`/`CFGBlock` layer in `cfg_ir.jl`, since the pullback's
#     control-flow shape is *not* 1:1 with the primal — it inserts extra phi-routing blocks and
#     lowers multi-way dispatches into `GotoIfNot` chains), popping the block stack to know which
#     predecessor to jump to and popping each block's comms stack to recover that visit's forward
#     values, accumulating rdata into per-SSA/per-argument `Ref`s along the way.
#
# Unlike Mooncake (two `OpaqueClosure`s sharing captured state), the two passes here are ordinary
# `CodeInstance`s and the shared state is an explicit `Tape` value: `reverse_fwds_impl` returns it,
# `reverse_pullback_impl` takes it as an argument. No `OpaqueClosure` anywhere in this engine.
#
# `rrule`/`gradient` (bottom of this file) are plain, uncompiled Julia: they hold the original
# argument fdata (from `zero_fcodual`) and combine it with the rdata the pullback carrier returns
# via `tangent(fdata, rdata)` — the pullback carrier itself only ever returns rdata.
#
# Scope: branches and loops (any number of back-edges, and any number of reachable exit blocks —
# Julia's optimizer commonly keeps a separate `return` per arm of a branch rather than merging into
# one via a phi, so multi-exit is the *common* case, not a corner case; see `_exit_blocks`). A block
# pushes to (forwards) / pops from (pullback) the block stack only when its predecessor identity is
# actually ambiguous — see `_unique_predecessor_info` — so a straight-line function emits zero block-
# stack traffic at all, and a branch/loop only pays for the joins that are genuinely ambiguous. Data-
# wise: scalar float arithmetic intrinsics (`intrinsics_reverse.jl`), immutable fully-initialised
# structs (`Core.getfield`/`Expr(:new,...)` via `increment_field!!`/`RData`), statically-resolvable
# recursive calls into another primal (concrete callee + concrete, trivial-fdata args only — see
# `_static_recursible_call`; a hand-written rule, `src/rrules.jl`, always takes priority over raw
# recursion via ordinary method-table dispatch, mirroring `frules.jl`), **direct self-recursion**
# (the callee resolves to the exact primal being differentiated — a finite, closed-form `Tape` type,
# resolved to a static self-`:invoke` with no fixed point to solve and no extra compilation; see
# `reverse_fwds_recursive_ci`, ISSUES #65), and read-only array indexing via a provenance chain
# traceable to a function argument (see `_fdata_tracked`). Still out of scope, bails cleanly: mutable
# structs, array mutation, dynamic-dispatch recursion, *mutual* recursion (A→B→A — needs a tape-type
# pre-pass across the whole SCC), try/catch (see the control-flow plan's Phase E).

_codual_primal_type(@nospecialize P) = fieldtype(P, 1)
_codual_fdata_type(@nospecialize P) = fieldtype(P, 2)

# ===========================================================================
# The rule interface.
#
# Everything is called on `CoDual`s and returns `(ycd, pullback)`; the pullback is called on an rdata
# seed and returns the tuple of argument rdatas. Following Mooncake, **the pullback closure *is* the
# tape** — there is no separate tape value threaded between two free functions. For a derived
# (compiler-generated) rule that closure is a `Tape` (below), holding the block stack and the
# per-block comms stacks; for a hand-written rule it is whatever's cheapest to remember (see
# `src/rrules.jl`).
#
# `rrule!!` is a single stateless generic function under one uniform convention:
#
#     rrule!!(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) -> (ycd, pullback)
#
# Both flavours of rule are *methods* of it:
#
#   * hand-written *primitives* (`src/rrules.jl`) — a method for a specific `fcd`/`argcds` shape;
#   * the *derived* path — a single `@generated` fallback that transforms a composite primal's IR.
#
# The tape (and any future per-call/per-task config) lives not in the rule but in the `ctx` argument
# threaded through — so `rrule!!` itself is stateless and shareable, and reentrancy is "one `Ctx`
# per task" rather than "one rule per task".
#
# Reintroducing a fallback on `rrule!!` (we removed the previous one) is ambiguity-free *because the
# `ctx` slot is dispatch-neutral*: every method — the fallback and every hand rule — declares it as
# `::AbstractCtx`, never a concrete subtype. Specificity is therefore decided purely by the fcd + arg
# slots, the fallback is strictly least specific, and a hand rule always wins cleanly. The only
# consequence is that "does `rrule!!` resolve?" no longer means "is there a hand rule?" (the fallback
# always resolves), so `hand_reverse_rule_match` must *recognize* the fallback (a `m.sig === …` test)
# and report "no hand rule" when that is what matched.
#
# RULE-AUTHORING CONSTRAINT: a hand rule's `ctx` slot must stay `::AbstractCtx`. Two rules dispatching
# on *different* concrete ctx subtypes would bring the ambiguity back.
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
reusable one with [`build_ctx`](@ref)); a hand rule that needs no tape ignores it. A hand rule **must**
declare its `ctx` slot as `::AbstractCtx` (never a concrete subtype) — that is what keeps dispatch
against the fallback unambiguous.
"""
function rrule!! end

"""
    AbstractCtx

Supertype of reverse-mode *differentiation contexts* — the argument [`rrule!!`](@ref) threads its
per-call/per-task state through, chiefly the tape. [`Ctx`](@ref) is the default. Every `rrule!!`
method dispatches this slot as `::AbstractCtx` (never a concrete subtype), which is what keeps the
derived-fallback-vs-hand-rule dispatch ambiguity-free.
"""
abstract type AbstractCtx end

"""
    Ctx{P} <: AbstractCtx

The default differentiation context, carrying a pre-allocated tape in `tape::P`.

`P === Nothing` means "allocate a fresh tape on every call" — the simple, stateless mode, and what a
nested/recursive inner call always uses (an inner pullback is a value pushed onto the *outer* block's
comms stack, one per execution, so there is nothing for an outer caller to pre-allocate on its
behalf). Any other `P` is a tape the caller allocated once, whose stacks are reset and reused per
call; that is what removes the last per-call allocations. Build one with [`build_ctx`](@ref).
"""
struct Ctx{P} <: AbstractCtx
    tape::P
end
Ctx() = Ctx(nothing)

# ===========================================================================
# The tape — which is also the pullback closure for the generated fallback.
# `ArgsTT` is a `Tuple` of the primal's `CoDual` argument types, carried so that a `Tape`'s own call
# specialization can recover which primal method it belongs to (mirroring how the forwards carrier's
# `specTypes` names it directly). `CS` is a `Tuple` of per-primal-block comms-stack types
# (`Stack{T}` for a block with something to communicate, `SingletonStack{Tuple{}}` for one with
# nothing — mirroring Mooncake's `SharedDataPairs`/singleton-type optimization).
# ===========================================================================
mutable struct Tape{ArgsTT<:Tuple,CS<:Tuple}
    const block_stack::Stack{Int32}
    const comms::CS
    # The primal's `(fcd, argcds...)` coduals, stored by the forwards pass so the pullback can reach
    # argument values at all: `reverse_pullback_impl`'s own signature is `(tape, seed)`, so an
    # `Argument(k)` of the primal is otherwise simply unavailable to it, and everything derived from
    # one has to be pushed per-execution instead of read once.
    #
    # Non-`const` because `build_ctx` allocates the tape knowing only the argument *types*
    # (`_fresh_tape_expr`) — the values arrive later, on each call. The other two stay `const`: the
    # pullback reads them once and hoists, which a mutable field would inhibit. Assigning an inline
    # tuple field like this allocates nothing, just a write barrier.
    # Reusable buffers for bulk primal save/restore, one slot per bulk-saved argument
    # (`_bulk_save_args`). `const`: the `Vector` object is fixed for the tape's life, only its
    # contents change — which is what lets a pre-allocated context reuse the buffers instead of
    # allocating a copy per call. `_NO_BULK_BUFS` (a shared empty) when this primal saves nothing.
    #
    # Declared before `args` so the 3-argument constructor can leave *only* `args` undef: `bufs` is
    # read unconditionally by the pullback and must always be assigned.
    const bufs::Vector{Any}
    args::ArgsTT
    Tape{ArgsTT,CS}(block_stack, comms, bufs=_NO_BULK_BUFS) where {ArgsTT<:Tuple,CS<:Tuple} =
        new{ArgsTT,CS}(block_stack, comms, bufs)    # `args` deliberately left undef
    Tape{ArgsTT,CS}(block_stack, comms, bufs, args) where {ArgsTT<:Tuple,CS<:Tuple} =
        new{ArgsTT,CS}(block_stack, comms, bufs, args)
end

# ===========================================================================
# Two layers, as in forward mode (`frule!!` entry / `dualized_impl` carrier):
#
#   * The *entry* is the public surface — the `@generated` fallback method of `rrule!!` and the
#     `@generated` pullback callable `(t::Tape)(seed)`. Both compile the corresponding carrier under
#     `ADInterpreter{Reverse}` (which is what calls `build_contextual_ir` at all — ordinary
#     code compiles under a `NativeInterpreter`, which never calls it) and emit a static
#     `:invoke` to the result. The fwds entry is a straight pass-through: it invokes the carrier with
#     exactly its own `(fcd, ctx, argcds...)`, because the carrier mirrors `rrule!!`'s signature.
#   * The *carrier* is the hidden function whose specializations `build_contextual_ir` actually
#     transforms.
#     Its body below is the stub that runs only if the transform bailed *and* the more specific
#     `reverse_error_ircode` (which names the offending construct) was not installed.
#
# The `ctx` is a real argument of the fwds carrier — that is how a `Ctx` hands its pre-allocated
# stacks to the generated body.
# ===========================================================================

reverse_fwds_impl(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) =
    error("Differ.reverse_fwds_impl ran directly: ADInterpreter could not build the reverse " *
          "forwards pass (likely control flow Differ doesn't support yet, a mutable/undef-field " *
          "struct, a surviving high-level call, or an intrinsic with no registered reverse rule).")

reverse_pullback_impl(tape, seed) =
    error("Differ.reverse_pullback_impl ran directly: ADInterpreter could not build the reverse " *
          "pullback pass.")

# Is `mi` a *carrier* specialization — the thing `build_contextual_ir` transforms? The carrier is
# `reverse_fwds_impl(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...)`, so `ctx` is at `params[3]`.
# The `isa(mi.specTypes, DataType)` guard is load-bearing, not defensive: these run on *every*
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

# The `@generated` derived fallback — the least-specific `rrule!!` method (see the note above the
# type definitions). Recognized by its exact signature so `hand_reverse_rule_match` can tell "matched
# a hand rule" from "matched the fallback": a `findsup` on a concrete query always resolves *some*
# method now that the fallback exists.
is_generated_reverse_fwds_fallback(m::Method) =
    m.sig === Tuple{typeof(rrule!!),CoDual,AbstractCtx,Vararg{CoDual}}

# The `rrule!!` signature a *hypothetical* reverse-mode differentiation of `callee_mi` would resolve
# against — mirrors `implicit_frule_tt` (`forward_interp.jl`), built from `callee_mi.specTypes`
# instead of an actual differentiation call site. Returns `nothing` for anything the shape doesn't
# apply to, rather than throwing: `callee_mi` is an arbitrary callee discovered via `frame.edges`, not
# something the differentiation call site validated.
function implicit_rrule_tt(callee_mi::MethodInstance)
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
        argcodualtys = Any[fcodual_type(P) for P in params[2:end]]
        return Tuple{typeof(rrule!!),CoDual{ftype,NoFData},Ctx{Nothing},argcodualtys...}
    catch
        return nothing
    end
end

# An mt-backedge on the `rrule!!` resolution a *hypothetical* differentiation of `callee_mi` would
# use, registered even though `callee_mi`'s call was (or may have been) inlined away and never
# actually reached `reverse_fwds_recursive_ci`'s resolution — see the call site in
# `_optimized_primal_ir`. So a user later hand-writing `rrule!!` for a callee that was inlined away
# before a rule existed for it still invalidates a derivative built before that rule existed. Mirrors
# `register_implicit_frule_backedge!` (`forward_interp.jl`).
function register_implicit_rrule_backedge!(edges::Vector{Any}, callee_mi::MethodInstance)
    rrule_tt = implicit_rrule_tt(callee_mi)
    rrule_tt === nothing || push!(edges, rrule_tt, Core.methodtable)
    return nothing
end

# Resolve the hand-written `rrule!!` for a call, or `nothing` if only the derived fallback applies.
# The `ctx` slot of the query is `Ctx{Nothing}` (the fresh-tape mode a recursive inner call uses);
# every method — hand rule or fallback — declares that slot `::AbstractCtx`, so it never affects which
# method wins, only the fcd + arg slots do.
function hand_reverse_rule_match(interp::ADInterpreter, @nospecialize(ftype), argtypes)
    tt = Tuple{typeof(rrule!!),CoDual{ftype,NoFData},Ctx{Nothing},(fcodual_type(P) for P in argtypes)...}
    m, _ = CC.findsup(tt, CC.method_table(interp))
    (m === nothing || !isa(m.method, Method)) && return nothing
    is_generated_reverse_fwds_fallback(m.method) && return nothing
    return tt, m
end

# Does a hand-written `rrule!!` apply to a hypothetical reverse-mode differentiation of `callee_mi`?
# Mirrors `has_hand_frule` (`forward_interp.jl`): used by `src_inlining_policy` below to keep such a
# call from being inlined away before it ever reaches `_static_recursible_call`'s recursion dispatch,
# regardless of how cheap the callee looks to Julia's ordinary cost heuristic.
function has_hand_reverse_rule(interp::ADInterpreter, callee_mi::MethodInstance)
    isa(callee_mi.def, Method) || return false
    isa(callee_mi.specTypes, DataType) || return false     # `UnionAll` sig — see `is_reverse_fwds_impl`
    params = callee_mi.specTypes.parameters
    isempty(params) && return false
    ftype = params[1]
    (ftype isa Type && isconcretetype(ftype)) || return false
    argtypes = params[2:end]
    all(P -> P isa Type && isconcretetype(P), argtypes) || return false
    return hand_reverse_rule_match(interp, ftype, argtypes) !== nothing
end

# Is `mi` itself a `reverse_fwds_impl`/`reverse_pullback_impl` specialization — i.e. the target of
# one of Part 1's recursive `:invoke`s (`reverse_fwds_recursive_ci`/`reverse_pullback_recursive_ci`),
# hand-written rule or generated fallback alike? A hand rule's body is typically *small* (e.g. one
# `sin(x)` call) — small enough that Julia's ordinary cost heuristic will happily inline it, unlike
# the generated fallback (always sizable) or forward mode's `frule!!` hand rules for the same functions
# (which call *two* transcendentals — `Dual(sin(x), cos(x)*dx)` — comfortably over the inlining
# threshold, so this exact hazard never arose there). Inlining one back into its recursive caller is
# never correct here: the inlined statements carry GlobalRefs resolved relative to the *callee's own*
# defining module (confirmed empirically — a `sin(x)` call inside a hand rule in `src/rrules.jl`
# inlines as `GlobalRef(Differ, :sin)`, not `GlobalRef(Base, :sin)`), which
# `Core.Compiler.verify_ir` rejects as an "unbound or partitioned GlobalRef... in value position" once
# re-embedded in the caller's own compiled unit — so every recursive carrier invoke must stay a
# genuine `:invoke`, never inlined, regardless of apparent cost.
#
# Covers the two carriers and the hand-written `rrule!!` primitives. A hand-written *pullback* has no
# common supertype to test (it is a method on the rule author's own type, e.g. `SinPullback`), so
# those are blocked at the call site instead: the emitted pullback-recursion `:invoke` carries
# `CC.IR_FLAG_NOINLINE`, which `resolve_todo` honours regardless of the callee's type.
_is_reverse_carrier_mi(mi::MethodInstance) = isa(mi.def, Method) && isa(mi.specTypes, DataType) &&
    !isempty(mi.specTypes.parameters) &&
    (mi.specTypes.parameters[1] === typeof(reverse_fwds_impl) ||
     mi.specTypes.parameters[1] === typeof(reverse_pullback_impl) ||
     mi.specTypes.parameters[1] === typeof(rrule!!))

# Mirrors forward mode's own override in `forward_interp.jl`: never inline a call whose callee has a
# hand-written reverse-mode rule (so it survives into the primal IR for `_static_recursible_call`'s
# recursion dispatch to see and route to that rule via ordinary method-table resolution), and never
# inline a recursive reverse-mode carrier invoke once emitted (see `_is_reverse_carrier_mi` above).
function CC.src_inlining_policy(interp::ADInterpreter{Reverse}, mi::MethodInstance,
                                @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::UInt32)
    (_is_reverse_carrier_mi(mi) || has_hand_reverse_rule(interp, mi)) && return false
    return @invoke CC.src_inlining_policy(interp::CC.AbstractInterpreter, mi::MethodInstance,
                                          src::Any, info::CC.CallInfo, stmt_flag::UInt32)
end

# Build a minimal IRCode whose only effect is to `error(msg)` when invoked, installed via the same
# `finishinfer!`/`optimize` path as a real reverse-mode body (mirrors `error_ircode`,
# `forward_interp.jl`). Works for either carrier's argument shape (`_impl_argtypes` below).
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
function build_contextual_ir(interp::ADInterpreter{Reverse}, mi::MethodInstance)
    if is_reverse_fwds_impl(mi)
        reason = Ref("Differ could not build the reverse forwards pass (no specific reason recorded).")
        edges = Any[]
        ir = build_reverse_fwds_ir(interp, mi, reason, edges)
        interp.transformed_edges[mi] = edges
        if ir === nothing
            interp.bail_reasons[mi] = reason[]   # so a recursing caller can report *this* reason
            return reverse_error_ircode(mi, reason[])
        end
        return ir
    elseif is_reverse_pullback_impl(mi)
        reason = Ref("Differ could not build the reverse pullback pass (no specific reason recorded).")
        edges = Any[]
        ir = build_reverse_pullback_ir(interp, mi, reason, edges)
        interp.transformed_edges[mi] = edges
        if ir === nothing
            interp.bail_reasons[mi] = reason[]
            return reverse_error_ircode(mi, reason[])
        end
        return ir
    end
    return nothing
end

# ===========================================================================
# Shared primal resolution: both carriers eventually need the same (primal_mi, n) pair, obtained
# from the tuple of `CoDual` argument types (directly, for the fwds carrier; recovered from the
# `Tape`'s `ArgsTT` parameter, for the pullback carrier — see `build_reverse_pullback_ir`).
# ===========================================================================
function resolve_reverse_primal(interp::ADInterpreter, codualparams::Vector{Any},
                                reason::Ref{String}, edges::Vector{Any})
    if !all(P -> P isa Type && P <: CoDual, codualparams)
        reason[] = "not every codual argument type is a `CoDual` (a vararg call?)"
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
    if pmatch.method.isva
        reason[] = "the primal method $(pmatch.method) is a vararg method (not yet supported)"
        return nothing
    end
    primal_mi = specialize_method(pmatch.method, pmatch.spec_types, pmatch.sparams)::MethodInstance
    CC.add_inlining_edge!(edges, primal_mi)
    return (primal_mi, length(codualparams))
end

# The primal's optimized `IRCode`, computed the same way `build_dual_ir`/`build_reverse_ir` do
# (mirroring `Core.Compiler.typeinf_ircode`'s own body so `frame.edges` is available too).
function _optimized_primal_ir(interp::ADInterpreter, primal_mi::MethodInstance,
                              reason::Ref{String}, edges::Vector{Any})
    frame = CC.typeinf_frame(interp, primal_mi, false)
    if frame === nothing
        reason[] = "inference failed to produce optimized IR for the primal method $(primal_mi)"
        return nothing
    end
    opt = CC.OptimizationState(frame, interp)
    pir = CC.run_passes_ipo_safe(opt.src, opt, nothing)
    append!(edges, frame.edges)
    # For every concrete callee discovered above (regardless of whether its call survived or was
    # inlined away), also register the mt-backedge a hand-written `rrule!!` for it would need — see
    # `register_implicit_rrule_backedge!`. Mirrors `dualize_to_ircode`'s base case in
    # `forward_interp.jl`; covers both carriers, since both `build_reverse_fwds_ir` and
    # `build_reverse_pullback_ir` call this function.
    for (_, item) in CC.ForwardToBackedgeIterator(Core.svec(frame.edges...))
        isa(item, MethodInstance) && register_implicit_rrule_backedge!(edges, item)
    end
    return pir
end

# Recursion cycle guard (see `interp.in_progress`, `contextual.jl`), keyed by the *carrier* mi —
# unchanged from before this file supported self-recursion. This guard's job is purely "don't
# recompile a carrier that's already being compiled higher up the call stack" (mutual recursion,
# A→B→A); it has nothing to do with whether an edge is *cyclic* (same primal), which is a separate,
# ctx-independent question `reverse_fwds_recursive_ci`/`reverse_pullback_recursive_ci` answer on
# their own from an explicitly-passed `primal_mi` — see their docstrings.
#
# Carrier-mi keying matters because the *fwds* carrier alone has two independent specializations per
# primal: `Ctx{Nothing}` (fresh-tape, what a recursive inner call always targets) and `Ctx{<:Tape}`
# (pre-allocated, what `build_ctx(...; prealloc=true)` uses). Building the pre-allocated variant of a
# self-recursive primal genuinely requires *also* compiling the `Ctx{Nothing}` sibling (the recursive
# edge always targets that one) — a real, bounded, one-off nested compile, not a cycle. A primal-mi-
# keyed guard would wrongly treat that nested compile as "this primal is already in progress" and
# bail, even though the `Ctx{Nothing}` build being triggered is a *different* carrier that has never
# been built and terminates cleanly (its own self-edge resolves via literal carrier-mi identity, no
# further nesting). Keying by carrier mi keeps the two independent builds from colliding.
function build_reverse_fwds_ir(interp::ADInterpreter, impl_mi::MethodInstance,
                               reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    if haskey(interp.in_progress, impl_mi)
        reason[] = "recursive reverse-mode forwards-pass build for $(impl_mi) detected (a self- or " *
                   "mutually-recursive primal) — not yet supported; bailing instead of recursing forever"
        return nothing
    end
    interp.in_progress[impl_mi] = nothing
    try
        # Carrier is `reverse_fwds_impl(fcd, ctx, argcds...)`: `params[2]` is `fcd`, `params[3]` the
        # `ctx`, `params[4:end]` the argument coduals. The full codual list is `(fcd, args...)`.
        p = impl_mi.specTypes.parameters
        codualparams = Any[p[2], p[4:end]...]
        info = resolve_reverse_primal(interp, codualparams, reason, edges)
        info === nothing && return nothing
        primal_mi, n = info
        pir = _optimized_primal_ir(interp, primal_mi, reason, edges)
        pir === nothing && return nothing
        return reverse_fwds_to_ircode(interp, impl_mi, pir, n, primal_mi; reason, edges)
    finally
        delete!(interp.in_progress, impl_mi)
    end
end

function build_reverse_pullback_ir(interp::ADInterpreter, impl_mi::MethodInstance,
                                   reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    if haskey(interp.in_progress, impl_mi)
        reason[] = "recursive reverse-mode pullback-pass build for $(impl_mi) detected (a self- or " *
                   "mutually-recursive primal) — not yet supported; bailing instead of recursing forever"
        return nothing
    end
    interp.in_progress[impl_mi] = nothing
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
        primal_mi, n = info
        pir = _optimized_primal_ir(interp, primal_mi, reason, edges)
        pir === nothing && return nothing
        return reverse_pullback_to_ircode(interp, impl_mi, pir, n, primal_mi; reason, edges)
    finally
        delete!(interp.in_progress, impl_mi)
    end
end

# The optimized IR for a carrier: exactly what `CC.optimize` installs. Used both by
# `code_reverse_fwds_ircode`/`code_reverse_pullback_ircode` (reflection.jl) and available for
# future higher-order composition.
function optimized_reverse_fwds_ir(interp::ADInterpreter, impl_mi::MethodInstance,
                                   reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    ir = build_reverse_fwds_ir(interp, impl_mi, reason, edges)
    ir === nothing && return nothing
    world = CC.get_inference_world(interp)
    opt = CC.OptimizationState(impl_mi, CC.retrieve_code_info(impl_mi, world), interp)
    return run_ipo_passes!(ir, opt)
end
function optimized_reverse_pullback_ir(interp::ADInterpreter, impl_mi::MethodInstance,
                                       reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    ir = build_reverse_pullback_ir(interp, impl_mi, reason, edges)
    ir === nothing && return nothing
    world = CC.get_inference_world(interp)
    opt = CC.OptimizationState(impl_mi, CC.retrieve_code_info(impl_mi, world), interp)
    return run_ipo_passes!(ir, opt)
end

# ===========================================================================
# Shared static analysis: both the forwards and pullback builders need to agree, byte-for-byte, on
# (a) which primal blocks are throw-only/unreachable, (b) that there's exactly one reachable exit,
# (c) that there's no back-edge (Phase B: no loops yet), and (d) which runtime operand values each
# block's own statements need communicated from forwards to pullback. Since both builders derive
# `pir` identically (same `primal_mi`, deterministic optimization), computing this twice (once per
# builder) always agrees.
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
# *normal* shape Julia's optimizer produces (it does not generally merge arms into one exit via a
# phi + single return — confirmed by inspecting real IR, per the `adnext-extending-ir-support`
# methodology), so multi-exit is the common case, not a corner case: every exit needs its own
# routing, exactly like a `PhiNode`'s per-predecessor routing (see `reverse_pullback_to_ircode`).
function _exit_blocks(pir, unreachable)
    exits = Int[]
    for b in eachindex(pir.cfg.blocks)
        unreachable[b] && continue
        term = pir.stmts[pir.cfg.blocks[b].stmts.stop][:stmt]
        isa(term, Core.ReturnNode) && isdefined(term, :val) && push!(exits, b)
    end
    return exits
end

# ===========================================================================
# Bulk save/restore analysis: which arguments have their primal contents saved once per call, rather
# than one overwritten element at a time.
#
# A `memoryrefset!`'s pullback restores the element it overwrote, so that the primal is left exactly
# as the call found it. Doing that per element costs, *per store executed*, two tape slots (the old
# value and the primal `MemoryRef` to put it back through), one load on the forwards pass and one
# store on the pullback. In a loop that is O(iterations).
#
# It can instead be done once: no pullback rule anywhere ever *reads* primal memory (only this
# restore ever writes it — see `src/rrules.jl`'s header, where that is stated as a rule for hand
# rules to honour), so nothing observes the primal between the start of the pullback and its end.
# Only the net effect at the boundary is visible, and one `copyto!` in and one out reproduces it.
#
# Restricted to arguments whose element type is `isbits`: the root has to be reachable from the
# pullback (arguments are, via `Tape.args`; a locally-allocated `Memory` would need a comms item of
# its own), and a non-bits element is a reference whose own contents the copy would not capture.
# ===========================================================================

# Blocks that can execute more than once per call: the union of the natural loops of every back edge
# (an edge `b -> s` whose target dominates its source). Over-approximating this is harmless — it only
# ever moves a store from the per-element scheme to the bulk one, which is a cost decision, not a
# correctness one.
function _loop_blocks(pir)
    cfg = pir.cfg
    nb = length(cfg.blocks)
    inloop = falses(nb)
    dt = CC.construct_domtree(cfg.blocks)
    for b in 1:nb, s in cfg.blocks[b].succs
        (1 <= s <= nb) || continue
        CC.dominates(dt, s, b) || continue        # `b -> s` is a back edge
        # Natural loop body: the header `s`, plus everything that reaches `b` without going through
        # `s`. Walk predecessors from `b`, stopping at the header.
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
# lowers array indexing to (and that `_fdata_tracked` already tracks): `PiNode` aliases,
# `memoryrefnew` (both the 1-arg `Memory` form and the 3-arg offsetting form), and an `Array`'s
# `.ref` field. Returns the terminal node, or `nothing` if the chain runs into anything else.
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
# `reverse_fwds_to_ircode`'s main loop, which fuses this with live code-emission bookkeeping and so
# keeps its own inline version).
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
# `memoryrefnew(getfield(arr, :ref), idx, boundscheck)` with `arr` an `Argument(k)` and `idx` a
# literal `Int` can be rebuilt in the pullback from `tape.args[k]` + `idx`, so its handles need not
# be pushed on the comms tuple (which would make it GC-tracked).
#
# Returns `(k::Int, idx::Int, bc::Bool)` when `node` is such a `MemoryRef` SSA, else `nothing`
# (dynamic index, 1-arg form over a fresh allocation, or non-argument root).
# ===========================================================================
function _static_ref_derivation(pir, iworld, @nospecialize(node))
    isa(node, Core.SSAValue) || return nothing
    s = pir.stmts[node.id][:stmt]
    (isa(s, Expr) && (s.head === :call || s.head === :invoke)) || return nothing
    fpos, actual = _call_parts(s)
    _calleeval(fpos, iworld) === Base.memoryrefnew || return nothing
    length(actual) >= 3 || return nothing          # need the 3-arg offsetting form
    idx = _calleeval(actual[2], iworld)
    (idx isa Int) || return nothing                 # literal Int index only
    bc = _calleeval(actual[3], iworld)
    (bc isa Bool) || return nothing                 # literal Bool boundscheck
    root = _provenance_root(pir, iworld, actual[1])
    isa(root, Core.Argument) || return nothing
    return (root.n, idx, bc)
end

# ===========================================================================
# Phase D: unique-predecessor analysis (Mooncake's `_characterise_unique_predecessor_blocks`,
# `reverse_mode.jl:660-702`), worked directly over primal block *numbers* rather than the `ID`s
# `cfg_ir.jl`'s copy of that algorithm uses — the forwards pass never leaves block-number space (see
# this file's header: it needs none of the `ID`/`CFGBlock` layer), and the pullback pass already
# tracks predecessors by primal block number too (`pir.cfg.blocks[b].preds`), so there is no need to
# round-trip through `_ircode_to_cfg_blocks` just to reuse the `ID`-indexed version.
#
# `is_unique_pred[b]`: is block `b` the *only* predecessor of every one of its successors? If so, no
# successor of `b` can ever be ambiguous about where it came from, so `b` need not push its own
# number onto the block stack before whatever runs next (an ordinary successor block, *or* — since a
# single reachable exit is a de facto unique predecessor of "the pullback's own entry", the one place
# control can leave the function — the pullback's exit-routing switch).
#
# `pred_is_unique_pred[b]`: does `b` have exactly one predecessor, and is *that* predecessor a unique
# predecessor? This is the condition that actually governs whether *this* block needs to pop the
# block stack — not merely "does `b` have a single static predecessor edge": if `b`'s sole
# predecessor `p` pushes anyway (because *another* successor of `p` is ambiguous), `b` must still pop
# to keep the stack balanced, even though its own arrival is individually unambiguous.
# ===========================================================================
function _unique_predecessor_info(pir, exit_blocks::Vector{Int}, unreachable::AbstractVector{Bool},
                                  block_comms_nodes::Vector{Vector{Any}})
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
    # region's blocks (entry `br`, interior `chk`/`pass`) need to push, exactly as if their
    # ambiguous successor were unique — nothing downstream can ever need to know whether the direct
    # or the checked edge into `merge` ran. `chk`/`pass` fail the plain test above for the same
    # reason `br` does (their only real successor besides a dead end is the still-ambiguous
    # `merge`), so all of `quiet` needs forcing, not just `br`.
    regions, quiet = _collapsible_regions(pir, unreachable, block_comms_nodes)
    for b in quiet
        is_unique_pred[b] = true
    end

    # Per-edge pop (ISSUES #52): `b` pops iff it has multiple predecessors, each of which pushed on
    # its edge into `b` so `b` can learn which one fired. A single-predecessor block's sole
    # predecessor pushes *only* if `b` is ambiguous -- but `b` having a single predecessor means it
    # is not ambiguous, so that predecessor never pushes on the edge into `b`, and `b` must not pop
    # either. (The old formula `length(preds[b])==1 && is_unique_pred[only(preds[b])]` popped for
    # *balance* whenever the sole predecessor pushed for some *other* successor; that balance was
    # only needed because the forwards push was per-block. `_split_ambiguous_block_pushes` makes the
    # push per-edge, so the balance-pop is gone -- the two changes are coupled, see #52.) The entry
    # block's 0-predecessor case is covered directly by `<= 1`.
    pred_is_unique_pred = falses(nblocks)
    for b in 1:nblocks
        pred_is_unique_pred[b] = length(preds[b]) <= 1
    end
    # A collapsible region's `merge` block has two real predecessors, so the generic formula above
    # never marks it — force it directly: `reverse_pullback_to_ircode` routes it through the
    # canonical `br` alone (see `regions`), so nothing is ever popped on its behalf either.
    for merge in keys(regions)
        pred_is_unique_pred[merge] = true
    end

    return is_unique_pred, pred_is_unique_pred, regions
end

# ===========================================================================
# Collapsible regions: extends the unique-predecessor optimization above from a single edge to a
# whole comms-free sub-region. The fixed shape Julia's `@boundscheck` lowering produces around
# every `getindex`/`setindex!` is exactly this: an entry block `br` branches to `{merge, chk}` (the
# "skip the check" vs "run the check" edge); `chk` is a single-predecessor, comms-free block that
# itself branches to `{thrw, onward}` (checked-fail vs checked-pass), where `onward` reaches `merge`
# either directly or through a linear chain of further single-predecessor, single-successor,
# comms-free pass-through blocks (`setindex!`'s lowering reaches `merge` directly; `getindex`'s adds
# one or two trivial routing blocks in between — a bare `goto` / a placeholder `nothing` statement,
# nothing that carries a value); `thrw` is an `_unreachable_blocks` dead end (an `invoke
# Base.throw_boundserror` that never returns). Since `chk` and every chain block push nothing onto
# any comms stack (checked below via `block_comms_nodes`, never assumed) and `merge` has no leading
# `PhiNode` (checked below too — nothing differs by which edge was actually taken, both arms compute
# the same statements), the pullback never needs to know whether the forwards pass took the direct
# `br`→`merge` edge or the checked detour through the chain: replaying `merge` in reverse and always
# routing back through `br` directly reaches the correct upstream state either way, and the chain's
# own (empty) reverse processing is safe to skip outright.
#
# Deliberately narrow (a linear, comms-free chain with no branching of its own) rather than a
# general dominance/SESE analysis — this exact shape (plus however many trivial routing blocks the
# optimizer happens to leave in the chain) is what every array index produces, and anything that
# doesn't match it byte-for-byte falls through to the ordinary (still correct, just not free)
# unique-predecessor handling above — never guess. Returns `(merges, quiet)`: `merges` is
# a `Dict{Int,Int}` mapping a collapsible `merge` block to its canonical entry `br`; `quiet` is the
# `Set{Int}` of every block in every matched region (`br` plus its interior `chk`/`pass`) that must
# stop pushing onto the block stack.
# ===========================================================================
function _collapsible_regions(pir, unreachable::AbstractVector{Bool},
                              block_comms_nodes::Vector{Vector{Any}})
    nblocks = length(pir.cfg.blocks)
    preds = [filter(!=(0), pir.cfg.blocks[b].preds) for b in 1:nblocks]
    succs = [pir.cfg.blocks[b].succs for b in 1:nblocks]
    comms_free(b) = isempty(block_comms_nodes[b])
    dead_end(b) = unreachable[b] && isempty(succs[b])
    solo_pred(b, from) = length(preds[b]) == 1 && only(preds[b]) == from
    no_leading_phi(b) = !isa(pir.stmts[pir.cfg.blocks[b].stmts.start][:stmt], Core.PhiNode)
    # `merge`'s full predecessor set must reduce to exactly `{br, exitb}` — no third, unrelated edge
    # feeding it from elsewhere — and it must be a genuine reachable block, not itself a dead end.
    closes_at(merge, br, exitb) = !unreachable[merge] && no_leading_phi(merge) &&
                                  Set(preds[merge]) == Set((br, exitb))

    # Does `chk` (the "run the check" arm out of `br`) lead only to `merge` (directly, or via a
    # linear chain of further comms-free pass-through blocks) and a throw dead end, with nothing of
    # its own — or the chain's — to communicate back? Returns the interior block list
    # (`[chk]`, `[chk, pass1]`, `[chk, pass1, pass2]`, ...) on a match, `nothing` otherwise.
    function matches_check(br, chk, merge)
        (solo_pred(chk, br) && comms_free(chk) && length(succs[chk]) == 2) || return nothing
        for (thrw, onward) in ((succs[chk][1], succs[chk][2]), (succs[chk][2], succs[chk][1]))
            dead_end(thrw) || continue
            interior = [chk]
            cur = onward
            # Bounded by `nblocks`: a comms-free chain can't legitimately revisit a block (that
            # would need a real loop back-edge, which fails `solo_pred` — this is just a hard stop
            # against ever spinning on a shape this analysis didn't anticipate).
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
    # `merge`'s reverse-replay through `br` alone. `quiet`: every block in every matched region
    # (`br` plus its interior `chk`/`pass`) that must stop pushing onto the block stack — not just
    # `br` itself, since `chk`/`pass` independently fail the plain unique-predecessor test too (their
    # only real successor besides the dead end is `merge`, whose predecessor count doesn't change
    # just because we've decided to stop caring about it).
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
# ISSUES #52: split a block's stack push per *edge* rather than per *block*.
#
# `is_unique_pred[b] == false` makes `emit_epilogue!` push block `b`'s number unconditionally, but a
# `GotoIfNot` with one unambiguous arm (unique predecessor) and one ambiguous arm (multiple
# predecessors) only ever needs the push on the ambiguous arm — the unambiguous arm's target never
# pops it. This is a post-processing pass over the already-built fwds-carrier `ir`: relocate the push
# into a new relay block reached only via the ambiguous arm, using the `ID`/`CFGBlock` working-IR
# layer (`cfg_ir.jl`) the pullback pass already uses for its own extra routing blocks.
#
# Redirecting one arm through a relay changes the ambiguous target's *real* predecessor from `b` to
# the relay — any `PhiNode` there still names `b`, so its `edges` must be patched to match, or the
# result miscompiles (a stale edge reference is not something `verify_ir`/codegen catch on their own:
# the block number the edge resolves to still exists, it just now names the wrong predecessor).
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
(no round trip through the `cfg_ir.jl` layer at all) when there is nothing to split.
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

# Part 2 (read-only array indexing) static provenance analysis, extended by Part 3 (`src/CLAUDE.md`
# mutable-struct/array-mutation milestone) to general struct fields: which SSA statements' values have
# a statically-known *fdata* (shadow), traceable back to a function argument whose own fdata is
# non-trivial. Two chains:
#
#  * The array-index chain (original Part 2): *exactly* the shape Julia 1.13 lowers `x[i]` to, back to
#    an `Array` argument — confirmed by direct inspection of `Base.code_ircode` on `x[1]`/a
#    hand-written summation loop, per the `adnext-extending-ir-support` methodology:
#    `Base.getfield(x, :ref)::MemoryRef{T}` then `Base.memoryrefnew(ref, i, false)`, with an optional
#    `PiNode` alias in between.
#  * The general-struct chain (Part 3): `Core.getfield(x, fld)` off a tracked, non-`Array` object,
#    whenever the result itself has non-trivial fdata (a nested array or mutable-struct field) —
#    `_get_fdata_field` (`fwds_rvs_data.jl`) covers `FData`/`MutableTangent`/`Tuple`/`NamedTuple`
#    uniformly, so an immutable closure holding a mutable ref and a mutable struct holding another
#    mutable struct both work with the same emission path (`apply_builtin_rrule_fwds!`,
#    `builtins_reverse.jl`).
#
# Bounds the feature precisely: an array/struct reachable any other way (nested inside a returned
# value, freshly allocated, read out of a container via ordinary indexing) is untracked — and
# untracked-but-differentiable is a real bail at the point of use (`_scan_block_comms`/the builtin
# dispatch layer), never silently mishandled.
#
# Which top-level (fwds-carrier) arguments carry non-trivial fdata (an `Array`, a mutable struct, an
# immutable struct/closure with a mutable-struct or array field, ...). Factored out of
# `_fdata_tracked` so `_static_recursible_call`'s Part 3 array-argument-recursion guard can check a
# bare `Core.Argument` operand directly without recomputing this.
function _arg_fdata_tracked(n::Int, codualparams::Vector{Any})
    arg_tracked = falses(n)
    for k in 1:n
        arg_tracked[k] = fdtype(_codual_primal_type(codualparams[k])) !== NoFData
    end
    return arg_tracked
end

# Returns a `BitVector` of length `length(pir.stmts)`, indexed by SSA id (`tracked[i]`). Argument
# provenance (`Core.Argument(k)`) is checked inline via `arg_tracked` rather than returned, since
# every caller that needs it (`_scan_block_comms`'s `memoryrefget` case, the builtin dispatch layer)
# only ever looks the chain up starting from an `SSAValue` (the `memoryrefnew`/`PiNode`/general-struct
# `getfield` result the shadow handle actually flows through), never a bare `Argument` directly. Part 3
# (array-argument recursion, `_static_recursible_call`) additionally needs `arg_tracked` itself, for
# a call whose argument is a bare `Core.Argument`, so it's exposed via `_arg_fdata_tracked` above
# rather than recomputed.
function _fdata_tracked(pir, iworld, n::Int, codualparams::Vector{Any})
    N = length(pir.stmts)
    tracked = falses(N)
    arg_tracked = _arg_fdata_tracked(n, codualparams)
    provenance_tracked(@nospecialize node) =
        isa(node, Core.SSAValue) ? tracked[node.id] :
        isa(node, Core.Argument) ? (node.n <= n && arg_tracked[node.n]) : false
    for i in 1:N
        s = pir.stmts[i][:stmt]
        if isa(s, Core.PiNode)
            tracked[i] = provenance_tracked(s.val)
        elseif isa(s, Expr) && s.head === :new
            # A locally-created mutable struct is a fresh provenance root: the fwds pass always
            # builds it a real `MutableTangent` shadow (see the `:new` case in
            # `reverse_fwds_to_ircode`), so downstream `getfield`/`setfield!` on it can resolve a
            # shadow with no function-argument ancestor at all. Only when `T` actually has
            # differentiable content (`fdtype(T) !== NoFData`) -- otherwise `tangent_type(T) ===
            # NoTangent` and there is no `MutableTangent` type to build a shadow of. `_calleeval`
            # resolves a `GlobalRef` type argument (the common shape after optimization -- confirmed
            # by inspecting real IR, `%new(Main.MPoint, ...)`, not a literal `DataType`) at the
            # inference world.
            T = _calleeval(s.args[1], iworld)
            tracked[i] = T isa DataType && ismutabletype(T) && fdtype(T) !== NoFData
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
                elseif fdtype(Ti) !== NoFData && !(Pobj isa DataType && Pobj <: Array)
                    # Part 3: a general struct field whose own value has non-trivial fdata (a nested
                    # array or mutable-struct field) — kept off the `Array` case above, whose shadow is
                    # a raw `.ref`-chain `MemoryRef`, not something `_get_fdata_field` handles.
                    tracked[i] = true
                end
            elseif f === Core.memorynew && !isempty(actual)
                # Array allocation step 1: a fresh `Memory{P}` is itself a provenance root (no
                # ancestor to check), exactly like a locally-`%new`'d mutable struct above — its
                # shadow `Memory` backs the `memoryrefnew`/`%new(Vector,...)` that follow in the
                # same 4-statement allocation sequence.
                MT = _widen(pir.stmts[i][:type])
                tracked[i] = MT isa DataType && MT <: Memory && fdtype(MT) !== NoFData
            elseif f === Base.memoryrefnew && !isempty(actual) && provenance_tracked(actual[1])
                tracked[i] = true
            elseif f === Base.memoryrefget && !isempty(actual) && provenance_tracked(actual[1])
                # A read off a tracked ref whose own *result* is itself array/mutable-struct-valued
                # (a nested array) is a provenance root in turn — its shadow is the corresponding
                # element of the shadow array (mirrored onto the shadow ref in the fwds emission,
                # `apply_builtin_rrule_fwds!(::Val{Base.memoryrefget}, ...)`, `builtins_reverse.jl`).
                # A scalar (bits) result carries no fdata of its own, so is deliberately left
                # untracked here — its gradient flows via rdata routing, not shadow aliasing.
                fdtype(pir.stmts[i][:type]) !== NoFData && (tracked[i] = true)
            end
        end
    end
    return tracked
end

# `P` is usually a plain `Type` (`pir.stmts[i][:type]`), but a statement whose result the primal's
# own const-prop narrowed (e.g. a `Core.memorynew` call with a literal length) can carry a
# `Core.PartialStruct`/`Const` lattice element instead — widen it first, since `tangent_type` is only
# ever defined on `Type`s. Mirrors `tt` in `forward_interp.jl` (`dualize_to_ircode`).
_widen(@nospecialize T) = T isa Type ? T : CC.widenconst(T)
rdtype(@nospecialize P) = rdata_type(tangent_type(_widen(P)))
fdtype(@nospecialize P) = fdata_type(tangent_type(_widen(P)))

# Extract the callee-position node and actual-argument nodes from a `:call`/`:invoke` statement —
# the same split used at every other call-statement site in this file (`_scan_block_comms`,
# the fwds/pullback per-statement dispatch), factored out once here since Part 1's recursion path
# needs it independently from the main dispatch loops (once during the comms scan, once again
# during emission).
_call_parts(s::Expr) = s.head === :invoke ? (s.args[2], @view s.args[3:end]) : (s.args[1], @view s.args[2:end])

# Is call statement `i` (`s`, already known to be a surviving, non-intrinsic, non-`getfield`,
# non-array-builtin `:call`/`:invoke`) a candidate for Part 1's recursive `rrule` support? Purely
# static (no compilation): resolves the callee value and argument/result *types* only. Returns
# `(fval, ftype, argtypes)` on success or `nothing` (with `reason[]` set) otherwise.
#
# Three guards, in order:
#  1. Callee must be statically resolvable to a concrete, non-tangent-carrying value (an ordinary
#     top-level function/singleton — `tangent_type(ftype) === NoTangent` — never a closure with
#     differentiable captures nor a dynamically-dispatched callee). This is what lets the recursive
#     invoke pass `CoDual{ftype,NoFData}` for the callee slot with no fdata to thread through.
#  2. Every argument type must be concrete, and its fdata must either be trivial (`NoFData`) or a
#     real `Array` whose *identity* is traceable back to a function argument of the current fwds
#     carrier (`arg_tracked`/`fdata_tracked`, Part 3). This is the load-bearing correctness guard,
#     not just a missing-feature guard: passing a recursive call's argument a freshly-zeroed fdata
#     with no link to the real shadow would be silently wrong, not just unsupported — so any
#     non-trivial fdata that isn't a *real, traceable* array shadow must still bail (a mutable
#     struct's fdata, or an array reachable any other way — nested in a struct field, returned from
#     a call, freshly allocated — is untracked and stays unsupported, mirroring
#     `_fdata_tracked`'s own read-indexing scope).
#  3. The call's own result type must likewise carry trivial fdata — array-*valued results* from a
#     recursive call are a separate, not-yet-supported feature (the fwds pass has nowhere to route a
#     result shadow today).
function _static_recursible_call(pir, iworld, i::Int, s::Expr, reason::Ref{String},
                                 arg_tracked::BitVector, fdata_tracked::BitVector)
    fpos, actual = _call_parts(s)
    fval = _calleeval(fpos, iworld)
    # `_calleeval` returns `nothing` for a callee in argument position (an `Argument`/`SSAValue`, e.g.
    # `mapreduce_impl(f, op, A, ...)`'s `f`) — no compile-time *value*, but the recursion machinery
    # below never needs one: `reverse_fwds_recursive_ci` takes `ftype`, not `fval`, and `fval` is used
    # only once, to build the callee's `CoDual` at emission (below, in the fwds pass) — where
    # `fval === nothing` means "resolve the operand there instead" (`presolve(fpos)`). So fall back to
    # the operand's *type* (`_optype_w`, which widens `Const`/`PartialStruct` and handles
    # `GlobalRef`/`QuoteNode`) rather than bailing outright. This is what lets a call like
    # `sum(sin, v)` recurse: `sin`'s type is `typeof(sin)`, a concrete singleton, even though its
    # *value* isn't statically known inside `mapreduce_impl`'s body.
    ftype = fval === nothing ? _optype_w(pir, iworld, fpos) : _typeof(fval)
    if !(ftype isa DataType && isconcretetype(ftype))
        reason[] = "callee type $(ftype) is not a concrete DataType at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    if ftype === DataType
        # "Some Type value, identity erased" — mirrors the argument-side rejection below
        # (`P === DataType`): a genuinely concrete singleton type value (`Type{Float64}`) is handled
        # fine by `fcodual_type`; a bare `DataType` isn't, and the `%new(CoDual{...}, ...)` this guard
        # protects (the fwds-pass emission, below) would be illegal IR.
        reason[] = "recursive call with a Type-valued callee of erased identity is not supported " *
                   "yet at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    if tangent_type(ftype) !== NoTangent
        reason[] = "recursive calls into a callee with differentiable captures ($(ftype)) are not " *
                   "supported yet at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    # `_optype_w`, not `_optype`: an operand can be a `GlobalRef`/`QuoteNode` naming a value (e.g.
    # `sum(sin, v)`, whose `sin` operand `_optype` would type as `GlobalRef`), and the types derived
    # here decide which `rrule!!` the recursion resolves *and* annotate the `%new(CoDual{…}, …)` the
    # emission side builds from `presolve`d values — so a node-shaped answer both misresolves the
    # rule and emits IR whose declared type doesn't match the value in it.
    argtypes = Any[_optype_w(pir, iworld, a) for a in actual]
    for (j, P) in enumerate(argtypes)
        if !(P isa DataType && isconcretetype(P))
            reason[] = "recursive call has a non-concrete argument type $(P) at %$i: `$(_stmt_str(s))`"
            return nothing
        end
        # `P === DataType` means "some Type value, identity erased" (unlike `Type{Float64}`, a
        # genuinely concrete singleton `fcodual_type` handles directly) — `fcodual_type(DataType)`
        # special-cases this to a bare abstract `CoDual`, which would make the `%new` this guard
        # exists to keep safe (fwds-pass emission, below) illegal IR. Reject it here instead.
        if P === DataType
            reason[] = "recursive call with a Type-valued argument of erased identity is not " *
                       "supported yet at %$i: `$(_stmt_str(s))`"
            return nothing
        end
        if fdtype(P) !== NoFData
            # An array or mutable-struct argument is allowed through — but only if its *identity*
            # is statically traceable back to a tracked function argument (or a tracked local
            # `%new`, see `_fdata_tracked`), so the emission side (below) always has a real shadow
            # value (`sresolve`) to thread through the recursive `:invoke` rather than a detached
            # `NoFData()`. A mutable-struct argument needs no rdata back from the inner call at
            # all: the inner call's rule accumulates into the *same*, shared `MutableTangent` in
            # place (fdata semantics), so the gradient is already there once the call returns.
            if !(fdata_type(tangent_type(P)) <: Array || ismutabletype(P))
                reason[] = "recursive call with a non-array, non-mutable-struct argument ($(P)) " *
                           "whose tangent carries fdata is not supported yet at %$i: " *
                           "`$(_stmt_str(s))`"
                return nothing
            end
            a = actual[j]
            tracked_here = isa(a, Core.Argument) ? (a.n <= length(arg_tracked) && arg_tracked[a.n]) :
                           isa(a, Core.SSAValue) ? fdata_tracked[a.id] : false
            if !tracked_here
                reason[] = "recursive call with an array/mutable-struct argument ($(P)) whose " *
                           "provenance is not traceable to a function argument is not supported " *
                           "yet at %$i: `$(_stmt_str(s))`"
                return nothing
            end
        end
    end
    if fdtype(pir.stmts[i][:type]) !== NoFData
        reason[] = "recursive call with a non-trivial-fdata result ($(pir.stmts[i][:type])) is not " *
                   "supported yet at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    return (fval, ftype, argtypes)
end

# Resolve (and compile, under the *caller's own* `interp`) the `CodeInstance` for the callee's
# `reverse_fwds_impl(CoDual{ftype,NoFData}(fval,NoFData()), argcoduals...)` specialization, so Part
# 1's recursion support can emit a static `:invoke` to it (mirroring `frule_codeinstance` in
# `forward_interp.jl`, but — unlike that function, which resolves under a *fresh*
# `CC.NativeInterpreter` because it targets the generic `frule!!` function whose own `@generated` body
# separately spins up a nested `ADInterpreter{Forward}` — this must reuse the caller's own `interp`:
# `reverse_fwds_impl` specializations only get transformed via `build_contextual_ir` when
# compiled under an `ADInterpreter`, so a bare `NativeInterpreter` would just compile the
# untransformed stub (`error("ran directly...")`) instead of a real recursive forwards pass. Reusing
# `interp` is also what makes the `in_progress` cycle guard (`build_reverse_fwds_ir` above) actually
# see a genuine cycle — see that guard's docstring.
#
# An inner call resolves one of two ways. Both share the *same* argument layout `(fcd, ctx, argcds...)`
# — the uniform `rrule!!` convention, which the fwds carrier mirrors exactly — so the only thing the
# caller needs from this is which value goes in the `:invoke`'s callee slot:
#
#   * a hand-written `rrule!!` for the callee, if one exists — invoked as `rrule!!`;
#   * otherwise the *derived* path: the fwds carrier `reverse_fwds_impl`, invoked directly.
#
# Either way the emitted `:invoke` passes a fresh `Ctx()` in the ctx slot: a recursive inner call
# always uses the tape-allocating mode — an inner pullback is a value pushed onto the *outer* block's
# comms stack, one per execution, so there is nothing an outer caller's pre-allocated tape could
# stand in for. (See the limitation note on `Ctx`.)
#
# Returns `(ci, callee_val, InnerFCoDualT, InnerPullbackT)` on success or `nothing` (with `reason[]`
# set); the emit sites build `invoke(callee_val, ci, fcd, Ctx(), argcds...)`.
#
# `impl_mi` is the *carrier* mi of the build this call is part of; `current_primal_mi` is that
# build's own primal mi (passed all the way down from `build_reverse_fwds_ir`'s
# `resolve_reverse_primal` call). `R` is this call statement's own (widened) primal result type, read
# straight off the primal IR — already fixed-point-solved by Julia's own inference when it built
# `pir`, so no extra resolution is needed.
#
# Direct self-recursion (the callee's own primal mi is exactly `current_primal_mi`) never needs a
# fixed point solved for its `Tape` type: the comms slot is declared as the bare `Tape` UnionAll
# (`InnerPullbackT` below), which is what makes the comms-tuple type finite. That part is
# ctx-independent and unconditional whenever the primal mi matches — it must be, since both the fwds
# and pullback builders, and every ctx-type variant of the fwds carrier (`Ctx{Nothing}` fresh-tape vs
# `Ctx{<:Tape}` pre-allocated — see `_scan_block_comms`'s callers), have to agree on it or the
# concrete `Tape` type they each separately compute would disagree.
#
# *Whether compiling is needed* is a narrower, ctx-dependent question, decided by comparing the
# resolved recursive-call target `callee_impl_mi` (always `Ctx{Nothing}`, per the fixed convention
# below) against `impl_mi` itself:
#   * `callee_impl_mi === impl_mi` — a literal self-edge (only possible when this build's own ctx is
#     already `Ctx{Nothing}`). No compile: `CC.typeinf_ext_toplevel` is skipped entirely, and codegen's
#     `mi == ctx.linfo` self-recursion fast path (`src/codegen.cpp`) triggers for the emitted `:invoke`.
#   * otherwise — same primal, different carrier (this build is the `Ctx{<:Tape}` pre-allocated
#     variant, whose recursive edge always targets the `Ctx{Nothing}` sibling). That sibling genuinely
#     has never been compiled, so it must be, via the ordinary `typeinf_ext_toplevel` path — but this
#     always terminates in exactly one bounded nested compile: building it hits *its own* recursive
#     edge, for which its own ctx *is* `Ctx{Nothing}`, so that inner resolution takes the literal-
#     identity branch above and stops. (This is also why `interp.in_progress` stays keyed by carrier
#     mi, not primal mi — see its docstring in `contextual.jl`: a primal-keyed guard would mistake
#     this legitimate nested compile for the primal already being in progress and bail incorrectly.)
function reverse_fwds_recursive_ci(interp, impl_mi::MethodInstance, current_primal_mi::MethodInstance,
                                   @nospecialize(ftype), argtypes::Vector{Any}, @nospecialize(R),
                                   edges::Vector{Any}, reason::Ref{String})
    argcodualtys = Any[fcodual_type(P) for P in argtypes]
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
                push!(edges, tt, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
                # `CC.findall`, not `findsup` — see the ABI note above `helper_ci`: `findall` gives a
                # `spec_types` already intersected with `tt`, so `specialize_method` yields a
                # `MethodInstance` whose `specTypes` is exactly `tt`.
                matches = CC.findall(tt, CC.method_table(interp))
                if matches === nothing || length(matches) != 1 || !matches[1].fully_covers
                    reason[] = "no reverse-mode rule resolves for recursive (self-cyclic) call " *
                               "signature $(tt)"
                    return nothing
                end
                callee_impl_mi = specialize_method(matches[1])::MethodInstance
                if callee_impl_mi === impl_mi
                    CC.add_inlining_edge!(edges, callee_impl_mi)
                    return callee_impl_mi, reverse_fwds_impl, CoDual{R,NoFData}, Tape
                else
                    ci = CC.typeinf_ext_toplevel(interp, callee_impl_mi, CC.SOURCE_MODE_ABI)::CodeInstance
                    CC.add_invoke_edge!(edges, tt, ci)
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
    # The pullback half (`InnerRT.parameters[2]`) is deliberately left unconstrained beyond "some
    # concrete type": the derived path always returns a `Tape{...}`, but a hand-written rule
    # (`src/rrules.jl`) is free to use whatever's cheapest to remember — this glue never inspects a
    # pullback's internals, only threads it through opaquely and calls it.
    if !(InnerRT isa DataType && InnerRT <: Tuple && length(InnerRT.parameters) == 2 &&
         InnerRT.parameters[1] isa DataType && InnerRT.parameters[1] <: CoDual &&
         isconcretetype(InnerRT.parameters[2]))
        # Two distinct causes, worth telling apart: `Union{}` means the callee's own carrier bailed
        # and only its error stub compiled (its recorded reason, if this same interpreter built it,
        # names the real problem); anything else means a rule *did* resolve but its return type
        # isn't the concrete `(CoDual, pullback)` pair the emission below needs — typically because
        # an argument type handed to it was wrong or too abstract for the rule's body to infer.
        inner = get(interp.bail_reasons, callee_impl_mi, nothing)
        reason[] = inner !== nothing ? "the reverse-mode build for `$(tt)` bailed: $(inner)" :
            InnerRT === Union{} ?
                "the reverse rule resolved for `$(tt)` never returns — either its own derived " *
                "build bailed, or it cannot run on those argument types" :
                "the reverse rule resolved for `$(tt)` returned `$(InnerRT)`, which is not a " *
                "concrete `(CoDual, pullback)` pair"
        return nothing
    end
    CC.add_invoke_edge!(edges, tt, ci)
    # (ci, callee_val, InnerFCoDualT, InnerPullbackT)
    return ci, callee_val, InnerRT.parameters[1], InnerRT.parameters[2]
end

# Mirrors `reverse_fwds_recursive_ci` for the pullback carrier: resolves
# `reverse_pullback_impl(tape::InnerTapeT, seed::InnerSeedT)`. Simpler than the fwds case — by the
# time the pullback builder needs this, `InnerTapeT` is already a known-concrete type (it was
# resolved once already, while building the *fwds* pass — see `_scan_block_comms`'s `:subtape`
# handling below — and both builders derive it identically since `CC.cache_owner` is a mode-level
# singleton, so the `CodeInstance` compiled there is found, not recompiled, here), so no "did the
# callee bail" recovery logic is needed beyond a defensive rettype-shape check.
#
# Cyclic edge (self-recursion): `InnerTapeT` arrives as the bare `Tape` UnionAll — the marker
# `reverse_fwds_recursive_ci` leaves in a cyclic block's comms type (see `_scan_block_comms`) — never
# a concrete type there, so it can't be resolved into a `tt` directly. The concrete type is
# `own_TapeT`, the *current* pullback build's own tape type (`impl_mi.specTypes.parameters[2]`): a
# self-recursive callee's tape is by construction the same type as the caller's own. The callee's
# rettype is likewise closed-form regardless of whether compiling turns out to be needed below
# (`zero_like_rdata_type` of each argument's primal type, exact per the derived pullback's own return
# convention, `:2196-2203` below), computed straight from `own_codualparams` (also the current build's
# own — same reasoning).
#
# Whether compiling is needed mirrors `reverse_fwds_recursive_ci`'s fwds-side reasoning, but the
# source of the mismatch differs: the pullback carrier has no ctx-type variance (`reverse_pullback_impl`
# takes only `(tape, seed)`), but the recursive call's own seed type (`InnerSeedT` — the local
# accumulator's type at *this* call site) need not equal the current build's own incoming seed type
# (`impl_mi`'s own `SeedT`) even for a literally self-recursive primal. `pb_mi === impl_mi` catches
# exactly the case where it does (both `own_TapeT` and the seed type agree) — codegen's self-recursion
# fast path, no compile. Otherwise `pb_mi` is a genuine, not-yet-compiled sibling (same tape, different
# seed type), compiled the ordinary way; that nested compile's own recursive edge targets *its own*
# seed type, so it resolves via the literal-identity branch and terminates.
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
    # *carrier* directly rather than the generated `(t::Tape)(seed)` entry — the entry is only a
    # one-line `:invoke` wrapper, and skipping it keeps this a single static call. A hand-written
    # pullback is a method on the rule author's own type, so the object is its own callee.
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
        inner = get(interp.bail_reasons, callee_impl_mi, nothing)
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
# Static resolution of the *runtime helpers* this engine emits (`push!`/`pop!`/`increment!!`/
# `Stack` construction/…), so they can be emitted as `Expr(:invoke, ci, f, args...)` rather than
# `Expr(:call, f, args...)`.
#
# Why this is mandatory rather than an optimization: both carriers build their IR by hand, so every
# statement carries `CC.NoCallInfo()` (`CC.InstructionStream(len)` in `reverse_fwds_to_ircode`;
# `new_inst` in `cfg_ir.jl`). Inference never ran over this IR, so nothing ever populates `info` —
# and `assemble_inline_todo!` bails on exactly that ("Inference determined this couldn't be
# analyzed. Don't question it.", `Compiler/src/ssair/inlining.jl`). A surviving `:call` to a
# non-builtin therefore compiles to a full `jl_apply_generic`: boxed arguments, method lookup,
# boxed return — so a single `increment!!(::Float64, ::Float64)`, which is one `add_float`, costs
# three heap allocations and a dynamic dispatch on *every* accumulation of *every* iteration.
# `:invoke` sidesteps the whole problem: `assemble_inline_todo!` dispatches `handle_invoke_expr!`
# *before* the `NoCallInfo` check, so an `:invoke` gets the unboxed specsig ABI and stays eligible
# for inlining. This is the same fix forward mode already uses for surviving high-level calls
# (`frule_codeinstance` + the `Expr(:invoke, ci, ...)` emission in `forward_interp.jl`).
#
# Two details that are load-bearing:
#
#  * `CC.findall`, not `CC.findsup`. `findall` returns matches whose `spec_types` are already
#    *intersected* with `tt`, so `specialize_method` yields a concrete `MethodInstance` and the
#    `:invoke` uses the specialized (unboxed) ABI. `findsup` can hand back the method's own widened
#    signature (e.g. `Tuple{typeof(push!),SingletonStack,Any}` for `push!(::SingletonStack,::Any)`),
#    which would still box every argument — a static call, but not a fast one.
#
#  * Compile under `interp`, not a fresh `CC.NativeInterpreter`. `resolve_todo` looks the callee up
#    in `code_cache(state.interp)`, keyed by `CC.cache_owner(::ADInterpreter{Reverse}) === Reverse()`;
#    a `CodeInstance` compiled by a `NativeInterpreter` lands in a *different* cache, so it would be
#    found by codegen but never by the inliner. (`frule_codeinstance` uses a `NativeInterpreter`
#    because forward mode deliberately blocks inlining of frules via `src_inlining_policy` — that
#    reasoning does not apply to these helpers, which we very much want inlined.) Routing them
#    through `interp` is safe: `build_contextual_ir` returns `nothing` for any non-carrier
#    MethodInstance, and `src_inlining_policy(::ADInterpreter{Reverse}, ...)` only blocks carrier
#    and hand-rule MIs, so `push!`/`increment!!`/… infer and inline exactly as they would natively.
#
# Returns `nothing` if the signature doesn't resolve to a single method (the caller then falls back
# to a plain `:call` — slow, but correct; nothing here is ever load-bearing for correctness).
# `cache` memoizes per builder invocation, since the same handful of signatures recur at many sites.
# ===========================================================================
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
# forwards to pullback (see this file's header). `getfield`/`memoryrefnew` need no runtime value at
# all (their reverse rules route by static type + node identity only, or — for the array shadow
# chain, Part 2 — are rebuilt deterministically by both passes independently), so only intrinsic
# operands (`:primal`), recursive-call inner tapes (`:subtape`), and tracked array reads
# (`:shadow_ref`) ever need comms. Plain tuples, not a new struct — they already work as `Dict` keys.
#
# NOT side-effect-free once recursion is present: resolving a `:subtape` candidate compiles that
# callee's own `reverse_fwds_impl` (via `reverse_fwds_recursive_ci`), a deliberate departure from
# "pure static scan" flagged here prominently. Both builders still derive it identically (so they
# agree on every block's comms-tuple type) because `CC.cache_owner(interp::ADInterpreter{M})` is a
# mode-level singleton: a `CodeInstance` compiled while scanning from one builder's `interp` is
# found, not recompiled, when the other builder's separate `interp` instance resolves the same
# callsite. `interp`/`reason`/`edges` are only exercised by the `:subtape`/`:shadow_ref` paths;
# plain intrinsic/`getfield` scanning is unchanged from before recursion/arrays existed.
function _scan_block_comms(interp, scan_impl_mi::MethodInstance, primal_mi::MethodInstance, pir, iworld,
                           unreachable, codualparams::Vector{Any}, reason::Ref{String}, edges::Vector{Any})
    nblocks = length(pir.cfg.blocks)
    nodes = [Any[] for _ in 1:nblocks]
    types = [Any[] for _ in 1:nblocks]
    n = length(codualparams)
    fdata_tracked = _fdata_tracked(pir, iworld, n, codualparams)
    arg_tracked = _arg_fdata_tracked(n, codualparams)
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
            # A tracked mutable `%new` (see `_fdata_tracked`) needs its fresh shadow communicated
            # forward to the pullback, exactly like any other object's `(:fshadow, obj)` item
            # (`getfield`/`setfield!`) — except here the object *is* this statement's own result,
            # so it's keyed by its own `SSAValue` rather than an existing operand.
            T = _calleeval(s.args[1], iworld)
            # `T <: Array` excluded: an array's shadow is a plain `Array`, never referenced via a
            # `(:fshadow, obj)` comms item (that's only ever produced by `getfield`/`setfield!` on a
            # genuinely mutable struct — `.ref`/`.size` reads on an array always have `NoRData`, so
            # `Core.getfield`'s own comms rule never fires for them). Declaring one here would just be
            # dead weight on every block that allocates an array.
            if T isa DataType && !(T <: Array) && ismutabletype(T) && fdtype(T) !== NoFData
                push!(nodes[b], (:fshadow, Core.SSAValue(i)))
                push!(types[b], fdtype(T))
            end
            continue
        end
        (isa(s, Expr) && (s.head === :call || s.head === :invoke)) || continue
        fpos, actual = _call_parts(s)
        f = _calleeval(fpos, iworld)
        if isa(f, Core.IntrinsicFunction)
            # Two reasons an intrinsic operand needs no comms slot — both decided purely from the
            # primal IR, so the fwds and pullback builders derive identical tuple types (the
            # invariant this whole scan exists to maintain):
            #
            #  * The statement carries no gradient at all (`NoRData` result — every integer/boolean/
            #    comparison intrinsic). The pullback never even reaches `apply_intrinsic_rrule!` for
            #    it, so nothing it consumed can ever be read back.
            #  * The rule for this specific intrinsic doesn't read that operand — see
            #    `intrinsic_rrule_operands`. A linear op (`add_float`, `neg_float`, …) reads none of
            #    them; before this, every `s += x` in a loop was pushing both addends per iteration
            #    and popping them into a value the pullback never used.
            rdtype(pir.stmts[i][:type]) === NoRData && continue
            needed = intrinsic_rrule_operands(Val(f))
            for (k, a) in enumerate(actual)
                (isa(a, Core.SSAValue) || isa(a, Core.Argument)) || continue
                (needed === nothing || k in needed) || continue
                any(nd -> nd == (:primal, a), nodes[b]) && continue
                push!(nodes[b], (:primal, a))
                push!(types[b], _optype(pir, a))
            end
        elseif isa(f, Core.Builtin)
            # The dispatch layer (`builtins_reverse.jl`) decides everything: whether this call needs
            # comms items, and whether this specific case is even in scope (types + static provenance
            # — a registered rule that declines sets `reason[]` and returns `false`; `nothing` means
            # "unregistered", which needs no comms and never bails here — any bail for an arbitrary
            # differentiable builtin with no rule happens at point-of-use, in the emission loops).
            ctx = (optype=(@nospecialize x) -> _optype(pir, x), ssa=Core.SSAValue(i),
                  tracked=fdata_tracked, arg_tracked=arg_tracked, reason=reason,
                  bulk_saved=bulk_saved,
                  # A `MemoryRef` statically re-derivable from an argument + literal index need not
                  # be pushed (see `_static_ref_derivation`). Consulted by `builtins_reverse.jl`.
                  static_ref=(@nospecialize x) -> _static_ref_derivation(pir, iworld, x))
            result = builtin_rrule_comms(Val(f), actual, pir.stmts[i][:type], ctx)
            result === false && return nothing
            if result !== nothing
                for (item, ty) in result
                    any(nd -> nd == item, nodes[b]) && continue
                    push!(nodes[b], item)
                    push!(types[b], ty)
                end
            end
        else
            # A surviving high-level call: attempt Part 1's static recursion. Not qualifying is *not*
            # an error at scan time (mirrors how an unregistered intrinsic isn't flagged here either
            # — only the main per-statement loop, reaching it for real, turns that into a bail); only
            # a genuine attempted-and-failed resolution propagates as a real bail here.
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
            # For a cyclic edge this is the bare `Tape` UnionAll (see `reverse_fwds_recursive_ci`),
            # which is exactly what makes the comms-tuple type finite — a concrete `TapeT` isn't known
            # yet at scan time (it depends on this very comms scan), and doesn't need to be: the
            # abstract element type is enough to close the loop.
            InnerTapeT = resolved[4]
            push!(nodes[b], (:subtape, Core.SSAValue(i)))
            push!(types[b], InnerTapeT)
        end
    end
    return nodes, types, bulk_args
end

# ===========================================================================
# Forwards pass: 1:1 block-topology-preserving replay of the primal (built the same way
# `dualize_to_ircode` is, minus any shadow/tangent — see this file's header), instrumented with the
# block-stack/comms pushes described above.
# ===========================================================================
function reverse_fwds_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int, primal_mi::MethodInstance;
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
    # `ctx`, `params[4:end]` are the argument coduals. `codualparams`/`ArgsTT` are the *full* codual
    # list `(fcd, args...)` — the shape the rest of this builder and the pullback side both expect.
    # `vararg_tt` is only the *argument* coduals: it types the packed vararg slot `Argument(4)`.
    codualparams = Any[impl_mi.specTypes.parameters[2], impl_mi.specTypes.parameters[4:end]...]
    vararg_tt = Tuple{impl_mi.specTypes.parameters[4:end]...}
    ArgsTT = Tuple{codualparams...}

    scan = _scan_block_comms(interp, impl_mi, primal_mi, pir, iworld, unreachable_block, codualparams, reason, edges)
    scan === nothing && return nothing
    # `bulk_args`: argument positions whose primal contents are saved/restored once per call rather
    # than per overwritten element. Derived inside the scan so both builders get it identically —
    # they must agree exactly, since it decides which comms items exist.
    block_comms_nodes, block_comms_types, bulk_args = scan
    bulk_slot = Dict(k => j for (j, k) in enumerate(sort!(collect(bulk_args))))
    # Needs `block_comms_nodes` (collapsible-region detection — see `_collapsible_regions`), hence
    # computed after the scan above rather than before it as its Phase-D-only predecessor was.
    is_unique_pred, _, _ = _unique_predecessor_info(pir, exit_blocks, unreachable_block, block_comms_nodes)
    # Same predicate `_scan_block_comms` used to decide the comms items, recomputed here so the
    # emission sides agree with the declaration side by construction.
    pb_bulk_saved(@nospecialize ref_node) = _is_bulk_saved(pir, iworld, bulk_args, ref_node)
    fdata_tracked = _fdata_tracked(pir, iworld, n, codualparams)
    arg_tracked = _arg_fdata_tracked(n, codualparams)

    getf = GlobalRef(Core, :getfield)
    setf = GlobalRef(Core, :setfield!)
    ctuple = GlobalRef(Core, :tuple)
    push_g = Base.push!
    zerofcodual_g = zero_fcodual

    code = Any[]; types = Any[]
    emit!(ex, @nospecialize(ty)) = (push!(code, ex); push!(types, ty); Core.SSAValue(length(code)))
    opf(name, ty, args...) = emit!(Expr(:call, GlobalRef(Core.Intrinsics, name), args...), ty)

    # Emit a call to a runtime helper as a static `:invoke` (see `helper_ci` for why a plain `:call`
    # here would be a full dynamic dispatch). `argtys` is the helper's *declared* argument types —
    # already known at every call site, since every emitted SSA value carries its type. `f` goes in
    # the `:invoke`'s callee value position as the function/type *object*, never a `GlobalRef`
    # (`verify_ir` rejects a `GlobalRef` there — see the same note in `dualize_to_ircode`).
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
        # codual 1 is `fcd` — `Argument(2)` directly; coduals 2..n are the argument coduals, packed
        # in the vararg tuple `Argument(4)` at position `i-1`. `Argument(3)` is the `ctx`.
        ci = i == 1 ? Core.Argument(2) : emit!(Expr(:call, getf, Core.Argument(4), i - 1), Ci)
        carg[i] = ci
        parg[i] = emit!(Expr(:call, getf, ci, 1), Pi)
        Fi !== NoFData && (farg[i] = emit!(Expr(:call, getf, ci, 2), Fi))
    end
    # Packed once here rather than at each use: both tape shapes below need it, and the
    # pre-allocated shape stores it in the prologue (not at the return) so an early bail can't sink
    # the store past a point where the tape is already visible to the caller.
    args_tup_ssa = emit!(Expr(:call, ctuple, carg...), ArgsTT)

    # --- Tape prologue. Two shapes, chosen by the `ctx` type in `Argument(3)`. ---
    # Select `CommsCell{T}` (single-slot inline holder) for a non-loop block whose comms tuple is
    # `isbits` (pushed once per call): the carrier emits a direct `setfield!`/`getfield`, no
    # `push!`/`pop!`/`position`/boundscheck. A loop block or non-`isbits` tuple keeps `Stack{T}`;
    # an empty tuple uses `SingletonStack`.
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
        # Tape-allocating mode: build the block stack and one comms stack per block.
        block_stack_ssa = icall!(Stack{Int32}, Stack{Int32}, ())
        for b in 1:nblocks
            ST = comms_stack_ty[b]
            # `SingletonStack` has no fields, so `%new`. `CommsCell{T}()` zero-fills `val`.
            # `Stack{T}` takes a capacity (pre-size to 1 for a single push).
            comms_stack_ssa[b] = ST <: SingletonStack ? emit!(Expr(:new, ST), ST) : icall!(ST, ST, ())
        end
    else
        # Pre-allocated mode: read the caller's tape out of the `ctx` and reuse its stacks. This is
        # the whole point of `build_ctx(...; prealloc=true)` — a `Stack` is three heap objects
        # (`Stack`, `Vector`, `Memory`), so allocating one per block per call was, by the end, the
        # entire remaining per-call allocation cost.
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
        # Reset every reusable stack to empty. Balanced push/pop already leaves them at 0 after a
        # completed round trip (the `check_stack_balance` tests cover exactly that), but resetting
        # here is what makes reuse correct unconditionally — a caller is free to run the forwards
        # pass and never call the pullback, or to bail out partway through.
        emit!(Expr(:call, setf, block_stack_ssa, 2, 0), Any)
        for b in 1:nblocks
            ST = comms_stack_ty[b]
            (ST <: SingletonStack || ST <: CommsCell) && continue
            # `Stack` only: reset `position` (field 2) to 0. `CommsCell` is overwritten in place;
            # `SingletonStack` is empty.
            emit!(Expr(:call, setf, comms_stack_ssa[b], 2, 0), Any)
        end
        # This call's coduals replace the previous call's. A re-used context therefore keeps the
        # *previous* call's arguments (and their shadows) alive until the next call overwrites them
        # — noted in `build_ctx`'s docstring; the field is concretely typed, so there is nothing to
        # null it to in between.
        emit!(Expr(:call, setf, tape_ssa, 4, args_tup_ssa), Any)
    end

    # --- Bulk primal save. Still in the prologue — before any primal statement has run — so it
    # captures the arguments exactly as the call found them. The buffers live on the tape, so a
    # pre-allocated context reuses them across calls and a steady-state call allocates nothing; a
    # `Ctx()` call allocates each buffer once, on its tape's first (and only) use. ---
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
        Pk = _widen(_codual_primal_type(codualparams[k]))
        icall!(_bulk_save!, Nothing, (Vector{Any}, Int, Pk), bufs_ssa, bulk_slot[k], parg[k])
    end

    primal_map = Vector{Any}(undef, N)
    shadow_map = Vector{Any}(undef, N)   # Part 2: array shadow (MemoryRef) chain, sparse — only
                                          # `fdata_tracked[i]` entries are ever assigned or read.
    # A bare `GlobalRef` operand (e.g. a `ReturnNode`'s `.val` for a function ending `return nothing`)
    # must be resolved to its actual value here, not passed through: an unresolved `GlobalRef` outside
    # `Core`/`Base` (a `Main` binding, say) is illegal in value position (`verify_ir`) — exactly the
    # hazard `_calleeval` already exists to handle for callees (`forward_interp.jl`).
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

    # Forward-reference patches for a `PhiNode` operand not yet resolved when the phi is processed
    # (a loop back-edge: the operand is defined later in linear statement order). Keyed by the
    # referenced *original* SSA index; each entry is (target values-vector, slot) — mirrors the
    # `pending` mechanism in `dualize_to_ircode` (`forward_interp.jl`), minus the shadow half (this
    # pass only ever computes one value per statement, never a primal+shadow pair).
    pending = Dict{Int,Vector{Tuple{Vector{Any},Int}}}()

    block_start_new = Vector{Int}(undef, nblocks)
    block_start_new[1] = 1
    bidx = 1

    emit_epilogue!(b) = begin
        nodes = block_comms_nodes[b]
        ST = comms_stack_ty[b]
        # Nothing to communicate: the stack is a `SingletonStack` whose `push!` is a no-op, so the
        # tuple construction and the push are both pure overhead. (The pullback already skips the
        # matching `pop!` for exactly this case.)
        if !isempty(nodes)
            vals = (nd[1] === :primal ? presolve(nd[2]) :
                    nd[1] === :subtape ? inner_tape_map[nd[2].id] :
                    (nd[1] === :old_primal || nd[1] === :old_tangent) ?
                        get(() -> error("Differ internal error: comms item $(nd) was declared but " *
                                        "never saved by its rule's fwds emission (builtin_rrule_comms/" *
                                        "apply_builtin_rrule_fwds! disagree)"),
                            block_saved[b], nd) :
                    sresolve(nd[2]) for nd in nodes)
            CommsT = Tuple{block_comms_types[b]...}
            tup = emit!(Expr(:call, ctuple, vals...), CommsT)
            if ST <: CommsCell
                # Unrolled single-slot store: `setfield!` to `val` (field 1) — no `push!`/`position`/boundscheck.
                emit!(Expr(:call, setf, comms_stack_ssa[b], 1, tup), Any)
            else
                icall!(push_g, Any, (ST, CommsT), comms_stack_ssa[b], tup)
            end
        end
        # Phase D: skip the block-stack push when `b` is a unique predecessor of whatever runs next
        # (an ordinary successor, or — for a lone exit — the pullback's own entry routing): nothing
        # downstream can ever be ambiguous about having come from `b`, so there is nothing to record.
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
        # next — an ordinary successor block (Phase B: unconditionally, no unique-pred skip yet) *or*
        # the pullback's own entry, which pops this same stack to learn which of possibly *several*
        # reachable exits actually ran (a branch with a `return` in each arm — the common case, not a
        # corner case, see `_exit_blocks`) and routes accordingly, exactly like a `PhiNode`'s
        # per-predecessor routing. Not conditioned on "the terminator is an explicit GotoNode/
        # GotoIfNot": Julia's optimizer leaves some fallthrough blocks with no explicit terminator at
        # all (last statement a bare placeholder like `nothing`), yet they still have a real successor.
        #
        # Defer the push when the terminator is a `PhiNode` (merge-only block with implicit
        # fallthrough): a push before the phi violates `verify_ir`'s "phi leads its block" rule.
        # Control-transfer terminators keep the pre-statement push; `_split_ambiguous_block_pushes`
        # later relocates the `GotoIfNot` case per-edge.
        defer_epilogue = is_terminator && !unreachable_block[bidx] && isa(s, Core.PhiNode)
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
                # Pure control marker (undef-var/boxed-capture guard) — see the matching arm in the
                # main (reachable-block) loop below for the full rationale. `args[1]` (name) is copied
                # verbatim; `args[2]` (condition) is resolved like any other operand.
                primal_map[i] = emit!(Expr(:throw_undef_if_not, s.args[1], presolve(s.args[2])), Ti)
            elseif isa(s, Expr) && s.head === :loopinfo
                # Same treatment (and same `julia.ivdep` filtering) as the main loop's `:loopinfo` arm
                # below — see there for the full rationale.
                args = filter(a -> a !== Symbol("julia.ivdep"), s.args)
                primal_map[i] = emit!(Expr(:loopinfo, args...), Ti)
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
            # `_optype` has no world to resolve a bare `GlobalRef` with (unlike `presolve` above), so
            # a `GlobalRef`-valued return (`return nothing` at `Main` scope, say) needs its type read
            # off the already-resolved `ret_val` instead of the unresolved node.
            R = isa(s.val, GlobalRef) ? Core.Typeof(ret_val) : _optype(pir, s.val)
            result_cd = icall!(zerofcodual_g, fcodual_type(R), (R,), ret_val)
            # Pre-allocated mode returns the caller's own tape object; otherwise `%new` one around
            # the stacks the prologue just built.
            tape = tape_ssa
            if tape === nothing
                comms_tuple = emit!(Expr(:call, ctuple, comms_stack_ssa...), Tuple{comms_stack_ty...})
                tape = emit!(Expr(:new, TapeT, block_stack_ssa, comms_tuple, bufs_ssa, args_tup_ssa), TapeT)
            end
            final = emit!(Expr(:call, ctuple, result_cd, tape), Tuple{fcodual_type(R),TapeT})
            emit!(Core.ReturnNode(final), Any)
        elseif isa(s, Core.PiNode)
            primal_map[i] = presolve(s.val)
            fdata_tracked[i] && (shadow_map[i] = sresolve(s.val))
        elseif isa(s, Expr) && s.head === :new
            # `_calleeval` resolves a `GlobalRef` type argument (the common post-optimization shape,
            # e.g. `%new(Main.MPoint, ...)`) at the inference world; a literal `DataType` passes
            # through unchanged.
            T = _calleeval(s.args[1], iworld)
            if !(T isa DataType) || !is_always_fully_initialised(T)
                reason[] = "reverse mode does not support structs with possibly-undef fields " *
                           "($(T)) at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            args = @view s.args[2:end]
            primal_map[i] = emit!(Expr(:new, T, (presolve(a) for a in args)...), Ti)
            if T <: Array
                # Array allocation step 4 of 4 (`%new(Vector{P}, ref, size)`): the shadow is a real
                # same-shape `Array{tangent_type(P),N}`, never a `MutableTangent` — this must run
                # BEFORE the generic mutable-struct branch below, which would otherwise wrongly claim
                # this `%new` too (`ismutabletype(Vector) === true`). Exactly 2 fields: `:ref`
                # (differentiable — `sresolve` to the shadow `MemoryRef` already built by the
                # `Core.memorynew`/`Base.memoryrefnew` steps above) and `:size` (structural, not
                # differentiable — the primal's own size tuple, `presolve`d not `sresolve`d). Mirrors
                # forward mode's identical `T <: Array` branch (`forward_interp.jl`).
                TT = tangent_type(T)
                shadow_map[i] = emit!(Expr(:new, TT, sresolve(args[1]), presolve(args[2])), TT)
            elseif ismutabletype(T) && fdtype(T) !== NoFData
                # Fresh shadow `MutableTangent`, mirroring Mooncake's `_new_` rrule
                # (`ismutabletype(P)` branch): each field's initial tangent is
                # `zero_tangent(field_primal, field_fdata)`, which *aliases* the assigned value's
                # own shadow when that field carries fdata (array/mutable-struct field) instead of
                # fabricating a detached zero — the same aliasing `Core.setfield!`'s rule relies on
                # (`builtins_reverse.jl`). This SSA is a tracked provenance root (`_fdata_tracked`
                # marks it), so downstream `getfield`/`setfield!` on it resolve to this shadow
                # exactly like a tracked function argument.
                field_tangents = Any[]
                for (j, a) in enumerate(args)
                    Fty = fieldtype(T, j)
                    FTj = fdtype(Fty)
                    if FTj !== NoFData
                        tracked_here = isa(a, Core.SSAValue) ? (a.id <= length(fdata_tracked) && fdata_tracked[a.id]) :
                                       isa(a, Core.Argument) ? (a.n <= length(arg_tracked) && arg_tracked[a.n]) : false
                        if !tracked_here
                            reason[] = "reverse mode `%new` of a mutable struct with a field " *
                                       "($(Fty)) whose assigned value's fdata is not traceable " *
                                       "to a function argument is not supported at %$i: " *
                                       "`$(_stmt_str(s))`"
                            return nothing
                        end
                        fdata_val = sresolve(a)
                    else
                        fdata_val = NoFData()
                    end
                    push!(field_tangents, icall!(_rr_zero_tangent2, tangent_type(Fty), (Fty, FTj), presolve(a), fdata_val))
                end
                argtys = (Type{T}, Tuple(tangent_type(fieldtype(T, j)) for j in eachindex(args))...)
                shadow_map[i] = icall!(_rr_build_tangent, tangent_type(T), argtys, T, field_tangents...)
            end
        elseif isa(s, Expr) && s.head === :boundscheck
            primal_map[i] = emit!(Expr(:boundscheck, s.args...), Ti)
        elseif isa(s, Expr) && s.head === :throw_undef_if_not
            # Pure control marker: raises `UndefVarError`/`UndefRefError` for an unassigned slot or
            # boxed-capture field (the guard around a captured, reassigned variable's read). `args[1]`
            # is a bare Symbol/GlobalRef name, never a value to resolve; `args[2]` is the Bool
            # condition, which is a genuine operand here (a literal in a throw-only block — handled
            # in the unreachable-block arm above — or an SSA reference on a live path). No fdata: its
            # result is never consumed.
            primal_map[i] = emit!(Expr(:throw_undef_if_not, s.args[1], presolve(s.args[2])), Ti)
        elseif isa(s, Expr) && s.head === :loopinfo
            # `@simd`'s loop marker. Pure metadata, copied through like `:boundscheck` above (no
            # `shadow_map` entry): its operands are `Symbol`s/`nothing`, and `:loopinfo` isn't in the
            # compiler's `is_relevant_expr`, so `userefs` never traverses them — `presolve`ing them
            # would be wrong, not just unnecessary. Same reasoning as forward mode's
            # `forward_interp.jl:1379-1392`; not restated here.
            #
            # One difference from forward mode: `julia.ivdep` is dropped, `julia.simdloop` and
            # anything else pass through. `ivdep` asserts no loop-carried memory dependence, which
            # forward mode's shadow honestly preserves (mirrors the primal's access pattern
            # one-for-one) but the reverse carrier does not — `emit_epilogue!` pushes onto the tape's
            # stacks on every iteration, which *is* a loop-carried dependence. Keeping the marker
            # would be a silent miscompile; dropping it only costs vectorization, which the carrier's
            # `rrule!!` calls and stack pushes preclude in practice anyway.
            #
            # The other difference — ISSUES #65's original worry — turned out not to apply: codegen's
            # `LowerSIMDLoop` resolves the loop from the marker's *basic block* via LLVM `LoopInfo`,
            # not by scanning backward from the terminator (measured on Julia 1.13), so
            # `emit_epilogue!`'s pushes landing between this statement and the block terminator is
            # harmless.
            args = filter(a -> a !== Symbol("julia.ivdep"), s.args)
            primal_map[i] = emit!(Expr(:loopinfo, args...), Ti)
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos, actual = _call_parts(s)
            f = _calleeval(fpos, iworld)
            if isa(f, Core.IntrinsicFunction)
                primal_map[i] = emit!(Expr(:call, f, (presolve(a) for a in actual)...), Ti)
            elseif isa(f, Core.Builtin)
                bctx = (; emit!, icall!, presolve, sresolve,
                       optype=(@nospecialize x) -> _optype(pir, x), tracked=fdata_tracked,
                       ssa=Core.SSAValue(i), bulk_saved=pb_bulk_saved)
                result = apply_builtin_rrule_fwds!(Val(f), actual, Ti, bctx)
                if result !== nothing
                    p, shadow, saved = result
                    primal_map[i] = p
                    shadow !== nothing && (shadow_map[i] = shadow)
                    isempty(saved) || merge!(block_saved[bidx], saved)
                elseif tangent_type(_widen(Ti)) === NoTangent
                    # A non-differentiable builtin result (`===`, `isa`, comparisons, ... — e.g. the
                    # `Base.iterate`-state check a `for i in 1:length(x)` loop's own lowering embeds)
                    # has no gradient to route, so no rule/comms is needed — just replay it primally,
                    # the same treatment `GlobalRef`/literal statements already get.
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
                                                     edges, reason)
                if resolved === nothing
                    reason[] *= " — at %$i: `$(_stmt_str(s))`"
                    return nothing
                end
                ci, callee_val, InnerFCoDualT, InnerTapeT0 = resolved
                # Cyclic edge: `reverse_fwds_recursive_ci` leaves this the bare `Tape` UnionAll (see
                # its docstring); the concrete type is this build's own `TapeT`, already computed
                # above (`:1276`) — a self-recursive callee's tape is, by construction, the same type
                # as the caller's own.
                InnerTapeT = InnerTapeT0 === Tape ? TapeT : InnerTapeT0
                FCT = CoDual{ftype,NoFData}
                # `fval === nothing` means `_static_recursible_call` resolved `ftype` from the
                # operand's type, not its value (an argument-position callee) — `presolve(fpos)`
                # resolves the operand itself uniformly, exactly as it already does for the argument
                # coduals just below.
                fcodual = emit!(Expr(:new, FCT, fval === nothing ? presolve(fpos) : fval, NoFData()), FCT)
                argcoduals = Any[]
                for (j, a) in enumerate(actual)
                    Cj = fcodual_type(argtypes[j])
                    # Part 3: thread the argument's *real* shadow through when its fdata is
                    # non-trivial (an array whose identity `_static_recursible_call` has already
                    # confirmed is traceable to a function argument) — `sresolve` resolves both a
                    # bare `Core.Argument` (via `farg`) and a tracked `SSAValue` (via `shadow_map`)
                    # uniformly. A hardcoded `NoFData()` here would silently detach the callee's
                    # accumulation from the caller's real buffer for any such argument.
                    fdata_val = fdtype(argtypes[j]) === NoFData ? NoFData() : sresolve(a)
                    push!(argcoduals, emit!(Expr(:new, Cj, presolve(a), fdata_val), Cj))
                end
                # Uniform layout `(fcd, ctx, argcds...)` for both callees; `Ctx()` is the fresh-tape
                # mode every recursive inner call uses (spliced as a singleton constant).
                pair = emit!(Expr(:invoke, ci, callee_val, fcodual, Ctx(), argcoduals...),
                            Tuple{InnerFCoDualT,InnerTapeT})
                result_cd = emit!(Expr(:call, getf, pair, 1), InnerFCoDualT)
                primal_map[i] = emit!(Expr(:call, getf, result_cd, 1), Ti)
                inner_tape_map[i] = emit!(Expr(:call, getf, pair, 2), InnerTapeT)
            end
        elseif isa(s, Core.GotoNode)
            emit!(Core.GotoNode(s.label), Any)
        elseif isa(s, Core.GotoIfNot)
            emit!(Core.GotoIfNot(presolve(s.cond), s.dest), Any)
        elseif isa(s, Core.PhiNode)
            k = length(s.values)
            pvals = Vector{Any}(undef, k)
            for j in 1:k
                isassigned(s.values, j) || continue
                v = s.values[j]
                if isa(v, Core.SSAValue) && !isassigned(primal_map, v.id)
                    push!(get!(() -> Tuple{Vector{Any},Int}[], pending, v.id), (pvals, j))
                else
                    pvals[j] = presolve(v)
                end
            end
            primal_map[i] = emit!(Core.PhiNode(s.edges, pvals), Ti)
        elseif isa(s, GlobalRef)
            primal_map[i] = emit!(s, Ti)
        elseif !isa(s, Expr)
            primal_map[i] = presolve(s)
        else
            reason[] = "unsupported statement kind $(typeof(s)) at %$i: `$(_stmt_str(s))`"
            return nothing
        end
        if haskey(pending, i)
            for (arr, slot) in pending[i]
                arr[slot] = primal_map[i]
            end
            delete!(pending, i)
        end
        # Deferred for a phi-terminator block (see above): the push follows the phi so it leads
        # its block, still within this block's range.
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
    # `_split_ambiguous_block_pushes` (ISSUES #52): relocate the per-block block-stack push onto
    # only the ambiguous arm of a mixed `GotoIfNot` (one ambiguous + one unambiguous successor), so
    # the forwards push becomes per-edge. This is the companion to the per-edge `pred_is_unique_pred`
    # formula above: with the push now per-edge, the pullback must stop balance-popping single-pred
    # successors (`pred_is_unique_pred[b] = length(preds[b]) <= 1`), which it now does. The two
    # changes are coupled -- neither is correct alone. See ISSUES.md #52.
    ir = _split_ambiguous_block_pushes(ir, pir, is_unique_pred)
    CC.verify_ir(ir)
    return ir
end

# ===========================================================================
# Pullback pass: walks the primal's blocks in reverse, over a *freshly built* CFG (not 1:1 with the
# primal — extra phi-routing blocks are inserted, and multi-way predecessor dispatch is lowered from
# a `Switch` into a `GotoIfNot` chain by `lower_cfg_blocks_to_ir`), using the `ID`/`CFGBlock` layer
# from `cfg_ir.jl`. rdata accumulators are real mutable `Ref`s (one per primal SSA + one per
# argument), not the flat per-statement SSA-merge scheme the old straight-line PoC used — that only
# worked because nothing was ever revisited. Push/pop is unconditional here (Phase B: no
# unique-predecessor skip yet — Phase D adds that as a pure optimization on top).
# ===========================================================================

# `Base.pop!`, not a bare `pop!`: this is inlined into synthetic carrier IR, where a bare call
# resolves as `GlobalRef(Differ, :pop!)` and trips `verify_ir` — see the note in `src/stack.jl`.
@inline __pop_blk_stack!(block_stack) = Base.pop!(block_stack)::Int32
# Fully qualified for the same reason as `__pop_blk_stack!` above: a bare `===` inlines as
# `GlobalRef(Differ, :(===))` — an implicit `using Base` binding, which is not a *defined const*
# binding and so trips `verify_ir`'s value-position check (the check exempts `Core`/`Base`, which is
# exactly what naming them explicitly buys).
@inline __switch_case(id::Int32, prev::Int32) = Base.:!(Core.:(===)(id, prev))

# Materializes a real zero rdata from `acc` when it's the `ZeroRData` placeholder. Needed wherever a
# `deref_and_zero!`-derived accumulator must be treated as an actual `RDataT` value (e.g. its own
# `NamedTuple` wrapper decomposed field-by-field for an immutable `%new`) rather than merely
# `increment!!`-ed into (which already handles `ZeroRData` generically, needing no instantiation).
# `@noinline`: this gets threaded through `icall` into hand-built carrier IR; without it,
# `CC.ssa_inlining_pass!` would inline the tiny `@inline`-marked body straight in, and that body
# contains a bare call that would resolve as `GlobalRef(Differ, ...)` in value position, which
# `verify_ir` rejects (same reasoning as the `_rr_*` helper convention in `src/builtins_reverse.jl`).
@noinline _rr_realize_rdata(acc, ::Type{RDataT}) where {RDataT} =
    (acc isa ZeroRData ? zero_rdata_from_type(RDataT) : acc)::RDataT

function reverse_pullback_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int,
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

    scan = _scan_block_comms(interp, impl_mi, primal_mi, pir, iworld, unreachable_block, codualparams, reason, edges)
    scan === nothing && return nothing
    # `bulk_args`: argument positions whose primal contents are saved/restored once per call rather
    # than per overwritten element. Derived inside the scan so both builders get it identically —
    # they must agree exactly, since it decides which comms items exist.
    block_comms_nodes, block_comms_types, bulk_args = scan
    # Needs `block_comms_nodes` (collapsible-region detection — see `_collapsible_regions`), hence
    # computed after the scan above rather than before it as its Phase-D-only predecessor was.
    _, pred_is_unique_pred, regions = _unique_predecessor_info(pir, exit_blocks, unreachable_block, block_comms_nodes)
    bulk_slot = Dict(k => j for (j, k) in enumerate(sort!(collect(bulk_args))))
    # Same predicate `_scan_block_comms` used to decide the comms items, recomputed here so the
    # emission sides agree with the declaration side by construction.
    pb_bulk_saved(@nospecialize ref_node) = _is_bulk_saved(pir, iworld, bulk_args, ref_node)
    fdata_tracked = _fdata_tracked(pir, iworld, n, codualparams)
    arg_tracked = _arg_fdata_tracked(n, codualparams)

    getf = GlobalRef(Core, :getfield)
    setf = GlobalRef(Core, :setfield!)
    ctuple = GlobalRef(Core, :tuple)
    pop_g = Base.pop!
    increment_g = increment!!

    # Build the statement for a call to a runtime helper: a static `:invoke` when the callee
    # resolves, else a plain `:call` (see `helper_ci` for why the difference is the whole ballgame
    # here). Returns the `Expr` rather than emitting it, because this builder has several distinct
    # emit closures (`eemit!` for the entry block, `remit!` per exit route, `emit!` per reverse
    # block) and every one of them needs it. `argtys` is the helper's declared argument types, known
    # at each site from the types of the values being passed.
    hcache = IdDict{Any,Any}()
    icall(@nospecialize(f), argtys::Tuple, args...) = begin
        ci = helper_ci(interp, Tuple{Core.Typeof(f),argtys...}, edges, hcache)
        ci === nothing ? Expr(:call, f, args...) : Expr(:invoke, ci, f, args...)
    end

    # Which block each statement belongs to (reused throughout).
    stmt_block = _stmt_block_map(pir)

    # Every statement except a pure control marker (or one living in a throw-only block) gets a
    # `Ref` to accumulate rdata into; literal/GlobalRef/`:boundscheck`/`:loopinfo` operands never do
    # (no gradient to route to — both always have `NoRData`, so skipping them here just avoids a
    # useless allocation, it isn't load-bearing).
    needs_ref(i) = !unreachable_block[stmt_block[i]] &&
                   !isa(pstmts[i][:stmt], Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode}) &&
                   !(isa(pstmts[i][:stmt], Expr) && (pstmts[i][:stmt]::Expr).head in (:boundscheck, :loopinfo))

    entry_id = ID()
    block_id = [ID() for _ in 1:nblocks]

    # The node each exit block returns — potentially a different one per exit (e.g. each arm of an
    # if/else returning its own value; see `_exit_blocks`).
    exit_ret_node = Dict(b => pstmts[pir.cfg.blocks[b].stmts.stop][:stmt].val for b in exit_blocks)

    # --- Entry block: unpack the tape and allocate every rdata `Ref`. Which exit actually ran is
    # not known statically (there may be several — an ordinary branch returning early in each arm is
    # the common case, not a corner case), so which one gets seeded from the incoming `seed` is
    # decided by a runtime switch below, exactly like a `PhiNode`'s per-predecessor routing. ---
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

    comms_obj_id = Vector{Any}(undef, nblocks)
    for b in 1:nblocks
        isempty(block_comms_types[b]) && continue
        comms_obj_id[b] = eemit!(Expr(:call, getf, comms_tuple_id, b), comms_stack_ty[b])
    end

    # --- The primal's own coduals, stored on the tape by the forwards pass (`Tape.args`). This is
    # the pullback's only route to an `Argument(k)` of the primal: its own signature is
    # `(tape, seed)`. Everything is unpacked eagerly, and unused entries are DCE'd by
    # `run_ipo_passes!` — emitting lazily is not an option, because the entry block's *terminator*
    # is emitted into this same `entry_stmts` vector by `_emit_switch!` below, and `CFGBlock` copies
    # the vector, so anything appended after that point would either land past a terminator or be
    # silently dropped. ---
    parg_pb = Vector{Any}(undef, n)
    farg_pb = Vector{Any}(undef, n)
    let args_tup_id = eemit!(Expr(:call, getf, tape_id, 4), ArgsTT)
        for k in 1:n
            Ck = codualparams[k]
            Pk = _codual_primal_type(Ck)
            Fk = _codual_fdata_type(Ck)
            cd = eemit!(Expr(:call, getf, args_tup_id, k), Ck)
            parg_pb[k] = eemit!(Expr(:call, getf, cd, 1), Pk)
            Fk !== NoFData && (farg_pb[k] = eemit!(Expr(:call, getf, cd, 2), Fk))
        end
    end

    arg_ref_id = Vector{Any}(undef, n)
    for k in 1:n
        Pk = _codual_primal_type(codualparams[k])
        # `zero_like_rdata_type`/`zero_like_rdata_from_type`, not `rdtype`/`zero_rdata_from_type`:
        # when `Pk` isn't concrete enough to produce a real zero rdata from its type alone (e.g. an
        # abstractly-typed argument slot), the ref's element type must include `ZeroRData` and the
        # zero literal must be `ZeroRData()` instead of crashing (`zero_rdata_from_type` returns the
        # `CannotProduceZeroRDataFromType` sentinel in that case, which `:new`'s field-type check
        # below rejects). Both collapse to the old behavior exactly when `Pk` is concrete.
        RT = zero_like_rdata_type(Pk)
        arg_ref_id[k] = eemit!(Expr(:new, Base.RefValue{RT}, zero_like_rdata_from_type(Pk)), Base.RefValue{RT})
    end

    ssa_ref_id = Vector{Any}(undef, N)
    for i in 1:N
        needs_ref(i) || continue
        # `_widen`: a statement's inferred type can be a lattice element (`Core.PartialStruct`), not
        # a bare `Type` — e.g. a `Core.memorynew` call with a literal length — and
        # `zero_rdata_from_type` (like `tangent_type`) is only ever defined on `Type`s.
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
            # `zero_like_rdata_type`, not `rdtype`: `target`'s actual declared element type (set in
            # the `arg_ref_id`/`ssa_ref_id` prologue above) may be `Union{R,ZeroRData}`, and `cur`'s
            # declared type here must agree with that or the `icall` below could resolve to (and
            # statically `:invoke`) a method compiled for the too-narrow `R` alone.
            RT = zero_like_rdata_type(_optype(pir, exit_ret_node[b]))
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
            # prologue above; `ref`'s actual declared element type is whichever of the two this
            # produces, so reading it back out (and re-zeroing it) must agree exactly.
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
                # the caller-supplied `ty` (which describes `contrib`, computed from whatever type
                # the *contribution* happens to come from, e.g. a field type rather than `node`'s
                # own type). Deriving `cur`'s type independently here, rather than trusting `ty`,
                # keeps this correct regardless of what each call site passes for `ty` — a too-narrow
                # declared type here would let `icall` resolve to (and statically `:invoke`) a method
                # compiled for a type narrower than what the ref can actually hold.
                rty = zero_like_rdata_type(_widen(_optype(pir, node)))
                cur = emit!(Expr(:call, getf, target, 1), rty)
                new = emit!(icall(increment_g, (rty, ty), cur, contrib), rty)
                emit!(Expr(:call, setf, target, 1, new), Any)
            end
            nothing
        end

        # (a) Recover this visit's forwards-computed operand values, if this block has any. Keyed by
        # the same tagged `(kind, node)` items `_scan_block_comms` produced. `comms_type_id` records
        # each item's *static* type alongside its SSA value — needed by the `:subtape` recursion case
        # below, which must pass the inner tape's concrete `InnerTapeT` to
        # `reverse_pullback_recursive_ci`, not just the runtime SSA reference to its value.
        comms_val_id = Dict{Any,Any}()
        comms_type_id = Dict{Any,Any}()
        if !isempty(block_comms_types[b])
            ST = comms_stack_ty[b]
            if ST <: CommsCell
                # Unrolled single-slot read: `getfield` of `val` (field 1) — no `pop!`/`position`.
                popped = emit!(Expr(:call, getf, comms_obj_id[b], 1), Tuple{block_comms_types[b]...})
            else
                popped = emit!(icall(pop_g, (ST,), comms_obj_id[b]),
                               Tuple{block_comms_types[b]...})
            end
            for (j, nd) in enumerate(block_comms_nodes[b])
                comms_val_id[nd] = emit!(Expr(:call, getf, popped, j), block_comms_types[b][j])
                comms_type_id[nd] = block_comms_types[b][j]
            end
        end
        # --- Comms resolvers. Every pullback-side read of a forwards-recorded value goes through
        # one of these three, rather than indexing `comms_val_id` directly. They are the single
        # place that knows *how* a value can be obtained, which is what lets that set be extended
        # (rematerialization, argument values off the tape) without touching a single rule.
        #
        # `fetch_shadow` accepts either spelling of "this node's fdata handle": `:shadow_ref` (what
        # `memoryrefget`/`memoryrefset!` declare for a `MemoryRef`) and `:fshadow` (what
        # `getfield`/`setfield!`/a tracked `%new` declare for a struct object). They mean the same
        # thing and are resolved identically by `emit_epilogue!`; the two node populations are
        # disjoint, so trying both here is unambiguous and saves renaming ten declaration sites.
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
            base = emit!(Expr(:call, getf, arr, QuoteNode(:ref)), reft)
            # Shadow ref forces boundscheck `true`; primal ref keeps the primal's own.
            return emit!(Expr(:call, Base.memoryrefnew, base, idx, shadow ? true : bc), reft)
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
        # `nothing` is the point: before, an operand a rule read but the scan never recorded would
        # emit `nothing` into the IR and fail far away with no indication of where.
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
            elseif isa(s, Core.PiNode)
                acc = deref_and_zero!(ssa_ref_id[i], Ti)
                route!(s.val, acc, zero_like_rdata_type(_widen(_optype(pir, s.val))))
            elseif isa(s, Expr) && s.head === :new
                T = _calleeval(s.args[1], iworld)
                args = @view s.args[2:end]
                if T <: Array
                    # Nothing to route: both fields (`:ref` — a `MemoryRef`, `:size` — a structural
                    # tuple) always have `NoRData`, and no `(:fshadow, ...)` comms item exists for an
                    # array's own `%new` (see `_scan_block_comms`) — everything accumulated during the
                    # reverse walk already landed directly in the shadow `Array` the fwds pass built.
                    # Must be checked before `ismutabletype(T)` below, which is also true for `Array`
                    # but assumes a `MutableTangent` shadow that doesn't exist here.
                elseif ismutabletype(T)
                    # A mutable struct carries no rdata of its own (`rdtype(T) === NoRData`
                    # always) -- everything accumulated during the reverse walk landed directly in
                    # the shadow `MutableTangent` built by the fwds pass (via `getfield`/
                    # `setfield!`'s own rules, `increment_field_rdata!`). By now (the pullback
                    # walks statements in reverse, so this `%new`'s own turn runs *last*, after
                    # every use of the object) that accumulation is complete, so each field's
                    # gradient is handed back as this `%new` argument's own rdata -- mirroring
                    # Mooncake's `_mutable_new_pullback!!`.
                    FT = fdtype(T)
                    if FT !== NoFData
                        shadow = pb_fetch_shadow(Core.SSAValue(i))
                        for j in eachindex(args)
                            Fty = fieldtype(T, j)
                            RFty = rdtype(Fty)
                            RFty === NoRData && continue
                            TFty = tangent_type(Fty)
                            field_tangent = emit!(icall(_rr_get_tangent_field, (FT, Int), shadow, j), TFty)
                            contrib = emit!(icall(_rr_rdata, (TFty,), field_tangent), RFty)
                            route!(args[j], contrib, RFty)
                        end
                    end
                else
                    acc = deref_and_zero!(ssa_ref_id[i], Ti)
                    RDataT = rdtype(T)
                    if RDataT !== NoRData
                        # `acc`'s own declared type can include `ZeroRData` when `Ti` (this SSA's own
                        # inferred type, which can be broader than `T` -- e.g. this `%new` sits behind
                        # a later `Union` merge) isn't concrete enough on its own to produce a real
                        # zero. `T` itself is always concrete here (required by `:new`), so a real
                        # zero of type `RDataT` is always compile-time constructible regardless --
                        # materialize it before treating `acc` as the real `RDataT` `NamedTuple`
                        # wrapper below (a raw `getfield` on the literal `ZeroRData()` singleton,
                        # which has no fields, would otherwise throw).
                        real_acc = emit!(
                            icall(_rr_realize_rdata, (zero_like_rdata_type(_widen(Ti)), Type{RDataT}),
                                  acc, RDataT),
                            RDataT)
                        NT = fields_type(RDataT)
                        data_id = emit!(Expr(:call, getf, real_acc, 1), NT)
                        for j in eachindex(args)
                            Fty = rdtype(fieldtype(T, j))
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
                    # Non-differentiable results (comparisons -> Bool, integer ops, ...) have
                    # `NoRData`; skip entirely rather than asking `apply_intrinsic_rrule!` for a rule
                    # that doesn't exist and shouldn't — nothing flows backward through them.
                    if rdtype(Ti) !== NoRData
                        acc = deref_and_zero!(ssa_ref_id[i], Ti)
                        # Operands the forwards pass deliberately didn't record (see
                        # `intrinsic_rrule_operands`, consulted identically in `_scan_block_comms`)
                        # come through as `UnrecordedOperand`, and `opf` refuses to emit anything
                        # referencing one. So a rule whose declaration understates what it reads
                        # fails here, loudly and located, instead of silently emitting IR against a
                        # value that was never put on the tape.
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
                              # An operand's own *declared* primal type, read straight from the primal
                              # IR — not derivable from `pvals` (a resolved value, not a type) or `Ti`
                              # (the statement's own result type, not an operand's). Needed by
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
                           optype=(@nospecialize x) -> _optype(pir, x), ssa=Core.SSAValue(i), ref_for)
                    contribs = apply_builtin_rrule!(Val(f), actual, Ti, cctx)
                    if contribs !== nothing
                        for (a, c) in zip(actual, contribs)
                            c === nothing && continue
                            route!(a, c, zero_like_rdata_type(_widen(_optype(pir, a))))
                        end
                    elseif tangent_type(_widen(Ti)) === NoTangent
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
                    subtape_key = (:subtape, Core.SSAValue(i))
                    inner_tape_raw = comms_val_id[subtape_key]
                    InnerTapeT = comms_type_id[subtape_key]
                    # `zero_like_rdata_type`, not `rdtype`: `acc`'s actual type (see `deref_and_zero!`
                    # above) may include `ZeroRData` when `Ti` isn't concrete enough on its own, so the
                    # inner pullback must be resolved to accept exactly that (possibly wider) seed type
                    # — its own exit-route `increment!!` already tolerates `ZeroRData` generically.
                    SeedT = zero_like_rdata_type(_widen(Ti))
                    # Cyclic edge: `comms_type_id` for this item is the bare `Tape` UnionAll (what the
                    # fwds pass's `_scan_block_comms` declared it as — see `reverse_fwds_recursive_ci`).
                    # The popped value's declared type is therefore abstract; narrow it to this build's
                    # own concrete `TapeT` (a self-recursive callee's tape is, by construction, the same
                    # type as the caller's own) via an unchecked `PiNode` before invoking with it.
                    inner_tape = InnerTapeT === Tape ? emit!(Core.PiNode(inner_tape_raw, TapeT), TapeT) :
                                                        inner_tape_raw
                    pb_resolved = reverse_pullback_recursive_ci(interp, impl_mi, TapeT, codualparams,
                                                                 InnerTapeT, SeedT, edges, reason)
                    if pb_resolved === nothing
                        reason[] *= " — at %$i: `$(_stmt_str(s))`"
                        return nothing
                    end
                    pb_ci, pb_derived, InnerRdatasT = pb_resolved
                    # A derived inner pullback goes through its carrier (so `reverse_pullback_impl`
                    # is the callee and the tape an argument); a hand-written one *is* the callee.
                    # The hand-written case is flagged `IR_FLAG_NOINLINE`: inlining a rule author's
                    # body back in here would re-embed its `GlobalRef`s relative to *its* defining
                    # module (a bare `cos` in `src/rrules.jl` becomes `GlobalRef(Differ, :cos)`),
                    # which `verify_ir` rejects in value position. `_is_reverse_carrier_mi` cannot
                    # cover this case by type — a hand pullback has no common supertype.
                    pb_stmt = pb_derived ? Expr(:invoke, pb_ci, reverse_pullback_impl, inner_tape, acc) :
                                           Expr(:invoke, pb_ci, inner_tape, acc)
                    inner_rdatas = emit!(pb_stmt, InnerRdatasT,
                                         pb_derived ? CC.IR_FLAG_REFINED : CC.IR_FLAG_NOINLINE)
                    # Slot 1 is the callee's own rdata — guaranteed `NoRData` by
                    # `_static_recursible_call`'s callee guard, discarded exactly like `ref_for`
                    # already discards any literal operand's contribution — so routing starts at 2.
                    for (j, a) in enumerate(actual)
                        # `zero_like_rdata_type`: the callee's own `argtypes[j]`-th argument rdata
                        # (this same function, recursively, for the callee) can likewise be
                        # `ZeroRData` when that argument's type isn't concrete enough.
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
        # through that entry alone — `pred_is_unique_pred[b]` is forced `true` for exactly this case
        # (`_unique_predecessor_info`), so `_emit_switch!` below never pops for it either.
        haskey(regions, b) && (preds = [regions[b]])
        phi_acc = Any[]
        for i in lo:phi_end
            Ti = pstmts[i][:type]
            push!(phi_acc, deref_and_zero!(ssa_ref_id[i], Ti))
        end

        if b == 1
            # No predecessors: this is the pullback's own final block — the last thing that runs.
            # Restore every bulk-saved argument's primal contents here, which is what makes the
            # whole scheme equivalent to restoring each overwritten element as the sweep passes it:
            # nothing reads primal memory in between, so only the state at this boundary is visible.
            if !isempty(bulk_args)
                bufs_id = emit!(Expr(:call, getf, Core.Argument(2), 3), Vector{Any})
                for k in sort!(collect(bulk_args))
                    Pk = _widen(_codual_primal_type(codualparams[k]))
                    emit!(icall(_bulk_restore!, (Vector{Any}, Int, Pk),
                                bufs_id, bulk_slot[k], parg_pb[k]), Nothing)
                end
            end
            # Read out every argument's accumulated rdata and return them as a tuple.
            result_ids = Vector{Any}(undef, n)
            for k in 1:n
                Pk = _codual_primal_type(codualparams[k])
                result_ids[k] = emit!(Expr(:call, getf, arg_ref_id[k], 1), zero_like_rdata_type(Pk))
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
                    # `tgt`'s actual declared element type is `zero_like_rdata_type` of `v`'s own
                    # primal type (the edge value) — generally *not* the same as the phi node's own
                    # (merged, typically wider) type `Ti`. `phi_acc[j]` is `zero_like_rdata_type` of
                    # `Ti` instead, since that's what `deref_and_zero!` actually produced for the phi
                    # itself. Deriving these independently (rather than a single `RT` from `Ti` used
                    # for both, as before `ZeroRData` support) is what makes this correct on a loop
                    # back-edge, where the phi's accumulator is genuinely re-zeroed every visit.
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

    # `blocks` is currently ordered by ascending *primal* block number, which is not a valid
    # topological/dominance order for the *pullback's own* control flow — the pullback's control
    # flow runs in the opposite direction (starting at the exit routing blocks, ending at primal
    # block 1's finalization), so primal block 1 sits first in the vector despite being reached
    # last. `_sort_cfg_blocks!` reorders by BFS distance from the pullback's own entry, matching
    # Mooncake's own `pullback_ir` (whose docstring on `_sort_cfg_blocks!` notes the optimizer needs
    # this for reasons not fully understood — confirmed here: without it, `sroa_pass!`/`adce_pass!`
    # crash with an `UndefRefError` inside `is_union_phi` on any primal with a loop, because SROA's
    # own phi-insertion when scalar-replacing the rdata `Ref`s assumes this order). Safe to reorder
    # even though blocks still hold `Switch` terminators here (not yet lowered to `IDGotoIfNot`,
    # whose *implicit* fallthrough-to-next-block *would* make reordering unsound) —
    # `_cfg_lower_switch_statements` (called inside `lower_cfg_blocks_to_ir`, after this sort) is what
    # introduces those. `_remove_unreachable_cfg_blocks!` drops the throw-only-primal-block stubs:
    # nothing in the pullback ever pushes/routes to them, so they're genuinely unreachable here (the
    # forwards pass's `unreachable_block` treatment has no reverse analogue to be reachable *from*).
    blocks = _remove_unreachable_cfg_blocks!(_sort_cfg_blocks!(blocks))
    ir2 = lower_cfg_blocks_to_ir(blocks, pir; argtypes=Any[impl_mi.specTypes.parameters...], def=impl_mi)
    CC.verify_ir(ir2)
    return ir2
end

# Emits a plain goto (`skip_pop`, the only path a single-predecessor block ever takes under
# the per-edge formula — ISSUES #52) or `pop!(block_stack)` followed by a `Switch` comparing the
# popped id against each candidate (`preds[1:end-1]`), falling through to `preds[end]`.
#
# The `length(preds) == 1` branch below is now dead code: with `pred_is_unique_pred[b] =
# length(preds[b]) <= 1`, every single-predecessor block has `skip_pop == true` and returns early
# above. It's kept as a defensive no-op rather than removed, because the old per-block world needed
# it (a sole predecessor that pushed for *another* successor required a balance-pop here). Do not
# rely on it without confirming the per-edge formula is live.
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
# what calls `build_contextual_ir` (see the two-layer note near the carrier stubs). The
# carrier mirrors this signature exactly, so the body is a straight pass-through invoke — no
# reordering. `ctx` reaches the generated carrier as an ordinary argument, which is how a `Ctx`
# carrying a pre-allocated tape hands its stacks in.
function rrule_entry_body(world::UInt, source, self, fcd, ctx, argcds)
    argnames = Any[Symbol("#self#"), :fcd, :ctx, :argcds]
    impl_tt = Tuple{typeof(reverse_fwds_impl),fcd,ctx,argcds...}
    interp = ADInterpreter{Reverse}(; world)
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
    interp = ADInterpreter{Reverse}(; world)
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
IR). Pass it to [`rrule!!`](@ref) / [`value_and_gradient!`](@ref) / [`gradient!`](@ref).

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
    # `Base.to_tuple_type` moves the argument types into a *type parameter*, which is the only way
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
    interp = ADInterpreter{Reverse}(; world)
    codualtys = Any[fcodual_type(ftype)]
    for T in ArgsT.parameters
        (T isa Type) || return bail("Differ.build_ctx: argtypes must be a tuple of types")
        push!(codualtys, fcodual_type(T))
    end
    # Carrier layout is `reverse_fwds_impl(fcd, ctx, argcds...)`: fcd first, then the `Ctx{Nothing}`
    # (tape-allocating mode — this reads its return type only), then the argument coduals.
    impl_tt = Tuple{typeof(reverse_fwds_impl),codualtys[1],Ctx{Nothing},codualtys[2:end]...}
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    match === nothing && return bail("Differ.build_ctx: no reverse_fwds_impl match")
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    cinst = CC.typeinf_ext_toplevel(interp, impl_mi, CC.SOURCE_MODE_ABI)
    RT = cinst.rettype
    if !(RT isa DataType && RT <: Tuple && length(RT.parameters) == 2 &&
         RT.parameters[2] isa DataType && RT.parameters[2] <: Tape)
        return bail("Differ.build_ctx: could not derive a rule for this signature (it bailed; " *
                    "call `build_ctx(f, argtypes; prealloc=false)` and then differentiate to see why)")
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
    return :($TapeT($(Stack{Int32})(1), ($(slots...),), $(Vector{Any})()))
end

function refresh_build_tape()
    @eval function _build_tape(f, ArgsT)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, _build_tape_body))
    end
end
refresh_build_tape()

"""
    gradient(f, args...) -> (df, dx1, dx2, ...)

Reverse-mode gradient of `f(args...)` for scalar output. Allocates everything it needs (zero shadows
for each argument, and a tape); see [`gradient!`](@ref) for the pre-allocated form and
[`build_ctx`](@ref) to hold onto a reusable context.
"""
function gradient(f, args...)
    y, grads = value_and_gradient!(Ctx(), zero_fcodual(f), map(zero_fcodual, args)...)
    return grads
end

# ---------------------------------------------------------------------------
# Pre-allocated entry points.
#
# `gradient` above calls `zero_fcodual` on `f` and every argument, allocating a fresh shadow per call
# (for an `Array` argument: the shadow array itself). These variants instead take the caller's own
# `CoDual`s, so the shadow buffers are owned and reused by the caller. Pair them with a
# `build_ctx(...; prealloc=true)` context and a steady-state call allocates essentially nothing.
#
# The gradient w.r.t. an argument arrives one of two ways, and which one it is is a property of the
# argument's *type*, not of this API:
#
#   * fdata-carried (an `Array`, a mutable struct): accumulated *in place* into the shadow the
#     caller supplied, so `dx` holds the answer when the call returns (and is also returned, as the
#     same object — no copy).
#   * rdata-carried (a scalar): there is nothing to pre-allocate; it is returned by value.
#
# The supplied fdata is zeroed on entry (`set_to_zero!!`, allocation-free for arrays), so one of these
# calls means the same thing as the equivalent `gradient` call — the caller owns the buffer, not the
# accumulation history. Pass `zero_fcodual(f)` for the function slot unless differentiating w.r.t. a
# closure's captures.
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

See also [`gradient!`](@ref) and [`gradient`](@ref).
"""
function value_and_gradient!(ctx::AbstractCtx, fcd::CoDual, argcds::CoDual...)
    set_to_zero!!(tangent(fcd))
    map(cd -> set_to_zero!!(tangent(cd)), argcds)
    result_cd, pb = rrule!!(fcd, ctx, argcds...)
    y = primal(result_cd)
    all_cds = (fcd, argcds...)
    # A derived pullback can hand back `ZeroRData` for an argument whose concrete type has an
    # abstractly-typed field (or is itself non-concrete) — see the `zero_like_rdata_type` machinery
    # in `reverse_pullback_to_ircode`. This is the one place `gradient`/`gradient!` funnel through,
    # so instantiate a real zero here rather than ever handing a `ZeroRData` back to the user.
    rdatas = map((cd, r) -> r isa ZeroRData ? zero_rdata(primal(cd)) : r, all_cds, pb(one(y)))
    fdatas = (tangent(fcd), map(tangent, argcds)...)
    return y, map(tangent, fdatas, rdatas)
end

"""
    gradient!(ctx::AbstractCtx, fcd::CoDual, argcds::CoDual...) -> (df, dx1, dx2, ...)

Pre-allocated form of [`gradient`](@ref). See [`value_and_gradient!`](@ref) for the full description;
this drops the primal value.
"""
gradient!(ctx::AbstractCtx, fcd::CoDual, argcds::CoDual...) =
    value_and_gradient!(ctx, fcd, argcds...)[2]
