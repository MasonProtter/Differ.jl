# Reverse-mode AD: branches and loops, Mooncake style.
#
# Two separately-compiled carriers, wired through the same `build_contextual_ir` seam as forward
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
# recursion via ordinary method-table dispatch, mirroring `frules.jl`), and read-only array indexing
# via a provenance chain traceable to a function argument (see `_array_shadow_tracked`). Still out of
# scope, bails cleanly: mutable structs, array mutation, dynamic-dispatch recursion, try/catch (see
# the control-flow plan's Phase E).

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
struct Tape{ArgsTT<:Tuple,CS<:Tuple}
    block_stack::Stack{Int32}
    comms::CS
end

# ===========================================================================
# Two layers, as in forward mode (`frule!!` entry / `dualized_impl` carrier):
#
#   * The *entry* is the public surface — the `@generated` fallback method of `rrule!!` and the
#     `@generated` pullback callable `(t::Tape)(seed)`. Both compile the corresponding carrier under
#     `ADInterpreter{Reverse}` (which is what runs the `build_contextual_ir` seam at all — ordinary
#     code compiles under a `NativeInterpreter`, where the seam never fires) and emit a static
#     `:invoke` to the result. The fwds entry is a straight pass-through: it invokes the carrier with
#     exactly its own `(fcd, ctx, argcds...)`, because the carrier mirrors `rrule!!`'s signature.
#   * The *carrier* is the hidden function whose specializations that seam actually transforms.
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
is_reverse_fwds_impl(mi) = isa(mi.def, Method) && length(mi.specTypes.parameters) >= 3 &&
                          mi.specTypes.parameters[1] === typeof(reverse_fwds_impl) &&
                          mi.specTypes.parameters[2] <: CoDual &&
                          mi.specTypes.parameters[3] <: AbstractCtx
is_reverse_pullback_impl(mi) = isa(mi.def, Method) && length(mi.specTypes.parameters) >= 2 &&
                              mi.specTypes.parameters[1] === typeof(reverse_pullback_impl) &&
                              mi.specTypes.parameters[2] <: Tape

# The `@generated` derived fallback — the least-specific `rrule!!` method (see the note above the
# type definitions). Recognized by its exact signature so `hand_reverse_rule_match` can tell "matched
# a hand rule" from "matched the fallback": a `findsup` on a concrete query always resolves *some*
# method now that the fallback exists.
is_generated_reverse_fwds_fallback(m::Method) =
    m.sig === Tuple{typeof(rrule!!),CoDual,AbstractCtx,Vararg{CoDual}}

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
_is_reverse_carrier_mi(mi::MethodInstance) = isa(mi.def, Method) && !isempty(mi.specTypes.parameters) &&
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
# `finishinfer!`/`optimize` seam as a real reverse-mode body (mirrors `error_ircode`,
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
        ir === nothing && return reverse_error_ircode(mi, reason[])
        return ir
    elseif is_reverse_pullback_impl(mi)
        reason = Ref("Differ could not build the reverse pullback pass (no specific reason recorded).")
        edges = Any[]
        ir = build_reverse_pullback_ir(interp, mi, reason, edges)
        interp.transformed_edges[mi] = edges
        ir === nothing && return reverse_error_ircode(mi, reason[])
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
    return pir
end

# Recursion cycle guard (see `interp.in_progress`, `contextual.jl`): a genuinely cyclic primal has
# no finite `Tape` type in this design (Part 1's recursion support nests the inner call's `Tape` type
# as a literal type parameter inside the outer block's comms-tuple type — see `reverse_fwds_recursive_ci`
# below), so a self- or mutually-recursive primal must bail cleanly here rather than recursing forever
# building ever-deeper `CodeInstance`s. Keying by the carrier `impl_mi` is sufficient: a genuine cycle
# (direct self-recursion, or A→B→A mutual recursion) always re-encounters the exact same `impl_mi`
# still on the stack, since recursive resolution always reuses *this same* `interp` instance (see
# `reverse_fwds_recursive_ci`/`reverse_pullback_recursive_ci`) rather than crossing through a fresh
# interpreter the way forward mode's `frule!!` `@generated`-function boundary does (see
# `DUALIZED_IMPL_IN_PROGRESS`, `forward_interp.jl`, for that cross-instance variant of this same guard).
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
        return reverse_fwds_to_ircode(interp, impl_mi, pir, n; reason, edges)
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
        return reverse_pullback_to_ircode(interp, impl_mi, pir, n; reason, edges)
    finally
        delete!(interp.in_progress, impl_mi)
    end
end

# The optimized IR for a carrier: exactly what the `optimize` seam installs. Used both by
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
function _unique_predecessor_info(pir, exit_blocks::Vector{Int})
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

    pred_is_unique_pred = falses(nblocks)
    for b in 1:nblocks
        pred_is_unique_pred[b] = length(preds[b]) == 1 && is_unique_pred[only(preds[b])]
    end
    # The entry block has no real predecessor (only ever entered one way: the function is called).
    pred_is_unique_pred[1] = isempty(preds[1])

    return is_unique_pred, pred_is_unique_pred
end

# Part 2 (read-only array indexing) static provenance analysis: which SSA statements' values are a
# `MemoryRef` traceable, through *exactly* the chain Julia 1.13 lowers `x[i]` to, back to a function
# argument whose fdata is non-trivial (i.e. an `Array` argument — confirmed by direct inspection of
# `Base.code_ircode` on `x[1]`/a hand-written summation loop, per the `adnext-extending-ir-support`
# methodology: `Base.getfield(x, :ref)::MemoryRef{T}` then `Base.memoryrefnew(ref, i, false)`, with an
# optional `PiNode` alias in between). Bounds the feature precisely: an array reachable any other way
# (nested in a struct field, returned from a call, locally `Vector{T}(undef,...)`-allocated) is
# untracked — and untracked-but-differentiable is a real bail at the point of use (`_scan_block_comms`
# /the fwds-pass builtin dispatch below), never silently mishandled.
#
# Which top-level (fwds-carrier) arguments carry non-trivial fdata (an `Array`, today — mutable
# structs are handled separately since this engine doesn't support them at all yet). Factored out of
# `_array_shadow_tracked` so `_static_recursible_call`'s Part 3 array-argument-recursion guard can
# check a bare `Core.Argument` operand directly without recomputing this.
function _arg_fdata_tracked(n::Int, codualparams::Vector{Any})
    arg_tracked = falses(n)
    for k in 1:n
        arg_tracked[k] = fdtype(_codual_primal_type(codualparams[k])) !== NoFData
    end
    return arg_tracked
end

# Returns a `BitVector` of length `length(pir.stmts)`, indexed by SSA id (`tracked[i]`). Argument
# provenance (`Core.Argument(k)`) is checked inline via `arg_tracked` rather than returned, since
# every caller that needs it (`_scan_block_comms`'s `memoryrefget` case, the fwds-pass builtin
# dispatch) only ever looks the chain up starting from an `SSAValue` (the `memoryrefnew`/`PiNode`
# result the `MemoryRef` handle actually flows through), never a bare `Argument` directly. Part 3
# (array-argument recursion, `_static_recursible_call`) additionally needs `arg_tracked` itself, for
# a call whose argument is a bare `Core.Argument`, so it's exposed via `_arg_fdata_tracked` above
# rather than recomputed.
function _array_shadow_tracked(pir, iworld, n::Int, codualparams::Vector{Any})
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
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos, actual = _call_parts(s)
            f = _calleeval(fpos, iworld)
            if f === Core.getfield && length(actual) >= 2 && provenance_tracked(actual[1])
                fk = actual[2]
                fname = isa(fk, QuoteNode) ? fk.value : fk
                if fname === :ref && (pir.stmts[i][:type] <: MemoryRef)
                    tracked[i] = true
                end
            elseif f === Base.memoryrefnew && !isempty(actual) && provenance_tracked(actual[1])
                tracked[i] = true
            end
        end
    end
    return tracked
end

rdtype(@nospecialize P) = rdata_type(tangent_type(P))
fdtype(@nospecialize P) = fdata_type(tangent_type(P))

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
#     carrier (`arg_tracked`/`array_tracked`, Part 3). This is the load-bearing correctness guard,
#     not just a missing-feature guard: passing a recursive call's argument a freshly-zeroed fdata
#     with no link to the real shadow would be silently wrong, not just unsupported — so any
#     non-trivial fdata that isn't a *real, traceable* array shadow must still bail (a mutable
#     struct's fdata, or an array reachable any other way — nested in a struct field, returned from
#     a call, freshly allocated — is untracked and stays unsupported, mirroring
#     `_array_shadow_tracked`'s own read-indexing scope).
#  3. The call's own result type must likewise carry trivial fdata — array-*valued results* from a
#     recursive call are a separate, not-yet-supported feature (the fwds pass has nowhere to route a
#     result shadow today).
function _static_recursible_call(pir, iworld, i::Int, s::Expr, reason::Ref{String},
                                 arg_tracked::BitVector, array_tracked::BitVector)
    fpos, actual = _call_parts(s)
    fval = _calleeval(fpos, iworld)
    if fval === nothing
        reason[] = "dynamic (non-statically-resolvable) callee not supported yet at %$i: " *
                   "`$(_stmt_str(s))`"
        return nothing
    end
    ftype = _typeof(fval)
    if !(ftype isa DataType && isconcretetype(ftype))
        reason[] = "callee type $(ftype) is not a concrete DataType at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    if tangent_type(ftype) !== NoTangent
        reason[] = "recursive calls into a callee with differentiable captures ($(ftype)) are not " *
                   "supported yet at %$i: `$(_stmt_str(s))`"
        return nothing
    end
    argtypes = Any[_optype(pir, a) for a in actual]
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
            # Part 3: an array argument is allowed through — but only if its *identity* is
            # statically traceable back to a tracked function argument, mirroring
            # `_array_shadow_tracked`'s own provenance scope, so the emission side (below) always has
            # a real shadow value (`sresolve`) to thread through the recursive `:invoke` rather than
            # a detached `NoFData()`.
            if !(fdata_type(tangent_type(P)) <: Array)
                reason[] = "recursive call with a mutable-struct argument ($(P)) is not supported " *
                           "yet at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            a = actual[j]
            tracked_here = isa(a, Core.Argument) ? (a.n <= length(arg_tracked) && arg_tracked[a.n]) :
                           isa(a, Core.SSAValue) ? array_tracked[a.id] : false
            if !tracked_here
                reason[] = "recursive call with an array argument ($(P)) whose provenance is not " *
                           "traceable to a function argument is not supported yet at %$i: " *
                           "`$(_stmt_str(s))`"
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
# `reverse_fwds_impl` specializations only get transformed via the `build_contextual_ir` seam when
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
function reverse_fwds_recursive_ci(interp, @nospecialize(ftype), argtypes::Vector{Any},
                                   edges::Vector{Any}, reason::Ref{String})
    argcodualtys = Any[fcodual_type(P) for P in argtypes]
    hand = hand_reverse_rule_match(interp, ftype, argtypes)
    if hand !== nothing
        tt, fm = hand
        callee_val = rrule!!
    else
        tt = Tuple{typeof(reverse_fwds_impl),CoDual{ftype,NoFData},Ctx{Nothing},argcodualtys...}
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
        reason[] = "recursion into the callee's own reverse-mode forwards pass failed " *
                  "(it bailed on something inside its own body)"
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
function reverse_pullback_recursive_ci(interp, @nospecialize(InnerTapeT), @nospecialize(InnerSeedT),
                                       edges::Vector{Any}, reason::Ref{String})
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
        reason[] = "recursion into the callee's own reverse-mode pullback pass failed " *
                  "(it bailed on something inside its own body)"
        return nothing
    end
    CC.add_invoke_edge!(edges, tt, ci)
    return ci, derived
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
function _scan_block_comms(interp, pir, iworld, unreachable, codualparams::Vector{Any},
                           reason::Ref{String}, edges::Vector{Any})
    nblocks = length(pir.cfg.blocks)
    nodes = [Any[] for _ in 1:nblocks]
    types = [Any[] for _ in 1:nblocks]
    n = length(codualparams)
    array_tracked = _array_shadow_tracked(pir, iworld, n, codualparams)
    arg_tracked = _arg_fdata_tracked(n, codualparams)
    bidx = 1
    for i in 1:length(pir.stmts)
        while bidx < nblocks && i > pir.cfg.blocks[bidx].stmts.stop
            bidx += 1
        end
        unreachable[bidx] && continue
        s = pir.stmts[i][:stmt]
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
                any(nd -> nd == (:primal, a), nodes[bidx]) && continue
                push!(nodes[bidx], (:primal, a))
                push!(types[bidx], _optype(pir, a))
            end
        elseif f === Core.getfield
            continue   # routes by static type + node identity only — no runtime value needed
        elseif f === Base.memoryrefget
            # A read whose result actually carries a gradient needs its shadow `MemoryRef` handle
            # communicated forward — but only if that handle has provenance traceable all the way
            # back to a tracked argument (see `_array_shadow_tracked`); otherwise this is a real bail
            # (silently dropping the contribution would be wrong, not just unsupported).
            Ti = pir.stmts[i][:type]
            if rdtype(Ti) !== NoRData
                ref_node = actual[1]
                if !(isa(ref_node, Core.SSAValue) && array_tracked[ref_node.id])
                    reason[] = "array read has no differentiable provenance traceable to a function " *
                               "argument at %$i: `$(_stmt_str(s))`"
                    return nothing
                end
                any(nd -> nd == (:shadow_ref, ref_node), nodes[bidx]) && continue
                push!(nodes[bidx], (:shadow_ref, ref_node))
                push!(types[bidx], _optype(pir, ref_node))
            end
        elseif isa(f, Core.Builtin)
            continue   # `memoryrefnew` needs no comms entry; any other builtin bails at point-of-use
        else
            # A surviving high-level call: attempt Part 1's static recursion. Not qualifying is *not*
            # an error at scan time (mirrors how an unregistered intrinsic isn't flagged here either
            # — only the main per-statement loop, reaching it for real, turns that into a bail); only
            # a genuine attempted-and-failed resolution propagates as a real bail here.
            info = _static_recursible_call(pir, iworld, i, s, Ref(""), arg_tracked, array_tracked)
            info === nothing && continue
            _, ftype, argtypes = info
            resolved = reverse_fwds_recursive_ci(interp, ftype, argtypes, edges, reason)
            resolved === nothing && return nothing
            InnerTapeT = resolved[4]
            push!(nodes[bidx], (:subtape, Core.SSAValue(i)))
            push!(types[bidx], InnerTapeT)
        end
    end
    return nodes, types
end

# ===========================================================================
# Forwards pass: 1:1 block-topology-preserving replay of the primal (built the same way
# `dualize_to_ircode` is, minus any shadow/tangent — see this file's header), instrumented with the
# block-stack/comms pushes described above.
# ===========================================================================
function reverse_fwds_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int;
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
    is_unique_pred, _ = _unique_predecessor_info(pir, exit_blocks)

    # Carrier is `reverse_fwds_impl(fcd, ctx, argcds...)`: `params[2]` is `fcd`, `params[3]` is the
    # `ctx`, `params[4:end]` are the argument coduals. `codualparams`/`ArgsTT` are the *full* codual
    # list `(fcd, args...)` — the shape the rest of this builder and the pullback side both expect.
    # `vararg_tt` is only the *argument* coduals: it types the packed vararg slot `Argument(4)`.
    codualparams = Any[impl_mi.specTypes.parameters[2], impl_mi.specTypes.parameters[4:end]...]
    vararg_tt = Tuple{impl_mi.specTypes.parameters[4:end]...}
    ArgsTT = Tuple{codualparams...}

    scan = _scan_block_comms(interp, pir, iworld, unreachable_block, codualparams, reason, edges)
    scan === nothing && return nothing
    block_comms_nodes, block_comms_types = scan
    array_tracked = _array_shadow_tracked(pir, iworld, n, codualparams)
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
    for i in 1:n
        Ci = codualparams[i]
        Pi = _codual_primal_type(Ci)
        Fi = _codual_fdata_type(Ci)
        # codual 1 is `fcd` — `Argument(2)` directly; coduals 2..n are the argument coduals, packed
        # in the vararg tuple `Argument(4)` at position `i-1`. `Argument(3)` is the `ctx`.
        ci = i == 1 ? Core.Argument(2) : emit!(Expr(:call, getf, Core.Argument(4), i - 1), Ci)
        parg[i] = emit!(Expr(:call, getf, ci, 1), Pi)
        Fi !== NoFData && (farg[i] = emit!(Expr(:call, getf, ci, 2), Fi))
    end

    # --- Tape prologue. Two shapes, chosen by the `ctx` type in `Argument(3)`. ---
    comms_stack_ty = Vector{Any}(undef, nblocks)
    for b in 1:nblocks
        CommsT = Tuple{block_comms_types[b]...}
        comms_stack_ty[b] = Base.issingletontype(CommsT) ? SingletonStack{CommsT} : Stack{CommsT}
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
            # A `SingletonStack` has no fields and stores nothing — construct it with `%new` rather
            # than paying a call to reach a constructor that does nothing.
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
            comms_stack_ty[b] <: SingletonStack && continue   # no fields, nothing to reset
            emit!(Expr(:call, setf, comms_stack_ssa[b], 2, 0), Any)
        end
    end

    primal_map = Vector{Any}(undef, N)
    shadow_map = Vector{Any}(undef, N)   # Part 2: array shadow (MemoryRef) chain, sparse — only
                                          # `array_tracked[i]` entries are ever assigned or read.
    presolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? primal_map[x.id] : isa(x, Core.Argument) ? parg[x.n] : x
    sresolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? shadow_map[x.id] : isa(x, Core.Argument) ? farg[x.n] : x

    # Part 1: which block's own recursive-call `:subtape` comms item already has its inner `Tape`
    # SSA value computed, so `emit_epilogue!` can find it (sparse, parallel to `primal_map`).
    inner_tape_map = Dict{Int,Any}()

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
        # Nothing to communicate: the stack is a `SingletonStack` whose `push!` is a no-op, so the
        # tuple construction and the push are both pure overhead. (The pullback already skips the
        # matching `pop!` for exactly this case.)
        if !isempty(nodes)
            vals = (nd[1] === :primal ? presolve(nd[2]) :
                    nd[1] === :subtape ? inner_tape_map[nd[2].id] : sresolve(nd[2]) for nd in nodes)
            CommsT = Tuple{block_comms_types[b]...}
            tup = emit!(Expr(:call, ctuple, vals...), CommsT)
            icall!(push_g, Any, (comms_stack_ty[b], CommsT), comms_stack_ssa[b], tup)
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
        if is_terminator && !unreachable_block[bidx]
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
            R = _optype(pir, s.val)
            result_cd = icall!(zerofcodual_g, fcodual_type(R), (R,), ret_val)
            # Pre-allocated mode returns the caller's own tape object; otherwise `%new` one around
            # the stacks the prologue just built.
            tape = tape_ssa
            if tape === nothing
                comms_tuple = emit!(Expr(:call, ctuple, comms_stack_ssa...), Tuple{comms_stack_ty...})
                tape = emit!(Expr(:new, TapeT, block_stack_ssa, comms_tuple), TapeT)
            end
            final = emit!(Expr(:call, ctuple, result_cd, tape), Tuple{fcodual_type(R),TapeT})
            emit!(Core.ReturnNode(final), Any)
        elseif isa(s, Core.PiNode)
            primal_map[i] = presolve(s.val)
            array_tracked[i] && (shadow_map[i] = sresolve(s.val))
        elseif isa(s, Expr) && s.head === :new
            T = s.args[1]
            if !(T isa DataType) || ismutabletype(T) || !is_always_fully_initialised(T)
                reason[] = "reverse mode does not support mutable structs or structs with " *
                           "possibly-undef fields ($(T)) at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            args = @view s.args[2:end]
            primal_map[i] = emit!(Expr(:new, T, (presolve(a) for a in args)...), Ti)
        elseif isa(s, Expr) && s.head === :boundscheck
            primal_map[i] = emit!(Expr(:boundscheck, s.args...), Ti)
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos, actual = _call_parts(s)
            f = _calleeval(fpos, iworld)
            if isa(f, Core.IntrinsicFunction)
                primal_map[i] = emit!(Expr(:call, f, (presolve(a) for a in actual)...), Ti)
            elseif f === Core.getfield
                primal_map[i] = emit!(Expr(:call, getf, presolve(actual[1]), actual[2]), Ti)
                # Part 2: the shadow `.ref`/element-of-tracked-chain mirrors the primal access
                # exactly (see `_array_shadow_tracked`) — only ever true for a `MemoryRef`-producing
                # `getfield(x, :ref)` off an already-tracked `x`.
                array_tracked[i] && (shadow_map[i] = emit!(Expr(:call, getf, sresolve(actual[1]), actual[2]), Ti))
            elseif f === Base.memoryrefnew
                primal_map[i] = emit!(Expr(:call, f, (presolve(a) for a in actual)...), Ti)
                if array_tracked[i]
                    shadow_args = Any[sresolve(actual[1])]
                    for a in actual[2:end]
                        push!(shadow_args, presolve(a))
                    end
                    shadow_map[i] = emit!(Expr(:call, f, shadow_args...), Ti)
                end
            elseif f === Base.memoryrefget
                # Primal replay only: the shadow *value* is never read on the fwds pass, only the
                # shadow `MemoryRef` *handle* (a mutable/pointer-like object) is communicated forward
                # for the pullback to mutate later — registered in `_scan_block_comms`, not here.
                primal_map[i] = emit!(Expr(:call, f, (presolve(a) for a in actual)...), Ti)
            elseif f === Base.memoryrefset!
                reason[] = "reverse mode does not support array mutation (`memoryrefset!`) at " *
                           "%$i: `$(_stmt_str(s))`"
                return nothing
            elseif isa(f, Core.Builtin) && rdtype(Ti) === NoRData
                # A non-differentiable builtin result (`===`, `isa`, comparisons, ... — e.g. the
                # `Base.iterate`-state check a `for i in 1:length(x)` loop's own lowering embeds) has
                # no gradient to route, so no rule/comms is needed — just replay it primally, the same
                # treatment `GlobalRef`/literal statements already get. Distinct from
                # `memoryrefget`/`memoryrefnew` above only in that there is no shadow chain to extend.
                primal_map[i] = emit!(Expr(:call, f, (presolve(a) for a in actual)...), Ti)
            elseif isa(f, Core.Builtin)
                reason[] = "reverse mode does not support builtin `$(f)` with a differentiable " *
                           "result ($(Ti)) and no reverse rule at %$i: `$(_stmt_str(s))`"
                return nothing
            else
                info = _static_recursible_call(pir, iworld, i, s, reason, arg_tracked, array_tracked)
                info === nothing && return nothing
                fval, ftype, argtypes = info
                resolved = reverse_fwds_recursive_ci(interp, ftype, argtypes, edges, reason)
                resolved === nothing && return nothing
                ci, callee_val, InnerFCoDualT, InnerTapeT = resolved
                FCT = CoDual{ftype,NoFData}
                fcodual = emit!(Expr(:new, FCT, fval, NoFData()), FCT)
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
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[])
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

function reverse_pullback_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int;
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
    _, pred_is_unique_pred = _unique_predecessor_info(pir, exit_blocks)

    params = impl_mi.specTypes.parameters
    TapeT = params[2]
    PullbackSeedT = params[3]
    ArgsTT = TapeT.parameters[1]
    codualparams = Any[ArgsTT.parameters...]
    CS = TapeT.parameters[2]
    comms_stack_ty = Any[CS.parameters...]

    scan = _scan_block_comms(interp, pir, iworld, unreachable_block, codualparams, reason, edges)
    scan === nothing && return nothing
    block_comms_nodes, block_comms_types = scan
    array_tracked = _array_shadow_tracked(pir, iworld, n, codualparams)
    arg_tracked = _arg_fdata_tracked(n, codualparams)

    getf = GlobalRef(Core, :getfield)
    setf = GlobalRef(Core, :setfield!)
    ctuple = GlobalRef(Core, :tuple)
    pop_g = Base.pop!
    increment_g = increment!!
    incfield_g = increment_field!!

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
    stmt_block = Vector{Int}(undef, N)
    let bidx = 1
        for i in 1:N
            while bidx < nblocks && i > pir.cfg.blocks[bidx].stmts.stop
                bidx += 1
            end
            stmt_block[i] = bidx
        end
    end

    # Every statement except a pure control marker (or one living in a throw-only block) gets a
    # `Ref` to accumulate rdata into; literal/GlobalRef/`:boundscheck` operands never do (no
    # gradient to route to — `:boundscheck`'s rdata is always `NoRData`, so skipping it here just
    # avoids a useless allocation, it isn't load-bearing).
    needs_ref(i) = !unreachable_block[stmt_block[i]] &&
                   !isa(pstmts[i][:stmt], Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode}) &&
                   !(isa(pstmts[i][:stmt], Expr) && (pstmts[i][:stmt]::Expr).head === :boundscheck)

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

    arg_ref_id = Vector{Any}(undef, n)
    for k in 1:n
        Pk = _codual_primal_type(codualparams[k])
        RT = rdtype(Pk)
        arg_ref_id[k] = eemit!(Expr(:new, Base.RefValue{RT}, zero_rdata_from_type(Pk)), Base.RefValue{RT})
    end

    ssa_ref_id = Vector{Any}(undef, N)
    for i in 1:N
        needs_ref(i) || continue
        Ti = pstmts[i][:type]
        RT = rdtype(Ti)
        ssa_ref_id[i] = eemit!(Expr(:new, Base.RefValue{RT}, zero_rdata_from_type(Ti)), Base.RefValue{RT})
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
            RT = rdtype(_optype(pir, exit_ret_node[b]))
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
        deref_and_zero!(ref, @nospecialize(Pi)) = begin
            RT = rdtype(Pi)
            cur = emit!(Expr(:call, getf, ref, 1), RT)
            emit!(Expr(:call, setf, ref, 1, zero_rdata_from_type(Pi)), Any)
            cur
        end
        route!(@nospecialize(node), contrib, @nospecialize(ty)) = begin
            target = ref_for(node)
            if target !== nothing
                cur = emit!(Expr(:call, getf, target, 1), ty)
                new = emit!(icall(increment_g, (ty, ty), cur, contrib), ty)
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
            popped = emit!(icall(pop_g, (comms_stack_ty[b],), comms_obj_id[b]),
                           Tuple{block_comms_types[b]...})
            for (j, nd) in enumerate(block_comms_nodes[b])
                comms_val_id[nd] = emit!(Expr(:call, getf, popped, j), block_comms_types[b][j])
                comms_type_id[nd] = block_comms_types[b][j]
            end
        end
        # `pb_presolve` only ever looks up `:primal` comms items (intrinsic operands) — `:subtape`/
        # `:shadow_ref` items are looked up directly by their own tag where needed, below.
        pb_presolve(@nospecialize a) =
            haskey(comms_val_id, (:primal, a)) ? comms_val_id[(:primal, a)] : _calleeval(a, iworld)

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
            elseif isa(s, Core.PiNode)
                acc = deref_and_zero!(ssa_ref_id[i], Ti)
                route!(s.val, acc, rdtype(_optype(pir, s.val)))
            elseif isa(s, Expr) && s.head === :new
                acc = deref_and_zero!(ssa_ref_id[i], Ti)
                T = s.args[1]
                args = @view s.args[2:end]
                RDataT = rdtype(T)
                if RDataT !== NoRData
                    NT = fields_type(RDataT)
                    data_id = emit!(Expr(:call, getf, acc, 1), NT)
                    for j in eachindex(args)
                        Fty = rdtype(fieldtype(T, j))
                        Fty === NoRData && continue
                        contrib = emit!(Expr(:call, getf, data_id, j), Fty)
                        route!(args[j], contrib, Fty)
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
                               end,)
                        contribs = apply_intrinsic_rrule!(Val(f), pvals, acc, Ti, ctx)
                        if contribs === nothing
                            reason[] = "no reverse rule for intrinsic `$(nameof(f))` at %$i: " *
                                       "`$(_stmt_str(s))` (no rule registered; add one in " *
                                       "src/intrinsics_reverse.jl via `apply_intrinsic_rrule!`)"
                            return nothing
                        end
                        for (a, c) in zip(actual, contribs)
                            route!(a, c, rdtype(_optype(pir, a)))
                        end
                    end
                elseif f === Core.getfield
                    # Check triviality *first* (the key simplification over the naive version): this
                    # automatically covers `getfield(x,:ref)`/`getfield(x,:size)`/`getfield(sizetuple,
                    # j)` on an `Array` — all confirmed `NoRData` — with no array-specific code needed
                    # here at all. Only a genuinely differentiable field access falls through to the
                    # existing mutable-struct-bail + `increment_field!!` logic below.
                    if rdtype(Ti) !== NoRData
                        acc = deref_and_zero!(ssa_ref_id[i], Ti)
                        obj = actual[1]
                        StructP = _optype(pir, obj)
                        if !(StructP isa DataType) || ismutabletype(StructP)
                            reason[] = "reverse mode does not support `getfield` on a mutable struct " *
                                       "($(StructP)) at %$i: `$(_stmt_str(s))` (needs fdata-based " *
                                       "in-place accumulation, not implemented)"
                            return nothing
                        end
                        target = ref_for(obj)
                        if target !== nothing
                            fk = actual[2]
                            fieldidx = isa(fk, QuoteNode) ? findfirst(==(fk.value), fieldnames(StructP)) : fk
                            RT = rdtype(StructP)
                            cur = emit!(Expr(:call, getf, target, 1), RT)
                            new = emit!(icall(incfield_g, (RT, rdtype(Ti), Val{fieldidx}),
                                              cur, acc, Val(fieldidx)), RT)
                            emit!(Expr(:call, setf, target, 1, new), Any)
                        end
                    end
                elseif f === Base.memoryrefget
                    if rdtype(Ti) !== NoRData
                        acc = deref_and_zero!(ssa_ref_id[i], Ti)
                        shadow_ref = comms_val_id[(:shadow_ref, actual[1])]
                        RT = rdtype(Ti)
                        # `MemoryRef`s are mutable/pointer-like: no `Ref`/`ssa_ref_id` accumulator is
                        # needed for the array itself — its whole tangent lives in fdata, read-
                        # incremented-written in place directly through the shadow handle threaded
                        # forward via the `:shadow_ref` comms item, exactly the way an `Array`'s
                        # tangent is a plain `Array` incremented in place, never routed through rdata.
                        cur = emit!(Expr(:call, Base.memoryrefget, shadow_ref, QuoteNode(:not_atomic), false), RT)
                        new = emit!(icall(increment_g, (RT, RT), cur, acc), RT)
                        emit!(Expr(:call, Base.memoryrefset!, shadow_ref, new, QuoteNode(:not_atomic), false), RT)
                    end
                elseif f === Base.memoryrefnew
                    # Its own rdata is always `NoRData` (a `MemoryRef` handle, not a differentiable
                    # value) — nothing to route.
                elseif f === Base.memoryrefset!
                    reason[] = "reverse mode does not support array mutation (`memoryrefset!`) at " *
                               "%$i: `$(_stmt_str(s))`"
                    return nothing
                elseif isa(f, Core.Builtin) && rdtype(Ti) === NoRData
                    # Mirrors the fwds pass's own treatment: a non-differentiable builtin result has
                    # nothing to route backward, so this is a genuine no-op here (unlike the fwds
                    # pass, the pullback never needs to *replay* the statement at all).
                elseif isa(f, Core.Builtin)
                    reason[] = "reverse mode does not support builtin `$(f)` with a differentiable " *
                               "result ($(Ti)) and no reverse rule at %$i: `$(_stmt_str(s))`"
                    return nothing
                else
                    info = _static_recursible_call(pir, iworld, i, s, reason, arg_tracked, array_tracked)
                    info === nothing && return nothing
                    _, ftype, argtypes = info
                    acc = deref_and_zero!(ssa_ref_id[i], Ti)   # this call's own seed for the inner pullback
                    subtape_key = (:subtape, Core.SSAValue(i))
                    inner_tape = comms_val_id[subtape_key]
                    InnerTapeT = comms_type_id[subtape_key]
                    SeedT = rdtype(Ti)
                    pb_resolved = reverse_pullback_recursive_ci(interp, InnerTapeT, SeedT, edges, reason)
                    pb_resolved === nothing && return nothing
                    pb_ci, pb_derived = pb_resolved
                    InnerRdatasT = pb_ci.rettype
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
                        Fty = rdtype(argtypes[j])
                        Fty === NoRData && continue
                        contrib = emit!(Expr(:call, getf, inner_rdatas, j + 1), Fty)
                        route!(a, contrib, Fty)
                    end
                end
            end
        end

        # (c) Leading PhiNodes: dereference+zero each accumulated rdata, then route per-predecessor.
        preds = filter(!=(0), pir.cfg.blocks[b].preds)
        phi_acc = Any[]
        for i in lo:phi_end
            Ti = pstmts[i][:type]
            push!(phi_acc, deref_and_zero!(ssa_ref_id[i], Ti))
        end

        if b == 1
            # No predecessors: this is the pullback's own final block. Read out every argument's
            # accumulated rdata and return them as a tuple.
            result_ids = Vector{Any}(undef, n)
            for k in 1:n
                Pk = _codual_primal_type(codualparams[k])
                result_ids[k] = emit!(Expr(:call, getf, arg_ref_id[k], 1), rdtype(Pk))
            end
            res = emit!(Expr(:call, ctuple, result_ids...), Tuple{(rdtype(_codual_primal_type(c)) for c in codualparams)...})
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
                    RT = rdtype(Ti)
                    cur = remit!(Expr(:call, getf, tgt, 1), RT)
                    new = remit!(icall(increment_g, (RT, RT), cur, phi_acc[j]), RT)
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

# Emits a plain goto (`skip_pop`, or a lone predecessor with nothing pushed for it to consume — see
# `_unique_predecessor_info`) or `pop!(block_stack)` followed by either a plain goto (a single
# candidate — the pop is still needed here purely for stack balance, since *some* predecessor pushed
# unconditionally) or a `Switch` comparing the popped id against each candidate (`preds[1:end-1]`),
# falling through to `preds[end]`.
function _emit_switch!(emit!, icall, block_stack_id, preds::Vector{Int}, targets::Vector{ID};
                       skip_pop::Bool=false)
    if skip_pop
        @assert length(targets) == 1 "skip_pop requires an unambiguous (single) target"
        emit!(IDGotoNode(targets[1]), Any)
        return nothing
    end
    prev_id = emit!(icall(__pop_blk_stack!, (Stack{Int32},), block_stack_id), Int32)
    if length(preds) == 1
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
# what runs the `build_contextual_ir` seam (see the two-layer note near the carrier stubs). The
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
    slots = Any[:($S()) for S in CS.parameters]
    return :($TapeT($(Stack{Int32})(), ($(slots...),)))
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
    rdatas = pb(one(y))
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
