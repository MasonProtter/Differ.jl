# Reverse-mode AD: branches and (Phase C) loops, Mooncake style.
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
# Phase B scope: branches only (no back-edges/loops — Phase C lifts this) and exactly one
# reachable `return` in the primal (multiple early-return exits is a follow-up). Data-wise: the
# same as the straight-line PoC this supersedes — scalar float arithmetic intrinsics
# (`intrinsics_reverse.jl`, unchanged) and immutable, fully-initialised structs (`Core.getfield`/
# `Expr(:new,...)` via `increment_field!!`/`RData`). Still out of scope, bails cleanly: mutable
# structs/arrays, surviving high-level calls, try/catch.

_codual_primal_type(@nospecialize P) = fieldtype(P, 1)
_codual_fdata_type(@nospecialize P) = fieldtype(P, 2)

# ===========================================================================
# The tape: what the forwards carrier returns and the pullback carrier consumes.
# `ArgsTT` is a `Tuple` of the primal's `CoDual` argument types — carried purely so a
# `reverse_pullback_impl` specialization can recover which primal method it belongs to (mirroring
# how `reverse_fwds_impl`'s own `specTypes` names it directly). `CS` is a `Tuple` of per-primal-block
# comms-stack types (`Stack{T}` for a block with something to communicate, `SingletonStack{Tuple{}}`
# for one with nothing — mirroring Mooncake's `SharedDataPairs`/singleton-type optimization).
# ===========================================================================
struct Tape{ArgsTT<:Tuple,CS<:Tuple}
    block_stack::Stack{Int32}
    comms::CS
end

# ===========================================================================
# Carrier stubs.
# ===========================================================================

reverse_fwds_impl(codualargs::CoDual...) =
    error("ADNext.reverse_fwds_impl ran directly: ADInterpreter could not build the reverse " *
          "forwards pass (likely control flow ADNext doesn't support yet, a mutable/undef-field " *
          "struct, a surviving high-level call, or an intrinsic with no registered reverse rule).")

reverse_pullback_impl(tape, seed) =
    error("ADNext.reverse_pullback_impl ran directly: ADInterpreter could not build the reverse " *
          "pullback pass.")

is_reverse_fwds_impl(mi) = isa(mi.def, Method) && !isempty(mi.specTypes.parameters) &&
                          mi.specTypes.parameters[1] === typeof(reverse_fwds_impl)
is_reverse_pullback_impl(mi) = isa(mi.def, Method) && length(mi.specTypes.parameters) >= 2 &&
                              mi.specTypes.parameters[1] === typeof(reverse_pullback_impl)

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

# `reverse_fwds_impl(codualargs::CoDual...)` is fully-vararg after `#self#`, so its IR sees one
# packed tuple argument; `reverse_pullback_impl(tape, seed)` is ordinary (two flat slots).
function _impl_argtypes(mi::MethodInstance)
    params = mi.specTypes.parameters
    return (mi.def::Method).isva ? Any[params[1], Tuple{params[2:end]...}] : Any[params...]
end

# Mode hook — the only thing `contextual.jl` needs from reverse mode.
function build_contextual_ir(interp::ADInterpreter{Reverse}, mi::MethodInstance)
    if is_reverse_fwds_impl(mi)
        reason = Ref("ADNext could not build the reverse forwards pass (no specific reason recorded).")
        edges = Any[]
        ir = build_reverse_fwds_ir(interp, mi, reason, edges)
        interp.transformed_edges[mi] = edges
        ir === nothing && return reverse_error_ircode(mi, reason[])
        return ir
    elseif is_reverse_pullback_impl(mi)
        reason = Ref("ADNext could not build the reverse pullback pass (no specific reason recorded).")
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

function build_reverse_fwds_ir(interp::ADInterpreter, impl_mi::MethodInstance,
                               reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    codualparams = Any[impl_mi.specTypes.parameters[2:end]...]
    info = resolve_reverse_primal(interp, codualparams, reason, edges)
    info === nothing && return nothing
    primal_mi, n = info
    pir = _optimized_primal_ir(interp, primal_mi, reason, edges)
    pir === nothing && return nothing
    return reverse_fwds_to_ircode(interp, impl_mi, pir, n; reason, edges)
end

function build_reverse_pullback_ir(interp::ADInterpreter, impl_mi::MethodInstance,
                                   reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    params = impl_mi.specTypes.parameters
    TapeT = length(params) >= 2 ? params[2] : Any
    if !(TapeT isa Type && TapeT <: Tape)
        reason[] = "reverse_pullback_impl's tape argument is not a `Tape` (malformed specialization)"
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

# Phase B restriction: bail on any back-edge (a jump to a block at or before the jumping block's own
# number). Julia's reverse-postorder block numbering guarantees only genuine loop back-edges violate
# strictly-increasing jump targets — see the `adnext-extending-ir-support` methodology note this
# mirrors (get the real IR shape/invariants before designing against them).
function _has_backedge(pir)
    for (b, blk) in enumerate(pir.cfg.blocks)
        for s in blk.succs
            s <= b && return true
        end
    end
    return false
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

# For each primal block, the list of (node, type) pairs — genuine `SSAValue`/`Argument` operands of
# an intrinsic call *within that block* — that must be communicated from forwards to pullback (see
# this file's header). `getfield`/`%new` need no runtime value at all (their reverse rules route by
# static type + node identity only), so only intrinsic-call operands ever need comms.
function _scan_block_comms(pir, iworld, unreachable)
    nblocks = length(pir.cfg.blocks)
    nodes = [Any[] for _ in 1:nblocks]
    types = [Any[] for _ in 1:nblocks]
    bidx = 1
    for i in 1:length(pir.stmts)
        while bidx < nblocks && i > pir.cfg.blocks[bidx].stmts.stop
            bidx += 1
        end
        unreachable[bidx] && continue
        s = pir.stmts[i][:stmt]
        (isa(s, Expr) && (s.head === :call || s.head === :invoke)) || continue
        fpos = s.head === :invoke ? s.args[2] : s.args[1]
        actual = s.head === :invoke ? s.args[3:end] : s.args[2:end]
        isa(_calleeval(fpos, iworld), Core.IntrinsicFunction) || continue
        for a in actual
            (isa(a, Core.SSAValue) || isa(a, Core.Argument)) || continue
            any(==(a), nodes[bidx]) && continue
            push!(nodes[bidx], a)
            push!(types[bidx], _optype(pir, a))
        end
    end
    return nodes, types
end

rdtype(@nospecialize P) = rdata_type(tangent_type(P))

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
    if _has_backedge(pir)
        reason[] = "reverse-mode control flow support does not yet handle loops/back-edges " *
                   "(Phase C lifts this restriction)"
        return nothing
    end
    unreachable_block = _unreachable_blocks(pir)
    exit_blocks = _exit_blocks(pir, unreachable_block)
    if isempty(exit_blocks)
        reason[] = "primal has no reachable `return` (every path throws) — reverse mode cannot " *
                   "differentiate a function that never returns"
        return nothing
    end
    block_comms_nodes, block_comms_types = _scan_block_comms(pir, iworld, unreachable_block)

    getf = GlobalRef(Core, :getfield)
    ctuple = GlobalRef(Core, :tuple)
    push_g = Base.push!
    zerofcodual_g = zero_fcodual

    code = Any[]; types = Any[]
    emit!(ex, @nospecialize(ty)) = (push!(code, ex); push!(types, ty); Core.SSAValue(length(code)))
    opf(name, ty, args...) = emit!(Expr(:call, GlobalRef(Core.Intrinsics, name), args...), ty)

    codualparams = Any[impl_mi.specTypes.parameters[2:end]...]
    vararg_tt = Tuple{codualparams...}
    ArgsTT = Tuple{codualparams...}

    # --- Argument-unpacking prologue: primal only (fdata for the *result* is handled at the plain-
    # Julia `rrule` level for the top-level arguments; here we only need primal values to replay). ---
    parg = Vector{Any}(undef, n)
    for i in 1:n
        Ci = codualparams[i]
        Pi = _codual_primal_type(Ci)
        ci = emit!(Expr(:call, getf, Core.Argument(2), i), Ci)
        parg[i] = emit!(Expr(:call, getf, ci, 1), Pi)
    end

    # --- Tape prologue: allocate the block stack and one comms stack per block. ---
    block_stack_ssa = emit!(Expr(:call, Stack{Int32}), Stack{Int32})
    comms_stack_ssa = Vector{Any}(undef, nblocks)
    comms_stack_ty  = Vector{Any}(undef, nblocks)
    for b in 1:nblocks
        CommsT = Tuple{block_comms_types[b]...}
        ST = Base.issingletontype(CommsT) ? SingletonStack{CommsT} : Stack{CommsT}
        comms_stack_ty[b] = ST
        comms_stack_ssa[b] = emit!(Expr(:call, ST), ST)
    end

    primal_map = Vector{Any}(undef, N)
    presolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? primal_map[x.id] : isa(x, Core.Argument) ? parg[x.n] : x

    block_start_new = Vector{Int}(undef, nblocks)
    block_start_new[1] = 1
    bidx = 1

    emit_epilogue!(b) = begin
        nodes = block_comms_nodes[b]
        tup = emit!(Expr(:call, ctuple, (presolve(nd) for nd in nodes)...), Tuple{block_comms_types[b]...})
        emit!(Expr(:call, push_g, comms_stack_ssa[b], tup), Any)
        emit!(Expr(:call, push_g, block_stack_ssa, Int32(b)), Any)
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
            result_cd = emit!(Expr(:call, zerofcodual_g, ret_val), fcodual_type(R))
            comms_tuple = emit!(Expr(:call, ctuple, comms_stack_ssa...), Tuple{comms_stack_ty...})
            tape = emit!(Expr(:new, Tape{ArgsTT,Tuple{comms_stack_ty...}}, block_stack_ssa, comms_tuple),
                        Tape{ArgsTT,Tuple{comms_stack_ty...}})
            final = emit!(Expr(:call, ctuple, result_cd, tape), Tuple{fcodual_type(R),Tape{ArgsTT,Tuple{comms_stack_ty...}}})
            emit!(Core.ReturnNode(final), Any)
        elseif isa(s, Core.PiNode)
            primal_map[i] = presolve(s.val)
        elseif isa(s, Expr) && s.head === :new
            T = s.args[1]
            if !(T isa DataType) || ismutabletype(T) || !is_always_fully_initialised(T)
                reason[] = "reverse mode does not support mutable structs or structs with " *
                           "possibly-undef fields ($(T)) at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            args = @view s.args[2:end]
            primal_map[i] = emit!(Expr(:new, T, (presolve(a) for a in args)...), Ti)
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos = s.head === :invoke ? s.args[2] : s.args[1]
            actual = s.head === :invoke ? s.args[3:end] : s.args[2:end]
            f = _calleeval(fpos, iworld)
            if isa(f, Core.IntrinsicFunction)
                primal_map[i] = emit!(Expr(:call, f, (presolve(a) for a in actual)...), Ti)
            elseif f === Core.getfield
                primal_map[i] = emit!(Expr(:call, getf, presolve(actual[1]), actual[2]), Ti)
            else
                reason[] = "reverse mode does not support surviving calls to " *
                           "`$(f === nothing ? fpos : f)` (only intrinsics/`getfield`/`%new` are " *
                           "supported; no derived-rule recursion yet) at %$i: `$(_stmt_str(s))`"
                return nothing
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
                    reason[] = "primal has a PhiNode operand not yet defined in linear order at " *
                               "%$i (a back-edge should have been caught earlier)"
                    return nothing
                end
                pvals[j] = presolve(v)
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
    end
    if length(code) < block_start_new[nblocks]
        emit!(nothing, Nothing)
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
    argtypes = Any[impl_mi.specTypes.parameters[1], vararg_tt]
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

@inline __pop_blk_stack!(block_stack) = pop!(block_stack)::Int32
@inline __switch_case(id::Int32, prev::Int32) = !(id === prev)

function reverse_pullback_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int;
                                    reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    pstmts = pir.stmts
    N = length(pstmts)
    nblocks = length(pir.cfg.blocks)
    iworld = CC.get_inference_world(interp)

    if _has_backedge(pir)
        reason[] = "reverse-mode control flow support does not yet handle loops/back-edges " *
                   "(Phase C lifts this restriction)"
        return nothing
    end
    unreachable_block = _unreachable_blocks(pir)
    exit_blocks = _exit_blocks(pir, unreachable_block)
    if isempty(exit_blocks)
        reason[] = "primal has no reachable `return` (every path throws) — reverse mode cannot " *
                   "differentiate a function that never returns"
        return nothing
    end
    block_comms_nodes, block_comms_types = _scan_block_comms(pir, iworld, unreachable_block)

    params = impl_mi.specTypes.parameters
    TapeT = params[2]
    ArgsTT = TapeT.parameters[1]
    codualparams = Any[ArgsTT.parameters...]
    CS = TapeT.parameters[2]
    comms_stack_ty = Any[CS.parameters...]

    getf = GlobalRef(Core, :getfield)
    setf = GlobalRef(Core, :setfield!)
    ctuple = GlobalRef(Core, :tuple)
    pop_g = Base.pop!
    increment_g = increment!!
    incfield_g = increment_field!!

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
    # `Ref` to accumulate rdata into; literal/GlobalRef operands never do (no gradient to route to).
    needs_ref(i) = !unreachable_block[stmt_block[i]] &&
                   !isa(pstmts[i][:stmt], Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode})

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
            new = remit!(Expr(:call, increment_g, cur, seed_id), RT)
            remit!(Expr(:call, setf, target, 1, new), Any)
        end
        rid = ID()
        remit!(IDGotoNode(block_id[b]), Any)
        push!(exit_route_blocks, CFGBlock(rid, rstmts))
        push!(exit_route_ids, rid)
    end
    _emit_switch!(eemit!, block_stack_id, exit_blocks, exit_route_ids)

    blocks = vcat(CFGBlock(entry_id, entry_stmts), exit_route_blocks)

    # --- One reverse block per primal block. ---
    for b in 1:nblocks
        if unreachable_block[b]
            push!(blocks, CFGBlock(block_id[b], [ID()], [new_inst(nothing, Nothing)]))
            continue
        end

        stmts = IDInstPair[]
        emit!(ex, @nospecialize(ty)) = begin
            id = ID()
            push!(stmts, (id, new_inst(ex, ty)))
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
                new = emit!(Expr(:call, increment_g, cur, contrib), ty)
                emit!(Expr(:call, setf, target, 1, new), Any)
            end
            nothing
        end

        # (a) Recover this visit's forwards-computed operand values, if this block has any.
        comms_val_id = Dict{Any,Any}()
        if !isempty(block_comms_types[b])
            popped = emit!(Expr(:call, pop_g, comms_obj_id[b]), Tuple{block_comms_types[b]...})
            for (j, nd) in enumerate(block_comms_nodes[b])
                comms_val_id[nd] = emit!(Expr(:call, getf, popped, j), block_comms_types[b][j])
            end
        end
        pb_presolve(@nospecialize a) = haskey(comms_val_id, a) ? comms_val_id[a] : _calleeval(a, iworld)

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
                fpos = s.head === :invoke ? s.args[2] : s.args[1]
                actual = s.head === :invoke ? s.args[3:end] : s.args[2:end]
                f = _calleeval(fpos, iworld)
                if isa(f, Core.IntrinsicFunction)
                    # Non-differentiable results (comparisons -> Bool, integer ops, ...) have
                    # `NoRData`; skip entirely rather than asking `apply_intrinsic_rrule!` for a rule
                    # that doesn't exist and shouldn't — nothing flows backward through them.
                    if rdtype(Ti) !== NoRData
                        acc = deref_and_zero!(ssa_ref_id[i], Ti)
                        pvals = Tuple(pb_presolve(a) for a in actual)
                        ctx = (opf=(name, ty, args...) -> emit!(Expr(:call, GlobalRef(Core.Intrinsics, name), args...), ty),)
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
                        new = emit!(Expr(:call, incfield_g, cur, acc, Val(fieldidx)), RT)
                        emit!(Expr(:call, setf, target, 1, new), Any)
                    end
                else
                    reason[] = "reverse mode does not support surviving calls to " *
                               "`$(f === nothing ? fpos : f)` at %$i: `$(_stmt_str(s))`"
                    return nothing
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
            _emit_switch!(emit!, block_stack_id, preds, ID[block_id[p] for p in preds])
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
                    new = remit!(Expr(:call, increment_g, cur, phi_acc[j]), RT)
                    remit!(Expr(:call, setf, tgt, 1, new), Any)
                end
                rid = ID()
                remit!(IDGotoNode(block_id[p]), Any)
                push!(new_blocks, CFGBlock(rid, rstmts))
                push!(target_ids, rid)
            end
            _emit_switch!(emit!, block_stack_id, preds, target_ids)
            append!(blocks, new_blocks)
        end

        push!(blocks, CFGBlock(block_id[b], stmts))
    end

    # `_sort_cfg_blocks!` is unnecessary here: blocks were built and appended without any forward
    # references that later reordering would need to resolve, and the `ID`-addressed scheme never
    # relies on vector-position ordering — only `lower_cfg_blocks_to_ir` needs a linear order, which
    # the append order already provides. `_remove_unreachable_cfg_blocks!` is skipped for the same
    # reason `dualize_to_ircode` doesn't need it: every block here really is reachable (a stub for
    # a primal throw-only block is included deliberately, mirroring `unreachable_block`'s treatment
    # in the forwards pass, not left dangling).
    ir2 = lower_cfg_blocks_to_ir(blocks, pir; argtypes=Any[impl_mi.specTypes.parameters...], def=impl_mi)
    CC.verify_ir(ir2)
    return ir2
end

# Emits `pop!(block_stack)` (for stack balance) followed by a plain goto (a single predecessor) or a
# `Switch` comparing the popped predecessor id against each candidate (`preds[1:end-1]`), falling
# through to `preds[end]` — Phase B always pops, even for a single predecessor (Phase D's
# unique-predecessor optimization is the only thing allowed to skip it).
function _emit_switch!(emit!, block_stack_id, preds::Vector{Int}, targets::Vector{ID})
    prev_id = emit!(Expr(:call, __pop_blk_stack!, block_stack_id), Int32)
    if length(preds) == 1
        emit!(IDGotoNode(targets[1]), Any)
        return nothing
    end
    conds = ID[]
    for p in preds[1:(end - 1)]
        push!(conds, emit!(Expr(:call, __switch_case, Int32(p), prev_id), Bool))
    end
    emit!(Switch(Any[c for c in conds], targets[1:(end - 1)], targets[end]), Any)
    return nothing
end

# ===========================================================================
# Generated entry points + public API.
# ===========================================================================

function reverse_fwds_body(world::UInt, source, self, codual_argtypes)
    argnames = Any[Symbol("#self#"), :codualargs]
    impl_tt = Tuple{typeof(reverse_fwds_impl),codual_argtypes...}
    interp = ADInterpreter{Reverse}(; world)
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                                :(error("ADNext: no reverse_fwds_impl match")), true)
    end
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    cinst = CC.typeinf_ext_toplevel(interp, impl_mi, CC.SOURCE_MODE_ABI)
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(reverse_fwds_impl, $cinst, codualargs...)), true)
    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

function refresh_reverse_fwds_seeded()
    @eval function reverse_fwds_seeded(codualargs::CoDual...)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, reverse_fwds_body))
    end
end
refresh_reverse_fwds_seeded()

function reverse_pullback_body(world::UInt, source, self, tapetype, seedtype)
    argnames = Any[Symbol("#self#"), :tape, :seed]
    impl_tt = Tuple{typeof(reverse_pullback_impl),tapetype,seedtype}
    interp = ADInterpreter{Reverse}(; world)
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                                :(error("ADNext: no reverse_pullback_impl match")), false)
    end
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    cinst = CC.typeinf_ext_toplevel(interp, impl_mi, CC.SOURCE_MODE_ABI)
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(reverse_pullback_impl, $cinst, tape, seed)), false)
    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

function refresh_reverse_pullback_seeded()
    @eval function reverse_pullback_seeded(tape, seed)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, reverse_pullback_body))
    end
end
refresh_reverse_pullback_seeded()

"""
    rrule(f, args...) -> (y, pullback)

Reverse-mode AD over `f(args...)`: `y` is the primal value, `pullback(seed)` returns a tuple of
tangents (one per `f, args...`) given the rdata seed `seed` for `y` (e.g. `1.0` for a scalar
output). Branches are supported (see this file's header for the two-carrier/tape design); loops are
Phase C.
"""
function rrule(f, args...)
    fcd = zero_fcodual(f)
    argcds = map(zero_fcodual, args)
    result_cd, tape = reverse_fwds_seeded(fcd, argcds...)
    y = primal(result_cd)
    function pullback(seed)
        rdatas = reverse_pullback_seeded(tape, seed)
        fdatas = (tangent(fcd), map(tangent, argcds)...)
        return map(tangent, fdatas, rdatas)
    end
    return y, pullback
end

"""
    gradient(f, args...) -> (df, dx1, dx2, ...)

Convenience wrapper around [`rrule`](@ref) for scalar output: seeds the pullback with `one(y)`.
"""
function gradient(f, args...)
    y, pb = rrule(f, args...)
    return pb(one(y))
end
