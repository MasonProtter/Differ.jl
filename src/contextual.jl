using Core: MethodMatch, MethodInstance, CodeInstance, Compiler
using Core.Compiler:
    _methods_by_ftype,
    InferenceParams,
    get_world_counter,
    tls_world_age,
    InferenceResult,
    typeinf,
    InternalMethodTable,
    InferenceState,
    NativeInterpreter,
    AbstractInterpreter,
    OptimizationParams,
    MethodTableView,
    OverlayMethodTable,
    WorldRange,
    finish,
    InferenceResult,
    OptimizationState,
    OptimizationParams,
    optimize,
    store_backedges,
    finish!,
    CodeInfo,
    isexpr,
    argextype,
    singleton_type,
    Builtin,
    GlobalRef,
    CodeInstance,
    ir_to_codeinf!,
    typeinf_ext_toplevel,
    IRCode,
    argextype,
    widenconst

const CC = Core.Compiler

using Base:
    specialize_method,
    isexpr

struct MyCtx end

struct ContextualInterpreter <: AbstractInterpreter
    inf_cache::Vector{InferenceResult}
    world::UInt
    inf_params::InferenceParams
    opt_params::OptimizationParams
    codegen_cache::IdDict{CodeInstance, CodeInfo}
    function ContextualInterpreter(world::UInt,
                                   ip::InferenceParams,
                                   op::OptimizationParams)
        @assert world <= Base.get_world_counter()
        return new(
            InferenceResult[],
            world,
            ip,
            op,
            IdDict{CodeInstance, CodeInfo}()
        )
    end
end
function ContextualInterpreter(;world=Base.get_world_counter(),
                               inf_params=InferenceParams(),
                               opt_params=OptimizationParams()) 
    ContextualInterpreter(world, inf_params, opt_params)
end

Core.Compiler.InferenceParams(interp::ContextualInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::ContextualInterpreter) = interp.opt_params
Core.Compiler.get_inference_world(interp::ContextualInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::ContextualInterpreter) = interp.inf_cache
Core.Compiler.cache_owner(interp::ContextualInterpreter) = MyCtx()
Core.Compiler.codegen_cache(interp::ContextualInterpreter) = interp.codegen_cache


@noinline function Core.OptimizedGenerics.CompilerPlugins.typeinf(::MyCtx, mi::MethodInstance, source_mode::UInt8)
    # Base.invoke_in_world(which(Core.OptimizedGenerics.CompilerPlugins.typeinf, Tuple{ContextOwner, MethodInstance, UInt8}).primary_world,
    Compiler.typeinf_ext_toplevel(ContextualInterpreter(; world=Base.tls_world_age()),
                                  mi, source_mode)
end

@noinline function Core.OptimizedGenerics.CompilerPlugins.typeinf_edge(::MyCtx, mi::MethodInstance, parent_frame::Compiler.InferenceState, world::UInt, source_mode::UInt8)
    interp = ContextualInterpreter(; world)
    Compiler.typeinf_edge(interp, mi.def, mi.specTypes, Core.svec(),
                          parent_frame, false, false)
end


function expr_to_codeinfo(m::Module, argnames, spnames, sp, e::Expr, isva::Bool=false)
    lam = Expr(:lambda, argnames,
               Expr(Symbol("scope-block"),
                    Expr(:block,
                         Expr(:return,
                              Expr(:block,
                                   e,
                                   )))))
    ex = if spnames === nothing || isempty(spnames)
        lam
    else
        Expr(Symbol("with-static-parameters"), lam, spnames...)
    end
    ci = Base.generated_body_to_codeinfo(ex, @__MODULE__(), isva)
    @assert ci isa Core.CodeInfo "Failed to create a CodeInfo from the given expression. This might mean it contains a closure or comprehension?\n Offending expression: $e"
    ci
end


function with_ctx_interpreter(f, args...)
    cinst = generated_ci_in_absint(f, args)
    invoke(f, cinst, args...)
end


function generated_ci_in_absint_body(world::UInt, lnn, this, f, args)
    sig = Type{Tuple{f, args.parameters...}}
    sig isa Type{<:Type{<:Tuple}} || error()
    tt = sig.parameters[1]
    interp = ContextualInterpreter(; world)

    match, valid_worlds = Core.Compiler.findsup(tt, Core.Compiler.method_table(interp))
    if match === nothing
        error(lazy"Unable to find matching $tt")
    end
    mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    
    cinst = Core.OptimizedGenerics.CompilerPlugins.typeinf(MyCtx(), mi, Compiler.SOURCE_MODE_ABI)
    
    ci = expr_to_codeinfo(@__MODULE__(), [Symbol("#self#"), :f, :args], [], (), :(return $cinst))
    
    matches = Base._methods_by_ftype(sig, -1, world)
    if !isnothing(matches)
        ci.edges = Core.MethodInstance[]
        for match in Base._methods_by_ftype(sig, -1, world)
            mi = Base.specialize_method(match) 
            push!(ci.edges, mi)
        end
    end
    return ci
end

function refresh_generated_ci_in_absint()
    @eval function generated_ci_in_absint(f, args)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, generated_ci_in_absint_body))
    end
end
refresh_generated_ci_in_absint()


# ---------------------------------------------------------------------------
# The dualization pass, injected at the interp-dispatched InferenceState seam.
#
# When ContextualInterpreter is asked to build an InferenceState for a
# `dualized_impl` MethodInstance (whose specTypes is the *dual* signature), we
# discard the stub source and instead hand inference the *dualized* body derived
# from the corresponding primal method. Inference then derives the correct `Dual`
# types itself, and optimization (inlining control, post-inlining transforms)
# runs normally under this interpreter.
# ---------------------------------------------------------------------------

is_dualized_impl(mi) = isa(mi.def, Method) && !isempty(mi.specTypes.parameters) &&
                       mi.specTypes.parameters[1] === typeof(dualized_impl)

# Resolve the primal MethodInstance and dual arity for a `dualized_impl` specialization.
function primal_of_impl(interp::ContextualInterpreter, impl_mi::MethodInstance)
    dualparams = impl_mi.specTypes.parameters[2:end]
    all(P -> P isa Type && P <: Dual, dualparams) || return nothing
    primal_tt = Base.to_tuple_type(Any[primal_type(P) for P in dualparams])
    pmatch, _ = CC.findsup(primal_tt, CC.method_table(interp))
    pmatch === nothing && return nothing
    isa(pmatch.method, Method) || return nothing
    pmatch.method.isva && return nothing
    primal_mi = specialize_method(pmatch.method, pmatch.spec_types, pmatch.sparams)::MethodInstance
    return (primal_mi, length(dualparams))
end

# Tier 2: build the dualized *IRCode* from the primal's optimized IRCode. Returns
# `(ir::IRCode, rettype)` or `nothing` (control flow / unsupported).
function tier2_dual_ircode(interp::ContextualInterpreter, impl_mi::MethodInstance)
    info = primal_of_impl(interp, impl_mi)
    info === nothing && return nothing
    primal_mi, n = info
    world = CC.get_inference_world(interp)
    # Optimized primal IR via the internal `typeinf_ircode` (reflection `code_ircode` is blocked
    # inside generators). NativeInterpreter so `sin`/`cos` survive as `:invoke`s for Tier 1.
    pir, _ = CC.typeinf_ircode(CC.NativeInterpreter(world), primal_mi, nothing)
    pir === nothing && return nothing
    return dualize_to_ircode(impl_mi, pir, n)
end

# ── Tier 1 (control-flow fallback) via the InferenceState seam ─────────────────────────────
# Tier 2 is preferred and installed in `optimize` below. When it can't apply (control flow), we
# inject a dualized *lowered* CodeInfo here so inference derives the `Dual` return type and
# `optimize` runs normally.
function Core.Compiler.InferenceState(result::CC.InferenceResult, cache_mode::UInt8,
                                      interp::ContextualInterpreter)
    mi = result.linfo
    if is_dualized_impl(mi) && tier2_dual_ircode(interp, mi) === nothing   # Tier 2 not applicable
        src = tier1_dualized_source(interp, mi)
        src !== nothing && return CC.InferenceState(result, src, cache_mode, interp)
    end
    return @invoke CC.InferenceState(result::CC.InferenceResult, cache_mode::UInt8,
                                     interp::CC.AbstractInterpreter)
end

function tier1_dualized_source(interp::ContextualInterpreter, impl_mi::MethodInstance)
    info = primal_of_impl(interp, impl_mi)
    info === nothing && return nothing
    primal_mi, n = info
    psrc = CC.retrieve_code_info(primal_mi, CC.get_inference_world(interp))
    isa(psrc, CodeInfo) || return nothing
    src = copy(psrc)
    dualize!(primal_mi, src, n) || return nothing
    errs = CC.validate_code(impl_mi, src)
    isempty(errs) || (foreach(Core.println, errs); return nothing)
    return src
end

# ── Tier 2 install: replace the optimized IR with the dualized IRCode and *determine* the
# CodeInstance return type (finishinfer! froze it before optimize; jl_update_codeinst won't
# revise it, so we re-fill it from the dual IR's return type).
function Core.Compiler.optimize(interp::ContextualInterpreter, opt::CC.OptimizationState,
                                caller::CC.InferenceResult)
    if is_dualized_impl(caller.linfo)
        res = tier2_dual_ircode(interp, caller.linfo)
        if res !== nothing
            dual_ir, rettype = res
            CC.ipo_dataflow_analysis!(interp, opt, dual_ir, caller)
            CC.finishopt!(interp, opt, dual_ir)
            if isdefined(caller, :ci)
                ci = caller.ci
                mw, Mw = first(caller.valid_worlds), last(caller.valid_worlds)
                ccall(:jl_fill_codeinst, Cvoid,
                      (Any, Any, Any, Any, Int32, UInt, UInt, UInt32, Any, Any, Any),
                      ci, rettype, Any, nothing, Int32(0), mw, Mw,
                      CC.encode_effects(caller.ipo_effects), nothing, nothing, Core.svec())
            end
            return nothing
        end
    end
    return @invoke CC.optimize(interp::CC.AbstractInterpreter, opt::CC.OptimizationState,
                               caller::CC.InferenceResult)
end

# Dualization: rewrite each `f(a...)` into `frule(Dual(f, struct_zero(f)), <dualified a>...)`,
# unpacking `dualargs`. Supports straight-line code, local-variable assignments, and control
# flow (branches/loops); every value flowing through the body is a `Dual`. `GotoIfNot`
# conditions are reduced to a primal `Bool` via `getfield(cond, 1)`. `n` = number of primal
# argument slots (including #self#) = number of dual args. Returns `false` (bails) on
# constructs not yet handled: exception handling, `:new`/`:foreigncall`, reassigned arguments.
function dualize!(primal_mi::MethodInstance, src::CodeInfo, n::Int)
    old = src.code
    nold = length(old)
    for s in old                                        # exception handling not yet supported
        if isa(s, Core.EnterNode) ||
           (isa(s, Expr) && s.head in (:enter, :leave, :pop_exception, :the_exception))
            return false
        end
    end

    fruleg = GlobalRef(@__MODULE__(), :frule)
    dualg  = GlobalRef(@__MODULE__(), :Dual)
    zerog  = GlobalRef(@__MODULE__(), :struct_zero)
    getf   = GlobalRef(Core, :getfield)

    new = Any[]
    for i in 1:n                                        # primal SlotNumber(i) -> SSAValue(i)
        push!(new, Expr(:call, getf, Core.SlotNumber(2), i))
    end
    ssamap = zeros(Int, nold)                           # orig stmt -> new index of its value
    stmt_start = zeros(Int, nold)                       # orig stmt -> new index of its first stmt
    produces_dual = falses(nold)
    fixups = Tuple{Int,Symbol}[]                        # (new index, :goto|:ifnot) to patch

    remapslot(id) = id <= n ? Core.SSAValue(id) : Core.SlotNumber(id - n + 2)
    remap(x) =
        isa(x, Core.SSAValue)   ? Core.SSAValue(ssamap[x.id]) :
        isa(x, Core.SlotNumber) ? remapslot(x.id) :
        isa(x, Core.Argument)   ? remapslot(x.n) : x
    # Every argument slot (unpacked to a Dual) and every local (only ever assigned dualified
    # values) holds a Dual; an SSA value is a Dual iff its defining statement produced one.
    is_dual(x) =
        isa(x, Core.SSAValue)   ? produces_dual[x.id] :
        isa(x, Core.SlotNumber) ? true :
        isa(x, Core.Argument)   ? true : false
    function dualify(x)
        rx = remap(x)
        is_dual(x) && return rx
        push!(new, Expr(:call, zerog, rx)); z = Core.SSAValue(length(new))
        push!(new, Expr(:call, dualg, rx, z)); return Core.SSAValue(length(new))
    end
    function emit_call!(ce::Expr)                       # -> SSAValue of the frule result (a Dual)
        callee = dualify(ce.args[1])
        dargs = Any[dualify(ce.args[k]) for k in 2:length(ce.args)]
        push!(new, Expr(:call, fruleg, callee, dargs...))
        return Core.SSAValue(length(new))
    end

    for (j, s) in enumerate(old)
        stmt_start[j] = length(new) + 1
        if isa(s, Expr) && s.head === :call
            ssamap[j] = emit_call!(s).id; produces_dual[j] = true
        elseif isa(s, Expr) && s.head === :(=)
            lhs = s.args[1]
            (isa(lhs, Core.SlotNumber) && lhs.id > n) || return false   # only local-slot assignment
            rhs = s.args[2]
            v = (isa(rhs, Expr) && rhs.head === :call) ? emit_call!(rhs) : dualify(rhs)
            push!(new, Expr(:(=), remapslot(lhs.id), v))
            ssamap[j] = length(new); produces_dual[j] = true
        elseif isa(s, Core.ReturnNode)
            push!(new, isdefined(s, :val) ? Core.ReturnNode(dualify(s.val)) : s)
            ssamap[j] = length(new)
        elseif isa(s, Core.GotoNode)
            push!(new, s); ssamap[j] = length(new)                      # label patched below
            push!(fixups, (length(new), :goto))
        elseif isa(s, Core.GotoIfNot)
            c = s.cond
            cond = is_dual(c) ?
                (push!(new, Expr(:call, getf, remap(c), 1)); Core.SSAValue(length(new))) : remap(c)
            push!(new, Core.GotoIfNot(cond, s.dest)); ssamap[j] = length(new)   # dest patched below
            push!(fixups, (length(new), :ifnot))
        elseif isa(s, Core.NewvarNode)
            push!(new, Core.NewvarNode(remapslot(s.slot.id))); ssamap[j] = length(new)
        elseif isa(s, GlobalRef) || isa(s, QuoteNode) || isa(s, Core.SSAValue) ||
               isa(s, Core.SlotNumber) || isa(s, Core.Argument) || !isa(s, Expr)
            push!(new, remap(s)); ssamap[j] = length(new)
            produces_dual[j] = is_dual(s)
        elseif isa(s, Expr) && s.head in (:boundscheck, :meta, :inbounds, :loopinfo,
                                          :gc_preserve_begin, :gc_preserve_end)
            push!(new, s); ssamap[j] = length(new)                      # non-value / passthrough
        else
            return false                                               # unsupported (e.g. :new, :foreigncall)
        end
    end

    for (pos, kind) in fixups                           # patch jump targets to new indices
        st = new[pos]
        new[pos] = kind === :goto ? Core.GotoNode(stmt_start[st.label]) :
                                    Core.GotoIfNot(st.cond, stmt_start[st.dest])
    end

    src.code = new
    src.slotnames = Symbol[Symbol("#self#"), :dualargs, src.slotnames[n+1:end]...]
    src.slotflags = UInt8[0x00, 0x00, src.slotflags[n+1:end]...]
    src.nargs = 2
    src.isva = true
    src.ssavaluetypes = length(new)
    src.ssaflags = fill(UInt32(0), length(new))
    src.debuginfo = Core.DebugInfo(:dualized)
    return true
end


