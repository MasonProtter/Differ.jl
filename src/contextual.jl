using Core: MethodInstance, CodeInstance, CodeInfo, Compiler
using Core.Compiler:
    AbstractInterpreter, InferenceParams, OptimizationParams, InferenceResult, IRCode
using Base: specialize_method

const CC = Core.Compiler

# ---------------------------------------------------------------------------
# ContextualInterpreter: a custom AbstractInterpreter that compiles the *dualized*
# version of a primal method. The dualization is a single source-to-source transform on the
# primal's post-optimization `IRCode` (see `ir_dualize.jl`), installed into the normal typeinf
# pipeline so it produces an ordinary `invoke`-able `CodeInstance`. Modeled on the plugin shape in
# `julia/Compiler/extras/CompilerDevTools`.
# ---------------------------------------------------------------------------

struct MyCtx
    world::UInt
end
MyCtx(;world) = MyCtx(world)

struct ContextualInterpreter <: AbstractInterpreter
    inf_cache::Vector{InferenceResult}
    world::UInt
    inf_params::InferenceParams
    opt_params::OptimizationParams
    codegen_cache::IdDict{CodeInstance, CodeInfo}
    # Per-compile scratch: the dualized IRCode built in `finishinfer!` (which also supplies the
    # return type) and installed as the optimization result in `optimize`. Keyed by the
    # `dualized_impl` MethodInstance being compiled. Safe because one interpreter instance serves
    # both hooks of a given frame.
    dual_ir::IdDict{MethodInstance, IRCode}
    function ContextualInterpreter(world::UInt,
                                   ip::InferenceParams,
                                   op::OptimizationParams)
        @assert world <= Base.get_world_counter()
        return new(
            InferenceResult[],
            world,
            ip,
            op,
            IdDict{CodeInstance, CodeInfo}(),
            IdDict{MethodInstance, IRCode}(),
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
Core.Compiler.cache_owner(interp::ContextualInterpreter) = MyCtx(interp.world)
Core.Compiler.codegen_cache(interp::ContextualInterpreter) = interp.codegen_cache


@noinline function Core.OptimizedGenerics.CompilerPlugins.typeinf(ctx::MyCtx, mi::MethodInstance, source_mode::UInt8)
    Compiler.typeinf_ext_toplevel(ContextualInterpreter(; world=ctx.world),
                                  mi, source_mode)
end

@noinline function Core.OptimizedGenerics.CompilerPlugins.typeinf_edge(ctx::MyCtx, mi::MethodInstance, parent_frame::Compiler.InferenceState, world::UInt, source_mode::UInt8)
    interp = ContextualInterpreter(; world)
    Compiler.typeinf_edge(interp, mi.def, mi.specTypes, Core.svec(),
                          parent_frame, false, false)
end


# Build a lowered `CodeInfo` from an expression. Used by `frule_body` (`forward_interp.jl`) to
# produce the trivial generated body that `invoke`s the compiled dual `CodeInstance`.
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


# ---------------------------------------------------------------------------
# The dualization seam.
#
# The generated `frule` fallback (`forward_interp.jl`) asks this interpreter to compile a
# `dualized_impl` MethodInstance whose `specTypes` is the *dual* signature. We compile it by
# transforming the corresponding primal method's post-optimization `IRCode` into a dualized
# `IRCode` (`build_dual_ir`), and splice that transform into the pipeline at two points:
#
#   * `finishinfer!` — the pipeline function that *supplies* the `CodeInstance` return type. We
#     build the dual IR here and set `me.bestguess` to its return type, so the ordinary
#     `jl_fill_codeinst` writes the correct dual type once (no post-hoc patching).
#   * `optimize` — installs the already-built dual IR as the optimization result via the ordinary
#     `ipo_dataflow_analysis!` + `finishopt!`.
#
# When the primal IR contains constructs the transform can't handle (control flow, unsupported
# builtins, …), `build_dual_ir` returns `nothing`: we leave `me.bestguess`/the cache untouched, the
# stub's throwing body flows through the pipeline normally, and the compiled CI raises the
# `dualized_impl` `ErrorException` when invoked — a graceful bail.
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

# Build the dualized `IRCode` for a `dualized_impl` specialization from the primal's optimized
# `IRCode`. Returns the dual `IRCode` or `nothing` (unsupported IR → bail).
function build_dual_ir(interp::ContextualInterpreter, impl_mi::MethodInstance)
    info = primal_of_impl(interp, impl_mi)
    info === nothing && return nothing
    primal_mi, n = info
    world = CC.get_inference_world(interp)
    # Optimized primal IR via the internal `typeinf_ircode`. A NativeInterpreter is used so that
    # `sin`/`cos` and other hand-ruled functions survive as `:invoke`s (routed through `frule`).
    pir, _ = CC.typeinf_ircode(CC.NativeInterpreter(world), primal_mi, nothing)
    pir === nothing && return nothing
    return dualize_to_ircode(interp, impl_mi, pir, n)
end

# Resolve and compile the `frule(Dual{typeof(f),NoFData}, dualargs...)` rule for a surviving
# high-level call to an *invoke-able `CodeInstance`*, so the dualized IR can emit a static
# `:invoke` (mirroring how the primal IR keeps `sin`/`cos` as `:invoke`s to a `CodeInstance`).
# `:invoke` targets *must* be `CodeInstance`s: `collectinvokes!` only JITs those, so a bare
# `MethodInstance` would fall back to a boxed dynamic call. Returns `nothing` if unresolved.
function frule_codeinstance(interp::ContextualInterpreter, @nospecialize(ftype), dual_argtypes)
    frule_tt = Tuple{typeof(frule), Dual{ftype,NoFData}, dual_argtypes...}
    fm, _ = CC.findsup(frule_tt, CC.method_table(interp))
    fm === nothing && return nothing
    isa(fm.method, Method) || return nothing
    frule_mi = specialize_method(fm.method, fm.spec_types, fm.sparams)::MethodInstance
    world = CC.get_inference_world(interp)
    return CC.typeinf_ext_toplevel(CC.NativeInterpreter(world), frule_mi, CC.SOURCE_MODE_ABI)::CodeInstance
end

# Return-type seam: for a `dualized_impl` MI, build the dual IR and set `me.bestguess` to its
# return type so the generic `finishinfer!` freezes the correct dual type into the CodeInstance.
function CC.finishinfer!(me::CC.InferenceState, interp::ContextualInterpreter, cycleid::Int,
                         opt_cache::IdDict{MethodInstance, CodeInstance})
    mi = me.linfo
    if is_dualized_impl(mi)
        ir = build_dual_ir(interp, mi)
        if ir !== nothing
            interp.dual_ir[mi] = ir
            me.bestguess = CC.compute_ir_rettype(ir)
        end
    end
    return @invoke CC.finishinfer!(me::CC.InferenceState, interp::CC.AbstractInterpreter,
                                   cycleid::Int, opt_cache::IdDict{MethodInstance, CodeInstance})
end

# Install seam: replace the optimization result with the dual IR built in `finishinfer!`, then run
# the ordinary IPO-safe optimization passes on it (inlining, SROA, ADCE, …) so the `Dual` /
# `struct_zero` / `getfield` / `frule` calls are inlined and the immutable duals scalar-replaced.
function CC.optimize(interp::ContextualInterpreter, opt::CC.OptimizationState,
                     caller::CC.InferenceResult)
    ir = get(interp.dual_ir, caller.linfo, nothing)
    if ir !== nothing
        ir = run_dual_passes!(ir, opt)
        CC.ipo_dataflow_analysis!(interp, opt, ir, caller)
        CC.finishopt!(interp, opt, ir)
        return nothing
    end
    return @invoke CC.optimize(interp::CC.AbstractInterpreter, opt::CC.OptimizationState,
                               caller::CC.InferenceResult)
end

# The IRCode half of `run_passes_ipo_safe` (our dual IR is already SSA IRCode, so CONVERT/SLOT2REG
# are skipped).
function run_dual_passes!(ir::IRCode, opt::CC.OptimizationState)
    ir = CC.compact!(ir)
    ir = CC.ssa_inlining_pass!(ir, opt.inlining, opt.src.propagate_inbounds)
    ir = CC.compact!(ir)
    ir = CC.sroa_pass!(ir, opt.inlining)
    ir, _ = CC.adce_pass!(ir, opt.inlining)
    ir = CC.compact!(ir, true)
    return ir
end
