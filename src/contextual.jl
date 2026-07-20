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
    interp = ContextualInterpreter{Owner}(; world)
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


