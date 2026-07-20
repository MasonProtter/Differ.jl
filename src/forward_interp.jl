function frule_body(world::UInt, lnn, this, dual_argtypes)
    primal_sig = map(primal_type, dual_argtypes)
    Core.println(primal_sig)
    interp = ContextualInterpreter(; world)

    error()
    
    match, valid_worlds = Core.Compiler.findsup(calltup_tt, Core.Compiler.method_table(interp))
    if match === nothing
        error(lazy"Unable to find matching $calltup_tt")
    end
    mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    
    cinst = Core.OptimizedGenerics.CompilerPlugins.typeinf(MyCtx(), mi, Compiler.SOURCE_MODE_ABI)
    
    ci = expr_to_codeinfo(@__MODULE__(), [Symbol("#self#"), :args], [], (), :(return $cinst), true)
    
    matches = Base._methods_by_ftype(dual_tt, -1, world)
    if !isnothing(matches)
        ci.edges = Core.MethodInstance[]
        for match in Base._methods_by_ftype(dual_tt, -1, world)
            mi = Base.specialize_method(match) 
            push!(ci.edges, mi)
        end
    end
    return ci
end

function refresh_frule()
    @eval function generated_frule(dualargs::Dual...)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, frule_body))
    end
end
refresh_frule()
