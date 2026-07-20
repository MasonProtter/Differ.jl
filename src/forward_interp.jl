# Forward-mode AD entry point.
#
# `frule(dualargs::Dual...)` is a `@generated` fallback: for a composite primal `g`
# with no hand-written rule, it compiles a *dualized* version of `g`'s body under
# `ContextualInterpreter` and invokes the resulting `CodeInstance`. The dualization
# (replacing each call `f(a...)` with `frule(Dual(f, struct_zero(f)), <dual a>...)`)
# happens inside the interpreter pipeline, at the `InferenceState` seam
# (see `contextual.jl`), not here.

# Carrier stub: gives a MethodInstance whose specTypes is the *dual* signature.
# `ContextualInterpreter` replaces its source with the dualized primal body, so this
# body must never actually run.
dualized_impl(dualargs::Dual...) =
    error("ADNext.dualized_impl ran directly: ContextualInterpreter could not dualize the primal ",
          "(likely unsupported IR — e.g. control flow, which is not handled in this first cut).")

function frule_body(world::UInt, source, self, dual_argtypes)
    argnames = Any[Symbol("#self#"), :dualargs]

    # Resolve the `dualized_impl` specialization for these dual argument types.
    impl_tt = Tuple{typeof(dualized_impl), dual_argtypes...}
    interp = ContextualInterpreter(; world)
    match, _ = Core.Compiler.findsup(impl_tt, Core.Compiler.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                                :(error("ADNext: no dualized_impl match")), true)
    end
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance

    # Compile the dualized body under ContextualInterpreter -> an invoke-able CodeInstance.
    # Call typeinf_ext_toplevel directly (not CompilerPlugins.typeinf, which would recreate the
    # interpreter at tls_world_age() — stale inside a generator) so it uses `interp`'s generation
    # world; the InferenceState seam then injects the dualized source (see contextual.jl).
    cinst = Compiler.typeinf_ext_toplevel(interp, impl_mi, Compiler.SOURCE_MODE_ABI)

    # Trivial generated body: return invoke(dualized_impl, cinst, dualargs...)
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(dualized_impl, $cinst, dualargs...)), true)
    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

function refresh_frule()
    @eval function frule(dualargs::Dual...)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, frule_body))
    end
end
refresh_frule()
