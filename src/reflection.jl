# Reflection: view the optimized dualized `IRCode` for a call.
#
# `code_dual_ircode(f, argtypes)` reproduces exactly what the `optimize` seam installs for the
# `dualized_impl` compile of `f`: the split-shadow transform on `f`'s post-optimization primal
# `IRCode` (`build_dual_ir`) followed by the IPO-safe passes (`run_dual_passes!`). It does not go
# through `typeinf_ircode`, which would recompute from the throwing stub and bypass the seam.

"""
    code_dual_ircode(f, argtypes::Tuple; world=Base.get_world_counter()) -> Pair{IRCode,Any}

Return `ir => rettype`, the optimized dualized `IRCode` for differentiating `f` at arguments of
types `argtypes` (each primal type `T` is seeded with a `Dual{T,T}`; the function itself with
`Dual{typeof(f),NoFData}`). Errors if dualization bails (e.g. control flow — not yet supported).

# Examples
```julia
ir, rt = code_dual_ircode(x -> sin(x) + 1, (Float64,))
code_dual_ircode(*, (ComplexF64, ComplexF64))
```
See also [`@code_dual_ircode`](@ref).
"""
function code_dual_ircode(@nospecialize(f), @nospecialize(argtypes::Tuple);
                          world::UInt = Base.get_world_counter())
    interp = ContextualInterpreter(; world)
    dualtys = Any[Dual{typeof(f), NoFData}]
    for T in argtypes
        (T isa Type) || throw(ArgumentError("argtypes must be a tuple of types, got $(repr(T))"))
        push!(dualtys, Dual{T, T})
    end
    impl_tt = Tuple{typeof(dualized_impl), dualtys...}
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    match === nothing && error("no primal method for $f with argument types $argtypes")
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance

    ir = build_dual_ir(interp, impl_mi)
    ir === nothing && error("ADNext could not dualize $f$argtypes on optimized IRCode " *
                            "(unsupported construct, e.g. control flow — not yet handled).")
    opt = CC.OptimizationState(impl_mi, CC.retrieve_code_info(impl_mi, world), interp)
    ir = run_dual_passes!(ir, opt)
    return ir => CC.compute_ir_rettype(ir)
end

"""
    @code_dual_ircode f(args...)

Convenience macro: show the optimized dualized `IRCode` for the call `f(args...)`, using the
runtime types of `args`. Equivalent to `code_dual_ircode(f, map(typeof, (args...,)))`.

```julia
@code_dual_ircode sin(0.5)
@code_dual_ircode (1.0+2.0im) * (3.0+4.0im)
```
"""
macro code_dual_ircode(call::Expr)
    call.head === :call || throw(ArgumentError("@code_dual_ircode expects a function call, e.g. `@code_dual_ircode f(x)`"))
    f = esc(call.args[1])
    args = Expr(:tuple, (esc(a) for a in call.args[2:end])...)
    return :(code_dual_ircode($f, map(typeof, $args)))
end
