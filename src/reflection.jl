# Reflection: view the optimized dualized `IRCode` for a call.
#
# `code_dual_ircode(f, argtypes)` reproduces exactly what the `optimize` seam installs for the
# `dualized_impl` compile of `f`: the split-shadow transform on `f`'s post-optimization primal
# `IRCode` (`build_dual_ir`) followed by the IPO-safe passes (`run_ipo_passes!`). It does not go
# through `typeinf_ircode`, which would recompute from the throwing stub and bypass the seam.

# Nest a value's dual seed `order` levels deep, per the Mooncake tangent system: the leaf tangent
# is `tangent_type(T)`, so order 1 → `Dual{T,tangent_type(T)}` (first derivative). Because a `Dual`
# is its own tangent type (`tangent_type(Dual{P,T}) == Dual{P,T}`, see `dual.jl`), the recursive
# step `Dual{S,tangent_type(S)}` collapses to `Dual{S,S}`, so order 2 → `Dual{Dual{T,U},Dual{T,U}}`
# etc. — matching ADNext's Option-A higher-order nesting.
#
# This is uniform for the function and every value argument (the function nests like any other value
# — see the peel in `build_dual_ir`). A plain function `f` has `tangent_type(typeof(f)) == NoTangent`
# (so `Dual{typeof(f),NoTangent}`); a closure has a `Tangent{…}` tangent over its captures, read in
# the body via the struct `getfield`→`get_tangent_field` path.
_nest_dual(@nospecialize(T), order::Int) =
    order <= 1 ? Dual{T,tangent_type(T)} : (S = _nest_dual(T, order - 1); Dual{S,tangent_type(S)})

"""
    code_dual_ircode(f, argtypes::Tuple; order=1, world=Base.get_world_counter()) -> Pair{IRCode,Any}

Return `ir => rettype`, the optimized dualized `IRCode` for differentiating `f` at arguments of
types `argtypes` (each primal type `T` is seeded with a `Dual{T,T}`; the function itself with
`Dual{typeof(f),NoTangent}`). `order` requests a higher derivative by nesting *all* seeds uniformly
to that depth — `order=2` seeds each `T` with `Dual{Dual{T,T},Dual{T,T}}` and the function with
`Dual{Dual{typeof(f),NoTangent},Dual{typeof(f),NoTangent}}` — which is useful for running
`Core.Compiler.verify_ir` on the higher-order IR without a live call. Errors if dualization bails
(e.g. array indexing — not yet supported).

# Examples
```julia
ir, rt = code_dual_ircode(x -> sin(x) + 1, (Float64,))
code_dual_ircode(*, (ComplexF64, ComplexF64))
code_dual_ircode(x -> x*x, (Float64,); order=2)   # second-derivative IR
```
See also [`@code_dual_ircode`](@ref).
"""
function code_dual_ircode(@nospecialize(f), @nospecialize(argtypes::Tuple);
                          order::Int = 1, world::UInt = Base.get_world_counter())
    interp = ADInterpreter{Forward}(; world)
    dualtys = Any[_nest_dual(typeof(f), order)]
    for T in argtypes
        (T isa Type) || throw(ArgumentError("argtypes must be a tuple of types, got $(repr(T))"))
        push!(dualtys, _nest_dual(T, order))
    end
    impl_tt = Tuple{typeof(dualized_impl), dualtys...}
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    match === nothing && error("no primal method for $f with argument types $argtypes")
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance

    reason = Ref("no specific reason recorded")
    ir = optimized_dual_ir(interp, impl_mi, reason)
    ir === nothing && error("ADNext could not dualize $f$argtypes on optimized IRCode: $(reason[])")
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
