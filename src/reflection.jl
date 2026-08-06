# Reflection: view the optimized dualized `IRCode` for a call.
#
# `code_dual_ircode(f, argtypes)` reproduces exactly what `CC.optimize` installs for the
# `dualized_impl` compile of `f`: the split-shadow transform on `f`'s post-optimization primal
# `IRCode` (`build_dual_ir`) followed by the IPO-safe passes (`run_ipo_passes!`). It does not go
# through `typeinf_ircode`, which would recompute from the throwing stub instead of calling
# `CC.optimize`.

# Nest a value's dual seed `order` levels deep, per the Mooncake tangent system: the leaf tangent is
# `tangent_type(T)`, so order 1 → `Dual{T,tangent_type(T)}` (first derivative). Because a `Dual` is
# its own tangent type (`tangent_type(Dual{P,T}) == Dual{P,T}`, see `dual.jl`), the recursive step
# `Dual{S,tangent_type(S)}` collapses to `Dual{S,S}`, so order 2 → `Dual{Dual{T,U},Dual{T,U}}`, etc.
# — matching Differ's Option-A higher-order nesting.
#
# Uniform for the function and every value argument (the function nests like any other value — see
# the peel in `build_dual_ir`). A plain function `f` has `tangent_type(typeof(f)) == NoTangent` (so
# `Dual{typeof(f),NoTangent}`); a closure has a `Tangent{…}` tangent over its captures, read in the
# body via the struct `getfield`→`get_tangent_field` path.
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
    dualtys = Any[_nest_dual(_typeof(f), order)]
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
    ir === nothing && error("Differ could not dualize $f$argtypes on optimized IRCode: $(reason[])")
    return ir => CC.compute_ir_rettype(ir)
end

"""
    @code_dual_ircode f(args...)

Convenience macro: show the optimized dualized `IRCode` for the call `f(args...)`, using the
runtime types of `args`. Equivalent to `code_dual_ircode(f, map(_typeof, (args...,)))`.

```julia
@code_dual_ircode sin(0.5)
@code_dual_ircode (1.0+2.0im) * (3.0+4.0im)
```
"""
macro code_dual_ircode(call::Expr)
    call.head === :call || throw(ArgumentError("@code_dual_ircode expects a function call, e.g. `@code_dual_ircode f(x)`"))
    f = esc(call.args[1])
    args = Expr(:tuple, (esc(a) for a in call.args[2:end])...)
    return :(code_dual_ircode($f, map(_typeof, $args)))
end

"""
    code_reverse_fwds_ircode(f, argtypes::Tuple; world=Base.get_world_counter()) -> Pair{IRCode,Any}

Return `ir => rettype`, the optimized `IRCode` for the reverse-mode *forwards* carrier
(`reverse_fwds_impl`) differentiating `f` at arguments of types `argtypes` (each primal type `T`,
and `f` itself, wrapped in `fcodual_type`). `rettype` is `Tuple{CoDual, Tape}` — the primal result
plus the tape `code_reverse_pullback_ircode` needs. Reproduces exactly what `CC.optimize`
installs, without going through the throwing carrier stub. Errors if the reverse-mode transform
bails (see the scope notes in `reverse_interp.jl`'s header).

# Examples
```julia
ir, rt = code_reverse_fwds_ircode(x -> x*x + x, (Float64,))
```
See also [`@code_reverse_fwds_ircode`](@ref) and [`code_reverse_pullback_ircode`](@ref).
"""
function code_reverse_fwds_ircode(@nospecialize(f), @nospecialize(argtypes::Tuple);
                                  world::UInt=Base.get_world_counter())
    interp = ADInterpreter{Reverse}(; world)
    codualtys = Any[fcodual_type(_typeof(f))]
    for T in argtypes
        (T isa Type) || throw(ArgumentError("argtypes must be a tuple of types, got $(repr(T))"))
        push!(codualtys, fcodual_type(T))
    end
    # Inspect the tape-allocating shape (`Ctx{Nothing}`) — the one `build_ctx(...; prealloc=false)`
    # uses, and the shape a pre-allocated context differs from only in its prologue. Carrier layout is
    # `reverse_fwds_impl(fcd, ctx, argcds...)`: fcd first, then the ctx, then the argument coduals.
    impl_tt = Tuple{typeof(reverse_fwds_impl), codualtys[1], Ctx{Nothing}, codualtys[2:end]...}
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    match === nothing && error("no primal method for $f with argument types $argtypes")
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance

    reason = Ref("no specific reason recorded")
    ir = optimized_reverse_fwds_ir(interp, impl_mi, reason)
    ir === nothing &&
        error("Differ could not build the reverse forwards pass for $f$argtypes on optimized IRCode: $(reason[])")
    return ir => CC.compute_ir_rettype(ir)
end

"""
    @code_reverse_fwds_ircode f(args...)

Convenience macro: show the optimized `IRCode` for the reverse-mode forwards carrier for the call
`f(args...)`, using the runtime types of `args`.
"""
macro code_reverse_fwds_ircode(call::Expr)
    call.head === :call || throw(ArgumentError("@code_reverse_fwds_ircode expects a function call, e.g. `@code_reverse_fwds_ircode f(x)`"))
    f = esc(call.args[1])
    args = Expr(:tuple, (esc(a) for a in call.args[2:end])...)
    return :(code_reverse_fwds_ircode($f, map(_typeof, $args)))
end

"""
    code_reverse_pullback_ircode(f, argtypes::Tuple; seedtype::Type=Float64, world=Base.get_world_counter()) -> Pair{IRCode,Any}

Return `ir => rettype`, the optimized `IRCode` for the reverse-mode *pullback* carrier
(`reverse_pullback_impl`) for `f` at arguments of types `argtypes`. `seedtype` is the type of the
rdata seed for the primal's return value (default `Float64`, for scalar output). The `Tape` type is
recovered from `code_reverse_fwds_ircode`'s own return type, so both carriers agree on it by
construction (mirroring how they'd be produced by an actual `rrule` call).

# Examples
```julia
ir, rt = code_reverse_pullback_ircode(x -> x*x + x, (Float64,))
```
See also [`@code_reverse_pullback_ircode`](@ref) and [`code_reverse_fwds_ircode`](@ref).
"""
function code_reverse_pullback_ircode(@nospecialize(f), @nospecialize(argtypes::Tuple);
                                      seedtype::Type=Float64, world::UInt=Base.get_world_counter())
    interp = ADInterpreter{Reverse}(; world)
    _, fwds_rt = code_reverse_fwds_ircode(f, argtypes; world)
    TapeT = fwds_rt.parameters[2]
    impl_tt = Tuple{typeof(reverse_pullback_impl), TapeT, seedtype}
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    match === nothing && error("no primal method for $f with argument types $argtypes")
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance

    reason = Ref("no specific reason recorded")
    ir = optimized_reverse_pullback_ir(interp, impl_mi, reason)
    ir === nothing &&
        error("Differ could not build the reverse pullback pass for $f$argtypes on optimized IRCode: $(reason[])")
    return ir => CC.compute_ir_rettype(ir)
end

"""
    tape_type(f, argtypes::Tuple; world=Base.get_world_counter()) -> Type{<:Tape}

The concrete `Tape` type the reverse-mode forwards carrier for `f` at `argtypes` returns — i.e. the
tape a real `rrule!!` call would build. Recovered from `code_reverse_fwds_ircode`'s return type,
exactly as `code_reverse_pullback_ircode` does it, so this reports the tape both carriers agree on.

Chiefly useful with [`comms_element_types`](@ref) for asserting on tape layout in tests.
"""
function tape_type(@nospecialize(f), @nospecialize(argtypes::Tuple);
                   world::UInt=Base.get_world_counter())
    return code_reverse_fwds_ircode(f, argtypes; world)[2].parameters[2]
end

"""
    comms_element_types(TapeT::Type{<:Tape}) -> Vector{Any}

The element type of every per-block comms stack in `TapeT` that actually stores something, in block
order. `SingletonStack` slots (blocks with nothing to communicate) are skipped — they carry no
storage, so they say nothing about tape size.

Each element is the `Tuple` type that block pushes once per execution. Assert on properties of the
whole collection (`all(isbitstype, …)`, `sum(sizeof, …)`) rather than on a particular block's index:
block numbering shifts with any unrelated change to Julia's optimizer.
"""
function comms_element_types(@nospecialize(TapeT::Type))
    (TapeT <: Tape) || throw(ArgumentError("expected a `Tape` type, got $(TapeT)"))
    return Any[S.parameters[1] for S in TapeT.parameters[2].parameters if S <: Stack]
end

"""
    @code_reverse_pullback_ircode f(args...)

Convenience macro: show the optimized `IRCode` for the reverse-mode pullback carrier for the call
`f(args...)`, using the runtime types of `args`.
"""
macro code_reverse_pullback_ircode(call::Expr)
    call.head === :call || throw(ArgumentError("@code_reverse_pullback_ircode expects a function call, e.g. `@code_reverse_pullback_ircode f(x)`"))
    f = esc(call.args[1])
    args = Expr(:tuple, (esc(a) for a in call.args[2:end])...)
    return :(code_reverse_pullback_ircode($f, map(_typeof, $args)))
end
