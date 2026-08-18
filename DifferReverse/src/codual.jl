# Portions of this file are derived from Mooncake.jl (https://github.com/chalk-lab/Mooncake.jl),
# Copyright (c) 2024 Will Tebbutt and Hong Ge, licensed under the MIT License.

"""
    CoDual{Tx,Tdx}

A primal value paired with its shadow. `Tdx` is one of:

- `fdata_type(tangent_type(Tx))` — the ordinary active carrier, built by [`fcodual_type`](@ref);
- `tangent_type(Tx)` — the full-tangent flavour, built by [`codual_type`](@ref);
- `Inactive` — the value is *held constant*: no derivative is propagated to or from it.

The third case is what `isactive` tests, and it is decidable from `Tdx` alone. That is why it is
`Inactive` rather than `NoTangent`: an active `Float64`'s shadow is `NoFData()`, so an empty fdata
cannot mean "constant", and `NoTangent` is already the tangent of a type with no tangent space.

Marking an argument constant is a promise about aliasing: an inactive value must not share memory
with an active one, and an active value must not be stored into an inactive container. Neither is
checkable here.
"""
struct CoDual{Tx,Tdx}
    x::Tx
    dx::Tdx
end

function Base.getproperty(d::CoDual, s::Symbol)
    if s === :x || s === :y || s === :z || s === :w || s === :primal
        getfield(d, :x)
    elseif s === :dx || s === :dy || s === :dz || s === :dw || s === :tangent
        getfield(d, :dx)
    else
        getfield(d, s)
    end
end
Base.propertynames(::CoDual) = (:primal, :tangent, :x, :y, :z, :w, :dx, :dy, :dz, :dw)

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function CoDual(x::Type{P}, dx::NoFData) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),NoFData}(P, dx)
end

function CoDual(x::Type{P}, dx::NoTangent) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(P, dx)
end

function CoDual(x::Type{P}, dx::Inactive) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),Inactive}(P, dx)
end

primal(x::CoDual) = x.x
tangent(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(tangent(x)))
# CoDual can be safely shared without copying
_copy(x::P) where {P<:CoDual} = x

"""
    extract(x::CoDual)

Returns the 2-tuple `x.x, x.dx`.
"""
extract(x::CoDual) = primal(x), tangent(x)

"""
    zero_codual(x)

Equivalent to `CoDual(x, zero_tangent(x))`.

For `Ptr{P}`, falls back to `uninit_codual(x)` instead: a true zero tangent needs newly allocated
derivative storage with unclear ownership, so the tangent pointer is a bitcast of the primal
address to `Ptr{tangent_type(P)}` — a type-correct placeholder, not real derivative storage.
"""
zero_codual(x) = CoDual(x, zero_tangent(x))
zero_codual(x::Ptr{P}) where {P} = uninit_codual(x)

"""
    uninit_codual(x)

Equivalent to `CoDual(x, uninit_tangent(x))`.
"""
uninit_codual(x) = CoDual(x, uninit_tangent(x))

function _codual_internal(::Type{P}, f::F, extractor::E) where {P,F,E}
    P == Union{} && return Union{}
    P == DataType && return CoDual
    P isa Union && return Union{f(P.a),f(P.b)}
    # Use `isa` not `<:`: generators like `NTuple{N,Int} where N` are instances of
    # UnionAll but not subtypes of it (`NTuple{N,Int} where N <: UnionAll` is false).
    # `P == UnionAll` handles the UnionAll metatype itself (`UnionAll isa UnionAll` is false).
    (P isa UnionAll || P == UnionAll) && return CoDual # P is abstract, tangent type unknown.

    if P <: Tuple && !all(isconcretetype, (P.parameters...,))
        field_types = (P.parameters...,)
        union_fields = _findall(Base.Fix2(isa, Union), field_types)
        if length(union_fields) == 1 &&
            all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_union_tuple_type(field_types)
            return Union{f(P_split.a),f(P_split.b)}
        end
    end

    return isconcretetype(P) ? CoDual{P,extractor(P)} : CoDual
end

"""
    codual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and associated tangents.
"""
function codual_type(::Type{P}) where {P}
    # @isdefined(P) is false when the static parameter couldn't be bound at dispatch, e.g. for
    # UnionAll(A, AbstractArray{T, A}) whose body has a free TypeVar T. Without this check, touching
    # P throws UndefVarError(:P, :static_parameter). Same check guards the overloads below and
    # dual_type in src/dual.jl.
    @isdefined(P) || return CoDual
    return _codual_internal(P, codual_type, tangent_type)
end

function codual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? CoDual{Type{P},NoTangent} : CoDual{_typeof(p),NoTangent}
end

"""
    fcodual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and its fdata.
"""
function fcodual_type(::Type{P}) where {P}
    @isdefined(P) || return CoDual
    return _codual_internal(P, fcodual_type, P -> fdata_type(tangent_type(P)))
end

function fcodual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? CoDual{Type{P},NoFData} : CoDual{_typeof(p),NoFData}
end

to_fwds(x::CoDual) = CoDual(primal(x), fdata(tangent(x)))

to_fwds(x::CoDual{Type{P}}) where {P} = CoDual{Type{P},NoFData}(primal(x), NoFData())

# No `NoTangent` arm: `zero_codual` yields `CoDual{P,NoTangent}` for every primal whose own
# `tangent_type` is `NoTangent`, and converting those to `NoFData` is exactly this function's job. An
# inactive carrier (`Inactive` shadow) is built by the caller, already in fdata form, and never
# routed through here.

"""
    zero_fcodual(x)

Equivalent to `CoDual(x, fdata(zero_tangent(x)))`.

For `Ptr{P}`, falls back to `uninit_fcodual(x)` for the same reason as `zero_codual` — the full
tangent is fdata for `Ptr`, so the placeholder bitcast applies here too.
"""
zero_fcodual(p) = to_fwds(zero_codual(p))
zero_fcodual(p::Ptr{P}) where {P} = uninit_fcodual(p)

"""
    uninit_fcodual(x)

Like `zero_fcodual`, but doesn't guarantee that the value of the fdata is initialised.
See implementation for details, as this function is subject to change.
"""
@inline uninit_fcodual(x::P) where {P} = CoDual(x, uninit_fdata(x))

struct NoPullback{R<:Tuple}
    r::R
end

# Recursively copy the contained reverse data
_copy(x::P) where {P<:NoPullback} = P(_copy(x.r))

"""
    NoPullback(args::CoDual...)

Construct a `NoPullback` from the arguments passed to an `rrule!!`: extracts each argument's
primal value and wraps it as a `LazyZeroRData`, which the reverse pass instantiates and returns
as that argument's rdata.

If every argument's zero rdata can be constructed lazily, the resulting `NoPullback` is a
singleton type, so AD can skip allocating a stack to store it.
"""
function NoPullback(args::Vararg{CoDual,N}) where {N}
    return NoPullback(tuple_map(lazy_zero_rdata ∘ primal, args))
end

@inline (pb::NoPullback)(_) = tuple_map(instantiate, pb.r)
