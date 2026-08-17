# Portions of this file are derived from Mooncake.jl (https://github.com/chalk-lab/Mooncake.jl),
# Copyright (c) 2024 Will Tebbutt and Hong Ge, licensed under the MIT License.

"""
    NoFData

Singleton indicating there is nothing to propagate on the forwards-pass beyond the primal data.
"""
struct NoFData end

Base.copy(::NoFData) = NoFData()

increment_internal!!(::IncCache, ::NoFData, ::NoFData) = NoFData()

# Zeroing NoFData is a no-op, like zeroing NoTangent (`set_to_zero_internal!!` in `tangents.jl`).
# Needed because `value_and_gradient!` zeroes every caller-supplied shadow uniformly, and an
# argument whose tangent is rdata-carried (any scalar) or absent (a plain function) has `NoFData`
# as its shadow.
set_to_zero_internal!!(::SetToZeroCache, ::NoFData) = NoFData()

"""
    FData(data::NamedTuple)

The component of a `struct` which is propagated alongside the primal on the forwards-pass of
AD. For example, the tangents for `Float64`s do not need to be propagated on the forwards-
pass of reverse-mode AD, so any `Float64` fields of `Tangent` do not need to appear in the
associated `FData`.
"""
struct FData{T<:NamedTuple}
    data::T
end

# Recursively copy the wrapped data
_copy(x::P) where {P<:FData} = P(_copy(x.data))

fields_type(::Type{FData{T}}) where {T<:NamedTuple} = T

# Recurse into the wrapped NamedTuple (bottoms out in the NamedTuple/PossiblyUninitTangent/
# MutableTangent methods in tangents.jl). Needed so `set_to_zero_internal!!` can zero a struct
# field whose tangent carries fdata (e.g. a `Core.Box`-boxed captured variable), not just NoFData.
set_to_zero_internal!!(c::SetToZeroCache, x::F) where {F<:FData} = F(set_to_zero_internal!!(c, x.data))

function increment_internal!!(c::IncCache, x::F, y::F) where {F<:FData}
    return F(tuple_map((a, b) -> increment_internal!!(c, a, b), x.data, y.data))
end

"""
    fdata_type(T)

Returns the type of the forwards data associated to a tangent of type `T`.

# Extended help

Rules in Mooncake.jl do not operate on tangents directly.
Rather, functionality is defined to split each tangent into two components, that we call _fdata_ (forwards-pass data) and _rdata_ (reverse-pass data).
In short, any component of a tangent which is identified by its address (e.g. a `mutable struct`s or an `Array`) gets passed around on the forwards-pass of AD and is incremented in-place on the reverse-pass, while components of tangents identified by their value get propagated and accumulated only on the reverse-pass.

Given a tangent type `T`, you can find out what type its fdata and rdata must be with `fdata_type(T)` and `rdata_type(T)` respectively.
A consequence of this is that there is exactly one valid fdata type and rdata type for each primal type.

Given a tangent `t`, you can get its fdata and rdata using `f = fdata(t)` and `r = rdata(t)` respectively.
`f` and `r` can be re-combined to recover the original tangent using the binary version of `tangent`: `tangent(f, r)`.
It must always hold that
```julia
tangent(fdata(t), rdata(t)) === t
```

The need for this split is explained in the docs; for now, just look at what fdata and rdata look like for our running examples.

#### Int

`Int`s are non-differentiable types, so there is nothing to pass around on the forwards- or reverse-pass.
Therefore
```jldoctest; setup = :(using DifferCore)
julia> fdata_type(tangent_type(Int)), rdata_type(tangent_type(Int))
(NoFData, NoRData)
```

#### Float64

The tangent type of `Float64` is `Float64`.
`Float64`s are identified by their value / have no fixed address, so
```jldoctest; setup = :(using DifferCore)
julia> (fdata_type(Float64), rdata_type(Float64))
(NoFData, Float64)
```

#### Vector{Float64}

The tangent type of `Vector{Float64}` is `Vector{Float64}`.
A `Vector{Float64}` is identified by its address, so
```jldoctest; setup = :(using DifferCore)
julia> (fdata_type(Vector{Float64}), rdata_type(Vector{Float64}))
(Vector{Float64}, NoRData)
```

`Vector{Float64}` does not need rdata because it is the fdata that is incremented in-place during the reverse pass.

#### Tuple{Float64, Vector{Float64}, Int}

This is an example of a type which has both fdata and rdata.
The tangent type for `Tuple{Float64, Vector{Float64}, Int}` is
`Tuple{Float64, Vector{Float64}, NoTangent}`.
`Tuple`s have no fixed memory address, so we interrogate each field on its own.
We have already established the fdata and rdata types for each element, so we recurse to obtain:
```jldoctest; setup = :(using DifferCore)
julia> T = tangent_type(Tuple{Float64, Vector{Float64}, Int})
Tuple{Float64, Vector{Float64}, NoTangent}

julia> (fdata_type(T), rdata_type(T))
(Tuple{NoFData, Vector{Float64}, NoFData}, Tuple{Float64, NoRData, NoRData})
```

The zero tangent for `(5.0, [5.0])` is `t = (0.0, [0.0])`.
`fdata(t)` returns `(NoFData(), [0.0])`, where the second element is `===` to the second element of `t`.
`rdata(t)` returns `(0.0, NoRData())`.
In this example, `t` contains a mixture of data, some of which is identified by its value, and some of which is identified by its address, so there is some fdata and some rdata.

#### Structs

Structs are handled in more-or-less the same way as `Tuple`s, albeit with the possibility of undefined fields needing to be explicitly handled.
For example, a struct such as
```jldoctest foo_fdata; setup = :(using DifferCore)
julia> struct Foo
           x::Float64
           y
           z::Int
       end
```
has tangent type
```jldoctest foo_fdata; setup = :(using DifferCore)
julia> tangent_type(Foo)
Tangent{@NamedTuple{x::Float64, y, z::NoTangent}}
```
Its fdata and rdata are given by special `FData` and `RData` types:
```jldoctest foo_fdata
julia> (fdata_type(tangent_type(Foo)), rdata_type(tangent_type(Foo)))
(FData{@NamedTuple{x::NoFData, y, z::NoFData}}, RData{@NamedTuple{x::Float64, y, z::NoRData}})
```
Practically speaking, `FData` and `RData` both have the same structure as `Tangent`s and are just used in different contexts.

#### Mutable Structs

The fdata for a `mutable struct`s is its tangent, and it has no rdata.
This is because `mutable struct`s have fixed memory addresses, and can therefore be incremented in-place.
For example,
```jldoctest bar_fdata; setup = :(using DifferCore)
julia> mutable struct Bar
           x::Float64
           y
           z::Int
       end
```
has tangent type
```jldoctest bar_fdata; setup = :(using DifferCore)
julia> tangent_type(Bar)
MutableTangent{@NamedTuple{x::Float64, y, z::NoTangent}}
```
and fdata / rdata types
```jldoctest bar_fdata
julia> (fdata_type(tangent_type(Bar)), rdata_type(tangent_type(Bar)))
(MutableTangent{@NamedTuple{x::Float64, y, z::NoTangent}}, NoRData)
```

#### Primitive Types

As with tangents, each primitive type must specify what its fdata and rdata is.
See specific examples for details.
"""
fdata_type(T)

fdata_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

@foldable fdata_type(::Type{Union{}}) = Union{}

fdata_type(::Type{T}) where {T<:IEEEFloat} = NoFData

function fdata_type(::Type{PossiblyUninitTangent{T}}) where {T}
    Tfields = fdata_type(T)
    return PossiblyUninitTangent{Tfields}
end

@generated function fdata_type(::Type{T}) where {T}
    T == NoTangent && return NoFData

    if isprimitivetype(T)
        msg = "$T is a primitive type. Implement a method of `fdata_type` for it."
        return :(error($msg))
    end

    T isa Union && return :(Union{fdata_type($(T.a)),fdata_type($(T.b))})

    ismutabletype(T) && return T

    (isabstracttype(T) || !isconcretetype(T)) && return Any

    T <: Tangent || return :(error("Unhandled type $T"))

    # Some fields of an immutable type may not need to be propagated on the forwards-pass.
    field_names = fieldnames(fields_type(T))
    Tfields = fieldtypes(fields_type(T))
    fdata_type_exprs = map(n -> :(fdata_type($(Tfields[n]))), 1:length(Tfields))
    return quote
        fwds_data_field_types = $(Expr(:call, :tuple, fdata_type_exprs...))
        stable_all(tuple_map(==(NoFData), fwds_data_field_types)) && return NoFData
        return FData{NamedTuple{$field_names,Tuple{fwds_data_field_types...}}}
    end
end

fdata_type(::Type{T}) where {T<:Ptr} = T

# MemoryRef/Memory are reference/address-identified like Ptr (tangent_type is self-typed, see
# tangents.jl): fdata is themselves with no rdata. Needed explicitly since the generic
# struct-derivation below can't handle these primitive-ish compiler types.
fdata_type(::Type{T}) where {T<:MemoryRef} = T
fdata_type(::Type{T}) where {T<:Memory} = T

@generated function fdata_type(::Type{P}) where {P<:Tuple}
    isa(P, Union) && return :(Union{fdata_type($(P.a)),fdata_type($(P.b))})
    isempty(P.parameters) && return NoFData
    isa(last(P.parameters), Core.TypeofVararg) && return Any
    nofdata_tt = Tuple{Vararg{NoFData,length(P.parameters)}}
    fdata_type_exprs = map(_P -> Expr(:call, :fdata_type, _P), P.parameters)
    return quote
        fdata_tt = $(Expr(:curly, Tuple, fdata_type_exprs...))
        fdata_tt <: $nofdata_tt && return NoFData
        return $nofdata_tt <: fdata_tt ? Union{NoFData,fdata_tt} : fdata_tt
    end
end

function fdata_type(::Type{NamedTuple{names,T}}) where {names,T<:Tuple}
    if fdata_type(T) == NoFData
        return NoFData
    elseif isconcretetype(fdata_type(T))
        return NamedTuple{names,fdata_type(T)}
    else
        return Any
    end
end

"""
    fdata_field_type(::Type{P}, n::Int) where {P}

Returns the type of to the nth field of the fdata type associated to `P`. Will be a
`PossiblyUninitTangent` if said field can be undefined.
"""
@inline function fdata_field_type(::Type{P}, n::Int) where {P}
    Tf = tangent_type(fieldtype(P, n))
    f = ismutabletype(P) ? Tf : fdata_type(Tf)
    return is_always_initialised(P, n) ? f : PossiblyUninitTangent{f}
end

"""
    fdata(t)::fdata_type(typeof(t))

Extract the forwards data from tangent `t`.
"""
function fdata(t::T) where {T}
    F = fdata_type(T)
    F == NoFData && return NoFData()
    F == T && return t   # mutable structs, arrays, ... : fdata is the whole object
    T <: Tangent || error("Unhandled type $T")
    return F(fdata(t.fields))
end

function fdata(::Type{T}) where {T}
    error("$T is a type. Perhaps you meant fdata_type($T) or fdata(instance_of_tangent)?")
end

function fdata(t::T) where {T<:PossiblyUninitTangent}
    F = fdata_type(T)
    return is_init(t) ? F(fdata(val(t))) : F()
end

function fdata(t::T) where {T<:Union{Tuple,NamedTuple}}
    return fdata_type(T) == NoFData ? NoFData() : tuple_map(fdata, t)
end

"""
    uninit_fdata(p)

Equivalent to `fdata(uninit_tangent(p))`.
"""
uninit_fdata(p) = fdata(uninit_tangent(p))

"""
    InvalidFDataException(msg::String)

Exception indicating that there is a problem with the fdata associated to a primal.
"""
struct InvalidFDataException <: Exception
    msg::String
end

function Base.showerror(io::IO, err::InvalidFDataException)
    _print_boxed_error(io, split("InvalidFDataException: $(err.msg)", '\n'))
end

"""
    verify_fdata_type(P::Type, F::Type)::Nothing

Check that `F` is a valid type for fdata associated to a primal of type `P`. Returns
`nothing` if valid, throws an `InvalidFDataException` if a problem is found.

This applies to both concrete and non-concrete `P`. For example, if `P` is the type inferred
for a primal `q::Q`, such that `Q <: P`, then this method is still applicable.
"""
function verify_fdata_type(P::Type, F::Type)::Nothing
    _F = fdata_type(tangent_type(P))
    F <: _F && return nothing
    throw(InvalidFDataException("Type $P has fdata type $_F, but got $F."))
end

"""
    verify_fdata_value(p, f)::Nothing

Check that `f` cannot be proven to be invalid fdata for `p`.

This method attempts to provide some confidence that `f` is valid fdata for `p` by checking
a collection of necessary conditions. We do not guarantee that these amount to a sufficient
condition, just that they rule out a variety of common problems.

Put differently, we cannot prove that `f` is valid fdata, only that it is not obviously
invalid.
"""
verify_fdata_value(p, f)::Nothing = _verify_fdata_value(IdDict{Any,Nothing}(), p, f)

function _verify_fdata_value(c::IdDict{Any,Nothing}, p, f)::Nothing
    verify_fdata_type(_typeof(p), typeof(f))
    return __verify_fdata_value(c, p, f)
end

__verify_fdata_value(::IdDict{Any,Nothing}, ::IEEEFloat, ::NoFData) = nothing

__verify_fdata_value(::IdDict{Any,Nothing}, ::Ptr, ::Ptr) = nothing

function __verify_fdata_value(c::IdDict{Any,Nothing}, p::Array, f::Array)
    if size(p) != size(f)
        throw(InvalidFDataException("p has size $(size(p)) but f has size $(size(f))"))
    end

    eltype(f) == NoFData && return nothing

    @static if VERSION > v"1.11-" && VERSION < v"1.12-"
        if p isa Vector && getfield(p, :size)[1] > length(p.ref.mem)
            # Bail out if the vector is mid-resize; validating the tangent here would segfault in
            # debug mode (e.g. inside _growend!'s inner function on Julia v1.11).
            return nothing
        end
    end

    # Recurse into each element. Array elements hold full tangents, so check the fdata and rdata
    # components separately.
    for n in eachindex(p)
        if isassigned(p, n)
            _p = p[n]
            ismutable(_p) && haskey(c, _p) && continue
            ismutable(_p) && !haskey(c, _p) && setindex!(c, nothing, _p)
            t = f[n]
            _verify_fdata_value(c, p[n], fdata(t))
            verify_rdata_value(p[n], rdata(t))
        end
    end

    return nothing
end

# (mutable) structs, Tuples, and NamedTuples all have slightly different storage.
@inline _get_fdata_field(f::NamedTuple, name) = getfield(f, name)
@inline _get_fdata_field(f::Tuple, name) = getfield(f, name)
@inline _get_fdata_field(f::FData, name) = val(getfield(f.data, name))
@inline _get_fdata_field(f::MutableTangent, name) = fdata(
    val(getfield(f.fields, name))
)

function __verify_fdata_value(c::IdDict{Any,Nothing}, p, f)
    f isa NoFData && return nothing

    # A primitive reaching here has no specific _verify_fdata_value method and a non-NoFData
    # fdata type — an error, since everything else here assumes p is a struct.
    P = _typeof(p)
    isprimitivetype(P) && error("Encountered primitive $p with fdata $f")

    for name in fieldnames(P)
        if isdefined(p, name)
            _p = getfield(p, name)
            ismutable(_p) && haskey(c, _p) && continue
            ismutable(_p) && !haskey(c, _p) && setindex!(c, nothing, _p)
            t = _get_fdata_field(f, name)
            _verify_fdata_value(c, _p, t)
            if f isa MutableTangent
                verify_rdata_value(_p, rdata(val(getfield(f.fields, name))))
            end
        end
    end

    return nothing
end

"""
    NoRData()

Nothing to propagate backwards on the reverse-pass.
"""
struct NoRData end

Base.copy(::NoRData) = NoRData()

@inline increment_internal!!(::IncCache, ::NoRData, ::NoRData) = NoRData()

@inline increment_field!!(::NoRData, y, ::Val) = NoRData()

"""
    RData(data::NamedTuple)

The component of a `struct`'s tangent which is *not* propagated on the forwards-pass of reverse-
mode AD, produced only going backwards by a pullback — the complement of [`FData`](@ref). For
example, a `Float64` field's tangent has no fdata, so it is entirely rdata.
"""
struct RData{T<:NamedTuple}
    data::T
end

# Recursively copy the wrapped data
_copy(x::P) where {P<:RData} = P(_copy(x.data))

fields_type(::Type{RData{T}}) where {T<:NamedTuple} = T

# See the matching FData method above for why this is needed.
set_to_zero_internal!!(c::SetToZeroCache, x::R) where {R<:RData} = R(set_to_zero_internal!!(c, x.data))

@inline function increment_internal!!(c::IncCache, x::RData{T}, y::RData{T}) where {T}
    return RData(increment_internal!!(c, x.data, y.data))
end

@inline function increment_field!!(x::RData{T}, y, ::Val{f}) where {T,f}
    y isa NoRData && return x
    new_val = fieldtype(T, f) <: PossiblyUninitTangent ? fieldtype(T, f)(y) : y
    return RData(increment_field!!(x.data, new_val, Val(f)))
end

"""
    rdata_type(T)

Returns the type of the reverse data of a tangent of type T.

# Extended help

See extended help in [`fdata_type`](@ref) docstring.
"""
rdata_type(T)

rdata_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

@foldable rdata_type(::Type{Union{}}) = Union{}

rdata_type(::Type{T}) where {T<:IEEEFloat} = T

function rdata_type(::Type{PossiblyUninitTangent{T}}) where {T}
    return PossiblyUninitTangent{rdata_type(T)}
end

@generated function rdata_type(::Type{T}) where {T}
    T == NoTangent && return NoRData

    if isprimitivetype(T)
        msg = "$T is a primitive type. Implement a method of `rdata_type` for it."
        return :(error($msg))
    end

    T isa Union && return :(Union{rdata_type($(T.a)),rdata_type($(T.b))})

    # Mutable type: all tangent info is propagated on the forwards-pass, so no rdata.
    ismutabletype(T) && return NoRData

    (isabstracttype(T) || !isconcretetype(T)) && return Any

    # Some fields of an immutable type may not need to be propagated on the forwards-pass.
    field_names = fieldnames(fields_type(T))
    Tfields = fieldtypes(fields_type(T))
    rdata_type_exprs = map(n -> :(rdata_type($(Tfields[n]))), 1:length(Tfields))
    return quote
        rvs_data_field_types = $(Expr(:call, :tuple, rdata_type_exprs...))
        stable_all(tuple_map(==(NoRData), rvs_data_field_types)) && return NoRData
        return RData{NamedTuple{$field_names,Tuple{rvs_data_field_types...}}}
    end
end

rdata_type(::Type{<:Ptr}) = NoRData

rdata_type(::Type{<:MemoryRef}) = NoRData
rdata_type(::Type{<:Memory}) = NoRData

@generated function rdata_type(::Type{P}) where {P<:Tuple}
    isa(P, Union) && return :(Union{rdata_type($(P.a)),rdata_type($(P.b))})
    isempty(P.parameters) && return NoRData
    isa(last(P.parameters), Core.TypeofVararg) && return Any
    nordata_tt = Tuple{Vararg{NoRData,length(P.parameters)}}
    rdata_type_exprs = map(_P -> Expr(:call, :rdata_type, _P), P.parameters)
    return quote
        rdata_tt = $(Expr(:curly, Tuple, rdata_type_exprs...))
        rdata_tt <: $nordata_tt && return NoRData
        return $nordata_tt <: rdata_tt ? Union{NoRData,rdata_tt} : rdata_tt
    end
end

function rdata_type(::Type{NamedTuple{names,T}}) where {names,T<:Tuple}
    if rdata_type(T) == NoRData
        return NoRData
    elseif isconcretetype(rdata_type(T))
        return NamedTuple{names,rdata_type(T)}
    else
        return Any
    end
end

"""
    rdata_field_type(::Type{P}, n::Int) where {P}

Returns the type of to the nth field of the rdata type associated to `P`. Will be a
`PossiblyUninitTangent` if said field can be undefined.
"""
function rdata_field_type(::Type{P}, n::Int) where {P}
    r = rdata_type(tangent_type(fieldtype(P, n)))
    return is_always_initialised(P, n) ? r : PossiblyUninitTangent{r}
end

"""
    rdata(t)::rdata_type(typeof(t))

Extract the reverse data from tangent `t`.

# Extended help

See extended help section of [fdata_type](@ref).
"""
function rdata(t::T) where {T}
    R = rdata_type(T)
    R == NoRData && return NoRData()
    R == T && return t   # Float64, isbits structs, ... : rdata is the whole object
    T <: Tangent || error("Unhandled type $T")
    return R(rdata(t.fields))
end

function rdata(::Type{T}) where {T}
    error("$T is a type. Perhaps you meant rdata_type($T) or rdata(instance_of_tangent)?")
end

function rdata(t::T) where {T<:PossiblyUninitTangent}
    R = rdata_type(T)
    return is_init(t) ? R(rdata(val(t))) : R()
end

@generated function rdata(t::Union{Tuple,NamedTuple})
    return :(rdata_type($t) == NoRData ? NoRData() : tuple_map(rdata, t))
end

"""
    rdata_field_types_exprs(::Type{P}) where {P}

Tuple of expressions. The nth computes the rdata backing type of the nth field of `P`.
"""
function rdata_field_types_exprs(::Type{P}) where {P}
    return map(1:fieldcount(P), always_initialised(P)) do n, init
        Pf = fieldtype(P, n)
        if init
            return :(rdata_type(tangent_type($Pf)))
        else
            return :(PossiblyUninitTangent{rdata_type(tangent_type($Pf))})
        end
    end
end

"""
    rdata_backing_type(::Type{P}) where {P}

The type of the field of `RData` for `P`.
"""
@generated function rdata_backing_type(::Type{P}) where {P}
    rdata_field_types_expr = Expr(:call, :tuple, rdata_field_types_exprs(P)...)
    return quote
        rdata_field_types = $rdata_field_types_expr
        stable_all(tuple_map(==(NoRData()), rdata_field_types)) && return NoRData
        return NamedTuple{$(fieldnames(P)),Tuple{rdata_field_types...}}
    end
end

"""
    zero_rdata(p)

Given value `p`, return the zero element associated to its reverse data type.
"""
zero_rdata(p)

zero_rdata(p::IEEEFloat) = zero(p)

@generated function zero_rdata(p::P) where {P}
    Rs = rdata_field_types_exprs(P)
    rdata_field_zeros_exprs = map(1:fieldcount(P), always_initialised(P), Rs) do n, init, R
        if init
            return :(zero_rdata(getfield(p, $n)))
        else
            return quote
                R_field = $R
                isdefined(p, $n) ? R_field(zero_rdata(getfield(p, $n))) : R_field()
            end
        end
    end
    backing_data_expr = Expr(:call, :tuple, rdata_field_zeros_exprs...)
    backing_expr = :(rdata_backing_type($P)($backing_data_expr))

    return quote
        T = tangent_type($P)
        R = rdata_type(T)
        R == NoRData && return NoRData()
        T <: Tangent || error("Unhandled type $T")
        return R($backing_expr)
    end
end

function zero_rdata(p::P) where {P<:Union{Tuple,NamedTuple}}
    return rdata_type(tangent_type(P)) == NoRData ? NoRData() : tuple_map(zero_rdata, p)
end

has_definite_fieldcount(P) = P isa DataType && Base.datatype_fieldcount(P) !== nothing

"""
    can_produce_zero_rdata_from_type(::Type{P}) where {P}

Returns whether or not the zero element of the rdata type for primal type `P` can be
obtained from `P` alone.
"""
@foldable @generated function can_produce_zero_rdata_from_type(::Type{P}) where {P}
    if isstructtype(P) && has_definite_fieldcount(P)
        can_produces = map(_P -> :(can_produce_zero_rdata_from_type($_P)), fieldtypes(P))
    else
        can_produces = ()
    end
    tuple_expr = Expr(:call, :tuple, can_produces...)

    return quote
        R = rdata_type(tangent_type($P))
        R == NoRData && return true
        $(isabstracttype(P)) && return false
        $(isconcretetype(P) || P <: Tuple) || return false
        $(P <: Tuple && !(P isa DataType)) && return false # catch Unions and UnionAlls
        $(P <: Tuple && !has_definite_fieldcount(P)) && return false

        # For general structs, just look at their fields.
        $(!isstructtype(P)) && return false
        return stable_all($tuple_expr)
    end
end

@foldable can_produce_zero_rdata_from_type(::Type{<:IEEEFloat}) = true

@foldable can_produce_zero_rdata_from_type(::Type{<:Type}) = true

@foldable can_produce_zero_rdata_from_type(::Type{Union{}}) = true

"""
    CannotProduceZeroRDataFromType()

Returned by `zero_rdata_from_type` if is not possible to construct the zero rdata element
for a given type. See `zero_rdata_from_type` for more info.
"""
struct CannotProduceZeroRDataFromType end

"""
    zero_rdata_from_type(::Type{P}) where {P}

Returns the zero element of `rdata_type(tangent_type(P))` if this is possible given only
`P`. If not possible, returns an instance of `CannotProduceZeroRDataFromType`.

For example, the zero rdata associated to any primal of type `Float64` is `0.0`, so for
`Float64`s this function is simple. Similarly, if the rdata type for `P` is `NoRData`, that
can simply be returned.

However, it is not possible to return the zero rdata element for abstract types e.g. `Real`
as the type does not uniquely determine the zero element -- the rdata type for `Real` is
`Any`.

These considerations apply recursively to tuples / namedtuples / structs, etc.

If you encounter a type which this function returns `CannotProduceZeroRDataFromType`, but
you believe this is done in error, please open an issue. This kind of problem does not
constitute a correctness problem, but can be detrimental to performance, so should be dealt
with.
"""
@generated function zero_rdata_from_type(::Type{P}) where {P}
    if P isa DataType && isconcretetype(P)
        names = fieldnames(P)
        types = fieldtypes(P)
        wrapped_field_zeros = map(enumerate(always_initialised(P))) do (n, init)
            fzero = :(zero_rdata_from_type($(types[n])))
            init && return fzero
            Q = :(rdata_type(tangent_type($(fieldtype(P, n)))))
            return :(PossiblyUninitTangent{$Q}($fzero))
        end
        wrapped_field_zeros_tuple = Expr(:call, :tuple, wrapped_field_zeros...)
        wrapped_expr = :(R(NamedTuple{$names}($wrapped_field_zeros_tuple)))
    else
        wrapped_expr = nothing
    end

    return quote
        can_produce_zero_rdata_from_type($P) || return CannotProduceZeroRDataFromType()
        R = rdata_type(tangent_type($P))
        R == NoRData && return NoRData()
        $(isstructtype(P)) || error("Unhandled type $P")
        return $wrapped_expr
    end
end

@generated function zero_rdata_from_type(::Type{P}) where {P<:Tuple}
    has_fields = P isa DataType && Base.datatype_fieldcount(P) !== nothing
    zero_exprs = has_fields ? map(_P -> :(zero_rdata_from_type($_P)), fieldtypes(P)) : []
    return quote
        can_produce_zero_rdata_from_type($P) || return CannotProduceZeroRDataFromType()
        rdata_type(tangent_type($P)) == NoRData && return NoRData()
        return $(Expr(:call, :tuple, zero_exprs...))
    end
end

function zero_rdata_from_type(::Type{P}) where {P<:NamedTuple}
    can_produce_zero_rdata_from_type(P) || return CannotProduceZeroRDataFromType()
    rdata_type(tangent_type(P)) == NoRData && return NoRData()
    return NamedTuple{fieldnames(P)}(tuple_map(zero_rdata_from_type, fieldtypes(P)))
end

zero_rdata_from_type(::Type{P}) where {P<:IEEEFloat} = zero(P)

zero_rdata_from_type(::Type{<:Type}) = NoRData()

zero_rdata_from_type(::Type{Union{}}) = NoRData()

"""
    InvalidRDataException(msg::String)

Exception indicating that there is a problem with the rdata associated to a primal.
"""
struct InvalidRDataException <: Exception
    msg::String
end

function Base.showerror(io::IO, err::InvalidRDataException)
    _print_boxed_error(io, split("InvalidRDataException: $(err.msg)", '\n'))
end

"""
    verify_rdata_type(P::Type, R::Type)::Nothing

Check that `R` is a valid type for rdata associated to a primal of type `P`. Returns
`nothing` if valid, throws an `InvalidRDataException` if a problem is found.

This applies to both concrete and non-concrete `P`. For example, if `P` is the type inferred
for a primal `q::Q`, such that `Q <: P`, then this method is still applicable.
"""
function verify_rdata_type(P::Type, R::Type)::Nothing
    _R = rdata_type(tangent_type(P))
    R <: _R && return nothing
    throw(InvalidRDataException("Type $P has rdata type $_R, but got $R."))
end

"""
    verify_rdata_value(p, r)::Nothing

Check that `r` cannot be proven to be invalid rdata for `p`.

This method attempts to provide some confidence that `r` is valid rdata for `p` by checking
a collection of necessary conditions. We do not guarantee that these amount to a sufficient
condition, just that they rule out a variety of common problems.

Put differently, we cannot prove that `r` is valid rdata, only that it is not obviously
invalid.
"""
function verify_rdata_value(p, r)::Nothing
    r isa ZeroRData && return nothing
    verify_rdata_type(_typeof(p), typeof(r))
    return _verify_rdata_value(p, r)
end

_verify_rdata_value(::P, ::P) where {P<:IEEEFloat} = nothing

_verify_rdata_value(::Array, ::NoRData) = nothing

function _verify_rdata_value(p, r)
    r isa NoRData && return nothing

    # A primitive reaching here has no specific _verify_rdata_value method and a non-NoRData
    # rdata type — an error, since everything else here assumes p is a struct.
    P = _typeof(p)
    isprimitivetype(P) && error("Encountered primitive $p with rdata $r")

    # (mutable) structs, Tuples, and NamedTuples all have slightly different storage.
    _get_rdata_field(r::NamedTuple, name) = getfield(r, name)
    _get_rdata_field(r::Tuple, name) = getfield(r, name)
    _get_rdata_field(r::RData, name) = val(getfield(r.data, name))

    for name in fieldnames(P)
        if isdefined(p, name)
            verify_rdata_value(getfield(p, name), _get_rdata_field(r, name))
        end
    end

    return nothing
end

"""
    LazyZeroRData{P, Tdata}()

Lazy placeholder for `zero_like_rdata_from_type`, deferring construction of the zero data to the
reverse pass. Calling `instantiate` on an instance produces the zero data.

Construct via `LazyZeroRData(p)` for `p::P`. Both the constructor and `instantiate` are specialised
to minimise stored data — e.g. `Float64`s need none, so `LazyZeroRData(0.0)` is a singleton
instance, enabling further optimisations in AD.
"""
struct LazyZeroRData{P,Tdata}
    data::Tdata
end

# Recursively copy the wrapped data
_copy(x::P) where {P<:LazyZeroRData} = P(_copy(x.data))

# Returns the type which must be output by LazyZeroRData whenever it is passed a `P`.
@inline function lazy_zero_rdata_type(::Type{P}) where {P}
    Tdata = can_produce_zero_rdata_from_type(P) ? Nothing : rdata_type(tangent_type(P))
    return LazyZeroRData{P,Tdata}
end

# Lazy when the zero element can be computed from the type alone; otherwise store it now for
# later use. L is the exact LazyZeroRData type to construct — occasionally needed when you want
# full control without working out laziness yourself.
@inline function lazy_zero_rdata(::Type{L}, p::P) where {S,L<:LazyZeroRData{S},P}
    return L(can_produce_zero_rdata_from_type(S) ? nothing : zero_rdata(p))
end

# If type parameters for `LazyZeroRData` are not provided, use the defaults.
@inline lazy_zero_rdata(p::P) where {P} = lazy_zero_rdata(lazy_zero_rdata_type(P), p)

# Ensure proper specialisation on types.
@inline function lazy_zero_rdata(p::Type{P}) where {P}
    Rtype = @isdefined(P) ? Type{P} : _typeof(p)
    return LazyZeroRData{Rtype,Nothing}(nothing)
end

@inline instantiate(::LazyZeroRData{P,Nothing}) where {P} = zero_rdata_from_type(P)
@inline instantiate(r::LazyZeroRData) = r.data
@inline instantiate(::NoRData) = NoRData()

"""
    tangent_type(F::Type, R::Type)::Type

Given the type of the fdata and rdata, `F` and `R` resp., for some primal type, compute its
tangent type. This method must be equivalent to `tangent_type(_typeof(primal))`.
"""

@foldable tangent_type(::Type{NoFData}, ::Type{NoRData}) = NoTangent
@foldable tangent_type(::Type{NoFData}, ::Type{R}) where {R<:NoRData} = NoTangent
@foldable tangent_type(::Type{F}, ::Type{NoRData}) where {F<:NoFData} = NoTangent
@foldable tangent_type(::Type{NoFData}, ::Type{R}) where {R<:IEEEFloat} = R
@foldable tangent_type(::Type{F}, ::Type{NoRData}) where {F<:Array} = F

# Union types. Supported shapes:
# - F=NoFData, R<:Union{NoRData, IEEEFloat or RData{...}}
# - F<:Union{NoFData, FData}, R<:Union{NoRData, RData}            (both unions)
# - F<:Union{NoFData, T} for fdata-only T (e.g. Array), R=NoRData
# Each branch recurses by binary splitting on R.a/R.b or F.a/F.b, which handles
# N-branch unions because Julia stores them as nested binary Unions.
@foldable function tangent_type(
    ::Type{NoFData}, ::Type{R}
) where {R<:Union{NoRData,Base.IEEEFloat}}
    @assert R isa Union  # R==NoRData hits a more specific method above; Any is excluded.
    Union{tangent_type(NoFData, R.a),tangent_type(NoFData, R.b)}
end
# Covers Union{NoRData, RData{...}}. More specific than the F-union method below on R, so
# dispatch lands here when F=NoFData and R<:Union{NoRData, RData}.
@foldable function tangent_type(::Type{NoFData}, ::Type{R}) where {R<:Union{NoRData,RData}}
    @assert R isa Union
    Union{tangent_type(NoFData, R.a),tangent_type(NoFData, R.b)}
end
# Both F and R are unions. Arises for primals Union{Nothing, T} where T has both fdata
# and rdata parts; pairing assumes that shape: NoFData ↔ NoRData (Nothing branch) and
# FData ↔ RData (T branch).
@foldable function tangent_type(
    ::Type{F}, ::Type{R}
) where {F<:Union{NoFData,FData},R<:Union{NoRData,RData}}
    @assert F isa Union && R isa Union
    Fa = F.a == NoFData ? F.a : F.b
    Fb = F.a == NoFData ? F.b : F.a
    Ra = R.a == NoRData ? R.a : R.b
    Rb = R.a == NoRData ? R.b : R.a
    Union{tangent_type(Fa, Ra),tangent_type(Fb, Rb)}
end
# More specific than the generic F<:Union{NoFData, T} method below on F. _validate_union
# is unnecessary: FData carries no rdata by construction.
@foldable function tangent_type(::Type{F}, ::Type{NoRData}) where {F<:Union{NoFData,FData}}
    @assert F isa Union
    Union{tangent_type(F.a, NoRData),tangent_type(F.b, NoRData)}
end
# Generic Union{NoFData, T} for non-FData T (e.g. Array). _validate_union rejects T
# values that would silently carry rdata.
@foldable function tangent_type(
    ::Type{F}, ::Type{NoRData}
) where {F<:Union{NoFData,T} where {T}}
    _validate_union(F)
    @assert F isa Union
    Union{tangent_type(F.a, NoRData),tangent_type(F.b, NoRData)}
end

function _validate_union(::Type{F}) where {F<:Union{NoFData,T} where {T}}
    _T = F isa Union ? (F.a == NoFData ? F.b : F.a) : F
    # rdata_type throws for non-IEEEFloat primitive types; guard before calling it.
    if isprimitivetype(_T) || rdata_type(_T) != NoRData
        throw(
            InvalidFDataException("Something went wrong: called tangent_type($F, NoRData)")
        )
    end
    return nothing
end

# Tuples
@foldable @generated function tangent_type(::Type{F}, ::Type{R}) where {F<:Tuple,R<:Tuple}
    tt_exprs = map((f, r) -> :(tangent_type($f, $r)), fieldtypes(F), fieldtypes(R))
    return Expr(:curly, :Tuple, tt_exprs...)
end
@foldable function tangent_type(::Type{NoFData}, ::Type{R}) where {R<:Tuple}
    return tangent_type(Tuple{tuple_fill(NoFData, Val(length(R.parameters)))...}, R)
end
@foldable function tangent_type(::Type{F}, ::Type{NoRData}) where {F<:Tuple}
    return tangent_type(F, Tuple{tuple_fill(NoRData, Val(length(F.parameters)))...})
end

# NamedTuples
@foldable function tangent_type(
    ::Type{F}, ::Type{R}
) where {ns,F<:NamedTuple{ns},R<:NamedTuple{ns}}
    return NamedTuple{ns,tangent_type(tuple_type(F), tuple_type(R))}
end
@foldable function tangent_type(::Type{NoFData}, ::Type{R}) where {ns,R<:NamedTuple{ns}}
    return NamedTuple{ns,tangent_type(NoFData, tuple_type(R))}
end
@foldable function tangent_type(::Type{F}, ::Type{NoRData}) where {ns,F<:NamedTuple{ns}}
    return NamedTuple{ns,tangent_type(tuple_type(F), NoRData)}
end
@foldable tuple_type(::Type{<:NamedTuple{<:Any,T}}) where {T<:Tuple} = T

# mutable structs
@foldable tangent_type(::Type{F}, ::Type{NoRData}) where {F<:MutableTangent} = F

# structs
# Note: Union{RData{A},RData{B}} <: RData in Julia's type system, so the R<:RData and F<:FData
# methods below also match unions of RData/FData subtypes. Guard with `isa Union` so those cases
# recurse via binary splitting instead of calling fields_type on a Union. The F<:FData,R<:RData
# combined case is NOT guarded: pairing two independently-ordered unions would be unreliable and
# shouldn't arise in valid use.
@foldable function tangent_type(::Type{F}, ::Type{R}) where {F<:FData,R<:RData}
    return Tangent{tangent_type(fields_type(F), fields_type(R))}
end
@foldable function tangent_type(::Type{NoFData}, ::Type{R}) where {R<:RData}
    R isa Union && return Union{tangent_type(NoFData, R.a),tangent_type(NoFData, R.b)}
    return Tangent{tangent_type(NoFData, fields_type(R))}
end
@foldable function tangent_type(::Type{F}, ::Type{NoRData}) where {F<:FData}
    F isa Union && return Union{tangent_type(F.a, NoRData),tangent_type(F.b, NoRData)}
    return Tangent{tangent_type(fields_type(F), NoRData)}
end

@foldable function tangent_type(
    ::Type{PossiblyUninitTangent{F}}, ::Type{PossiblyUninitTangent{R}}
) where {F,R}
    return PossiblyUninitTangent{tangent_type(F, R)}
end

# Abstract types.
@foldable tangent_type(::Type{Any}, ::Type{Any}) = Any

"""
    tangent(f, r)

Reconstruct the tangent `t` for which `fdata(t) == f` and `rdata(t) == r`.
"""
tangent(::NoFData, ::NoRData) = NoTangent()
tangent(::NoFData, r::IEEEFloat) = r
tangent(f::Array, ::NoRData) = f
tangent(f::Ptr, ::NoRData) = f

# Tuples
tangent(f::Tuple, r::Tuple) = tuple_map(tangent, f, r)
tangent(::NoFData, r::Tuple) = tuple_map(_r -> tangent(NoFData(), _r), r)
tangent(f::Tuple, ::NoRData) = tuple_map(_f -> tangent(_f, NoRData()), f)

# NamedTuples
function tangent(f::NamedTuple{n}, r::NamedTuple{n}) where {n}
    return NamedTuple{n}(tangent(Tuple(f), Tuple(r)))
end
function tangent(::NoFData, r::NamedTuple{ns}) where {ns}
    return NamedTuple{ns}(tangent(NoFData(), Tuple(r)))
end
function tangent(f::NamedTuple{ns}, ::NoRData) where {ns}
    return NamedTuple{ns}(tangent(Tuple(f), NoRData()))
end

# mutable structs
tangent(f::MutableTangent, r::NoRData) = f

# structs
function tangent(f::F, r::R) where {F<:FData,R<:RData}
    return tangent_type(F, R)(tangent(f.data, r.data))
end
function tangent(::NoFData, r::R) where {R<:RData}
    return tangent_type(NoFData, R)(tangent(NoFData(), r.data))
end
function tangent(f::F, ::NoRData) where {F<:FData}
    return tangent_type(F, NoRData)(tangent(f.data, NoRData()))
end

function tangent(f::PossiblyUninitTangent{F}, r::PossiblyUninitTangent{R}) where {F,R}
    T = PossiblyUninitTangent{tangent_type(F, R)}
    is_init(f) && is_init(r) && return T(tangent(val(f), val(r)))
    !is_init(f) && !is_init(r) && return T()
    throw(ArgumentError("Initialisation mismatch"))
end
function tangent(f::PossiblyUninitTangent{F}, ::PossiblyUninitTangent{NoRData}) where {F}
    T = PossiblyUninitTangent{tangent_type(F, NoRData)}
    return is_init(f) ? T(tangent(val(f), NoRData())) : T()
end
function tangent(::PossiblyUninitTangent{NoFData}, r::PossiblyUninitTangent{R}) where {R}
    T = PossiblyUninitTangent{tangent_type(NoFData, R)}
    return is_init(r) ? T(tangent(NoFData(), val(r))) : T()
end
function tangent(::PossiblyUninitTangent{NoFData}, ::PossiblyUninitTangent{NoRData})
    return PossiblyUninitTangent(NoTangent())
end

"""
    increment_rdata!!(t::T, r)::T where {T}

Increment the rdata component of tangent `t` by `r`, and return the updated tangent.
Useful for implementation getfield-like rules for mutable structs, pointers, dicts, etc.
"""
function increment_rdata!!(t::T, r) where {T}
    return tangent(fdata(t), increment_internal!!(NoCache(), rdata(t), r))::T
end

"""
    increment_field_rdata!(dx::MutableTangent, dy_rdata, f) -> dx

Increment field `f` of a mutable struct's tangent by an rdata contribution `dy_rdata`, in place.
Mutable-struct analogue of `increment_field!!` (`tangents.jl`): a mutable struct has no rdata of
its own (its whole tangent lives in fdata), so a field access routes its contribution here
instead of into an object-level `RData` accumulator.
"""
increment_field_rdata!(dx::MutableTangent, ::NoRData, ::Val) = dx
increment_field_rdata!(dx::NoFData, ::NoRData, ::Val) = dx
function increment_field_rdata!(dx::T, dy_rdata, ::Val{f}) where {T<:MutableTangent,f}
    set_tangent_field!(dx, f, increment_rdata!!(get_tangent_field(dx, f), dy_rdata))
    return dx
end

# Runtime-`Int` field index, for a dynamic (non-literal) `getfield` into a *homogeneous* mutable
# struct (every field shares one tangent type, so field `i` is well-typed regardless of which
# runtime index lands there — `_bi_homog_tangent_type`, `builtins.jl`). Body identical to the `Val`
# method above with `f` -> `i`; used by reverse mode's `Core.getfield` rule (`builtins_reverse.jl`)
# once its comms scan has proven the object homogeneous.
increment_field_rdata!(dx::MutableTangent, ::NoRData, ::Int) = dx
increment_field_rdata!(dx::NoFData, ::NoRData, ::Int) = dx
function increment_field_rdata!(dx::T, dy_rdata, i::Int) where {T<:MutableTangent}
    set_tangent_field!(dx, i, increment_rdata!!(get_tangent_field(dx, i), dy_rdata))
    return dx
end

"""
    zero_tangent(primal, fdata)

Equivalent to `tangent(fdata, zero_rdata(primal))`.

Prefer this two-argument form over the single-argument `zero_tangent(primal)` whenever
`primal` is or contains a `Ptr`, since single-argument `zero_tangent` is not safe for
pointer types (it will throw). The two-argument form handles pointer-containing types
correctly by using `zero_rdata`, whose result for a `Ptr` field is just `NoRData()`.
"""
zero_tangent(p, ::NoFData) = zero_tangent(p)

function zero_tangent(p::P, f::F) where {P,F}
    return tangent_type(P) == F ? f : tangent(f, zero_rdata(p))
end

zero_tangent(p::Tuple, f::Union{Tuple,NamedTuple}) = tuple_map(zero_tangent, p, f)

# ---------------------------------------------------------------------------
# ZeroRData: ported from Mooncake's `src/interpreter/zero_like_rdata.jl`. It belongs with the
# rdata system (a lazy zero-rdata placeholder) and is referenced by `verify_rdata_value`.
# ---------------------------------------------------------------------------

"""
    ZeroRData()

Singleton type indicating zero-valued rdata. This should only ever appear as an
intermediate quantity in the reverse-pass of AD when the type of the primal is not fully
inferable, or a field of a type is abstractly typed.
"""
struct ZeroRData end

@inline increment!!(::ZeroRData, r::R) where {R} = r
@inline increment!!(r::R, ::ZeroRData) where {R} = r
@inline increment!!(::ZeroRData, ::ZeroRData) = ZeroRData()

"""
    zero_like_rdata_type(::Type{P}) where {P}

Indicates the type which will be returned by `zero_like_rdata_from_type`.
"""
function zero_like_rdata_type(::Type{P}) where {P}
    R = rdata_type(tangent_type(P))
    return can_produce_zero_rdata_from_type(P) ? R : Union{R,ZeroRData}
end

"""
    zero_like_rdata_from_type(::Type{P}) where {P}

Returns either the zero element of type `rdata_type(tangent_type(P))`, or a `ZeroRData`.
"""
function zero_like_rdata_from_type(::Type{P}) where {P}
    return can_produce_zero_rdata_from_type(P) ? zero_rdata_from_type(P) : ZeroRData()
end
