# Portions of this file are derived from Mooncake.jl (https://github.com/chalk-lab/Mooncake.jl),
# Copyright (c) 2024 Will Tebbutt and Hong Ge, licensed under the MIT License.

using Random: AbstractRNG, MersenneTwister, randn

"""
    NoTangent

The type in question has no meaningful notion of a tangent space.
Generally, you shouldn't use this -- just let the default recursive tangent
construction work.
You might need to use this for `primitive type`s though.
"""
struct NoTangent end

"""
    PossiblyUninitTangent{T}

Represents a `T` that may or may not be present. Does not distinguish between zero and
not being present.
"""
struct PossiblyUninitTangent{T}
    tangent::T
    PossiblyUninitTangent{T}(tangent::T) where {T} = new{T}(tangent)
    PossiblyUninitTangent{T}() where {T} = new{T}()
end

# Copy only if initialized, otherwise create new uninitialized instance
_copy(x::P) where {P<:PossiblyUninitTangent} = is_init(x) ? P(_copy(x.tangent)) : P()

@inline PossiblyUninitTangent(tangent::T) where {T} = PossiblyUninitTangent{T}(tangent)
@inline PossiblyUninitTangent(T::Type) = PossiblyUninitTangent{T}()

@inline is_init(t::PossiblyUninitTangent) = isdefined(t, :tangent)
is_init(t) = true

@inline val(x::PossiblyUninitTangent) =
    (!is_init(x) && error("Uninitialised"); x.tangent)
@inline val(x) = x

"""
    Tangent{Tfields<:NamedTuple}

Default type used to represent the tangent of a `struct`. See [`tangent_type`](@ref) for
more info.
"""
struct Tangent{Tfields<:NamedTuple}
    fields::Tfields
end

_copy(x::T) where {T<:Tangent} = T(_copy(x.fields))

Base.:(==)(x::Tangent, y::Tangent) = x.fields == y.fields

"""
    MutableTangent{Tfields<:NamedTuple}

Default type used to represent the tangent of a `mutable struct`. See [`tangent_type`](@ref)
for more info.
"""
mutable struct MutableTangent{Tfields<:NamedTuple}
    fields::Tfields
    MutableTangent{Tfields}() where {Tfields} = new{Tfields}()
    MutableTangent(fields::Tfields) where {Tfields} = MutableTangent{Tfields}(fields)
    function MutableTangent{Tfields}(
        fields::NamedTuple{names}
    ) where {names,Tfields<:NamedTuple{names}}
        return new{Tfields}(fields)
    end
end

Base.:(==)(x::MutableTangent, y::MutableTangent) = x.fields == y.fields

fields_type(::Type{MutableTangent{Tfields}}) where {Tfields<:NamedTuple} = Tfields
fields_type(::Type{Tangent{Tfields}}) where {Tfields<:NamedTuple} = Tfields
fields_type(::Type{<:Union{MutableTangent,Tangent}}) = NamedTuple

const PossiblyMutableTangent{T} = Union{MutableTangent{T},Tangent{T}}

"""
    get_tangent_field(t::Union{MutableTangent, Tangent}, i::Int)

Gets the `i`th field of data in `t`. The moral equivalent of `getfield` for
[`MutableTangent`](@ref), treating `t.fields` as if its entries were fields of `t` directly.
"""
@inline get_tangent_field(t::PossiblyMutableTangent, i::Int) = val(
    getfield(t.fields, i)
)

@inline function get_tangent_field(
    t::PossiblyMutableTangent{F}, s::Symbol
) where {F}
    return get_tangent_field(t, _sym_to_int(F, Val(s)))
end

"""
    set_tangent_field!(t::MutableTangent{Tfields}, i::Int, x) where {Tfields}

Sets the `i`th field of the data in `t` to `x`. The moral equivalent of `setfield!` for
[`MutableTangent`](@ref), treating `t.fields` as if its entries were fields of `t` directly.
"""
@inline function set_tangent_field!(t::MutableTangent{Tfields}, i::Int, x) where {Tfields}
    fields = t.fields
    Ti = fieldtype(Tfields, i)
    new_val = Ti <: PossiblyUninitTangent ? Ti(x) : x
    new_fields = Tfields(ntuple(n -> n == i ? new_val : fields[n], fieldcount(Tfields)))
    t.fields = new_fields
    return x
end

@inline function set_tangent_field!(t::MutableTangent{T}, s::Symbol, x) where {T}
    return set_tangent_field!(t, _sym_to_int(T, Val(s)), x)
end

@generated function _sym_to_int(::Type{Tfields}, ::Val{s}) where {Tfields,s}
    return findfirst(==(s), fieldnames(Tfields))
end

function tangent_field_types_exprs(P::Type)
    tangent_type_exprs = map(fieldtypes(P), always_initialised(P)) do _P, init
        T_expr = Expr(:call, :tangent_type, _P)
        return init ? T_expr : Expr(:curly, PossiblyUninitTangent, T_expr)
    end
    return tangent_type_exprs
end

# Must be inlined, or the recursion to compute nested tangent types gets slow.
@generated function tangent_field_types(::Type{P}) where {P}
    return Expr(:call, :tuple, tangent_field_types_exprs(P)...)
end

function build_tangent(::Type{P}, fields::Vararg{Any,N}) where {P,N}
    TP = tangent_type(P)
    _ftypes = tangent_field_types(P)
    ftypes = Tuple{_ftypes...}
    fnames = fieldnames(P)
    return _build_tangent_cartesian(
        TP, fields, ftypes, Val(fnames), Val(length(_ftypes))
    )::TP
end
@generated function _build_tangent_cartesian(
    ::Type{TP}, fields::Tuple{Vararg{Any,N}}, ::Type{ftypes}, ::Val{fnames}, ::Val{nfields}
) where {TP,N,ftypes,fnames,nfields}
    quote
        full_fields = Base.Cartesian.@ntuple(
            $nfields, n -> let
                tt = ftypes.types[n]
                if tt <: PossiblyUninitTangent
                    n <= $N ? tt(fields[n]) : tt()
                else
                    fields[n]
                end
            end
        )
        return TP(NamedTuple{$fnames}(full_fields))
    end
end

function build_tangent(
    ::Type{P}, fields::Vararg{Any,N}
) where {P<:Union{Tuple,NamedTuple},N}
    TP = tangent_type(P)
    TP === NoTangent && return NoTangent()::TP
    isconcretetype(P) && return TP(fields)
    return __tangent_from_non_concrete(P, fields)::TP
end

# `@foldable` is defined in `tangent_utils.jl` (included before this file).

"""
    tangent_type(P)

Each primal type `P` has a single tangent type, given by `tangent_type(P)`.

Warning: this function assumes the effects `:removable` and `:consistent`, which is necessary
for performance but imposes precise constraints on your implementation. If adding new methods
to `tangent_type`, consult the extended help of `Base.@assume_effects` to see what this
imposes.

# Extended help

Mooncake.jl's tangent types are similar in spirit to ChainRules.jl. For example, tangent
"vectors" for
1. `Float64`s are `Float64`s,
1. `Vector{Float64}`s are `Vector{Float64}`s, and
1. `struct`s are another (special) `struct` with field types specified recursively.

There are major differences, though. Those tangent types are permissible in ChainRules.jl,
but not uniquely so: `ZeroTangent` is also a valid tangent for any of them, and `Float32` is
valid for `Float64`. ChainRules.jl intentionally declines to restrict what type can represent
the tangent of a given type.

Mooncake.jl differs from this.
**It insists that each primal type is associated to a _single_ tangent type.**
Furthermore, this type is _always_ given by the function `Mooncake.tangent_type(primal_type)`.

Some worked examples.

#### Int

`Int` is not a differentiable type, so its tangent type is [`NoTangent`](@ref):
```jldoctest; setup = :(using Mooncake: tangent_type)
julia> tangent_type(Int)
NoTangent
```

#### Tuples

The tangent type of a `Tuple` is defined recursively based on its field types. For example
```jldoctest; setup = :(using Mooncake: tangent_type)
julia> tangent_type(Tuple{Float64, Vector{Float64}, Int})
Tuple{Float64, Vector{Float64}, NoTangent}
```

Edge case: if all fields of a `Tuple` are non-differentiable, the tangent type is
`NoTangent`. For example,
```jldoctest; setup = :(using Mooncake: tangent_type)
julia> tangent_type(Tuple{Int, Int})
NoTangent
```

#### Structs

As with `Tuple`s, the tangent type of a struct is given recursively by default: it's
[`Tangent`](@ref), which wraps a `NamedTuple` containing the tangent for each field.

As with `Tuple`s, if all field types are non-differentiable, the tangent type of the entire
struct is `NoTangent`.

Two subtleties beyond `Tuple`s. First, not all fields of a `struct` have to be defined; Julia
tracks how many fields are always defined, and the tangent for any field that might not be is
wrapped in a `PossiblyUninitTangent`.

Second, `struct`s can have fields with abstract static types. For example
```jldoctest foo; setup = :(using Mooncake: tangent_type)
julia> struct Foo
           x
       end
```
If you ask for the tangent type of `Foo`, you will see that it is
```jldoctest foo
julia> tangent_type(Foo)
Tangent{@NamedTuple{x}}
```
The field type for `x` is `Any`, because: `x` could have any type at runtime, so its tangent
type isn't known until runtime; and the tangent type of `Foo` must be unique. Together these
mean `Foo`'s tangent must be able to hold any tangent type in its `x` field, so that field's
type has to be `Any`.

#### Mutable Structs

The tangent type for `mutable struct`s follows the same considerations as `struct`s, except
it must itself be mutable. We use [`MutableTangent`](@ref), a `mutable struct` with the same
shape as `Tangent`.

For example, if you ask for the `tangent_type` of
```jldoctest bar; setup = :(using Mooncake: tangent_type)
julia> mutable struct Bar
           x::Float64
       end
```
you will find that it is
```jldoctest bar
julia> tangent_type(Bar)
MutableTangent{@NamedTuple{x::Float64}}
```

#### Primitive Types

We've already seen two primitive types (`Float64` and `Int`). Every primitive type requires
an explicit `tangent_type` method.

The tangent type of a `Ptr{P}` is `Ptr{T}`, where `T = tangent_type(P)`.
For example
```julia
julia> tangent_type(Ptr{Float64})
Ptr{Float64}
```

"""
tangent_type(T)

tangent_type(x) = throw(error("$x is not a type. Perhaps you meant typeof(x)?"))

# The "Bottom" type.
@foldable tangent_type(::Type{Union{}}) = Union{}

# Needed because the recursive definition would otherwise recurse infinitely: one of
# DataType's fieldtypes is itself always a DataType, and we'll eventually hit `Any`, whose
# `super` field is `Any`. Recursive tangent construction can't handle self-referential types.
tangent_type(::Type{<:Type}) = NoTangent

tangent_type(::Type{<:TypeVar}) = NoTangent

@foldable tangent_type(::Type{Ptr{P}}) where {P} = Ptr{tangent_type(P)}

tangent_type(::Type{<:Ptr}) = NoTangent

tangent_type(::Type{Bool}) = NoTangent

tangent_type(::Type{Char}) = NoTangent

tangent_type(::Type{Symbol}) = NoTangent

tangent_type(::Type{Cstring}) = NoTangent

tangent_type(::Type{Cwstring}) = NoTangent

tangent_type(::Type{Module}) = NoTangent

tangent_type(::Type{Nothing}) = NoTangent

tangent_type(::Type{Expr}) = NoTangent

tangent_type(::Type{Core.TypeofVararg}) = NoTangent

tangent_type(::Type{SimpleVector}) = Vector{Any}

tangent_type(::Type{P}) where {P<:Union{UInt8,UInt16,UInt32,UInt64,UInt128}} = NoTangent

tangent_type(::Type{P}) where {P<:Union{Int8,Int16,Int32,Int64,Int128,BigInt}} = NoTangent

tangent_type(::Type{<:Core.Builtin}) = NoTangent

@foldable tangent_type(::Type{P}) where {P<:IEEEFloat} = P

tangent_type(::Type{<:Core.LLVMPtr}) = NoTangent

tangent_type(::Type{String}) = NoTangent

@foldable tangent_type(::Type{<:Array{P,N}}) where {P,N} = Array{tangent_type(P),N}

tangent_type(::Type{<:Array{P,N} where {P}}) where {N} = Array

# A `MemoryRef`'s (and bare `Memory`'s) tangent is the same wrapper over the shadow `Memory`.
# The array-indexing dualization in `forward_interp.jl` never builds these by hand; it mirrors
# the primal's `memoryrefnew`/`getfield(:ref)` onto a genuine `Array{tangent_type(P),N}` shadow
# array. Only matches the default `MemoryRef{P}`/`Memory{P}` (`:not_atomic`, CPU) aliases. A
# `GenericMemoryRef` with a different `Kind`/`AddrSpace` (e.g. atomic memory) falls through to
# the generic struct derivation below instead. This is a known sharp edge, undocumented elsewhere.
@foldable tangent_type(::Type{<:MemoryRef{P}}) where {P} = MemoryRef{tangent_type(P)}
@foldable tangent_type(::Type{<:Memory{P}}) where {P} = Memory{tangent_type(P)}

tangent_type(::Type{<:MersenneTwister}) = NoTangent

tangent_type(::Type{Core.TypeName}) = NoTangent

tangent_type(::Type{Core.MethodTable}) = NoTangent

tangent_type(::Type{DimensionMismatch}) = NoTangent

tangent_type(::Type{Method}) = NoTangent

tangent_type(::Type{<:Enum}) = NoTangent

tangent_type(::Type{<:Base.TTY}) = NoTangent

tangent_type(::Type{<:IOStream}) = NoTangent

tangent_type(::Type{<:Base.LibuvStream}) = NoTangent

tangent_type(::Type{<:Base.CoreLogging.AbstractLogger}) = NoTangent

tangent_type(::Type{Core.CodeInstance}) = NoTangent

tangent_type(::Type{Core.MethodInstance}) = NoTangent

tangent_type(::Type{Core.Binding}) = NoTangent

tangent_type(::Type{Core.Compiler.InferenceState}) = NoTangent

tangent_type(::Type{Core.Compiler.Timings.Timing}) = NoTangent

tangent_type(::Type{Core.Compiler.InferenceResult}) = NoTangent

@static if VERSION >= v"1.11"
    tangent_type(::Type{Core.Compiler.AnalysisResults}) = NoTangent
end

function split_union_tuple_type(tangent_types)

    # Create first split.
    ta_types = map(tangent_types) do T
        return T isa Union ? T.a : T
    end
    ta = Tuple{ta_types...}

    # Create second split.
    tb_types = map(tangent_types) do T
        return T isa Union ? T.b : T
    end
    tb = Tuple{tb_types...}

    return Union{ta,tb}
end

# Generated functions cannot emit closures, so this is defined here for use below.
isconcrete_or_union(p) = p isa Union || isconcretetype(p)

@foldable @generated function tangent_type(::Type{P}) where {N,P<:Tuple{Vararg{Any,N}}}

    # As with other types, tangent type of Union is Union of tangent types.
    P isa Union && return :(Union{tangent_type($(P.a)),tangent_type($(P.b))})

    # Determine whether P isa a Tuple with a Vararg, e.g, Tuple, or Tuple{Float64, Vararg}.
    # Exclude `UnionAll`s by checking `isa(P, DataType)` first, so `Base.datatype_fieldcount(P)`
    # doesn't fail below.
    isa(P, DataType) && !(@isdefined(N)) && return Any

    # Tuple{} can only have `NoTangent` as its tangent type. Again check for `UnionAll` first
    # so datatype_fieldcount doesn't fail.
    isa(P, DataType) && N == 0 && return NoTangent

    # Expression to construct `Tuple` type containing tangent type for all fields.
    tangent_type_exprs = map(n -> :(tangent_type(fieldtype(P, $n))), 1:N)
    tangent_types = Expr(:call, tuple, tangent_type_exprs...)

    # Construct a Tuple type of the same length as `P`, containing all `NoTangent`s.
    T_all_notangent = Tuple{Vararg{NoTangent,N}}

    return quote

        # Get tangent types for all fields. If they're all `NoTangent`, return `NoTangent`.
        # i.e. if `P = Tuple{Int, Int}`, do not return `Tuple{NoTangent, NoTangent}`.
        # Simplify and return `NoTangent`.
        tangent_types = $tangent_types
        T = Tuple{tangent_types...}
        T <: $T_all_notangent && return NoTangent

        # If exactly one of the field types is a Union, then split.
        union_fields = _findall(Base.Fix2(isa, Union), tangent_types)
        if length(union_fields) == 1 && all(tuple_map(isconcrete_or_union, tangent_types))
            return split_union_tuple_type(tangent_types)
        end

        # If it's _possible_ for a subtype of `P` to have tangent type `NoTangent`, then we
        # must account for that by returning the union of `NoTangent` and `T`. For example,
        # if `P = Tuple{Any, Int}`, then `P2 = Tuple{Int, Int}` is a subtype. Since `P2` has
        # tangent type `NoTangent`, it must be true that `NoTangent <: tangent_type(P)`. If,
        # on the other hand, it's not possible for `NoTangent` to be the tangent type, e.g.
        # for `Tuple{Float64, Any}`, then there's no need to take the union.
        return $T_all_notangent <: T ? Union{T,NoTangent} : T
    end
end

@foldable function tangent_type(::Type{P}) where {N,P<:NamedTuple{N}}
    P isa Union && return Union{tangent_type(P.a),tangent_type(P.b)}
    !isconcretetype(P) && return Union{NoTangent,NamedTuple{N}}
    TT = tangent_type(Tuple{fieldtypes(P)...})
    TT == NoTangent && return NoTangent
    return isconcretetype(TT) ? NamedTuple{N,TT} : Any
end

@foldable @generated function tangent_type(::Type{P}) where {P}

    # This method can only handle struct types. Something has gone wrong if P is primitive.
    if isprimitivetype(P)
        return error("$P is a primitive type. Implement a method of `tangent_type` for it.")
    end

    # If the type is a Union, then take the union type of its arguments.
    P isa Union && return :(Union{tangent_type($(P.a)),tangent_type($(P.b))})

    # If the type is itself abstract, its tangent could be anything.
    # The same goes for if the type has any undetermined type parameters.
    (isabstracttype(P) || !isconcretetype(P)) && return Any

    # No self-reference guard here, deliberately. A structurally self-referential `P` (one whose
    # transitive field/element types reach `P` again) does NOT imply non-termination: `GlobalRef`
    # cycles through `binding::Core.Binding`'s own `globalref::GlobalRef`, and derives fine —
    # inference resolves the mutually-recursive `@foldable` calls to a fixpoint, which converges
    # because every other field is `NoTangent`. `Tape`'s cycle does not converge, because it runs
    # through `Stack`/`Vector` overrides that map the element type back through `tangent_type` and so
    # keep producing new types. Telling the two apart at generation time means computing that
    # fixpoint, which is exactly what inference already does — a static reachability walk rejects
    # both, and was measured breaking working `GlobalRef` derivation. A self-referential type whose
    # cycle does not converge must therefore still be given its own `tangent_type` method (see
    # `DifferReverse`'s `Stack`/`Tape`), and gets a hang rather than a diagnostic if it isn't.

    tangent_fields_types_expr = Expr(:curly, Tuple, tangent_field_types_exprs(P)...)
    T_all_notangent = Tuple{Vararg{NoTangent,fieldcount(P)}}
    return quote

        # Construct a `Tuple{...}` whose fields are the tangent types of the fields of `P`.
        tangent_field_types_tuple = $tangent_fields_types_expr

        # If all fields are definitely `NoTangent`s, then return `NoTangent`.
        tangent_field_types_tuple <: $T_all_notangent && return NoTangent

        # Derive tangent type.
        bt = NamedTuple{$(fieldnames(P)),tangent_field_types_tuple}
        return $(ismutabletype(P) ? MutableTangent : Tangent){bt}
    end
end

backing_type(P::Type) = NamedTuple{fieldnames(P),Tuple{tangent_field_types(P)...}}

struct NoCache end

Base.haskey(::NoCache, x) = false
Base.setindex!(::NoCache, v, x) = nothing

const MaybeCache = Union{NoCache,IdDict{Any,Any}}

"""
    zero_tangent(x)

Returns the unique zero element of the tangent space of `x`.
It is an error for the zero element of the tangent space of `x` to be represented by
anything other than that which this function returns.
"""
zero_tangent(x)
function zero_tangent(x::P) where {P}
    # Use `require_tangent_cache`, not a bare `isbitstype(P)`. The cache exists only to handle
    # circular references and aliasing, and `require_tangent_cache` is this system's authority on
    # when a tangent can contain either (`set_to_zero!!` consults the same thing). `isbitstype`
    # is cruder: it allocates an `IdDict` for every `Array`, including `Array{<:IEEEFloat}`, whose
    # tangent is provably tree-like. That allocation was showing up on every `rev_gradient` call, just
    # to build an argument's zero shadow.
    return zero_tangent_internal(x, _tangent_cache(require_tangent_cache(P)))
end
@inline _tangent_cache(::Val{true}) = IdDict()
@inline _tangent_cache(::Val{false}) = NoCache()
function zero_tangent(x::Ptr)
    throw(
        ArgumentError(
            "`zero_tangent` is not safe to call on `Ptr` types with a single argument. " *
            "Use the two-argument form `zero_tangent(primal, fdata)` instead, where `fdata` " *
            "is the fdata component of the `CoDual` for this value.",
        ),
    )
end

"""
    as_tangent(x)::tangent_type(typeof(x))

Returns a value matching a primal type `x` projected to the tangent space.

Examples:

```julia-repl
julia> as_tangent(1.0)
1.0

julia> as_tangent(1)
NoTangent()

julia> as_tangent(1.0 + im)
Tangent{@NamedTuple{re::Float64, im::Float64}}((re = 1.0, im = 1.0))

julia> as_tangent(1 + im)
NoTangent()
```
"""
as_tangent(x) = primal_to_tangent!!(zero_tangent(x), x)::tangent_type(typeof(x))


"""
    unit_tangent(x)::tangent_type(typeof(x))

Equivalent to `as_tangent(oneunit(x))`, i.e. get the equivalent of `oneunit(x)` but
converted to an appropriate tangent type (if such an object exists).

```julia-repl
julia> unit_tangent(10.0)
1.0

julia> unit_tangent(10.0 + im)
Tangent{@NamedTuple{re::Float64, im::Float64}}((re = 1.0, im = 0.0))

julia> unit_tangent(1)
NoTangent()
```

Not every object (e.g. vectors) has a canonical `oneunit`, so these will error.
```
julia> unit_tangent([1.0])
ERROR: MethodError: no method matching one(::Vector{Float64})
The function `one` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  one(::Type{Union{}}, Any...)
   @ Base number.jl:401
  one(::Type{Missing})
   @ Base missing.jl:107
  one(::BitMatrix)
   @ Base bitarray.jl:418
  ...

Stacktrace:
 [1] oneunit(x::Vector{Float64})
   @ Base ./number.jl:424
 [2] unit_tangent(x::Vector{Float64})
   @ Differ ~/Nextcloud/Julia/DifferPlayground/Differ/src/tangents.jl:555
 [3] top-level scope
   @ REPL[114]:1
```
 
"""
unit_tangent(x) = primal_to_tangent!!(zero_tangent(x), oneunit(x))::tangent_type(typeof(x))


"""
    zero_tangent_internal(x, d::MaybeCache)

Implementation of [`zero_tangent`](@ref). Uses `d` the way `Base.deepcopy_internal` uses an
`IdDict` (see `Base.deepcopy`'s docstring).

`d` ensures aliasing is respected: if `x = (a, b)` with `a === b`, then in
`(da, db) = zero_tangent((a, b))` it must hold that `da === db`. See the `struct`/
`mutable struct` methods of `zero_tangent_internal` for reference if implementing this for
your own type.

Similarly, if `x` contains circular references, its tangent generally needs matching circular
references (unless the tangent is trivial, e.g. [`NoTangent`](@ref)).

If `d` is a `NoCache`, assume `x` contains neither aliasing nor circular references.
"""
zero_tangent_internal(::Union{Int8,Int16,Int32,Int64,Int128}, ::MaybeCache) = NoTangent()
zero_tangent_internal(x::IEEEFloat, ::MaybeCache) = zero(x)
@generated function zero_tangent_internal(x::Tuple, dict::MaybeCache)
    zt_exprs = map(n -> :(zero_tangent_internal(x[$n], dict)), 1:fieldcount(x))
    return quote
        tangent_type($x) == NoTangent && return NoTangent()
        return $(Expr(:call, :tuple, zt_exprs...))
    end
end
function zero_tangent_internal(x::NamedTuple, dict::MaybeCache)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    return tuple_map(Base.Fix2(zero_tangent_internal, dict), x)
end
# Ptr fields in Arrays/structs: bitcast to Ptr{tangent_type(P)} as a type-correct
# placeholder. Must not be dereferenced. See uninit_tangent(x::Ptr) for the full WHY.
function zero_tangent_internal(x::Ptr{P}, ::MaybeCache) where {P}
    return bitcast(Ptr{tangent_type(P)}, x)
end
function zero_tangent_internal(x::SimpleVector, dict::MaybeCache)
    return map!(
        n -> zero_tangent_internal(x[n], dict), Vector{Any}(undef, length(x)), eachindex(x)
    )
end
@inline @generated function zero_tangent_internal(x::P, d::MaybeCache) where {P}

    # Loop over fields, constructing expressions to construct zeros depending on the
    # field type and initialisation status.
    inits = always_initialised(P)
    tangent_field_exprs = map(1:fieldcount(P)) do n
        if inits[n]
            return :(zero_tangent_internal(getfield(x, $n), d))
        else
            P_field = fieldtype(P, n)
            T_field_expr = :(PossiblyUninitTangent{tangent_type($P_field)})
            return quote
                if isdefined(x, $n)
                    $T_field_expr(zero_tangent_internal(getfield(x, $n), d))
                else
                    $T_field_expr()
                end
            end
        end
    end
    tangent_fields_tuple_expr = Expr(:call, :tuple, tangent_field_exprs...)

    return quote
        tangent_type(P) == NoTangent && return NoTangent()

        # If dealing with a mutable type, ensure that we have an entry in `d`.
        if tangent_type(P) <: MutableTangent
            haskey(d, x) && return d[x]::tangent_type(P)
            d[x] = tangent_type(P)() # create an uninitialised MutableTangent
        end

        # For each field in `x`, construct its zero tangent. This is where the generated
        # expression above is used. Everything else is regular code.
        fields = backing_type(P)($tangent_fields_tuple_expr)

        if tangent_type(P) <: MutableTangent
            # if circular reference exists, then the recursive call will first look up d
            # and return the uninitialised MutableTangent
            # after the recursive call returns, d will be initialised
            d[x].fields = fields
            return d[x]::tangent_type(P)
        else
            return tangent_type(P)(fields)
        end
    end
end

"""
    normalize_tangent(x)

Returns a normalized copy of tangent `x`, with all numerical fields promoted to `Float64`. Used
to normalize randomly generated tangents from [`randn_tangent`](@ref).
"""
function normalize_tangent(x)
    total_norm = sqrt(_dot(x, x))
    scaling_factor = iszero(total_norm) ? 1.0 : 1 / total_norm
    return _scale(scaling_factor, x)
end

"""
    uninit_tangent(x)

Related to [`zero_tangent`](@ref), but a bit different. Check the implementation for details;
this docstring is intentionally non-specific so it doesn't go stale.
"""
@inline uninit_tangent(x) = zero_tangent(x)
# The tangent of Ptr{P} is a Ptr{tangent_type(P)}, pointing to derivative storage for whatever
# the primal pointer addresses. Gradients accumulate through dereferenced values, not the
# address itself (hence rdata_type(Ptr) = NoRData).
#
# When no derivative storage exists yet (e.g. before a rule fills it in), bitcast the primal
# address to Ptr{tangent_type(P)}. The result must NOT be dereferenced; it's a type-correct
# placeholder only. Single-arg zero_tangent(x::Ptr) throws instead, since allocating fresh
# storage there would have unclear ownership; use zero_tangent(primal, fdata) instead.
@inline uninit_tangent(x::Ptr{P}) where {P} = bitcast(Ptr{tangent_type(P)}, x)

"""
    randn_tangent(rng::AbstractRNG, x::P) where {P}

Required for testing. Generate a randomly-chosen tangent to `x`. Very similar to
[`zero_tangent`](@ref), except that the elements are randomly chosen, rather than
being equal to zero.
"""
function randn_tangent(rng::AbstractRNG, x::P) where {P}
    return randn_tangent_internal(rng, x, isbitstype(P) ? NoCache() : IdDict())
end

"""
    randn_tangent_internal(rng::AbstractRNG, x, dict::MaybeCache)

Implementation for [`randn_tangent`](@ref). Please consult the docstring for
[`zero_tangent_internal`](@ref) for more information on how this implementation works, As
the same implementation strategy is adopted for both this function and that one.
"""
function randn_tangent_internal(rng::AbstractRNG, ::P, ::MaybeCache) where {P<:IEEEFloat}
    return randn(rng, P)
end
@generated function randn_tangent_internal(rng::AbstractRNG, x::Tuple, dict::MaybeCache)
    rt_exprs = map(n -> :(randn_tangent_internal(rng, x[$n], dict)), 1:fieldcount(x))
    return quote
        tangent_type($x) == NoTangent && return NoTangent()
        return $(Expr(:call, :tuple, rt_exprs...))
    end
end
function randn_tangent_internal(rng::AbstractRNG, x::NamedTuple, dict::MaybeCache)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    return tuple_map(x -> randn_tangent_internal(rng, x, dict), x)
end
function randn_tangent_internal(rng::AbstractRNG, x::SimpleVector, dict::MaybeCache)
    return map!(Vector{Any}(undef, length(x)), eachindex(x)) do n
        return randn_tangent_internal(rng, x[n], dict)
    end
end
@generated function randn_tangent_internal(rng::AbstractRNG, x::P, d::MaybeCache) where {P}

    # Loop over fields, constructing expressions to construct randn tangents depending on
    # the field type and initialisation status.
    inits = always_initialised(P)
    tangent_field_exprs = map(1:fieldcount(P)) do n
        if inits[n]
            return :(randn_tangent_internal(rng, getfield(x, $n), d))
        else
            P_field = fieldtype(P, n)
            T_field_expr = :(PossiblyUninitTangent{tangent_type($P_field)})
            return quote
                if isdefined(x, $n)
                    $T_field_expr(randn_tangent_internal(rng, getfield(x, $n), d))
                else
                    $T_field_expr()
                end
            end
        end
    end
    tangent_fields_tuple_expr = Expr(:call, :tuple, tangent_field_exprs...)

    return quote
        tangent_type(P) == NoTangent && return NoTangent()

        # If dealing with a mutable type, ensure that we have an entry in `d`.
        if tangent_type(P) <: MutableTangent
            haskey(d, x) && return d[x]::tangent_type(P)
            d[x] = tangent_type(P)() # create an uninitialised MutableTangent
        end

        # For each field in `x`, construct its randn tangent. This is where the generated
        # expression above is used. Everything else is regular code.
        fields = backing_type(P)($tangent_fields_tuple_expr)

        if tangent_type(P) <: MutableTangent
            # if circular reference exists, then the recursive call will first look up d
            # and return the uninitialised MutableTangent
            # after the recursive call returns, d will be initialised
            d[x].fields = fields
            return d[x]::tangent_type(P)
        else
            return tangent_type(P)(fields)
        end
    end
end

"""
    require_tangent_cache(::Type{P}) where {P}

Whether operations on tangents of primal type `P` need a cache to handle circular references
or aliasing. Returns `Val{true}()` if caching is required (the default), or `Val{false}()` if
tangents of type [`tangent_type(P)`](@ref) are guaranteed free of circular references,
uninitialized fields that could create them, and aliasing.

Used internally by operations like `set_to_zero!!`. `Val{false}()` avoids cache overhead but
is only safe when the tangent type's memory layout is provably tree-like.

!!! warning
    The default (`Val{true}()`) is always correct. Only override it after proving the tangent
    type's memory layout is tree-like.

# Extended help

`Val{false}()` is safe only when the tangent type's memory layout cannot contain circular
references or aliasing:

- **Safe**: pure [`Tangent`](@ref) structures (no [`MutableTangent`](@ref)); `MutableTangent`s
  with only concrete, non-reference fields; `PossiblyUninitTangent` over a concrete
  non-reference type.
- **Unsafe**: any `Any`/abstract-typed field; a type that could reference its own container;
  multiple fields that could alias the same mutable object.

#### Example 1: Circular References with Abstract-Typed Fields

`Ref` with abstract types can lead to circular references:

```jldoctest; setup = :(using Mooncake: tangent_type, zero_tangent)
julia> # Ref{Any} is dangerous because Any can hold circular references
       struct Evil
           r::Ref{Any}
           data::Float64
       end

julia> e = Evil(Ref{Any}(nothing), 1.0);

julia> e.r[] = e;  # Store the struct in its own field!

julia> # The tangent type has PossiblyUninitTangent{Any}
       tangent_type(Evil)
Tangent{@NamedTuple{r, data::Float64}}

julia> # Let's trace what happens with zero_tangent
       zt = zero_tangent(e);

julia> # The Ref field's tangent is a MutableTangent
       typeof(zt.fields.r)
MutableTangent{@NamedTuple{x::Mooncake.PossiblyUninitTangent{Any}}}

julia> # And it contains a circular reference to zt itself!
       zt.fields.r.fields.x.tangent === zt
true
```

#### Example 2: Aliasing in Tangent Structures

When a primal contains aliased references, the tangent must preserve that aliasing. Without
caching, operations would process the aliased tangents twice:

```jldoctest; setup = :(using Mooncake: zero_tangent)
julia> # Create a mutable primal with aliased references
       mutable struct Container
           data::Vector{Float64}
       end

julia> x = Container([1.0, 2.0, 3.0]);

julia> # Create aliasing: both fields reference the same Container
       primal_object = (x, x);

julia> # The tangent preserves the aliasing structure
       zt = zero_tangent(primal_object);

julia> # Verify the tangent type: tuple of two MutableTangents
       typeof(zt)
Tuple{MutableTangent{@NamedTuple{data::Vector{Float64}}}, MutableTangent{@NamedTuple{data::Vector{Float64}}}}

julia> # Both elements are the SAME tangent object (aliased)
       zt[1] === zt[2]
true
```

This aliasing mirrors the primal's: without caching, [`increment!!`](@ref) would visit `zt[1]`
and `zt[2]` separately and double-count the shared tangent.

"""
require_tangent_cache(::Type{P}) where {P} = Val{!isbitstype(P)}()
require_tangent_cache(::Type{<:Array{P}}) where {P} = Val{!isbitstype(P) && tangent_type(P) !== NoTangent}()

const IncCache = Union{NoCache,IdDict{Any,Bool}}
const SetToZeroCache = Union{NoCache,Vector{UInt}}

"""
    _already_tracked!(c::SetToZeroCache, x)

Check if an object has already been tracked and add it to the cache if not.
Returns `true` if the object was already tracked, `false` otherwise.
Mutates the cache by adding untracked objects.
"""
@inline function _already_tracked!(c::Vector{UInt}, x)
    oid = objectid(x)
    oid in c && return true
    push!(c, oid)
    return false
end

@inline _already_tracked!(::NoCache, x) = false

"""
    increment!!(x::T, y::T) where {T}

Add `x` to `y`. If `ismutabletype(T)`, then `increment!!(x, y) === x` must hold.
That is, `increment!!` will mutate `x`.
This must apply recursively if `T` is a composite type whose fields are mutable.
"""
# Use `require_tangent_cache` for the aliasing/circular-reference cache decision, same as
# `zero_tangent` (`_tangent_cache`) and `set_to_zero!!`, keyed here on the tangent type `T`.
# A bare `isbitstype(T)` is cruder: it allocates an `IdDict` for every non-bits tangent,
# including provably tree-like ones like `Vector{<:IEEEFloat}` (`require_tangent_cache` says
# `NoCache` for those). This keeps `increment!!` and `zero_tangent` agreeing on when a cache
# is needed.
@inline _inc_cache(::Val{true}) = IdDict{Any,Bool}()
@inline _inc_cache(::Val{false}) = NoCache()
function increment!!(x::T, y::T) where {T}
    return increment_internal!!(_inc_cache(require_tangent_cache(T)), x, y)
end

"""
    increment_internal!!(c::IncCache, x::T, y::T) where {T}

Implementation of [`Mooncake.increment!!`](@ref). Make use the cache `c` to avoid "double
counting". If `c` is a `NoCache`, assume no aliasing or circular referencing.
"""
increment_internal!!(::IncCache, ::NoTangent, ::NoTangent) = NoTangent()
increment_internal!!(::IncCache, x::T, y::T) where {T<:IEEEFloat} = x + y
function increment_internal!!(::IncCache, x::Ptr{T}, y::Ptr{T}) where {T}
    return x === y ? x : throw(error("Incrementing pointers is not supported!"))
end
@generated function increment_internal!!(c::IncCache, x::T, y::T) where {T<:Tuple}
    inc_exprs = map(n -> :(increment_internal!!(c, x[$n], y[$n])), 1:fieldcount(T))
    return Expr(:call, :tuple, inc_exprs...)
end
@generated function increment_internal!!(c::IncCache, x::T, y::T) where {T<:NamedTuple}
    inc_exprs = map(n -> :(increment_internal!!(c, x[$n], y[$n])), 1:fieldcount(T))
    return Expr(:new, T, inc_exprs...)
end
function increment_internal!!(c::IncCache, x::T, y::T) where {T<:PossiblyUninitTangent}
    is_init(x) && is_init(y) && return T(increment_internal!!(c, val(x), val(y)))
    is_init(x) && !is_init(y) && error("x is initialised, but y is not")
    !is_init(x) && is_init(y) && error("x is not initialised, but y is")
    return x
end
function increment_internal!!(c::IncCache, x::T, y::T) where {T<:Tangent}
    return T(increment_internal!!(c, x.fields, y.fields))
end
function increment_internal!!(c::IncCache, x::T, y::T) where {T<:MutableTangent}
    (x === y || haskey(c, x)) && return x
    c[x] = true
    x.fields = increment_internal!!(c, x.fields, y.fields)
    return x
end

"""
    set_to_zero!!(x)

Set `x` to its zero element (`x` should be a tangent, so the zero must exist).
"""
# `set_to_zero!!` uses a more permissive cache decision than `require_tangent_cache`. Zeroing
# is idempotent, so the `Vector{UInt}` visited-cache is only a perf optimization (skips
# re-zeroing aliased mutable subtrees), never needed for correctness. `increment!!` shares
# `require_tangent_cache` because it does need the `IdDict` cache: it double-counts on aliasing.
# Skip the per-call `Vector{UInt}()` allocation whenever the tangent's reachable substructure
# has no `MutableTangent`, i.e. a `MutableTangent{Tfields}` with isbits `Tfields`.
@inline _set_to_zero_cache(::Type{MutableTangent{Tfields}}) where {Tfields<:NamedTuple} =
    Val{!isbitstype(Tfields)}()
@inline _set_to_zero_cache(@nospecialize T) = require_tangent_cache(T)

set_to_zero!!(x) = set_to_zero!!(x, _set_to_zero_cache(typeof(x)))
set_to_zero!!(x, ::Val{true}) = set_to_zero_internal!!(Vector{UInt}(), x)
set_to_zero!!(x, ::Val{false}) = set_to_zero_internal!!(NoCache(), x)

"""
    set_to_zero_maybe!!(x, doit::Bool)

If `doit` is `true`, return `set_to_zero!!(x)`, otherwise return `x`.
"""
function set_to_zero_maybe!!(x, doit::Bool)
    if doit
        return set_to_zero!!(x)
    else
        return x
    end
end

"""
    set_to_zero_internal!!(c::SetToZeroCache, x)

Implementation for [`Mooncake.set_to_zero!!`](@ref). Use `c` to ensure that circular
references are correctly handled. If `c` is a `NoCache`, assume no circular references.
"""
set_to_zero_internal!!(::SetToZeroCache, ::NoTangent) = NoTangent()
set_to_zero_internal!!(::SetToZeroCache, x::Base.IEEEFloat) = zero(x)
function set_to_zero_internal!!(c::SetToZeroCache, x::Union{Tuple,NamedTuple})
    return tuple_map(Base.Fix1(set_to_zero_internal!!, c), x)
end
function set_to_zero_internal!!(c::SetToZeroCache, x::T) where {T<:PossiblyUninitTangent}
    return is_init(x) ? T(set_to_zero_internal!!(c, val(x))) : x
end
function set_to_zero_internal!!(c::SetToZeroCache, x::T) where {T<:Tangent}
    return T(set_to_zero_internal!!(c, x.fields))
end
function set_to_zero_internal!!(c::SetToZeroCache, x::MutableTangent)
    _already_tracked!(c, x) && return x
    x.fields = set_to_zero_internal!!(c, x.fields)
    return x
end

"""
    _scale(a::Float64, t::T) where {T}

Required for testing.
Should be defined for all standard tangent types.

Multiply tangent `t` by scalar `a`. Always possible since any tangent type corresponds to a
vector field. Not using `*`, to avoid piracy.
"""
_scale(a::Float64, t) = _scale_internal(IdDict{Any,Any}(), a, t)

"""
    _scale_internal(c::MaybeCache, a::Float64, t)

Implementation for [`_scale`](@ref). Use `c` to handle circular references and aliasing in
`t`. If `c` is a `NoCache` assume no circular references or aliasing in `c`.
"""
_scale_internal(::MaybeCache, ::Float64, ::NoTangent) = NoTangent()
_scale_internal(::MaybeCache, a::Float64, t::T) where {T<:IEEEFloat} = T(a * t)
function _scale_internal(c::MaybeCache, a::Float64, t::Union{Tuple,NamedTuple})
    return map(ti -> _scale_internal(c, a, ti)::typeof(ti), t)
end
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:PossiblyUninitTangent}
    return is_init(t) ? T(_scale_internal(c, a, val(t))) : T()
end
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:Tangent}
    return T(_scale_internal(c, a, t.fields))
end
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:MutableTangent}
    haskey(c, t) && return c[t]::T
    y = T()
    c[t] = y
    y.fields = _scale_internal(c, a, t.fields)
    return y
end

struct FieldUndefined end

"""
    _dot(t::T, s::T)::Float64 where {T}

Required for testing.
Should be defined for all standard tangent types.

Inner product between tangents `t` and `s`. Must return a `Float64`.
Always available because all tangent types correspond to finite-dimensional vector spaces.
"""
_dot(t::T, s::T) where {T} = _dot_internal(IdDict{Any,Any}(), t, s)::Float64

"""
    _dot_internal(c::MaybeCache, t::T, s::T) where {T}

Implementation for [`_dot`](@ref). Use `c` to handle circular references and aliasing.
If `c` is a `NoCache`, assume that neither `t` nor `s` contain either circular references
or aliasing.
"""
_dot_internal(::MaybeCache, ::NoTangent, ::NoTangent) = 0.0
_dot_internal(::MaybeCache, t::T, s::T) where {T<:Union{IEEEFloat,Integer}} = Float64(t * s)
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:Union{Tuple,NamedTuple}}
    return sum(map((t, s) -> _dot_internal(c, t, s)::Float64, t, s); init=0.0)::Float64
end
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:PossiblyUninitTangent}
    is_init(t) && is_init(s) && return _dot_internal(c, val(t), val(s))::Float64
    return 0.0
end
function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:Union{Tangent,MutableTangent}}
    key = (t, s)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    return sum(
        _map((t, s) -> _dot_internal(c, t, s)::Float64, t.fields, s.fields); init=0.0
    )::Float64
end

"""
    _add_to_primal(p::P, t::T, unsafe::Bool=false) where {P, T}

Adds `t` to `p`, returning a `P`. It must be the case that `tangent_type(P) == T`.

If `unsafe` is `true` and `P` is a composite type, `_add_to_primal` constructs the new
instance by invoking the `:new` instruction directly, instead of `P`'s default constructor.
Fine if you're confident the perturbed value will always be a valid `P`; can cause problems
otherwise.

This is, for example, fine for the following type:
```julia
struct Foo{T}
    x::Vector{T}
    y::Vector{T}
    function Foo(x::Vector{T}, y::Vector{T}) where {T}
        @assert length(x) == length(y)
        return new{T}(x, y)
    end
end
```
Here, the value returned by `_add_to_primal` will satisfy the invariant asserted in the
inner constructor for `Foo`.
"""
function _add_to_primal(p, t, unsafe::Bool=false)
    return _add_to_primal_internal(IdDict{Any,Any}(), p, t, unsafe)::typeof(p)
end

"""
    _add_to_primal_internal(c::MaybeCache, x, t, ::Bool)

Implementation for [`_add_to_primal`](@ref). Use `c` to handle circular referencing and
aliasing correctly. If `c` is a `NoCache`, assume there is no circular references or
aliasing in either `x` or `t`.
"""
_add_to_primal_internal(::MaybeCache, x, ::NoTangent, ::Bool) = x
_add_to_primal_internal(::MaybeCache, x::T, t::T, ::Bool) where {T<:IEEEFloat} = x + t
function _add_to_primal_internal(
    c::MaybeCache, x::SimpleVector, t::Vector{Any}, unsafe::Bool
)
    haskey(c, (x, t, unsafe)) && return c[(x, t, unsafe)]::SimpleVector
    x′ = svec(map(n -> _add_to_primal_internal(c, x[n], t[n], unsafe), eachindex(x))...)
    c[(x, t, unsafe)] = x′
    return x′
end
function _add_to_primal_internal(c::MaybeCache, x::Tuple, t::Tuple, unsafe::Bool)
    return _map((x, t) -> _add_to_primal_internal(c, x, t, unsafe), x, t)::typeof(x)
end
function _add_to_primal_internal(c::MaybeCache, x::NamedTuple, t::NamedTuple, unsafe::Bool)
    return _map((x, t) -> _add_to_primal_internal(c, x, t, unsafe), x, t)::typeof(x)
end

struct AddToPrimalException <: Exception
    primal_type::Type
end

function Base.showerror(io::IO, err::AddToPrimalException)
    msg =
        "Attempted to construct an instance of $(err.primal_type) using the default " *
        "constuctor. In most cases, this error is caused by the lack of existence of the " *
        "default constructor for this type. There are two approaches to dealing with " *
        "this problem. The first is to avoid having to call `_add_to_primal` on this " *
        "type, which can be achieved by avoiding testing functions whose arguments are " *
        "of this type. If this cannot be avoided, you should consider using calling " *
        "`Mooncake._add_to_primal` with its third positional argument set to `true`. " *
        "If you are using some of Mooncake's testing functionality, this can be achieved " *
        "by setting the `unsafe_perturb` setting to `true` -- check the docstring " *
        "for `Mooncake._add_to_primal` to ensure that your use case is unlikely to " *
        "cause problems."
    return _print_boxed_error(io, split("AddToPrimalException: $msg", '\n'))
end

function __construct_type(::Type{P}, unsafe::Bool, fields::Vararg{Any,N})::P where {P,N}
    i = findfirst(==(FieldUndefined()), fields)

    # If unsafe mode is enabled, then call `_new_` directly, and avoid the possibility that
    # the default inner constructor for `P` does not exist.
    if unsafe
        return i === nothing ? _new_(P, fields...) : _new_(P, fields[1:(i - 1)]...)
    end

    # If unsafe mode is disabled, try to use the default constructor for `P`. If this does
    # not work, then throw an informative error message.
    try
        return i === nothing ? P(fields...) : P(fields[1:(i - 1)]...)
    catch e
        if e isa MethodError
            throw(AddToPrimalException(P))
        else
            rethrow(e)
        end
    end
end

function _add_to_primal_internal(
    c::MaybeCache, p::P, t::T, unsafe::Bool
) where {P,T<:Tangent}
    Tt = tangent_type(P)
    if Tt != typeof(t)
        throw(ArgumentError("p of type $P has tangent_type $Tt, but t is of type $T"))
    end
    fields = map(fieldnames(P)) do f
        tf = getfield(t.fields, f)
        isdefined(p, f) &&
            is_init(tf) &&
            return _add_to_primal_internal(c, getfield(p, f), val(tf), unsafe)
        !isdefined(p, f) && !is_init(tf) && return FieldUndefined()
        throw(error("unable to handle undefined-ness"))
    end
    return __construct_type(P, unsafe, fields...)::P
end

function _add_to_primal_internal(
    c::MaybeCache, p::P, t::T, unsafe::Bool
) where {P,T<:MutableTangent}

    # Do not recompute if we already have a perturbed primal.
    key = (p, t, unsafe)
    haskey(c, key) && return c[key]::P

    # Check that `T` is the correct tangent type for `P`.
    Tt = tangent_type(P)
    if Tt != typeof(t)
        throw(ArgumentError("p of type $P has tangent_type $Tt, but t is of type $T"))
    end

    # Safe to recurse immediately for const fields, since a const field can't contain a
    # circular reference back to `p`. Other (defined) fields get placeholder values and are
    # revisited in a second pass.
    init_fields = map(fieldnames(P)) do f
        tf = getfield(t.fields, f)
        if isdefined(p, f) && is_init(tf) && isconst(P, f)
            return _add_to_primal_internal(c, getfield(p, f), val(tf), unsafe)
        elseif isdefined(p, f) && is_init(tf) && !isconst(P, f)
            return getfield(p, f)
        elseif !isdefined(p, f) && !is_init(tf)
            return FieldUndefined()
        else
            throw(error("unable to handle undefined-ness"))
        end
    end

    # Construct an initial perturbed `p`: const fields perturbed, non-const fields unchanged.
    p′ = __construct_type(P, unsafe, init_fields...)
    c[key] = p′

    # `c[key]` now guards against circular references, so perturb each defined mutable field.
    map(fieldnames(P)) do f
        tf = getfield(t.fields, f)
        if isdefined(p, f) && is_init(tf) && !isconst(P, f)
            setfield!(p′, f, _add_to_primal_internal(c, getfield(p, f), val(tf), unsafe))
        end
    end
    return p′::P
end

"""
    increment_field!!(x::T, y::V, f) where {T, V}

`increment!!` the field `f` of `x` by `y`, and return the updated `x`.
"""
@inline @generated function increment_field!!(x::Tuple, y, ::Val{i}) where {i}
    exprs = map(n -> n == i ? :(increment!!(x[$n], y)) : :(x[$n]), fieldnames(x))
    return Expr(:tuple, exprs...)
end

# Optimal for homogeneously-typed Tuples with dynamic field choice. Uses `ifelse` to keep the
# whole function a single basic block; `n -> n == i ? v : x[n]` would produce one basic block
# per element of `x`, fine for small-medium `x` but bad for large `x` (length > 1_000, and
# probably smaller too).
function increment_field!!(x::Tuple, y, i::Int)
    v = increment!!(x[i], y)
    return ntuple(n -> ifelse(n == i, v, x[n]), Val(length(x)))
end

@inline @generated function increment_field!!(x::T, y, ::Val{f}) where {T<:NamedTuple,f}
    i = f isa Symbol ? findfirst(==(f), fieldnames(T)) : f
    new_fields = Expr(:call, increment_field!!, :(Tuple(x)), :y, :(Val($i)))
    return Expr(:call, T, new_fields)
end

# Optimal for homogeneously-typed NamedTuples with dynamic field choice.
function increment_field!!(x::T, y, i::Int) where {T<:NamedTuple}
    return T(increment_field!!(Tuple(x), y, i))
end
function increment_field!!(x::T, y, s::Symbol) where {T<:NamedTuple}
    return T(tuple_map(n -> n == s ? increment!!(x[n], y) : x[n], fieldnames(T)))
end

function increment_field!!(x::Tangent{T}, y, f::Val{F}) where {T,F}
    y isa NoTangent && return x
    new_val = fieldtype(T, F) <: PossiblyUninitTangent ? fieldtype(T, F)(y) : y
    return Tangent(increment_field!!(x.fields, new_val, f))
end
function increment_field!!(x::MutableTangent{T}, y, f::V) where {T,F,V<:Val{F}}
    y isa NoTangent && return x
    new_val = fieldtype(T, F) <: PossiblyUninitTangent ? fieldtype(T, F)(y) : y
    setfield!(x, :fields, increment_field!!(x.fields, new_val, f))
    return x
end

@inline increment_field!!(x, y, f::Symbol) = increment_field!!(x, y, Val(f))
@inline increment_field!!(x, y, n::Int) = increment_field!!(x, y, Val(n))

# Fallback method for when a tangent type for a struct is declared to be `NoTangent`.
for T in [Symbol, Int, Val]
    @eval increment_field!!(::NoTangent, ::NoTangent, f::Union{$T}) = NoTangent()
end

"""
    tangent_to_primal!!(primal::P, tangent)::P where {P}

Translate a tangent back to a primal type, modifying the differentiable fields
of the primal in place as much as possible to minimize allocations.
The tangent is not modified, and the returned primal will not alias it.

!!! warning
    This function will be removed in the next breaking release (0.6).
    It is retained solely for backward compatibility with downstream packages.
"""
function tangent_to_primal!!(primal::P, tangent) where {P}
    @assert typeof(tangent) <: tangent_type(P)
    return tangent_to_primal_internal!!(
        primal, tangent, isbitstype(P) ? NoCache() : IdDict()
    )::P
end

"""
    primal_to_tangent!!(tangent::T, primal)::T where {T}

Extract the differentiable data from a primal into a tangent type,
modifying the tangent in place as much as possible to minimize allocations.
The primal is not modified, and the returned tangent will not alias it.
"""
function primal_to_tangent!!(tangent, primal::P) where {P}
    @assert typeof(tangent) <: tangent_type(P)
    return primal_to_tangent_internal!!(
        tangent, primal, isbitstype(P) ? NoCache() : IdDict()
    )::tangent_type(P)
end


"""
    tangent_to_primal_internal!!(x, tx, c::MaybeCache)

Internal implementation used to convert tangents back to primal types.

For mutable types, the cache should be used to avoid infinite recursion.
For every mutable `x`, if there is an entry `c[x]`, then it can be returned directly.
Otherwise, the corresponding updated primal should be stored in the cache.
"""
function tangent_to_primal_internal!! end
"""
    primal_to_tangent_internal!!(tx, x, c::MaybeCache)

Implementation of [`primal_to_tangent!!`](@ref).

For mutable types, the cache should be used to avoid infinite recursion.
For every mutable `x`, if there is an entry `c[x]`, then it can be returned directly.
Otherwise, the corresponding updated tangent should be stored in the cache.
"""
function primal_to_tangent_internal!! end

function tangent_to_primal_internal!!(
    x::Union{Int8,Int16,Int32,Int64,Int128}, tx, c::MaybeCache
)
    x
end
function primal_to_tangent_internal!!(
    tx, x::Union{Int8,Int16,Int32,Int64,Int128}, c::MaybeCache
)
    NoTangent()
end
tangent_to_primal_internal!!(x::IEEEFloat, tx, c::MaybeCache) = tx
primal_to_tangent_internal!!(tx, x::IEEEFloat, c::MaybeCache) = x
@generated function tangent_to_primal_internal!!(x::Tuple, tx, c::MaybeCache)
    ttp_exprs = map(n -> :(tangent_to_primal_internal!!(x[$n], tx[$n], c)), 1:fieldcount(x))
    return quote
        tx isa NoTangent && return x
        return $(Expr(:call, :tuple, ttp_exprs...))
    end
end
@generated function primal_to_tangent_internal!!(tx, x::Tuple, c::MaybeCache)
    ptt_exprs = map(n -> :(primal_to_tangent_internal!!(tx[$n], x[$n], c)), 1:fieldcount(x))
    return quote
        tx isa NoTangent && return NoTangent()
        return $(Expr(:call, :tuple, ptt_exprs...))
    end
end
function tangent_to_primal_internal!!(x::NamedTuple, tx, c::MaybeCache)
    tx isa NoTangent && return x
    return tuple_map((xn, txn) -> tangent_to_primal_internal!!(xn, txn, c), x, tx)
end
function primal_to_tangent_internal!!(tx, x::NamedTuple, c::MaybeCache)
    tx isa NoTangent && return NoTangent()
    return tuple_map((txn, xn) -> primal_to_tangent_internal!!(txn, xn, c), tx, x)
end
function tangent_to_primal_internal!!(x::Ptr{T}, tx, c::MaybeCache) where {T}
    tangent_type(T) == NoTangent && return x
    return throw(ArgumentError("tangent_to_primal_internal!! not available for pointers."))
end
function primal_to_tangent_internal!!(tx, x::Ptr{T}, c::MaybeCache) where {T}
    tangent_type(T) == NoTangent && return NoTangent()
    return throw(ArgumentError("primal_to_tangent!! not available for pointers."))
end
function tangent_to_primal_internal!!(x::SimpleVector, tx, c::MaybeCache)
    haskey(c, x) && return c[x]::SimpleVector
    # There doesn't seem to be a nice way to modify a SimpleVector in-place,
    # so we just create a new one.
    x′ = svec(map(x, tx) do xn, txn
        tangent_to_primal_internal!!(xn, txn, c)
    end...)
    c[x] = x′
    return x′
end
function primal_to_tangent_internal!!(tx, x::SimpleVector, c::MaybeCache)
    haskey(c, x) && return c[x]::Vector{Any}
    @assert length(tx) == length(x)
    c[x] = tx
    for i in eachindex(x)
        tx[i] = primal_to_tangent_internal!!(tx[i], x[i], c)
    end
    return tx
end
@generated function tangent_to_primal_internal!!(x::P, tx, c::MaybeCache) where {P}
    if ismutabletype(P)
        # Mutable type: set fields one by one if initialized
        ttp_exprs = map(1:fieldcount(P)) do n
            return quote
                if is_init(tx.fields[$n])
                    isdefined(x, $n) || error(
                        "The field #$($n) of a tangent of type $(typeof(tx)) is initialized " *
                        "but the corresponding primal field is not.",
                    )
                    ccall(
                        :jl_set_nth_field,
                        Cvoid,
                        (Any, Csize_t, Any),
                        x,
                        $(n-1),
                        tangent_to_primal_internal!!(
                            getfield(x, $n), val(tx.fields[$n]), c
                        ),
                    )
                end
                # If the tangent is not initialized, we leave the primal field as-is.
                # It might make sense to unset the field instead.
            end
        end
        return quote
            tx isa NoTangent && return x
            tx isa MutableTangent || error(
                "Generic tangent_to_primal_internal!! implementation expected " *
                "a MutableTangent but received a $(typeof(tx)) tangent type for " *
                "a primal of type $P.\n" *
                "This likely means that a specialized implementation of " *
                "Mooncake.tangent_to_primal_internal!! is missing.",
            )
            haskey(c, x) && return c[x]::P
            c[x] = x
            $(ttp_exprs...)
            return x
        end
    else
        ttp_exprs = map(1:fieldcount(P)) do n
            return :(tangent_to_primal_internal!!(getfield(x, $n), val(tx.fields[$n]), c))
        end
        # Generate a chain of if statements to handle partially-initialized structs
        ninit = CC.datatype_min_ninitialized(P)
        ex = :(return $(Expr(:new, P, ttp_exprs[1:fieldcount(P)]...)))
        for n in (fieldcount(P) - 1):-1:ninit
            cond = :(is_init(tx.fields[$(n + 1)]))
            expr = :(return $(Expr(:new, P, ttp_exprs[1:n]...)))
            ex = Expr(:if, cond, ex, expr)
        end
        return quote
            tx isa NoTangent && return x
            tx isa Tangent || error(
                "Generic tangent_to_primal_internal!! implementation expected " *
                "a Tangent but received a $(typeof(tx)) tangent type for " *
                "a primal of type $P.\n" *
                "This likely means that a specialized implementation of " *
                "Mooncake.tangent_to_primal_internal!! is missing.",
            )
            $ex
        end
    end
end
@generated function primal_to_tangent_internal!!(tx, x::P, c::MaybeCache) where {P}
    inits = always_initialised(P)
    ptt_exprs = map(1:fieldcount(P)) do n
        if inits[n]
            return :(primal_to_tangent_internal!!(tx.fields[$n], getfield(x, $n), c))
        else
            P_field = fieldtype(P, n)
            T_field_expr = :(PossiblyUninitTangent{tangent_type($P_field)})
            return quote
                if isdefined(x, $n)
                    is_init(tx.fields[$n]) || error(
                        "The field #$($n) of an object of type $(typeof(x)) is " *
                        "initialized but the corresponding tangent field is not.",
                    )
                    $T_field_expr(
                        primal_to_tangent_internal!!(
                            val(tx.fields[$n]), getfield(x, $n), c
                        ),
                    )
                else
                    is_init(tx.fields[$n]) && error(
                        "The field #$($n) of an object of type $(typeof(x)) is " *
                        "not initialized but the corresponding tangent field is.",
                    )
                    $T_field_expr()
                end
            end
        end
    end
    if ismutabletype(P)
        return quote
            tx isa NoTangent && return NoTangent()
            tx isa MutableTangent || error(
                "Generic primal_to_tangent_internal!! implementation expected " *
                "a MutableTangent but received a $(typeof(tx)) tangent type for " *
                "a primal of type $P.\n" *
                "This likely means that a specialized implementation of " *
                "Mooncake.primal_to_tangent_internal!! is missing.",
            )
            haskey(c, x) && return c[x]::tangent_type(P)
            c[x] = tx
            Tfields = typeof(tx).parameters[1]
            tx.fields = Tfields(($(ptt_exprs...),))
            return tx
        end
    else
        return quote
            tx isa NoTangent && return NoTangent()
            tx isa Tangent || error(
                "Generic primal_to_tangent_internal!! implementation expected " *
                "a Tangent but received a $(typeof(tx)) tangent type for " *
                "a primal of type $P.\n" *
                "This likely means that a specialized implementation of " *
                "Mooncake.primal_to_tangent_internal!! is missing.",
            )
            Tfields = typeof(tx).parameters[1]
            return Tangent(Tfields(($(ptt_exprs...),)))
        end
    end
end

