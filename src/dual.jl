macro outline(args...)
    fargs = esc.(args[1:end-1])
    body = args[end]
    @gensym f
    quote
        $f($(fargs...),) = $body
        @noinline $f($(fargs...),)
    end
end


"""
    Dual(primal::P, tangent::T)

Used to pair together a `primal` value and a `tangent` to it. In the context of foward mode
AD (aka computing Fréchet derivatives), `primal` governs the point at which the derivative
is computed, and `tangent` the direction in which it is computed.

Must satisfy `tangent_type(P) == T`.
"""
struct Dual{P,T}
    primal::P
    tangent::T
    function Dual{P, T}(x, dx) where {P, T}
        if !(tangent_type(P) == T)
            @outline P T throw(ArgumentError(
                "Invalid Dual{P,T} construction for primal type P=$P\n\tgot tangent type T=$T\n\tADNext requires that tangent_type(P) == T"
            ))
        end
        new{P, T}(convert(P, x), convert(T, dx))
    end
    function Dual(x::P, dx::T) where {P, T}
        Tproper = tangent_type(P)
        if !(T <: Tproper)
            tdx = as_tangent(dx)
            if typeof(tdx) <: Tproper
                return new{P, Tproper}(x, tdx)
            else
                @outline P T throw(ArgumentError(
                    "Invalid Dual construction, Dual(::$(P), ::$(T)). The tangent type T=$T does not match tangent_type(P)=$(tangent_type(P))"
                ))
            end
        else 
            new{P, T}(x, dx)
        end
    end
end

primal(x::Dual) = x.primal
tangent(x::Dual) = x.tangent
Base.copy(x::Dual) = Dual(copy(primal(x)), copy(tangent(x)))
# Dual can be safely shared without copying
_copy(x::P) where {P<:Dual} = x

"""
    extract(x::Dual)

Helper function. Returns the 2-tuple `x.x, x.dx`.
"""
extract(x::Dual) = primal(x), tangent(x)

zero_dual(x) = Dual(x, zero_tangent(x))
randn_dual(rng::AbstractRNG, x) = Dual(x, randn_tangent(rng, x))

@unstable function dual_type(::Type{P}) where {P}
    @isdefined(P) || return Dual
    P == Union{} && return Union{}
    P == DataType && return Dual
    P isa Union && return Union{dual_type(P.a),dual_type(P.b)}
    # Use `isa` not `<:`: generators like `NTuple{N,Int} where N` are instances of
    # UnionAll but not subtypes of it (`NTuple{N,Int} where N <: UnionAll` is false).
    # `P == UnionAll` handles the UnionAll metatype itself (`UnionAll isa UnionAll` is false).
    (P isa UnionAll || P == UnionAll) && return Dual # P is abstract, tangent type unknown.

    # Union Splitting
    if P <: Tuple && !all(isconcretetype, (P.parameters...,))
        field_types = (P.parameters...,)
        union_fields = _findall(Base.Fix2(isa, Union), field_types)

        # If there is exactly one Union field, split it to help the compiler
        if length(union_fields) == 1 &&
            all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_union_tuple_type(field_types)
            return Union{dual_type(P_split.a),dual_type(P_split.b)}
        end
    end

    return isconcretetype(P) ? Dual{P,tangent_type(P)} : Dual
end

function dual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? Dual{Type{P},NoTangent} : Dual{_typeof(p),NoTangent}
end

_primal(x) = x
_primal(x::Dual) = primal(x)

"""
    verify_dual_type(x::Dual)

Check that the type of `tangent(x)` is the tangent type of the type of `primal(x)`.
"""
verify_dual_type(x::Dual) = tangent_type(typeof(primal(x))) == typeof(tangent(x))

function error_if_incorrect_dual_types(duals::Vararg{Dual,N}) where {N}
    correct_types = map(verify_dual_type, duals)
    if !all(correct_types)
        primals = map(primal, duals)
        tangents = map(tangent, duals)
        throw(ArgumentError("""
        Tangent types do not match primal types:
          - primal types:           $(map(typeof, primals))
          - provided tangent types: $(map(typeof, tangents))
          - required tangent types: $(map(tangent_type, map(typeof, primals)))
        """))
    end
end

@inline uninit_dual(x::P) where {P} = Dual(x, uninit_tangent(x))

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function Dual(x::Type{P}, dx::NoTangent) where {P}
    return Dual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(x, dx)
end

# ===========================================================================
# ADNext-specific integration of `Dual` with the tangent system.
# ===========================================================================

# Property aliases used across ADNext's engine and tests. `x`/`y`/`z` alias the primal field
# (`primal`); `dx`/`dy`/`dz` alias the tangent field (`tangent`). Any other symbol (including the
# real field names `:primal`/`:tangent`) falls through to `getfield`, so `primal(d)`/`tangent(d)`
# keep working.
function Base.getproperty(d::Dual, s::Symbol)
    if s === :x || s === :y || s === :z
        getfield(d, :primal)
    elseif s === :dx || s === :dy || s === :dz
        getfield(d, :tangent)
    else
        getfield(d, s)
    end
end
Base.propertynames(::Dual) = (:primal, :tangent, :x, :y, :z, :dx, :dy, :dz)

# A `Dual` is its own tangent type. This is the key to preserving ADNext's higher-order
# (Option A) nesting under the Mooncake tangent system: after the engine peels one `Dual` level
# off a primal that is itself a `Dual`, the tangent it must produce is again a `Dual` — because we
# define it to be. This reproduces every documented order-≥2 seed and satisfies Mooncake's
# invariant `Dual{P,T} ⟹ T == tangent_type(P)` (checked, e.g. `tangent_type(Dual{Float64,Float64})
# == Dual{Float64,Float64}`, `tangent_type(Dual{typeof(sin),NoTangent}) == Dual{typeof(sin),
# NoTangent}`). A `Dual` therefore keeps *same-typed-shadow* semantics, unlike a general struct
# (which strips to a `Tangent`/`MutableTangent`).
tangent_type(::Type{P}) where {P<:Dual} = P

# Type-level field accessors used by the dualization engine (replacing the old
# `primal_type`/`tangent_type`-on-`Dual` accessors).
_dual_primal_type(::Type{Dual{P,T}}) where {P,T} = P
_dual_tangent_type(::Type{Dual{P,T}}) where {P,T} = T

# Same-typed zero tangent for a `Dual` carrier: differentiable leaves are zeroed, while
# non-differentiable singletons (functions, `NoTangent`) are carried through unchanged so the
# result stays a value of the (self) tangent type `typeof(d)`. This mirrors the old `struct_zero`
# behavior for `Dual`s; it is rarely reached (Duals are constructed by the transform, not present
# as primal constants), but keeps `zero_tangent` coherent if a `Dual`-typed value ever flows into
# the engine's constant / non-differentiable path.
_carrier_zero(x::IEEEFloat) = zero(x)
_carrier_zero(::NoTangent) = NoTangent()
_carrier_zero(x::Dual) = zero_tangent_internal(x, NoCache())
# For any other carried field: a singleton (function/constant) carries through; a self-tangent type
# (`tangent_type(X) === X`, e.g. `Vector{Float64}`) takes its ordinary `zero_tangent`. A non-self-
# tangent type (`tangent_type(X) !== X`, e.g. a struct/closure with a `Float64` field, whose tangent
# is a `Tangent`) *cannot* be represented in a `Dual`'s same-typed field, so a same-typed zero does
# not exist. This is a fundamental limit of the self-tangent `Dual` scheme used for higher-order
# forward mode — surface it as a clear error rather than the cryptic `%new` `TypeError` it would
# otherwise become downstream. (Differentiating such a value at order ≥2 — e.g. a closure with
# differentiable captures under a nested `D` — is what lands here.)
_carrier_zero(x::X) where {X} =
    Base.issingletontype(X) ? x :
    tangent_type(X) === X    ? zero_tangent(x) :
    error("ADNext: cannot build a higher-order zero tangent for a `Dual` carrying a value of type ",
          X, " (whose tangent type ", tangent_type(X), " differs from itself). The self-tangent ",
          "`Dual` scheme used for higher-order forward mode requires each carried type to be its own ",
          "tangent type — true for scalars and arrays, but not for a struct/closure with ",
          "differentiable fields — so differentiating such a value at order ≥2 is unsupported.")

@generated function zero_tangent_internal(d::D, ::MaybeCache) where {D<:Dual}
    return Expr(:new, :D, (:(_carrier_zero(getfield(d, $i))) for i in 1:fieldcount(D))...)
end
