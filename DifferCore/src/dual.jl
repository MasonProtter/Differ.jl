# Portions of this file are derived from Mooncake.jl (https://github.com/chalk-lab/Mooncake.jl),
# Copyright (c) 2024 Will Tebbutt and Hong Ge, licensed under the MIT License.

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

`T` is one of two things:

- `tangent_type(P)`, the ordinary case — forward mode carries the whole tangent, not the fdata half.
- [`Inactive`](@ref), marking the value as held constant by the caller. Legal for any `P`;
  `tangent_shadow_type(P)` is the pair of them.

`Dual{P,Inactive}` reaches `frule!!` only in argument position, but it does reach it: an inactive
value consumed by an active call is handed to the rule as-is, so a hand rule can skip the term
instead of multiplying by a zero. Consumers that read a tangent as a value — intrinsic and builtin
rules, phi-likes, `%new`, the return — get a materialised zero at the value's definition instead.
A rule never *returns* `Inactive` (see `dualize_to_ircode`'s `frule_split!`).
"""
struct Dual{P,T}
    primal::P
    tangent::T
    function Dual{P, T}(x, dx) where {P, T}
        if T === Inactive
            return new{P, Inactive}(convert(P, x), Inactive())
        end
        if !(tangent_type(P) == T)
            @outline P T throw(ArgumentError(
                "Invalid Dual{P,T} construction for primal type P=$P\n\tgot tangent type T=$T\n\tDiffer requires that tangent_type(P) == T, or T === Inactive"
            ))
        end
        new{P, T}(convert(P, x), convert(T, dx))
    end
    # Must precede the general arm: `as_tangent` below would try to convert an `Inactive` into a
    # tangent of `P` and then throw, rather than recognising it as a constancy marker.
    Dual(x::P, dx::Inactive) where {P} = new{P, Inactive}(x, dx)
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

Returns the 2-tuple `x.x, x.dx`.
"""
extract(x::Dual) = primal(x), tangent(x)

zero_dual(x) = Dual(x, zero_tangent(x))
randn_dual(rng::AbstractRNG, x) = Dual(x, randn_tangent(rng, x))

function dual_type(::Type{P}) where {P}
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
verify_dual_type(x::Dual) =
    tangent(x) isa Inactive || tangent_type(_typeof(primal(x))) == typeof(tangent(x))

function error_if_incorrect_dual_types(duals::Vararg{Dual,N}) where {N}
    correct_types = map(verify_dual_type, duals)
    if !all(correct_types)
        primals = map(primal, duals)
        tangents = map(tangent, duals)
        throw(ArgumentError("""
        Tangent types do not match primal types:
          - primal types:           $(map(_typeof, primals))
          - provided tangent types: $(map(typeof, tangents))
          - required tangent types: $(map(tangent_type, map(_typeof, primals)))
        """))
    end
end

@inline uninit_dual(x::P) where {P} = Dual(x, uninit_tangent(x))

# A shadow that contributes nothing to a derivative, for the two distinct reasons a hand rule has to
# treat alike: `NoTangent` (the type has no tangent space, e.g. an `Int` exponent) and `Inactive`
# (the caller declared this particular value constant). `isactive` knows only the second.
_inert(@nospecialize dx) = isa(dx, NoTangent) || isa(dx, Inactive)

# `_require_active_dest` (destination-activity refusal for a mutating rule) is defined in
# `DifferCore/src/inactive.jl` and shared with `DifferReverse`.

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function Dual(x::Type{P}, dx::NoTangent) where {P}
    return Dual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(x, dx)
end
function Dual(x::Type{P}, dx::Inactive) where {P}
    return Dual{@isdefined(P) ? Type{P} : typeof(x),Inactive}(x, dx)
end

# ===========================================================================
# Differ-specific integration of `Dual` with the tangent system.
# ===========================================================================

# Property aliases used across Differ's engine and tests. `x`/`y`/`z` alias the primal field
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

# A `Dual` is its own tangent type whenever both of its fields can live in a same-typed shadow:
# peeling one `Dual` level off a primal that is itself a `Dual` must again produce a `Dual` for
# higher-order nesting to work. A field qualifies when its tangent type is itself (the shadow slot
# holds the tangent) or when it has no tangent space at all (the shadow slot carries the primal
# through unchanged — what lets a `Dual{typeof(sin),NoTangent}` be re-dualized at higher order).
#
# A field that is neither — a closure with differentiable captures, whose tangent is a `Tangent` —
# has no same-typed representation, so that `Dual` gets the ordinary two-field struct tangent
# instead. Reached at order >= 2, where the outer pass dualizes inner-pass code that builds such a
# `Dual`. Either way the result satisfies Mooncake's invariant `Dual{P,T} ⟹ T == tangent_type(P)`.
@foldable _dual_selfsim_field(::Type{F}) where {F} = (T = tangent_type(F); T === F || T === NoTangent)

@foldable function tangent_type(::Type{Dual{P,T}}) where {P,T}
    isconcretetype(Dual{P,T}) || return Dual{P,T}
    (_dual_selfsim_field(P) && _dual_selfsim_field(T)) && return Dual{P,T}
    return Tangent{NamedTuple{(:primal, :tangent),Tuple{tangent_type(P),tangent_type(T)}}}
end
tangent_type(::Type{D}) where {D<:Dual} = D

# A `Dual` can sit inside an ordinary tangent — DifferentiationInterface keeps an inner-pass `Dual`
# in its prep object, which the outer pass meets as a code constant — so `fdata_type` is asked about
# one (via `inactive_constant_type`) and must answer rather than aborting the whole compilation with
# "Unhandled type". Every slot of a self-similar `Dual` holds either a tangent or a primal carried
# through a `NoTangent` slot, the latter contributing no forward-pass storage.
#
# A `Dual` that does hold fdata has no answer: `FData`/`RData` are `Tangent`-shaped, so a split one
# could not be told apart from a `Tangent` with the same two field names on the way back. It stays
# loud, as does `rdata_type`, which is left undefined — a `Dual` has no usable reverse half.
_dual_slot_fdata_type(::Type{F}) where {F} = tangent_type(F) === NoTangent ? NoFData : fdata_type(F)

function fdata_type(::Type{Dual{P,T}}) where {P,T}
    tangent_type(Dual{P,T}) === Dual{P,T} ||
        error("Differ: `$(Dual{P,T})` is not a tangent type (its tangent is " *
              "`$(tangent_type(Dual{P,T}))`), so it has no fdata.")
    Fp = _dual_slot_fdata_type(P)
    Ft = _dual_slot_fdata_type(T)
    (Fp === NoFData && Ft === NoFData) && return NoFData
    (Fp === Any || Ft === Any) && return Any        # abstract slot: unknown, as elsewhere
    error("Differ: `$(Dual{P,T})` is a tangent holding fdata ($Fp, $Ft), which has no " *
          "`FData`/`RData` split.")
end

# Type-level field accessors used by the dualization engine (replacing the old
# `primal_type`/`tangent_type`-on-`Dual` accessors).
_dual_primal_type(::Type{Dual{P,T}}) where {P,T} = P
_dual_tangent_type(::Type{Dual{P,T}}) where {P,T} = T

# Same-typed zero tangent for a `Dual` carrier: differentiable leaves are zeroed, values with no
# tangent space pass through unchanged, so the result stays a value of the same tangent type
# `typeof(d)`. Rarely reached — Duals are built by the transform, not present as primal constants.
_carrier_zero(x::IEEEFloat) = zero(x)
_carrier_zero(::NoTangent) = NoTangent()
_carrier_zero(::Inactive) = Inactive()
_carrier_zero(x::Dual) = zero_tangent_internal(x, NoCache())
# A field with no tangent space carries its primal value through, matching what the same-typed
# shadow `%new` does. Anything else has no same-typed zero, and only `_dual_selfsim_field` decides
# which `Dual`s take this path at all, so reaching the error means those two disagree.
_carrier_zero(x::X) where {X} =
    Base.issingletontype(X)       ? x :
    tangent_type(X) === X         ? zero_tangent(x) :
    tangent_type(X) === NoTangent ? x :
    error("Differ: cannot build a same-typed zero tangent for a `Dual` carrying a value of type ",
          X, " (whose tangent type ", tangent_type(X), " is neither itself nor `NoTangent`).")

@generated function zero_tangent_internal(d::D, c::MaybeCache) where {D<:Dual}
    # The `tangent_type(D) === D` test stays in the emitted code, not the generator body: a
    # generator's world is pinned, and this must see whatever `tangent_type` methods exist at the
    # call site. It folds away during inference.
    selfsim = Expr(:new, :D, (:(_carrier_zero(getfield(d, $i))) for i in 1:fieldcount(D))...)
    return quote
        tangent_type(D) === D && return $selfsim
        return tangent_type(D)(backing_type(D)((zero_tangent_internal(getfield(d, 1), c),
                                                zero_tangent_internal(getfield(d, 2), c))))
    end
end
