"""
    Inactive

Marks a value as *held constant*: no derivative is propagated to or from it. A carrier
(`Dual`/`CoDual`) puts `Inactive()` in its shadow slot to say so, and the engines put it in an
aggregate's shadow slot for a constant component.

Distinct from [`NoTangent`](@ref), which says the *type* has no tangent space. The two cannot be
merged: `fdata_type(NoTangent) === NoFData`, which is also an active `Float64`'s fdata, so
`NoTangent` cannot encode constancy once a value is nested. `Inactive` is preserved by both
`fdata` and `rdata`, so it survives into aggregates.

`Inactive` is a *strong zero*, and the two directions are not the same operation:

- `increment!!(Inactive(), y) === Inactive()` — accumulating into a constant discards, by
  definition of having declared it constant.
- `increment!!(x, Inactive()) === x` — a constant contributes nothing, leaving `x` untouched.

So `increment!!` is not commutative in the presence of `Inactive`: the first slot is the
accumulator that owns storage, the second is a contribution. Both follow the general rule that
`increment!!` returns a value of the accumulator's type.

`NoTangent` deliberately keeps *no* absorbing arm, so a mis-analysed active value still raises a
`MethodError` rather than silently dropping a gradient. `Inactive` absorbs only because it is
written exactly where constancy was declared.

Because `Inactive` is a singleton, the no-op arms cost nothing: each activity signature compiles
separately, with the dead work erased.
"""
struct Inactive end

"""
    isactive(dx)

Whether the shadow `dx` carries a derivative — `false` exactly for `Inactive()`.

Decidable from the shadow's type in every position, which is the point of `Inactive` being its own
type: an active `Float64`'s shadow is `NoFData()`, indistinguishable from what a constant one would
carry if constancy were encoded as an empty fdata.
"""
isactive(@nospecialize dx) = !isa(dx, Inactive)

"""
    @ifactive dx expr

`expr` if the shadow `dx` is active, `NoRData()` otherwise. `Inactive` is a singleton so the test
folds away — the two activity specialisations of a rule compile separately, with no `Union` result.

`NoRData()`, not `Inactive()`: activity has to be visible in the *fdata* half, where a slot declared
at the primal-derived type would otherwise be a type lie. An inactive argument's rdata slot carries
nothing either way, and `NoRData()` is what the derived path emits for one, so hand rules and derived
rules return the same shape. `tangent(Inactive(), NoRData())` reassembles to `Inactive()` regardless.

For a pullback component in a hand-written `rrule!!`:

    mul_pullback(dz) = (NoRData(), @ifactive(dx, dz * y), @ifactive(dy, dz * x))
"""
macro ifactive(dx, expr)
    return :(isactive($(esc(dx))) ? $(esc(expr)) : NoRData())
end

"""
    fdata_shadow_type(P::Type)

The types a *reverse-mode* shadow for a primal of type `P` may legally have: its ordinary fdata
type, or `Inactive` if the value is held constant. This is what a `CoDual`'s second slot admits.

**A validity predicate, never a declaration.** Use it in `<:` checks and rule-signature
constraints. Every declared slot, field, comms item and SSA type must stay concrete — the engines
pick the concrete alternative from their own per-value activity, so no union ever reaches a hot
path.

See [`tangent_shadow_type`](@ref) for the forward-mode half.
"""
fdata_shadow_type(::Type{P}) where {P} = Union{fdata_type(tangent_type(P)),Inactive}

"""
    tangent_shadow_type(P::Type)

The types a *forward-mode* shadow for a primal of type `P` may legally have: its ordinary tangent
type, or `Inactive` if the value is held constant. This is what a `Dual`'s second slot admits.

Forward mode carries the whole tangent rather than the fdata half, which is the only difference
from [`fdata_shadow_type`](@ref). The same "validity predicate, never a declaration" rule applies.
"""
tangent_shadow_type(::Type{P}) where {P} = Union{tangent_type(P),Inactive}

"""
    shadow_type(P::Type)

Deprecated alias for [`fdata_shadow_type`](@ref). Named before forward mode carried `Inactive`,
when there was only one shadow half to describe.
"""
shadow_type(::Type{P}) where {P} = fdata_shadow_type(P)

# `Inactive` is its own fdata and its own rdata — that is what lets it survive into an aggregate's
# shadow, which `NoTangent` cannot do.
fdata_type(::Type{Inactive}) = Inactive
rdata_type(::Type{Inactive}) = Inactive
tangent_type(::Type{Inactive}) = Inactive

fdata(::Inactive) = Inactive()
rdata(::Inactive) = Inactive()
# An inactive fdata half determines the whole tangent whatever the rdata half is: a derived
# pullback emits a `NoRData()` literal for an inactive slot, a hand rule's `@ifactive` an
# `Inactive()`, and both reassemble to the same thing.
tangent(::Inactive, @nospecialize(_)) = Inactive()

_copy(::Inactive) = Inactive()

# Accumulation. The third method is not redundant: without it `(Inactive, Inactive)` is ambiguous
# between the first two.
increment_internal!!(::IncCache, ::Inactive, @nospecialize(_)) = Inactive()
increment_internal!!(::IncCache, x, ::Inactive) = x
increment_internal!!(::IncCache, ::Inactive, ::Inactive) = Inactive()

# The `increment!!` entry point is homogeneously typed (`increment!!(x::T, y::T)`), so an
# `Inactive` on one side alone matches nothing without these.
increment!!(::Inactive, @nospecialize(_)) = Inactive()
increment!!(x, ::Inactive) = x
increment!!(::Inactive, ::Inactive) = Inactive()
# `ZeroRData` is the additive *identity* rather than an absorbing element, so these two disagree
# on which side wins and have to be spelled out.
increment!!(::Inactive, ::ZeroRData) = Inactive()
increment!!(::ZeroRData, ::Inactive) = ZeroRData()

# A mixed-activity aggregate's shadow is narrower than the seed a caller builds from the primal
# type alone (`increment!!(tangent(ycd), fdata(seed))`), so the two tuples' types differ slot for
# slot and the homogeneous `x::T, y::T` arms do not apply. Accumulate structurally instead and let
# each slot's own arm decide — an `Inactive` slot discards, as it would anywhere else. Result keeps
# the accumulator's type.
@generated function increment_internal!!(c::IncCache, x::Tuple, y::Tuple)
    fieldcount(x) == fieldcount(y) || return :(throw(ArgumentError(
        "cannot increment a $(fieldcount(x))-element tuple by a $(fieldcount(y))-element one")))
    return Expr(:call, :tuple,
                (:(increment_internal!!(c, x[$n], y[$n])) for n in 1:fieldcount(x))...)
end
# Cache keyed on `typeof(x)`, not `y`: the `MutableTangent` arm of `increment_internal!!` only
# checks/records against the accumulator side (`haskey(c, x)`), so aliasing that matters for
# double-counting lives in `x`'s structure — `y` having a different type slot-for-slot is exactly
# what makes this the mixed arm, so `x`'s own type is the only thing there is to key on.
increment!!(x::Tuple, y::Tuple) =
    increment_internal!!(_inc_cache(require_tangent_cache(typeof(x))), x, y)
# Diagonal, so more specific than the arm above: a homogeneously-typed tuple keys off the static
# `T` directly (equal to `typeof(x)` here) instead of falling through to the mixed arm.
increment!!(x::T, y::T) where {T<:Tuple} =
    increment_internal!!(_inc_cache(require_tangent_cache(T)), x, y)

increment_rdata!!(::Inactive, @nospecialize(_)) = Inactive()
increment_rdata!!(t, ::Inactive) = t
increment_rdata!!(::Inactive, ::Inactive) = Inactive()

# Field-wise accumulation, both directions. The `Union{Symbol,Int,Val}` index shapes mirror the
# `NoTangent` fallback in `tangents.jl`.
for I in (Symbol, Int, Val)
    @eval increment_field!!(::Inactive, @nospecialize(_), ::$I) = Inactive()
end
increment_field!!(x::Tangent, ::Inactive, ::Val) = x
increment_field!!(x::MutableTangent, ::Inactive, ::Val) = x
increment_field_rdata!(dx::MutableTangent, ::Inactive, ::Val) = dx
increment_field_rdata!(dx::MutableTangent, ::Inactive, ::Int) = dx
increment_field_rdata!(::Inactive, @nospecialize(_), ::Val) = Inactive()
increment_field_rdata!(::Inactive, @nospecialize(_), ::Int) = Inactive()

set_to_zero_internal!!(::SetToZeroCache, ::Inactive) = Inactive()

instantiate(::Inactive) = Inactive()

# Validation: an `Inactive` shadow is valid for any primal, so there is nothing to check.
verify_fdata_type(::Type, ::Type{Inactive}) = nothing
verify_rdata_type(::Type, ::Type{Inactive}) = nothing
__verify_fdata_value(::IdDict{Any,Nothing}, @nospecialize(_), ::Inactive) = nothing
_verify_rdata_value(@nospecialize(_), ::Inactive) = nothing

# Test-harness arms (finite differencing, tangent/primal round-trips): an inactive slot takes no
# part in any of them.
_scale_internal(::MaybeCache, ::Float64, ::Inactive) = Inactive()
_dot_internal(::MaybeCache, ::Inactive, @nospecialize(_)) = 0.0
_dot_internal(::MaybeCache, @nospecialize(_), ::Inactive) = 0.0
_dot_internal(::MaybeCache, ::Inactive, ::Inactive) = 0.0
_add_to_primal_internal(::MaybeCache, x, ::Inactive, ::Bool) = x
