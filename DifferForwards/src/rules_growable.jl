# Hand-written `frule!!`s for Base's array growth/shrink helpers. Every growable `Vector` operation
# (`push!`, `pop!`, `pushfirst!`, `popfirst!`, `insert!`, `deleteat!`, `resize!`, `empty!`,
# `append!`, `prepend!`) funnels through these six. `src_inlining_policy` keeps a hand-ruled callee
# from being inlined away, so ruling them here is what keeps `Core.memoryrefoffset` and `setfield!`
# on an `Array`'s `:ref`/`:size` out of dualized IR entirely.
#
# They move structure, not values: only the length changes, and surviving elements keep their
# tangents because the shadow undergoes the identical operation through its own layout. Primal and
# shadow therefore never need matching capacity or offset.
#
# Newly exposed slots are left uninitialized on both sides, same as `Core.memorynew`'s shadow: the
# mirrored `memoryrefset!` that fills the primal slot fills the shadow slot too, before any read.
#
# An `Inactive` array shadow is a no-op here rather than a refusal. These helpers carry no value, so
# skipping the shadow discards nothing; a later active write into a constant array is refused where
# the derivative actually flows, by `memoryrefset!`'s own destination check.

function frule!!(::Dual{typeof(Base._growend!)}, ad::Dual{V}, dd::Dual{<:Integer}) where {V<:Vector}
    a, da = extract(ad)
    d = primal(dd)
    Base._growend!(a, d)
    isactive(da) && Base._growend!(da, d)
    return Dual(nothing, NoTangent())
end

function frule!!(::Dual{typeof(Base._growbeg!)}, ad::Dual{V}, dd::Dual{<:Integer}) where {V<:Vector}
    a, da = extract(ad)
    d = primal(dd)
    Base._growbeg!(a, d)
    isactive(da) && Base._growbeg!(da, d)
    return Dual(nothing, NoTangent())
end

# `Base._growat!` returns whatever its last `setfield!` returned (`Union{Nothing,Tuple{Int}}`), and
# every caller discards it. Returning `nothing` keeps the result type concrete.
function frule!!(::Dual{typeof(Base._growat!)}, ad::Dual{V}, id::Dual{<:Integer},
                 dd::Dual{<:Integer}) where {V<:Vector}
    a, da = extract(ad)
    i, d = primal(id), primal(dd)
    Base._growat!(a, i, d)
    isactive(da) && Base._growat!(da, i, d)
    return Dual(nothing, NoTangent())
end

function frule!!(::Dual{typeof(Base._deleteend!)}, ad::Dual{V}, dd::Dual{<:Integer}) where {V<:Vector}
    a, da = extract(ad)
    d = primal(dd)
    Base._deleteend!(a, d)
    isactive(da) && Base._deleteend!(da, d)
    return Dual(nothing, NoTangent())
end

function frule!!(::Dual{typeof(Base._deletebeg!)}, ad::Dual{V}, dd::Dual{<:Integer}) where {V<:Vector}
    a, da = extract(ad)
    d = primal(dd)
    Base._deletebeg!(a, d)
    isactive(da) && Base._deletebeg!(da, d)
    return Dual(nothing, NoTangent())
end

function frule!!(::Dual{typeof(Base._deleteat!)}, ad::Dual{V}, id::Dual{<:Integer},
                 dd::Dual{<:Integer}) where {V<:Vector}
    a, da = extract(ad)
    i, d = primal(id), primal(dd)
    Base._deleteat!(a, i, d)
    isactive(da) && Base._deleteat!(da, i, d)
    return Dual(nothing, NoTangent())
end

# `sizehint!` only moves an array's spare capacity around; the length is unchanged and the shadow's
# own capacity is invisible to values, so the primal is hinted alone. Base implements the grow case
# as `_growend!` followed by a raw `setfield!(a, :size, …)` undoing the length, and that undo is not
# mirrored onto the shadow — without this rule the shadow array is left at the hinted length.
#
# Unlike the helpers above, `sizehint!` returns the array, so the result's shadow is the
# destination's. A constant array contributes nothing and nothing flows into it, so its zero tangent
# is materialised here rather than refused: the alternative would be returning `Inactive`, which no
# rule may do.
function frule!!(::Dual{typeof(Base.sizehint!)}, ad::Dual{V}, nd::Dual{<:Integer}) where {V<:Vector}
    a, da = extract(ad)
    Base.sizehint!(a, primal(nd))
    isactive(da) || return Dual(a, zero_tangent(a))
    return Dual(a, da)
end
