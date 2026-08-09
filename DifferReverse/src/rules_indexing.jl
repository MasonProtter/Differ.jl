# Hand-written rrule!! for fancy/logical indexing (getindex only).
# Forward-mode frule!!s (including setindex!) live in DifferForwards/src/rules_indexing.jl.
#
# Both getindex rules allocate the result's own fdata array (`dy`) as a fresh zero, handed back as
# part of `ycd`. Later reads of `y` accumulate into `dy` through the ordinary tracked-array
# machinery; the pullback then scatters `dy`'s contents back into the source array's fdata (`dA`)
# via `increment!!` — needed for the index-vector case, where a repeated source index (e.g.
# `A[[1,1,2]]`) must accumulate rather than overwrite.
#
# Reachable only via a direct top-level `rev_gradient(Base.getindex, A, mask_or_idxvec)` call, not
# through a user-defined wrapper (`f(A, m) = A[m]`): `_static_recursible_call`'s static eligibility
# gate (`reverse_interp.jl`) rejects any high-level call whose result carries non-trivial fdata
# before hand-rule resolution is attempted, regardless of the rule below. Tested here by calling
# `rev_gradient`/`rrule!!` on `Base.getindex` directly.

struct MaskGetindexPullback{DA<:Array,M<:AbstractArray{Bool},DY<:Array}
    dA::DA
    mask::M
    dy::DY
end
function (pb::MaskGetindexPullback)(seed)
    dA, mask, dy = pb.dA, pb.mask, pb.dy
    k = 0
    for i in eachindex(dA, mask)
        if mask[i]
            k += 1
            dA[i] = increment!!(dA[i], dy[k])
        end
    end
    return (NoRData(), NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(Base.getindex),NoFData}, ::AbstractCtx,
    Acd::CoDual{X,X}, maskcd::CoDual{M}
) where {N,X<:Array{<:IEEEFloat,N},M<:AbstractArray{Bool}}
    a, da = extract(Acd)
    mask = primal(maskcd)
    n = count(mask)
    y = Vector{eltype(X)}(undef, n)
    k = 0
    for i in eachindex(a, mask)
        if mask[i]
            k += 1
            y[k] = a[i]
        end
    end
    dy = zero_tangent(y)
    return CoDual(y, dy), MaskGetindexPullback(da, mask, dy)
end

struct IdxvecGetindexPullback{DA<:Array,I<:AbstractVector,DY<:Array}
    dA::DA
    idxvec::I
    dy::DY
end
function (pb::IdxvecGetindexPullback)(seed)
    dA, idxvec, dy = pb.dA, pb.idxvec, pb.dy
    for k in eachindex(idxvec)
        i = idxvec[k]
        dA[i] = increment!!(dA[i], dy[k])
    end
    return (NoRData(), NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(Base.getindex),NoFData}, ::AbstractCtx,
    Acd::CoDual{X,X}, idxcd::CoDual{I}
) where {N,X<:Array{<:IEEEFloat,N},I<:AbstractVector{<:Union{Signed,Unsigned}}}
    a, da = extract(Acd)
    idxvec = primal(idxcd)
    n = length(idxvec)
    y = Vector{eltype(X)}(undef, n)
    for k in eachindex(idxvec)
        y[k] = a[idxvec[k]]
    end
    dy = zero_tangent(y)
    return CoDual(y, dy), IdxvecGetindexPullback(da, idxvec, dy)
end
