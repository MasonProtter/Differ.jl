# Hand-written rrule!! for fancy/logical indexing (getindex only).
# Forward-mode frule!!s (including setindex!) live in DifferForwards/src/rules_indexing.jl.
#
# Both getindex rules allocate the result's own fdata array (`dout`) as a fresh zero, handed back
# as part of the returned `CoDual`. Later reads of the result accumulate into `dout` through the
# ordinary tracked-array machinery; the pullback then scatters `dout`'s contents back into the
# source array's fdata (`dx`) via `increment!!` — needed for the index-vector case, where a
# repeated source index (e.g. `A[[1,1,2]]`) must accumulate rather than overwrite.
#
# Reachable only via a direct top-level `rev_gradient(Base.getindex, A, mask_or_idxvec)` call, not
# through a user-defined wrapper (`f(A, m) = A[m]`): `_static_recursible_call`'s static eligibility
# gate (`reverse_interp.jl`) rejects any high-level call whose result carries non-trivial fdata
# before hand-rule resolution is attempted, regardless of the rule below. Tested here by calling
# `rev_gradient`/`rrule!!` on `Base.getindex` directly.

function rrule!!(
    ::CoDual{typeof(Base.getindex),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{X,<:Union{X,Inactive}}, (; y)::CoDual{M}
) where {N,X<:Array{<:IEEEFloat,N},M<:AbstractArray{Bool}}
    xactive = isactive(dx)
    n = count(y)
    out = Vector{eltype(X)}(undef, n)
    k = 0
    for i in eachindex(x, y)
        if y[i]
            k += 1
            out[k] = x[i]
        end
    end
    dout = zero_tangent(out)
    function mask_getindex_pullback(_)
        # Named `kk`, not `k`: reusing `k` here would capture (and box) the forward pass's own `k`
        # above, rather than declaring a fresh local — `function`-in-`function` doesn't shadow.
        kk = 0
        if xactive
            for i in eachindex(dx, y)
                if y[i]
                    kk += 1
                    dx[i] = increment!!(dx[i], dout[kk])
                end
            end
        end
        return (NoRData(), NoRData(), NoRData())
    end
    return CoDual(out, dout), mask_getindex_pullback
end

function rrule!!(
    ::CoDual{typeof(Base.getindex),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{X,<:Union{X,Inactive}}, (; y)::CoDual{I}
) where {N,X<:Array{<:IEEEFloat,N},I<:AbstractVector{<:Union{Signed,Unsigned}}}
    xactive = isactive(dx)
    n = length(y)
    out = Vector{eltype(X)}(undef, n)
    for k in eachindex(y)
        out[k] = x[y[k]]
    end
    dout = zero_tangent(out)
    function idxvec_getindex_pullback(_)
        if xactive
            for k in eachindex(y)
                i = y[k]
                dx[i] = increment!!(dx[i], dout[k])
            end
        end
        return (NoRData(), NoRData(), NoRData())
    end
    return CoDual(out, dout), idxvec_getindex_pullback
end
