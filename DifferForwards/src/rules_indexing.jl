# Hand-written frule!! for fancy/logical indexing. See ISSUES.md #30. Reverse-mode rrule!!s for
# the same functions live in DifferReverse/src/rules_indexing.jl.
#
# Scalar/multi-dimensional Int-indexed getindex/setindex! on a concrete `Array` already works
# without any rule here: it lowers to `memoryrefnew`/`memoryrefget`/`memoryrefset!`, which the
# dualization engine handles natively (see `differ-architecture`'s status notes). This file covers
# the two things that don't: mask (logical) indexing and index-vector indexing, both of which call
# into Base's real `LogicalIndex`/`_unsafe_getindex`/checkbounds machinery — not dualizable — so
# every rule below is an explicit loop over primal/tangent data instead.

# ---------------------------------------------------------------------------
# Logical (mask) indexing: `A[mask]` for `mask::AbstractArray{Bool}`. Returns the elements where
# `mask` is true, as a 1-D array (matching `Base.getindex`'s shape convention for logical indexing,
# regardless of `A`'s dimensionality). `eachindex(a, mask)` does the same axes-match check Base's
# `checkbounds` would (raises `DimensionMismatch` on mismatched axes).
# ---------------------------------------------------------------------------
function frule!!(
    ::Dual{typeof(Base.getindex)}, Acd::Dual{A}, maskcd::Dual{M}
) where {T,N,A<:Array{T,N},M<:AbstractArray{Bool}}
    a, da = extract(Acd)
    mask = primal(maskcd)
    n = count(mask)
    y = Vector{T}(undef, n)
    dy = Vector{tangent_type(T)}(undef, n)
    k = 0
    for i in eachindex(a, mask)
        if mask[i]
            k += 1
            y[k] = a[i]
            dy[k] = da[i]
        end
    end
    return Dual(y, dy)
end

# ---------------------------------------------------------------------------
# Index-vector indexing: `A[idxvec]` for `idxvec::AbstractVector{<:Integer}` — a gather,
# `y[k] = A[idxvec[k]]`. Element type restricted to `Union{Signed,Unsigned}` (excluding `Bool`,
# which is also `<:Integer` in Julia) so this doesn't overlap with the mask rule above.
# ---------------------------------------------------------------------------
function frule!!(
    ::Dual{typeof(Base.getindex)}, Acd::Dual{A}, idxcd::Dual{I}
) where {T,N,A<:Array{T,N},I<:AbstractVector{<:Union{Signed,Unsigned}}}
    a, da = extract(Acd)
    idxvec = primal(idxcd)
    n = length(idxvec)
    y = Vector{T}(undef, n)
    dy = Vector{tangent_type(T)}(undef, n)
    for k in eachindex(idxvec)
        i = idxvec[k]
        y[k] = a[i]
        dy[k] = da[i]
    end
    return Dual(y, dy)
end

# ---------------------------------------------------------------------------
# `setindex!` companions: `A[mask] = v` / `A[idxvec] = v`, `v` an array of replacement values
# (length `count(mask)`/`length(idxvec)`). `Base.setindex!` returns the mutated container.
# (No reverse-mode rule for `setindex!` — see DifferReverse/src/rules_indexing.jl, which only
# needs `getindex` rules; `setindex!` mutation is handled by the general array-mutation machinery
# there instead.)
# ---------------------------------------------------------------------------
function frule!!(
    ::Dual{typeof(Base.setindex!)}, Acd::Dual{A}, vcd::Dual{V}, maskcd::Dual{M}
) where {T,N,A<:Array{T,N},V<:AbstractArray,M<:AbstractArray{Bool}}
    a, da = extract(Acd)
    v, dv = extract(vcd)
    mask = primal(maskcd)
    length(v) == count(mask) || throw(DimensionMismatch(
        "setindex!(A, v, mask): length(v) = $(length(v)) does not match count(mask) = $(count(mask))"))
    k = 0
    for i in eachindex(a, mask)
        if mask[i]
            k += 1
            a[i] = v[k]
            da[i] = dv[k]
        end
    end
    return Dual(a, da)
end

function frule!!(
    ::Dual{typeof(Base.setindex!)}, Acd::Dual{A}, vcd::Dual{V}, idxcd::Dual{I}
) where {T,N,A<:Array{T,N},V<:AbstractArray,I<:AbstractVector{<:Union{Signed,Unsigned}}}
    a, da = extract(Acd)
    v, dv = extract(vcd)
    idxvec = primal(idxcd)
    length(v) == length(idxvec) || throw(DimensionMismatch(
        "setindex!(A, v, idxvec): length(v) = $(length(v)) does not match " *
        "length(idxvec) = $(length(idxvec))"))
    for k in eachindex(idxvec)
        i = idxvec[k]
        a[i] = v[k]
        da[i] = dv[k]
    end
    return Dual(a, da)
end
