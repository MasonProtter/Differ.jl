# Hand-written frule!!/rrule!! for fancy/logical indexing. See ISSUES.md #30.
#
# Scalar/multi-dimensional Int-indexed getindex/setindex! on a concrete `Array` already works
# without any rule here: it lowers to `memoryrefnew`/`memoryrefget`/`memoryrefset!`, which the
# dualization engine handles natively (see `differ-architecture`'s status notes). This file covers
# the two things that don't: mask (logical) indexing and index-vector indexing, both of which call
# into Base's real `LogicalIndex`/`_unsafe_getindex`/checkbounds machinery — not dualizable — so
# every rule below is written as an explicit loop over primal/tangent data instead of letting the
# engine try to derive through Base's implementation.
#
# NOTE on scope: a generic `AbstractArray`-dispatched scalar-Int getindex/setindex! rule (to cover
# a custom array subtype whose indexing doesn't inline to `memoryrefget`/`memoryrefset!`) was
# attempted and dropped. Once such a rule for `Base.getindex`/`Base.setindex!` exists,
# `src_inlining_policy` (forward_interp.jl / reverse_interp.jl) blocks Julia from inlining *any*
# call matching it — including deep inside unrelated Base library code (e.g. `Base._mapreduce`'s
# own body, reached while differentiating `sum(transpose(M))`). That changes what Julia's optimizer
# constant-propagates there, which was empirically confirmed to newly trigger a pre-existing,
# unrelated reverse-mode gap (`_static_recursible_call` in reverse_interp.jl not handling a
# `Core.Const`-narrowed callee argument — the reverse-mode counterpart of an already-fixed
# forward-mode issue, ISSUES.md #34) and broke `test_linalg_rules.jl`'s "transpose/adjoint already
# work with NO new rule" test. Fixing that gap is out of this file's scope (reverse_interp.jl), so
# the generic rule was dropped rather than risk that class of collateral breakage elsewhere.

# ===========================================================================
# Part 1: forward mode (frule!!)
# ===========================================================================

# ---------------------------------------------------------------------------
# Logical (mask) indexing: `A[mask]` for `mask::AbstractArray{Bool}`. Returns the elements of `A`
# where `mask` is `true`, as a 1-D array (matching `Base.getindex`'s own shape convention for
# logical indexing, regardless of `A`'s dimensionality). `eachindex(a, mask)` does the axes-match
# check Base's own `checkbounds` would (raises `DimensionMismatch` on mismatched axes).
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
# which is also `<:Integer` in Julia) so this doesn't overlap/become ambiguous with the mask rule
# above.
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
# (length `count(mask)`/`length(idxvec)`). `Base.setindex!` returns the mutated container itself.
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

# ===========================================================================
# Part 2: reverse mode (rrule!!)
#
# Both getindex rules below allocate the result's own fdata array (`dy`) as a fresh zero and hand
# it back as part of `ycd` — exactly like the derived array-allocation path does for a freshly
# `%new`'d array. Whatever later code does with the gathered array `y` accumulates into `dy`
# through the ordinary tracked-array machinery; the pullback (called once the whole forward sweep
# is done, during the backward sweep) reads `dy`'s now-populated contents and scatters them back
# into the source array's own fdata (`dA`) via `increment!!` — required for the index-vector case,
# where the same source index can repeat (e.g. `A[[1,1,2]]`), so a later occurrence must accumulate
# into `dA[i]`, not overwrite an earlier occurrence's contribution.
#
# NOTE: reachable only via a direct top-level `gradient(Base.getindex, A, mask_or_idxvec)` call, not
# through a user-defined wrapper (`f(A, m) = A[m]`) — `_static_recursible_call`'s static eligibility
# gate (reverse_interp.jl) rejects *any* surviving high-level call whose result carries non-trivial
# fdata ("array-valued results from a recursive call are a separate, not-yet-supported feature"),
# before hand-rule resolution is even attempted. That gate lives outside this file's scope, so a
# wrapped call bails with that message regardless of the rule below existing. Tested here by calling
# `gradient`/`rrule!!` on `Base.getindex` directly.
# ===========================================================================

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
