# Hand-written `rrule!!`s for Base's array growth/shrink helpers, the reverse-mode counterparts of
# `DifferForwards/src/rules_growable.jl`. Every growable `Vector` operation funnels through these
# six, and `src_inlining_policy` keeps a hand-ruled callee from being inlined away, so ruling them
# here is what keeps `Core.memoryrefoffset` and `setfield!` on an `Array`'s `:ref`/`:size` out of
# the carrier IR.
#
# The forwards half just performs the same operation on primal and shadow, each through its own
# layout. The pullback puts both back on the ref and length they had on entry, moving the
# accumulated gradient from the post-operation indexing to the pre-operation one. Restoring the ref
# is what the reverse engine needs beyond the length: comms items hold `MemoryRef` handles and
# re-derived refs taken at specific points of the forwards pass, and a grow that reallocated would
# otherwise leave them addressing memory nothing reads back.

# Undo a length change with one surviving run: `n` elements sitting at post-operation indices
# `p0+1 … p0+n` belong at pre-operation indices `q0+1 … q0+n`. Reading through the post indexing
# before `a` goes back on `old_ref` keeps this independent of which internal branch Base took; when
# nothing moved, source and destination are the same address and nothing is copied.
function _rewind_run!(a::Vector, old_ref, len::Int, p0::Int, q0::Int, n::Int)
    if n > 0
        src = p0 == 0 ? getfield(a, :ref) : Base.memoryrefnew(getfield(a, :ref), p0 + 1, false)
        dst = q0 == 0 ? old_ref : Base.memoryrefnew(old_ref, q0 + 1, false)
        src === dst || Base.unsafe_copyto!(dst, src, n)
    end
    setfield!(a, :ref, old_ref)
    setfield!(a, :size, (len,))
    return nothing
end

# Undo a length change with two surviving runs, each `(post_start, pre_start, count)`. An insert or
# an interior delete always moves data and the two runs can overlap each other, so the post-operation
# values are snapshotted before `a` goes back on `old_ref`.
function _rewind_runs!(a::Vector, old_ref, len::Int, runs::NTuple{2,NTuple{3,Int}})
    post = copy(a)
    setfield!(a, :ref, old_ref)
    setfield!(a, :size, (len,))
    for (ps, qs, n) in runs
        for k in 0:(n - 1)
            @inbounds a[qs + k] = post[ps + k]
        end
    end
    return nothing
end

function rrule!!(::CoDual{typeof(Base._growend!),NoFData}, ::AbstractCtx,
                 acd::CoDual{V,<:Union{Vector,Inactive}}, dcd::CoDual{<:Integer}) where {V<:Vector}
    a, da = extract(acd)
    d = Int(primal(dcd))
    len = length(a)
    old_ref = getfield(a, :ref)
    old_dref = isactive(da) ? getfield(da, :ref) : nothing
    Base._growend!(a, d)
    old_dref === nothing || Base._growend!(da, d)
    function growend_pullback(::NoRData)
        old_dref === nothing || _rewind_run!(da, old_dref, len, 0, 0, len)
        _rewind_run!(a, old_ref, len, 0, 0, len)
        return (NoRData(), NoRData(), NoRData())
    end
    return zero_fcodual(nothing), growend_pullback
end

function rrule!!(::CoDual{typeof(Base._growbeg!),NoFData}, ::AbstractCtx,
                 acd::CoDual{V,<:Union{Vector,Inactive}}, dcd::CoDual{<:Integer}) where {V<:Vector}
    a, da = extract(acd)
    d = Int(primal(dcd))
    len = length(a)
    old_ref = getfield(a, :ref)
    old_dref = isactive(da) ? getfield(da, :ref) : nothing
    Base._growbeg!(a, d)
    old_dref === nothing || Base._growbeg!(da, d)
    function growbeg_pullback(::NoRData)
        old_dref === nothing || _rewind_run!(da, old_dref, len, d, 0, len)
        _rewind_run!(a, old_ref, len, d, 0, len)
        return (NoRData(), NoRData(), NoRData())
    end
    return zero_fcodual(nothing), growbeg_pullback
end

# Base's `_growat!` returns whatever its last `setfield!` returned; every caller discards it, so the
# result is normalised to `nothing` to keep the carrier's result type concrete.
function rrule!!(::CoDual{typeof(Base._growat!),NoFData}, ::AbstractCtx,
                 acd::CoDual{V,<:Union{Vector,Inactive}}, icd::CoDual{<:Integer},
                 dcd::CoDual{<:Integer}) where {V<:Vector}
    a, da = extract(acd)
    i, d = Int(primal(icd)), Int(primal(dcd))
    len = length(a)
    old_ref = getfield(a, :ref)
    old_dref = isactive(da) ? getfield(da, :ref) : nothing
    Base._growat!(a, i, d)
    old_dref === nothing || Base._growat!(da, i, d)
    runs = ((1, 1, i - 1), (i + d, i, len - i + 1))
    function growat_pullback(::NoRData)
        old_dref === nothing || _rewind_runs!(da, old_dref, len, runs)
        _rewind_runs!(a, old_ref, len, runs)
        return (NoRData(), NoRData(), NoRData(), NoRData())
    end
    return zero_fcodual(nothing), growat_pullback
end

# No slice save: `_deleteend!` moves nothing and leaves bits elements in place, so the dropped slots
# still hold, on both carriers, what they held when the call ran. A later `push!` reusing a slot is
# already covered by `memoryrefset!`'s own old-value save and restore.
function rrule!!(::CoDual{typeof(Base._deleteend!),NoFData}, ::AbstractCtx,
                 acd::CoDual{V,<:Union{Vector,Inactive}}, dcd::CoDual{<:Integer}) where {V<:Vector}
    a, da = extract(acd)
    d = Int(primal(dcd))
    len = length(a)
    old_ref = getfield(a, :ref)
    old_dref = isactive(da) ? getfield(da, :ref) : nothing
    Base._deleteend!(a, d)
    old_dref === nothing || Base._deleteend!(da, d)
    function deleteend_pullback(::NoRData)
        old_dref === nothing || _rewind_run!(da, old_dref, len, 0, 0, len - d)
        _rewind_run!(a, old_ref, len, 0, 0, len - d)
        return (NoRData(), NoRData(), NoRData())
    end
    return zero_fcodual(nothing), deleteend_pullback
end

# Same reasoning as `_deleteend!`: Base only advances the ref, so nothing moves.
function rrule!!(::CoDual{typeof(Base._deletebeg!),NoFData}, ::AbstractCtx,
                 acd::CoDual{V,<:Union{Vector,Inactive}}, dcd::CoDual{<:Integer}) where {V<:Vector}
    a, da = extract(acd)
    d = Int(primal(dcd))
    len = length(a)
    old_ref = getfield(a, :ref)
    old_dref = isactive(da) ? getfield(da, :ref) : nothing
    Base._deletebeg!(a, d)
    old_dref === nothing || Base._deletebeg!(da, d)
    function deletebeg_pullback(::NoRData)
        old_dref === nothing || _rewind_run!(da, old_dref, len, 0, d, len - d)
        _rewind_run!(a, old_ref, len, 0, d, len - d)
        return (NoRData(), NoRData(), NoRData())
    end
    return zero_fcodual(nothing), deletebeg_pullback
end

# The one delete that really moves memory: the shift runs over the deleted zone, so unlike the other
# two the dropped values have to be saved on both carriers and put back.
function rrule!!(::CoDual{typeof(Base._deleteat!),NoFData}, ::AbstractCtx,
                 acd::CoDual{V,<:Union{Vector,Inactive}}, icd::CoDual{<:Integer},
                 dcd::CoDual{<:Integer}) where {V<:Vector}
    a, da = extract(acd)
    i, d = Int(primal(icd)), Int(primal(dcd))
    len = length(a)
    old_ref = getfield(a, :ref)
    cut = a[i:(i + d - 1)]
    old_dref = isactive(da) ? getfield(da, :ref) : nothing
    dcut = old_dref === nothing ? nothing : da[i:(i + d - 1)]
    Base._deleteat!(a, i, d)
    old_dref === nothing || Base._deleteat!(da, i, d)
    runs = ((1, 1, i - 1), (i, i + d, len - i - d + 1))
    function deleteat_pullback(::NoRData)
        if old_dref !== nothing
            _rewind_runs!(da, old_dref, len, runs)
            copyto!(da, i, dcut, 1, d)
        end
        _rewind_runs!(a, old_ref, len, runs)
        copyto!(a, i, cut, 1, d)
        return (NoRData(), NoRData(), NoRData(), NoRData())
    end
    return zero_fcodual(nothing), deleteat_pullback
end

# `sizehint!` leaves the length alone and only moves spare capacity, so the shadow is not hinted:
# its own capacity is invisible to values, and leaving it untouched means no shadow reallocation to
# rewind. The primal can still be reallocated, hence the rewind on that side. Base implements the
# grow case as `_growend!` followed by a raw `setfield!(a, :size, …)` undoing the length, which is
# not mirrored onto the shadow — without this rule the shadow is left at the hinted length.
#
# `sizehint!` returns the array, so a constant one has its zero tangent materialised here: returning
# an `Inactive` shadow is not open to a rule.
function rrule!!(::CoDual{typeof(Base.sizehint!),NoFData}, ::AbstractCtx,
                 acd::CoDual{V,<:Union{Vector,Inactive}}, ncd::CoDual{<:Integer}) where {V<:Vector}
    a, da = extract(acd)
    len = length(a)
    old_ref = getfield(a, :ref)
    Base.sizehint!(a, Int(primal(ncd)))
    dout = isactive(da) ? da : zero_tangent(a)
    function sizehint_pullback(::NoRData)
        _rewind_run!(a, old_ref, len, 0, 0, len)
        return (NoRData(), NoRData(), NoRData())
    end
    return CoDual(a, dout), sizehint_pullback
end
