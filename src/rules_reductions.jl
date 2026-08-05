# Hand-written frule!!/rrule!! for reductions (cumsum/extrema). See ISSUES.md #32.
#
# `sum`/`prod`/`maximum`/`minimum`/`mapreduce(f,+,x)` moved to `src/rules_perf_backstop.jl` (not
# included by `src/Differ.jl`) — they were pure performance backstops (Base's real implementations
# go through self-recursive `mapreduce`/`_mapreduce`/`mapfoldl_impl` machinery, which the derived
# path now handles correctly and, since the nested-tape recycling in `src/reverse_interp.jl`/
# `src/stack.jl`, efficiently — see ISSUES #65 and that file's own header). `cumsum`/`extrema` stay
# here: `cumsum`'s comment below documents a real functional limitation in the generic path (not
# purely a perf question), so it isn't in scope for that move pending a separate audit.

# ---- cumsum ----
# Forward is linear, so `cumsum` on the tangent array works directly. Reverse: `y = cumsum(x)`
# is itself array-valued, so (like every array-valued primal in this system) its rdata is
# `NoRData` — the real gradient information flows through its *fdata*, a fresh zero array `dy`
# allocated here and returned as `y`'s shadow. Whatever downstream code reads/writes `y` in the
# larger differentiated function accumulates into `dy` in place (same mechanism as array element
# mutation generally); by the time this pullback runs (topologically after all of that, since
# pullback order is the reverse of forward execution order), `dy` holds the total seed vector, and
# `dx[i] += sum(dy[i:end])` is exactly the reverse cumulative sum, computed here as a running
# suffix sum in a plain loop (ordinary code, not re-dualized, so calling `cumsum`/`reverse` would
# also be fine — the loop is just as simple).
#
# Known limitation: using `cumsum(x)`'s result (e.g. indexing into it) inside another
# differentiated function currently fails in reverse mode — the static provenance scan
# (`_fdata_tracked`, `reverse_interp.jl`) only recognizes specific known operations (`%new`,
# `memorynew`/`memoryrefnew`/`memoryrefget`) as differentiable-array provenance roots, not an
# arbitrary hand-ruled call's result, so downstream reads off `y` are rejected as untracked. Not
# fixable from here; see `test_reduction_rules.jl` for the direct `rrule!!`-level regression test.

function frule!!(::Dual{typeof(cumsum)}, xd::Dual{X}) where {X<:Array{<:IEEEFloat}}
    return Dual(cumsum(primal(xd)), cumsum(tangent(xd)))
end

struct CumsumPullback{X<:Array}
    dx::X
    dy::X
end
function (pb::CumsumPullback)(seed)
    dx, dy = pb.dx, pb.dy
    running = zero(eltype(dy))
    for i in length(dy):-1:1
        running += dy[i]
        dx[i] = increment!!(dx[i], running)
    end
    return (NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(cumsum),NoFData}, ::AbstractCtx, xcd::CoDual{X,X}
) where {X<:Array{<:IEEEFloat}}
    x = primal(xcd)
    dx = tangent(xcd)
    y = cumsum(x)
    dy = zero_tangent(y)
    return CoDual(y, dy), CumsumPullback{X}(dx, dy)
end

# ---- extrema ----
# `(minimum(x), maximum(x))`, sharing one pass over the array. Unlike arrays, a `Tuple`'s rdata is
# a plain tuple of its fields' rdata (see `fwds_rvs_data.jl`), so the pullback receives a real
# `(seed_min, seed_max)` seed directly (no fdata-aliasing trick needed, unlike `cumsum` above).

function frule!!(::Dual{typeof(extrema)}, xd::Dual{X}) where {X<:Array{<:IEEEFloat}}
    x = primal(xd)
    dx = tangent(xd)
    n = length(x)
    n == 0 && error("Differ: extrema of an empty array is not supported by this rule")
    mn = x[1]
    mx = x[1]
    imn = 1
    imx = 1
    for i in 2:n
        if x[i] < mn
            mn = x[i]
            imn = i
        end
        if x[i] > mx
            mx = x[i]
            imx = i
        end
    end
    return Dual((mn, mx), (dx[imn], dx[imx]))
end

struct ExtremaPullback{X<:Array}
    dx::X
    imn::Int
    imx::Int
end
function (pb::ExtremaPullback)((seed_mn, seed_mx))
    pb.dx[pb.imn] = increment!!(pb.dx[pb.imn], seed_mn)
    pb.dx[pb.imx] = increment!!(pb.dx[pb.imx], seed_mx)
    return (NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(extrema),NoFData}, ::AbstractCtx, xcd::CoDual{X,X}
) where {X<:Array{<:IEEEFloat}}
    x = primal(xcd)
    dx = tangent(xcd)
    n = length(x)
    n == 0 && error("Differ: extrema of an empty array is not supported by this rule")
    mn = x[1]
    mx = x[1]
    imn = 1
    imx = 1
    for i in 2:n
        if x[i] < mn
            mn = x[i]
            imn = i
        end
        if x[i] > mx
            mx = x[i]
            imx = i
        end
    end
    return zero_fcodual((mn, mx)), ExtremaPullback{X}(dx, imn, imx)
end
