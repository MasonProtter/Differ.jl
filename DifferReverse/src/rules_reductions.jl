# Hand-written rrule!! for reductions (cumsum/extrema). Forward-mode frule!!s for the same
# functions live in DifferForwards/src/rules_reductions.jl.
#
# `sum`/`prod`/`maximum`/`minimum`/`mapreduce(f,+,x)` moved to `rules_perf_backstop.jl` (not
# included by default) — pure performance backstops. `cumsum`/`extrema` stay here.

# ---- cumsum ----
# `y = cumsum(x)` is array-valued, so its rdata is `NoRData`; gradient flows through its fdata
# instead — a fresh zero array `dy` returned as `y`'s shadow, accumulated into in place by
# downstream reads/writes. By the time this pullback runs, `dy` holds the total seed vector, and
# `dx[i] += sum(dy[i:end])` is the reverse cumulative sum, computed here as a running suffix sum.
#
# Known limitation: using `cumsum(x)`'s result (e.g. indexing into it) inside another
# differentiated function fails in reverse mode — the static provenance scan (`_fdata_tracked`)
# doesn't recognize a hand-ruled call's result as a differentiable-array provenance root, so
# downstream reads off `y` are rejected as untracked. Not fixable from here.

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
# Unlike arrays, a `Tuple`'s rdata is a plain tuple of its fields' rdata, so the pullback receives
# a real `(seed_min, seed_max)` seed directly — no fdata-aliasing trick needed, unlike `cumsum`.

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
