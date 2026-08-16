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

function rrule!!(
    ::CoDual{typeof(cumsum),NoFData}, ::AbstractCtx, (; x, dx)::CoDual{X,X}
) where {X<:Array{<:IEEEFloat}}
    y = cumsum(x)
    dy = zero_tangent(y)
    function cumsum_pullback(_)
        running = zero(eltype(dy))
        for i in length(dy):-1:1
            running += dy[i]
            dx[i] = increment!!(dx[i], running)
        end
        return (NoRData(), NoRData())
    end
    return CoDual(y, dy), cumsum_pullback
end

# ---- extrema ----
# Unlike arrays, a `Tuple`'s rdata is a plain tuple of its fields' rdata, so the pullback receives
# a real `(seed_min, seed_max)` seed directly — no fdata-aliasing trick needed, unlike `cumsum`.

function rrule!!(
    ::CoDual{typeof(extrema),NoFData}, ::AbstractCtx, (; x, dx)::CoDual{X,X}
) where {X<:Array{<:IEEEFloat}}
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
    # single-assignment copies: `imn`/`imx` are reassigned in the loop above and must not be captured
    min_idx, max_idx = imn, imx
    function extrema_pullback((dmn, dmx))
        dx[min_idx] = increment!!(dx[min_idx], dmn)
        dx[max_idx] = increment!!(dx[max_idx], dmx)
        return (NoRData(), NoRData())
    end
    return zero_fcodual((mn, mx)), extrema_pullback
end
