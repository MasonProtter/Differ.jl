# Hand-written frule!! for reductions (cumsum/extrema). Reverse-mode rrule!!s for the same
# functions live in DifferReverse/src/rules_reductions.jl.
#
# `sum`/`prod`/`maximum`/`minimum`/`mapreduce(f,+,x)` moved to `rules_perf_backstop.jl` (not
# included by default) — the derived path now handles Base's self-recursive
# `mapreduce`/`mapfoldl_impl` machinery correctly and efficiently. `cumsum`/`extrema` stay here:
# `cumsum` has a real functional limitation in the generic path, not just a perf question.

# ---- cumsum ----
# Forward is linear, so `cumsum` on the tangent array works directly.

function frule!!(::Dual{typeof(cumsum)}, xd::Dual{X}) where {X<:Array{<:IEEEFloat}}
    return Dual(cumsum(primal(xd)), cumsum(tangent(xd)))
end

# ---- extrema ----
# `(minimum(x), maximum(x))` in one pass.

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
