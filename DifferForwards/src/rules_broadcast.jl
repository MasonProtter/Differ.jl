# Hand-written frule!! for map/map!. See ISSUES.md #31. Reverse-mode rrule!!s for the same
# functions live in DifferReverse/src/rules_broadcast.jl.
#
# Same motivation as `sum`/`sum(f, ·)`: Base's real `map`/`map!` implementations use IR constructs
# the dualization engine doesn't support (chiefly self-recursive pairwise reduction). Every rule
# here is an explicit per-element loop calling `frule!!` on the user's function `f`, never
# touching Base's actual `map`/`broadcast` internals.
#
# Unmodified `.`-syntax dualizes for a single array argument (`sin.(x)`) and array-with-scalar
# forms (`x .* 2.0`) via the `memmove`/`copyto!` path (`foreigncalls.jl`); two-array broadcast
# (`x .* y`) is the remaining gap, blocked by ISSUES #60 rather than anything here. These rules
# still matter for `map`/`map!` proper and the two-array case.
#
# Scope: unary and binary `map`/`map!` over same-shape `Array`s.

# ===========================================================================
# map(f, x) — unary
# ===========================================================================

function frule!!(::Dual{typeof(map)}, gdual::Dual{G}, xdual::Dual{X}) where {G,X<:Array}
    x = primal(xdual)
    dx = tangent(xdual)
    n = length(x)
    n == 0 && error("Differ: map(f, x) over an empty array is not supported by this rule")
    y1 = frule!!(gdual, Dual(x[1], dx[1]))
    Y, DY = typeof(primal(y1)), typeof(tangent(y1))
    y = Array{Y}(undef, size(x))
    dy = Array{DY}(undef, size(x))
    y[1] = primal(y1)
    dy[1] = tangent(y1)
    for i in 2:n
        yi = frule!!(gdual, Dual(x[i], dx[i]))
        y[i] = primal(yi)
        dy[i] = tangent(yi)
    end
    return Dual(y, dy)
end

# ===========================================================================
# map(f, x, y) — binary
# ===========================================================================

function frule!!(
    ::Dual{typeof(map)}, gdual::Dual{G}, xdual::Dual{X}, ydual::Dual{Y}
) where {G,X<:Array,Y<:Array}
    x = primal(xdual)
    dx = tangent(xdual)
    y = primal(ydual)
    dy = tangent(ydual)
    size(x) == size(y) || throw(DimensionMismatch("Differ: map(f, x, y) requires same-shape arrays"))
    n = length(x)
    n == 0 && error("Differ: map(f, x, y) over empty arrays is not supported by this rule")
    r1 = frule!!(gdual, Dual(x[1], dx[1]), Dual(y[1], dy[1]))
    R, DR = typeof(primal(r1)), typeof(tangent(r1))
    out = Array{R}(undef, size(x))
    dout = Array{DR}(undef, size(x))
    out[1] = primal(r1)
    dout[1] = tangent(r1)
    for i in 2:n
        ri = frule!!(gdual, Dual(x[i], dx[i]), Dual(y[i], dy[i]))
        out[i] = primal(ri)
        dout[i] = tangent(ri)
    end
    return Dual(out, dout)
end

# ===========================================================================
# map!(f, dest, x) — unary source
# ===========================================================================

function frule!!(
    ::Dual{typeof(map!)}, gdual::Dual{G}, destdual::Dual{D}, xdual::Dual{X}
) where {G,D<:Array,X<:Array}
    dest = primal(destdual)
    ddest = tangent(destdual)
    x = primal(xdual)
    dx = tangent(xdual)
    size(dest) == size(x) ||
        throw(DimensionMismatch("Differ: map!(f, dest, x) requires matching shapes"))
    for i in eachindex(x)
        yi = frule!!(gdual, Dual(x[i], dx[i]))
        dest[i] = primal(yi)
        ddest[i] = tangent(yi)
    end
    return destdual
end

# ===========================================================================
# map!(f, dest, x, y) — binary source
# ===========================================================================

function frule!!(
    ::Dual{typeof(map!)}, gdual::Dual{G}, destdual::Dual{D}, xdual::Dual{X}, ydual::Dual{Y}
) where {G,D<:Array,X<:Array,Y<:Array}
    dest = primal(destdual)
    ddest = tangent(destdual)
    x = primal(xdual)
    dx = tangent(xdual)
    y = primal(ydual)
    dy = tangent(ydual)
    (size(dest) == size(x) == size(y)) ||
        throw(DimensionMismatch("Differ: map!(f, dest, x, y) requires matching shapes"))
    for i in eachindex(x)
        ri = frule!!(gdual, Dual(x[i], dx[i]), Dual(y[i], dy[i]))
        dest[i] = primal(ri)
        ddest[i] = tangent(ri)
    end
    return destdual
end
