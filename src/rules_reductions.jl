# Hand-written frule!!/rrule!! for reductions (sum/prod/maximum/minimum/mapreduce/cumsum/extrema).
# See ISSUES.md #32.
#
# Same motivation as `sum`/`sum(f,·)` in `rrules.jl`: Base's real implementations of these
# (`prod`, `maximum`/`minimum`, `mapreduce`, `cumsum`) go through `mapreduce`/`_mapreduce` /
# `mapfoldl_impl` machinery that is self-recursive (pairwise divide-and-conquer) and `@simd`
# annotated above `Base.pairwise_blocksize` elements. Differ's reverse-mode recursion engine bails
# on genuine self-recursion, which is what makes generic recursion into these fail above ~1024
# elements. (Forward mode's half of that reason is gone as of 2026-08-01 — `:loopinfo` is now carried
# through, ISSUES #62 — but the self-recursion half stands, so these rules stay.) Every rule below is
# a plain hand-written loop that never touches Base's internals, so it is correct and efficient at
# any size.

# `sum`'s REVERSE rule already lives in `rrules.jl` (`SumPullback`). Only the forward half is
# missing; `sum` is linear so both primal and tangent are computed by the same accumulation.
function frule!!(::Dual{typeof(sum)}, xd::Dual{X}) where {X<:Array{<:IEEEFloat}}
    return Dual(sum(primal(xd)), sum(tangent(xd)))
end

# ---- prod ----

function frule!!(::Dual{typeof(prod)}, xd::Dual{X}) where {X<:Array{<:IEEEFloat}}
    x = primal(xd)
    dx = tangent(xd)
    n = length(x)
    n == 0 && error("Differ: prod of an empty array is not supported by this rule")
    p = one(eltype(x))
    for xi in x
        p *= xi
    end
    dp = zero(eltype(dx))
    for i in eachindex(x)
        dp += dx[i] * p / x[i]
    end
    return Dual(p, dp)
end

struct ProdPullback{X<:Array}
    x::X
    dx::X
    p::eltype(X)
end
function (pb::ProdPullback)(seed)
    x, dx, p = pb.x, pb.dx, pb.p
    for i in eachindex(dx)
        dx[i] = increment!!(dx[i], seed * p / x[i])
    end
    return (NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(prod),NoFData}, ::AbstractCtx, xcd::CoDual{X,X}
) where {X<:Array{<:IEEEFloat}}
    x = primal(xcd)
    dx = tangent(xcd)
    n = length(x)
    n == 0 && error("Differ: prod of an empty array is not supported by this rule")
    p = one(eltype(x))
    for xi in x
        p *= xi
    end
    return zero_fcodual(p), ProdPullback{X}(x, dx, p)
end

# ---- maximum / minimum ----
# Derivative flows entirely to the (first, on ties) extremal element; every other element gets zero.

function frule!!(::Dual{typeof(maximum)}, xd::Dual{X}) where {X<:Array{<:IEEEFloat}}
    x = primal(xd)
    dx = tangent(xd)
    n = length(x)
    n == 0 && error("Differ: maximum of an empty array is not supported by this rule")
    m = x[1]
    idx = 1
    for i in 2:n
        if x[i] > m
            m = x[i]
            idx = i
        end
    end
    return Dual(m, dx[idx])
end

function frule!!(::Dual{typeof(minimum)}, xd::Dual{X}) where {X<:Array{<:IEEEFloat}}
    x = primal(xd)
    dx = tangent(xd)
    n = length(x)
    n == 0 && error("Differ: minimum of an empty array is not supported by this rule")
    m = x[1]
    idx = 1
    for i in 2:n
        if x[i] < m
            m = x[i]
            idx = i
        end
    end
    return Dual(m, dx[idx])
end

struct ExtremalPullback{X<:Array}
    dx::X
    idx::Int
end
function (pb::ExtremalPullback)(seed)
    pb.dx[pb.idx] = increment!!(pb.dx[pb.idx], seed)
    return (NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(maximum),NoFData}, ::AbstractCtx, xcd::CoDual{X,X}
) where {X<:Array{<:IEEEFloat}}
    x = primal(xcd)
    dx = tangent(xcd)
    n = length(x)
    n == 0 && error("Differ: maximum of an empty array is not supported by this rule")
    m = x[1]
    idx = 1
    for i in 2:n
        if x[i] > m
            m = x[i]
            idx = i
        end
    end
    return zero_fcodual(m), ExtremalPullback{X}(dx, idx)
end

function rrule!!(
    ::CoDual{typeof(minimum),NoFData}, ::AbstractCtx, xcd::CoDual{X,X}
) where {X<:Array{<:IEEEFloat}}
    x = primal(xcd)
    dx = tangent(xcd)
    n = length(x)
    n == 0 && error("Differ: minimum of an empty array is not supported by this rule")
    m = x[1]
    idx = 1
    for i in 2:n
        if x[i] < m
            m = x[i]
            idx = i
        end
    end
    return zero_fcodual(m), ExtremalPullback{X}(dx, idx)
end

# ---- mapreduce(f, +, x) ----
# Scoped to `op === +` (dispatch on `Dual{typeof(+)}`/`CoDual{typeof(+),NoFData}` enforces this);
# general `op` isn't supported. Mirrors `sum(f, x)`'s hand rule (`rrules.jl`, `SumMapPullback`)
# exactly, including its non-concrete-`G` accumulator handling.
#
# Known limitation: calling `mapreduce(f, +, x)` from inside another differentiated function
# currently fails in reverse mode, separately from anything below — Differ's reverse-mode
# interpreter compiles the 3-positional-argument call as a "dynamic invoke" with `f`/`+` widened to
# abstract `Function` (confirmed via `Base.code_typed(...; interp=ADInterpreter{Reverse}())`),
# unlike `sum(f, x)`'s 2-argument shape, which stays concrete. The concrete-`op` dispatch constraint
# on this rule then never matches at that call site, and the call falls through to failing generic
# recursion instead. This is in the recursive-call resolution (`reverse_interp.jl`), not fixable
# from here; see `test_reduction_rules.jl` for the direct `rrule!!`-level regression test this rule
# does support.

function frule!!(
    ::Dual{typeof(mapreduce)}, fd::Dual, ::Dual{typeof(+)}, xd::Dual{X}
) where {X<:Array{<:IEEEFloat}}
    x = primal(xd)
    dx = tangent(xd)
    n = length(x)
    n == 0 && error("Differ: mapreduce(f, +, x) over an empty array is not supported by this rule")
    y1 = frule!!(fd, Dual(x[1], dx[1]))
    s = primal(y1)
    ds = tangent(y1)
    for i in 2:n
        yi = frule!!(fd, Dual(x[i], dx[i]))
        s += primal(yi)
        ds += tangent(yi)
    end
    return Dual(s, ds)
end

# See `SumMapPullback` in `rrules.jl` for why `zero_like_rdata_from_type(G)` (not
# `zero_rdata_from_type`) is required here: the derived recursion glue can resolve this hand rule
# via a non-concrete static call-site type for `f`, and only the "like" variant handles that.
struct MapReducePullback{G,PB,Dx<:Array}
    pbs::Vector{PB}
    dx::Dx
end
function (pb::MapReducePullback{G})(seed) where {G}
    pbs = pb.pbs
    dx = pb.dx
    grdata = zero_like_rdata_from_type(G)
    for i in length(pbs):-1:1
        gi_r, xi_r = pbs[i](seed)
        grdata = increment!!(grdata, gi_r)
        dx[i] = increment!!(dx[i], xi_r)
    end
    return (NoRData(), grdata, NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(mapreduce),NoFData}, ::AbstractCtx,
    gcd::CoDual{G,FG}, ::CoDual{typeof(+),NoFData}, xcd::CoDual{X,X}
) where {G,FG,X<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: mapreduce requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
    x = primal(xcd)
    dx = tangent(xcd)
    n = length(x)
    n == 0 && error("Differ: mapreduce(f, +, x) over an empty array is not supported by this rule")
    y1, pb1 = rrule!!(gcd, Ctx(), CoDual(x[1], NoFData()))
    s = primal(y1)
    pbs = Vector{typeof(pb1)}(undef, n)
    pbs[1] = pb1
    for i in 2:n
        yi, pbi = rrule!!(gcd, Ctx(), CoDual(x[i], NoFData()))
        s += primal(yi)
        pbs[i] = pbi
    end
    return zero_fcodual(s), MapReducePullback{G,typeof(pb1),typeof(dx)}(pbs, dx)
end

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
