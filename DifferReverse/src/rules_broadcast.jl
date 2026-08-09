# Hand-written rrule!! for map/map!. Forward-mode frule!!s for the same functions live in
# DifferForwards/src/rules_broadcast.jl.
#
# Follows `SumMapPullback`'s structure (`rrules.jl`): call `rrule!!` on each element, collect the
# per-element pullbacks in a `Vector`, replay them in reverse, accumulating both the array
# argument(s)' tangent(s) and `f`'s own gradient contribution via `zero_like_rdata_from_type` (not
# `zero_rdata_from_type` — the derived recursion glue can resolve this hand rule via a non-concrete
# static call-site type for `f`, even though `G` is usually concrete).
#
# Every rule requires `G` concrete: reverse mode has no dynamic dispatch, so a `map`/`map!` call
# whose function argument's static type isn't concrete can't resolve a per-element `rrule!!` call.
#
# Scope: unary and binary `map`/`map!` over same-shape `Array`s, restricted to `Array{<:IEEEFloat}`
# element types — the "read the array's accumulated fdata back as the per-element seed" trick below
# needs `rdata_type(tangent_type(Y)) == tangent_type(Y)` (a pure-rdata type), which every
# `IEEEFloat` satisfies.

# ===========================================================================
# map(f, x) — unary
# ===========================================================================

struct MapPullback{G,PB,Dx<:Array,Dy<:Array}
    pbs::Vector{PB}
    dx::Dx
    dy::Dy
end
function (pb::MapPullback{G})(seed) where {G}
    pbs, dx, dy = pb.pbs, pb.dx, pb.dy
    # `dy` is `y`'s own fdata array; by the time this pullback runs, every downstream use of `y`
    # has accumulated its cotangent into `dy` in place, so `dy[i]` is the full backward-accumulated
    # seed for element `i`.
    grdata = zero_like_rdata_from_type(G)
    for i in length(pbs):-1:1
        gi_r, xi_r = pbs[i](dy[i])
        grdata = increment!!(grdata, gi_r)
        dx[i] = increment!!(dx[i], xi_r)
    end
    return (NoRData(), grdata, NoRData())
end

function rrule!!(
    ::CoDual{typeof(map),NoFData}, ::AbstractCtx, gcd::CoDual{G,FG}, xcd::CoDual{X,X}
) where {G,FG,X<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: map requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
    x = primal(xcd)
    dx = tangent(xcd)
    n = length(x)
    n == 0 && error("Differ: map(f, x) over an empty array is not supported by this rule")
    y1, pb1 = rrule!!(gcd, Ctx(), CoDual(x[1], NoFData()))
    Y = typeof(primal(y1))
    y = Array{Y}(undef, size(x))
    dy = zeros(Y, size(x))
    y[1] = primal(y1)
    pbs = Vector{typeof(pb1)}(undef, n)
    pbs[1] = pb1
    for i in 2:n
        yi, pbi = rrule!!(gcd, Ctx(), CoDual(x[i], NoFData()))
        y[i] = primal(yi)
        pbs[i] = pbi
    end
    return CoDual(y, dy), MapPullback{G,typeof(pb1),typeof(dx),typeof(dy)}(pbs, dx, dy)
end

# ===========================================================================
# map(f, x, y) — binary
# ===========================================================================

struct Map2Pullback{G,PB,Dx<:Array,Dy<:Array,Dout<:Array}
    pbs::Vector{PB}
    dx::Dx
    dy::Dy
    dout::Dout
end
function (pb::Map2Pullback{G})(seed) where {G}
    pbs, dx, dy, dout = pb.pbs, pb.dx, pb.dy, pb.dout
    grdata = zero_like_rdata_from_type(G)
    for i in length(pbs):-1:1
        gi_r, xi_r, yi_r = pbs[i](dout[i])
        grdata = increment!!(grdata, gi_r)
        dx[i] = increment!!(dx[i], xi_r)
        dy[i] = increment!!(dy[i], yi_r)
    end
    return (NoRData(), grdata, NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(map),NoFData}, ::AbstractCtx, gcd::CoDual{G,FG}, xcd::CoDual{X,X}, ycd::CoDual{Y,Y}
) where {G,FG,X<:Array{<:IEEEFloat},Y<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: map requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
    x = primal(xcd)
    dx = tangent(xcd)
    y = primal(ycd)
    dy = tangent(ycd)
    size(x) == size(y) || throw(DimensionMismatch("Differ: map(f, x, y) requires same-shape arrays"))
    n = length(x)
    n == 0 && error("Differ: map(f, x, y) over empty arrays is not supported by this rule")
    r1, pb1 = rrule!!(gcd, Ctx(), CoDual(x[1], NoFData()), CoDual(y[1], NoFData()))
    R = typeof(primal(r1))
    out = Array{R}(undef, size(x))
    dout = zeros(R, size(x))
    out[1] = primal(r1)
    pbs = Vector{typeof(pb1)}(undef, n)
    pbs[1] = pb1
    for i in 2:n
        ri, pbi = rrule!!(gcd, Ctx(), CoDual(x[i], NoFData()), CoDual(y[i], NoFData()))
        out[i] = primal(ri)
        pbs[i] = pbi
    end
    return CoDual(out, dout),
           Map2Pullback{G,typeof(pb1),typeof(dx),typeof(dy),typeof(dout)}(pbs, dx, dy, dout)
end

# ===========================================================================
# map!(f, dest, x) — unary source
# ===========================================================================

struct MapBangPullback{G,PB,Dx<:Array,Ddest<:Array}
    pbs::Vector{PB}
    dx::Dx
    ddest::Ddest
    old_ddest::Ddest
end
function (pb::MapBangPullback{G})(seed) where {G}
    pbs, dx, ddest, old = pb.pbs, pb.dx, pb.ddest, pb.old_ddest
    grdata = zero_like_rdata_from_type(G)
    for i in length(pbs):-1:1
        gi_r, xi_r = pbs[i](ddest[i])
        grdata = increment!!(grdata, gi_r)
        dx[i] = increment!!(dx[i], xi_r)
        # Restore what was in `ddest[i]` before this call (same old-tangent restore as the
        # `memoryrefset!` builtin rule): `map!` overwrites rather than accumulates, so gradient
        # contributions from after this call must not reach what was overwritten.
        ddest[i] = old[i]
    end
    return (NoRData(), grdata, NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(map!),NoFData}, ::AbstractCtx,
    gcd::CoDual{G,FG}, destcd::CoDual{D,D}, xcd::CoDual{X,X}
) where {G,FG,D<:Array{<:IEEEFloat},X<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: map! requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
    dest = primal(destcd)
    ddest = tangent(destcd)
    x = primal(xcd)
    dx = tangent(xcd)
    size(dest) == size(x) ||
        throw(DimensionMismatch("Differ: map!(f, dest, x) requires matching shapes"))
    n = length(x)
    n == 0 && error("Differ: map!(f, dest, x) over empty arrays is not supported by this rule")
    old_ddest = copy(ddest)
    y1, pb1 = rrule!!(gcd, Ctx(), CoDual(x[1], NoFData()))
    dest[1] = primal(y1)
    ddest[1] = zero(eltype(ddest))
    pbs = Vector{typeof(pb1)}(undef, n)
    pbs[1] = pb1
    for i in 2:n
        yi, pbi = rrule!!(gcd, Ctx(), CoDual(x[i], NoFData()))
        dest[i] = primal(yi)
        ddest[i] = zero(eltype(ddest))
        pbs[i] = pbi
    end
    return CoDual(dest, ddest),
           MapBangPullback{G,typeof(pb1),typeof(dx),typeof(ddest)}(pbs, dx, ddest, old_ddest)
end

# ===========================================================================
# map!(f, dest, x, y) — binary source
# ===========================================================================

struct MapBang2Pullback{G,PB,Dx<:Array,Dy<:Array,Ddest<:Array}
    pbs::Vector{PB}
    dx::Dx
    dy::Dy
    ddest::Ddest
    old_ddest::Ddest
end
function (pb::MapBang2Pullback{G})(seed) where {G}
    pbs, dx, dy, ddest, old = pb.pbs, pb.dx, pb.dy, pb.ddest, pb.old_ddest
    grdata = zero_like_rdata_from_type(G)
    for i in length(pbs):-1:1
        gi_r, xi_r, yi_r = pbs[i](ddest[i])
        grdata = increment!!(grdata, gi_r)
        dx[i] = increment!!(dx[i], xi_r)
        dy[i] = increment!!(dy[i], yi_r)
        ddest[i] = old[i]
    end
    return (NoRData(), grdata, NoRData(), NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(map!),NoFData}, ::AbstractCtx,
    gcd::CoDual{G,FG}, destcd::CoDual{D,D}, xcd::CoDual{X,X}, ycd::CoDual{Y,Y}
) where {G,FG,D<:Array{<:IEEEFloat},X<:Array{<:IEEEFloat},Y<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: map! requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
    dest = primal(destcd)
    ddest = tangent(destcd)
    x = primal(xcd)
    dx = tangent(xcd)
    y = primal(ycd)
    dy = tangent(ycd)
    (size(dest) == size(x) == size(y)) ||
        throw(DimensionMismatch("Differ: map!(f, dest, x, y) requires matching shapes"))
    n = length(x)
    n == 0 && error("Differ: map!(f, dest, x, y) over empty arrays is not supported by this rule")
    old_ddest = copy(ddest)
    r1, pb1 = rrule!!(gcd, Ctx(), CoDual(x[1], NoFData()), CoDual(y[1], NoFData()))
    dest[1] = primal(r1)
    ddest[1] = zero(eltype(ddest))
    pbs = Vector{typeof(pb1)}(undef, n)
    pbs[1] = pb1
    for i in 2:n
        ri, pbi = rrule!!(gcd, Ctx(), CoDual(x[i], NoFData()), CoDual(y[i], NoFData()))
        dest[i] = primal(ri)
        ddest[i] = zero(eltype(ddest))
        pbs[i] = pbi
    end
    return CoDual(dest, ddest),
           MapBang2Pullback{G,typeof(pb1),typeof(dx),typeof(dy),typeof(ddest)}(pbs, dx, dy, ddest, old_ddest)
end
