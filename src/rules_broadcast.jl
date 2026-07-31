# Hand-written frule!!/rrule!! for map/map! and (where feasible) broadcast. See ISSUES.md #31.
#
# Same motivation as `sum`/`sum(f, ·)` in `rrules.jl`: Base's real `map`/`map!`/broadcast
# implementations use IR constructs (self-recursive pairwise reduction, `Expr(:gc_preserve_begin)`/
# `:gc_preserve_end` inside broadcast's `copy`/`copyto!`) that Differ's dualization engine does not
# support. Every rule below is an explicit per-element loop calling `frule!!`/`rrule!!` on the
# user's function `f`, never touching Base's actual `map`/`broadcast` internals — this is what lets
# `map(sin, x)` differentiate even though `sin.(x)` (unmodified, no hand rule) would fail on the
# `:gc_preserve` construct inside `copy`.
#
# Reverse-mode rules follow `SumMapPullback`'s structure exactly (`rrules.jl`): call `rrule!!` on
# each element, collect the per-element pullbacks in a `Vector`, and replay them in reverse in the
# rule's own pullback, accumulating both the array argument(s)' tangent(s) and `f`'s own gradient
# contribution via `zero_like_rdata_from_type` (not `zero_rdata_from_type` — see the comment on
# `SumMapPullback` for why: `G` is usually concrete, but the derived recursion glue can resolve this
# hand rule via a non-concrete static call-site type for `f`).
#
# Every reverse-mode rule below carries the ISSUES.md #43 guard: reverse mode has no dynamic
# dispatch, so a `map`/`map!` call whose function argument's static type is not concrete (e.g. a
# `Union` of two different closures reached through an abstractly-typed field) cannot be
# differentiated — the per-element `rrule!!(gcd, Ctx(), ...)` call would need to resolve a rule for
# a non-concrete callee type, which is exactly what #43 says is unsupported. Forward mode has no
# such restriction (`frule!!` dispatches on a genuine `Dual`, and a non-concrete function type simply
# routes to `dynamic_frule` like any other dynamic call), so no guard is needed there.
#
# Scope: unary and binary `map`/`map!` over same-shape `Array`s. Reverse-mode rules are further
# restricted to `Array{<:IEEEFloat}` element types (matching `SumMapPullback`): a reverse-mode
# per-element result must have `rdata_type(tangent_type(Y)) == tangent_type(Y)` (a "pure rdata"
# type) for the "read the array's own accumulated fdata straight back out as the per-element seed"
# trick below to be valid, and every `IEEEFloat` satisfies this.

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

struct MapPullback{G,PB,Dx<:Array,Dy<:Array}
    pbs::Vector{PB}
    dx::Dx
    dy::Dy
end
function (pb::MapPullback{G})(seed) where {G}
    pbs, dx, dy = pb.pbs, pb.dx, pb.dy
    # `dy` is `y`'s own fdata array (fresh, zero-initialised at construction below): by the time this
    # pullback runs, every downstream use of `y` has already accumulated its cotangent into `dy` in
    # place (fdata semantics — see the header comment on `SumMapPullback`), so `dy[i]` *is* the full
    # backward-accumulated seed for element `i`.
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
        # Restore whatever was in `ddest[i]` before this call, mirroring the `memoryrefset!` builtin
        # rule's old-tangent restore (`builtins_reverse.jl`): `dest[i]` is overwritten (not
        # accumulated into) by `map!`, so anything that was there before is genuinely gone from the
        # primal's perspective and must not receive gradient contributions from after this call.
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
