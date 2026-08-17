# Hand-written rrule!! for map/map!. Forward-mode frule!!s for the same functions live in
# DifferForwards/src/rules_broadcast.jl.
#
# Follows `sum_map_pullback`'s structure (`rules_perf_backstop.jl`): call `rrule!!` on each element,
# collect the per-element pullbacks in a `Vector`, replay them in reverse, accumulating both the
# array argument(s)' tangent(s) and `f`'s own gradient contribution via `zero_like_rdata_from_type`
# (not `zero_rdata_from_type` — the derived recursion glue can resolve this hand rule via a
# non-concrete static call-site type for `f`, even though `G` is usually concrete).
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

function rrule!!(
    ::CoDual{typeof(map),NoFData}, ::AbstractCtx, gcd::CoDual{G,FG}, (; x, dx)::CoDual{X,X}
) where {G,FG,X<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: map requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
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
    function map_pullback(_)
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
    return CoDual(y, dy), map_pullback
end

# ===========================================================================
# map(f, x, y) — binary
# ===========================================================================

function rrule!!(
    ::CoDual{typeof(map),NoFData}, ::AbstractCtx,
    gcd::CoDual{G,FG}, (; x, dx)::CoDual{X,X}, (; y, dy)::CoDual{Y,Y}
) where {G,FG,X<:Array{<:IEEEFloat},Y<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: map requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
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
    function map2_pullback(_)
        grdata = zero_like_rdata_from_type(G)
        for i in length(pbs):-1:1
            gi_r, xi_r, yi_r = pbs[i](dout[i])
            grdata = increment!!(grdata, gi_r)
            dx[i] = increment!!(dx[i], xi_r)
            dy[i] = increment!!(dy[i], yi_r)
        end
        return (NoRData(), grdata, NoRData(), NoRData())
    end
    return CoDual(out, dout), map2_pullback
end

# ===========================================================================
# map!(f, dest, x) — unary source; args destructured positionally as (x, y) = (dest, source)
# ===========================================================================

function rrule!!(
    ::CoDual{typeof(map!),NoFData}, ::AbstractCtx,
    gcd::CoDual{G,FG}, (; x, dx)::CoDual{D,D}, (; y, dy)::CoDual{X,X}
) where {G,FG,D<:Array{<:IEEEFloat},X<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: map! requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
    size(x) == size(y) ||
        throw(DimensionMismatch("Differ: map!(f, dest, x) requires matching shapes"))
    n = length(y)
    n == 0 && error("Differ: map!(f, dest, x) over empty arrays is not supported by this rule")
    old_dx = copy(dx)
    r1, pb1 = rrule!!(gcd, Ctx(), CoDual(y[1], NoFData()))
    x[1] = primal(r1)
    dx[1] = zero(eltype(dx))
    pbs = Vector{typeof(pb1)}(undef, n)
    pbs[1] = pb1
    for i in 2:n
        ri, pbi = rrule!!(gcd, Ctx(), CoDual(y[i], NoFData()))
        x[i] = primal(ri)
        dx[i] = zero(eltype(dx))
        pbs[i] = pbi
    end
    function mapbang_pullback(_)
        grdata = zero_like_rdata_from_type(G)
        for i in length(pbs):-1:1
            gi_r, yi_r = pbs[i](dx[i])
            grdata = increment!!(grdata, gi_r)
            dy[i] = increment!!(dy[i], yi_r)
            # Restore what was in `dx[i]` before this call (same old-tangent restore as the
            # `memoryrefset!` builtin rule): `map!` overwrites rather than accumulates, so gradient
            # contributions from after this call must not reach what was overwritten.
            dx[i] = old_dx[i]
        end
        return (NoRData(), grdata, NoRData(), NoRData())
    end
    return CoDual(x, dx), mapbang_pullback
end

# ===========================================================================
# map!(f, dest, x, y) — binary source; args destructured positionally as (x, y, z) = (dest, x, y)
# ===========================================================================

function rrule!!(
    ::CoDual{typeof(map!),NoFData}, ::AbstractCtx,
    gcd::CoDual{G,FG}, (; x, dx)::CoDual{D,D}, (; y, dy)::CoDual{X,X}, (; z, dz)::CoDual{Y,Y}
) where {G,FG,D<:Array{<:IEEEFloat},X<:Array{<:IEEEFloat},Y<:Array{<:IEEEFloat}}
    isconcretetype(G) || error("Differ: map! requires a concretely-typed function argument in " *
                                "reverse mode (see ISSUES.md #43)")
    (size(x) == size(y) == size(z)) ||
        throw(DimensionMismatch("Differ: map!(f, dest, x, y) requires matching shapes"))
    n = length(y)
    n == 0 && error("Differ: map!(f, dest, x, y) over empty arrays is not supported by this rule")
    old_dx = copy(dx)
    r1, pb1 = rrule!!(gcd, Ctx(), CoDual(y[1], NoFData()), CoDual(z[1], NoFData()))
    x[1] = primal(r1)
    dx[1] = zero(eltype(dx))
    pbs = Vector{typeof(pb1)}(undef, n)
    pbs[1] = pb1
    for i in 2:n
        ri, pbi = rrule!!(gcd, Ctx(), CoDual(y[i], NoFData()), CoDual(z[i], NoFData()))
        x[i] = primal(ri)
        dx[i] = zero(eltype(dx))
        pbs[i] = pbi
    end
    function mapbang2_pullback(_)
        grdata = zero_like_rdata_from_type(G)
        for i in length(pbs):-1:1
            gi_r, yi_r, zi_r = pbs[i](dx[i])
            grdata = increment!!(grdata, gi_r)
            dy[i] = increment!!(dy[i], yi_r)
            dz[i] = increment!!(dz[i], zi_r)
            dx[i] = old_dx[i]
        end
        return (NoRData(), grdata, NoRData(), NoRData(), NoRData())
    end
    return CoDual(x, dx), mapbang2_pullback
end
