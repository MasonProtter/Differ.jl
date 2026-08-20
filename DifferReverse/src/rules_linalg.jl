# Hand-written rrule!! for LinearAlgebra basics (dot/norm/*/tr/mul!). Forward-mode frule!!s for
# the same functions live in DifferForwards/src/rules_linalg.jl.
#
# `dot`/`norm`/`*` bottom out in BLAS `ccall`s once inlined, which the dualization engine can't
# see through — a hand rule is the only way to differentiate these. Each rule computes the
# primal via a plain untracked call (or an explicit loop) and supplies the gradient via the
# closed-form identity, never by dualizing the target's actual body.
#
# `transpose`/`adjoint` deliberately have no rule here: both already differentiate correctly via
# the generic struct-tangent machinery, so a rule would be redundant and risk dispatch ambiguity
# with the generic fallback. See `test/test_linalg_rules.jl` for the regression test.

# ---------------------------------------------------------------------------
# dot(x, y)
# ---------------------------------------------------------------------------

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.dot),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Vector{Float64},<:Union{Vector{Float64},Inactive}},
    (; y, dy)::CoDual{Vector{Float64},<:Union{Vector{Float64},Inactive}},
)
    length(x) == length(y) ||
        throw(DimensionMismatch("dot: vectors have different lengths"))
    s = dot(x, y)
    xactive, yactive = isactive(dx), isactive(dy)
    function dot_pullback(dz)
        for i in eachindex(x, y)
            xactive && (dx[i] = increment!!(dx[i], dz * y[i]))
            yactive && (dy[i] = increment!!(dy[i], dz * x[i]))
        end
        return (NoRData(), NoRData(), NoRData())
    end
    return zero_fcodual(s), dot_pullback
end

# ---------------------------------------------------------------------------
# norm(x) — 2-norm only
# ---------------------------------------------------------------------------

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.norm),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Vector{Float64},<:Union{Vector{Float64},Inactive}},
)
    nrm = norm(x)
    xactive = isactive(dx)
    function norm_pullback(dy)
        if xactive
            c = dy / nrm
            for i in eachindex(x, dx)
                dx[i] = increment!!(dx[i], c * x[i])
            end
        end
        return (NoRData(), NoRData())
    end
    return zero_fcodual(nrm), norm_pullback
end

# ---------------------------------------------------------------------------
# tr(A) — explicit diagonal loop, mirroring `sum_pullback`'s shape in `rules_perf_backstop.jl`.
# ---------------------------------------------------------------------------

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.tr),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Matrix{Float64},<:Union{Matrix{Float64},Inactive}},
)
    xactive = isactive(dx)
    n = size(x, 1)
    s = 0.0
    for i in 1:n
        s += x[i, i]
    end
    function tr_pullback(dy)
        if xactive
            for i in 1:n
                dx[i, i] = increment!!(dx[i, i], dy)
            end
        end
        return (NoRData(), NoRData())
    end
    return zero_fcodual(s), tr_pullback
end

# ---------------------------------------------------------------------------
# `*` — matrix-vector and matrix-matrix, `Matrix{Float64}`/`Vector{Float64}` only. Signatures are
# deliberately narrow (never `Any`) so they don't interfere with plain scalar `*`, which the
# intrinsic layer (`mul_float`) handles entirely and never reaches `frule!!`/`rrule!!` dispatch.
# ---------------------------------------------------------------------------

# --- matrix * vector — first arg destructured as `x` (the matrix), second as `y` (the vector) ---

function rrule!!(
    ::CoDual{typeof(Base.:*),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Matrix{Float64},<:Union{Matrix{Float64},Inactive}},
    (; y, dy)::CoDual{Vector{Float64},<:Union{Vector{Float64},Inactive}},
)
    z = x * y
    zcd = zero_fcodual(z)
    dz = tangent(zcd)
    m, n = size(x)
    xactive, yactive = isactive(dx), isactive(dy)
    function matvecmul_pullback(::NoRData)
        if xactive
            for i in 1:m, j in 1:n
                dx[i, j] = increment!!(dx[i, j], dz[i] * y[j])
            end
        end
        if yactive
            for j in 1:n
                s = 0.0
                for i in 1:m
                    s += x[i, j] * dz[i]
                end
                dy[j] = increment!!(dy[j], s)
            end
        end
        return (NoRData(), NoRData(), NoRData())
    end
    return zcd, matvecmul_pullback
end

# --- matrix * matrix — `x`/`y` are the two matrix arguments ---

function rrule!!(
    ::CoDual{typeof(Base.:*),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Matrix{Float64},<:Union{Matrix{Float64},Inactive}},
    (; y, dy)::CoDual{Matrix{Float64},<:Union{Matrix{Float64},Inactive}},
)
    z = x * y
    zcd = zero_fcodual(z)
    dz = tangent(zcd)
    xactive, yactive = isactive(dx), isactive(dy)
    function matmul_pullback(::NoRData)
        # dx += dz*y',  dy += x'*dz
        if xactive
            gx = dz * adjoint(y)
            for i in eachindex(dx, gx)
                dx[i] = increment!!(dx[i], gx[i])
            end
        end
        if yactive
            gy = adjoint(x) * dz
            for i in eachindex(dy, gy)
                dy[i] = increment!!(dy[i], gy[i])
            end
        end
        return (NoRData(), NoRData(), NoRData())
    end
    return zcd, matmul_pullback
end

# ---------------------------------------------------------------------------
# mul!(C, A, B) — in-place `C = A*B`, `Matrix{Float64}`/`Vector{Float64}` only (3-arg form; the
# α/β-scaled 5-arg form isn't covered). `C` is overwritten, not accumulated, so the pullback reads
# `C`'s old fdata as the backward seed, zeroes it, and restores the old fdata afterward — same
# old-tangent-restore pattern as the `memoryrefset!` builtin rule.
# ---------------------------------------------------------------------------

# --- mul!(y, A, x) — matrix * vector; args destructured positionally as (x, y, z) = (dest, A, source) ---

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.mul!),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Vector{Float64},<:Union{Vector{Float64},Inactive}},
    (; y, dy)::CoDual{Matrix{Float64},<:Union{Matrix{Float64},Inactive}},
    (; z, dz)::CoDual{Vector{Float64},<:Union{Vector{Float64},Inactive}},
)
    # `dx` carries both the backward seed and the result's shadow, so a constant destination would
    # silently drop the sources' gradients. A write-only buffer needs a zeroed shadow, not `Inactive`.
    _require_active_dest(dx, "mul!", "gradient flowing to the factors")
    old_dx = copy(dx)
    mul!(x, y, z)
    fill!(dx, 0.0)
    m, n = size(y)
    yactive, zactive = isactive(dy), isactive(dz)
    function mulmatvec_pullback(::NoRData)
        if yactive
            for i in 1:m, j in 1:n
                dy[i, j] = increment!!(dy[i, j], dx[i] * z[j])
            end
        end
        if zactive
            for j in 1:n
                s = 0.0
                for i in 1:m
                    s += y[i, j] * dx[i]
                end
                dz[j] = increment!!(dz[j], s)
            end
        end
        dx .= old_dx
        return (NoRData(), NoRData(), NoRData(), NoRData())
    end
    return CoDual(x, dx), mulmatvec_pullback
end

# --- mul!(C, A, B) — matrix * matrix; args destructured positionally as (x, y, z) = (dest, A, B) ---

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.mul!),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Matrix{Float64},<:Union{Matrix{Float64},Inactive}},
    (; y, dy)::CoDual{Matrix{Float64},<:Union{Matrix{Float64},Inactive}},
    (; z, dz)::CoDual{Matrix{Float64},<:Union{Matrix{Float64},Inactive}},
)
    # See the matrix-vector rule above: a constant destination would silently drop the factors'
    # gradients.
    _require_active_dest(dx, "mul!", "gradient flowing to the factors")
    old_dx = copy(dx)
    mul!(x, y, z)
    fill!(dx, 0.0)
    yactive, zactive = isactive(dy), isactive(dz)
    function mulmatmat_pullback(::NoRData)
        # dy += dx*z',  dz += y'*dx
        if yactive
            gy = dx * adjoint(z)
            for i in eachindex(dy, gy)
                dy[i] = increment!!(dy[i], gy[i])
            end
        end
        if zactive
            gz = adjoint(y) * dx
            for i in eachindex(dz, gz)
                dz[i] = increment!!(dz[i], gz[i])
            end
        end
        dx .= old_dx
        return (NoRData(), NoRData(), NoRData(), NoRData())
    end
    return CoDual(x, dx), mulmatmat_pullback
end
