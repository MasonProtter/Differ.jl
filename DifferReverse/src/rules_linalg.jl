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
#
# `Base`/`LinearAlgebra` names are qualified throughout (`Base.:*`, `LinearAlgebra.dot`, ...) for
# the same GlobalRef-inlining reason as `rrules.jl`.

# ---------------------------------------------------------------------------
# dot(x, y)
# ---------------------------------------------------------------------------

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.dot),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Vector{Float64},Vector{Float64}},
    (; y, dy)::CoDual{Vector{Float64},Vector{Float64}},
)
    Base.length(x) == Base.length(y) ||
        throw(DimensionMismatch("dot: vectors have different lengths"))
    s = LinearAlgebra.dot(x, y)
    function dot_pullback(dz)
        for i in Base.eachindex(x, y, dx, dy)
            dx[i] = increment!!(dx[i], dz * y[i])
            dy[i] = increment!!(dy[i], dz * x[i])
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
    (; x, dx)::CoDual{Vector{Float64},Vector{Float64}},
)
    nrm = LinearAlgebra.norm(x)
    function norm_pullback(dy)
        c = dy / nrm
        for i in Base.eachindex(x, dx)
            dx[i] = increment!!(dx[i], c * x[i])
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
    (; x, dx)::CoDual{Matrix{Float64},Matrix{Float64}},
)
    n = Base.size(x, 1)
    s = 0.0
    for i in 1:n
        s += x[i, i]
    end
    function tr_pullback(dy)
        for i in 1:n
            dx[i, i] = increment!!(dx[i, i], dy)
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
    (; x, dx)::CoDual{Matrix{Float64},Matrix{Float64}},
    (; y, dy)::CoDual{Vector{Float64},Vector{Float64}},
)
    z = Base.:*(x, y)
    zcd = zero_fcodual(z)
    dz = tangent(zcd)
    m, n = Base.size(x)
    function matvecmul_pullback(::NoRData)
        for i in 1:m, j in 1:n
            dx[i, j] = increment!!(dx[i, j], dz[i] * y[j])
        end
        for j in 1:n
            s = 0.0
            for i in 1:m
                s += x[i, j] * dz[i]
            end
            dy[j] = increment!!(dy[j], s)
        end
        return (NoRData(), NoRData(), NoRData())
    end
    return zcd, matvecmul_pullback
end

# --- matrix * matrix — `x`/`y` are the two matrix arguments ---

function rrule!!(
    ::CoDual{typeof(Base.:*),NoFData},
    ::AbstractCtx,
    (; x, dx)::CoDual{Matrix{Float64},Matrix{Float64}},
    (; y, dy)::CoDual{Matrix{Float64},Matrix{Float64}},
)
    z = Base.:*(x, y)
    zcd = zero_fcodual(z)
    dz = tangent(zcd)
    function matmul_pullback(::NoRData)
        # dx += dz*y',  dy += x'*dz
        gx = Base.:*(dz, Base.adjoint(y))
        gy = Base.:*(Base.adjoint(x), dz)
        for i in Base.eachindex(dx, gx)
            dx[i] = increment!!(dx[i], gx[i])
        end
        for i in Base.eachindex(dy, gy)
            dy[i] = increment!!(dy[i], gy[i])
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
    (; x, dx)::CoDual{Vector{Float64},Vector{Float64}},
    (; y, dy)::CoDual{Matrix{Float64},Matrix{Float64}},
    (; z, dz)::CoDual{Vector{Float64},Vector{Float64}},
)
    old_dx = Base.copy(dx)
    LinearAlgebra.mul!(x, y, z)
    Base.fill!(dx, 0.0)
    m, n = Base.size(y)
    function mulmatvec_pullback(::NoRData)
        for i in 1:m, j in 1:n
            dy[i, j] = increment!!(dy[i, j], dx[i] * z[j])
        end
        for j in 1:n
            s = 0.0
            for i in 1:m
                s += y[i, j] * dx[i]
            end
            dz[j] = increment!!(dz[j], s)
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
    (; x, dx)::CoDual{Matrix{Float64},Matrix{Float64}},
    (; y, dy)::CoDual{Matrix{Float64},Matrix{Float64}},
    (; z, dz)::CoDual{Matrix{Float64},Matrix{Float64}},
)
    old_dx = Base.copy(dx)
    LinearAlgebra.mul!(x, y, z)
    Base.fill!(dx, 0.0)
    function mulmatmat_pullback(::NoRData)
        # dy += dx*z',  dz += y'*dx
        gy = Base.:*(dx, Base.adjoint(z))
        gz = Base.:*(Base.adjoint(y), dx)
        for i in Base.eachindex(dy, gy)
            dy[i] = increment!!(dy[i], gy[i])
        end
        for i in Base.eachindex(dz, gz)
            dz[i] = increment!!(dz[i], gz[i])
        end
        dx .= old_dx
        return (NoRData(), NoRData(), NoRData(), NoRData())
    end
    return CoDual(x, dx), mulmatmat_pullback
end
