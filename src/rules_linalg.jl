# Hand-written frule!!/rrule!! for LinearAlgebra basics (dot/norm/*/tr/mul!).
#
# `dot`/`norm`/`*` (on `Matrix`/`Vector`) all bottom out in BLAS `ccall`s once inlined, which the
# dualization engine cannot see through. Forward mode does now have a per-target `:foreigncall` rule
# layer (`src/foreigncalls.jl`, ISSUES #62), but that changes nothing here: it registers bulk memory
# copies, and a BLAS kernel like `:cblas_ddot64_` is opaque native code with no rule and no prospect
# of one, so it bails with a located reason. A hand rule is therefore still the only way to
# differentiate these: there is no generic-recursion fallback, unlike an ordinary composite function.
# Each rule below computes the primal via a plain, untracked call to the real function (or an
# explicit loop) and supplies the tangent/gradient via the closed-form identity — never by trying
# to dualize the target's actual body.
#
# `transpose`/`adjoint` deliberately have NO rule here: both already differentiate correctly via
# the existing generic struct-tangent machinery (`tangent_type(Transpose{...})` resolves to a real
# `Tangent`), so a rule would be redundant, and risks a dispatch ambiguity with the generic
# fallback. See `test/test_linalg_rules.jl` for regression tests proving this.
#
# Bare `Base`/`LinearAlgebra` names are qualified throughout (`Base.:*`, `LinearAlgebra.dot`, ...),
# matching `src/rrules.jl`'s `Base.sin`/`Base.cos` qualification style — see the note at the top of
# that file for why an unqualified name is unsafe once a rule body gets embedded elsewhere.

import LinearAlgebra

# ---------------------------------------------------------------------------
# dot(x, y)
# ---------------------------------------------------------------------------

function frule!!(
    ::Dual{typeof(LinearAlgebra.dot)}, xd::Dual{Vector{Float64}}, yd::Dual{Vector{Float64}}
)
    x, dx = xd.x, xd.dx
    y, dy = yd.x, yd.dx
    Base.length(x) == Base.length(y) ||
        throw(DimensionMismatch("dot: vectors have different lengths"))
    s = LinearAlgebra.dot(x, y)
    ds = LinearAlgebra.dot(dx, y) + LinearAlgebra.dot(x, dy)
    return Dual(s, ds)
end

struct DotPullback
    x::Vector{Float64}
    y::Vector{Float64}
    dx::Vector{Float64}
    dy::Vector{Float64}
end
function (pb::DotPullback)(seed::Float64)
    x, y, dx, dy = pb.x, pb.y, pb.dx, pb.dy
    for i in Base.eachindex(x, y, dx, dy)
        dx[i] = increment!!(dx[i], seed * y[i])
        dy[i] = increment!!(dy[i], seed * x[i])
    end
    return (NoRData(), NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.dot),NoFData},
    ::AbstractCtx,
    xcd::CoDual{Vector{Float64},Vector{Float64}},
    ycd::CoDual{Vector{Float64},Vector{Float64}},
)
    x, y = primal(xcd), primal(ycd)
    Base.length(x) == Base.length(y) ||
        throw(DimensionMismatch("dot: vectors have different lengths"))
    s = LinearAlgebra.dot(x, y)
    return zero_fcodual(s), DotPullback(x, y, tangent(xcd), tangent(ycd))
end

# ---------------------------------------------------------------------------
# norm(x) — 2-norm only
# ---------------------------------------------------------------------------

function frule!!(::Dual{typeof(LinearAlgebra.norm)}, xd::Dual{Vector{Float64}})
    x, dx = xd.x, xd.dx
    nrm = LinearAlgebra.norm(x)
    dnrm = LinearAlgebra.dot(x, dx) / nrm
    return Dual(nrm, dnrm)
end

struct NormPullback
    x::Vector{Float64}
    dx::Vector{Float64}
    nrm::Float64
end
function (pb::NormPullback)(seed::Float64)
    x, dx, nrm = pb.x, pb.dx, pb.nrm
    c = seed / nrm
    for i in Base.eachindex(x, dx)
        dx[i] = increment!!(dx[i], c * x[i])
    end
    return (NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.norm),NoFData},
    ::AbstractCtx,
    xcd::CoDual{Vector{Float64},Vector{Float64}},
)
    x = primal(xcd)
    nrm = LinearAlgebra.norm(x)
    return zero_fcodual(nrm), NormPullback(x, tangent(xcd), nrm)
end

# ---------------------------------------------------------------------------
# tr(A) — explicit diagonal loop, mirroring `SumPullback`'s shape in `src/rrules.jl`.
# ---------------------------------------------------------------------------

function frule!!(::Dual{typeof(LinearAlgebra.tr)}, Ad::Dual{Matrix{Float64}})
    A, dA = Ad.x, Ad.dx
    n = Base.size(A, 1)
    s = 0.0
    ds = 0.0
    for i in 1:n
        s += A[i, i]
        ds += dA[i, i]
    end
    return Dual(s, ds)
end

struct TrPullback
    dA::Matrix{Float64}
end
function (pb::TrPullback)(seed::Float64)
    dA = pb.dA
    n = Base.size(dA, 1)
    for i in 1:n
        dA[i, i] = increment!!(dA[i, i], seed)
    end
    return (NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.tr),NoFData},
    ::AbstractCtx,
    Acd::CoDual{Matrix{Float64},Matrix{Float64}},
)
    A = primal(Acd)
    n = Base.size(A, 1)
    s = 0.0
    for i in 1:n
        s += A[i, i]
    end
    return zero_fcodual(s), TrPullback(tangent(Acd))
end

# ---------------------------------------------------------------------------
# `*` — matrix-vector and matrix-matrix, `Matrix{Float64}`/`Vector{Float64}` only. These method
# signatures are deliberately narrow (never `Any`) so they don't interfere with plain scalar `*`,
# which is handled entirely by the intrinsic layer (`mul_float`) and never reaches `frule!!`/
# `rrule!!` dispatch at all.
# ---------------------------------------------------------------------------

# --- matrix * vector ---

function frule!!(
    ::Dual{typeof(Base.:*)}, Ad::Dual{Matrix{Float64}}, xd::Dual{Vector{Float64}}
)
    A, dA = Ad.x, Ad.dx
    x, dx = xd.x, xd.dx
    y = Base.:*(A, x)
    dy = Base.:+(Base.:*(dA, x), Base.:*(A, dx))
    return Dual(y, dy)
end

struct MatVecMulPullback
    A::Matrix{Float64}
    x::Vector{Float64}
    dA::Matrix{Float64}
    dx::Vector{Float64}
    dy::Vector{Float64}
end
function (pb::MatVecMulPullback)(::NoRData)
    A, x, dA, dx, dy = pb.A, pb.x, pb.dA, pb.dx, pb.dy
    m, n = Base.size(A)
    for i in 1:m, j in 1:n
        dA[i, j] = increment!!(dA[i, j], dy[i] * x[j])
    end
    for j in 1:n
        s = 0.0
        for i in 1:m
            s += A[i, j] * dy[i]
        end
        dx[j] = increment!!(dx[j], s)
    end
    return (NoRData(), NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(Base.:*),NoFData},
    ::AbstractCtx,
    Acd::CoDual{Matrix{Float64},Matrix{Float64}},
    xcd::CoDual{Vector{Float64},Vector{Float64}},
)
    A, x = primal(Acd), primal(xcd)
    y = Base.:*(A, x)
    ycd = zero_fcodual(y)
    return ycd, MatVecMulPullback(A, x, tangent(Acd), tangent(xcd), tangent(ycd))
end

# --- matrix * matrix ---

function frule!!(
    ::Dual{typeof(Base.:*)}, Ad::Dual{Matrix{Float64}}, Bd::Dual{Matrix{Float64}}
)
    A, dA = Ad.x, Ad.dx
    B, dB = Bd.x, Bd.dx
    Y = Base.:*(A, B)
    dY = Base.:+(Base.:*(dA, B), Base.:*(A, dB))
    return Dual(Y, dY)
end

struct MatMulPullback
    A::Matrix{Float64}
    B::Matrix{Float64}
    dA::Matrix{Float64}
    dB::Matrix{Float64}
    dY::Matrix{Float64}
end
function (pb::MatMulPullback)(::NoRData)
    A, B, dA, dB, dY = pb.A, pb.B, pb.dA, pb.dB, pb.dY
    # Ā += dY * B',  B̄ += A' * dY
    dABt = Base.:*(dY, Base.adjoint(B))
    dAtB = Base.:*(Base.adjoint(A), dY)
    for i in Base.eachindex(dA, dABt)
        dA[i] = increment!!(dA[i], dABt[i])
    end
    for i in Base.eachindex(dB, dAtB)
        dB[i] = increment!!(dB[i], dAtB[i])
    end
    return (NoRData(), NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(Base.:*),NoFData},
    ::AbstractCtx,
    Acd::CoDual{Matrix{Float64},Matrix{Float64}},
    Bcd::CoDual{Matrix{Float64},Matrix{Float64}},
)
    A, B = primal(Acd), primal(Bcd)
    Y = Base.:*(A, B)
    Ycd = zero_fcodual(Y)
    return Ycd, MatMulPullback(A, B, tangent(Acd), tangent(Bcd), tangent(Ycd))
end

# ---------------------------------------------------------------------------
# mul!(C, A, B) — in-place `C = A*B`, `Matrix{Float64}`/`Vector{Float64}` only (3-arg form; the
# `α`/`β`-scaled 5-arg form is not covered). `C` is mutated, so both modes follow the mutating-array
# convention used by `map!` (`rules_broadcast.jl`): forward returns the same `Dual` (mutated shadow
# in place); reverse zeroes `C`'s fdata after reading the old contents as the backward seed (`C`'s
# previous value is overwritten, not accumulated, so whatever cotangent had built up on it before
# this call belongs to that overwritten value, not to `A`/`B`) and restores the old fdata afterwards
# so an earlier write to the same array (if any) still sees its own seed correctly.
# ---------------------------------------------------------------------------

# --- mul!(y, A, x) — matrix * vector ---

function frule!!(
    ::Dual{typeof(LinearAlgebra.mul!)}, yd::Dual{Vector{Float64}},
    Ad::Dual{Matrix{Float64}}, xd::Dual{Vector{Float64}},
)
    y, dy = yd.x, yd.dx
    A, dA = Ad.x, Ad.dx
    x, dx = xd.x, xd.dx
    LinearAlgebra.mul!(y, A, x)
    LinearAlgebra.mul!(dy, dA, x)
    LinearAlgebra.mul!(dy, A, dx, true, true)
    return yd
end

struct MulMatVecPullback
    A::Matrix{Float64}
    x::Vector{Float64}
    dA::Matrix{Float64}
    dx::Vector{Float64}
    dy::Vector{Float64}
    old_dy::Vector{Float64}
end
function (pb::MulMatVecPullback)(::NoRData)
    A, x, dA, dx, dy, old_dy = pb.A, pb.x, pb.dA, pb.dx, pb.dy, pb.old_dy
    m, n = Base.size(A)
    for i in 1:m, j in 1:n
        dA[i, j] = increment!!(dA[i, j], dy[i] * x[j])
    end
    for j in 1:n
        s = 0.0
        for i in 1:m
            s += A[i, j] * dy[i]
        end
        dx[j] = increment!!(dx[j], s)
    end
    dy .= old_dy
    return (NoRData(), NoRData(), NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.mul!),NoFData},
    ::AbstractCtx,
    ycd::CoDual{Vector{Float64},Vector{Float64}},
    Acd::CoDual{Matrix{Float64},Matrix{Float64}},
    xcd::CoDual{Vector{Float64},Vector{Float64}},
)
    y, dy = primal(ycd), tangent(ycd)
    A, x = primal(Acd), primal(xcd)
    old_dy = Base.copy(dy)
    LinearAlgebra.mul!(y, A, x)
    Base.fill!(dy, 0.0)
    return CoDual(y, dy), MulMatVecPullback(A, x, tangent(Acd), tangent(xcd), dy, old_dy)
end

# --- mul!(C, A, B) — matrix * matrix ---

function frule!!(
    ::Dual{typeof(LinearAlgebra.mul!)}, Cd::Dual{Matrix{Float64}},
    Ad::Dual{Matrix{Float64}}, Bd::Dual{Matrix{Float64}},
)
    C, dC = Cd.x, Cd.dx
    A, dA = Ad.x, Ad.dx
    B, dB = Bd.x, Bd.dx
    LinearAlgebra.mul!(C, A, B)
    LinearAlgebra.mul!(dC, dA, B)
    LinearAlgebra.mul!(dC, A, dB, true, true)
    return Cd
end

struct MulMatMatPullback
    A::Matrix{Float64}
    B::Matrix{Float64}
    dA::Matrix{Float64}
    dB::Matrix{Float64}
    dC::Matrix{Float64}
    old_dC::Matrix{Float64}
end
function (pb::MulMatMatPullback)(::NoRData)
    A, B, dA, dB, dC, old_dC = pb.A, pb.B, pb.dA, pb.dB, pb.dC, pb.old_dC
    # Ā += dC * B',  B̄ += A' * dC
    dABt = Base.:*(dC, Base.adjoint(B))
    dAtB = Base.:*(Base.adjoint(A), dC)
    for i in Base.eachindex(dA, dABt)
        dA[i] = increment!!(dA[i], dABt[i])
    end
    for i in Base.eachindex(dB, dAtB)
        dB[i] = increment!!(dB[i], dAtB[i])
    end
    dC .= old_dC
    return (NoRData(), NoRData(), NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(LinearAlgebra.mul!),NoFData},
    ::AbstractCtx,
    Ccd::CoDual{Matrix{Float64},Matrix{Float64}},
    Acd::CoDual{Matrix{Float64},Matrix{Float64}},
    Bcd::CoDual{Matrix{Float64},Matrix{Float64}},
)
    C, dC = primal(Ccd), tangent(Ccd)
    A, B = primal(Acd), primal(Bcd)
    old_dC = Base.copy(dC)
    LinearAlgebra.mul!(C, A, B)
    Base.fill!(dC, 0.0)
    return CoDual(C, dC), MulMatMatPullback(A, B, tangent(Acd), tangent(Bcd), dC, old_dC)
end
