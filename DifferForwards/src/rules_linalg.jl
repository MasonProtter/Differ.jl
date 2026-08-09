# Hand-written frule!! for LinearAlgebra basics (dot/norm/*/tr/mul!). Reverse-mode rrule!!s for
# the same functions live in DifferReverse/src/rules_linalg.jl.
#
# `dot`/`norm`/`*` (on `Matrix`/`Vector`) bottom out in BLAS `ccall`s once inlined, which the
# dualization engine can't see through. Forward mode has a per-target `:foreigncall` rule layer
# (`src/foreigncalls.jl`, ISSUES #62), but that doesn't help here: it registers bulk memory copies,
# while a BLAS kernel like `:cblas_ddot64_` is opaque native code with no rule and no prospect of
# one. A hand rule is the only way to differentiate these — there's no generic-recursion fallback
# like an ordinary composite function gets. Each rule computes the primal via a plain untracked call
# (or an explicit loop) and supplies the tangent via the closed-form identity, never by dualizing
# the target's actual body.
#
# `transpose`/`adjoint` deliberately have no rule here: both already differentiate correctly via the
# generic struct-tangent machinery (`tangent_type(Transpose{...})` resolves to a real `Tangent`), so
# a rule would be redundant and risk dispatch ambiguity with the generic fallback. See
# `test/test_linalg_rules.jl` for the regression test.

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

# ---------------------------------------------------------------------------
# norm(x) — 2-norm only
# ---------------------------------------------------------------------------

function frule!!(::Dual{typeof(LinearAlgebra.norm)}, xd::Dual{Vector{Float64}})
    x, dx = xd.x, xd.dx
    nrm = LinearAlgebra.norm(x)
    dnrm = LinearAlgebra.dot(x, dx) / nrm
    return Dual(nrm, dnrm)
end

# ---------------------------------------------------------------------------
# tr(A) — explicit diagonal loop, mirroring `SumPullback`'s shape in `rrules.jl`.
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

# ---------------------------------------------------------------------------
# `*` — matrix-vector and matrix-matrix, `Matrix{Float64}`/`Vector{Float64}` only. Signatures are
# deliberately narrow (never `Any`) so they don't interfere with plain scalar `*`, which the
# intrinsic layer (`mul_float`) handles entirely and never reaches `frule!!`/`rrule!!` dispatch.
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

# ---------------------------------------------------------------------------
# mul!(C, A, B) — in-place `C = A*B`, `Matrix{Float64}`/`Vector{Float64}` only (3-arg form; the
# α/β-scaled 5-arg form isn't covered). `C` is mutated: forward returns the same `Dual` with the
# shadow mutated in place, mirroring `map!`'s mutating-array convention (`rules_broadcast.jl`).
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
