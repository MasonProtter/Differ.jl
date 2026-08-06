# Hand-written frule!!/rrule!! for scalar math functions: Base.Math transcendentals, plus
# reverse-only rules for functions that inline to LLVM intrinsics before rrule!! dispatch can fire.
# See ISSUES.md #20.
#
# Every rule uses the closed-form derivative directly, never by differentiating through Base's
# actual implementation. Some of those internals (e.g. asin's bitcast/and_int precision trick) use
# intrinsics on the `@inactive_intrinsic` list (`src/intrinsics.jl`), which would silently produce a
# zero tangent rather than error.
#
# As in `rrules.jl`, Base calls inside `rrule!!` bodies and pullback closures are qualified
# (`Base.sin`, not `sin`): these bodies get inlined into synthetic carrier IR, where a bare name
# re-embeds as an implicit-`using` GlobalRef that `Core.Compiler.verify_ir` rejects. `frule!!` bodies
# are ordinary compiled methods, not synthetic IR, so they keep bare names, matching `frules.jl`.
# Arithmetic/comparison operators stay unqualified everywhere since they lower to intrinsics, not
# generic-function GlobalRefs, matching `SumPullback`/`SumMapPullback` in `rrules.jl`.

# ===========================================================================
# exp
# ===========================================================================

function frule!!(::Dual{typeof(exp)}, (; x, dx)::Dual)
    y = exp(x)
    Dual(y, y*dx)
end

struct ExpPullback
    y::Float64
end
(pb::ExpPullback)(seed::Float64) = (NoRData(), pb.y*seed)

function rrule!!(::CoDual{typeof(exp),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    y = Base.exp(primal(xcd))
    return CoDual(y, NoFData()), ExpPullback(y)
end

# ===========================================================================
# log
# ===========================================================================

function frule!!(::Dual{typeof(log)}, (; x, dx)::Dual)
    Dual(log(x), dx/x)
end

struct LogPullback
    x::Float64
end
(pb::LogPullback)(seed::Float64) = (NoRData(), seed/pb.x)

function rrule!!(::CoDual{typeof(log),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.log(x), NoFData()), LogPullback(x)
end

# ===========================================================================
# log1p
# ===========================================================================

function frule!!(::Dual{typeof(log1p)}, (; x, dx)::Dual)
    Dual(log1p(x), dx/(1+x))
end

struct Log1pPullback
    x::Float64
end
(pb::Log1pPullback)(seed::Float64) = (NoRData(), seed/(1+pb.x))

function rrule!!(::CoDual{typeof(log1p),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.log1p(x), NoFData()), Log1pPullback(x)
end

# ===========================================================================
# expm1
# ===========================================================================

function frule!!(::Dual{typeof(expm1)}, (; x, dx)::Dual)
    Dual(expm1(x), exp(x)*dx)
end

struct Expm1Pullback
    x::Float64
end
(pb::Expm1Pullback)(seed::Float64) = (NoRData(), Base.exp(pb.x)*seed)

function rrule!!(::CoDual{typeof(expm1),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.expm1(x), NoFData()), Expm1Pullback(x)
end

# ===========================================================================
# log2
# ===========================================================================

function frule!!(::Dual{typeof(log2)}, (; x, dx)::Dual)
    Dual(log2(x), dx/(x*log(2)))
end

struct Log2Pullback
    x::Float64
end
(pb::Log2Pullback)(seed::Float64) = (NoRData(), seed/(pb.x*Base.log(2)))

function rrule!!(::CoDual{typeof(log2),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.log2(x), NoFData()), Log2Pullback(x)
end

# ===========================================================================
# log10
# ===========================================================================

function frule!!(::Dual{typeof(log10)}, (; x, dx)::Dual)
    Dual(log10(x), dx/(x*log(10)))
end

struct Log10Pullback
    x::Float64
end
(pb::Log10Pullback)(seed::Float64) = (NoRData(), seed/(pb.x*Base.log(10)))

function rrule!!(::CoDual{typeof(log10),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.log10(x), NoFData()), Log10Pullback(x)
end

# ===========================================================================
# exp2
# ===========================================================================

function frule!!(::Dual{typeof(exp2)}, (; x, dx)::Dual)
    y = exp2(x)
    Dual(y, y*log(2)*dx)
end

struct Exp2Pullback
    y::Float64
end
(pb::Exp2Pullback)(seed::Float64) = (NoRData(), pb.y*Base.log(2)*seed)

function rrule!!(::CoDual{typeof(exp2),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    y = Base.exp2(primal(xcd))
    return CoDual(y, NoFData()), Exp2Pullback(y)
end

# ===========================================================================
# exp10
# ===========================================================================

function frule!!(::Dual{typeof(exp10)}, (; x, dx)::Dual)
    y = exp10(x)
    Dual(y, y*log(10)*dx)
end

struct Exp10Pullback
    y::Float64
end
(pb::Exp10Pullback)(seed::Float64) = (NoRData(), pb.y*Base.log(10)*seed)

function rrule!!(::CoDual{typeof(exp10),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    y = Base.exp10(primal(xcd))
    return CoDual(y, NoFData()), Exp10Pullback(y)
end

# ===========================================================================
# sinh / cosh / tanh
# ===========================================================================

function frule!!(::Dual{typeof(sinh)}, (; x, dx)::Dual)
    Dual(sinh(x), cosh(x)*dx)
end

struct SinhPullback
    x::Float64
end
(pb::SinhPullback)(seed::Float64) = (NoRData(), Base.cosh(pb.x)*seed)

function rrule!!(::CoDual{typeof(sinh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.sinh(x), NoFData()), SinhPullback(x)
end

function frule!!(::Dual{typeof(cosh)}, (; x, dx)::Dual)
    Dual(cosh(x), sinh(x)*dx)
end

struct CoshPullback
    x::Float64
end
(pb::CoshPullback)(seed::Float64) = (NoRData(), Base.sinh(pb.x)*seed)

function rrule!!(::CoDual{typeof(cosh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.cosh(x), NoFData()), CoshPullback(x)
end

function frule!!(::Dual{typeof(tanh)}, (; x, dx)::Dual)
    y = tanh(x)
    Dual(y, (1-y^2)*dx)
end

struct TanhPullback
    y::Float64
end
(pb::TanhPullback)(seed::Float64) = (NoRData(), (1-pb.y^2)*seed)

function rrule!!(::CoDual{typeof(tanh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    y = Base.tanh(primal(xcd))
    return CoDual(y, NoFData()), TanhPullback(y)
end

# ===========================================================================
# asinh / acosh / atanh
# ===========================================================================

function frule!!(::Dual{typeof(asinh)}, (; x, dx)::Dual)
    Dual(asinh(x), dx/sqrt(x^2+1))
end

struct AsinhPullback
    x::Float64
end
(pb::AsinhPullback)(seed::Float64) = (NoRData(), seed/Base.sqrt(pb.x^2+1))

function rrule!!(::CoDual{typeof(asinh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.asinh(x), NoFData()), AsinhPullback(x)
end

function frule!!(::Dual{typeof(acosh)}, (; x, dx)::Dual)
    Dual(acosh(x), dx/sqrt(x^2-1))
end

struct AcoshPullback
    x::Float64
end
(pb::AcoshPullback)(seed::Float64) = (NoRData(), seed/Base.sqrt(pb.x^2-1))

function rrule!!(::CoDual{typeof(acosh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.acosh(x), NoFData()), AcoshPullback(x)
end

function frule!!(::Dual{typeof(atanh)}, (; x, dx)::Dual)
    Dual(atanh(x), dx/(1-x^2))
end

struct AtanhPullback
    x::Float64
end
(pb::AtanhPullback)(seed::Float64) = (NoRData(), seed/(1-pb.x^2))

function rrule!!(::CoDual{typeof(atanh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.atanh(x), NoFData()), AtanhPullback(x)
end

# ===========================================================================
# asin / acos
# ===========================================================================

function frule!!(::Dual{typeof(asin)}, (; x, dx)::Dual)
    Dual(asin(x), dx/sqrt(1-x^2))
end

struct AsinPullback
    x::Float64
end
(pb::AsinPullback)(seed::Float64) = (NoRData(), seed/Base.sqrt(1-pb.x^2))

function rrule!!(::CoDual{typeof(asin),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.asin(x), NoFData()), AsinPullback(x)
end

function frule!!(::Dual{typeof(acos)}, (; x, dx)::Dual)
    Dual(acos(x), -dx/sqrt(1-x^2))
end

struct AcosPullback
    x::Float64
end
(pb::AcosPullback)(seed::Float64) = (NoRData(), -seed/Base.sqrt(1-pb.x^2))

function rrule!!(::CoDual{typeof(acos),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.acos(x), NoFData()), AcosPullback(x)
end

# ===========================================================================
# atan (1-arg and 2-arg)
# ===========================================================================

function frule!!(::Dual{typeof(atan)}, (; x, dx)::Dual)
    Dual(atan(x), dx/(1+x^2))
end

struct AtanPullback
    x::Float64
end
(pb::AtanPullback)(seed::Float64) = (NoRData(), seed/(1+pb.x^2))

function rrule!!(::CoDual{typeof(atan),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.atan(x), NoFData()), AtanPullback(x)
end

function frule!!(::Dual{typeof(atan)}, dy::Dual, dx::Dual)
    y, dyv = primal(dy), tangent(dy)
    x, dxv = primal(dx), tangent(dx)
    r2 = x^2+y^2
    Dual(atan(y, x), (x*dyv-y*dxv)/r2)
end

struct Atan2Pullback
    x::Float64
    y::Float64
end
function (pb::Atan2Pullback)(seed::Float64)
    x, y = pb.x, pb.y
    r2 = x^2+y^2
    return (NoRData(), x*seed/r2, -y*seed/r2)
end

function rrule!!(
    ::CoDual{typeof(atan),NoFData}, ::AbstractCtx,
    ycd::CoDual{Float64,NoFData}, xcd::CoDual{Float64,NoFData},
)
    y, x = primal(ycd), primal(xcd)
    return CoDual(Base.atan(y, x), NoFData()), Atan2Pullback(x, y)
end

# ===========================================================================
# cbrt
# ===========================================================================

function frule!!(::Dual{typeof(cbrt)}, (; x, dx)::Dual)
    y = cbrt(x)
    Dual(y, dx/(3*y^2))
end

struct CbrtPullback
    y::Float64
end
(pb::CbrtPullback)(seed::Float64) = (NoRData(), seed/(3*pb.y^2))

function rrule!!(::CoDual{typeof(cbrt),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    y = Base.cbrt(primal(xcd))
    return CoDual(y, NoFData()), CbrtPullback(y)
end

# ===========================================================================
# ^ (x::Float64, y::Float64) — non-literal real exponent
# ===========================================================================

function frule!!(::Dual{typeof(^)}, dx::Dual, dy::Dual)
    x, dxv = primal(dx), tangent(dx)
    y, dyv = primal(dy), tangent(dy)
    yp = x^y
    Dual(yp, y*x^(y-1)*dxv + yp*log(x)*dyv)
end

struct PowPullback
    x::Float64
    y::Float64
    yp::Float64
end
function (pb::PowPullback)(seed::Float64)
    x, y, yp = pb.x, pb.y, pb.yp
    return (NoRData(), y*Base.:^(x, y-1)*seed, yp*Base.log(x)*seed)
end

function rrule!!(
    ::CoDual{typeof(^),NoFData}, ::AbstractCtx,
    xcd::CoDual{Float64,NoFData}, ycd::CoDual{Float64,NoFData},
)
    x, y = primal(xcd), primal(ycd)
    yp = Base.:^(x, y)
    return CoDual(yp, NoFData()), PowPullback(x, y, yp)
end

# ===========================================================================
# hypot(x, y)
# ===========================================================================

function frule!!(::Dual{typeof(hypot)}, (; x, dx)::Dual, (; y, dy)::Dual)
    r = hypot(x, y)
    Dual(r, (x*dx+y*dy)/r)
end

struct HypotPullback
    x::Float64
    y::Float64
    r::Float64
end
function (pb::HypotPullback)(seed::Float64)
    return (NoRData(), pb.x/pb.r*seed, pb.y/pb.r*seed)
end

function rrule!!(
    ::CoDual{typeof(hypot),NoFData}, ::AbstractCtx,
    xcd::CoDual{Float64,NoFData}, ycd::CoDual{Float64,NoFData},
)
    x, y = primal(xcd), primal(ycd)
    r = Base.hypot(x, y)
    return CoDual(r, NoFData()), HypotPullback(x, y, r)
end

# ===========================================================================
# sqrt(::Complex)
# ===========================================================================

function frule!!(::Dual{typeof(sqrt)}, dz::Dual{ComplexF64})
    z = primal(dz)
    dzt = tangent(dz)
    dz_re = get_tangent_field(dzt, :re)
    dz_im = get_tangent_field(dzt, :im)
    y = sqrt(z)
    dy = Complex(dz_re, dz_im)/(2*y)
    Dual(y, build_tangent(ComplexF64, real(dy), imag(dy)))
end

# Reverse mode for a holomorphic scalar map f: the output rdata seed s = sr + i*si (read straight
# off the (re, im) fields, no conjugation) pulls back via conj(f'(z)) * s — the standard adjoint of
# the real 2x2 Jacobian of a holomorphic map.
struct SqrtComplexPullback
    z::ComplexF64
    y::ComplexF64
end
function (pb::SqrtComplexPullback)(seed::RData{@NamedTuple{re::Float64,im::Float64}})
    s = Base.Complex(seed.data.re, seed.data.im)
    fprime = 1/(2*pb.y)
    dz = Base.conj(fprime)*s
    return (NoRData(), RData((re=Base.real(dz), im=Base.imag(dz))))
end

function rrule!!(
    ::CoDual{typeof(sqrt),NoFData}, ::AbstractCtx, zcd::CoDual{ComplexF64,NoFData}
)
    z = primal(zcd)
    y = Base.sqrt(z)
    return CoDual(y, NoFData()), SqrtComplexPullback(z, y)
end

# ===========================================================================
# Reverse-only rules — these inline straight to an LLVM intrinsic before a call-level rrule!! gets
# a chance to fire. Forward mode already handles them via `src/intrinsics.jl`.
# ===========================================================================

struct AbsPullback
    x::Float64
end
(pb::AbsPullback)(seed::Float64) = (NoRData(), Base.sign(pb.x)*seed)

function rrule!!(::CoDual{typeof(abs),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.abs(x), NoFData()), AbsPullback(x)
end

struct MaxPullback
    x::Float64
    y::Float64
end
function (pb::MaxPullback)(seed::Float64)
    x_wins = pb.x >= pb.y
    return (NoRData(), x_wins ? seed : 0.0, x_wins ? 0.0 : seed)
end

function rrule!!(
    ::CoDual{typeof(max),NoFData}, ::AbstractCtx,
    xcd::CoDual{Float64,NoFData}, ycd::CoDual{Float64,NoFData},
)
    x, y = primal(xcd), primal(ycd)
    return CoDual(Base.max(x, y), NoFData()), MaxPullback(x, y)
end

struct MinPullback
    x::Float64
    y::Float64
end
function (pb::MinPullback)(seed::Float64)
    x_wins = pb.x <= pb.y
    return (NoRData(), x_wins ? seed : 0.0, x_wins ? 0.0 : seed)
end

function rrule!!(
    ::CoDual{typeof(min),NoFData}, ::AbstractCtx,
    xcd::CoDual{Float64,NoFData}, ycd::CoDual{Float64,NoFData},
)
    x, y = primal(xcd), primal(ycd)
    return CoDual(Base.min(x, y), NoFData()), MinPullback(x, y)
end

struct SignPullback end
(pb::SignPullback)(::Float64) = (NoRData(), 0.0)

function rrule!!(::CoDual{typeof(sign),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    return CoDual(Base.sign(primal(xcd)), NoFData()), SignPullback()
end

struct CopysignPullback
    x::Float64
    y::Float64
end
function (pb::CopysignPullback)(seed::Float64)
    return (NoRData(), Base.sign(pb.x)*Base.sign(pb.y)*seed, 0.0)
end

function rrule!!(
    ::CoDual{typeof(copysign),NoFData}, ::AbstractCtx,
    xcd::CoDual{Float64,NoFData}, ycd::CoDual{Float64,NoFData},
)
    x, y = primal(xcd), primal(ycd)
    return CoDual(Base.copysign(x, y), NoFData()), CopysignPullback(x, y)
end

struct FmaPullback
    x::Float64
    y::Float64
end
(pb::FmaPullback)(seed::Float64) = (NoRData(), pb.y*seed, pb.x*seed, seed)

function rrule!!(
    ::CoDual{typeof(fma),NoFData}, ::AbstractCtx,
    xcd::CoDual{Float64,NoFData}, ycd::CoDual{Float64,NoFData}, zcd::CoDual{Float64,NoFData},
)
    x, y, z = primal(xcd), primal(ycd), primal(zcd)
    return CoDual(Base.fma(x, y, z), NoFData()), FmaPullback(x, y)
end

struct MuladdPullback
    x::Float64
    y::Float64
end
(pb::MuladdPullback)(seed::Float64) = (NoRData(), pb.y*seed, pb.x*seed, seed)

function rrule!!(
    ::CoDual{typeof(muladd),NoFData}, ::AbstractCtx,
    xcd::CoDual{Float64,NoFData}, ycd::CoDual{Float64,NoFData}, zcd::CoDual{Float64,NoFData},
)
    x, y, z = primal(xcd), primal(ycd), primal(zcd)
    return CoDual(Base.muladd(x, y, z), NoFData()), MuladdPullback(x, y)
end

struct Abs2Pullback
    x::Float64
end
(pb::Abs2Pullback)(seed::Float64) = (NoRData(), 2*pb.x*seed)

function rrule!!(::CoDual{typeof(abs2),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.abs2(x), NoFData()), Abs2Pullback(x)
end

struct InvPullback
    y::Float64
end
(pb::InvPullback)(seed::Float64) = (NoRData(), -pb.y^2*seed)

function rrule!!(::CoDual{typeof(inv),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    y = Base.inv(primal(xcd))
    return CoDual(y, NoFData()), InvPullback(y)
end
