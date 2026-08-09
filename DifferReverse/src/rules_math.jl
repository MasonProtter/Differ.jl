# Hand-written rrule!! for scalar math functions: Base.Math transcendentals, plus reverse-only
# rules for functions that inline to LLVM intrinsics before rrule!! dispatch can fire.
# Forward-mode frule!!s for the same functions live in DifferForwards/src/rules_math.jl.
#
# Every rule uses the closed-form derivative directly, never by differentiating through Base's
# actual implementation.
#
# As in `rrules.jl`, Base calls inside `rrule!!` bodies and pullback closures are qualified
# (`Base.sin`, not `sin`): these bodies get inlined into synthetic carrier IR, where a bare name
# re-embeds as an implicit-`using` GlobalRef that `Core.Compiler.verify_ir` rejects. Arithmetic/
# comparison operators stay unqualified since they lower to intrinsics, not generic-function
# GlobalRefs, matching `SumPullback`/`SumMapPullback` in `rrules.jl`.

# ===========================================================================
# exp
# ===========================================================================

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

struct SinhPullback
    x::Float64
end
(pb::SinhPullback)(seed::Float64) = (NoRData(), Base.cosh(pb.x)*seed)

function rrule!!(::CoDual{typeof(sinh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.sinh(x), NoFData()), SinhPullback(x)
end

struct CoshPullback
    x::Float64
end
(pb::CoshPullback)(seed::Float64) = (NoRData(), Base.sinh(pb.x)*seed)

function rrule!!(::CoDual{typeof(cosh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.cosh(x), NoFData()), CoshPullback(x)
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

struct AsinhPullback
    x::Float64
end
(pb::AsinhPullback)(seed::Float64) = (NoRData(), seed/Base.sqrt(pb.x^2+1))

function rrule!!(::CoDual{typeof(asinh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.asinh(x), NoFData()), AsinhPullback(x)
end

struct AcoshPullback
    x::Float64
end
(pb::AcoshPullback)(seed::Float64) = (NoRData(), seed/Base.sqrt(pb.x^2-1))

function rrule!!(::CoDual{typeof(acosh),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.acosh(x), NoFData()), AcoshPullback(x)
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

struct AsinPullback
    x::Float64
end
(pb::AsinPullback)(seed::Float64) = (NoRData(), seed/Base.sqrt(1-pb.x^2))

function rrule!!(::CoDual{typeof(asin),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.asin(x), NoFData()), AsinPullback(x)
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

struct AtanPullback
    x::Float64
end
(pb::AtanPullback)(seed::Float64) = (NoRData(), seed/(1+pb.x^2))

function rrule!!(::CoDual{typeof(atan),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.atan(x), NoFData()), AtanPullback(x)
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
