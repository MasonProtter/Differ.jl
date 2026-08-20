# Hand-written `rrule!!` for SpecialFunctions.jl's scalar functions. The forward-mode `frule!!`s
# for the same functions live in `DifferForwards/ext/DifferForwardsSpecialFunctionsExt.jl`, and
# that file's header lists what neither mode covers and why.
#
# The derivatives are the closed forms SpecialFunctions' own ChainRulesCore extension uses. Rules
# are not optional here the way they are for a plain Julia function: nearly every one of these
# bottoms out in a `ccall` to openspecfun, which the transform cannot see through.
#
# Some arguments have no implemented derivative — a Bessel order, an incomplete gamma or
# exponential-integral parameter — because no closed form for one is implemented upstream either.
# Constancy is visible in the shadow type by the time the rule runs, so `_no_param_rdata` decides
# from it: an `Inactive` shadow, or an integer parameter with no tangent space, gets `NoRData()`;
# a live float shadow is a caller genuinely asking for the missing derivative, and is refused
# eagerly at the forwards call. Literals reach the rule as `CoDual{P,Inactive}`, so `besselj(1.5, x)`
# still works — same refusal forward mode makes, at the same point.
module DifferReverseSpecialFunctionsExt

using SpecialFunctions
using SpecialFunctions: sqrtπ, invπ

import DifferReverse: CoDual, AbstractCtx, rrule!!, NoFData, NoRData, Inactive, primal, @ifactive

# The order/parameter slot: an integer, or any real — held constant or not.
const ParamCoDual = CoDual{<:Real,<:Union{NoFData,Inactive}}
# The same slot restricted to integers, which have no tangent space at all.
const IntParamCoDual = CoDual{<:Integer,<:Union{NoFData,Inactive}}

# The rdata for a parameter slot with no implemented derivative. Nothing to hand back when the
# parameter is held constant or is an integer. A live float shadow asks for the missing derivative,
# so refuse it — called from the rule body, before the pullback is built, so it throws at the
# forwards call rather than inside the reverse pass.
_no_param_rdata(::CoDual{<:Any,Inactive}, ::Any, ::Any) = NoRData()
_no_param_rdata(::CoDual{<:Integer,NoFData}, ::Any, ::Any) = NoRData()
_no_param_rdata(::CoDual{<:AbstractFloat,NoFData}, f, what) =
    error("Differ: the derivative of `", f, "` with respect to its ", what,
          " is not implemented — hold that argument constant")

# ===========================================================================
# Airy functions
# ===========================================================================

function rrule!!(::CoDual{typeof(airyai),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    airyai_pullback(dy) = (NoRData(), @ifactive(dx, airyaiprime(x)*dy))
    CoDual(airyai(x), NoFData()), airyai_pullback
end

function rrule!!(::CoDual{typeof(airyaix),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = airyaix(x)
    airyaix_pullback(dy) = (NoRData(), @ifactive(dx, (airyaiprimex(x) + sqrt(x)*y)*dy))
    CoDual(y, NoFData()), airyaix_pullback
end

function rrule!!(::CoDual{typeof(airyaiprime),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    airyaiprime_pullback(dy) = (NoRData(), @ifactive(dx, x*airyai(x)*dy))
    CoDual(airyaiprime(x), NoFData()), airyaiprime_pullback
end

function rrule!!(::CoDual{typeof(airyaiprimex),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = airyaiprimex(x)
    airyaiprimex_pullback(dy) = (NoRData(), @ifactive(dx, (x*airyaix(x) + sqrt(x)*y)*dy))
    CoDual(y, NoFData()), airyaiprimex_pullback
end

function rrule!!(::CoDual{typeof(airybi),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    airybi_pullback(dy) = (NoRData(), @ifactive(dx, airybiprime(x)*dy))
    CoDual(airybi(x), NoFData()), airybi_pullback
end

function rrule!!(::CoDual{typeof(airybiprime),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    airybiprime_pullback(dy) = (NoRData(), @ifactive(dx, x*airybi(x)*dy))
    CoDual(airybiprime(x), NoFData()), airybiprime_pullback
end

# ===========================================================================
# Bessel functions of fixed order
# ===========================================================================

function rrule!!(::CoDual{typeof(besselj0),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    besselj0_pullback(dy) = (NoRData(), @ifactive(dx, -besselj1(x)*dy))
    CoDual(besselj0(x), NoFData()), besselj0_pullback
end

function rrule!!(::CoDual{typeof(besselj1),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    besselj1_pullback(dy) = (NoRData(), @ifactive(dx, (besselj0(x) - besselj(2, x))/2*dy))
    CoDual(besselj1(x), NoFData()), besselj1_pullback
end

function rrule!!(::CoDual{typeof(bessely0),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    bessely0_pullback(dy) = (NoRData(), @ifactive(dx, -bessely1(x)*dy))
    CoDual(bessely0(x), NoFData()), bessely0_pullback
end

function rrule!!(::CoDual{typeof(bessely1),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    bessely1_pullback(dy) = (NoRData(), @ifactive(dx, (bessely0(x) - bessely(2, x))/2*dy))
    CoDual(bessely1(x), NoFData()), bessely1_pullback
end

# ===========================================================================
# Bessel functions of general order — differentiable in the argument only
# ===========================================================================

function rrule!!(::CoDual{typeof(besselj),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "besselj", "order")
    besselj_pullback(dy) = (NoRData(), νrd, @ifactive(dx, (besselj(ν-1, x) - besselj(ν+1, x))/2*dy))
    CoDual(besselj(ν, x), NoFData()), besselj_pullback
end

function rrule!!(::CoDual{typeof(besseli),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "besseli", "order")
    besseli_pullback(dy) = (NoRData(), νrd, @ifactive(dx, (besseli(ν-1, x) + besseli(ν+1, x))/2*dy))
    CoDual(besseli(ν, x), NoFData()), besseli_pullback
end

function rrule!!(::CoDual{typeof(bessely),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "bessely", "order")
    bessely_pullback(dy) = (NoRData(), νrd, @ifactive(dx, (bessely(ν-1, x) - bessely(ν+1, x))/2*dy))
    CoDual(bessely(ν, x), NoFData()), bessely_pullback
end

function rrule!!(::CoDual{typeof(besselk),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "besselk", "order")
    besselk_pullback(dy) = (NoRData(), νrd,
                            @ifactive(dx, -(besselk(ν-1, x) + besselk(ν+1, x))/2*dy))
    CoDual(besselk(ν, x), NoFData()), besselk_pullback
end

function rrule!!(::CoDual{typeof(besselkx),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "besselkx", "order")
    y = besselkx(ν, x)
    besselkx_pullback(dy) = (NoRData(), νrd,
                             @ifactive(dx, (-(besselkx(ν-1, x) + besselkx(ν+1, x))/2 + y)*dy))
    CoDual(y, NoFData()), besselkx_pullback
end

# The scaling factor — `exp(-|Re x|)` for `besselix`, `exp(-|Im x|)` for the other two — is not
# holomorphic, so these three cover real arguments only. On the real axis its own derivative is the
# `-sign(x)*y` term in `besselix`, and vanishes for the other two.

function rrule!!(::CoDual{typeof(besselix),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "besselix", "order")
    y = besselix(ν, x)
    besselix_pullback(dy) = (NoRData(), νrd,
                             @ifactive(dx, ((besselix(ν-1, x) + besselix(ν+1, x))/2 - sign(x)*y)*dy))
    CoDual(y, NoFData()), besselix_pullback
end

function rrule!!(::CoDual{typeof(besseljx),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "besseljx", "order")
    besseljx_pullback(dy) = (NoRData(), νrd,
                             @ifactive(dx, (besseljx(ν-1, x) - besseljx(ν+1, x))/2*dy))
    CoDual(besseljx(ν, x), NoFData()), besseljx_pullback
end

function rrule!!(::CoDual{typeof(besselyx),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "besselyx", "order")
    besselyx_pullback(dy) = (NoRData(), νrd,
                             @ifactive(dx, (besselyx(ν-1, x) - besselyx(ν+1, x))/2*dy))
    CoDual(besselyx(ν, x), NoFData()), besselyx_pullback
end

# ===========================================================================
# dawson
# ===========================================================================

function rrule!!(::CoDual{typeof(dawson),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = dawson(x)
    dawson_pullback(dy) = (NoRData(), @ifactive(dx, (1 - 2*x*y)*dy))
    CoDual(y, NoFData()), dawson_pullback
end

# ===========================================================================
# gamma and friends
# ===========================================================================

function rrule!!(::CoDual{typeof(gamma),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = gamma(x)
    gamma_pullback(dy) = (NoRData(), @ifactive(dx, y*digamma(x)*dy))
    CoDual(y, NoFData()), gamma_pullback
end

function rrule!!(::CoDual{typeof(loggamma),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    loggamma_pullback(dy) = (NoRData(), @ifactive(dx, digamma(x)*dy))
    CoDual(loggamma(x), NoFData()), loggamma_pullback
end

function rrule!!(::CoDual{typeof(digamma),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    digamma_pullback(dy) = (NoRData(), @ifactive(dx, trigamma(x)*dy))
    CoDual(digamma(x), NoFData()), digamma_pullback
end

function rrule!!(::CoDual{typeof(trigamma),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    trigamma_pullback(dy) = (NoRData(), @ifactive(dx, polygamma(2, x)*dy))
    CoDual(trigamma(x), NoFData()), trigamma_pullback
end

function rrule!!(::CoDual{typeof(invdigamma),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = invdigamma(x)
    invdigamma_pullback(dy) = (NoRData(), @ifactive(dx, inv(trigamma(y))*dy))
    CoDual(y, NoFData()), invdigamma_pullback
end

# The order `m` is an `Integer`, so it has no tangent space to begin with — no activity check.
function rrule!!(::CoDual{typeof(polygamma),NoFData}, ::AbstractCtx,
                 mcd::IntParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    m = primal(mcd)
    polygamma_pullback(dz) = (NoRData(), NoRData(), @ifactive(dx, polygamma(m+1, x)*dz))
    CoDual(polygamma(m, x), NoFData()), polygamma_pullback
end

function rrule!!(::CoDual{typeof(beta),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
                 (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}})
    b = beta(x, y)
    dxy = digamma(x + y)
    beta_pullback(dz) = (NoRData(), @ifactive(dx, b*(digamma(x) - dxy)*dz),
                         @ifactive(dy, b*(digamma(y) - dxy)*dz))
    CoDual(b, NoFData()), beta_pullback
end

function rrule!!(::CoDual{typeof(logbeta),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
                 (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}})
    dxy = digamma(x + y)
    logbeta_pullback(dz) = (NoRData(), @ifactive(dx, (digamma(x) - dxy)*dz),
                            @ifactive(dy, (digamma(y) - dxy)*dz))
    CoDual(logbeta(x, y), NoFData()), logbeta_pullback
end

# Upper incomplete gamma `gamma(a, x)` and its log — differentiable in `x` only.
function rrule!!(::CoDual{typeof(gamma),NoFData}, ::AbstractCtx,
                 acd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    a = primal(acd)
    ard = _no_param_rdata(acd, "gamma", "parameter `a`")
    gamma_upper_pullback(dy) = (NoRData(), ard, @ifactive(dx, -exp(-x)*x^(a-1)*dy))
    CoDual(gamma(a, x), NoFData()), gamma_upper_pullback
end

function rrule!!(::CoDual{typeof(loggamma),NoFData}, ::AbstractCtx,
                 acd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    a = primal(acd)
    ard = _no_param_rdata(acd, "loggamma", "parameter `a`")
    y = loggamma(a, x)
    loggamma_upper_pullback(dy) = (NoRData(), ard, @ifactive(dx, -exp(-(x + y))*x^(a-1)*dy))
    CoDual(y, NoFData()), loggamma_upper_pullback
end

# `logabsgamma` returns `(log|Γ(x)|, sign(Γ(x)))`; the sign is an `Int`, so the rdata seed's
# second half carries nothing.
function rrule!!(::CoDual{typeof(logabsgamma),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    logabsgamma_pullback(dy) = (NoRData(), @ifactive(dx, digamma(x)*dy[1]))
    CoDual(logabsgamma(x), NoFData()), logabsgamma_pullback
end

# `gamma_inc(a, x, IND)` returns the regularised lower and upper incomplete gamma pair, which sum
# to one — hence the opposite signs. `IND` selects the accuracy and has no tangent space.
function rrule!!(::CoDual{typeof(gamma_inc),NoFData}, ::AbstractCtx,
                 acd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
                 indcd::IntParamCoDual)
    a = primal(acd)
    ard = _no_param_rdata(acd, "gamma_inc", "parameter `a`")
    z = exp(-x)*x^(a-1)/gamma(a)
    gamma_inc_pullback(dy) = (NoRData(), ard, @ifactive(dx, z*(dy[1] - dy[2])), NoRData())
    CoDual(gamma_inc(a, x, primal(indcd)), NoFData()), gamma_inc_pullback
end

# ===========================================================================
# error functions
# ===========================================================================

function rrule!!(::CoDual{typeof(erf),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    erf_pullback(dy) = (NoRData(), @ifactive(dx, 2*exp(-x^2)/sqrtπ*dy))
    CoDual(erf(x), NoFData()), erf_pullback
end

function rrule!!(::CoDual{typeof(erf),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
                 (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}})
    erf2_pullback(dz) = (NoRData(), @ifactive(dx, -2*exp(-x^2)/sqrtπ*dz),
                         @ifactive(dy, 2*exp(-y^2)/sqrtπ*dz))
    CoDual(erf(x, y), NoFData()), erf2_pullback
end

function rrule!!(::CoDual{typeof(erfc),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    erfc_pullback(dy) = (NoRData(), @ifactive(dx, -2*exp(-x^2)/sqrtπ*dy))
    CoDual(erfc(x), NoFData()), erfc_pullback
end

function rrule!!(::CoDual{typeof(logerfc),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = logerfc(x)
    logerfc_pullback(dy) = (NoRData(), @ifactive(dx, -2*exp(-x^2 - y)/sqrtπ*dy))
    CoDual(y, NoFData()), logerfc_pullback
end

function rrule!!(::CoDual{typeof(erfcx),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = erfcx(x)
    erfcx_pullback(dy) = (NoRData(), @ifactive(dx, 2*(x*y - inv(oftype(y, sqrtπ)))*dy))
    CoDual(y, NoFData()), erfcx_pullback
end

function rrule!!(::CoDual{typeof(logerfcx),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = logerfcx(x)
    logerfcx_pullback(dy) = (NoRData(), @ifactive(dx, 2*(x - exp(-y)/sqrtπ)*dy))
    CoDual(y, NoFData()), logerfcx_pullback
end

function rrule!!(::CoDual{typeof(erfi),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    erfi_pullback(dy) = (NoRData(), @ifactive(dx, 2*exp(x^2)/sqrtπ*dy))
    CoDual(erfi(x), NoFData()), erfi_pullback
end

function rrule!!(::CoDual{typeof(erfinv),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = erfinv(x)
    erfinv_pullback(dy) = (NoRData(), @ifactive(dx, sqrtπ*exp(y^2)/2*dy))
    CoDual(y, NoFData()), erfinv_pullback
end

function rrule!!(::CoDual{typeof(erfcinv),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = erfcinv(x)
    erfcinv_pullback(dy) = (NoRData(), @ifactive(dx, -sqrtπ*exp(y^2)/2*dy))
    CoDual(y, NoFData()), erfcinv_pullback
end

# ===========================================================================
# exponential, sine and cosine integrals
# ===========================================================================

function rrule!!(::CoDual{typeof(expint),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    expint_pullback(dy) = (NoRData(), @ifactive(dx, -exp(-x)/x*dy))
    CoDual(expint(x), NoFData()), expint_pullback
end

function rrule!!(::CoDual{typeof(expint),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "expint", "order")
    expint2_pullback(dy) = (NoRData(), νrd, @ifactive(dx, -expint(ν-1, x)*dy))
    CoDual(expint(ν, x), NoFData()), expint2_pullback
end

function rrule!!(::CoDual{typeof(expintx),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = expintx(x)
    expintx_pullback(dy) = (NoRData(), @ifactive(dx, (y - inv(x))*dy))
    CoDual(y, NoFData()), expintx_pullback
end

function rrule!!(::CoDual{typeof(expintx),NoFData}, ::AbstractCtx,
                 νcd::ParamCoDual, (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    ν = primal(νcd)
    νrd = _no_param_rdata(νcd, "expintx", "order")
    y = expintx(ν, x)
    expintx2_pullback(dy) = (NoRData(), νrd, @ifactive(dx, (y - expintx(ν-1, x))*dy))
    CoDual(y, NoFData()), expintx2_pullback
end

function rrule!!(::CoDual{typeof(expinti),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    expinti_pullback(dy) = (NoRData(), @ifactive(dx, exp(x)/x*dy))
    CoDual(expinti(x), NoFData()), expinti_pullback
end

function rrule!!(::CoDual{typeof(sinint),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    sinint_pullback(dy) = (NoRData(), @ifactive(dx, sinc(invπ*x)*dy))
    CoDual(sinint(x), NoFData()), sinint_pullback
end

function rrule!!(::CoDual{typeof(cosint),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    cosint_pullback(dy) = (NoRData(), @ifactive(dx, cos(x)/x*dy))
    CoDual(cosint(x), NoFData()), cosint_pullback
end

# ===========================================================================
# complete elliptic integrals
# ===========================================================================

function rrule!!(::CoDual{typeof(ellipk),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = ellipk(x)
    ellipk_pullback(dy) = (NoRData(),
                           @ifactive(dx, (iszero(x) ? oftype(y, π)/8 : (ellipe(x)/(1-x) - y)/(2*x))*dy))
    CoDual(y, NoFData()), ellipk_pullback
end

function rrule!!(::CoDual{typeof(ellipe),NoFData}, ::AbstractCtx,
                 (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}})
    y = ellipe(x)
    ellipe_pullback(dy) = (NoRData(),
                           @ifactive(dx, (iszero(x) ? -oftype(y, π)/8 : (y - ellipk(x))/(2*x))*dy))
    CoDual(y, NoFData()), ellipe_pullback
end

end # module DifferReverseSpecialFunctionsExt
