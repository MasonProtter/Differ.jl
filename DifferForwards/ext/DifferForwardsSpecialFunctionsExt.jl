# Hand-written `frule!!` for SpecialFunctions.jl's scalar functions. The reverse-mode `rrule!!`s
# for the same functions live in `DifferReverse/ext/DifferReverseSpecialFunctionsExt.jl`.
#
# The derivatives are the closed forms SpecialFunctions' own ChainRulesCore extension uses. Rules
# are not optional here the way they are for a plain Julia function: nearly every one of these
# bottoms out in a `ccall` to openspecfun, which the dualizer cannot see through.
#
# Some arguments have no implemented derivative — a Bessel order, an incomplete gamma or
# exponential-integral parameter — because no closed form for one is implemented upstream either.
# Those slots are not quietly skipped: a constant in that position reaches the rule as `Inactive()`,
# so `besselj(1.5, x)` still works, and anything else is refused rather than dropped
# (`_no_param_derivative`). Reverse mode refuses at the same point — see its file.
#
# Not covered: `hankelh1`/`hankelh2` and their scaled forms, which are complex-valued.
# `besselix`/`besseljx`/`besselyx` are covered for real arguments only — their complex derivative
# is not holomorphic and needs a different rule.
module DifferForwardsSpecialFunctionsExt

using SpecialFunctions
using SpecialFunctions: sqrtπ, invπ

import DifferForwards: Dual, primal, tangent, frule!!, isactive, _inert, zero_tangent,
    NoTangent

# A parameter slot with no implemented derivative: a Bessel order, or an incomplete gamma /
# exponential-integral parameter. `Inactive` is the normal path — a literal or a caller-declared
# constant both arrive that way. An explicitly seeded zero direction is also fine: the missing term
# would be multiplied by zero. Anything else would drop a real contribution, so refuse it.
_no_param_derivative(dp, f, what) =
    _inert(dp) || iszero(dp) ||
    error("Differ: the derivative of `", f, "` with respect to its ", what,
          " is not implemented — hold that argument constant")

# ===========================================================================
# Airy functions
# ===========================================================================

function frule!!(::Dual{typeof(airyai)}, (; x, dx)::Dual)
    y = airyai(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, airyaiprime(x)*dx)
end

function frule!!(::Dual{typeof(airyaix)}, (; x, dx)::Dual)
    y = airyaix(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (airyaiprimex(x) + sqrt(x)*y)*dx)
end

function frule!!(::Dual{typeof(airyaiprime)}, (; x, dx)::Dual)
    y = airyaiprime(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, x*airyai(x)*dx)
end

function frule!!(::Dual{typeof(airyaiprimex)}, (; x, dx)::Dual)
    y = airyaiprimex(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (x*airyaix(x) + sqrt(x)*y)*dx)
end

function frule!!(::Dual{typeof(airybi)}, (; x, dx)::Dual)
    y = airybi(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, airybiprime(x)*dx)
end

function frule!!(::Dual{typeof(airybiprime)}, (; x, dx)::Dual)
    y = airybiprime(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, x*airybi(x)*dx)
end

# ===========================================================================
# Bessel functions of fixed order
# ===========================================================================

function frule!!(::Dual{typeof(besselj0)}, (; x, dx)::Dual)
    y = besselj0(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -besselj1(x)*dx)
end

function frule!!(::Dual{typeof(besselj1)}, (; x, dx)::Dual)
    y = besselj1(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (besselj0(x) - besselj(2, x))/2*dx)
end

function frule!!(::Dual{typeof(bessely0)}, (; x, dx)::Dual)
    y = bessely0(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -bessely1(x)*dx)
end

function frule!!(::Dual{typeof(bessely1)}, (; x, dx)::Dual)
    y = bessely1(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (bessely0(x) - bessely(2, x))/2*dx)
end

# ===========================================================================
# Bessel functions of general order — differentiable in the argument only
# ===========================================================================

function frule!!(::Dual{typeof(besselj)}, dν::Dual, (; x, dx)::Dual)
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "besselj", "order")
    y = besselj(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (besselj(ν-1, x) - besselj(ν+1, x))/2*dx)
end

function frule!!(::Dual{typeof(besseli)}, dν::Dual, (; x, dx)::Dual)
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "besseli", "order")
    y = besseli(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (besseli(ν-1, x) + besseli(ν+1, x))/2*dx)
end

function frule!!(::Dual{typeof(bessely)}, dν::Dual, (; x, dx)::Dual)
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "bessely", "order")
    y = bessely(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (bessely(ν-1, x) - bessely(ν+1, x))/2*dx)
end

function frule!!(::Dual{typeof(besselk)}, dν::Dual, (; x, dx)::Dual)
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "besselk", "order")
    y = besselk(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -(besselk(ν-1, x) + besselk(ν+1, x))/2*dx)
end

function frule!!(::Dual{typeof(besselkx)}, dν::Dual, (; x, dx)::Dual)
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "besselkx", "order")
    y = besselkx(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (-(besselkx(ν-1, x) + besselkx(ν+1, x))/2 + y)*dx)
end

# The scaling factor — `exp(-|Re x|)` for `besselix`, `exp(-|Im x|)` for the other two — is not
# holomorphic, so these three cover real arguments only. On the real axis its own derivative is the
# `-sign(x)*y` term in `besselix`, and vanishes for the other two.

function frule!!(::Dual{typeof(besselix)}, dν::Dual, (; x, dx)::Dual{<:Real})
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "besselix", "order")
    y = besselix(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, ((besselix(ν-1, x) + besselix(ν+1, x))/2 - sign(x)*y)*dx)
end

function frule!!(::Dual{typeof(besseljx)}, dν::Dual, (; x, dx)::Dual{<:Real})
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "besseljx", "order")
    y = besseljx(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (besseljx(ν-1, x) - besseljx(ν+1, x))/2*dx)
end

function frule!!(::Dual{typeof(besselyx)}, dν::Dual, (; x, dx)::Dual{<:Real})
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "besselyx", "order")
    y = besselyx(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (besselyx(ν-1, x) - besselyx(ν+1, x))/2*dx)
end

# ===========================================================================
# dawson
# ===========================================================================

function frule!!(::Dual{typeof(dawson)}, (; x, dx)::Dual)
    y = dawson(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (1 - 2*x*y)*dx)
end

# ===========================================================================
# gamma and friends
# ===========================================================================

function frule!!(::Dual{typeof(gamma)}, (; x, dx)::Dual)
    y = gamma(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, y*digamma(x)*dx)
end

function frule!!(::Dual{typeof(loggamma)}, (; x, dx)::Dual)
    y = loggamma(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, digamma(x)*dx)
end

function frule!!(::Dual{typeof(digamma)}, (; x, dx)::Dual)
    y = digamma(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, trigamma(x)*dx)
end

function frule!!(::Dual{typeof(trigamma)}, (; x, dx)::Dual)
    y = trigamma(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, polygamma(2, x)*dx)
end

function frule!!(::Dual{typeof(invdigamma)}, (; x, dx)::Dual)
    y = invdigamma(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, inv(trigamma(y))*dx)
end

# The order `m` is an `Integer`, so its shadow can only ever be `NoTangent`/`Inactive`.
function frule!!(::Dual{typeof(polygamma)}, dm::Dual{<:Integer}, (; x, dx)::Dual)
    m = primal(dm)
    y = polygamma(m, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, polygamma(m+1, x)*dx)
end

function frule!!(::Dual{typeof(beta)}, (; x, dx)::Dual, (; y, dy)::Dual)
    b = beta(x, y)
    dxy = digamma(x + y)
    dbx = _inert(dx) ? zero(b) : b*(digamma(x) - dxy)*dx
    dby = _inert(dy) ? zero(b) : b*(digamma(y) - dxy)*dy
    Dual(b, dbx + dby)
end

function frule!!(::Dual{typeof(logbeta)}, (; x, dx)::Dual, (; y, dy)::Dual)
    b = logbeta(x, y)
    dxy = digamma(x + y)
    dbx = _inert(dx) ? zero(b) : (digamma(x) - dxy)*dx
    dby = _inert(dy) ? zero(b) : (digamma(y) - dxy)*dy
    Dual(b, dbx + dby)
end

# Upper incomplete gamma `gamma(a, x)` and its log — differentiable in `x` only.
function frule!!(::Dual{typeof(gamma)}, da::Dual, (; x, dx)::Dual)
    a = primal(da)
    _no_param_derivative(tangent(da), "gamma", "parameter `a`")
    y = gamma(a, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -exp(-x)*x^(a-1)*dx)
end

function frule!!(::Dual{typeof(loggamma)}, da::Dual, (; x, dx)::Dual)
    a = primal(da)
    _no_param_derivative(tangent(da), "loggamma", "parameter `a`")
    y = loggamma(a, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -exp(-(x + y))*x^(a-1)*dx)
end

# `logabsgamma` returns `(log|Γ(x)|, sign(Γ(x)))`; the sign is an `Int` and has no tangent space.
function frule!!(::Dual{typeof(logabsgamma)}, (; x, dx)::Dual)
    y = logabsgamma(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (digamma(x)*dx, NoTangent()))
end

# `gamma_inc(a, x, IND)` returns the regularised lower and upper incomplete gamma pair, which sum
# to one — hence the opposite signs. `IND` selects the accuracy and has no tangent space.
function frule!!(::Dual{typeof(gamma_inc)}, da::Dual, (; x, dx)::Dual, dind::Dual{<:Integer})
    a = primal(da)
    _no_param_derivative(tangent(da), "gamma_inc", "parameter `a`")
    y = gamma_inc(a, x, primal(dind))
    isactive(dx) || return Dual(y, zero_tangent(y))
    z = exp(-x)*x^(a-1)/gamma(a)
    Dual(y, (z*dx, -z*dx))
end

# ===========================================================================
# error functions
# ===========================================================================

function frule!!(::Dual{typeof(erf)}, (; x, dx)::Dual)
    y = erf(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, 2*exp(-x^2)/sqrtπ*dx)
end

function frule!!(::Dual{typeof(erf)}, (; x, dx)::Dual, (; y, dy)::Dual)
    z = erf(x, y)
    dzx = _inert(dx) ? zero(z) : -2*exp(-x^2)/sqrtπ*dx
    dzy = _inert(dy) ? zero(z) : 2*exp(-y^2)/sqrtπ*dy
    Dual(z, dzx + dzy)
end

function frule!!(::Dual{typeof(erfc)}, (; x, dx)::Dual)
    y = erfc(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -2*exp(-x^2)/sqrtπ*dx)
end

function frule!!(::Dual{typeof(logerfc)}, (; x, dx)::Dual)
    y = logerfc(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -2*exp(-x^2 - y)/sqrtπ*dx)
end

function frule!!(::Dual{typeof(erfcx)}, (; x, dx)::Dual)
    y = erfcx(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, 2*(x*y - inv(oftype(y, sqrtπ)))*dx)
end

function frule!!(::Dual{typeof(logerfcx)}, (; x, dx)::Dual)
    y = logerfcx(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, 2*(x - exp(-y)/sqrtπ)*dx)
end

function frule!!(::Dual{typeof(erfi)}, (; x, dx)::Dual)
    y = erfi(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, 2*exp(x^2)/sqrtπ*dx)
end

function frule!!(::Dual{typeof(erfinv)}, (; x, dx)::Dual)
    y = erfinv(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, sqrtπ*exp(y^2)/2*dx)
end

function frule!!(::Dual{typeof(erfcinv)}, (; x, dx)::Dual)
    y = erfcinv(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -sqrtπ*exp(y^2)/2*dx)
end

# ===========================================================================
# exponential, sine and cosine integrals
# ===========================================================================

function frule!!(::Dual{typeof(expint)}, (; x, dx)::Dual)
    y = expint(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -exp(-x)/x*dx)
end

function frule!!(::Dual{typeof(expint)}, dν::Dual, (; x, dx)::Dual)
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "expint", "order")
    y = expint(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -expint(ν-1, x)*dx)
end

function frule!!(::Dual{typeof(expintx)}, (; x, dx)::Dual)
    y = expintx(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (y - inv(x))*dx)
end

function frule!!(::Dual{typeof(expintx)}, dν::Dual, (; x, dx)::Dual)
    ν = primal(dν)
    _no_param_derivative(tangent(dν), "expintx", "order")
    y = expintx(ν, x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (y - expintx(ν-1, x))*dx)
end

function frule!!(::Dual{typeof(expinti)}, (; x, dx)::Dual)
    y = expinti(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, exp(x)/x*dx)
end

function frule!!(::Dual{typeof(sinint)}, (; x, dx)::Dual)
    y = sinint(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, sinc(invπ*x)*dx)
end

function frule!!(::Dual{typeof(cosint)}, (; x, dx)::Dual)
    y = cosint(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, cos(x)/x*dx)
end

# ===========================================================================
# complete elliptic integrals
# ===========================================================================

function frule!!(::Dual{typeof(ellipk)}, (; x, dx)::Dual)
    y = ellipk(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (iszero(x) ? oftype(y, π)/8 : (ellipe(x)/(1-x) - y)/(2*x))*dx)
end

function frule!!(::Dual{typeof(ellipe)}, (; x, dx)::Dual)
    y = ellipe(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (iszero(x) ? -oftype(y, π)/8 : (y - ellipk(x))/(2*x))*dx)
end

end # module DifferForwardsSpecialFunctionsExt
