# Hand-written frule!! for scalar math functions (Base.Math transcendentals).
# Reverse-mode rrule!!s for the same functions live in DifferReverse/src/rules_math.jl.
#
# Every rule uses the closed-form derivative directly, never by differentiating through Base's
# actual implementation. Some of those internals (e.g. asin's bitcast/and_int precision trick) use
# intrinsics on the `@inactive_intrinsic` list (`src/intrinsics.jl`), which would silently produce a
# zero tangent rather than error.
#
# `frule!!` bodies are ordinary compiled methods, not synthetic IR, so they keep bare names
# (matching `frules.jl`).

# ===========================================================================
# exp
# ===========================================================================

function frule!!(::Dual{typeof(exp)}, (; x, dx)::Dual)
    y = exp(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, y*dx)
end

# ===========================================================================
# log
# ===========================================================================

function frule!!(::Dual{typeof(log)}, (; x, dx)::Dual)
    y = log(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/x)
end

# ===========================================================================
# log1p
# ===========================================================================

function frule!!(::Dual{typeof(log1p)}, (; x, dx)::Dual)
    y = log1p(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/(1+x))
end

# ===========================================================================
# expm1
# ===========================================================================

function frule!!(::Dual{typeof(expm1)}, (; x, dx)::Dual)
    y = expm1(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, exp(x)*dx)
end

# ===========================================================================
# log2
# ===========================================================================

function frule!!(::Dual{typeof(log2)}, (; x, dx)::Dual)
    y = log2(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/(x*log(2)))
end

# ===========================================================================
# log10
# ===========================================================================

function frule!!(::Dual{typeof(log10)}, (; x, dx)::Dual)
    y = log10(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/(x*log(10)))
end

# ===========================================================================
# exp2
# ===========================================================================

function frule!!(::Dual{typeof(exp2)}, (; x, dx)::Dual)
    y = exp2(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, y*log(2)*dx)
end

# ===========================================================================
# exp10
# ===========================================================================

function frule!!(::Dual{typeof(exp10)}, (; x, dx)::Dual)
    y = exp10(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, y*log(10)*dx)
end

# ===========================================================================
# sinh / cosh / tanh
# ===========================================================================

function frule!!(::Dual{typeof(sinh)}, (; x, dx)::Dual)
    y = sinh(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, cosh(x)*dx)
end

function frule!!(::Dual{typeof(cosh)}, (; x, dx)::Dual)
    y = cosh(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, sinh(x)*dx)
end

function frule!!(::Dual{typeof(tanh)}, (; x, dx)::Dual)
    y = tanh(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, (1-y^2)*dx)
end

# ===========================================================================
# asinh / acosh / atanh
# ===========================================================================

function frule!!(::Dual{typeof(asinh)}, (; x, dx)::Dual)
    y = asinh(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/sqrt(x^2+1))
end

function frule!!(::Dual{typeof(acosh)}, (; x, dx)::Dual)
    y = acosh(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/sqrt(x^2-1))
end

function frule!!(::Dual{typeof(atanh)}, (; x, dx)::Dual)
    y = atanh(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/(1-x^2))
end

# ===========================================================================
# asin / acos
# ===========================================================================

function frule!!(::Dual{typeof(asin)}, (; x, dx)::Dual)
    y = asin(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/sqrt(1-x^2))
end

function frule!!(::Dual{typeof(acos)}, (; x, dx)::Dual)
    y = acos(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -dx/sqrt(1-x^2))
end

# ===========================================================================
# atan (1-arg and 2-arg)
# ===========================================================================

function frule!!(::Dual{typeof(atan)}, (; x, dx)::Dual)
    y = atan(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/(1+x^2))
end

function frule!!(::Dual{typeof(atan)}, (; y, dy)::Dual, (; x, dx)::Dual)
    r2 = x^2+y^2
    z = atan(y, x)
    dzy = _inert(dy) ? zero(z) : x*dy/r2
    dzx = _inert(dx) ? zero(z) : -y*dx/r2
    Dual(z, dzy + dzx)
end

# ===========================================================================
# cbrt
# ===========================================================================

function frule!!(::Dual{typeof(cbrt)}, (; x, dx)::Dual)
    y = cbrt(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, dx/(3*y^2))
end

# ===========================================================================
# ^ (x::Float64, y::Float64 or y::Int) — non-literal exponent, possibly non-differentiable
# ===========================================================================

function frule!!(::Dual{typeof(^)}, (; x, dx)::Dual, (; y, dy)::Dual)
    z = x^y
    dzx = _inert(dx) ? zero(z) : y*x^(y-1)*dx
    dzy = _inert(dy) ? zero(z) : z*log(x)*dy
    Dual(z, dzx + dzy)
end

# ===========================================================================
# hypot(x, y)
# ===========================================================================

function frule!!(::Dual{typeof(hypot)}, (; x, dx)::Dual, (; y, dy)::Dual)
    r = hypot(x, y)
    drx = _inert(dx) ? zero(r) : x*dx/r
    dry = _inert(dy) ? zero(r) : y*dy/r
    Dual(r, drx + dry)
end

# ===========================================================================
# sqrt(::Complex)
# ===========================================================================

function frule!!(::Dual{typeof(sqrt)}, dz::Dual{ComplexF64})
    z = primal(dz)
    dzt = tangent(dz)
    y = sqrt(z)
    isactive(dzt) || return Dual(y, zero_tangent(y))
    dz_re = get_tangent_field(dzt, :re)
    dz_im = get_tangent_field(dzt, :im)
    dy = Complex(dz_re, dz_im)/(2*y)
    Dual(y, build_tangent(ComplexF64, real(dy), imag(dy)))
end

# ===========================================================================
# Note: `abs`/`max`/`min`/`sign`/`copysign`/`fma`/`muladd`/`abs2`/`inv` have no `frule!!` here —
# forward mode already handles them via dispatch on the underlying LLVM intrinsic
# (`src/intrinsics.jl`), never reaching `frule!!`. Reverse-mode rules for these live in
# DifferReverse/src/rules_math.jl (they inline straight to the intrinsic before a call-level
# `rrule!!` gets a chance to fire, so reverse mode needs an explicit rule where forward doesn't).
# ===========================================================================
