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
# GlobalRefs, matching `sum_pullback`/`sum_map_pullback` in `rules_perf_backstop.jl`.

# ===========================================================================
# exp
# ===========================================================================

function rrule!!(::CoDual{typeof(exp),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    y = Base.exp(x)
    exp_pullback(dy) = (NoRData(), y*dy)
    CoDual(y, NoFData()), exp_pullback
end

# ===========================================================================
# log
# ===========================================================================

function rrule!!(::CoDual{typeof(log),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    log_pullback(dy) = (NoRData(), dy/x)
    CoDual(Base.log(x), NoFData()), log_pullback
end

# ===========================================================================
# log1p
# ===========================================================================

function rrule!!(::CoDual{typeof(log1p),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    log1p_pullback(dy) = (NoRData(), dy/(1+x))
    CoDual(Base.log1p(x), NoFData()), log1p_pullback
end

# ===========================================================================
# expm1
# ===========================================================================

function rrule!!(::CoDual{typeof(expm1),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    expm1_pullback(dy) = (NoRData(), Base.exp(x)*dy)
    CoDual(Base.expm1(x), NoFData()), expm1_pullback
end

# ===========================================================================
# log2
# ===========================================================================

function rrule!!(::CoDual{typeof(log2),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    log2_pullback(dy) = (NoRData(), dy/(x*Base.log(2)))
    CoDual(Base.log2(x), NoFData()), log2_pullback
end

# ===========================================================================
# log10
# ===========================================================================

function rrule!!(::CoDual{typeof(log10),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    log10_pullback(dy) = (NoRData(), dy/(x*Base.log(10)))
    CoDual(Base.log10(x), NoFData()), log10_pullback
end

# ===========================================================================
# exp2
# ===========================================================================

function rrule!!(::CoDual{typeof(exp2),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    y = Base.exp2(x)
    exp2_pullback(dy) = (NoRData(), y*Base.log(2)*dy)
    CoDual(y, NoFData()), exp2_pullback
end

# ===========================================================================
# exp10
# ===========================================================================

function rrule!!(::CoDual{typeof(exp10),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    y = Base.exp10(x)
    exp10_pullback(dy) = (NoRData(), y*Base.log(10)*dy)
    CoDual(y, NoFData()), exp10_pullback
end

# ===========================================================================
# sinh / cosh / tanh
# ===========================================================================

function rrule!!(::CoDual{typeof(sinh),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    sinh_pullback(dy) = (NoRData(), Base.cosh(x)*dy)
    CoDual(Base.sinh(x), NoFData()), sinh_pullback
end

function rrule!!(::CoDual{typeof(cosh),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    cosh_pullback(dy) = (NoRData(), Base.sinh(x)*dy)
    CoDual(Base.cosh(x), NoFData()), cosh_pullback
end

function rrule!!(::CoDual{typeof(tanh),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    y = Base.tanh(x)
    tanh_pullback(dy) = (NoRData(), (1-y^2)*dy)
    CoDual(y, NoFData()), tanh_pullback
end

# ===========================================================================
# asinh / acosh / atanh
# ===========================================================================

function rrule!!(::CoDual{typeof(asinh),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    asinh_pullback(dy) = (NoRData(), dy/Base.sqrt(x^2+1))
    CoDual(Base.asinh(x), NoFData()), asinh_pullback
end

function rrule!!(::CoDual{typeof(acosh),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    acosh_pullback(dy) = (NoRData(), dy/Base.sqrt(x^2-1))
    CoDual(Base.acosh(x), NoFData()), acosh_pullback
end

function rrule!!(::CoDual{typeof(atanh),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    atanh_pullback(dy) = (NoRData(), dy/(1-x^2))
    CoDual(Base.atanh(x), NoFData()), atanh_pullback
end

# ===========================================================================
# asin / acos
# ===========================================================================

function rrule!!(::CoDual{typeof(asin),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    asin_pullback(dy) = (NoRData(), dy/Base.sqrt(1-x^2))
    CoDual(Base.asin(x), NoFData()), asin_pullback
end

function rrule!!(::CoDual{typeof(acos),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    acos_pullback(dy) = (NoRData(), -dy/Base.sqrt(1-x^2))
    CoDual(Base.acos(x), NoFData()), acos_pullback
end

# ===========================================================================
# atan (1-arg and 2-arg)
# ===========================================================================

function rrule!!(::CoDual{typeof(atan),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    atan_pullback(dy) = (NoRData(), dy/(1+x^2))
    CoDual(Base.atan(x), NoFData()), atan_pullback
end

function rrule!!(
    ::CoDual{typeof(atan),NoFData}, ::AbstractCtx,
    (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
)
    r2 = x^2+y^2
    atan2_pullback(dz) = (NoRData(), @ifactive(dy, x*dz/r2), @ifactive(dx, -y*dz/r2))
    CoDual(Base.atan(y, x), NoFData()), atan2_pullback
end

# ===========================================================================
# cbrt
# ===========================================================================

function rrule!!(::CoDual{typeof(cbrt),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    y = Base.cbrt(x)
    cbrt_pullback(dy) = (NoRData(), dy/(3*y^2))
    CoDual(y, NoFData()), cbrt_pullback
end

# ===========================================================================
# ^ (x::Float64, y::Float64) — non-literal real exponent
# ===========================================================================

function rrule!!(
    ::CoDual{typeof(^),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}},
)
    yp = Base.:^(x, y)
    pow_pullback(dz) = (NoRData(), @ifactive(dx, y*Base.:^(x, y-1)*dz), @ifactive(dy, yp*Base.log(x)*dz))
    CoDual(yp, NoFData()), pow_pullback
end

# ===========================================================================
# ^ (x::Union{Float32,Float64}, n::Integer) — keeps `^` un-inlined (hand rules block inlining, see
# `src_inlining_policy`) so the branchless bit-twiddling of `Base.Math.pow_body` never surfaces.
# ===========================================================================

function rrule!!(
    ::CoDual{typeof(^),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{P,<:Union{NoFData,Inactive}}, (; y)::CoDual{N,NoFData},
) where {P<:Union{Float32,Float64},N<:Integer}
    n = Int(y)
    function intpow_pullback(dy)
        # n == 0 short-circuits: `n*x^(n-1)` would be `0 * Inf == NaN` at x == 0.
        dxval = iszero(n) ? zero(P) : P(n) * (x^(n - 1))
        return (NoRData(), @ifactive(dx, dxval * dy), NoRData())
    end
    CoDual(x^n, NoFData()), intpow_pullback
end

# ===========================================================================
# hypot(x, y)
# ===========================================================================

function rrule!!(
    ::CoDual{typeof(hypot),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}},
)
    r = Base.hypot(x, y)
    hypot_pullback(dz) = (NoRData(), @ifactive(dx, x/r*dz), @ifactive(dy, y/r*dz))
    CoDual(r, NoFData()), hypot_pullback
end

# ===========================================================================
# sqrt(::Complex)
# ===========================================================================

# Reverse mode for a holomorphic scalar map f: the output rdata seed s = sr + i*si (read straight
# off the (re, im) fields, no conjugation) pulls back via conj(f'(z)) * s — the standard adjoint of
# the real 2x2 Jacobian of a holomorphic map.
function rrule!!(
    ::CoDual{typeof(sqrt),NoFData}, ::AbstractCtx, (; z)::CoDual{ComplexF64,NoFData}
)
    y = Base.sqrt(z)
    function sqrt_pullback(dy)
        s = Base.Complex(dy.data.re, dy.data.im)
        fprime = 1/(2*y)
        dz = Base.conj(fprime)*s
        return (NoRData(), RData((re=Base.real(dz), im=Base.imag(dz))))
    end
    CoDual(y, NoFData()), sqrt_pullback
end

# ===========================================================================
# Reverse-only rules — these inline straight to an LLVM intrinsic before a call-level rrule!! gets
# a chance to fire. Forward mode already handles them via `src/intrinsics.jl`.
# ===========================================================================

function rrule!!(::CoDual{typeof(abs),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    abs_pullback(dy) = (NoRData(), Base.sign(x)*dy)
    CoDual(Base.abs(x), NoFData()), abs_pullback
end

function rrule!!(
    ::CoDual{typeof(max),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}},
)
    x_wins = x >= y
    max_pullback(dz) = (NoRData(), @ifactive(dx, x_wins ? dz : 0.0), @ifactive(dy, x_wins ? 0.0 : dz))
    CoDual(Base.max(x, y), NoFData()), max_pullback
end

function rrule!!(
    ::CoDual{typeof(min),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}},
)
    x_wins = x <= y
    min_pullback(dz) = (NoRData(), @ifactive(dx, x_wins ? dz : 0.0), @ifactive(dy, x_wins ? 0.0 : dz))
    CoDual(Base.min(x, y), NoFData()), min_pullback
end

function rrule!!(::CoDual{typeof(sign),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    sign_pullback(_) = (NoRData(), 0.0)
    CoDual(Base.sign(x), NoFData()), sign_pullback
end

function rrule!!(
    ::CoDual{typeof(copysign),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}},
)
    copysign_pullback(dz) = (NoRData(), @ifactive(dx, Base.sign(x)*Base.sign(y)*dz), @ifactive(dy, 0.0))
    CoDual(Base.copysign(x, y), NoFData()), copysign_pullback
end

function rrule!!(
    ::CoDual{typeof(fma),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; z, dz)::CoDual{Float64,<:Union{NoFData,Inactive}},
)
    fma_pullback(dw) = (NoRData(), @ifactive(dx, y*dw), @ifactive(dy, x*dw), @ifactive(dz, dw))
    CoDual(Base.fma(x, y, z), NoFData()), fma_pullback
end

function rrule!!(
    ::CoDual{typeof(muladd),NoFData}, ::AbstractCtx,
    (; x, dx)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; y, dy)::CoDual{Float64,<:Union{NoFData,Inactive}},
    (; z, dz)::CoDual{Float64,<:Union{NoFData,Inactive}},
)
    muladd_pullback(dw) = (NoRData(), @ifactive(dx, y*dw), @ifactive(dy, x*dw), @ifactive(dz, dw))
    CoDual(Base.muladd(x, y, z), NoFData()), muladd_pullback
end

function rrule!!(::CoDual{typeof(abs2),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    abs2_pullback(dy) = (NoRData(), 2*x*dy)
    CoDual(Base.abs2(x), NoFData()), abs2_pullback
end

function rrule!!(::CoDual{typeof(inv),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    y = Base.inv(x)
    inv_pullback(dy) = (NoRData(), -y^2*dy)
    CoDual(y, NoFData()), inv_pullback
end
