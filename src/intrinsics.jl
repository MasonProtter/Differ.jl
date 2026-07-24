# ===========================================================================
# Intrinsic wrappers — dispatch-based handling of `Core.Intrinsics`.
#
# Every `Core.Intrinsics` function (`add_float`, `mul_float`, …) is an instance of the *single*
# type `Core.IntrinsicFunction`, so — unlike an ordinary function — we cannot write an `frule`
# method keyed on a particular intrinsic (`Dual{typeof(sin)}` works; `Dual{typeof(add_float)}` would
# collide with every other intrinsic). Instead we give each intrinsic a thin wrapper *function*,
# which does have its own singleton type, rewrite intrinsic calls in the primal IR to their wrappers
# during dualization (`dualize_to_ircode`), and hang `frule` methods off the wrappers — so tangent
# rules are ordinary `frule` dispatch instead of hand-coded IR emission.
#
# Handling is **explicit, not implicit**: `translate` maps an intrinsic value to its wrapper and has
# *no* identity fallback. An intrinsic with no registered wrapper hits the error method below, so an
# unsupported intrinsic surfaces loudly rather than silently miscompiling (e.g. a missing derivative
# returning a wrong zero tangent). Register a *differentiable* intrinsic with `@intrinsic` + a
# hand-written `frule`; register a *non-differentiable* one (comparisons, integer/bit ops, …) with
# `@inactive_intrinsic`, which auto-generates an `frule` computing the primal with a zero tangent.
#
# frule bodies below call the *wrappers* (not the raw intrinsics), so they re-dualize cleanly to any
# order: a nested-differentiation pass sees each inner op as a surviving wrapper call and routes it
# back through these same rules.
# ===========================================================================

# No identity fallback: an unregistered intrinsic is an error, not a silent passthrough.
translate(::Val{F}) where {F} = error(
    "ADNext: unsupported intrinsic `", nameof(F), "`. No wrapper/`frule` is registered for it. ",
    "Add one in `src/intrinsics.jl`: `@intrinsic` + an `frule` for a differentiable intrinsic, or ",
    "`@inactive_intrinsic` for a non-differentiable one.")

# Define a wrapper function for a `Core.Intrinsics` function and register it with `translate`.
# Usage: `@intrinsic add_float`, followed by an `frule(::Dual{typeof(add_float)}, …)`.
macro intrinsic(name)
    intr = :(Core.Intrinsics.$name)          # e.g. `Core.Intrinsics.add_float`
    esc(quote
        $name(x...) = $intr(x...)
        translate(::Val{$intr}) = $name
    end)
end

# Like `@intrinsic`, but also generates the `frule` for a *non-differentiable* intrinsic: it
# computes the primal from the argument primals and wraps it in a `Dual` with a zero tangent
# (`NoTangent()` for `Int`/`Bool`, `zero` for a `Number`, …). Variadic to cover unary/binary/n-ary
# intrinsics uniformly.
macro inactive_intrinsic(name)
    intr = :(Core.Intrinsics.$name)
    esc(quote
        $name(x...) = $intr(x...)
        translate(::Val{$intr}) = $name
        function frule(::Dual{typeof($name)}, args::Dual...)
            p = $name(map(primal, args)...)
            Dual(p, zero_tangent(p))
        end
    end)
end

# ---------------------------------------------------------------------------
# Differentiable float intrinsics — hand-written rules.
# ---------------------------------------------------------------------------

# Linear binary ops (`add_float`, `sub_float`): d(a ∘ b) = da ∘ db
for op in (:add_float, :add_float_fast, :sub_float, :sub_float_fast)
    @eval @intrinsic $op
    @eval @inline function frule(::Dual{typeof($op)}, (; x, dx)::Dual, (; y, dy)::Dual)
        Dual($op(x, y), $op(dx, dy))
    end
end

# Linear unary op (`neg_float`): d(-a) = -da
for op in (:neg_float, :neg_float_fast)
    @eval @intrinsic $op
    @eval @inline function frule(::Dual{typeof($op)}, (; x, dx)::Dual)
        Dual($op(x), $op(dx))
    end
end

# Product rule (`mul_float`): d(a·b) = da·b + a·db
for (mul, add) in ((:mul_float, :add_float), (:mul_float_fast, :add_float_fast))
    @eval @intrinsic $mul
    @eval @inline function frule(::Dual{typeof($mul)}, (; x, dx)::Dual, (; y, dy)::Dual)
        Dual($mul(x, y), $add($mul(dx, y), $mul(x, dy)))
    end
end

# Quotient rule (`div_float`): d(a/b) = (da·b − a·db) / b²
for (div, sub, mul) in ((:div_float, :sub_float, :mul_float),
                        (:div_float_fast, :sub_float_fast, :mul_float_fast))
    @eval @intrinsic $div
    @eval function frule(::Dual{typeof($div)}, (; x, dx)::Dual, (; y, dy)::Dual)
        Dual($div(x, y), $div($sub($mul(dx, y), $mul(x, dy)), $mul(y, y)))
    end
end

# `sqrt_llvm(a)`: d(√a) = da / (2√a)
for op in (:sqrt_llvm, :sqrt_llvm_fast)
    @eval @intrinsic $op
    @eval function frule(::Dual{typeof($op)}, (; x, dx)::Dual)
        s = $op(x)
        Dual(s, div_float(dx, add_float(s, s)))     # dx / (2√a)
    end
end

# `abs_float(a)`: d|a| = sign(a)·da. `copysign_float(da, da·a)` = |da|·sign(da·a) = da·sign(a).
@intrinsic abs_float
function frule(::Dual{typeof(abs_float)}, (; x, dx)::Dual)
    Dual(abs_float(x), copysign_float(dx, mul_float(dx, x)))
end

# `max_float`/`min_float`: the tangent follows whichever operand is selected. A `?:` branch (not
# `Core.ifelse`, a builtin) keeps the rule dualizable at higher order.
for (op, cmp) in ((:max_float, :(>=)), (:max_float_fast, :(>=)),
                  (:min_float, :(<=)), (:min_float_fast, :(<=)))
    @eval @intrinsic $op
    @eval function frule(::Dual{typeof($op)}, (; x, dx)::Dual, (; y, dy)::Dual)
        Dual($op(x, y), $cmp(x, y) ? dx : dy)
    end
end

# `fma_float`/`muladd_float`, both `a·b + c`: d = da·b + a·db + dc.
for op in (:fma_float, :muladd_float)
    @eval @intrinsic $op
    @eval function frule(::Dual{typeof($op)}, (; x, dx)::Dual, (; y, dy)::Dual, (; z, dz)::Dual)
        Dual($op(x, y, z), add_float(add_float(mul_float(dx, y), mul_float(x, dy)), dz))
    end
end

# `copysign_float(a, b)` = |a|·sign(b): d/da = sign(a)·sign(b), d/db = 0.
# `copysign_float(da, a·b·da)` = |da|·sign(a·b·da) = da·sign(a)·sign(b).
@intrinsic copysign_float
function frule(::Dual{typeof(copysign_float)}, (; x, dx)::Dual, (; y, dy)::Dual)
    Dual(copysign_float(x, y), copysign_float(dx, mul_float(x, mul_float(y, dx))))
end

# Floating-point width conversions (`Float32(::Float64)` → `fptrunc`, `Float64(::Float32)` →
# `fpext`) carry a *type* as their first argument, which dualizes to a `Dual{DataType,NoTangent}`
# (see the type-arg note in the inactive section below). They are linear in the value:
# d(convert(T, a)) = convert(T, da).
for op in (:fpext, :fptrunc)
    @eval @intrinsic $op
    @eval function frule(::Dual{typeof($op)}, T::Dual, a::Dual)
        Dual($op(primal(T), primal(a)), $op(primal(T), tangent(a)))
    end
end

# ---------------------------------------------------------------------------
# Non-differentiable intrinsics — comparisons, integer arithmetic, bit/boolean ops, rounding to an
# integer value, and int↔float / bit conversions. Each gets a wrapper + an auto-generated
# zero-tangent `frule` via `@inactive_intrinsic`.
#
# The conversion intrinsics (`sitofp`, `fptosi`, `bitcast`, `trunc_int`, …) carry a *type* as their
# first argument. That constant has type `DataType` (which is concrete), so `frule_split!` keeps it
# on the static path and wraps it as a `Dual{DataType,NoTangent}`; the variadic `@inactive_intrinsic`
# rule's `map(primal, args)` then recovers `(T, value)` and re-applies the intrinsic. The result's
# tangent is a genuine zero (a converted-to/from-integer or bit-reinterpreted value has no
# meaningful derivative here — c.f. the differentiable `fpext`/`fptrunc` float-width conversions,
# which DO have rules above).
# ---------------------------------------------------------------------------
for name in (
    # integer & float comparisons (result is a `Bool`)
    :eq_int, :ne_int, :slt_int, :sle_int, :ult_int, :ule_int,
    :eq_float, :ne_float, :lt_float, :le_float, :fpiseq,
    :eq_float_fast, :ne_float_fast, :lt_float_fast, :le_float_fast,
    # integer arithmetic
    :add_int, :sub_int, :mul_int, :neg_int, :sdiv_int, :udiv_int, :srem_int, :urem_int,
    :checked_sadd_int, :checked_ssub_int, :checked_smul_int, :checked_sdiv_int, :checked_srem_int,
    :checked_uadd_int, :checked_usub_int, :checked_umul_int, :checked_udiv_int, :checked_urem_int,
    # bit / boolean ops
    :and_int, :or_int, :xor_int, :not_int, :shl_int, :lshr_int, :ashr_int,
    :bswap_int, :ctlz_int, :ctpop_int, :cttz_int, :flipsign_int,
    # misc queries (result is a `Bool`) — e.g. `have_fma(T)`, emitted by `fma`
    :have_fma,
    # rounding to an integral floating-point value (piecewise-constant → zero derivative)
    :floor_llvm, :ceil_llvm, :trunc_llvm, :rint_llvm,
    # int↔float and integer-width conversions (first argument is a type). NB: `bitcast` is omitted
    # — the name is already taken by `const bitcast` in tangent_utils.jl; it errors if ever hit.
    :sitofp, :uitofp, :fptosi, :fptoui, :trunc_int, :sext_int, :zext_int,
)
    @eval @inactive_intrinsic $name
end
