# ===========================================================================
# Intrinsic rules — dispatch-based, direct-IR-emission handling of `Core.Intrinsics`.
#
# Every `Core.Intrinsics` function (`add_float`, `mul_float`, …) is an instance of the *single*
# type `Core.IntrinsicFunction`, so ordinary dispatch on `typeof(f)` can't tell them apart — but an
# intrinsic *value* is itself a valid type parameter, so `Val{Core.Intrinsics.add_float}` names one
# specific intrinsic and ordinary multiple dispatch on `Val` works.
#
# `apply_intrinsic_frule!(Val(f), actual, Ti, ctx)` is called from the main statement loop in
# `dualize_to_ircode` (`forward_interp.jl`) for every intrinsic call in the primal IR. It emits the
# primal + shadow IR *directly* into the caller's instruction stream and returns
# `(primal_ssa, shadow_ssa)` (or `nothing` if unregistered) — there is no `Dual` boxing, no `frule`
# dispatch, and no `CodeInstance` resolution/compile the way a surviving high-level call
# (`frule_split!`, e.g. `sin`/`cos`) needs. That machinery is fine for the handful of calls that
# survive a function's body, but *every* arithmetic op in a differentiated function is an intrinsic
# call — routing each one through a full `frule`/`CodeInstance` round trip (as an earlier version of
# this file did: wrap each intrinsic in a thin wrapper function with its own singleton type, rewrite
# the call to it, and dispatch `frule` on that) bloated both compile time and the generated code.
# Direct emission keeps intrinsics exactly as cheap as the primal computation itself, while still
# reaching each rule via ordinary dispatch instead of a hand-rolled if-else chain.
#
# `ctx` is a `NamedTuple` of the closures `dualize_to_ircode` builds once per call:
#   * `ctx.opf(name, ty, args...)` — emit `Expr(:call, GlobalRef(Core.Intrinsics, name), args...)`
#   * `ctx.emit!(ex, ty)`          — emit any other typed IR statement (e.g. a `Core.ifelse` select)
#   * `ctx.presolve(x)`/`ctx.tresolve(x)` — resolve an operand AST node to its primal/shadow SSA
#   * `ctx.zero_shadow(Ti, primal_ssa)` — the zero tangent of a computed non-differentiable result
#
# Explicit, not implicit: the fallback method below returns `nothing`. An intrinsic with no
# registered rule bails (in `dualize_to_ircode`) with a clear, located reason instead of silently
# miscompiling — e.g. a missing derivative silently returning a wrong zero tangent. Register a
# *differentiable* intrinsic by hand (see below); register a *non-differentiable* one (comparisons,
# integer/bit ops, …) with `@inactive_intrinsic`, which emits the primal and its zero tangent.
# ===========================================================================

apply_intrinsic_frule!(::Val{F}, actual, Ti, ctx) where {F} = nothing

# ---------------------------------------------------------------------------
# Differentiable float intrinsics — hand-written rules, emitted directly.
# ---------------------------------------------------------------------------

# Linear binary ops (`add_float`, `sub_float`): d(a ∘ b) = da ∘ db
for op in (:add_float, :add_float_fast, :sub_float, :sub_float_fast)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a, b = actual[1], actual[2]
        ctx.opf($(QuoteNode(op)), Ti, ctx.presolve(a), ctx.presolve(b)),
        ctx.opf($(QuoteNode(op)), Ti, ctx.tresolve(a), ctx.tresolve(b))
    end
end

# Linear unary op (`neg_float`): d(-a) = -da
for op in (:neg_float, :neg_float_fast)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a = actual[1]
        ctx.opf($(QuoteNode(op)), Ti, ctx.presolve(a)), ctx.opf($(QuoteNode(op)), Ti, ctx.tresolve(a))
    end
end

# Product rule (`mul_float`): d(a·b) = da·b + a·db
for (mul, add) in ((:mul_float, :add_float), (:mul_float_fast, :add_float_fast))
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$mul}, actual, Ti, ctx)
        a, b = actual[1], actual[2]

        pa, pb = ctx.presolve(a), ctx.presolve(b)
        ta, tb = ctx.tresolve(a), ctx.tresolve(b)
        ctx.opf($(QuoteNode(mul)), Ti, pa, pb),
        ctx.opf($(QuoteNode(add)), Ti, ctx.opf($(QuoteNode(mul)), Ti, ta, pb), ctx.opf($(QuoteNode(mul)), Ti, pa, tb))
    end
end

# Quotient rule (`div_float`): d(a/b) = (da·b − a·db) / b²
for (div, sub, mul) in ((:div_float, :sub_float, :mul_float),
                        (:div_float_fast, :sub_float_fast, :mul_float_fast))
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$div}, actual, Ti, ctx)
        a, b = actual[1], actual[2]
        pa, pb = ctx.presolve(a), ctx.presolve(b)
        ta, tb = ctx.tresolve(a), ctx.tresolve(b)
        num = ctx.opf($(QuoteNode(sub)), Ti, ctx.opf($(QuoteNode(mul)), Ti, ta, pb), ctx.opf($(QuoteNode(mul)), Ti, pa, tb))
        ctx.opf($(QuoteNode(div)), Ti, pa, pb),
        ctx.opf($(QuoteNode(div)), Ti, num, ctx.opf($(QuoteNode(mul)), Ti, pb, pb))
    end
end

# `sqrt_llvm(a)`: d(√a) = da / (2√a)
for op in (:sqrt_llvm, :sqrt_llvm_fast)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a = actual[1]
        s = ctx.opf($(QuoteNode(op)), Ti, ctx.presolve(a))
        s, ctx.opf(:div_float, Ti, ctx.tresolve(a), ctx.opf(:add_float, Ti, s, s))
    end
end

# `abs_float(a)`: d|a| = sign(a)·da. `copysign_float(da, da·a)` = |da|·sign(da·a) = da·sign(a).
function apply_intrinsic_frule!(::Val{Core.Intrinsics.abs_float}, actual, Ti, ctx)
    a = actual[1]
    pa, da = ctx.presolve(a), ctx.tresolve(a)
    ctx.opf(:abs_float, Ti, pa), ctx.opf(:copysign_float, Ti, da, ctx.opf(:mul_float, Ti, da, pa))
end

# `max_float`/`min_float`: the tangent follows whichever operand is selected. A branchless
# `Core.ifelse` select — not a Julia `?:`, which would require splitting the block and so break the
# 1:1 block-topology invariant `dualize_to_ircode` relies on — picks it out. `Core.ifelse` is itself
# dualizable (see the builtin case next to `getfield` in `forward_interp.jl`), so this remains
# correct if re-dualized at a higher order.
for (op, le) in ((:max_float, :le_float), (:max_float_fast, :le_float_fast))
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a, b = actual[1], actual[2]
        pa, pb = ctx.presolve(a), ctx.presolve(b)
        cond = ctx.opf($(QuoteNode(le)), Bool, pb, pa)      # pb <= pa  <=>  a is the max
        ctx.opf($(QuoteNode(op)), Ti, pa, pb),
        ctx.emit!(Expr(:call, GlobalRef(Core, :ifelse), cond, ctx.tresolve(a), ctx.tresolve(b)), Ti)
    end
end
for (op, le) in ((:min_float, :le_float), (:min_float_fast, :le_float_fast))
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a, b = actual[1], actual[2]
        pa, pb = ctx.presolve(a), ctx.presolve(b)
        cond = ctx.opf($(QuoteNode(le)), Bool, pa, pb)      # pa <= pb  <=>  a is the min
        ctx.opf($(QuoteNode(op)), Ti, pa, pb),
        ctx.emit!(Expr(:call, GlobalRef(Core, :ifelse), cond, ctx.tresolve(a), ctx.tresolve(b)), Ti)
    end
end

# `fma_float`/`muladd_float`, both `a·b + c`: d = da·b + a·db + dc.
for op in (:fma_float, :muladd_float)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a, b, c = actual[1], actual[2], actual[3]
        pa, pb, pc = ctx.presolve(a), ctx.presolve(b), ctx.presolve(c)
        ta, tb, tc = ctx.tresolve(a), ctx.tresolve(b), ctx.tresolve(c)
        ctx.opf($(QuoteNode(op)), Ti, pa, pb, pc),
        ctx.opf(:add_float, Ti, ctx.opf(:add_float, Ti, ctx.opf(:mul_float, Ti, ta, pb), ctx.opf(:mul_float, Ti, pa, tb)), tc)
    end
end

# `copysign_float(a, b)` = |a|·sign(b): d/da = sign(a)·sign(b), d/db = 0.
# `copysign_float(da, a·b·da)` = |da|·sign(a·b·da) = da·sign(a)·sign(b).
function apply_intrinsic_frule!(::Val{Core.Intrinsics.copysign_float}, actual, Ti, ctx)
    a, b = actual[1], actual[2]
    pa, pb = ctx.presolve(a), ctx.presolve(b)
    da = ctx.tresolve(a)
    ctx.opf(:copysign_float, Ti, pa, pb),
    ctx.opf(:copysign_float, Ti, da, ctx.opf(:mul_float, Ti, pa, ctx.opf(:mul_float, Ti, pb, da)))
end

# Floating-point width conversions (`Float32(::Float64)` → `fptrunc`, `Float64(::Float32)` →
# `fpext`) carry a *type* as their first argument (arg 1 has no tangent — only `ctx.presolve` is
# ever called on it, never `ctx.tresolve`). They are linear in the value:
# d(convert(T, a)) = convert(T, da).
for op in (:fpext, :fptrunc)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        T, a = actual[1], actual[2]
        pT = ctx.presolve(T)
        ctx.opf($(QuoteNode(op)), Ti, pT, ctx.presolve(a)), ctx.opf($(QuoteNode(op)), Ti, pT, ctx.tresolve(a))
    end
end

# ---------------------------------------------------------------------------
# Non-differentiable intrinsics — comparisons, integer arithmetic, bit/boolean ops, rounding to an
# integer value, and int↔float / bit conversions. Each gets an auto-generated rule via
# `@inactive_intrinsic`: compute the primal from the argument primals, give the result a zero
# tangent.
#
# The conversion intrinsics (`sitofp`, `fptosi`, `bitcast`, `trunc_int`, …) carry a *type* as their
# first argument too. `ctx.presolve` is called uniformly over every argument (never `ctx.tresolve`),
# so the type argument just passes through unchanged — no special-casing needed, unlike the
# `Dual`-boxing approach this replaced (which had to resolve a `GlobalRef` type argument to its
# actual `DataType` value before it could be wrapped in a `Dual{DataType,NoTangent}`).
# ---------------------------------------------------------------------------
macro inactive_intrinsic(name)
    intr = :(Core.Intrinsics.$name)
    nmq = QuoteNode(name)
    esc(quote
        function apply_intrinsic_frule!(::Val{$intr}, actual, Ti, ctx)
            p = ctx.opf($nmq, Ti, (ctx.presolve(a) for a in actual)...)
            p, ctx.zero_shadow(Ti, p)
        end
    end)
end

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
