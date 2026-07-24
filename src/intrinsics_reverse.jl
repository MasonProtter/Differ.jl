# ===========================================================================
# Reverse-mode intrinsic rules — dispatch-based, direct-IR-emission backward (vjp) rules for
# `Core.Intrinsics`, mirroring the forward-mode `apply_intrinsic_frule!` dispatch in
# `intrinsics.jl` exactly (see that file's header for the `Val(f)`-dispatch trick).
#
# `apply_intrinsic_rrule!(Val(f), pvals, dz, Ti, ctx)` is called from the reverse walk in
# `reverse_to_ircode` (`reverse_interp.jl`) for every intrinsic call whose result has a
# non-`nothing` rdata accumulator. `pvals` is the tuple of *already forward-replayed* primal SSA
# values for the statement's operands (in argument order — literals pass through as themselves);
# `dz` is the SSA value (or literal) holding the statement's own accumulated rdata; `Ti` is the
# statement's primal result type, used as the type for every emitted arithmetic op (rdata for a
# scalar intrinsic is always same-typed as the primal, so one type suffices for the whole rule).
# Returns a tuple of rdata *contributions*, one per operand, in the same order as `pvals` — the
# caller routes each into that operand's accumulator (skipping operands that are literals/`GlobalRef`s
# and so have no accumulator slot at all).
#
# `ctx.opf(name, ty, args...)` is the same tiny helper as forward mode's: emit
# `Expr(:call, GlobalRef(Core.Intrinsics, name), args...)` typed `ty`, return its `SSAValue`.
#
# Explicit, not implicit: the fallback returns `nothing`, so an intrinsic with no registered
# reverse rule bails (in `reverse_to_ircode`) with a clear, located reason instead of silently
# dropping a gradient contribution.
# ===========================================================================

apply_intrinsic_rrule!(::Val{F}, pvals, dz, Ti, ctx) where {F} = nothing

# Linear binary ops (`add_float`): z = a + b  =>  da = dz, db = dz
for op in (:add_float, :add_float_fast)
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return dz, dz
    end
end

# `sub_float`: z = a - b  =>  da = dz, db = -dz
for (op, neg) in ((:sub_float, :neg_float), (:sub_float_fast, :neg_float_fast))
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return dz, ctx.opf($(QuoteNode(neg)), Ti, dz)
    end
end

# Linear unary op (`neg_float`): z = -a  =>  da = -dz
for op in (:neg_float, :neg_float_fast)
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return (ctx.opf($(QuoteNode(op)), Ti, dz),)
    end
end

# Product rule (`mul_float`): z = a·b  =>  da = b·dz, db = a·dz
for op in (:mul_float, :mul_float_fast)
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        a, b = pvals[1], pvals[2]
        da = ctx.opf($(QuoteNode(op)), Ti, b, dz)
        db = ctx.opf($(QuoteNode(op)), Ti, a, dz)
        return da, db
    end
end

# Quotient rule (`div_float`): z = a/b  =>  da = dz/b, db = -a·dz/b²
for (div, neg, mul) in ((:div_float, :neg_float, :mul_float), (:div_float_fast, :neg_float_fast, :mul_float_fast))
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$div}, pvals, dz, Ti, ctx)
        a, b = pvals[1], pvals[2]
        da = ctx.opf($(QuoteNode(div)), Ti, dz, b)
        num = ctx.opf($(QuoteNode(mul)), Ti, a, dz)
        db = ctx.opf($(QuoteNode(neg)), Ti,
                     ctx.opf($(QuoteNode(div)), Ti, num, ctx.opf($(QuoteNode(mul)), Ti, b, b)))
        return da, db
    end
end
