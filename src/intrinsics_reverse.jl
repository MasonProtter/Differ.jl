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
# `ctx.optype(k)` reads operand `k`'s own *declared* primal type straight from the primal IR — needed
# by a rule whose backward conversion depends on an operand's type rather than the statement's own
# result type `Ti` (`fpext`/`fptrunc` below).
#
# The fallback returns `nothing`, so an intrinsic with no registered reverse rule bails (in
# `reverse_to_ircode`) with a clear, located reason instead of silently dropping a gradient
# contribution.
# ===========================================================================

apply_intrinsic_rrule!(::Val{F}, pvals, dz, Ti, ctx) where {F} = nothing

"""
    intrinsic_rrule_operands(::Val{f}) -> NTuple{N,Int} or nothing

Which of `f`'s operand positions its vjp rule actually reads out of `pvals`. Consulted by
`_scan_block_comms` (`reverse_interp.jl`) to decide what the forwards pass must record on the tape:
an operand no rule ever looks at costs a slot in that block's comms tuple, a push per execution and
a pop per reverse execution, for nothing. A linear rule (`add_float`, `neg_float`, …) needs *none*
of its operands — the whole point of it being linear.

`nothing` (the fallback) means "assume every operand is needed", so a newly added
`apply_intrinsic_rrule!` without a matching declaration here is conservative and correct, just not
minimal. A declaration that is *wrong* — claiming fewer operands than the rule reads — is caught at
build time: the pullback passes an `UnrecordedOperand` marker for every position it didn't record,
and `ctx.opf` refuses to emit a statement referencing one (see `reverse_pullback_to_ircode`).
"""
intrinsic_rrule_operands(::Val{F}) where {F} = nothing

"""
    UnrecordedOperand(position)

Placeholder passed in `pvals` for an operand the forwards pass deliberately did not record on the
tape, because `intrinsic_rrule_operands` said no rule reads it. Reaching `ctx.opf` is a bug in that
declaration, not in user code — see `intrinsic_rrule_operands`.
"""
struct UnrecordedOperand
    position::Int
end

# Linear binary ops (`add_float`): z = a + b  =>  da = dz, db = dz
for op in (:add_float, :add_float_fast)
    @eval intrinsic_rrule_operands(::Val{Core.Intrinsics.$op}) = ()
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return dz, dz
    end
end

# `sub_float`: z = a - b  =>  da = dz, db = -dz
for (op, neg) in ((:sub_float, :neg_float), (:sub_float_fast, :neg_float_fast))
    @eval intrinsic_rrule_operands(::Val{Core.Intrinsics.$op}) = ()
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return dz, ctx.opf($(QuoteNode(neg)), Ti, dz)
    end
end

# Linear unary op (`neg_float`): z = -a  =>  da = -dz
for op in (:neg_float, :neg_float_fast)
    @eval intrinsic_rrule_operands(::Val{Core.Intrinsics.$op}) = ()
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return (ctx.opf($(QuoteNode(op)), Ti, dz),)
    end
end

# Product rule (`mul_float`): z = a·b  =>  da = b·dz, db = a·dz
for op in (:mul_float, :mul_float_fast)
    @eval intrinsic_rrule_operands(::Val{Core.Intrinsics.$op}) = (1, 2)
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        a, b = pvals[1], pvals[2]
        da = ctx.opf($(QuoteNode(op)), Ti, b, dz)
        db = ctx.opf($(QuoteNode(op)), Ti, a, dz)
        return da, db
    end
end

# Quotient rule (`div_float`): z = a/b  =>  da = dz/b, db = -a·dz/b²
for (div, neg, mul) in ((:div_float, :neg_float, :mul_float), (:div_float_fast, :neg_float_fast, :mul_float_fast))
    @eval intrinsic_rrule_operands(::Val{Core.Intrinsics.$div}) = (1, 2)
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$div}, pvals, dz, Ti, ctx)
        a, b = pvals[1], pvals[2]
        da = ctx.opf($(QuoteNode(div)), Ti, dz, b)
        num = ctx.opf($(QuoteNode(mul)), Ti, a, dz)
        db = ctx.opf($(QuoteNode(neg)), Ti,
                     ctx.opf($(QuoteNode(div)), Ti, num, ctx.opf($(QuoteNode(mul)), Ti, b, b)))
        return da, db
    end
end

# `sitofp`/`uitofp` (Int→Float conversion): the RESULT carries rdata (its primal type is a float),
# but both operands — the integer value and the leading type argument — are non-differentiable (an
# integer's tangent is `NoTangent`, a type's is `NoTangent` too). This is the *inactive* bucket,
# mirroring `@inactive_intrinsic` on the forward side (`intrinsics.jl`): the pullback consumes the
# seed and contributes `NoRData()` to every operand. Do not confuse with the linear bucket below —
# these have a differentiable *result* but non-differentiable *operands*, the opposite shape from a
# typical inactive intrinsic (whose result is also non-differentiable), which is why they need their
# own rule at all rather than being skipped by the `rdtype(Ti) === NoRData` check in the caller.
for op in (:sitofp, :uitofp)
    @eval intrinsic_rrule_operands(::Val{Core.Intrinsics.$op}) = ()
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return ntuple(_ -> NoRData(), length(pvals))
    end
end

# `fpext`/`fptrunc` (`Float32`<->`Float64` width conversion): genuinely differentiable, unlike the
# int/float conversions above — `d(convert(T,a))/da = convert(T,da)`, mirroring forward mode's linear
# rule (`intrinsics.jl:148-154`). The operand's contribution is `dz` converted back to the operand's
# own (narrower/wider) type via the *opposite* conversion. The operand's own primal type isn't
# derivable from `pvals` (a resolved value, not a type) or `Ti` (the statement's own result type) —
# `ctx.optype(2)` reads it straight from the primal IR.
for (op, invop) in ((:fpext, :fptrunc), (:fptrunc, :fpext))
    @eval intrinsic_rrule_operands(::Val{Core.Intrinsics.$op}) = ()
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        Pa = ctx.optype(2)
        return NoRData(), ctx.opf($(QuoteNode(invop)), Pa, Pa, dz)
    end
end
