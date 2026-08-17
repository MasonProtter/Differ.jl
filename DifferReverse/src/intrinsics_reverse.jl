# Reverse-mode intrinsic rules — dispatch-based, direct-IR-emission vjp rules for `Core.Intrinsics`,
# mirroring forward mode's `apply_intrinsic_frule!` `Val(f)`-dispatch (`intrinsics.jl`).
#
# `apply_intrinsic_rrule!(Val(f), pvals, dz, Ti, ctx)`: `pvals` are the forward-replayed primal
# operand values (literals pass through as themselves), `dz` is the statement's accumulated rdata,
# `Ti` is the statement's primal result type (used for every emitted op, since rdata for a scalar
# intrinsic is always same-typed as the primal). Returns one rdata contribution per operand, same
# order as `pvals`. `ctx.opf(name, ty, args...)` emits `Expr(:call, GlobalRef(Core.Intrinsics,
# name), args...)` typed `ty`. `ctx.optype(k)` reads operand `k`'s declared primal type from the
# primal IR, for a rule whose conversion depends on operand type rather than `Ti` (`fpext`/`fptrunc`
# below).
#
# The fallback returns `nothing`, so an unregistered intrinsic bails with a located reason instead
# of silently dropping a gradient contribution.

apply_intrinsic_rrule!(::Val{F}, pvals, dz, Ti, ctx) where {F} = nothing

"""
    intrinsic_rrule_deps(::Val{f}) -> NTuple{N,Tuple{Vararg{Int}}} or nothing

One entry per contribution `f`'s vjp rule returns (contributions are 1:1 with operands): entry `j`
lists the operand positions the rule reads out of `pvals` while computing contribution `j`. Consulted
by `_scan_block_comms` and the pullback (`reverse_interp.jl`, via `_intrinsic_needed_operands`) to
decide what the forwards pass must record on the tape: contribution `j` is only relevant when
`_has_rdata_sink`/`ref_for` says operand `j` (what contribution `j` routes back to) actually has
somewhere to accumulate into — a discarded contribution's own operand reads cost nothing. A linear
rule (`add_float`, `neg_float`, …) needs none of its operands — the whole point of it being linear.

`nothing` (the fallback), or an entry count that disagrees with the callee's arity, means "assume
every operand is needed everywhere", so a newly added `apply_intrinsic_rrule!` without a matching
declaration here is conservative and correct, just not minimal. A declaration that *understates* what
its rule reads is caught at build time: the pullback passes an `UnrecordedOperand` marker for every
position it didn't record, `ctx.opf` propagates that marker through any computation that touches it,
and the pullback's route loop raises an error if a live (non-discarded) contribution ever turns out to
be one (see `reverse_pullback_to_ircode`).
"""
intrinsic_rrule_deps(::Val{F}) where {F} = nothing

"""
    UnrecordedOperand(position)

Placeholder passed in `pvals` for an operand the forwards pass deliberately did not record on the
tape, because `intrinsic_rrule_deps` said no *live* contribution reads it. `ctx.opf` propagates this
marker through any computation built from it rather than erroring, since the contribution that reads
it is expected to be a discarded one — the whole point of the optimisation. It only becomes an error
if the pullback's route loop finds one attached to a contribution that *does* have somewhere to route
to, which means `intrinsic_rrule_deps` understated what its rule reads — a bug in that declaration,
not in user code. See `intrinsic_rrule_deps`.
"""
struct UnrecordedOperand
    position::Int
end

# Linear binary ops (`add_float`): z = a + b  =>  da = dz, db = dz
for op in (:add_float, :add_float_fast)
    @eval intrinsic_rrule_deps(::Val{Core.Intrinsics.$op}) = ((), ())
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return dz, dz
    end
end

# `sub_float`: z = a - b  =>  da = dz, db = -dz
for (op, neg) in ((:sub_float, :neg_float), (:sub_float_fast, :neg_float_fast))
    @eval intrinsic_rrule_deps(::Val{Core.Intrinsics.$op}) = ((), ())
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return dz, ctx.opf($(QuoteNode(neg)), Ti, dz)
    end
end

# Linear unary op (`neg_float`): z = -a  =>  da = -dz
for op in (:neg_float, :neg_float_fast)
    @eval intrinsic_rrule_deps(::Val{Core.Intrinsics.$op}) = ((),)
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return (ctx.opf($(QuoteNode(op)), Ti, dz),)
    end
end

# Product rule (`mul_float`): z = a·b  =>  da = b·dz, db = a·dz. Crossed dependency: `da` (routed to
# `a`) reads `b`, `db` (routed to `b`) reads `a` — an inactive `a` still needs `b`'s primal recorded
# (for `db`), not `a`'s own.
for op in (:mul_float, :mul_float_fast)
    @eval intrinsic_rrule_deps(::Val{Core.Intrinsics.$op}) = ((2,), (1,))
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        a, b = pvals[1], pvals[2]
        da = ctx.opf($(QuoteNode(op)), Ti, b, dz)
        db = ctx.opf($(QuoteNode(op)), Ti, a, dz)
        return da, db
    end
end

# `fma_float`/`muladd_float`: z = a·b + c  =>  da = b·dz, db = a·dz, dc = dz. Same crossed a/b
# dependency as `mul_float`; `dc` reads neither operand.
for op in (:fma_float, :muladd_float)
    @eval intrinsic_rrule_deps(::Val{Core.Intrinsics.$op}) = ((2,), (1,), ())
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        a, b = pvals[1], pvals[2]
        return ctx.opf(:mul_float, Ti, b, dz), ctx.opf(:mul_float, Ti, a, dz), dz
    end
end

# Quotient rule (`div_float`): z = a/b  =>  da = dz/b, db = -a·dz/b². `da` (routed to `a`) reads only
# `b`; `db` (routed to `b`) reads both — an inactive `b` still needs `a`'s primal recorded (for `db`).
for (div, neg, mul) in ((:div_float, :neg_float, :mul_float), (:div_float_fast, :neg_float_fast, :mul_float_fast))
    @eval intrinsic_rrule_deps(::Val{Core.Intrinsics.$div}) = ((2,), (1, 2))
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$div}, pvals, dz, Ti, ctx)
        a, b = pvals[1], pvals[2]
        da = ctx.opf($(QuoteNode(div)), Ti, dz, b)
        num = ctx.opf($(QuoteNode(mul)), Ti, a, dz)
        db = ctx.opf($(QuoteNode(neg)), Ti,
                     ctx.opf($(QuoteNode(div)), Ti, num, ctx.opf($(QuoteNode(mul)), Ti, b, b)))
        return da, db
    end
end

# `sitofp`/`uitofp` (Int->Float conversion): result carries rdata (primal type is float), but both
# operands (integer value, type argument) are non-differentiable, so the pullback contributes
# `NoRData()` to each — mirrors `@inactive_intrinsic` on the forward side. Needs its own rule rather
# than being skipped by the `rdtype(Ti) === NoRData` check: it's differentiable result / non-
# differentiable operands, the opposite of the usual inactive shape.
for op in (:sitofp, :uitofp)
    @eval intrinsic_rrule_deps(::Val{Core.Intrinsics.$op}) = ((), ())
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        return ntuple(_ -> NoRData(), length(pvals))
    end
end

# `fpext`/`fptrunc` (`Float32`<->`Float64` width conversion): genuinely differentiable —
# `d(convert(T,a))/da = convert(T,da)`, mirroring forward mode's linear rule. The operand's
# contribution is `dz` converted back via the opposite conversion; its type isn't derivable from
# `pvals` (values, not types) or `Ti` (the result type), so `ctx.optype(2)` reads it from the
# primal IR.
for (op, invop) in ((:fpext, :fptrunc), (:fptrunc, :fpext))
    @eval intrinsic_rrule_deps(::Val{Core.Intrinsics.$op}) = ((), ())
    @eval function apply_intrinsic_rrule!(::Val{Core.Intrinsics.$op}, pvals, dz, Ti, ctx)
        Pa = ctx.optype(2)
        return NoRData(), ctx.opf($(QuoteNode(invop)), Pa, Pa, dz)
    end
end
