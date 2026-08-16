module DifferReverseDifferentiationInterfaceExt

import DifferentiationInterface as DI
using DifferReverse: DifferReverse, AutoDifferReverse, CoDual, primal, tangent, rrule!!, Ctx, build_ctx,
    zero_fcodual, set_to_zero!!, ZeroRData, zero_rdata, fdata, rdata, increment!!

DI.check_available(::AutoDifferReverse) = true

# ===========================================================================
# Reverse mode: pullback, built on rrule!!/CoDual/Ctx.
#
# `build_ctx(f, argtypes)` preallocates a tape-holding `Ctx` once, whose stacks are reset and
# reused on every subsequent `rrule!!` call — that's DI's "prepare once, reuse many times" story.
# `value_and_pullback!` builds `CoDual(x, tx[i])` directly against the caller's buffer instead of
# allocating a fresh zero buffer and copying afterward, mirroring Differ's own
# `value_and_gradient!(ctx, fcd, argcds::CoDual...)` convention (fdata-carried args — arrays,
# mutable structs — accumulate in place; rdata-carried args — scalars — have nothing to
# preallocate and are returned by value instead).
# ===========================================================================

struct DifferPullbackPrep{SIG,CtxT} <: DI.PullbackPrep{SIG}
    _sig::Val{SIG}
    ctx::CtxT
end

# `pb` restores pre-write primal values in place, so an fdata-carrying result must be copied out
# before it runs. Mooncake's `_copy_output` does the same.
_di_out_copy(y::AbstractArray) = copy(y)
_di_out_copy(y::Tuple) = map(_di_out_copy, y)
_di_out_copy(y) = y

function DI.prepare_pullback_nokwarg(
        strict::Val, f::F, backend::AutoDifferReverse, x, ty::NTuple, contexts::Vararg{DI.Context,C}
    ) where {F,C}
    _sig = DI.signature(f, backend, x, ty, contexts...; strict)
    cargs = map(DI.unwrap, contexts)
    ctx = build_ctx(f, (typeof(x), map(typeof, cargs)...))
    return DifferPullbackPrep(_sig, ctx)
end

function DI.value_and_pullback!(
        f::F, tx::NTuple{B}, prep::DifferPullbackPrep, backend::AutoDifferReverse, x, ty::NTuple{B},
        contexts::Vararg{DI.Context,C},
    ) where {F,B,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    cargs = map(DI.unwrap, contexts)
    # `y` must come out of the closure's return value, not a reassigned outer variable — assigning
    # to a captured variable from inside a closure forces Julia to heap-box it, and a closure that
    # calls out to `rrule!!` (non-inlinable) is exactly the case escape analysis can't undo.
    outs = ntuple(Val(B)) do i
        fcd = zero_fcodual(f)
        buf = tx[i]
        set_to_zero!!(buf)
        xcd = CoDual(x, buf)
        ccds = map(zero_fcodual, cargs)
        ycd, pb = rrule!!(fcd, prep.ctx, xcd, ccds...)
        y = _di_out_copy(primal(ycd))
        seed = ty[i]
        increment!!(tangent(ycd), fdata(seed))
        rdatas = pb(rdata(seed))
        rd_x = rdatas[2] isa ZeroRData ? zero_rdata(x) : rdatas[2]
        (y, tangent(tangent(xcd), rd_x))
    end
    return outs[1][1], map(last, outs)
end

function DI.value_and_pullback(
        f::F, prep::DifferPullbackPrep, backend::AutoDifferReverse, x, ty::NTuple{B},
        contexts::Vararg{DI.Context,C},
    ) where {F,B,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    tx = ntuple(Val(B)) do _
        tangent(zero_fcodual(x))
    end
    return DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...)
end

function DI.pullback(
        f::F, prep::DifferPullbackPrep, backend::AutoDifferReverse, x, ty::NTuple,
        contexts::Vararg{DI.Context,C},
    ) where {F,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return DI.value_and_pullback(f, prep, backend, x, ty, contexts...)[2]
end

function DI.pullback!(
        f::F, tx::NTuple, prep::DifferPullbackPrep, backend::AutoDifferReverse, x, ty::NTuple,
        contexts::Vararg{DI.Context,C},
    ) where {F,C}
    DI.check_prep(f, prep, backend, x, ty, contexts...)
    return DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...)[2]
end

# `gradient`/`value_and_gradient` need no code — DI's generic `PullbackGradientPrep`
# (src/first_order/gradient.jl) calls pullback with seed `oneunit(typeof(y))`, exactly what
# `rev_gradient`'s own `pb(one(y))` does internally.

end # module DifferReverseDifferentiationInterfaceExt
