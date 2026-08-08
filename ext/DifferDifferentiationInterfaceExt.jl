module DifferDifferentiationInterfaceExt

import DifferentiationInterface as DI
using Differ: Differ, AutoDifferForwards, AutoDifferReverse,
    Dual, CoDual, primal, tangent, frule!!, rrule!!, Ctx, build_ctx,
    zero_fcodual, zero_tangent, set_to_zero!!, ZeroRData, zero_rdata

DI.check_available(::AutoDifferForwards) = true
DI.check_available(::AutoDifferReverse) = true

# ===========================================================================
# Forward mode: pushforward, built on frule!!/Dual.
#
# There is no tape/config to preallocate — frule!! is stateless per call — so preparation only
# caches `Dual(f, zero_tangent(f))`, which matters when `f` is a closure with a non-trivial
# capture tangent. The output tangent is always freshly allocated by the dualized code itself (no
# hook to write into a caller buffer), so the mutating variant is a real `frule!!` call followed
# by `copyto!` into the caller's buffer.
# ===========================================================================

struct DifferPushforwardPrep{SIG,FD} <: DI.PushforwardPrep{SIG}
    _sig::Val{SIG}
    fdual::FD
end

function DI.prepare_pushforward_nokwarg(
        strict::Val, f::F, backend::AutoDifferForwards, x, tx::NTuple, contexts::Vararg{DI.Context,C}
    ) where {F,C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    return DifferPushforwardPrep(_sig, Dual(f, zero_tangent(f)))
end

function DI.value_and_pushforward(
        f::F, prep::DifferPushforwardPrep, backend::AutoDifferForwards, x, tx::NTuple{B},
        contexts::Vararg{DI.Context,C},
    ) where {F,B,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    cargs = map(DI.unwrap, contexts)
    # `y` must come out of the closure's return value, not a reassigned outer variable — assigning
    # to a captured variable from inside a closure forces Julia to heap-box it, and a closure that
    # calls out to `frule!!` (non-inlinable) is exactly the case escape analysis can't undo.
    outs = ntuple(Val(B)) do i
        cduals = map(c -> Dual(c, zero_tangent(c)), cargs)
        yd = frule!!(prep.fdual, Dual(x, tx[i]), cduals...)
        (primal(yd), tangent(yd))
    end
    return outs[1][1], map(last, outs)
end

function DI.value_and_pushforward!(
        f::F, ty::NTuple{B}, prep::DifferPushforwardPrep, backend::AutoDifferForwards, x, tx::NTuple{B},
        contexts::Vararg{DI.Context,C},
    ) where {F,B,C}
    y, new_ty = DI.value_and_pushforward(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return y, ty
end

function DI.pushforward(
        f::F, prep::DifferPushforwardPrep, backend::AutoDifferForwards, x, tx::NTuple,
        contexts::Vararg{DI.Context,C},
    ) where {F,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.value_and_pushforward(f, prep, backend, x, tx, contexts...)[2]
end

function DI.pushforward!(
        f::F, ty::NTuple, prep::DifferPushforwardPrep, backend::AutoDifferForwards, x, tx::NTuple,
        contexts::Vararg{DI.Context,C},
    ) where {F,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    return DI.value_and_pushforward!(f, ty, prep, backend, x, tx, contexts...)[2]
end

# `derivative`/`value_and_derivative` need no code — DI's generic `PushforwardDerivativePrep`
# (src/first_order/derivative.jl) derives them from pushforward via `oneunit(x)`.

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
        rdatas = pb(ty[i])
        rd_x = rdatas[2] isa ZeroRData ? zero_rdata(x) : rdatas[2]
        (primal(ycd), tangent(tangent(xcd), rd_x))
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
# `Differ.rev_gradient`'s own `pb(one(y))` does internally.

end # module DifferDifferentiationInterfaceExt
