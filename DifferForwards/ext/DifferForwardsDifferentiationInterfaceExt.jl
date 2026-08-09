module DifferForwardsDifferentiationInterfaceExt

import DifferentiationInterface as DI
using DifferForwards: DifferForwards, AutoDifferForwards, Dual, primal, tangent, frule!!, zero_tangent

DI.check_available(::AutoDifferForwards) = true

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

end # module DifferForwardsDifferentiationInterfaceExt
