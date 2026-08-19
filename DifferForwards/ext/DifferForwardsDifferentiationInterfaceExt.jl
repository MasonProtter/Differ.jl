module DifferForwardsDifferentiationInterfaceExt

import DifferentiationInterface as DI
using DifferForwards: DifferForwards, AutoDifferForwards, Dual, primal, tangent, frule!!, zero_tangent,
    set_to_zero!!, Inactive

DI.check_available(::AutoDifferForwards) = true

# Only `DI.Constant` maps to an inactive `Dual` — not `Cache` (its own docstring example writes an
# *active* argument into it), not `ConstantOrCache` (conservatively active), not `FunctionContext`
# (may carry a function with differentiable captures). Never dispatch on `GeneralizedConstant`.
# Mirrors `_ctx_codual` in the DifferReverse extension.
_ctx_dual(c::DI.Constant) = Dual(DI.unwrap(c), Inactive())
_ctx_dual(c::DI.Context) = (v = DI.unwrap(c); Dual(v, zero_tangent(v)))

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
    # `y` must come out of the closure's return value, not a reassigned outer variable — assigning
    # to a captured variable from inside a closure forces Julia to heap-box it, and a closure that
    # calls out to `frule!!` (non-inlinable) is exactly the case escape analysis can't undo.
    outs = ntuple(Val(B)) do i
        cduals = map(_ctx_dual, contexts)
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

# Two-argument (in-place) primal f!(y, x). DI's generic PushforwardSlow-only
# `_prepare_pushforward_aux` doesn't cover forward mode, so `prepare`/`value_and_pushforward` need
# their own overloads here; two-arg `pushforward`/`pushforward!`/`value_and_pushforward!` and the
# jacobian path are generic on `PushforwardPrep` and come for free.

function DI.prepare_pushforward_nokwarg(
        strict::Val, f!::F, y, backend::AutoDifferForwards, x, tx::NTuple, contexts::Vararg{DI.Context,C}
    ) where {F,C}
    _sig = DI.signature(f!, y, backend, x, tx, contexts...; strict)
    return DifferPushforwardPrep(_sig, Dual(f!, zero_tangent(f!)))
end

function DI.value_and_pushforward(
        f!::F, y, prep::DifferPushforwardPrep, backend::AutoDifferForwards, x, tx::NTuple{B},
        contexts::Vararg{DI.Context,C},
    ) where {F,B,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    ty = ntuple(Val(B)) do i
        dy = zero_tangent(y)  # fresh per tangent, and zero since y is an output, not an input
        cduals = map(_ctx_dual, contexts)
        frule!!(prep.fdual, Dual(y, dy), Dual(x, tx[i]), cduals...)
        dy
    end
    return y, ty
end

# Writes directly into the caller's `ty` buffers instead of DI's generic fallback (allocate + copyto!).
function DI.value_and_pushforward!(
        f!::F, y, ty::NTuple{B}, prep::DifferPushforwardPrep, backend::AutoDifferForwards, x, tx::NTuple{B},
        contexts::Vararg{DI.Context,C},
    ) where {F,B,C}
    DI.check_prep(f!, y, prep, backend, x, tx, contexts...)
    for i in 1:B
        dy = set_to_zero!!(ty[i])
        cduals = map(_ctx_dual, contexts)
        frule!!(prep.fdual, Dual(y, dy), Dual(x, tx[i]), cduals...)
    end
    return y, ty
end

end # module DifferForwardsDifferentiationInterfaceExt
