# Hand-written `frule!!` methods. `Dual`, `NoTangent`, `tangent_type`, and the zero-tangent machinery
# come from the ported Mooncake tangent system (`tangents.jl`/`dual.jl`); this file only holds
# transcendental rules we'd rather not differentiate through at the intrinsic level. See the
# `@generated frule!!` fallback in `forward_interp.jl` for everything else.

# These route through `sin`/`cos` (which have hand rules) rather than `sincos`, which isn't itself
# dualizable — at higher order, frule!!-of-frule!! would re-dualize this body and bail on the
# `sincos` call. First-order results and allocation-freedom are unchanged (both stay `:invoke`s to
# libm).
function frule!!(::Dual{typeof(sin)}, (; x, dx)::Dual)
    y = sin(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, cos(x)*dx)
end
function frule!!(::Dual{typeof(cos)}, (; x, dx)::Dual)
    y = cos(x)
    isactive(dx) || return Dual(y, zero_tangent(y))
    Dual(y, -sin(x)*dx)
end

# Arithmetic (`+`, `-`, `*`, `/`) and comparisons deliberately have no `frule!!` methods here. They
# inline to intrinsics (`add_float`, `mul_float`, `lt_float`, …), which the dualization engine
# differentiates directly by dispatching on `Val(f)` (`apply_intrinsic_frule!` in
# `src/intrinsics.jl`): differentiable ones get hand-written rules emitting tangent IR inline,
# non-differentiable ones (comparisons, integer/bit ops) get an auto-generated
# primal-plus-zero-tangent rule via `@inactive_intrinsic`. A bare `frule!!(Dual(+), …)` on a
# *composite* primal still reaches the generated fallback and that same pass, so `+`/`*` work for any
# type (Complex, Float32, …) without a per-type rule. Hand-written `frule!!` methods belong here only
# for functions we don't want to differentiate through at the intrinsic level (transcendentals like
# `sin`/`cos` above); intrinsic rules live in `src/intrinsics.jl`.
