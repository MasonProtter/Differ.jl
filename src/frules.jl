# Hand-written `frule` methods. The `Dual` type, `NoTangent`, `tangent_type`, and the zero-tangent
# machinery now come from the ported Mooncake tangent system (`tangents.jl`/`dual.jl`); this file
# only holds the transcendental rules that we'd rather not differentiate through at the intrinsic
# level. See the `@generated frule` fallback in `forward_interp.jl` for everything else.

# NB: these route through `sin`/`cos` (which have hand rules) rather than `sincos`. `sincos` is not
# itself dualizable, so at higher order the frule-of-frule base case (which re-dualizes this rule
# body) would bail on a `sincos` call. Using `sin`/`cos` keeps the body dualizable to any order.
# First-order results and allocation-freedom are unchanged (both remain `:invoke`s to libm).
function frule(::Dual{typeof(sin)}, (; x, dx)::Dual)
    Dual(sin(x), cos(x)*dx)
end
function frule(::Dual{typeof(cos)}, (; x, dx)::Dual)
    Dual(cos(x), -sin(x)*dx)
end

# NOTE: arithmetic (`+`, `-`, `*`, `/`) and comparisons are intentionally NOT given `frule` methods
# here. They inline to intrinsics (`add_float`, `mul_float`, `lt_float`, …). The dualization engine
# rewrites every intrinsic to its wrapper function and routes it through the wrapper's `frule`
# (`src/intrinsics.jl`): differentiable ones (`add_float`, …) have hand-written rules, and
# non-differentiable ones (comparisons, integer/bit ops) get an auto-generated primal-plus-zero-
# tangent rule via `@inactive_intrinsic`. A bare `frule(Dual(+), …)` therefore routes through the generated fallback into
# that pass, so `+`/`*` work for any type (Complex, Float32, …) without a per-type rule. Keep
# hand-written `frule` methods here only for functions we'd rather not differentiate through
# (transcendentals like `sin`/`cos` above); intrinsic rules live in `src/intrinsics.jl`.
