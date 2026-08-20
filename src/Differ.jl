"""
    Differ

Thin re-export meta-package: `using Differ` pulls in `Contextual`, `DifferCore`,
`DifferForwards`, and `DifferReverse` together, reproducing the pre-split monolith's combined
namespace for existing `using Differ` consumers. No logic of its own — every real implementation
lives in one of the four sub-packages, each independently installable and usable on its own.
`primal`/`tangent` are single generic functions owned by `DifferCore`, with `DifferForwards`
adding the `Dual` method and `DifferReverse` adding the `CoDual` one, so they're already unified
without `Differ` needing to do anything.

Forward-over-reverse (`D(x -> rev_gradient(f, x), v)` and friends) works: load `DifferForwards` and
`DifferReverse` in either order, and `DifferForwardsOverReverseExt` wires the two engines together.
"""
module Differ

using Contextual
using DifferCore
using DifferForwards
using DifferReverse
# `rev_gradient`/`rev_gradient!` are `public` (not `export`ed) in DifferReverse.jl itself, so a
# bare `using DifferReverse` doesn't bring them into scope — `public` alone declares naming
# intent, it isn't an import mechanism. Bring them in explicitly so `Differ`'s own `public`
# declaration below has a real binding behind it.
using DifferReverse: rev_gradient, rev_gradient!

export DifferCore, DifferForwards, DifferReverse

# Carriers and the forward-mode entry points.
export Dual, CoDual, primal, tangent, NoTangent, Inactive, frule!!
export code_dual_ircode, @code_dual_ircode

# Reverse-mode entry points.
export rrule!!, AbstractCtx, Ctx, build_ctx
export value_and_gradient!, zero_fcodual
export code_reverse_fwds_ircode, @code_reverse_fwds_ircode
export code_reverse_pullback_ircode, @code_reverse_pullback_ircode
export tape_type, comms_element_types

# `public`, not exported: DifferentiationInterface.jl is the primary user-facing entry point now;
# these remain available for direct use without polluting `using Differ`'s namespace.
public rev_gradient, rev_gradient!

# Tangent / fdata / rdata type system.
export tangent_type, fdata_type, rdata_type
export Tangent, MutableTangent, PossiblyUninitTangent
export NoFData, NoRData, FData, RData
export fdata, rdata, zero_tangent
export as_tangent, unit_tangent

# DifferentiationInterface.jl integration (method impls in DifferForwards'/DifferReverse's own
# ext/ extensions).
export AutoDifferForwards, AutoDifferReverse

end # module Differ
