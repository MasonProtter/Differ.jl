"""
    Differ

Thin re-export meta-package: `using Differ` pulls in `Contextual`, `DifferCore`,
`DifferForwards`, and `DifferReverse` together, reproducing the pre-split monolith's combined
namespace for existing `using Differ` consumers. No logic of its own — every real implementation
lives in one of the four sub-packages, each independently installable and usable on its own.
`primal`/`tangent` are single generic functions owned by `DifferCore`, with `DifferForwards`
adding the `Dual` method and `DifferReverse` adding the `CoDual` one, so they're already unified
without `Differ` needing to do anything.

# Known limitation: forward-over-reverse is currently unsupported

Differentiating a function that itself calls `rev_gradient`/`value_and_gradient!`/a `Tape`
pullback, *under forward mode* (`frule!!`/`D` applied to such a function) — hangs or crashes. This
composition worked in the pre-split single-module version of this package; it broke as a result of
splitting the tangent/fdata/rdata system (`DifferCore`) and the two AD engines
(`DifferForwards`/`DifferReverse`) into separate packages connected only through
`DifferForwards/ext/DifferForwardsOverReverseExt.jl`'s coupling-point hooks. The hooks themselves
are implemented correctly (verified individually); the failure is a deeper `tangent_type`
dispatch/compilation issue specific to a self-referential struct type (`Tape`) under the custom
`AbstractInterpreter`-based compiler plugin this package is built on. See ISSUES.md #85 for the
full investigation, root-cause findings so far, and suggested next steps — it is a known,
documented, **currently-accepted** limitation, not an actively-worked bug. Every other use of
`DifferForwards`/`DifferReverse`, together or separately (including loading both in one session
without composing them), is unaffected.
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

# Carriers and the forward-mode entry points.
export Dual, CoDual, primal, tangent, NoTangent, frule!!
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
