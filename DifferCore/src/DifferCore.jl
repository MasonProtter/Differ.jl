module DifferCore

# Tangent / fdata / rdata type system, ported from Mooncake.jl (`tangent_type`/`fdata_type`/
# `rdata_type` keyed on the *primal* type, `Tangent`/`MutableTangent`/`PossiblyUninitTangent`,
# `FData`/`RData`, `zero_tangent`/`increment!!`). Shared by Differ's forward-mode `Dual` and
# reverse-mode `CoDual` carriers, which live in DifferForwards.jl/DifferReverse.jl respectively —
# nothing in this package is forward- or reverse-mode-specific.
include("tangent_utils.jl")
include("tangents.jl")
include("fwds_rvs_data.jl")
include("array_tangents.jl")

# Small, mode-agnostic IR-inspection helpers shared by DifferForwards' and DifferReverse's own
# dualization/pullback engines (both `include` neither owns exclusively) — see the file header.
include("shared_ir_helpers.jl")

"""
    primal(x)

The primal value carried by `x` — a `Dual` (forward mode, `DifferForwards.jl`) or a `CoDual`
(reverse mode, `DifferReverse.jl`). A bare stub here: `DifferForwards`/`DifferReverse` each add
their own method, so `primal` is one shared generic function across both, the same way `tangent`
already is (`tangent(f, r)` above).
"""
function primal end

export tangent_type, fdata_type, rdata_type
export Tangent, MutableTangent, PossiblyUninitTangent, NoTangent
export NoFData, NoRData, FData, RData
export fdata, rdata, tangent, primal, zero_tangent, zero_rdata, randn_tangent
export increment!!, set_to_zero!!
export build_tangent, get_tangent_field, set_tangent_field!
export as_tangent, unit_tangent
export LazyZeroRData

end # module DifferCore
