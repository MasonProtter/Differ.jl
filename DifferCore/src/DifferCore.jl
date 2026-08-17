module DifferCore

# Tangent / fdata / rdata type system, ported from Mooncake.jl. Shared by Differ's forward-mode
# `Dual` and reverse-mode `CoDual` carriers (DifferForwards.jl/DifferReverse.jl) — nothing here
# is forward- or reverse-mode-specific.
include("tangent_utils.jl")
include("tangents.jl")
include("fwds_rvs_data.jl")
include("array_tangents.jl")

# Mode-agnostic IR-inspection helpers shared by DifferForwards' and DifferReverse's own
# dualization/pullback engines.
include("shared_ir_helpers.jl")

"""
    primal(x)

The primal value carried by `x` — a `Dual` (forward mode) or a `CoDual` (reverse mode). A bare
stub here; `DifferForwards`/`DifferReverse` each add their own method.
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
export isactive, @ifactive

end # module DifferCore
