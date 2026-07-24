module ADNext

# Tangent / fdata / rdata type system (ported from Mooncake), plus the Mooncake-shaped
# `Dual` / `CoDual` carriers that sit on top of it.
include("tangent_utils.jl")
include("tangents.jl")
include("fwds_rvs_data.jl")
include("array_tangents.jl")
include("dual.jl")
include("codual.jl")

# Forward-mode AD engine, rebuilt on the ported `Dual`.
include("intrinsics.jl")   # intrinsic wrappers + `frule`s (dispatch-based intrinsic handling)
include("frules.jl")
include("contextual.jl")
include("forward_interp.jl")
include("reflection.jl")

# using PrecompileTools: @compile_workload
# include("precomp.jl")

# NOTE: `reverse_interp.jl` (a WIP barebones reverse-mode engine) is intentionally NOT included.
# Its mutable `CoDual` is incompatible with the ported immutable `CoDual{Tx,Tdx}`, and reverse-mode
# AD is out of scope for the tangent/fdata/rdata port. The file is retained on disk.

# Carriers and the forward-mode entry points.
export Dual, CoDual, primal, tangent, NoTangent, frule
export code_dual_ircode, @code_dual_ircode

# Tangent / fdata / rdata type system.
export tangent_type, fdata_type, rdata_type
export Tangent, MutableTangent, PossiblyUninitTangent
export NoFData, NoRData, FData, RData
export fdata, rdata, zero_tangent
export as_tangent, unit_tangent

end # module ADNext
