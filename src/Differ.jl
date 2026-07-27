module Differ

# Tangent / fdata / rdata type system (ported from Mooncake), plus the Mooncake-shaped
# `Dual` / `CoDual` carriers that sit on top of it.
include("tangent_utils.jl")
include("tangents.jl")
include("fwds_rvs_data.jl")
include("array_tangents.jl")
include("dual.jl")
include("codual.jl")
include("stack.jl")        # Stack/SingletonStack — reverse-mode control-flow replay tape

# Forward-mode AD engine, compiler-level Dual arithmetic
include("intrinsics.jl")   # dispatch-based intrinsic handling (apply_intrinsic_frule!)
include("builtins.jl")     # dispatch-based Core.Builtin handling (apply_builtin_frule!)
include("frules.jl")
include("contextual.jl")
include("cfg_ir.jl")       # ID/CFGBlock working-IR layer (reverse-mode control flow only)
include("forward_interp.jl")

# Reverse-mode AD engine (proof of concept, straight-line code only — see reverse_interp.jl header)
include("intrinsics_reverse.jl")   # dispatch-based intrinsic vjp rules (apply_intrinsic_rrule!)
include("builtins_reverse.jl")     # dispatch-based Core.Builtin vjp rules (apply_builtin_rrule!/etc.)
include("reverse_interp.jl")
include("rrules.jl")               # hand-written reverse-mode rules (mirrors frules.jl)

include("reflection.jl")

# using PrecompileTools: @compile_workload
# include("precomp.jl")

# Carriers and the forward-mode entry points.
export Dual, CoDual, primal, tangent, NoTangent, frule!!
export code_dual_ircode, @code_dual_ircode

# Reverse-mode entry points (branches supported; loops are Phase C — see reverse_interp.jl header).
export rrule!!, AbstractCtx, Ctx, build_ctx
export gradient, gradient!, value_and_gradient!, zero_fcodual
export code_reverse_fwds_ircode, @code_reverse_fwds_ircode
export code_reverse_pullback_ircode, @code_reverse_pullback_ircode

# Tangent / fdata / rdata type system.
export tangent_type, fdata_type, rdata_type
export Tangent, MutableTangent, PossiblyUninitTangent
export NoFData, NoRData, FData, RData
export fdata, rdata, zero_tangent
export as_tangent, unit_tangent

end # module Differ
