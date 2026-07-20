module ADNext

include("frules.jl")
include("contextual.jl")
include("ir_dualize.jl")
include("forward_interp.jl")

export Dual, frule, NoFData, struct_zero

end # module ADNext
