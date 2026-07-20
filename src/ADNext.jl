module ADNext

include("frules.jl")
include("contextual.jl")
include("forward_interp.jl")

export Dual, frule, NoFData, make_zero

end # module ADNext
