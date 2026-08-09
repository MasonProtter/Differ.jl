using SafeTestsets

@safetestset "forward: scalar" begin include("test_forward_scalar.jl") end
@safetestset "forward: control flow" begin include("test_forward_control_flow.jl") end
@safetestset "forward: arrays" begin include("test_forward_arrays.jl") end
@safetestset "forward: pointers & GC.@preserve" begin include("test_forward_pointers.jl") end
@safetestset "forward: foreigncall (ccall)" begin include("test_forward_foreigncall.jl") end
@safetestset "forward: dispatch" begin include("test_forward_dispatch.jl") end
@safetestset "forward: closures & higher-order" begin include("test_forward_closures_higher_order.jl") end
@safetestset "forward: recursion" begin include("test_forward_recursion.jl") end
@safetestset "intrinsic dispatch" begin include("test_intrinsic_dispatch.jl") end
@safetestset "rules: math" begin include("test_math_rules.jl") end
@safetestset "rules: reductions" begin include("test_reduction_rules.jl") end
@safetestset "rules: broadcast" begin include("test_broadcast_rules.jl") end
@safetestset "rules: indexing" begin include("test_indexing_rules.jl") end
@safetestset "rules: linalg" begin include("test_linalg_rules.jl") end
@safetestset "DifferentiationInterface.jl integration" begin include("test_differentiation_interface.jl") end
