using SafeTestsets

@safetestset "tangent system" begin include("test_tangent_system.jl") end
@safetestset "forward: scalar" begin include("test_forward_scalar.jl") end
@safetestset "forward: control flow" begin include("test_forward_control_flow.jl") end
@safetestset "forward: arrays" begin include("test_forward_arrays.jl") end
@safetestset "forward: pointers & GC.@preserve" begin include("test_forward_pointers.jl") end
@safetestset "forward: foreigncall (ccall)" begin include("test_forward_foreigncall.jl") end
@safetestset "forward: dispatch" begin include("test_forward_dispatch.jl") end
@safetestset "forward: closures & higher-order" begin include("test_forward_closures_higher_order.jl") end
@safetestset "reverse: scalar & struct" begin include("test_reverse_scalar_struct.jl") end
@safetestset "reverse: control flow" begin include("test_reverse_control_flow.jl") end
@safetestset "reverse: dispatch & recursion" begin include("test_reverse_dispatch_recursion.jl") end
@safetestset "reverse: arrays" begin include("test_reverse_arrays.jl") end
@safetestset "reverse: mutation & aliasing" begin include("test_reverse_mutation_aliasing.jl") end
@safetestset "reverse: closures & globals" begin include("test_reverse_closures_globals.jl") end
@safetestset "backedges: derivative invalidation" begin include("test_backedges.jl") end
@safetestset "cfg / IR" begin include("test_cfg_ir.jl") end
@safetestset "intrinsic dispatch" begin include("test_intrinsic_dispatch.jl") end
@safetestset "rules: math" begin include("test_math_rules.jl") end
@safetestset "rules: reductions" begin include("test_reduction_rules.jl") end
@safetestset "rules: broadcast" begin include("test_broadcast_rules.jl") end
@safetestset "rules: indexing" begin include("test_indexing_rules.jl") end
@safetestset "rules: linalg" begin include("test_linalg_rules.jl") end
@safetestset "rules: no ambiguities" begin include("test_rule_ambiguities.jl") end
