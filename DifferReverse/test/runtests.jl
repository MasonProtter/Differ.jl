using SafeTestsets

@safetestset "reverse: scalar & struct" begin include("test_reverse_scalar_struct.jl") end
@safetestset "reverse: control flow" begin include("test_reverse_control_flow.jl") end
@safetestset "reverse: block-stack push/edge split (ISSUES #52)" begin include("test_reverse_block_stack_split.jl") end
@safetestset "reverse: counted-loop trip-count compression" begin include("test_reverse_counted_loops.jl") end
@safetestset "reverse: implied-merge predecessor recovery" begin include("test_reverse_implied_merges.jl") end
@safetestset "reverse: dispatch & recursion" begin include("test_reverse_dispatch_recursion.jl") end
@safetestset "reverse: arrays" begin include("test_reverse_arrays.jl") end
@safetestset "reverse: growable arrays" begin include("test_reverse_growable.jl") end
@safetestset "reverse: `.`-broadcast" begin include("test_reverse_broadcast.jl") end
@safetestset "reverse: mutation & aliasing" begin include("test_reverse_mutation_aliasing.jl") end
@safetestset "reverse: tuples" begin include("test_reverse_tuples.jl") end
@safetestset "reverse: Core.ifelse" begin include("test_reverse_ifelse.jl") end
@safetestset "reverse: activity (constant arguments)" begin include("test_reverse_activity.jl") end
@safetestset "reverse: hand-rule activity audit" begin include("test_reverse_rule_activity.jl") end
@safetestset "reverse: closures & globals" begin include("test_reverse_closures_globals.jl") end
@safetestset "cfg / IR" begin include("test_cfg_ir.jl") end
@safetestset "carrier world ranges" begin include("test_world_ranges.jl") end
@safetestset "rules: math" begin include("test_math_rules.jl") end
@safetestset "rules: reductions" begin include("test_reduction_rules.jl") end
@safetestset "rules: broadcast" begin include("test_broadcast_rules.jl") end
@safetestset "rules: indexing" begin include("test_indexing_rules.jl") end
@safetestset "rules: linalg" begin include("test_linalg_rules.jl") end
@safetestset "rules: special functions" begin include("test_specialfunctions_rules.jl") end
@safetestset "reverse: threads" begin include("test_reverse_threads.jl") end
@safetestset "DifferentiationInterface.jl integration" begin include("test_differentiation_interface.jl") end
