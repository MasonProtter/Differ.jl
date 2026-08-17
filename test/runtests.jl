using SafeTestsets

# `Differ` is now a thin re-export meta-package (see src/Differ.jl) — the real implementation and
# most of the test suite live in the four sub-packages (Contextual/, DifferCore/, DifferForwards/,
# DifferReverse/), each with its own full test suite. What remains here are genuine cross-package
# integration tests that don't belong to any one sub-package: backedge/invalidation behavior
# spanning both `frule!!` and `rrule!!`, and a combined no-method-ambiguities check across the
# whole re-exported namespace.
@safetestset "backedges: derivative invalidation" begin include("test_backedges.jl") end
@safetestset "rules: no ambiguities" begin include("test_rule_ambiguities.jl") end
@safetestset "DifferentiationInterfaceTest" begin include("dit.jl") end
