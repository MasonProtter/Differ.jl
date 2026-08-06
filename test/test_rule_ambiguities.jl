using Test
using Differ

# Two hand rules that are mutually ambiguous do NOT error at definition time: `hand_reverse_rule_match`/
# `has_hand_frule` treat "ambiguous" the same as "no match" (`Core.Compiler`'s `findsup` returns
# `nothing` for both), so inlining suppression silently turns off and the generic derived-recursion
# fallback runs instead, which can be silently wrong rather than loudly broken (see ISSUES.md). This is
# the permanent regression guard against the 5 new rule files (`rules_math.jl`/`rules_reductions.jl`/
# `rules_broadcast.jl`/`rules_indexing.jl`/`rules_linalg.jl`) accidentally introducing one.
@testset "no method ambiguities introduced by hand-written frule!!/rrule!! rules" begin
    @test isempty(Test.detect_ambiguities(Differ))
end
