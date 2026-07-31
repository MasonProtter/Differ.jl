using Test
using Differ
using Differ: Dual, NoTangent, frule!!, gradient, CoDual, NoFData, NoRData, AbstractCtx, primal

# Tests that Differ attaches real Julia backedges to a compiled derivative, so redefining a primal
# method (or filling in a `frule!!`/`rrule!!` for something that previously errored) invalidates and
# recompiles it — instead of silently keeping a stale `dualized_impl`/`frule!!`/`rrule!!`
# `CodeInstance` around forever. Covers both forward mode (`frule!!`, dualization) and reverse mode
# (`rrule!!`, derived-rule building), which use separate but parallel invalidation machinery.
#
# The forward mechanism (see `contextual.jl`'s `finishinfer!` and `forward_interp.jl`'s
# `primal_of_impl`/`frule_codeinstance`/`compose`): while dualizing a primal's IR, Differ collects
# the MethodInstances it depends on (the primal method itself, any `frule!!` resolved for a surviving
# high-level call, an inner carrier for a composed higher-order derivative) plus mt-backedges keyed
# on the resolution's call signature (so a *new* method — one that didn't exist, or wasn't as
# specific, before — invalidates too, not just a redefinition of the exact same method). These are
# folded into `me.src.edges` before the primal `dualized_impl` `MethodInstance`'s own `finishinfer!`
# runs, so Julia's ordinary `compute_edges!`/`store_backedges` machinery registers them as real
# backedges on *that* `CodeInstance` — the one whose staleness actually gates recompilation, not
# merely some caller of it. Reverse mode's `_optimized_primal_ir`/`resolve_reverse_primal`/
# `reverse_fwds_recursive_ci` (`reverse_interp.jl`) mirror this for `rrule!!`.
#
# Separately (last two testsets in each mode): a related but distinct bug — a callee small enough for
# Julia's ordinary inliner to merge into its caller vanishes from the IR *before*
# `dualize_to_ircode`/`_optimized_primal_ir` ever runs, so a hand-written `frule!!`/`rrule!!` for it
# can never be consulted, no matter the compile order or how good the backedges are.
# `src_inlining_policy` (both files) fixes the "never consulted at all" half by refusing to inline any
# call whose callee has a hand-written rule. `register_implicit_frule_backedge!`/
# `register_implicit_rrule_backedge!` fix the other half — invalidation — by registering a
# speculative mt-backedge on the rule resolution a hypothetical differentiation of each discovered
# callee would use, even for callees whose call was inlined away before a rule existed for them.
@testset "backedges: derivative invalidation" begin
    @testset "redefining a differentiated primal invalidates its derivative" begin
        redefinable_sqr(x) = x * x
        d1 = frule!!(Dual(redefinable_sqr, NoTangent()), Dual(2.0, 1.0))
        @test d1.x == 4.0 && d1.dx == 4.0                     # d/dx(x^2) at 2 = 4

        redefinable_sqr(x) = (x * x) * x                      # redefine: same signature, x^3 now

        d2 = frule!!(Dual(redefinable_sqr, NoTangent()), Dual(2.0, 1.0))
        @test d2.x == 8.0 && d2.dx == 12.0                    # d/dx(x^3) at 2 = 12, not the stale 4
    end

    @testset "a primal that initially errors recompiles once given a real body" begin
        # The first call bails inside the primal itself (an ordinary runtime `error`, not a
        # dualization bail) — `frule!!` still successfully differentiates *through* it, it just
        # throws when invoked, same as calling the primal directly would.
        placeholder_then_real(x) = error("not implemented yet")
        @test_throws ErrorException frule!!(Dual(placeholder_then_real, NoTangent()), Dual(3.0, 1.0))

        placeholder_then_real(x) = (x * x) * (x * x)          # redefine: x^4

        d = frule!!(Dual(placeholder_then_real, NoTangent()), Dual(3.0, 1.0))
        @test d.x == 81.0 && d.dx == 108.0                    # d/dx(x^4) at 3 = 4*27 = 108
    end

    @testset "a hand rule for an inlinable callee is honored, not inlined away" begin
        # Without `src_inlining_policy` (see `forward_interp.jl`), Julia's ordinary cost-based
        # inliner would merge `inlinable_callee`'s tiny body directly into `inlinable_caller`'s
        # optimized IR before `dualize_to_ircode` ever runs, erasing the call entirely — so a
        # hand-written `frule!!` for it could never be consulted, regardless of compile order.
        inlinable_callee(x) = x + 1
        inlinable_caller(x) = inlinable_callee(x)

        function Differ.frule!!(::Dual{typeof(inlinable_callee)}, (; x, dx)::Dual)
            Dual(x + 10, 10 * dx * one(x))
        end

        d = frule!!(Dual(inlinable_caller, NoTangent()), Dual(1.0, 1.0))
        @test d.dx == 10.0                                    # the hand rule, not the inlined d/dx(x+1)=1
    end

    @testset "a hand rule added after first use still invalidates + is honored" begin
        inlinable_callee2(x) = x + 1
        inlinable_caller2(x) = inlinable_callee2(x)

        d1 = frule!!(Dual(inlinable_caller2, NoTangent()), Dual(1.0, 1.0))
        @test d1.dx == 1.0                                    # no rule yet: plain d/dx(x+1) = 1

        function Differ.frule!!(::Dual{typeof(inlinable_callee2)}, (; x, dx)::Dual)
            Dual(x + 10, 10 * dx * one(x))
        end

        d2 = frule!!(Dual(inlinable_caller2, NoTangent()), Dual(1.0, 1.0))
        @test d2.dx == 10.0                                   # recompiled + honors the new hand rule
    end

    @testset "a hand rule for an inlinable *vararg* callee is honored + invalidates" begin
        # Julia's compilation-signature heuristic collapses a vararg callee's trailing arguments, so
        # this callee's `MethodInstance` has `specTypes == Tuple{typeof(inlinable_vcallee), Float64,
        # Vararg{Float64}}` — the arity isn't recorded. `implicit_frule_tt` mirrors that collapse into
        # an open-ended `Vararg{Dual{Float64,Float64}}` tail rather than giving up, which is what makes
        # both halves work here: `src_inlining_policy` keeps the call from being inlined away, and
        # `register_implicit_frule_backedge!` registers the mt-backedge that invalidates the already
        # compiled derivative once the rule appears.
        inlinable_vcallee(a, bs...) = a + sum(bs)
        inlinable_vcaller(x) = inlinable_vcallee(x, x, 2x)

        d1 = frule!!(Dual(inlinable_vcaller, NoTangent()), Dual(1.0, 1.0))
        @test d1.dx == 4.0                                    # no rule yet: d/dx(x + x + 2x) = 4

        function Differ.frule!!(::Dual{typeof(inlinable_vcallee)}, da::Dual, dbs::Dual...)
            Dual(da.x + 100, 100 * da.dx * one(da.x))
        end

        d2 = frule!!(Dual(inlinable_vcaller, NoTangent()), Dual(1.0, 1.0))
        @test d2.dx == 100.0                                  # recompiled + honors the new hand rule
    end

    @testset "reverse: redefining a differentiated primal invalidates its derivative" begin
        redefinable_rsqr(x) = x * x
        _, d1 = gradient(redefinable_rsqr, 2.0)
        @test d1 == 4.0                                       # d/dx(x^2) at 2 = 4

        redefinable_rsqr(x) = (x * x) * x                     # redefine: same signature, x^3 now

        _, d2 = gradient(redefinable_rsqr, 2.0)
        @test d2 == 12.0                                      # d/dx(x^3) at 2 = 12, not the stale 4
    end

    @testset "reverse: a primal that initially errors recompiles once given a real body" begin
        # Unlike forward mode, this doesn't get to the point of an ordinary runtime `error`: a
        # function whose every path throws has no reachable `return`, so reverse mode bails *at
        # compile time* — there's no primal return value to build a pullback structure around
        # (`reverse_error_ircode` embeds `error(msg)` as the generated carrier body). Still an
        # `ErrorException` either way, so the assertion is unchanged.
        placeholder_then_real_rev(x) = error("not implemented yet")
        @test_throws ErrorException gradient(placeholder_then_real_rev, 3.0)

        placeholder_then_real_rev(x) = (x * x) * (x * x)      # redefine: x^4

        _, d = gradient(placeholder_then_real_rev, 3.0)
        @test d == 108.0                                      # d/dx(x^4) at 3 = 4*27 = 108
    end

    @testset "reverse: a hand rule for an inlinable callee is honored, not inlined away" begin
        # Mirrors the forward testset above, but via `rrule!!`/`has_hand_reverse_rule`/
        # `src_inlining_policy` (`reverse_interp.jl`) instead of `frule!!`/`has_hand_frule`.
        inlinable_rcallee(x) = x + 1
        inlinable_rcaller(x) = inlinable_rcallee(x)

        function Differ.rrule!!(::CoDual{typeof(inlinable_rcallee),NoFData}, ::AbstractCtx,
                                xcd::CoDual{Float64,NoFData})
            x = primal(xcd)
            return CoDual(x + 10, NoFData()), Returns((NoRData(), 10.0))
        end

        _, d = gradient(inlinable_rcaller, 1.0)
        @test d == 10.0                                       # the hand rule, not the inlined d/dx(x+1)=1
    end

    @testset "reverse: a hand rule added after first use still invalidates + is honored" begin
        inlinable_rcallee2(x) = x + 1
        inlinable_rcaller2(x) = inlinable_rcallee2(x)

        _, d1 = gradient(inlinable_rcaller2, 1.0)
        @test d1 == 1.0                                       # no rule yet: plain d/dx(x+1) = 1

        function Differ.rrule!!(::CoDual{typeof(inlinable_rcallee2),NoFData}, ::AbstractCtx,
                                xcd::CoDual{Float64,NoFData})
            x = primal(xcd)
            return CoDual(x + 10, NoFData()), Returns((NoRData(), 10.0))
        end

        _, d2 = gradient(inlinable_rcaller2, 1.0)
        @test d2 == 10.0                                      # recompiled + honors the new hand rule
    end
end
