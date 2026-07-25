# Tests that Differ attaches real Julia backedges to a compiled derivative, so redefining a primal
# method (or filling in a `frule!!` for something that previously errored) invalidates and recompiles
# it — instead of silently keeping a stale `dualized_impl`/`frule!!` `CodeInstance` around forever.
#
# The mechanism (see `contextual.jl`'s `finishinfer!` and `forward_interp.jl`'s
# `primal_of_impl`/`frule_codeinstance`/`compose`): while dualizing a primal's IR, Differ collects
# the MethodInstances it depends on (the primal method itself, any `frule!!` resolved for a surviving
# high-level call, an inner carrier for a composed higher-order derivative) plus mt-backedges keyed
# on the resolution's call signature (so a *new* method — one that didn't exist, or wasn't as
# specific, before — invalidates too, not just a redefinition of the exact same method). These are
# folded into `me.src.edges` before the primal `dualized_impl` `MethodInstance`'s own `finishinfer!`
# runs, so Julia's ordinary `compute_edges!`/`store_backedges` machinery registers them as real
# backedges on *that* `CodeInstance` — the one whose staleness actually gates recompilation, not
# merely some caller of it.
#
# Separately (last two testsets): a related but distinct bug — a callee small enough for Julia's
# ordinary inliner to merge into its caller vanishes from the IR *before* `dualize_to_ircode` ever
# runs, so a hand-written `frule!!` for it can never be consulted, no matter the compile order or how
# good the backedges are. `src_inlining_policy` (`forward_interp.jl`) fixes this by refusing to
# inline any call whose callee has a hand-written `frule!!`.
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
end
