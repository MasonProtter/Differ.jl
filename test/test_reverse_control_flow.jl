using Test
using Differ
using Differ: Dual, NoTangent, frule!!, gradient

include(joinpath(@__DIR__, "testutils.jl"))

@testset "reverse mode: branches" begin
    # `relu` is the multiple-reachable-`return`s shape (the common shape Julia's optimizer
    # actually produces for an `if/else` with a value in each arm — see `_exit_blocks`'s
    # docstring); `branch3` merges all three arms into a single `return` via one `PhiNode` with
    # three predecessors (exercises the `Switch`-with-more-than-two-targets path). Both are checked
    # against finite differences AND the trusted forward-mode `frule!!` (already supports control
    # flow) as an independent cross-check of the new tape/block-stack machinery.
    relu(x) = x > 0.0 ? x : -x
    function branch3(x)
        if x > 2.0
            x*x
        elseif x > 0.0
            x + 1.0
        else
            -x
        end
    end

    for (f, x) in ((relu, 2.0), (relu, -2.0), (branch3, 3.0), (branch3, 1.0), (branch3, -1.0))
        _, dx = gradient(f, x)
        h = 1e-6
        @test dx ≈ (f(x + h) - f(x - h)) / 2h rtol = 1e-5
        @test dx ≈ frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx
    end

    # A branch combined with `%new`/`getfield` (both mechanisms exercised together).
    struct V2; a::Float64; b::Float64; end
    function branch_struct(a, b)
        v = V2(a, b)
        return a > b ? v.a * v.b : v.a + v.b
    end
    for (a, b) in ((5.0, 2.0), (1.0, 4.0))
        _, da, db = gradient(branch_struct, a, b)
        h = 1e-6
        @test da ≈ (branch_struct(a + h, b) - branch_struct(a - h, b)) / 2h rtol = 1e-5
        @test db ≈ (branch_struct(a, b + h) - branch_struct(a, b - h)) / 2h rtol = 1e-5
    end

    checkverify_rev(relu, (Float64,))
    checkverify_rev(branch3, (Float64,))
    checkverify_rev(branch_struct, (Float64, Float64))

    check_stack_balance(relu, 2.0)
    check_stack_balance(relu, -2.0)
    check_stack_balance(branch3, 3.0)
    check_stack_balance(branch3, 1.0)
    check_stack_balance(branch3, -1.0)
end

@testset "reverse mode: Union-typed phis" begin
    # `Union`-typed rdata (Phase 2): a phi merging two branches of different concrete type
    # (`Float64` vs. an `Int` literal) makes reverse mode's rdata accumulator for that SSA value
    # non-concrete. Before the `ZeroRData`-aware `Ref`/`deref_and_zero!`/`route!` machinery, this
    # crashed with `TypeError: in new, expected Union{NoRData, Float64}, got a value of type
    # Differ.CannotProduceZeroRDataFromType`.
    unionphi_ternary(x)  = (x > 0 ? x : 1) * x          # d/dx = 2x for x>0
    unionphi_ternary2(x) = (x > 0 ? x*x : 1) * 1.0      # d/dx = 2x for x>0

    @test Differ.gradient(unionphi_ternary, 1.5) == (NoTangent(), 3.0)
    @test Differ.gradient(unionphi_ternary2, 1.5) == (NoTangent(), 3.0)

    # `Union`-typed rdata across a loop *back-edge* (Phase 2): `unionphi_loop`'s loop-carried `s`
    # is `Union{Int,Float64}` (starts as the `Int` literal `0`, becomes `Float64` after the first
    # iteration), so its accumulator `Ref`'s `deref_and_zero!`/`route!` treatment is exercised
    # repeatedly (once per iteration), not just once.
    function unionphi_loop(x)
        s = 0
        for i in 1:3
            s = s + x
        end
        return s
    end
    @test Differ.gradient(unionphi_loop, 1.5) == (NoTangent(), 3.0)

    checkverify_rev(unionphi_ternary, (Float64,))
    checkverify_rev(unionphi_ternary2, (Float64,))
    checkverify_rev(unionphi_loop, (Float64,))
end

@testset "reverse mode: loops" begin
    # A loop body may execute an unknown number of times, so this is the first place the block
    # stack and per-block comms `Stack`s are actually needed (not just degenerate 0-or-1-entry
    # stacks, as in the branch-only cases) — and the first place rdata `Ref`s must correctly
    # reset/accumulate across repeated visits in exact LIFO order. `sumk`/`sumk2`/`sumk_multi` are
    # the existing forward-mode loop fixtures (a single while-loop, nested while-loops, and two
    # live loop-carried accumulators in one block, respectively) — reused here and cross-checked
    # against the already-trusted `frule!!`.
    function sumk(x, k)
        s = x - x
        i = 0
        while i < k
            s = s + x
            i = i + 1
        end
        s
    end
    function sumk2(x, k, m)
        s = x - x
        i = 0
        while i < k
            j = 0
            while j < m
                s = s + x
                j = j + 1
            end
            i = i + 1
        end
        s
    end
    function sumk_multi(x, y, k)
        s = x - x
        t = y - y
        i = 0
        while i < k
            s = s + x
            t = t + y
            i = i + 1
        end
        s + t
    end

    _, dx_sumk = gradient(sumk, 3.0, 4)
    @test dx_sumk ≈ frule!!(Dual(sumk, NoTangent()), Dual(3.0, 1.0), Dual(4, 0)).dx
    _, dx_sumk2 = gradient(sumk2, 2.0, 3, 5)
    @test dx_sumk2 ≈ frule!!(Dual(sumk2, NoTangent()), Dual(2.0, 1.0), Dual(3, 0), Dual(5, 0)).dx
    _, dx_multi, dy_multi = gradient(sumk_multi, 2.0, 3.0, 4)
    @test dx_multi ≈ frule!!(Dual(sumk_multi, NoTangent()), Dual(2.0, 1.0), Dual(3.0, 0.0), Dual(4, 0)).dx
    @test dy_multi ≈ frule!!(Dual(sumk_multi, NoTangent()), Dual(2.0, 0.0), Dual(3.0, 1.0), Dual(4, 0)).dx
    # A zero-iteration loop (the loop-carried accumulator never updates) is a good edge case.
    _, dx_zero = gradient(sumk, 3.0, 0)
    @test dx_zero == 0.0

    checkverify_rev(sumk, (Float64, Int))
    checkverify_rev(sumk2, (Float64, Int, Int))
    checkverify_rev(sumk_multi, (Float64, Float64, Int))

    check_stack_balance(sumk, 2.0, 5)
    check_stack_balance(sumk, 2.0, 0)   # zero iterations
    check_stack_balance(sumk2, 1.5, 3, 4)
end
