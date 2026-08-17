using Test
using DifferReverse
using DifferReverse: rev_gradient, rrule!!, Ctx, CoDual, NoFData, primal, tangent, zero_fcodual

include(joinpath(@__DIR__, "testutils.jl"))

# `cumsum`/`extrema` (rules_reductions.jl, always built in) are covered by the first testset
# below. `sum`/`prod`/`maximum`/`minimum`/`mapreduce(f,+,x)` (rules_perf_backstop.jl, NOT built in
# by default — see that file's header) are covered by the second, which opts the rules in the
# same way a benchmark script would.

@testset "rules: reductions (reverse)" begin

    @testset "cumsum" begin
        x = [1.0, 2.0, -3.0, 4.5, 0.5]

        # y = cumsum(x) is array-valued, so its rdata is NoRData; gradient flows through its fdata
        # (see the comment on the rule in rules_reductions.jl). Exercised directly here: seed the
        # output fdata dy and confirm the pullback computes the reverse cumulative sum into dx.
        # `sum(cumsum(x))` (below) exercises the same rule mid-expression, through the full engine.
        dx = zeros(length(x))
        ycd, pb = rrule!!(zero_fcodual(cumsum), Ctx(), CoDual(x, dx))
        @test primal(ycd) == cumsum(x)
        dy = tangent(ycd)
        @test dy == zeros(length(x))   # fresh zero shadow, to be accumulated into by downstream code
        seed = [1.0, 0.5, -2.0, 3.0, 0.25]
        dy .= seed
        pb(nothing)
        expected = [sum(seed[i:end]) for i in eachindex(seed)]
        @test dx ≈ expected

        # Cross-check the formula against finite differences of `cumsum` directly: `dx` above
        # should equal `J' * seed` where `J` is `cumsum`'s (lower-triangular-ones) Jacobian.
        n = length(x)
        J = [i <= j ? 1.0 : 0.0 for i in 1:n, j in 1:n]   # dy[j]/dx[i] = 1 for i <= j
        @test dx ≈ J * seed
    end

    @testset "cumsum mid-expression (not the function's final return)" begin
        # `cumsum`'s own rule always followed the "caller accumulates into the shadow I returned"
        # contract; the engine previously couldn't route a recursive call's own result shadow at
        # all, so this only ever worked when `cumsum(x)` was itself the function's return value.
        f_cs(x) = sum(cumsum(x))
        x = [1.0, 2.0, -3.0, 4.5, 0.5]
        _, dx = rev_gradient(f_cs, x)
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (f_cs(xp) - f_cs(xm)) / 2e-6 rtol = 1e-5
        end
        # d(sum(cumsum(x)))/dx_i = number of cumsum entries that include x_i = n - i + 1
        @test dx ≈ Float64[length(x) - i + 1 for i in eachindex(x)]
        checkverify_rev(f_cs, (Vector{Float64},))
        check_stack_balance(f_cs, x)
    end

    @testset "sum(f, x) with a closure over a non-differentiable capture" begin
        # `f`'s type is a non-singleton `DataType` (a closure struct with an `Int` field), exercising
        # the argument-position-callee path through the default derived (non-hand-ruled) reduction:
        # `sum(f, v)` recurses into Base's own `mapreduce`/`mapreduce_impl`, calling this closure per
        # element. `n` non-literal means `x^n` runs through the `^(x, ::Integer)` hand rule rather
        # than being constant-folded away.
        n = 3
        v = [0.3, -1.2, 2.0, 0.75]
        sumpown(v) = sum(x -> x^n, v)

        _, dv = rev_gradient(sumpown, v)
        @test dv ≈ n .* v .^ (n - 1)
        for k in eachindex(v)
            vp = copy(v); vp[k] += 1e-6
            vm = copy(v); vm[k] -= 1e-6
            @test dv[k] ≈ (sumpown(vp) - sumpown(vm)) / 2e-6 rtol = 1e-5
        end

        checkverify_rev(sumpown, (Vector{Float64},))
        check_stack_balance(sumpown, v)
    end

    @testset "extrema" begin
        x = [3.0, -1.0, 5.0, 2.0]

        function extrema_wrap(x)
            mn, mx = extrema(x)
            return 2mn + 3mx
        end

        _, dx = rev_gradient(extrema_wrap, x)
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (extrema_wrap(xp) - extrema_wrap(xm)) / 2e-6 rtol = 1e-5
        end
        expected = zeros(length(x))
        expected[argmin(x)] += 2.0
        expected[argmax(x)] += 3.0
        @test dx ≈ expected

        checkverify_rev(extrema_wrap, (Vector{Float64},))
        check_stack_balance(extrema_wrap, x)

        @test_throws ErrorException rev_gradient(extrema, Float64[])
    end

end
