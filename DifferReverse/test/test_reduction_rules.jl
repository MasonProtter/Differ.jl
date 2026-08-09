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

        # y = cumsum(x) is array-valued, so (like every array-valued primal here) its rdata is
        # NoRData; real gradient information flows through its fdata (see the comment on the rule
        # in rules_reductions.jl). Reading y[i] back out of a hand-ruled call's result inside a
        # larger differentiated function needs the surrounding call's result to be recognized as
        # a differentiable-array provenance root, which the current static provenance scan
        # (_fdata_tracked, reverse_interp.jl) does not do for arbitrary calls (only for specific
        # known operations: %new, memorynew/memoryrefnew/memoryrefget). That's a separate,
        # pre-existing engine limitation, not fixable from a rules file. So the rule is exercised
        # directly: seed the output fdata dy and confirm the pullback computes the reverse
        # cumulative sum into dx.
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

# Opt in `rules_perf_backstop.jl` (mirrors how a test or benchmark script loads it per that
# file's own header comment) so `sum`/`prod`/`maximum`/`minimum`/`mapreduce(f,+,x)`'s hand
# `rrule!!`s are available below.
Core.eval(DifferReverse, :(include(joinpath(pkgdir(DifferReverse), "src", "rules_perf_backstop.jl"))))

@testset "rules: reductions (reverse, perf-backstop rules_perf_backstop.jl)" begin

    @testset "sum" begin
        f = sum
        x = [1.0, 2.0, -3.0, 4.5]
        wrap_sum(x) = sum(x)

        _, dx = rev_gradient(f, x)
        @test dx == ones(length(x))
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (f(xp) - f(xm)) / 2e-6 rtol = 1e-5
        end
        checkverify_rev(wrap_sum, (Vector{Float64},))
        check_stack_balance(wrap_sum, x)

        # Regression size: Base's real internals switch to a self-recursive pairwise algorithm
        # above `Base.pairwise_blocksize` elements (1024 for `+`); confirm the hand rule still
        # works well past that.
        xbig = rand(2000) .+ 0.5
        _, dxbig = rev_gradient(f, xbig)
        @test dxbig == ones(length(xbig))
    end

    @testset "sum over a SubArray (ISSUES #65, reverse mode)" begin
        # `sum`'s hand rule is constrained to `X<:Array{<:IEEEFloat}` (above), so `sum(view(v,…))`
        # misses it and falls through to Base's own self-recursive `_mapreduce`/`mapreduce_impl`
        # (`reverse_fwds_recursive_ci`), which now works end to end.
        wrap_view_sum(v) = sum(view(v, 2:4))
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, dv = rev_gradient(wrap_view_sum, v)
        @test dv == [0.0, 1.0, 1.0, 1.0, 0.0]
        checkverify_rev(wrap_view_sum, (Vector{Float64},))
        check_stack_balance(wrap_view_sum, v)
    end

    @testset "sum(f, x) through a composite" begin
        v = [0.3, -1.2, 2.0, 0.75]
        sumsin(v) = sum(sin, v)
        sumabs2(v) = sum(abs2, v)
        sumsq(v)  = sum(y -> y * y, v)   # a local closure operand, this shape always worked

        _, dsin = rev_gradient(sumsin, v)
        @test dsin ≈ cos.(v)
        _, dabs2 = rev_gradient(sumabs2, v)
        @test dabs2 ≈ 2 .* v
        _, dsq = rev_gradient(sumsq, v)
        @test dsq ≈ 2 .* v

        for wrap in (sumsin, sumabs2, sumsq)
            checkverify_rev(wrap, (Vector{Float64},))
            check_stack_balance(wrap, v)
        end
    end

    @testset "reverse mode: :loopinfo (@simd) is carried through" begin
        function simdsum(x)
            s = 0.0
            @simd for i in eachindex(x)
                s += x[i]
            end
            return s
        end
        n = 2000
        x = rand(n)
        _, dx = rev_gradient(simdsum, x)
        @test dx == ones(n)
        checkverify_rev(simdsum, (Vector{Float64},))
        check_stack_balance(simdsum, x)
    end

    @testset "prod" begin
        f = prod
        x = [1.0, 2.0, 3.0, 1.5]
        wrap_prod(x) = prod(x)

        _, dx = rev_gradient(f, x)
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (f(xp) - f(xm)) / 2e-6 rtol = 1e-5
        end
        checkverify_rev(wrap_prod, (Vector{Float64},))
        check_stack_balance(wrap_prod, x)

        @test_throws ErrorException rev_gradient(prod, Float64[])
    end

    @testset "maximum / minimum" begin
        wrap_max(x) = maximum(x)
        wrap_min(x) = minimum(x)
        for (f, cmp, wrap) in ((maximum, >, wrap_max), (minimum, <, wrap_min))
            x = [3.0, -1.0, 5.0, 2.0]   # no ties
            _, dx = rev_gradient(f, x)
            expected_idx = 1; m = x[1]
            for i in 2:length(x)
                cmp(x[i], m) && (m = x[i]; expected_idx = i)
            end
            expected = zeros(length(x)); expected[expected_idx] = 1.0
            @test dx == expected
            checkverify_rev(wrap, (Vector{Float64},))
            check_stack_balance(wrap, x)
            @test_throws ErrorException rev_gradient(f, Float64[])
        end
    end

    @testset "mapreduce(f, +, x)" begin
        g(y) = y^2 + 3y   # g'(y) = 2y + 3
        x = [1.0, 2.0, -1.5, 0.5]
        mr(x) = mapreduce(g, +, x)

        _, dx_mr = rev_gradient(mr, x)
        @test dx_mr ≈ 2 .* x .+ 3
        checkverify_rev(mr, (Vector{Float64},))
        check_stack_balance(mr, x)

        dx = zeros(length(x))
        ycd, pb = rrule!!(zero_fcodual(mapreduce), Ctx(), zero_fcodual(g), zero_fcodual(+), CoDual(x, dx))
        @test primal(ycd) ≈ mr(x)
        pb(1.0)
        @test dx ≈ 2 .* x .+ 3

        # Reverse-mode guard: a Union-typed function argument must fail with a clear, located
        # error rather than crash deep inside (ISSUES.md #43: reverse mode has no dynamic
        # dispatch).
        h1 = y -> y^2
        h2 = y -> 2y
        G2 = Union{typeof(h1),typeof(h2)}
        gcd_u = CoDual{G2,DifferReverse.fdata_type(DifferReverse.tangent_type(G2))}(
            h1, DifferReverse.fdata(DifferReverse.zero_tangent(h1)))
        dx2 = zeros(length(x))
        err = try
            rrule!!(zero_fcodual(mapreduce), Ctx(), gcd_u, zero_fcodual(+), CoDual(x, dx2))
            nothing
        catch e
            e
        end
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("mapreduce", msg)
        @test occursin("concrete", msg)
    end

end
