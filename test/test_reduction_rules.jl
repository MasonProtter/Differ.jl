using Test
using Differ
using Differ: gradient, rrule!!, Ctx, CoDual, NoFData, primal, tangent, zero_fcodual
using Differ: fdata_type, tangent_type, zero_tangent, fdata

include(joinpath(@__DIR__, "testutils.jl"))

# `checkverify`/`checkverify_rev`/`check_stack_balance` build/verify the *derived* dualized/reverse
# IR for a function's own body — that only makes sense for a genuinely dualizable composite, never
# for a function that is itself only ever reached via a hand rule (same reason the existing test
# suite never calls `checkverify_rev(sin, ...)`/`checkverify_rev(sum, ...)` directly). So each
# hand-ruled reduction below is exercised through a trivial wrapper for those checks, exactly like
# `mysum`/`arr_sum`/`sinloop` elsewhere in the test suite.

@testset "rules: reductions" begin

    @testset "sum" begin
        f = sum
        x = [1.0, 2.0, -3.0, 4.5]
        wrap_sum(x) = sum(x)

        # forward mode
        d = Differ.frule!!(Differ.zero_dual(f), Differ.Dual(x, ones(length(x))))
        @test d.x == sum(x)
        @test d.dx == sum(ones(length(x)))

        # reverse mode
        _, dx = gradient(f, x)
        @test dx == ones(length(x))
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (f(xp) - f(xm)) / 2e-6 rtol = 1e-5
        end

        checkverify(wrap_sum, (Vector{Float64},))
        checkverify_rev(wrap_sum, (Vector{Float64},))
        check_stack_balance(wrap_sum, x)

        # Regression size: Base's real internals switch to a self-recursive pairwise algorithm
        # above `Base.pairwise_blocksize` elements (1024 for `+`); confirm the hand rule still
        # works well past that.
        xbig = rand(2000) .+ 0.5
        _, dxbig = gradient(f, xbig)
        @test dxbig == ones(length(xbig))
    end

    @testset "prod" begin
        f = prod
        x = [1.0, 2.0, 3.0, 1.5]
        wrap_prod(x) = prod(x)

        d = Differ.frule!!(Differ.zero_dual(f), Differ.Dual(x, ones(length(x))))
        @test d.x ≈ prod(x)
        p = prod(x)
        @test d.dx ≈ sum(p / xi for xi in x)

        _, dx = gradient(f, x)
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (f(xp) - f(xm)) / 2e-6 rtol = 1e-5
        end

        checkverify(wrap_prod, (Vector{Float64},))
        checkverify_rev(wrap_prod, (Vector{Float64},))
        check_stack_balance(wrap_prod, x)

        xbig = rand(2000) .+ 0.5   # avoid zeros: prod's pullback divides by each element
        _, dxbig = gradient(f, xbig)
        pbig = prod(xbig)
        for k in (1, 500, 2000)
            @test dxbig[k] ≈ pbig / xbig[k] rtol = 1e-8
        end

        @test_throws ErrorException gradient(prod, Float64[])
    end

    @testset "maximum / minimum" begin
        wrap_max(x) = maximum(x)
        wrap_min(x) = minimum(x)
        for (f, cmp, wrap) in ((maximum, >, wrap_max), (minimum, <, wrap_min))
            x = [3.0, -1.0, 5.0, 2.0]   # no ties, so central differences are well defined everywhere

            d = Differ.frule!!(Differ.zero_dual(f), Differ.Dual(x, collect(1.0:length(x))))
            m = x[1]; expected_idx = 1
            for i in 2:length(x)
                if cmp(x[i], m)
                    m = x[i]; expected_idx = i
                end
            end
            @test d.x == f(x)
            @test d.dx == Float64(expected_idx)

            _, dx = gradient(f, x)
            expected = zeros(length(x))
            expected[expected_idx] = 1.0
            @test dx == expected
            for k in eachindex(x)
                xp = copy(x); xp[k] += 1e-6
                xm = copy(x); xm[k] -= 1e-6
                @test dx[k] ≈ (f(xp) - f(xm)) / 2e-6 rtol = 1e-5
            end

            checkverify(wrap, (Vector{Float64},))
            checkverify_rev(wrap, (Vector{Float64},))
            check_stack_balance(wrap, x)

            xbig = rand(2000)
            _, dxbig = gradient(f, xbig)
            expected_idx_big = cmp === (>) ? argmax(xbig) : argmin(xbig)
            expected_big = zeros(length(xbig))
            expected_big[expected_idx_big] = 1.0
            @test dxbig == expected_big

            @test_throws ErrorException gradient(f, Float64[])
        end

        # Tie-breaking: the *first* occurrence of the extremal value gets the gradient. A central
        # difference cross-check would be ill-defined here (perturbing the non-first tied element
        # also changes the max/min by the same amount), so this is checked exactly instead.
        xtie_max = [3.0, 5.0, 5.0, -1.0]
        _, dx_tie_max = gradient(maximum, xtie_max)
        @test dx_tie_max == [0.0, 1.0, 0.0, 0.0]

        xtie_min = [-1.0, -1.0, 5.0, 2.0]
        _, dx_tie_min = gradient(minimum, xtie_min)
        @test dx_tie_min == [1.0, 0.0, 0.0, 0.0]
    end

    @testset "mapreduce(f, +, x)" begin
        # `mapreduce(f, +, x)`'s 3-positional-argument call shape, once compiled under Differ's
        # reverse-mode interpreter, is resolved as a "dynamic invoke" with `f`/`+` widened to
        # abstract `Function` (confirmed by inspecting `Base.code_typed(...; interp)` directly) —
        # unlike `sum(f, x)`'s 2-argument shape, which survives with concrete argument types. A
        # concrete-type dispatch constraint on `op` therefore never matches in that composite-call
        # path, and the call falls through to (failing) generic recursion — a pre-existing engine
        # limitation in the reverse-mode recursive-call resolution (`_static_recursible_call` /
        # `reverse_interp.jl`), not something fixable from a rules file. So the reverse-mode rule
        # below is exercised directly (`rrule!!` + its pullback), which exercises the exact same
        # rule code a working composite call would eventually reach.
        g(y) = y^2 + 3y   # g'(y) = 2y + 3
        x = [1.0, 2.0, -1.5, 0.5]
        mr(x) = mapreduce(g, +, x)

        # forward mode: composes normally (forward mode dispatches `frule!!` dynamically at the
        # actual call, so the static-type-widening issue above doesn't apply).
        d = Differ.frule!!(
            Differ.zero_dual(mapreduce), Differ.Dual(g, Differ.zero_tangent(g)),
            Differ.zero_dual(+), Differ.Dual(x, ones(length(x))),
        )
        @test d.x ≈ mr(x)
        checkverify(mr, (Vector{Float64},))

        # `d.dx` above is the directional derivative along the all-ones tangent, i.e.
        # sum_i g'(x[i]) = sum_i (2*x[i] + 3); cross-check against that closed form.
        @test d.dx ≈ sum(2 .* x .+ 3) rtol = 1e-8
        # And against a central difference along that same direction.
        h = 1e-6
        central_dir = (mr(x .+ h) - mr(x .- h)) / 2h
        @test d.dx ≈ central_dir rtol = 1e-5

        # reverse mode, direct `rrule!!` call (see note above for why not via `gradient`).
        dx = zeros(length(x))
        ycd, pb = rrule!!(zero_fcodual(mapreduce), Ctx(), zero_fcodual(g), zero_fcodual(+), CoDual(x, dx))
        @test primal(ycd) ≈ mr(x)
        pb(1.0)
        @test dx ≈ 2 .* x .+ 3
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (mr(xp) - mr(xm)) / 2e-6 rtol = 1e-5
        end

        # Reverse-mode guard: a `Union`-typed function argument (its static type can't be resolved
        # to a concrete type) must fail with a clear, located error rather than crash deep inside
        # (ISSUES.md #43 — reverse mode has no dynamic dispatch). Constructed directly against
        # `rrule!!`, mirroring `SumMapPullback`'s own non-concrete-`G` test in
        # `test_reverse_arrays.jl` (the actual call-site route into this state needs reverse-mode
        # dynamic dispatch, not yet implemented — see ISSUES.md #43).
        h1 = y -> y^2
        h2 = y -> 2y
        G2 = Union{typeof(h1),typeof(h2)}
        gcd_u = CoDual{G2,fdata_type(tangent_type(G2))}(h1, fdata(zero_tangent(h1)))
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

    @testset "cumsum" begin
        f = cumsum
        x = [1.0, 2.0, -3.0, 4.5, 0.5]

        # forward mode composes normally.
        d = Differ.frule!!(Differ.zero_dual(f), Differ.Dual(x, ones(length(x))))
        @test d.x == cumsum(x)
        @test d.dx == cumsum(ones(length(x)))

        function cumsum_wrap(x)
            y = cumsum(x)
            s = 0.0
            for i in eachindex(y)
                s += i * y[i]
            end
            return s
        end
        d2 = Differ.frule!!(Differ.zero_dual(cumsum_wrap), Differ.Dual(x, ones(length(x))))
        @test d2.x ≈ cumsum_wrap(x)
        checkverify(cumsum_wrap, (Vector{Float64},))

        # Reverse mode: `y = cumsum(x)` is array-valued, so (like every array-valued primal here)
        # its rdata is `NoRData` — real gradient information flows through its *fdata* (see the
        # comment on the rule in `rules_reductions.jl`). Reading `y[i]` back out of a hand-ruled
        # call's result inside a larger differentiated function needs the surrounding call's result
        # to be recognized as a differentiable-array provenance root, which the current static
        # provenance scan (`_fdata_tracked`, `reverse_interp.jl`) does not do for arbitrary calls
        # (only for specific known operations: `%new`, `memorynew`/`memoryrefnew`/`memoryrefget`) —
        # a separate, pre-existing engine limitation, not fixable from a rules file. So the rule is
        # exercised directly: seed the output fdata `dy` and confirm the pullback computes the
        # reverse cumulative sum into `dx`.
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

        # Cross-check the formula against finite differences of `cumsum` directly: `dx` above should
        # equal `J' * seed` where `J` is `cumsum`'s (lower-triangular-ones) Jacobian.
        n = length(x)
        J = [i <= j ? 1.0 : 0.0 for i in 1:n, j in 1:n]   # dy[j]/dx[i] = 1 for i <= j
        @test dx ≈ J * seed
    end

    @testset "extrema" begin
        f = extrema
        x = [3.0, -1.0, 5.0, 2.0]

        d = Differ.frule!!(Differ.zero_dual(f), Differ.Dual(x, collect(1.0:length(x))))
        @test d.x == extrema(x)
        @test d.dx == (2.0, 3.0)   # tangent of -1.0 (idx 2) and 5.0 (idx 3)

        function extrema_wrap(x)
            mn, mx = extrema(x)
            return 2mn + 3mx
        end

        _, dx = gradient(extrema_wrap, x)
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (extrema_wrap(xp) - extrema_wrap(xm)) / 2e-6 rtol = 1e-5
        end
        expected = zeros(length(x))
        expected[argmin(x)] += 2.0
        expected[argmax(x)] += 3.0
        @test dx ≈ expected

        checkverify(extrema_wrap, (Vector{Float64},))
        checkverify_rev(extrema_wrap, (Vector{Float64},))
        check_stack_balance(extrema_wrap, x)

        @test_throws ErrorException gradient(extrema, Float64[])
    end

end
