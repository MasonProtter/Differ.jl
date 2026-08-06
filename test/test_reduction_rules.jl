using Test
using Differ
using Differ: gradient, rrule!!, Ctx, CoDual, NoFData, primal, tangent, zero_fcodual
using Differ: fdata_type, tangent_type, zero_tangent, fdata

include(joinpath(@__DIR__, "testutils.jl"))

# sum/prod/maximum/minimum/mapreduce(f,+,x)'s hand rules moved to src/rules_perf_backstop.jl
# (nested-tape-recycling plan, Stage 3), a known-efficient fallback not included by src/Differ.jl by
# default, so both configurations (derived path alone vs. derived path + these rules) can be
# benchmarked separately. This test suite exercises them directly, so load the file into Differ's
# own namespace (mirroring what Differ.jl would do) rather than this test module's: its bodies
# reference plenty of Differ-internal names unqualified, exactly as they did when the file was
# included from Differ.jl itself.
Core.eval(Differ, :(include(joinpath($(@__DIR__), "..", "src", "rules_perf_backstop.jl"))))

# checkverify/checkverify_rev/check_stack_balance build/verify the derived dualized/reverse IR for
# a function's own body, which only makes sense for a genuinely dualizable composite, never for a
# function that is itself only ever reached via a hand rule (same reason the existing test suite
# never calls checkverify_rev(sin, ...) / checkverify_rev(sum, ...) directly). So each hand-ruled
# reduction below is exercised through a trivial wrapper for those checks, exactly like
# mysum/arr_sum/sinloop elsewhere in the test suite.

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

    @testset "sum over a SubArray (ISSUES #65, reverse mode)" begin
        # `sum`'s hand rules are constrained to `X<:Array{<:IEEEFloat}` (above), so `sum(view(v,…))`
        # misses them and falls through to Base's own `_mapreduce`/`mapreduce_impl`, which is
        # self-recursive. That used to bail unconditionally on the `in_progress` cycle guard; direct
        # self-recursion is now supported (see `reverse_fwds_recursive_ci`, `src/reverse_interp.jl`),
        # and this specific reduction works end to end through it.
        wrap_view_sum(v) = sum(view(v, 2:4))
        v = [1.0, 2.0, 3.0, 4.0, 5.0]

        _, dv = gradient(wrap_view_sum, v)
        @test dv == [0.0, 1.0, 1.0, 1.0, 0.0]
        d = Differ.frule!!(Differ.zero_dual(wrap_view_sum), Differ.Dual(v, ones(length(v))))
        @test d.x == sum(view(v, 2:4))
        @test d.dx ≈ sum(dv)   # JVP in direction `ones(5)` == dot(gradient, direction)

        checkverify(wrap_view_sum, (Vector{Float64},))
        checkverify_rev(wrap_view_sum, (Vector{Float64},))
        check_stack_balance(wrap_view_sum, v)
    end

    @testset "sum(f, x) through a composite" begin
        # Regression for the reverse-mode half of ISSUES #63: in `sum(sin, v)` the `sin` operand of
        # the surviving `sum` call is a `GlobalRef`, which `_static_recursible_call` typed as
        # `GlobalRef` (the node's type, not the named value's). The `sum(f,x)` hand rule then
        # matched with `G === GlobalRef`, its body wouldn't infer, and the whole thing bailed with
        # "recursion into the callee's own reverse-mode forwards pass failed".
        v = [0.3, -1.2, 2.0, 0.75]

        sumsin(v) = sum(sin, v)
        sumabs2(v) = sum(abs2, v)
        sumsq(v)  = sum(y -> y * y, v)   # a local closure operand, this shape always worked

        _, dsin = gradient(sumsin, v)
        @test dsin ≈ cos.(v)
        _, dabs2 = gradient(sumabs2, v)
        @test dabs2 ≈ 2 .* v
        _, dsq = gradient(sumsq, v)
        @test dsq ≈ 2 .* v

        for k in eachindex(v)
            vp = copy(v); vp[k] += 1e-6
            vm = copy(v); vm[k] -= 1e-6
            @test dsin[k] ≈ (sumsin(vp) - sumsin(vm)) / 2e-6 rtol = 1e-5
        end

        for wrap in (sumsin, sumabs2, sumsq)
            checkverify(wrap, (Vector{Float64},))
            checkverify_rev(wrap, (Vector{Float64},))
            check_stack_balance(wrap, v)
        end
    end

    @testset "sum(f, x) with a closure over a non-differentiable capture" begin
        # f's type here is a non-singleton DataType (a closure struct with an Int field), unlike
        # sumsin/sumabs2 above (top-level functions, i.e. singletons). This exercises the other half
        # of the argument-position-callee fix (_static_recursible_call, src/reverse_interp.jl):
        # tangent_type(ftype) === NoTangent is what actually licenses recursion, not
        # Base.issingletontype. `n = 3; sum(x -> x^n, v)` was the case this was designed to cover, but
        # ^ for a non-literal integer exponent goes through Base.Math.pow_body, which recurses
        # through Core.ifelse. Reverse mode has no rule for that builtin (a separate, pre-existing
        # gap, unrelated to this fix), so that exact case still bails. `y -> y + n` isolates the same
        # closure-capture shape without hitting it.
        n = 3
        v = [0.3, -1.2, 2.0, 0.75]
        sumaddn(v) = sum(y -> y + n, v)

        _, dv = gradient(sumaddn, v)
        @test dv == ones(length(v))
        for k in eachindex(v)
            vp = copy(v); vp[k] += 1e-6
            vm = copy(v); vm[k] -= 1e-6
            @test dv[k] ≈ (sumaddn(vp) - sumaddn(vm)) / 2e-6 rtol = 1e-5
        end

        checkverify(sumaddn, (Vector{Float64},))
        checkverify_rev(sumaddn, (Vector{Float64},))
        check_stack_balance(sumaddn, v)
    end

    @testset "reverse mode: :loopinfo (@simd) is carried through" begin
        # Mirrors forward mode's ":loopinfo (@simd) is carried through" (test_forward_foreigncall.jl)
        # for the reverse-mode arm (src/reverse_interp.jl). This is what the "sum"/"sum(f, x) through
        # a composite" testsets above rely on once sum/mapreduce fall through to Base's own
        # mapreduce_impl (an @simd for loop). A hand-written @simd loop isolates the :loopinfo
        # passthrough itself from anything about mapreduce's own recursive structure.
        function simdsum(x)
            s = 0.0
            @simd for i in eachindex(x)
                s += x[i]
            end
            return s
        end
        n = 2000   # past `Base.pairwise_blocksize`, matching forward mode's own regression size
        x = rand(n)
        _, dx = gradient(simdsum, x)
        @test dx == ones(n)

        checkverify_rev(simdsum, (Vector{Float64},))
        check_stack_balance(simdsum, x)

        # @simd ivdep: reverse mode drops the julia.ivdep marker (it would otherwise assert no
        # loop-carried memory dependence, which is false of the reverse carrier's epilogue; see the
        # comment on the fwds carrier's :loopinfo arm). Only affects vectorization, so the gradient
        # must still be correct.
        function simdsum_ivdep(x)
            s = 0.0
            @simd ivdep for i in eachindex(x)
                s += x[i]
            end
            return s
        end
        _, dx_ivdep = gradient(simdsum_ivdep, x)
        @test dx_ivdep == ones(n)
        checkverify_rev(simdsum_ivdep, (Vector{Float64},))
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
        # mapreduce(f, +, x) compiles to a "dynamic invoke" whose specialization signature has f/op
        # widened to abstract Function (Base doesn't specialize on a function argument it only passes
        # along). That widening is in specTypes only; the operands themselves are a closure value and
        # a GlobalRef, whose real types _static_recursible_call reads from the IR. It used to read
        # the GlobalRef as type GlobalRef and never match the rule's concrete-op constraint, which is
        # why this case was once documented as an engine limitation and exercised only through a
        # direct rrule!! call. It now composes through gradient (see the composite checks below).
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

        # reverse mode through the composite call (the case the note above used to say was
        # impossible), and then the same rule driven directly.
        _, dx_mr = gradient(mr, x)
        @test dx_mr ≈ 2 .* x .+ 3
        checkverify_rev(mr, (Vector{Float64},))
        check_stack_balance(mr, x)

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

        # Reverse-mode guard: a Union-typed function argument (its static type can't be resolved to
        # a concrete type) must fail with a clear, located error rather than crash deep inside
        # (ISSUES.md #43: reverse mode has no dynamic dispatch). Constructed directly against
        # rrule!!, mirroring SumMapPullback's own non-concrete-G test in test_reverse_arrays.jl (the
        # actual call-site route into this state needs reverse-mode dynamic dispatch, not yet
        # implemented; see ISSUES.md #43).
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

        # Reverse mode: y = cumsum(x) is array-valued, so (like every array-valued primal here) its
        # rdata is NoRData; real gradient information flows through its fdata (see the comment on
        # the rule in rules_reductions.jl). Reading y[i] back out of a hand-ruled call's result
        # inside a larger differentiated function needs the surrounding call's result to be
        # recognized as a differentiable-array provenance root, which the current static provenance
        # scan (_fdata_tracked, reverse_interp.jl) does not do for arbitrary calls (only for specific
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
