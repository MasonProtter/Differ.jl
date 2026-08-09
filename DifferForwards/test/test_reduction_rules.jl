using Test
using DifferForwards
using DifferForwards: Dual, zero_dual, frule!!, NoTangent

include(joinpath(@__DIR__, "testutils.jl"))

# `cumsum`/`extrema` (rules_reductions.jl, always built in) are covered by the first testset
# below. `sum`/`prod`/`maximum`/`minimum`/`mapreduce(f,+,x)` (rules_perf_backstop.jl, NOT built in
# by default — see that file's header) are covered by the second, which opts the rules in the
# same way a benchmark script would.

@testset "rules: reductions (forward)" begin

    @testset "cumsum" begin
        f = cumsum
        x = [1.0, 2.0, -3.0, 4.5, 0.5]

        d = frule!!(zero_dual(f), Dual(x, ones(length(x))))
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
        d2 = frule!!(zero_dual(cumsum_wrap), Dual(x, ones(length(x))))
        @test d2.x ≈ cumsum_wrap(x)
        checkverify(cumsum_wrap, (Vector{Float64},))
    end

    @testset "extrema" begin
        f = extrema
        x = [3.0, -1.0, 5.0, 2.0]

        d = frule!!(zero_dual(f), Dual(x, collect(1.0:length(x))))
        @test d.x == extrema(x)
        @test d.dx == (2.0, 3.0)   # tangent of -1.0 (idx 2) and 5.0 (idx 3)

        function extrema_wrap(x)
            mn, mx = extrema(x)
            return 2mn + 3mx
        end
        checkverify(extrema_wrap, (Vector{Float64},))
    end

end

# Opt in `rules_perf_backstop.jl` (mirrors how a test or benchmark script loads it per that
# file's own header comment) so `sum`/`prod`/`maximum`/`minimum`/`mapreduce(f,+,x)`'s hand
# `frule!!`s are available below.
Core.eval(DifferForwards, :(include(joinpath(pkgdir(DifferForwards), "src", "rules_perf_backstop.jl"))))

@testset "rules: reductions (forward, perf-backstop rules_perf_backstop.jl)" begin

    @testset "sum" begin
        x = [1.0, 2.0, -3.0, 4.5]
        d = frule!!(zero_dual(sum), Dual(x, ones(length(x))))
        @test d.x == sum(x)
        @test d.dx == sum(ones(length(x)))
        wrap_sum(x) = sum(x)
        checkverify(wrap_sum, (Vector{Float64},))
    end

    @testset "sum(f, x) through a composite" begin
        v = [0.3, -1.2, 2.0, 0.75]
        sumsin(v) = sum(sin, v)
        for k in eachindex(v)
            dvk = zeros(length(v)); dvk[k] = 1.0
            d = frule!!(Dual(sumsin, NoTangent()), Dual(v, dvk))
            expected = central_diff(x -> (v2 = copy(v); v2[k] = x; sumsin(v2)), v[k])
            @test d.dx ≈ expected atol=1e-6
        end
        checkverify(sumsin, (Vector{Float64},))
    end

    @testset "prod" begin
        x = [1.0, 2.0, 3.0, 1.5]
        d = frule!!(zero_dual(prod), Dual(x, ones(length(x))))
        @test d.x ≈ prod(x)
        p = prod(x)
        @test d.dx ≈ sum(p / xi for xi in x)
        wrap_prod(x) = prod(x)
        checkverify(wrap_prod, (Vector{Float64},))
    end

    @testset "maximum / minimum" begin
        for (f, cmp, wrap) in ((maximum, >, x -> maximum(x)), (minimum, <, x -> minimum(x)))
            x = [3.0, -1.0, 5.0, 2.0]   # no ties
            d = frule!!(zero_dual(f), Dual(x, collect(1.0:length(x))))
            m = x[1]; expected_idx = 1
            for i in 2:length(x)
                if cmp(x[i], m)
                    m = x[i]; expected_idx = i
                end
            end
            @test d.x == f(x)
            @test d.dx == Float64(expected_idx)
            checkverify(wrap, (Vector{Float64},))
        end
    end

    @testset "mapreduce(f, +, x)" begin
        g(y) = y^2 + 3y   # g'(y) = 2y + 3
        x = [1.0, 2.0, -1.5, 0.5]
        mr(x) = mapreduce(g, +, x)
        d = frule!!(
            zero_dual(mapreduce), Dual(g, NoTangent()),
            zero_dual(+), Dual(x, ones(length(x))),
        )
        @test d.x ≈ mr(x)
        @test d.dx ≈ sum(2 .* x .+ 3) rtol = 1e-8
        checkverify(mr, (Vector{Float64},))
    end

end
