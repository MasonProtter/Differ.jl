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

    @testset "sum(::Generator)" begin
        # No hand-written `sum` rule is loaded here — this goes through the generic composite
        # path, which has to dualize `Base._foldl_impl`'s `Base._InitialValue` sentinel check (a
        # `Core.isa` builtin; see `test_forward_control_flow.jl`'s "inactive builtins" testset).
        x = [1.0, 2.0, -3.0, 4.5]

        gs(x) = sum(xi^2 for xi in x)
        d1 = frule!!(zero_dual(gs), Dual(x, ones(length(x))))
        @test d1.x ≈ gs(x)
        @test d1.dx ≈ sum(2 .* x)
        checkverify(gs, (Vector{Float64},))

        eig(x) = sum(x[i]*i for i in eachindex(x))
        d2 = frule!!(zero_dual(eig), Dual(x, ones(length(x))))
        @test d2.x ≈ eig(x)
        @test d2.dx ≈ sum(eachindex(x))
        checkverify(eig, (Vector{Float64},))
    end
end
