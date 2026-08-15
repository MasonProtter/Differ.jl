using Test
using DifferReverse
using DifferReverse: rev_gradient
import DifferentiationInterface as DI

include(joinpath(@__DIR__, "testutils.jl"))

@testset "DI reverse: gradient matches rev_gradient" begin
    fscalar(x) = sin(x) * x * x
    x = 1.3
    @test DI.gradient(fscalar, AutoDifferReverse(), x) ≈ rev_gradient(fscalar, x)[2]
    @test DI.gradient(fscalar, AutoDifferReverse(), x) ≈ central_diff(fscalar, x) rtol = 1e-5

    fvec(v) = sum(sin, v)   # vector -> scalar, known-supported reduction
    v = [0.3, -1.1, 2.4]
    @test DI.gradient(fvec, AutoDifferReverse(), v) ≈ rev_gradient(fvec, v)[2]

    # Regression test for `.`-broadcast through the derived path; d/dx sum(x .+ x) = 2 elementwise.
    @test DI.gradient(x -> sum(x .+ x), AutoDifferReverse(), [1.0, 2.0]) ≈ [2.0, 2.0]
end

@testset "DI reverse: prepared pullback! accumulates in place" begin
    fvec(v) = sum(sin, v)
    v1 = [0.3, -1.1, 2.4]
    v2 = [1.0, 2.0, -0.5]

    prep = DI.prepare_pullback(fvec, AutoDifferReverse(), v1, (1.0,))
    dx = zeros(3)
    y1, (result1,) = DI.value_and_pullback!(fvec, (dx,), prep, AutoDifferReverse(), v1, (1.0,))
    @test y1 ≈ fvec(v1)
    @test result1 === dx           # true in-place accumulation for fdata-carried (array) x
    @test dx ≈ rev_gradient(fvec, v1)[2]

    # reuse the same prep (and its preallocated tape) at a different point
    y2, (result2,) = DI.value_and_pullback!(fvec, (dx,), prep, AutoDifferReverse(), v2, (1.0,))
    @test y2 ≈ fvec(v2)
    @test result2 === dx
    @test dx ≈ rev_gradient(fvec, v2)[2]
end
