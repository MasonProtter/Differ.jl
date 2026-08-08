using Test
using Differ
using Differ: rev_gradient, frule!!, Dual, NoTangent
import DifferentiationInterface as DI

include(joinpath(@__DIR__, "testutils.jl"))

@testset "DI forward: derivative & pushforward" begin
    f(x) = sin(x) * x * x   # ∂/∂x = cos(x)*x^2 + 2*x*sin(x)
    x = 1.3

    d_hand = frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx

    # unprepared
    @test DI.derivative(f, AutoDifferForwards(), x) ≈ d_hand
    @test DI.derivative(f, AutoDifferForwards(), x) ≈ central_diff(f, x) rtol = 1e-5

    y0, t0 = DI.value_and_pushforward(f, AutoDifferForwards(), x, (1.0,))
    @test y0 ≈ f(x)
    @test only(t0) ≈ d_hand

    # prepared, batched over two seed directions — linear in the seed
    prep = DI.prepare_pushforward(f, AutoDifferForwards(), x, (1.0, 2.0))
    y, (ta, tb) = DI.value_and_pushforward(f, prep, AutoDifferForwards(), x, (1.0, 2.0))
    @test y ≈ f(x)
    @test ta ≈ d_hand
    @test tb ≈ 2d_hand
end

@testset "DI forward: pushforward! (mutating, vector output)" begin
    h(x) = [x, 2x, 3x]   # scalar -> vector, so the mutating `ty` buffers make sense
    x = 0.7

    prep = DI.prepare_pushforward(h, AutoDifferForwards(), x, (1.0, 2.0))
    ty1, ty2 = zeros(3), zeros(3)
    y, _ = DI.value_and_pushforward!(h, (ty1, ty2), prep, AutoDifferForwards(), x, (1.0, 2.0))
    @test y ≈ h(x)
    @test ty1 ≈ [1.0, 2.0, 3.0]
    @test ty2 ≈ 2 .* [1.0, 2.0, 3.0]
end

@testset "DI reverse: gradient matches Differ.rev_gradient" begin
    fscalar(x) = sin(x) * x * x
    x = 1.3
    @test DI.gradient(fscalar, AutoDifferReverse(), x) ≈ rev_gradient(fscalar, x)[2]
    @test DI.gradient(fscalar, AutoDifferReverse(), x) ≈ central_diff(fscalar, x) rtol = 1e-5

    fvec(v) = sum(sin, v)   # vector -> scalar, known-supported reduction
    v = [0.3, -1.1, 2.4]
    @test DI.gradient(fvec, AutoDifferReverse(), v) ≈ rev_gradient(fvec, v)[2]
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
