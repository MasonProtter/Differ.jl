using Test
using DifferForwards
using DifferForwards: frule!!, Dual, NoTangent
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

@testset "DI forward: two-arg (in-place) pushforward/derivative/jacobian" begin
    g!(y, x) = (y .= sin.(x); nothing)
    g_pushforward(x, dx) = cos.(x) .* dx
    x = [1.0, 2.0]
    dx = [3.0, 4.0]
    y = zeros(2)

    ty = DI.pushforward(g!, y, AutoDifferForwards(), x, (dx,))
    @test only(ty) ≈ g_pushforward(x, dx)

    prep = DI.prepare_pushforward(g!, y, AutoDifferForwards(), x, (dx,))
    yv, tv = DI.value_and_pushforward(g!, y, prep, AutoDifferForwards(), x, (dx,))
    @test yv ≈ sin.(x)
    @test only(tv) ≈ g_pushforward(x, dx)

    ty1 = zeros(2)
    _, tv2 = DI.value_and_pushforward!(g!, y, (ty1,), prep, AutoDifferForwards(), x, (dx,))
    @test tv2[1] === ty1   # written directly into the caller's buffer, not copied
    @test ty1 ≈ g_pushforward(x, dx)

    h!(y, x::Number) = (y .= [sin(x), sin(2x)]; nothing)
    xs = 1.3
    der = DI.derivative(h!, zeros(2), AutoDifferForwards(), xs)
    @test der ≈ [cos(xs), 2cos(2xs)]

    J = DI.jacobian(g!, y, AutoDifferForwards(), x)
    @test J ≈ [cos(x[1]) 0.0; 0.0 cos(x[2])]
end

@testset "DI forward: two-arg pushforward with a Constant context" begin
    g!(y, x, a) = (y .= a .* sin.(x); nothing)
    x = [1.0, 2.0]
    dx = [3.0, 4.0]
    a = 2.0
    ty = DI.pushforward(g!, zeros(2), AutoDifferForwards(), x, (dx,), DI.Constant(a))
    @test only(ty) ≈ a .* cos.(x) .* dx
end
