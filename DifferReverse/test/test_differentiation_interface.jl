using Test
using DifferReverse
using DifferReverse: rev_gradient
using LinearAlgebra: Diagonal
import DifferentiationInterface as DI

include(joinpath(@__DIR__, "testutils.jl"))

# `arr_to_num_linalg`, ported from DifferentiationInterfaceTest's default scenario set: a regression
# pin for the reverse `^(::Union{Float32,Float64}, ::Integer)` rule at the DI level. `α`/`β` must stay
# module-level `const`s, not literals — a literal integer exponent (`x .^ 4`) goes through Julia's
# `Base.literal_pow` broadcast fusion (a different, currently-unsupported `RefValue`-boxed shape),
# while a `const` global exponent compiles to a plain broadcasted `^` call, matching the DIT scenario.
const _arr_to_num_α = 4
const _arr_to_num_β = 6
arr_to_num_linalg(x::AbstractArray) = sum(vec(x .^ _arr_to_num_α) .* transpose(vec(x .^ _arr_to_num_β)))

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

    # `Int`-element array literal construction and read.
    @test DI.derivative(x -> sin.([1, 2] .* x), AutoDifferReverse(), 0.7) ≈ [cos(0.7), 2cos(1.4)]
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

@testset "DI reverse: pullback and jacobian for a vector-valued function" begin
    fvec(x) = sin.(x)
    x = [0.3, -1.1, 2.4]

    (dx1,) = DI.pullback(fvec, AutoDifferReverse(), x, ([1.0, 0.0, 0.0],))
    @test dx1 ≈ [cos(x[1]), 0.0, 0.0]
    (dx2,) = DI.pullback(fvec, AutoDifferReverse(), x, ([0.0, 1.0, 0.0],))
    @test dx2 ≈ [0.0, cos(x[2]), 0.0]

    J = DI.jacobian(fvec, AutoDifferReverse(), x)
    @test J ≈ Diagonal(cos.(x))

    y, (dx3,) = DI.value_and_pullback(fvec, AutoDifferReverse(), x, ([1.0, 1.0, 1.0],))
    @test y ≈ sin.(x)
    @test dx3 ≈ cos.(x)
end

@testset "DI reverse: arr_to_num_linalg pullback" begin
    x = [0.386, 1.520, 1.979, 0.528]

    (dx,) = DI.pullback(arr_to_num_linalg, AutoDifferReverse(), x, (1.0,))
    @test dx ≈ rev_gradient(arr_to_num_linalg, x)[2]
    for k in eachindex(x)
        xp = copy(x); xp[k] += 1e-6
        xm = copy(x); xm[k] -= 1e-6
        @test dx[k] ≈ (arr_to_num_linalg(xp) - arr_to_num_linalg(xm)) / 2e-6 rtol = 1e-4
    end
end
