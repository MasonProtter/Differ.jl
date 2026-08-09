using Test
using DifferReverse
using DifferReverse: rev_gradient, CoDual, NoFData, RData, Ctx, rrule!!, zero_fcodual, primal, tangent

include(joinpath(@__DIR__, "testutils.jl"))

# Generic checks for a unary scalar function: reverse gradient against central differences, at
# every x in xs, plus an IR-legality check. Forward-mode half of this file's original combined
# check lives in DifferForwards/test/test_math_rules.jl.
function check_unary(f, xs; rtol=1e-6)
    for x in xs
        _, gx = rev_gradient(f, x)
        @test gx ≈ central_diff(f, x) rtol = rtol
    end
    # See the comment in DifferForwards/test/test_math_rules.jl's `check_unary` for why `f` is
    # wrapped before verifying.
    wrapped(x) = f(x)
    checkverify_rev(wrapped, (Float64,))
end

# Generic checks for a binary scalar function f(x, y): reverse gradient against central
# differences, plus an IR-legality check.
function check_binary(f, xys; rtol=1e-6)
    for (x, y) in xys
        _, gx, gy = rev_gradient(f, x, y)
        @test gx ≈ central_diff(f, x, y, 1) rtol = rtol
        @test gy ≈ central_diff(f, x, y, 2) rtol = rtol
    end
    wrapped(x, y) = f(x, y)
    checkverify_rev(wrapped, (Float64, Float64))
end

# Central difference of a 3-argument function w.r.t. argument k (1, 2, or 3). Local to this file,
# mirroring `central_diff(f, x, y, k)` in `testutils.jl` for one more argument.
function central_diff3(f, x, y, z, k::Int; h=1e-6)
    if k == 1
        return (f(x + h, y, z) - f(x - h, y, z)) / 2h
    elseif k == 2
        return (f(x, y + h, z) - f(x, y - h, z)) / 2h
    else
        return (f(x, y, z + h) - f(x, y, z - h)) / 2h
    end
end

# Generic checks for a ternary scalar function f(x, y, z), mirroring `check_binary`.
function check_ternary(f, xyzs; rtol=1e-6)
    for (x, y, z) in xyzs
        _, gx, gy, gz = rev_gradient(f, x, y, z)
        @test gx ≈ central_diff3(f, x, y, z, 1) rtol = rtol
        @test gy ≈ central_diff3(f, x, y, z, 2) rtol = rtol
        @test gz ≈ central_diff3(f, x, y, z, 3) rtol = rtol
    end
    wrapped(x, y, z) = f(x, y, z)
    checkverify_rev(wrapped, (Float64, Float64, Float64))
end

@testset "exp/log family" begin
    check_unary(exp, (-1.3, 0.2, 2.5))
    check_unary(log, (0.3, 1.7, 4.2))
    check_unary(log1p, (-0.5, 0.3, 2.0))
    check_unary(expm1, (-1.0, 0.2, 1.5))
    check_unary(log2, (0.3, 1.7, 4.2))
    check_unary(log10, (0.3, 1.7, 4.2))
    check_unary(exp2, (-1.3, 0.2, 2.5))
    check_unary(exp10, (-1.3, 0.2, 1.5))
end

@testset "sinh/cosh/tanh" begin
    check_unary(sinh, (-1.3, 0.2, 2.5))
    check_unary(cosh, (-1.3, 0.2, 2.5))
    check_unary(tanh, (-1.3, 0.2, 2.5))
end

@testset "asinh/acosh/atanh" begin
    check_unary(asinh, (-1.3, 0.2, 2.5))
    check_unary(acosh, (1.2, 2.0, 5.0))
    check_unary(atanh, (-0.6, 0.3, 0.7))
end

@testset "asin/acos" begin
    check_unary(asin, (-0.7, 0.3, 0.6))
    check_unary(acos, (-0.7, 0.3, 0.6))
end

@testset "atan" begin
    check_unary(atan, (-2.0, 0.3, 3.0))
    check_binary(atan, ((1.0, 2.0), (-3.0, 1.5), (2.0, -0.5)))
end

@testset "cbrt" begin
    check_unary(cbrt, (-2.0, 0.5, 3.0))
end

@testset "^ (Float64, Float64)" begin
    check_binary(^, ((0.5, 0.3), (1.5, 2.0), (2.0, -1.5)))
end

@testset "hypot" begin
    check_binary(hypot, ((3.0, 4.0), (1.0, 2.0), (-3.0, 4.0)))
end

@testset "sqrt(::Complex)" begin
    zs = (1.0 + 2.0im, 3.0 - 1.0im, 0.5 + 0.5im)

    for z in zs
        # sqrt(::Complex) isn't a scalar loss, so `rev_gradient`'s `one(y)` seeding convention
        # doesn't apply here — call the hand rule directly and cross-check the vjp against the
        # closed-form holomorphic derivative for each of the two real seed directions.
        ycd, pb = rrule!!(zero_fcodual(sqrt), Ctx(), CoDual(z, NoFData()))
        @test primal(ycd) == sqrt(z)
        fprime = 1 / (2 * sqrt(z))
        for (sr, si) in ((1.0, 0.0), (0.0, 1.0))
            _, dzr = pb(RData((re=sr, im=si)))
            expected = conj(fprime) * Complex(sr, si)
            @test dzr.data.re ≈ real(expected)
            @test dzr.data.im ≈ imag(expected)
        end
    end
    wrapped(z) = sqrt(z)
    checkverify_rev(wrapped, (ComplexF64,))
end

@testset "reverse-only: abs/sign/copysign" begin
    check_unary(abs, (-2.3, 0.7, 3.1))
    check_unary(sign, (-2.3, 0.7, 3.1))
    check_binary(copysign, ((3.0, -2.0), (-4.0, 5.0), (2.0, 2.0)))
end

@testset "reverse-only: max/min" begin
    check_binary(max, ((1.0, 2.0), (3.0, -1.0), (-2.0, -5.0)))
    check_binary(min, ((1.0, 2.0), (3.0, -1.0), (-2.0, -5.0)))
end

@testset "reverse-only: fma/muladd" begin
    check_ternary(fma, ((2.0, 3.0, 4.0), (-1.0, 2.0, 0.5), (0.3, -2.0, 5.0)))
    check_ternary(muladd, ((2.0, 3.0, 4.0), (-1.0, 2.0, 0.5), (0.3, -2.0, 5.0)))
end

@testset "reverse-only: abs2/inv" begin
    check_unary(abs2, (-2.0, 0.5, 3.0))
    check_unary(inv, (-2.0, 0.5, 3.0))
end
