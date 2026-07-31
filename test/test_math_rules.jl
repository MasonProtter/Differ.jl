using Test
using Differ
using Differ: Dual, NoTangent, frule!!, gradient, build_tangent
using Differ: CoDual, NoFData, RData, Ctx, rrule!!, zero_fcodual, primal, tangent

include(joinpath(@__DIR__, "testutils.jl"))

# Generic checks for a unary scalar function: forward tangent and reverse gradient both against
# central differences, at every x in xs, plus IR-legality checks for both modes.
function check_unary(f, xs; rtol=1e-6)
    for x in xs
        d = frule!!(Dual(f, NoTangent()), Dual(x, 1.0))
        @test d.x ≈ f(x)
        @test d.dx ≈ central_diff(f, x) rtol = rtol

        _, gx = Differ.gradient(f, x)
        @test gx ≈ central_diff(f, x) rtol = rtol
    end
    # `checkverify`/`checkverify_rev` dualize the trivial wrapper rather than `f` directly: passing
    # `f` itself as the top-level dualization target hits an unrelated pre-existing quirk where
    # Julia's own inliner unfolds `f`'s real Base body into the generic `dualized_impl` wrapper
    # *before* Differ's call-site hand-rule interception ever gets a look-in (confirmed harmless to
    # this task by checking that a composite caller of `f` — the realistic scenario — gets a single
    # clean `invoke` to our hand rule instead). Wrapping one level deep, as any real caller of `f`
    # would look, sidesteps that and exercises the interception path this task actually cares about.
    wrapped(x) = f(x)
    checkverify(wrapped, (Float64,))
    checkverify_rev(wrapped, (Float64,))
end

# Generic checks for a binary scalar function f(x, y): forward tangent (both argument directions)
# and reverse gradient, both against central differences, plus IR-legality checks.
function check_binary(f, xys; rtol=1e-6)
    for (x, y) in xys
        dx = frule!!(Dual(f, NoTangent()), Dual(x, 1.0), Dual(y, 0.0))
        @test dx.x ≈ f(x, y)
        @test dx.dx ≈ central_diff(f, x, y, 1) rtol = rtol
        dy = frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 1.0))
        @test dy.dx ≈ central_diff(f, x, y, 2) rtol = rtol

        _, gx, gy = Differ.gradient(f, x, y)
        @test gx ≈ central_diff(f, x, y, 1) rtol = rtol
        @test gy ≈ central_diff(f, x, y, 2) rtol = rtol
    end
    # See the comment in `check_unary` for why `f` is wrapped before verifying.
    wrapped(x, y) = f(x, y)
    checkverify(wrapped, (Float64, Float64))
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
        dx = frule!!(Dual(f, NoTangent()), Dual(x, 1.0), Dual(y, 0.0), Dual(z, 0.0))
        @test dx.x ≈ f(x, y, z)
        @test dx.dx ≈ central_diff3(f, x, y, z, 1) rtol = rtol
        dy = frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 1.0), Dual(z, 0.0))
        @test dy.dx ≈ central_diff3(f, x, y, z, 2) rtol = rtol
        dz = frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 0.0), Dual(z, 1.0))
        @test dz.dx ≈ central_diff3(f, x, y, z, 3) rtol = rtol

        _, gx, gy, gz = Differ.gradient(f, x, y, z)
        @test gx ≈ central_diff3(f, x, y, z, 1) rtol = rtol
        @test gy ≈ central_diff3(f, x, y, z, 2) rtol = rtol
        @test gz ≈ central_diff3(f, x, y, z, 3) rtol = rtol
    end
    # See the comment in `check_unary` for why `f` is wrapped before verifying.
    wrapped(x, y, z) = f(x, y, z)
    checkverify(wrapped, (Float64, Float64, Float64))
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
    # |x| >= 0.5 and |x| < 0.5: precisely the branch where Base's own atanh kernel does a
    # bitcast/and_int precision trick that would silently zero the tangent via the generic
    # fallback if this rule weren't hand-written with the closed-form derivative.
    check_unary(atanh, (-0.6, 0.3, 0.7))
end

@testset "asin/acos" begin
    # Same deliberate regression check as atanh above: mix points below and above |x| = 0.5.
    check_unary(asin, (-0.7, 0.3, 0.6))
    check_unary(acos, (-0.7, 0.3, 0.6))
end

@testset "atan" begin
    check_unary(atan, (-2.0, 0.3, 3.0))
    # atan(y, x), 2-arg
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
    ct(c) = build_tangent(ComplexF64, real(c), imag(c))
    zs = (1.0 + 2.0im, 3.0 - 1.0im, 0.5 + 0.5im)
    dirs = (1.0 + 0.0im, 0.0 + 1.0im, 0.3 - 0.2im)

    for z in zs
        for dz in dirs
            d = frule!!(Dual(sqrt, NoTangent()), Dual(z, ct(dz)))
            @test d.x == sqrt(z)
            expected = dz / (2 * sqrt(z))
            @test d.dx == ct(expected)
        end

        # Reverse: call the hand rule directly (sqrt(::Complex) isn't a scalar loss, so
        # `Differ.gradient`'s `one(y)` seeding convention doesn't apply here) and cross-check the
        # vjp against the forward-mode Jacobian for each of the two real seed directions.
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
    # See the comment in `check_unary` for why `sqrt` is wrapped before verifying.
    wrapped(z) = sqrt(z)
    checkverify(wrapped, (ComplexF64,))
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
