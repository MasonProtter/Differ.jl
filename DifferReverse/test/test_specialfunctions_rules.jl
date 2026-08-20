using Test
using DifferReverse
using DifferReverse: rev_gradient, CoDual, NoFData, NoRData, Ctx, rrule!!, zero_fcodual, primal
using SpecialFunctions

include(joinpath(@__DIR__, "testutils.jl"))

@testset "airy" begin
    check_unary(airyai, (-1.3, 0.4, 2.0))
    check_unary(airyaiprime, (-1.3, 0.4, 2.0))
    check_unary(airybi, (-1.3, 0.4, 2.0))
    check_unary(airybiprime, (-1.3, 0.4, 2.0))
    # The scaled Airy functions are defined for non-negative arguments only.
    check_unary(airyaix, (0.4, 2.0, 5.0))
    check_unary(airyaiprimex, (0.4, 2.0, 5.0))
end

@testset "bessel, fixed order" begin
    check_unary(besselj0, (-1.3, 0.4, 2.0))
    check_unary(besselj1, (-1.3, 0.4, 2.0))
    check_unary(bessely0, (0.4, 1.3, 4.0))
    check_unary(bessely1, (0.4, 1.3, 4.0))
end

@testset "bessel, general order" begin
    check_order(besselj, 2, (0.4, 1.3, 4.0))
    check_order(besseli, 2, (0.4, 1.3, 4.0))
    check_order(bessely, 2, (0.4, 1.3, 4.0))
    check_order(besselk, 2, (0.4, 1.3, 4.0))
    check_order(besselkx, 2, (0.4, 1.3, 4.0))
    # The scaled forms' extra term is the scaling factor's own derivative — sign-dependent for
    # `besselix`, so both signs are covered.
    check_order(besselix, 2, (-1.3, 0.4, 4.0))
    check_order(besseljx, 2, (-1.3, 0.4, 4.0))
    check_order(besselyx, 2, (0.4, 1.3, 4.0))
end

@testset "dawson" begin
    check_unary(dawson, (-1.3, 0.4, 2.0))
end

@testset "gamma and friends" begin
    check_unary(gamma, (0.4, 1.3, 4.0))
    check_unary(loggamma, (0.4, 1.3, 4.0))
    check_unary(digamma, (0.4, 1.3, 4.0))
    check_unary(trigamma, (0.4, 1.3, 4.0))
    check_unary(invdigamma, (-1.0, 0.5, 2.0))
    check_order(polygamma, 2, (0.4, 1.3, 4.0))
    check_binary(beta, ((1.5, 2.5), (0.7, 3.0)))
    check_binary(logbeta, ((1.5, 2.5), (0.7, 3.0)))
    check_param(gamma, 1.5, (0.4, 1.3, 4.0))
    check_param(loggamma, 1.5, (0.4, 1.3, 4.0))
end

@testset "error functions" begin
    check_unary(erf, (-1.3, 0.4, 2.0))
    check_binary(erf, ((0.3, 1.2), (-1.0, 0.5)))
    check_unary(erfc, (-1.3, 0.4, 2.0))
    check_unary(logerfc, (-1.0, 0.5, 3.0))
    check_unary(erfcx, (-1.0, 0.5, 3.0))
    check_unary(logerfcx, (-1.0, 0.5, 3.0))
    check_unary(erfi, (-1.3, 0.4, 1.5))
    check_unary(erfinv, (-0.6, 0.3, 0.8))
    check_unary(erfcinv, (0.3, 1.0, 1.7))
end

@testset "exponential, sine and cosine integrals" begin
    check_unary(expint, (0.4, 1.3, 4.0))
    check_unary(expintx, (0.4, 1.3, 4.0))
    check_unary(expinti, (0.4, 1.3, 4.0))
    check_unary(sinint, (-1.3, 0.4, 2.0))
    check_unary(cosint, (0.4, 1.3, 4.0))
    check_order(expint, 2, (0.4, 1.3, 4.0))
    check_order(expintx, 2, (0.4, 1.3, 4.0))
end

@testset "elliptic integrals" begin
    # `m == 0` is its own branch in both derivatives.
    check_unary(ellipk, (-0.5, 0.0, 0.2, 0.7))
    check_unary(ellipe, (-0.5, 0.0, 0.2, 0.7))
end

@testset "a constant operand contributes nothing" begin
    fcd = zero_fcodual(beta)
    a, b = 1.5, 2.5
    dbda = beta(a, b)*(digamma(a) - digamma(a + b))
    both = rrule!!(fcd, Ctx(), CoDual(a, NoFData()), CoDual(b, NoFData()))[2](1.0)
    @test both[2] ≈ dbda
    bcon = rrule!!(fcd, Ctx(), CoDual(a, NoFData()), const_codual(b))[2](1.0)
    @test bcon == (NoRData(), both[2], NoRData())
end

@testset "tuple-valued gamma functions" begin
    for x in (-2.5, 0.4, 1.3, 4.0)
        y, pb = rrule!!(zero_fcodual(logabsgamma), Ctx(), CoDual(x, NoFData()))
        @test primal(y) == logabsgamma(x)
        # The rdata seed pairs the log's own seed with the `Int` sign's empty one.
        @test pb((1.0, NoRData()))[2] ≈ central_diff(t -> logabsgamma(t)[1], x) rtol = 1e-6
    end
    for x in (0.4, 1.3, 4.0)
        y, pb = rrule!!(zero_fcodual(gamma_inc), Ctx(), const_codual(1.5), CoDual(x, NoFData()),
                        CoDual(0, NoFData()))
        @test primal(y) == gamma_inc(1.5, x, 0)
        @test pb((1.0, 0.0))[3] ≈ central_diff(t -> gamma_inc(1.5, t, 0)[1], x) rtol = 1e-6
        @test pb((0.0, 1.0))[3] ≈ central_diff(t -> gamma_inc(1.5, t, 0)[2], x) rtol = 1e-6
    end
    # Through the transform, where the tuple is taken apart afterwards.
    lg(x) = logabsgamma(x)[1]
    gi(x) = gamma_inc(1.5, x)[1]
    for x in (0.4, 1.3, 4.0)
        @test rev_gradient(lg, x)[2] ≈ digamma(x) rtol = 1e-12
        @test rev_gradient(gi, x)[2] ≈ central_diff(gi, x) rtol = 1e-6
    end
    checkverify_rev(lg, (Float64,))
    checkverify_rev(gi, (Float64,))
end

@testset "a parameter with no derivative is refused, not poisoned" begin
    # A literal order reaches the rule as `CoDual{Float64,Inactive}`, whose slot is `NoRData` — the
    # common call is unaffected. A live shadow in that slot means the caller really is asking for
    # the order's derivative, so the rule says so at the forwards call.
    f(x) = besselj(1.5, x)
    _, gx = rev_gradient(f, 1.3)
    @test gx ≈ central_diff(f, 1.3) rtol = 1e-6

    # The literal-parameter closure form, spelled out.
    g = rev_gradient(x -> besselj(1.5, x), 1.3)
    @test length(g) == 2
    @test g[2] ≈ central_diff(f, 1.3) rtol = 1e-6

    @test_throws ErrorException rev_gradient(besselj, 1.5, 1.3)

    # Declared constant, or an integer, the slot carries no rdata at all.
    @test rrule!!(zero_fcodual(besselj), Ctx(), const_codual(1.5), CoDual(1.3, NoFData()))[2](1.0) ==
        (NoRData(), NoRData(), gx)
    @test rrule!!(zero_fcodual(besselj), Ctx(), CoDual(2, NoFData()),
                  CoDual(1.3, NoFData()))[2](1.0)[2] === NoRData()
end

@testset "through the transform" begin
    # The rules reached the way real code reaches them: a composite the transform has to route
    # call by call, rather than a direct `rrule!!` on the primitive.
    f(x) = erf(x)*gamma(x) + besselj(2, x)
    for x in (0.4, 1.3, 2.5)
        _, gx = rev_gradient(f, x)
        @test gx ≈ central_diff(f, x) rtol = 1e-6
    end
    checkverify_rev(f, (Float64,))
end
