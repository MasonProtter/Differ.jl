using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, Inactive, frule!!
using SpecialFunctions

include(joinpath(@__DIR__, "testutils.jl"))

# `f(ν, x)` with an integer order: `ν` has no tangent space at all, so only the `x` tangent is
# checked, and the shadow the transform would hand the rule for `ν` is `NoTangent()`.
function check_order(f, ν::Integer, xs; rtol=1e-6)
    for x in xs
        d = frule!!(Dual(f, NoTangent()), Dual(ν, NoTangent()), Dual(x, 1.0))
        @test d.x ≈ f(ν, x)
        @test d.dx ≈ central_diff(t -> f(ν, t), x) rtol = rtol
    end
    # See the comment in testutils.jl's `check_unary` for why `f` is wrapped before verifying.
    wrapped(ν, x) = f(ν, x)
    checkverify(wrapped, (typeof(ν), Float64))
end

# `f(a, x)` with a real parameter whose derivative is not implemented: `a` is held constant.
function check_param(f, a::Real, xs; rtol=1e-6)
    for x in xs
        d = frule!!(Dual(f, NoTangent()), const_dual(a), Dual(x, 1.0))
        @test d.x ≈ f(a, x)
        @test d.dx ≈ central_diff(t -> f(a, t), x) rtol = rtol
    end
    wrapped(a, x) = f(a, x)
    checkverify(wrapped, (typeof(a), Float64); inactive=(1,))
end

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

@testset "tuple-valued gamma functions" begin
    for x in (-2.5, 0.4, 1.3, 4.0)
        d = frule!!(Dual(logabsgamma, NoTangent()), Dual(x, 1.0))
        @test d.x == logabsgamma(x)
        @test d.dx[1] ≈ central_diff(t -> logabsgamma(t)[1], x) rtol = 1e-6
        # The sign half is an `Int`: no tangent space, so no tangent.
        @test d.dx[2] === NoTangent()
    end
    for x in (0.4, 1.3, 4.0)
        d = frule!!(Dual(gamma_inc, NoTangent()), const_dual(1.5), Dual(x, 1.0),
                    Dual(0, NoTangent()))
        @test d.x == gamma_inc(1.5, x, 0)
        @test d.dx[1] ≈ central_diff(t -> gamma_inc(1.5, t, 0)[1], x) rtol = 1e-6
        @test d.dx[2] ≈ central_diff(t -> gamma_inc(1.5, t, 0)[2], x) rtol = 1e-6
    end
    # Through the dualizer, where the tuple is taken apart afterwards.
    lg(x) = logabsgamma(x)[1]
    gi(x) = gamma_inc(1.5, x)[1]
    for x in (0.4, 1.3, 4.0)
        @test frule!!(Dual(lg, NoTangent()), Dual(x, 1.0)).dx ≈ digamma(x) rtol = 1e-12
        @test frule!!(Dual(gi, NoTangent()), Dual(x, 1.0)).dx ≈ central_diff(gi, x) rtol = 1e-6
    end
    checkverify(lg, (Float64,))
    checkverify(gi, (Float64,))
end

@testset "a nonzero order or parameter tangent is refused" begin
    # No closed form is implemented for these directions. A zero tangent is fine — the missing term
    # would be multiplied by zero — but anything else would drop a real contribution.
    @test_throws ErrorException frule!!(Dual(besselj, NoTangent()), Dual(1.5, 1.0), Dual(1.3, 1.0))
    @test_throws ErrorException frule!!(Dual(gamma, NoTangent()), Dual(1.5, 1.0), Dual(1.3, 1.0))
    @test frule!!(Dual(besselj, NoTangent()), Dual(1.5, 0.0), Dual(1.3, 1.0)).x == besselj(1.5, 1.3)
    @test frule!!(Dual(besselj, NoTangent()), const_dual(1.5), Dual(1.3, 1.0)).x == besselj(1.5, 1.3)
end

@testset "through the dualizer" begin
    # The rules reached the way real code reaches them: a composite the transform has to route
    # call by call, rather than a direct `frule!!` on the primitive.
    f(x) = erf(x)*gamma(x) + besselj(2, x)
    for x in (0.4, 1.3, 2.5)
        d = frule!!(Dual(f, NoTangent()), Dual(x, 1.0))
        @test d.x ≈ f(x)
        @test d.dx ≈ central_diff(f, x) rtol = 1e-6
    end
    checkverify(f, (Float64,))

    # A literal parameter has a tangent space but no fdata, so the transform mints it `Inactive()`
    # rather than an active zero — `_no_param_derivative`'s `_inert` arm is what accepts it. (Its
    # `iszero` arm still matters, for a direction the caller explicitly seeds with zero.)
    g(x) = besselj(1.5, x) + gamma(2.5, x)
    for x in (0.4, 1.3, 2.5)
        @test frule!!(Dual(g, NoTangent()), Dual(x, 1.0)).dx ≈ central_diff(g, x) rtol = 1e-6
    end

    # The literal really does arrive as `Dual{Float64,Inactive}` — built as one, and named as one in
    # the rule invoke's own signature.
    bj(x) = besselj(1.5, x)
    ir, _ = code_dual_ircode(bj, (Float64,))
    # Types print module-qualified inside a `SafeTestset`, so match around the qualifiers.
    inactive_dual = r"(\w+\.)?Dual\{Float64, (\w+\.)?Inactive\}"
    stmts = [sprint(show, ir.stmts[i][:stmt]) for i in 1:length(ir.stmts)]
    @test any(s -> occursin(Regex("%new\\(" * inactive_dual.pattern * ", 1\\.5, "), s), stmts)
    @test any(s -> occursin("frule!!", s) && occursin(inactive_dual, s), stmts)
    @test !any(s -> occursin("zero_tangent", s), stmts)
end
