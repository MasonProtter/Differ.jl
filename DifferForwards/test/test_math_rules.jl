using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, frule!!, build_tangent

include(joinpath(@__DIR__, "testutils.jl"))

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
    # atan(y, x), 2-arg, one point per quadrant of (y, x)
    check_binary(atan, ((1.0, 2.0), (-3.0, 1.5), (2.0, -0.5), (-2.0, -1.5)))
end

@testset "cbrt" begin
    check_unary(cbrt, (-2.0, 0.5, 3.0))
end

@testset "^ (Float64, Float64)" begin
    check_binary(^, ((0.5, 0.3), (1.5, 2.0), (2.0, -1.5)))
end

@testset "^ (non-literal Int exponent — regression)" begin
    # The reported bug: `x^n` with `n::Int` gives the exponent's `Dual` a `NoTangent` tangent, and
    # the old rule unconditionally computed `yp*log(x)*dyv`, which both errors on `Float64*NoTangent`
    # and (independently) evaluates `log(x)` eagerly even when its term is structurally zero.
    f4(x) = x^4
    d = frule!!(Dual(f4, NoTangent()), Dual(2.0, 1.0))
    @test d.x == 2.0^4
    @test d.dx ≈ 4*2.0^3

    # Negative base with an odd Int exponent: must not throw DomainError (old rule would evaluate
    # log(negative) even though its coefficient is multiplied by a structural zero).
    f3(x) = x^3
    dneg = frule!!(Dual(f3, NoTangent()), Dual(-2.0, 1.0))
    @test dneg.x == (-2.0)^3
    @test dneg.dx ≈ 3*(-2.0)^2

    # IR-legality for a wrapper with a genuine Int argument (x differentiable, exponent not).
    wrapped(x, n) = x^n
    checkverify(wrapped, (Float64, Int))

    # Both base and exponent non-differentiable (Int^Int): tangent_type(Int) === NoTangent, so the
    # result must be `Dual{Int,NoTangent}`, never a `Dual{Int,Int}` — see the reachability
    # investigation in the PR description: the dualizer's `frule_split!` routes any surviving `^`
    # call through `frule!!` regardless of whether its operands are differentiable, so this case is
    # reachable (e.g. `f(x::Float64, n::Int, m::Int) = x + Float64(n^m)`), not merely speculative.
    rboth = frule!!(Dual(^, NoTangent()), Dual(3, NoTangent()), Dual(4, NoTangent()))
    @test rboth.x == 3^4
    @test rboth.dx === NoTangent()
    @test rboth isa Dual{Int,NoTangent}

    # Differentiating w.r.t. the exponent alone (both Float64, base held fixed).
    fexp(y) = 2.0^y
    check_unary(fexp, (0.5, 2.0, -1.0))
end

@testset "hypot" begin
    check_binary(hypot, ((3.0, 4.0), (1.0, 2.0), (-3.0, 4.0)))
end

@testset "atan/hypot with a non-differentiable (Int) operand — regression" begin
    # `atan(y, x)` and `hypot(x, y)` promote internally when called directly, but Differ's own
    # `src_inlining_policy` blocks inlining of any call whose callee has a hand-written `frule!!` —
    # so a composite caller like `f(n::Int, x::Float64) = atan(n, x)` reaches `frule!!` with the
    # *unpromoted* `Dual{Int,NoTangent}`/`Dual{Float64,Float64}` pair, not two same-typed Duals.
    # Confirmed via `code_dual_ircode` that this survives as a direct call, not something the
    # ordinary optimizer would ever produce. The old unguarded rules threw `MethodError(*, ...)`
    # on exactly this input.
    fatan(n, x) = atan(n, x)
    fhypot(n, x) = hypot(n, x)

    datan = frule!!(Dual(fatan, NoTangent()), Dual(3, NoTangent()), Dual(2.0, 1.0))
    @test datan.x == atan(3, 2.0)
    @test datan.dx ≈ central_diff(x -> atan(3, x), 2.0)

    dhypot = frule!!(Dual(fhypot, NoTangent()), Dual(3, NoTangent()), Dual(2.0, 1.0))
    @test dhypot.x == hypot(3, 2.0)
    @test dhypot.dx ≈ central_diff(x -> hypot(3, x), 2.0)

    checkverify(fatan, (Int, Float64))
    checkverify(fhypot, (Int, Float64))
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
    end
    # See the comment in `check_unary` for why `sqrt` is wrapped before verifying.
    wrapped(z) = sqrt(z)
    checkverify(wrapped, (ComplexF64,))
end

@testset "reverse-only rules: forward-mode composite fallback still differentiates them" begin
    # abs/sign/copysign/max/min/fma/muladd/abs2/inv have no hand-written `frule!!` — forward mode
    # already dualizes them correctly via the generic composite fallback dispatching on the
    # underlying LLVM intrinsic (`src/intrinsics.jl`), so this is just confirming that path works
    # for exactly the functions that need an explicit hand *reverse* rule
    # (DifferReverse/test/test_math_rules.jl) because they inline straight to the intrinsic before
    # a call-level `rrule!!` gets a chance to fire.
    check_unary(abs, (-2.3, 0.7, 3.1))
    check_unary(sign, (-2.3, 0.7, 3.1))
    check_binary(copysign, ((3.0, -2.0), (-4.0, 5.0), (2.0, 2.0)))
    check_binary(max, ((1.0, 2.0), (3.0, -1.0), (-2.0, -5.0)))
    check_binary(min, ((1.0, 2.0), (3.0, -1.0), (-2.0, -5.0)))
    check_ternary(fma, ((2.0, 3.0, 4.0), (-1.0, 2.0, 0.5), (0.3, -2.0, 5.0)))
    check_ternary(muladd, ((2.0, 3.0, 4.0), (-1.0, 2.0, 0.5), (0.3, -2.0, 5.0)))
    check_unary(abs2, (-2.0, 0.5, 3.0))
    check_unary(inv, (-2.0, 0.5, 3.0))
end
