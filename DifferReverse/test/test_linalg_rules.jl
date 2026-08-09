using Test
using DifferReverse
using DifferReverse: CoDual, NoRData, Ctx, rrule!!, rev_gradient, zero_fcodual, primal, tangent
using LinearAlgebra

include(joinpath(@__DIR__, "testutils.jl"))

# Throwaway named wrappers, one per rule under test — see DifferForwards/test/test_linalg_rules.jl
# for why they're needed (a direct dualization of `dot`/`norm`/`*` fails since their real bodies
# bottom out in a BLAS `ccall`; a named wrapper's *call* to the target survives as an ordinary
# `:invoke`, which is what exercises the hand rule).
dot_fn(x, y) = dot(x, y)
norm_fn(x) = norm(x)
tr_fn(A) = tr(A)

@testset "rules: linalg (reverse)" begin

    @testset "dot" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, -1.0, 0.5]

        _, dx_dot, dy_dot = rev_gradient(dot_fn, x, y)
        @test dx_dot ≈ y
        @test dy_dot ≈ x
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx_dot[k] ≈ (dot(xp, y) - dot(xm, y)) / 2e-6 rtol = 1e-5
        end
        for k in eachindex(y)
            yp = copy(y); yp[k] += 1e-6
            ym = copy(y); ym[k] -= 1e-6
            @test dy_dot[k] ≈ (dot(x, yp) - dot(x, ym)) / 2e-6 rtol = 1e-5
        end

        checkverify_rev(dot_fn, (Vector{Float64}, Vector{Float64}))
        check_stack_balance(dot_fn, x, y)
    end

    @testset "norm" begin
        x = [3.0, -4.0, 1.0]

        _, dx_norm = rev_gradient(norm_fn, x)
        @test dx_norm ≈ x ./ norm(x)
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx_norm[k] ≈ (norm(xp) - norm(xm)) / 2e-6 rtol = 1e-5
        end

        checkverify_rev(norm_fn, (Vector{Float64},))
        check_stack_balance(norm_fn, x)
    end

    @testset "tr" begin
        A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]

        _, dA_tr = rev_gradient(tr_fn, A)
        @test dA_tr == Matrix(1.0I, 3, 3)
        for i in 1:3, j in 1:3
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dA_tr[i, j] ≈ (tr(Ap) - tr(Am)) / 2e-6 rtol = 1e-5
        end

        checkverify_rev(tr_fn, (Matrix{Float64},))
        check_stack_balance(tr_fn, A)
    end

    @testset "* (matrix-vector)" begin
        # Composite-recursion into a hand rule whose *result* is array-shaped (fdata-carried, not
        # rdata) isn't supported by the general derived-call engine yet (see
        # `_static_recursible_call`'s guard #3 in `reverse_interp.jl` — a pre-existing scope limit
        # of the composite recursion machinery, unrelated to this rule's correctness and not
        # fixable from this file). So `A*x`'s reverse rule is exercised by calling `rrule!!`
        # directly — the same shape `rev_gradient`/`value_and_gradient!` use under the hood —
        # rather than through a wrapping scalar composite function.
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]   # 3x2
        x = [1.5, -0.5]

        dA = zeros(size(A)); dx = zeros(size(x))
        Acd, xcd = CoDual(A, dA), CoDual(x, dx)
        ycd, pb = rrule!!(zero_fcodual(*), Ctx(), Acd, xcd)
        @test primal(ycd) == A * x
        seed = [1.0, 2.0, 3.0]
        tangent(ycd) .= seed
        pb(NoRData())
        @test dA ≈ seed * x'
        @test dx ≈ A' * seed
        # Cross-check `dA`/`dx` against central differences of `dot(seed, A*x)` (a plain, untracked
        # scalar function with the same Jacobian-vector-product structure as the rule).
        floss(A, x) = dot(seed, A * x)
        for i in 1:3, j in 1:2
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dA[i, j] ≈ (floss(Ap, x) - floss(Am, x)) / 2e-6 rtol = 1e-5
        end
        for k in 1:2
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (floss(A, xp) - floss(A, xm)) / 2e-6 rtol = 1e-5
        end
    end

    @testset "* (matrix-matrix)" begin
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]

        # Same reasoning as the matrix-vector case above — direct `rrule!!` call.
        dA = zeros(size(A)); dB = zeros(size(B))
        Acd, Bcd = CoDual(A, dA), CoDual(B, dB)
        Ycd, pb = rrule!!(zero_fcodual(*), Ctx(), Acd, Bcd)
        @test primal(Ycd) == A * B
        seed = [1.0 0.5; -0.5 2.0]
        tangent(Ycd) .= seed
        pb(NoRData())
        @test dA ≈ seed * B'
        @test dB ≈ A' * seed
        floss(A, B) = tr(seed' * (A * B))
        for i in 1:2, j in 1:2
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dA[i, j] ≈ (floss(Ap, B) - floss(Am, B)) / 2e-6 rtol = 1e-5
        end
        for i in 1:2, j in 1:2
            Bp = copy(B); Bp[i, j] += 1e-6
            Bm = copy(B); Bm[i, j] -= 1e-6
            @test dB[i, j] ≈ (floss(A, Bp) - floss(A, Bm)) / 2e-6 rtol = 1e-5
        end
    end

    @testset "mul! (matrix-vector)" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]   # 3x2
        x = [1.5, -0.5]

        # Same reasoning as `*`'s matrix-vector case above — direct `rrule!!` call.
        y = zeros(3)
        dA = zeros(size(A)); dx = zeros(size(x))
        ycd, Acd, xcd = CoDual(y, zeros(3)), CoDual(A, dA), CoDual(x, dx)
        outcd, pb = rrule!!(zero_fcodual(mul!), Ctx(), ycd, Acd, xcd)
        @test primal(outcd) === y
        @test y == A * x
        seed = [1.0, 2.0, 3.0]
        tangent(outcd) .= seed
        pb(NoRData())
        @test dA ≈ seed * x'
        @test dx ≈ A' * seed
        floss(A, x) = dot(seed, A * x)
        for i in 1:3, j in 1:2
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dA[i, j] ≈ (floss(Ap, x) - floss(Am, x)) / 2e-6 rtol = 1e-5
        end
        for k in 1:2
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx[k] ≈ (floss(A, xp) - floss(A, xm)) / 2e-6 rtol = 1e-5
        end

        # Overwrite semantics: stale fdata already on `y` must not leak into `dA`/`dx`, and must be
        # restored after the pullback runs (mirrors `map!`'s `old_ddest` restore, `rules_broadcast.jl`).
        dy2 = [10.0, 20.0, 30.0]
        dA2 = zeros(size(A)); dx2 = zeros(size(x))
        ycd2, Acd2, xcd2 = CoDual(zeros(3), dy2), CoDual(A, dA2), CoDual(x, dx2)
        outcd2, pb2 = rrule!!(zero_fcodual(mul!), Ctx(), ycd2, Acd2, xcd2)
        @test tangent(outcd2) == zeros(3)
        tangent(outcd2) .= seed
        pb2(NoRData())
        @test dA2 ≈ seed * x'
        @test dx2 ≈ A' * seed
        @test dy2 == [10.0, 20.0, 30.0]
    end

    @testset "mul! (matrix-matrix)" begin
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]

        # Direct `rrule!!` call.
        C = zeros(2, 2)
        dA = zeros(size(A)); dB = zeros(size(B))
        Ccd, Acd, Bcd = CoDual(C, zeros(2, 2)), CoDual(A, dA), CoDual(B, dB)
        outcd, pb = rrule!!(zero_fcodual(mul!), Ctx(), Ccd, Acd, Bcd)
        @test primal(outcd) === C
        @test C == A * B
        seed = [1.0 0.5; -0.5 2.0]
        tangent(outcd) .= seed
        pb(NoRData())
        @test dA ≈ seed * B'
        @test dB ≈ A' * seed
        floss(A, B) = tr(seed' * (A * B))
        for i in 1:2, j in 1:2
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dA[i, j] ≈ (floss(Ap, B) - floss(Am, B)) / 2e-6 rtol = 1e-5
        end
        for i in 1:2, j in 1:2
            Bp = copy(B); Bp[i, j] += 1e-6
            Bm = copy(B); Bm[i, j] -= 1e-6
            @test dB[i, j] ≈ (floss(A, Bp) - floss(A, Bm)) / 2e-6 rtol = 1e-5
        end

        # Overwrite semantics: stale fdata already on `C` must not leak into `dA`/`dB`, and must be
        # restored after the pullback runs.
        dC2 = [10.0 20.0; 30.0 40.0]
        dA2 = zeros(size(A)); dB2 = zeros(size(B))
        Ccd2, Acd2, Bcd2 = CoDual(zeros(2, 2), dC2), CoDual(A, dA2), CoDual(B, dB2)
        outcd2, pb2 = rrule!!(zero_fcodual(mul!), Ctx(), Ccd2, Acd2, Bcd2)
        @test tangent(outcd2) == zeros(2, 2)
        tangent(outcd2) .= seed
        pb2(NoRData())
        @test dA2 ≈ seed * B'
        @test dB2 ≈ A' * seed
        @test dC2 == [10.0 20.0; 30.0 40.0]
    end

    @testset "reverse mode over a Transpose/Adjoint (ISSUES #65)" begin
        # `sum(::Transpose)` misses the `sum` hand rules (they require `X<:Array{<:IEEEFloat}`) and
        # falls through to Base's *pairwise* `mapreduce_impl`, which is self-recursive. Direct
        # self-recursion is supported (`reverse_fwds_recursive_ci`, `src/reverse_interp.jl`, ISSUES
        # #65), so this no longer bails on the `in_progress` cycle guard. It used to still bail past
        # that, on `mapreduce_impl`'s non-recursive base case: an `@simd for` loop, which reverse
        # mode had no `Expr(:loopinfo)` support for. Reverse mode now carries `:loopinfo` through,
        # so this composes correctly end to end.
        g_t(M) = sum(transpose(M))

        for n in (2, 40)   # 40x40 is the size that used to reach the illegal statement
            M = rand(n, n)
            _, dM = rev_gradient(g_t, M)
            @test dM == ones(n, n)

            k = (1, min(2, n))   # a single perturbed entry, central-differenced against g_t
            Mp = copy(M); Mp[k...] += 1e-6
            Mm = copy(M); Mm[k...] -= 1e-6
            @test dM[k...] ≈ (g_t(Mp) - g_t(Mm)) / 2e-6 rtol = 1e-5
        end

        checkverify_rev(g_t, (Matrix{Float64},))
        check_stack_balance(g_t, ones(2, 2))
    end

end
