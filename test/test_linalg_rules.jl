using Test
using Differ
using LinearAlgebra

include(joinpath(@__DIR__, "testutils.jl"))

# Throwaway named wrappers, one per rule under test. `checkverify`/`checkverify_rev` reflect on the
# dualized/derived IR for a *call site* to the target function — a call that must survive as a
# genuine `:invoke` for the hand rule dispatch to be exercised at all. Passing the bare
# `LinearAlgebra` function directly asks Differ to dualize/derive *that function's own* body instead
# (the "differentiate this function directly" entry point, unrelated to whether a hand rule exists
# for calls to it elsewhere) — which fails for `dot`/`norm`/`*`, since their real bodies bottom out
# in a BLAS `ccall` Differ can't see through regardless of the hand rule. Wrapping in a trivial
# named function makes the call to the target survive as an ordinary `:invoke`, which is what
# actually exercises the hand rule.
dot_fn(x, y) = dot(x, y)
norm_fn(x) = norm(x)
tr_fn(A) = tr(A)

@testset "rules: linalg" begin

    @testset "dot" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, -1.0, 0.5]

        # Forward mode: cross-check each input entry's directional derivative against central diffs.
        for k in eachindex(x)
            dxseed = zeros(3); dxseed[k] = 1.0
            dd = Differ.frule!!(Differ.Dual(dot_fn, Differ.NoTangent()), Differ.Dual(x, dxseed),
                                 Differ.Dual(y, zeros(3)))
            @test dd.x == dot(x, y)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dd.dx ≈ (dot(xp, y) - dot(xm, y)) / 2e-6 rtol = 1e-6
        end

        # Reverse mode.
        _, dx_dot, dy_dot = Differ.gradient(dot_fn, x, y)
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

        checkverify(dot_fn, (Vector{Float64}, Vector{Float64}))
        checkverify_rev(dot_fn, (Vector{Float64}, Vector{Float64}))
        check_stack_balance(dot_fn, x, y)
    end

    @testset "norm" begin
        x = [3.0, -4.0, 1.0]

        # Forward mode.
        for k in eachindex(x)
            dxseed = zeros(3); dxseed[k] = 1.0
            dd = Differ.frule!!(Differ.Dual(norm_fn, Differ.NoTangent()), Differ.Dual(x, dxseed))
            @test dd.x == norm(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dd.dx ≈ (norm(xp) - norm(xm)) / 2e-6 rtol = 1e-6
        end

        # Reverse mode.
        _, dx_norm = Differ.gradient(norm_fn, x)
        @test dx_norm ≈ x ./ norm(x)
        for k in eachindex(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dx_norm[k] ≈ (norm(xp) - norm(xm)) / 2e-6 rtol = 1e-5
        end

        checkverify(norm_fn, (Vector{Float64},))
        checkverify_rev(norm_fn, (Vector{Float64},))
        check_stack_balance(norm_fn, x)
    end

    @testset "tr" begin
        A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]

        # Forward mode.
        for i in 1:3, j in 1:3
            dAseed = zeros(3, 3); dAseed[i, j] = 1.0
            dd = Differ.frule!!(Differ.Dual(tr_fn, Differ.NoTangent()), Differ.Dual(A, dAseed))
            @test dd.x == tr(A)
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dd.dx ≈ (tr(Ap) - tr(Am)) / 2e-6 rtol = 1e-6
        end

        # Reverse mode.
        _, dA_tr = Differ.gradient(tr_fn, A)
        @test dA_tr == Matrix(1.0I, 3, 3)
        for i in 1:3, j in 1:3
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dA_tr[i, j] ≈ (tr(Ap) - tr(Am)) / 2e-6 rtol = 1e-5
        end

        checkverify(tr_fn, (Matrix{Float64},))
        checkverify_rev(tr_fn, (Matrix{Float64},))
        check_stack_balance(tr_fn, A)
    end

    @testset "* (matrix-vector)" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]   # 3x2
        x = [1.5, -0.5]
        mv(A, x) = A * x

        # Forward mode: cross-check each entry of `A` and `x` against central differences.
        for i in 1:3, j in 1:2
            dAseed = zeros(3, 2); dAseed[i, j] = 1.0
            dd = Differ.frule!!(Differ.Dual(mv, Differ.NoTangent()), Differ.Dual(A, dAseed),
                                 Differ.Dual(x, zeros(2)))
            @test dd.x == A * x
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dd.dx ≈ (Ap * x - Am * x) / 2e-6 atol = 1e-6
        end
        for k in 1:2
            dxseed = zeros(2); dxseed[k] = 1.0
            dd = Differ.frule!!(Differ.Dual(mv, Differ.NoTangent()), Differ.Dual(A, zeros(3, 2)),
                                 Differ.Dual(x, dxseed))
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dd.dx ≈ (A * xp - A * xm) / 2e-6 atol = 1e-6
        end
        checkverify(mv, (Matrix{Float64}, Vector{Float64}))

        # Reverse mode: composite-recursion into a hand rule whose *result* is array-shaped
        # (fdata-carried, not rdata) isn't supported by the general derived-call engine yet (see
        # `_static_recursible_call`'s guard #3 in `reverse_interp.jl` — unrelated to this rule being
        # correct, a pre-existing scope limit of the composite recursion machinery, not something
        # fixable from this file). So `A*x`'s reverse rule is exercised by calling `rrule!!` directly
        # — exactly the shape `Differ.gradient`/`value_and_gradient!` themselves use under the hood —
        # rather than through a wrapping scalar composite function.
        dA = zeros(size(A)); dx = zeros(size(x))
        Acd, xcd = Differ.CoDual(A, dA), Differ.CoDual(x, dx)
        ycd, pb = Differ.rrule!!(Differ.zero_fcodual(*), Differ.Ctx(), Acd, xcd)
        @test Differ.primal(ycd) == A * x
        seed = [1.0, 2.0, 3.0]
        Differ.tangent(ycd) .= seed
        pb(Differ.NoRData())
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
        mm(A, B) = A * B

        # Forward mode.
        for i in 1:2, j in 1:2
            dAseed = zeros(2, 2); dAseed[i, j] = 1.0
            dd = Differ.frule!!(Differ.Dual(mm, Differ.NoTangent()), Differ.Dual(A, dAseed),
                                 Differ.Dual(B, zeros(2, 2)))
            @test dd.x == A * B
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dd.dx ≈ (Ap * B - Am * B) / 2e-6 atol = 1e-6
        end
        checkverify(mm, (Matrix{Float64}, Matrix{Float64}))

        # Reverse mode: same reasoning as the matrix-vector case above — direct `rrule!!` call.
        dA = zeros(size(A)); dB = zeros(size(B))
        Acd, Bcd = Differ.CoDual(A, dA), Differ.CoDual(B, dB)
        Ycd, pb = Differ.rrule!!(Differ.zero_fcodual(*), Differ.Ctx(), Acd, Bcd)
        @test Differ.primal(Ycd) == A * B
        seed = [1.0 0.5; -0.5 2.0]
        Differ.tangent(Ycd) .= seed
        pb(Differ.NoRData())
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

    @testset "transpose/adjoint already work with NO new rule" begin
        # Regression tests documenting that `transpose`/`adjoint` are deliberately *not* given
        # rules: their tangent already routes through the generic struct-tangent machinery
        # (`tangent_type(Transpose{Float64,Matrix{Float64}})` is a real `Tangent`), both in forward
        # and reverse mode.
        M = [1.0 2.0; 3.0 4.0]

        g_t(M) = sum(transpose(M))
        g_a(M) = sum(adjoint(M))

        @test sum(transpose(M)) == sum(M)
        @test sum(adjoint(M)) == sum(M)

        _, dM_t = Differ.gradient(g_t, M)
        @test dM_t == ones(2, 2)
        _, dM_a = Differ.gradient(g_a, M)
        @test dM_a == ones(2, 2)

        for i in 1:2, j in 1:2
            Mp = copy(M); Mp[i, j] += 1e-6
            Mm = copy(M); Mm[i, j] -= 1e-6
            @test dM_t[i, j] ≈ (g_t(Mp) - g_t(Mm)) / 2e-6 rtol = 1e-5
            @test dM_a[i, j] ≈ (g_a(Mp) - g_a(Mm)) / 2e-6 rtol = 1e-5
        end

        # Forward mode too.
        dMseed = zeros(2, 2); dMseed[1, 2] = 1.0
        dd_t = Differ.frule!!(Differ.Dual(g_t, Differ.NoTangent()), Differ.Dual(M, dMseed))
        @test dd_t.dx ≈ 1.0

        checkverify(g_t, (Matrix{Float64},))
        checkverify(g_a, (Matrix{Float64},))
        checkverify_rev(g_t, (Matrix{Float64},))
        checkverify_rev(g_a, (Matrix{Float64},))
        check_stack_balance(g_t, M)
        check_stack_balance(g_a, M)
    end

end
