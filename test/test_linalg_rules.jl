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

    @testset "mul! (matrix-vector)" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]   # 3x2
        x = [1.5, -0.5]
        mulv(A, x) = mul!(zeros(3), A, x)

        # Forward mode: cross-check each entry of `A` and `x` against central differences.
        for i in 1:3, j in 1:2
            y = zeros(3)
            dAseed = zeros(3, 2); dAseed[i, j] = 1.0
            dd = Differ.frule!!(Differ.Dual(mul!, Differ.NoTangent()), Differ.Dual(y, zeros(3)),
                                 Differ.Dual(A, dAseed), Differ.Dual(x, zeros(2)))
            @test Differ.primal(dd) === y  # mul! returns (and mutates) y in place
            @test y == A * x
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test Differ.tangent(dd) ≈ (Ap * x - Am * x) / 2e-6 atol = 1e-6
        end
        for k in 1:2
            dxseed = zeros(2); dxseed[k] = 1.0
            dd = Differ.frule!!(Differ.Dual(mul!, Differ.NoTangent()), Differ.Dual(zeros(3), zeros(3)),
                                 Differ.Dual(A, zeros(3, 2)), Differ.Dual(x, dxseed))
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test Differ.tangent(dd) ≈ (A * xp - A * xm) / 2e-6 atol = 1e-6
        end
        checkverify(mulv, (Matrix{Float64}, Vector{Float64}))

        # Reverse mode: same reasoning as `*`'s matrix-vector case above — direct `rrule!!` call.
        y = zeros(3)
        dA = zeros(size(A)); dx = zeros(size(x))
        ycd, Acd, xcd = Differ.CoDual(y, zeros(3)), Differ.CoDual(A, dA), Differ.CoDual(x, dx)
        outcd, pb = Differ.rrule!!(Differ.zero_fcodual(mul!), Differ.Ctx(), ycd, Acd, xcd)
        @test Differ.primal(outcd) === y
        @test y == A * x
        seed = [1.0, 2.0, 3.0]
        Differ.tangent(outcd) .= seed
        pb(Differ.NoRData())
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
        ycd2, Acd2, xcd2 = Differ.CoDual(zeros(3), dy2), Differ.CoDual(A, dA2), Differ.CoDual(x, dx2)
        outcd2, pb2 = Differ.rrule!!(Differ.zero_fcodual(mul!), Differ.Ctx(), ycd2, Acd2, xcd2)
        @test Differ.tangent(outcd2) == zeros(3)
        Differ.tangent(outcd2) .= seed
        pb2(Differ.NoRData())
        @test dA2 ≈ seed * x'
        @test dx2 ≈ A' * seed
        @test dy2 == [10.0, 20.0, 30.0]
    end

    @testset "mul! (matrix-matrix)" begin
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]
        mulm(A, B) = mul!(zeros(2, 2), A, B)

        # Forward mode.
        for i in 1:2, j in 1:2
            C = zeros(2, 2)
            dAseed = zeros(2, 2); dAseed[i, j] = 1.0
            dd = Differ.frule!!(Differ.Dual(mul!, Differ.NoTangent()), Differ.Dual(C, zeros(2, 2)),
                                 Differ.Dual(A, dAseed), Differ.Dual(B, zeros(2, 2)))
            @test Differ.primal(dd) === C
            @test C == A * B
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test Differ.tangent(dd) ≈ (Ap * B - Am * B) / 2e-6 atol = 1e-6
        end
        checkverify(mulm, (Matrix{Float64}, Matrix{Float64}))

        # Reverse mode: direct `rrule!!` call.
        C = zeros(2, 2)
        dA = zeros(size(A)); dB = zeros(size(B))
        Ccd, Acd, Bcd = Differ.CoDual(C, zeros(2, 2)), Differ.CoDual(A, dA), Differ.CoDual(B, dB)
        outcd, pb = Differ.rrule!!(Differ.zero_fcodual(mul!), Differ.Ctx(), Ccd, Acd, Bcd)
        @test Differ.primal(outcd) === C
        @test C == A * B
        seed = [1.0 0.5; -0.5 2.0]
        Differ.tangent(outcd) .= seed
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

        # Overwrite semantics: stale fdata already on `C` must not leak into `dA`/`dB`, and must be
        # restored after the pullback runs.
        dC2 = [10.0 20.0; 30.0 40.0]
        dA2 = zeros(size(A)); dB2 = zeros(size(B))
        Ccd2, Acd2, Bcd2 = Differ.CoDual(zeros(2, 2), dC2), Differ.CoDual(A, dA2), Differ.CoDual(B, dB2)
        outcd2, pb2 = Differ.rrule!!(Differ.zero_fcodual(mul!), Differ.Ctx(), Ccd2, Acd2, Bcd2)
        @test Differ.tangent(outcd2) == zeros(2, 2)
        Differ.tangent(outcd2) .= seed
        pb2(Differ.NoRData())
        @test dA2 ≈ seed * B'
        @test dB2 ≈ A' * seed
        @test dC2 == [10.0 20.0; 30.0 40.0]
    end

    @testset "transpose/adjoint work in forward mode with NO new rule" begin
        # `transpose`/`adjoint` are deliberately *not* given rules: their tangent already routes
        # through the generic struct-tangent machinery (`tangent_type(Transpose{Float64,
        # Matrix{Float64}})` is a real `Tangent`).
        M = [1.0 2.0; 3.0 4.0]

        g_t(M) = sum(transpose(M))
        g_a(M) = sum(adjoint(M))

        @test sum(transpose(M)) == sum(M)
        @test sum(adjoint(M)) == sum(M)

        dMseed = zeros(2, 2); dMseed[1, 2] = 1.0
        dd_t = Differ.frule!!(Differ.Dual(g_t, Differ.NoTangent()), Differ.Dual(M, dMseed))
        @test dd_t.dx ≈ 1.0
        dd_a = Differ.frule!!(Differ.Dual(g_a, Differ.NoTangent()), Differ.Dual(M, dMseed))
        @test dd_a.dx ≈ 1.0

        checkverify(g_t, (Matrix{Float64},))
        checkverify(g_a, (Matrix{Float64},))
    end

    @testset "reverse mode over a Transpose/Adjoint (ISSUES #65)" begin
        # `sum(::Transpose)` misses the `sum` hand rules (they require `X<:Array{<:IEEEFloat}`) and
        # falls through to Base's *pairwise* `mapreduce_impl`, which is self-recursive. Direct
        # self-recursion is supported (`reverse_fwds_recursive_ci`, `src/reverse_interp.jl`, ISSUES
        # #65), so this no longer bails on the `in_progress` cycle guard. It used to still bail past
        # that, on `mapreduce_impl`'s non-recursive base case: an `@simd for` loop, which reverse mode
        # had no `Expr(:loopinfo)` support for. Reverse mode now carries `:loopinfo` through (mirroring
        # forward mode, dropping only `julia.ivdep` — see the comment on the fwds carrier's `:loopinfo`
        # arm, `src/reverse_interp.jl`), so this composes correctly end to end.
        #
        # This testset used to assert only a graceful bail, and before that (briefly) unsound working
        # gradients: `mapreduce_impl`'s `op` operand is a `GlobalRef`, which `_static_recursible_call`
        # used to mistype as `GlobalRef` (the node's type, not the value's — the reverse-mode half of
        # ISSUES #63), resolving a *different*, non-self-recursive specialization and emitting
        # `%new(CoDual{GlobalRef,NoFData}, Base.add_sum, …)`. At 2x2 that statement sits on a branch
        # below `pairwise_blocksize` and never runs; at 40x40 it does, and the same call died with
        # `TypeError: in new, expected GlobalRef, got a value of type typeof(Base.add_sum)`. Keep both
        # sizes so a regression in either the base case or the pairwise-recursive branch is caught.
        g_t(M) = sum(transpose(M))

        for n in (2, 40)   # 40x40 is the size that used to reach the illegal statement
            M = rand(n, n)
            _, dM = Differ.gradient(g_t, M)
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
