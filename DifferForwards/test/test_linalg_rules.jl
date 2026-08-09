using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, frule!!, primal, tangent
using LinearAlgebra

include(joinpath(@__DIR__, "testutils.jl"))

# Throwaway named wrappers, one per rule under test. `checkverify` reflects on the dualized IR for
# a *call site* to the target function — a call that must survive as a genuine `:invoke` for the
# hand rule dispatch to be exercised at all. Passing the bare `LinearAlgebra` function directly
# asks Differ to dualize *that function's own* body instead (the "differentiate this function
# directly" entry point, unrelated to whether a hand rule exists for calls to it elsewhere), which
# fails for `dot`/`norm`/`*` since their real bodies bottom out in a BLAS `ccall` Differ can't see
# through regardless of the hand rule. Wrapping in a trivial named function makes the call to the
# target survive as an ordinary `:invoke`, which is what exercises the hand rule.
#
# Reverse-mode tests for the same rules live in DifferReverse/test/test_linalg_rules.jl.
dot_fn(x, y) = dot(x, y)
norm_fn(x) = norm(x)
tr_fn(A) = tr(A)

@testset "rules: linalg (forward)" begin

    @testset "dot" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, -1.0, 0.5]

        for k in eachindex(x)
            dxseed = zeros(3); dxseed[k] = 1.0
            dd = frule!!(Dual(dot_fn, NoTangent()), Dual(x, dxseed), Dual(y, zeros(3)))
            @test dd.x == dot(x, y)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dd.dx ≈ (dot(xp, y) - dot(xm, y)) / 2e-6 rtol = 1e-6
        end

        checkverify(dot_fn, (Vector{Float64}, Vector{Float64}))
    end

    @testset "norm" begin
        x = [3.0, -4.0, 1.0]

        for k in eachindex(x)
            dxseed = zeros(3); dxseed[k] = 1.0
            dd = frule!!(Dual(norm_fn, NoTangent()), Dual(x, dxseed))
            @test dd.x == norm(x)
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dd.dx ≈ (norm(xp) - norm(xm)) / 2e-6 rtol = 1e-6
        end

        checkverify(norm_fn, (Vector{Float64},))
    end

    @testset "tr" begin
        A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]

        for i in 1:3, j in 1:3
            dAseed = zeros(3, 3); dAseed[i, j] = 1.0
            dd = frule!!(Dual(tr_fn, NoTangent()), Dual(A, dAseed))
            @test dd.x == tr(A)
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dd.dx ≈ (tr(Ap) - tr(Am)) / 2e-6 rtol = 1e-6
        end

        checkverify(tr_fn, (Matrix{Float64},))
    end

    @testset "* (matrix-vector)" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]   # 3x2
        x = [1.5, -0.5]
        mv(A, x) = A * x

        for i in 1:3, j in 1:2
            dAseed = zeros(3, 2); dAseed[i, j] = 1.0
            dd = frule!!(Dual(mv, NoTangent()), Dual(A, dAseed), Dual(x, zeros(2)))
            @test dd.x == A * x
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dd.dx ≈ (Ap * x - Am * x) / 2e-6 atol = 1e-6
        end
        for k in 1:2
            dxseed = zeros(2); dxseed[k] = 1.0
            dd = frule!!(Dual(mv, NoTangent()), Dual(A, zeros(3, 2)), Dual(x, dxseed))
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test dd.dx ≈ (A * xp - A * xm) / 2e-6 atol = 1e-6
        end
        checkverify(mv, (Matrix{Float64}, Vector{Float64}))
    end

    @testset "* (matrix-matrix)" begin
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]
        mm(A, B) = A * B

        for i in 1:2, j in 1:2
            dAseed = zeros(2, 2); dAseed[i, j] = 1.0
            dd = frule!!(Dual(mm, NoTangent()), Dual(A, dAseed), Dual(B, zeros(2, 2)))
            @test dd.x == A * B
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test dd.dx ≈ (Ap * B - Am * B) / 2e-6 atol = 1e-6
        end
        checkverify(mm, (Matrix{Float64}, Matrix{Float64}))
    end

    @testset "mul! (matrix-vector)" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]   # 3x2
        x = [1.5, -0.5]
        mulv(A, x) = mul!(zeros(3), A, x)

        for i in 1:3, j in 1:2
            y = zeros(3)
            dAseed = zeros(3, 2); dAseed[i, j] = 1.0
            dd = frule!!(Dual(mul!, NoTangent()), Dual(y, zeros(3)), Dual(A, dAseed), Dual(x, zeros(2)))
            @test primal(dd) === y  # mul! returns (and mutates) y in place
            @test y == A * x
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test tangent(dd) ≈ (Ap * x - Am * x) / 2e-6 atol = 1e-6
        end
        for k in 1:2
            dxseed = zeros(2); dxseed[k] = 1.0
            dd = frule!!(Dual(mul!, NoTangent()), Dual(zeros(3), zeros(3)), Dual(A, zeros(3, 2)), Dual(x, dxseed))
            xp = copy(x); xp[k] += 1e-6
            xm = copy(x); xm[k] -= 1e-6
            @test tangent(dd) ≈ (A * xp - A * xm) / 2e-6 atol = 1e-6
        end
        checkverify(mulv, (Matrix{Float64}, Vector{Float64}))
    end

    @testset "mul! (matrix-matrix)" begin
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]
        mulm(A, B) = mul!(zeros(2, 2), A, B)

        for i in 1:2, j in 1:2
            C = zeros(2, 2)
            dAseed = zeros(2, 2); dAseed[i, j] = 1.0
            dd = frule!!(Dual(mul!, NoTangent()), Dual(C, zeros(2, 2)), Dual(A, dAseed), Dual(B, zeros(2, 2)))
            @test primal(dd) === C
            @test C == A * B
            Ap = copy(A); Ap[i, j] += 1e-6
            Am = copy(A); Am[i, j] -= 1e-6
            @test tangent(dd) ≈ (Ap * B - Am * B) / 2e-6 atol = 1e-6
        end
        checkverify(mulm, (Matrix{Float64}, Matrix{Float64}))
    end

    @testset "transpose/adjoint work in forward mode with NO new rule" begin
        # `transpose`/`adjoint` deliberately have no rules: their tangent already routes through
        # the generic struct-tangent machinery (`tangent_type(Transpose{Float64,
        # Matrix{Float64}})` is a real `Tangent`).
        M = [1.0 2.0; 3.0 4.0]

        g_t(M) = sum(transpose(M))
        g_a(M) = sum(adjoint(M))

        @test sum(transpose(M)) == sum(M)
        @test sum(adjoint(M)) == sum(M)

        dMseed = zeros(2, 2); dMseed[1, 2] = 1.0
        dd_t = frule!!(Dual(g_t, NoTangent()), Dual(M, dMseed))
        @test dd_t.dx ≈ 1.0
        dd_a = frule!!(Dual(g_a, NoTangent()), Dual(M, dMseed))
        @test dd_a.dx ≈ 1.0

        checkverify(g_t, (Matrix{Float64},))
        checkverify(g_a, (Matrix{Float64},))
    end

end
