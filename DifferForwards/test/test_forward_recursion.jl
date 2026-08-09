using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, frule!!, zero_tangent, code_dual_ircode

include(joinpath(@__DIR__, "testutils.jl"))

# Forward-mode recursion: self-recursion resolves to a static self-`:invoke` against the bare,
# uncompiled `MethodInstance` currently being dualized (legal and fast only because it *is* that
# carrier — see `codegen.cpp`'s `mi == ctx.linfo` self-recursion fast path); mutual recursion (A→B→A)
# falls back to the ordinary dynamic `Expr(:call, frule!!, …)` form on whichever edge closes the
# cycle. Neither shape needs a fixed point solved for the `Dual` type the way reverse mode's `Tape`
# type does: `Dual{P,tangent_type(P)}` doesn't grow with recursion depth, and a call site's result
# type is read straight off `pir` — already fixed-point-solved by Julia's own inference. See
# `src/forward_interp.jl`'s `dual_recursive_impl_mi`/`frule_split!` for the resolver.

# Top-level, not testset-local: a self-recursive function defined in local scope closes over its own
# boxed binding, which gives it a real (non-`NoTangent`) tangent type instead of the plain singleton a
# top-level function has — that would defeat the `Dual(f, NoTangent())`/`zero_tangent(f)` seeding
# below (same reason documented at the top of `test_forward_closures_higher_order.jl`).

@noinline function rec_pow(x::Float64, n::Int)
    n <= 0 && return one(x)
    return x * rec_pow(x, n - 1)
end

@noinline function rec_arrsum(v::Vector{Float64}, i::Int)
    i > length(v) && return 0.0
    return v[i] + rec_arrsum(v, i + 1)
end

# Recursive call inside a real `if`/`else` branch (not just an early-return guard), so control flow
# and the self-edge interact: the recursive `:invoke` sits in one arm's block, not the entry block.
@noinline function rec_ifbranch(x::Float64, n::Int)
    if n <= 0
        return x
    else
        y = x * x
        return rec_ifbranch(y, n - 1)
    end
end

@noinline mutA(x::Float64, n::Int) = n <= 0 ? x : 2 * mutB(x, n - 1)
@noinline mutB(x::Float64, n::Int) = n <= 0 ? x : 3 * mutA(x, n - 1)

# Differentiate a scalar function of `x` w.r.t. `x`, holding any trailing arguments (a recursion-depth
# `Int`, say) fixed and non-differentiable.
dfdx(f, x, others...) =
    frule!!(Dual(f, zero_tangent(f)), Dual(x, one(x)),
            (Dual(a, NoTangent()) for a in others)...).dx

# The dual IR for a self-recursive primal contains a static self-`:invoke` — its target is a bare
# `MethodInstance`, unlike every other `:invoke` this engine emits (always a compiled `CodeInstance`).
# Guards against a silent regression to the dynamic `:call` form, or worse, to a boxed `jl_invoke`
# against the wrong (native) method cache, which would be a silently wrong answer, not an error.
function has_self_invoke(ir)
    for i in 1:length(ir.stmts)
        s = ir.stmts[i][:stmt]
        isa(s, Expr) && s.head === :invoke && isa(s.args[1], Core.MethodInstance) && return true
    end
    return false
end

@testset "self-recursion: scalar power" begin
    # rec_pow(x,n) = xⁿ, d/dx = n·xⁿ⁻¹.
    for n in (0, 1, 4)
        d = dfdx(rec_pow, 1.3, n)
        @test d ≈ central_diff(x -> rec_pow(x, n), 1.3)
    end
    checkverify(x -> rec_pow(x, 4), (Float64,))
    ir, _ = code_dual_ircode(rec_pow, (Float64, Int))
    @test has_self_invoke(ir)
    @test bail_reason(rec_pow, (Float64, Int)) === nothing
end

@testset "self-recursion: accumulating over an array" begin
    # rec_arrsum(v,1) = Σv — linear, so d/dv₁ = 1 regardless of the other entries or array length.
    # A plain `[x, 2.0, 3.0]` literal — the shape a user would actually write.
    arrsum_at(x) = rec_arrsum([x, 2.0, 3.0], 1)
    d = dfdx(arrsum_at, 0.0)
    @test d ≈ central_diff(arrsum_at, 0.0)
    checkverify(arrsum_at, (Float64,))
    ir, _ = code_dual_ircode(rec_arrsum, (Vector{Float64}, Int))
    @test has_self_invoke(ir)
    @test bail_reason(rec_arrsum, (Vector{Float64}, Int)) === nothing
end

@testset "self-recursion: recursive call inside a branch" begin
    for n in (0, 1, 3)
        d = dfdx(rec_ifbranch, 1.2, n)
        @test d ≈ central_diff(x -> rec_ifbranch(x, n), 1.2)
    end
    checkverify(x -> rec_ifbranch(x, 3), (Float64,))
    ir, _ = code_dual_ircode(rec_ifbranch, (Float64, Int))
    @test has_self_invoke(ir)
    @test bail_reason(rec_ifbranch, (Float64, Int)) === nothing
end

@testset "self-recursion: depth ~50 (no compile-time blow-up)" begin
    d = dfdx(rec_pow, 1.0, 50)
    @test d ≈ 50.0
    @test d ≈ central_diff(x -> rec_pow(x, 50), 1.0)
end

@testset "mutual recursion (A→B→A), both entry points" begin
    for n in (0, 1, 2, 5)
        dA = dfdx(mutA, 1.1, n)
        @test dA ≈ central_diff(x -> mutA(x, n), 1.1)
        dB = dfdx(mutB, 1.1, n)
        @test dB ≈ central_diff(x -> mutB(x, n), 1.1)
    end
    checkverify(x -> mutA(x, 5), (Float64,))
    checkverify(x -> mutB(x, 5), (Float64,))
    @test bail_reason(mutA, (Float64, Int)) === nothing
    @test bail_reason(mutB, (Float64, Int)) === nothing
    # Genuinely mutual (A calls B, B calls A) — neither side's own carrier is ever its own callee, so
    # neither dual IR should contain the bare-MI self-`:invoke` form; the cycle is broken by the
    # dynamic `:call` back-edge instead (see `dual_recursive_impl_mi`'s three-way branch).
    irA, _ = code_dual_ircode(mutA, (Float64, Int))
    irB, _ = code_dual_ircode(mutB, (Float64, Int))
    @test !has_self_invoke(irA)
    @test !has_self_invoke(irB)
end
