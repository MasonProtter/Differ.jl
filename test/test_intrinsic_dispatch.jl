# Tests for Differ's dispatch-based handling of `Core.Intrinsics` (see `src/intrinsics.jl`).
#
# The mechanism:
#   1. Every `Core.Intrinsics` function is an instance of the single type `Core.IntrinsicFunction`,
#      but an intrinsic *value* is a valid type parameter, so `Val{Core.Intrinsics.add_float}` names
#      one specific intrinsic and ordinary multiple dispatch on `Val` works.
#   2. `dualize_to_ircode` calls `apply_intrinsic_frule!(Val(f), actual, Ti, ctx)` for every
#      intrinsic call in the primal IR. Each method emits the primal + shadow IR *directly* — no
#      `Dual` boxing, no `frule!!` dispatch, no `CodeInstance` resolution.
#
# *Differentiable* float intrinsics get a hand-written rule (`add_float`, `sqrt_llvm`, `max_float`,
# `fma_float`, `fpext`, …); *non-differentiable* ones (comparisons, integer/bit ops, rounding,
# int↔float conversions) are registered with `@inactive_intrinsic`, which emits the primal and a
# zero tangent. The fallback method returns `nothing` — no identity fallback.

@testset "intrinsic dispatch" begin
    @testset "unregistered intrinsic bails with a located reason" begin
        # `bitcast` is deliberately never registered (see `src/intrinsics.jl`); hitting it must bail
        # gracefully with a message naming the intrinsic, not throw/miscompile.
        f(x) = reinterpret(Int64, x)
        err = try
            code_dual_ircode(f, (Float64,))
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("bitcast", sprint(showerror, err))
    end

    @testset "inactive (non-differentiable) intrinsics" begin
        D(f, x) = frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx

        # Integer arithmetic, comparisons, and bit ops all contribute a zero tangent.
        addi(x) = Int(x) + 1
        @test frule!!(Dual(addi, NoTangent()), Dual(2.0, 1.0)) === Dual(3, NoTangent())

        # A comparison: primal is the `Bool`, tangent is `NoTangent`.
        lt(x, y) = x < y
        @test frule!!(Dual(lt, NoTangent()), Dual(1.0, 1.0), Dual(2.0, 5.0)) === Dual(true, NoTangent())

        # A branch driven by a comparison dualizes end-to-end (relu-style).
        relu(x) = x > 0.0 ? x : -x
        rr = Dual(relu, NoTangent())
        @test frule!!(rr, Dual(3.0, 1.0)) === Dual(3.0, 1.0)
        @test frule!!(rr, Dual(-3.0, 1.0)) === Dual(3.0, -1.0)

        # inactive conversions / rounding: primal computed, tangent is zero
        @test D(x -> x + Float64(3), 2.0) == 1.0          # sitofp on a constant contributes 0
        @test D(floor, 2.7) == 0.0                        # floor_llvm
        @test D(round, 2.7) == 0.0                        # rint_llvm
        @test frule!!(Dual(x -> Float64(trunc(Int, x)), NoTangent()), Dual(2.7, 1.0)) === Dual(2.0, 0.0)
    end

    @testset "differentiable float intrinsics (vs finite differences)" begin
        D(f, x)     = frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx           # d/dx f(x)
        Dx(f, x, y) = frule!!(Dual(f, NoTangent()), Dual(x, 1.0), Dual(y, 0.0)).dx   # ∂/∂x f(x,y)
        Dy(f, x, y) = frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 1.0)).dx   # ∂/∂y f(x,y)
        fd(f, x; h=1e-6) = (f(x + h) - f(x - h)) / 2h

        # add/sub/neg/mul/div_float
        @test Dx((x, y) -> x + y, 3.0, 5.0) == 1.0
        @test Dx((x, y) -> x - y, 3.0, 5.0) == 1.0
        @test D(x -> -x, 3.0) == -1.0
        @test Dx((x, y) -> x * y, 3.0, 5.0) == 5.0
        @test Dx((x, y) -> x / y, 3.0, 5.0) == 0.2
        # sqrt_llvm: d√x = 1/(2√x)
        @test D(sqrt, 4.0) ≈ 0.25
        @test D(sqrt, 2.0) ≈ fd(sqrt, 2.0)
        # abs_float: sign(x)
        @test D(abs, 3.0)  == 1.0
        @test D(abs, -3.0) == -1.0
        # max_float / min_float: tangent follows the selected operand
        @test Dx(max, 5.0, 2.0) == 1.0 && Dy(max, 5.0, 2.0) == 0.0
        @test Dx(max, 2.0, 5.0) == 0.0 && Dy(max, 2.0, 5.0) == 1.0
        @test Dx(min, 2.0, 5.0) == 1.0 && Dy(min, 2.0, 5.0) == 0.0
        # fma / muladd (a·b + c): ∂/∂a = b, ∂/∂b = a
        @test Dx((a, b) -> fma(a, b, 1.0), 3.0, 4.0)    == 4.0
        @test Dy((a, b) -> fma(a, b, 1.0), 3.0, 4.0)    == 3.0
        @test Dx((a, b) -> muladd(a, b, 1.0), 3.0, 4.0) == 4.0
        # copysign: ∂/∂x = sign(x)·sign(y), ∂/∂y = 0
        @test Dx(copysign, 3.0, 5.0)  == 1.0
        @test Dx(copysign, 3.0, -5.0) == -1.0
        @test Dy(copysign, 3.0, 5.0)  == 0.0
        # fpext / fptrunc (Float32 ↔ Float64 width conversion): linear, derivative 1
        @test D(x -> Float64(Float32(x)), 2.5) == 1.0
        r32 = frule!!(Dual(x -> Float64(x), NoTangent()), Dual(2.5f0, 1.0f0))
        @test r32 === Dual(2.5, 1.0)
    end

    @testset "dualized IR emits intrinsics directly (no frule!! round trip)" begin
        # A primal `+` inlines to `add_float`, which now dualizes to a directly-emitted `add_float`
        # op for both primal and shadow — no boxing into a `Dual` and no `invoke frule!!(...)`.
        myadd(x, y) = x + y
        ir, rt = code_dual_ircode(myadd, (Float64, Float64))
        @test rt === Dual{Float64, Float64}
        stmts = string.(ir.stmts.stmt)
        @test count(s -> occursin("add_float", s), stmts) == 2
        @test !any(s -> occursin("frule!!", s), stmts)
    end

    @testset "end-to-end derivatives + allocation-free" begin
        # d/dx of each arithmetic combination, differentiating the first argument (dx=1, dy=0).
        f_add(x, y) = x + y
        f_mul(x, y) = x * y
        f_div(x, y) = x / y
        f_sub(x, y) = x - y
        deriv(f, x, y) = frule!!(Dual(f, NoTangent()), Dual(x, 1.0), Dual(y, 0.0)).dx
        @test deriv(f_add, 3.0, 5.0) === 1.0          # ∂/∂x (x+y) = 1
        @test deriv(f_sub, 3.0, 5.0) === 1.0          # ∂/∂x (x-y) = 1
        @test deriv(f_mul, 3.0, 5.0) === 5.0          # ∂/∂x (x*y) = y
        @test deriv(f_div, 3.0, 5.0) === 0.2          # ∂/∂x (x/y) = 1/y

        # Allocation-free after warmup — the direct-emission path never boxes a `Dual` per op.
        df = Dual(f_mul, NoTangent())
        frule!!(df, Dual(3.0, 1.0), Dual(5.0, 0.0))
        @test (@allocated frule!!(df, Dual(3.0, 1.0), Dual(5.0, 0.0))) == 0
    end
end
