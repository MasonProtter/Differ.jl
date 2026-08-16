using Test
using DifferReverse
using DifferReverse: NoTangent, NoRData, NoFData, rev_gradient
using DifferReverse: zero_fcodual, rrule!!, Ctx, primal, tangent, code_reverse_fwds_ircode
# `Dual`/`frule!!` here are DifferForwards' forward-mode carrier, used purely as an independent
# numerical oracle — forward mode's `Core.ifelse` rule predates reverse mode's.
using DifferForwards: Dual, frule!!

include(joinpath(@__DIR__, "testutils.jl"))

# `Core.ifelse` reverse rule (`builtins_reverse.jl`). Before this, any primal reaching it bailed
# unconditionally in reverse mode.

# Module-level: plain, non-capturing function definitions, so no closure/boxing hazard (see
# test_reverse_tuples.jl's header for when that hazard applies — it doesn't here, nothing below
# captures a local).
f_ifelse_straight(x::Float64) = Core.ifelse(x > 0, 2x, 3x)

function loop_ifelse_sum(x::Float64, y::Float64, n::Int)
    s = 0.0
    for i in 1:n
        s += Core.ifelse(iseven(i), x, y)
    end
    return s
end

ifelse_vec(cond::Bool, a::Vector{Float64}, b::Vector{Float64}) = Core.ifelse(cond, a, b)

@testset "reverse mode: Core.ifelse — straight-line, both branches" begin
    # Confirm the optimized IR genuinely contains a `Core.ifelse` call — otherwise this is vacuous.
    ir = first(only(Base.code_ircode(f_ifelse_straight, (Float64,))))
    @test any(s -> s isa Expr && s.head === :call && s.args[1] === Core.ifelse, ir.stmts.stmt)

    for (x, expected) in ((2.0, 2.0), (-3.0, 3.0))   # x>0 => 2x branch (d=2); x<0 => 3x branch (d=3)
        _, dx = rev_gradient(f_ifelse_straight, x)
        @test dx ≈ expected
        @test dx ≈ central_diff(f_ifelse_straight, x)
        dfx = frule!!(Dual(f_ifelse_straight, NoTangent()), Dual(x, 1.0)).dx
        @test dx ≈ dfx
    end

    checkverify_rev(f_ifelse_straight, (Float64,))
    check_stack_balance(f_ifelse_straight, 2.0)
    check_stack_balance(f_ifelse_straight, -3.0)
end

@testset "reverse mode: Core.ifelse — loop-varying condition (taped Bool comms slot)" begin
    # `n` is a runtime argument (not a literal), so the loop genuinely iterates rather than
    # unrolling; `iseven(i)` varies per iteration, so the comms `Bool` slot sees both values across
    # the run. n=5: cond true at i=2,4 (selects x), false at i=1,3,5 (selects y) => ds/dx=2, ds/dy=3.
    ir = first(only(Base.code_ircode(loop_ifelse_sum, (Float64, Float64, Int))))
    @test any(s -> s isa Expr && s.head === :call && s.args[1] === Core.ifelse, ir.stmts.stmt)
    @test length(ir.cfg.blocks) > 1   # genuine control flow, not folded to straight-line

    x0, y0, n = 2.0, 5.0, 5
    _, dx, dy = rev_gradient(loop_ifelse_sum, x0, y0, n)
    @test dx ≈ 2.0
    @test dy ≈ 3.0
    @test dx ≈ central_diff(x -> loop_ifelse_sum(x, y0, n), x0)
    @test dy ≈ central_diff(y -> loop_ifelse_sum(x0, y, n), y0)

    dfx = frule!!(Dual(loop_ifelse_sum, NoTangent()), Dual(x0, 1.0), Dual(y0, 0.0), Dual(n, NoTangent())).dx
    dfy = frule!!(Dual(loop_ifelse_sum, NoTangent()), Dual(x0, 0.0), Dual(y0, 1.0), Dual(n, NoTangent())).dx
    @test dx ≈ dfx
    @test dy ≈ dfy

    checkverify_rev(loop_ifelse_sum, (Float64, Float64, Int))
    check_stack_balance(loop_ifelse_sum, x0, y0, n)
end

@testset "reverse mode: Core.ifelse — declines with its own reason (not the generic bail)" begin
    # An fdata-carrying result (array select) is a deliberate, located bail — soundly expressible
    # via shadow aliasing, but not implemented.
    err = try
        code_reverse_fwds_ircode(ifelse_vec, (Bool, Vector{Float64}, Vector{Float64}))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("ifelse", msg)
    @test !occursin("no reverse rule", msg)
end
