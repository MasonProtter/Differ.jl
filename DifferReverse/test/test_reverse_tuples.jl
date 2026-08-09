using Test
using DifferReverse
using DifferReverse: NoTangent, NoRData, NoFData, rev_gradient
using DifferReverse: zero_fcodual, rrule!!, Ctx, primal, tangent, code_reverse_fwds_ircode
# `Dual`/`frule!!` here are DifferForwards' forward-mode carrier, used purely as an independent
# numerical oracle.
using DifferForwards: Dual, frule!!

include(joinpath(@__DIR__, "testutils.jl"))

# `Core.tuple` reverse rule (`builtins_reverse.jl`, `_fdata_tracked`'s tuple arm). Before this, any
# tuple whose result carried a tangent (a multi-value return, `[a, b]` array literals, ...) bailed
# with "reverse mode does not support builtin `tuple` with a differentiable result".

# Module-level (not testset-local): a locally-defined function/global picks up a boxed/closure
# tangent type instead of the plain singleton these tests want (same requirement documented in
# test_reverse_dispatch_recursion.jl).
mutable struct TupBox
    t::Tuple{Vector{Float64},Float64}
end

# A `Vector{Float64}`-typed, non-`const` global: its read is its own `GlobalRef` statement with no
# provenance the `_fdata_tracked` scan marks as tracked (unlike an `Argument` or a locally-`%new`'d
# object) — used by the negative test below.
global _tuple_untracked_g::Vector{Float64} = [1.0, 2.0]
tuple_untracked(x::Float64) = (_tuple_untracked_g, x)

@testset "reverse mode: Core.tuple — motivating case (sum(abs2,x), :hello)" begin
    # The reported case: a tuple whose first slot is differentiable (rdata) and second isn't
    # (Symbol, NoTangent) — `rrule!!`'s own multi-value return does exactly this. `rev_gradient`
    # doesn't apply (it seeds via `one(y)`, undefined for a tuple); seed explicitly instead.
    f(x) = (sum(abs2, x), :hello)
    x = [0.386, 1.520, 1.979, 0.528, 1.853, 0.439, 1.468, 1.198, 1.780, 1.008]
    xshadow = zeros(length(x))

    y, back = rrule!!(zero_fcodual(f), Ctx(), DifferReverse.CoDual(x, xshadow))
    @test primal(y) == f(x)
    @test tangent(y) === NoFData()
    @test back((1.0, NoRData())) == (NoRData(), NoRData())
    @test xshadow ≈ 2 .* x

    checkverify_rev(f, (Vector{Float64},))
    check_stack_balance(f, x; seed=(1.0, NoRData()))
end

@testset "reverse mode: Core.tuple — pure-rdata, both slots live" begin
    # Both slots differentiable (Float64, Float64): (x*y, x+y). Seeded one component at a time,
    # cross-checked against the already-trusted forward-mode `frule!!` (row vs. column of the same
    # Jacobian — reverse seed e_i gives d(out_i)/d(every input), forward seed e_j gives
    # d(every output)/d(input_j); they agree on the shared entries).
    g(x, y) = (x*y, x+y)
    x0, y0 = 2.0, 3.0

    yc, back = rrule!!(zero_fcodual(g), Ctx(), zero_fcodual(x0), zero_fcodual(y0))
    @test primal(yc) == g(x0, y0)
    _, dx1, dy1 = back((1.0, 0.0))
    @test dx1 ≈ y0   # d(x*y)/dx
    @test dy1 ≈ x0   # d(x*y)/dy

    yc2, back2 = rrule!!(zero_fcodual(g), Ctx(), zero_fcodual(x0), zero_fcodual(y0))
    _, dx2, dy2 = back2((0.0, 1.0))
    @test dx2 ≈ 1.0  # d(x+y)/dx
    @test dy2 ≈ 1.0  # d(x+y)/dy

    dfx = frule!!(Dual(g, NoTangent()), Dual(x0, 1.0), Dual(y0, 0.0)).dx
    dfy = frule!!(Dual(g, NoTangent()), Dual(x0, 0.0), Dual(y0, 1.0)).dx
    @test dx1 ≈ dfx[1]
    @test dx2 ≈ dfx[2]
    @test dy1 ≈ dfy[1]
    @test dy2 ≈ dfy[2]

    checkverify_rev(g, (Float64, Float64))
    check_stack_balance(g, x0, y0; seed=(1.0, 0.0))
    check_stack_balance(g, x0, y0; seed=(0.0, 1.0))
end

@testset "reverse mode: Core.tuple — built and consumed inside the body (dynamic index)" begin
    # `t = (x, y)` then indexed dynamically in a loop (`mod1` defeats constant-index unrolling, which
    # would otherwise let SROA scalarize the tuple away before it ever reaches this rule — confirmed
    # via `Base.code_ircode`). Scalar output, so `rev_gradient`/`check_stack_balance` apply directly.
    # t[1] is read `mod1(i,2)==1` times, t[2] the rest: for n=5, indices are 1,2,1,2,1 => dx=3, dy=2.
    function tuple_loop_sum(x::Float64, y::Float64, n::Int)
        t = (x, y)
        s = 0.0
        for i in 1:n
            s += t[mod1(i, 2)]
        end
        return s
    end

    _, dx, dy = rev_gradient(tuple_loop_sum, 2.0, 3.0, 5)
    @test dx ≈ 3.0
    @test dy ≈ 2.0
    @test dx ≈ central_diff(x -> tuple_loop_sum(x, 3.0, 5), 2.0) rtol = 1e-5
    @test dy ≈ central_diff(y -> tuple_loop_sum(2.0, y, 5), 3.0) rtol = 1e-5

    checkverify_rev(tuple_loop_sum, (Float64, Float64, Int))
    check_stack_balance(tuple_loop_sum, 2.0, 3.0, 5)
end

@testset "reverse mode: Core.tuple — fdata-carrying tuple, field read back via getfield" begin
    # `(v, x)` — `v::Vector{Float64}` carries fdata. Wrapping it in a mutable struct field is what
    # keeps the tuple materialized as a real `Core.tuple` call reachable by literal `getfield`s
    # (confirmed via `Base.code_ircode`: without the wrapper, SROA scalarizes a straight-line tuple
    # away before it ever reaches this rule, same as the previous testset's constant-index case).
    function tuple_array_field(v::Vector{Float64}, x::Float64)
        box = TupBox((v, x))
        arr, y = box.t
        return arr[1] * y
    end

    v = [2.0, 3.0]
    x = 5.0
    _, dv, dx = rev_gradient(tuple_array_field, v, x)
    @test dv ≈ [x, 0.0]
    @test dx ≈ v[1]
    @test dx ≈ central_diff(x -> tuple_array_field(v, x), x) rtol = 1e-5

    checkverify_rev(tuple_array_field, (Vector{Float64}, Float64))
    check_stack_balance(tuple_array_field, v, x)
end

@testset "reverse mode: Core.tuple — [a, b] array literal (Base.vect)" begin
    # `[a, b]` lowers through `Base.vect`'s `X = Core.tuple(a, b)` capture, filled into the result
    # array by a dynamic-index loop. Scalar elements only: `Base.vect`'s fill loop reads `X[i]` at
    # a dynamic index, and a dynamic `getfield` into an fdata-carrying (array-valued) tuple field is
    # a separate, still-open limitation (`getfield`'s own homogeneous-pure-rdata restriction).
    g_vect(a, b) = sum(x -> x^2, [a, b])
    a0, b0 = 2.0, 3.0

    _, da, db = rev_gradient(g_vect, a0, b0)
    @test da ≈ 2a0
    @test db ≈ 2b0
    @test da ≈ central_diff(a -> g_vect(a, b0), a0) rtol = 1e-5
    @test db ≈ central_diff(b -> g_vect(a0, b), b0) rtol = 1e-5

    checkverify_rev(g_vect, (Float64, Float64))
    check_stack_balance(g_vect, a0, b0)
end

@testset "reverse mode: Core.tuple — declines with its own reason (not the generic bail)" begin
    # `_tuple_untracked_g`'s read has no provenance the `_fdata_tracked` scan marks tracked, so the
    # comms-scan gate (`builtin_rrule_comms(::Val{Core.tuple}, ...)`) declines instead of emitting IR
    # against an unresolvable shadow. Must be the rule's own located reason, not the pre-existing
    # generic "no reverse rule for builtin `tuple`" message this rule replaces for every other case.
    err = try
        code_reverse_fwds_ircode(tuple_untracked, (Float64,))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test !(err isa MethodError)
    msg = sprint(showerror, err)
    @test occursin("tuple", msg)
    @test occursin("carries fdata", msg)
    @test occursin("traceable to a function argument", msg)
    @test !occursin("no reverse rule", msg)
end
