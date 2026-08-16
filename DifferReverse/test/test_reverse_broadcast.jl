using Test
using DifferReverse
using DifferReverse: rev_gradient

include(joinpath(@__DIR__, "testutils.jl"))

function cdiff_vec(f, x::Vector{Float64}; h=1e-6)
    g = similar(x)
    for k in eachindex(x)
        xp = copy(x); xp[k] += h
        xm = copy(x); xm[k] -= h
        g[k] = (f(xp) - f(xm)) / 2h
    end
    return g
end

const v9 = [0.3, -1.2, 2.5]
const w9 = [1.7, 0.4, -0.9]

@testset "reverse mode: `.`-broadcast through the derived path" begin
    f_plus(x) = sum(x .+ x)
    _, dplus = rev_gradient(f_plus, v9)
    @test dplus == [2.0, 2.0, 2.0]
    @test dplus ≈ cdiff_vec(f_plus, v9) rtol = 1e-5
    checkverify_rev(f_plus, (Vector{Float64},))
    check_stack_balance(f_plus, copy(v9))

    f_sin(x) = sum(sin.(x))
    _, dsin = rev_gradient(f_sin, v9)
    @test dsin ≈ cos.(v9)
    @test dsin ≈ cdiff_vec(f_sin, v9) rtol = 1e-5
    checkverify_rev(f_sin, (Vector{Float64},))
    check_stack_balance(f_sin, copy(v9))

    f_mul(x, y) = sum(x .* y)
    _, dmul_x, dmul_y = rev_gradient(f_mul, v9, w9)
    @test dmul_x == w9
    @test dmul_y == v9
    @test dmul_x ≈ cdiff_vec(x -> f_mul(x, w9), v9) rtol = 1e-5
    @test dmul_y ≈ cdiff_vec(y -> f_mul(v9, y), w9) rtol = 1e-5
    checkverify_rev(f_mul, (Vector{Float64}, Vector{Float64}))
    check_stack_balance(f_mul, copy(v9), copy(w9))

    # Fused chain `v .* w .+ 2.0 .* v`: two nested broadcasts fused into one loop. Scalar operand
    # `2.0` hits the reflection-only gap, hence `checkverify_rev_no_pb_reflection` rather than
    # `checkverify_rev` — see that helper and the testset below.
    f_fused(x, y) = sum(x .* y .+ 2.0 .* x)
    _, dfused_x, dfused_y = rev_gradient(f_fused, v9, w9)
    @test dfused_x == [3.7, 2.4, 1.1]
    @test dfused_y == v9
    @test dfused_x ≈ cdiff_vec(x -> f_fused(x, w9), v9) rtol = 1e-5
    @test dfused_y ≈ cdiff_vec(y -> f_fused(v9, y), w9) rtol = 1e-5
    checkverify_rev_no_pb_reflection(f_fused, (Vector{Float64}, Vector{Float64}))
    check_stack_balance(f_fused, copy(v9), copy(w9))

    # Same reflection-only gap as `f_fused` above.
    f_scalar(x) = sum(x .+ 1.0)
    _, dscalar = rev_gradient(f_scalar, v9)
    @test dscalar == [1.0, 1.0, 1.0]
    checkverify_rev_no_pb_reflection(f_scalar, (Vector{Float64},))
    check_stack_balance(f_scalar, copy(v9))

    # `copy(v)` exercises the `memmove`/`memcpy` foreigncall path directly, not through a broadcast.
    f_copy(x) = sum(copy(x))
    _, dcopy = rev_gradient(f_copy, v9)
    @test dcopy == [1.0, 1.0, 1.0]
    checkverify_rev(f_copy, (Vector{Float64},))
    check_stack_balance(f_copy, copy(v9))

    # let-bound captures: array and scalar closed over by the closure. `.* arr .+ s`'s scalar operand
    # hits the same reflection-only gap as `f_fused`/`f_scalar`.
    let arr = v9, s = 2.0
        f_let = x -> sum(x .* arr .+ s)
        _, dlet = rev_gradient(f_let, v9)
        @test dlet == arr
        @test dlet ≈ cdiff_vec(f_let, v9) rtol = 1e-5
        checkverify_rev_no_pb_reflection(f_let, (Vector{Float64},))
        check_stack_balance(f_let, copy(v9))
    end
end

# Scalar-operand broadcast pullback hits a reflection-tool-only `verify_ir` failure in
# `code_reverse_pullback_ircode` — confirmed not to affect the real carrier: `checkverify_prealloc`
# and `code_reverse_fwds_ircode` both pass, and `f_scalar`/`f_fused` above confirm gradient
# correctness via central differences. Pins the reflection-tool gap so a fix gets noticed rather
# than silently regressing to a crash.
@testset "reverse mode: broadcast pullback reflection gap" begin
    f_scalar2(x) = sum(x .+ 1.0)
    checkverify_prealloc(f_scalar2, (Vector{Float64},))
    Core.Compiler.verify_ir(code_reverse_fwds_ircode(f_scalar2, (Vector{Float64},))[1])
    @test_throws ErrorException Core.Compiler.verify_ir(
        code_reverse_pullback_ircode(f_scalar2, (Vector{Float64},))[1])
end

@testset "reverse mode: branch-merged array phi feeding an element read" begin
    # `y` is a `PhiNode` merging the original argument with a freshly-copied array, read through via
    # a broadcast — exactly the shape `_fdata_tracked`'s `PhiNode` arm exists for.
    function branchphi(x::Vector{Float64}, take_arg::Bool)
        y = take_arg ? x : copy(x)
        return sum(y .* 2.0)
    end

    # `.* 2.0` is a scalar-operand broadcast, so this also hits the reflection-only gap.
    _, dbp_t, = rev_gradient(branchphi, v9, true)
    @test dbp_t == [2.0, 2.0, 2.0]
    _, dbp_f, = rev_gradient(branchphi, v9, false)
    @test dbp_f == [2.0, 2.0, 2.0]
    checkverify_rev_no_pb_reflection(branchphi, (Vector{Float64}, Bool))
    check_stack_balance(branchphi, copy(v9), true)
    check_stack_balance(branchphi, copy(v9), false)
end

@testset "reverse mode: loop-carried array phi at >=2 iterations (fixpoint regression)" begin
    # `y` reassigned every iteration via broadcast, so the loop back-edge phi carries a different
    # tracked array each pass. A control-flow-replay/provenance change can look `verify_ir`-clean
    # yet be wrong — only gradient correctness at >=2 iterations is real evidence.
    # `y = x .+ x .+ ...` (n times) + initial `x` means `sum(y) = (n+1)*sum(x)`, so d/dx = (n+1)
    # elementwise.
    function loopphi(x::Vector{Float64}, n::Int)
        y = x
        for _ in 1:n
            y = y .+ x
        end
        return sum(y)
    end

    for n in (2, 3, 5)
        _, dlp = rev_gradient(loopphi, v9, n)
        @test dlp == fill(Float64(n + 1), length(v9))
    end
    checkverify_rev(loopphi, (Vector{Float64}, Int))
    check_stack_balance(loopphi, copy(v9), 3)
end

@testset "reverse mode: locally constructed immutable struct wrapping a Vector{Float64}" begin
    struct BroadcastWrap9
        v::Vector{Float64}
    end
    function wrapsum(x::Vector{Float64})
        w = BroadcastWrap9(x)
        return sum(w.v .* 2.0)
    end

    # `.* 2.0` is a scalar-operand broadcast, so this also hits the reflection-only gap.
    _, dws = rev_gradient(wrapsum, v9)
    @test dws == [2.0, 2.0, 2.0]
    checkverify_rev_no_pb_reflection(wrapsum, (Vector{Float64},))
    check_stack_balance(wrapsum, copy(v9))
end

@testset "reverse mode: literal integer exponent (`x .^ p`)" begin
    # `x .^ 4` (syntactic literal) lowers through `Base.literal_pow`, which broadcasting wraps as
    # `broadcasted(literal_pow, Ref(^), x, Ref(Val(p)))` — a different shape from a `const`-exponent
    # broadcast (`arr_to_num_linalg` in test_differentiation_interface.jl), since both `^` and
    # `Val(p)` arrive boxed in a `Base.RefValue`. `p == 2`/`3` hit Base's own dedicated small-exponent
    # `literal_pow` methods (plain multiplications), a different path from the general `Val{p}` one.
    f_p2(x) = sum(x .^ 2)
    f_p3(x) = sum(x .^ 3)
    f_p4(x) = sum(x .^ 4)
    f_p5(x) = sum(x .^ 5)

    for (f, dfdx) in ((f_p2, x -> 2 .* x), (f_p3, x -> 3 .* x .^ 2),
                      (f_p4, x -> 4 .* x .^ 3), (f_p5, x -> 5 .* x .^ 4))
        _, dv = rev_gradient(f, v9)
        @test dv ≈ dfdx(v9)
        @test dv ≈ cdiff_vec(f, v9) rtol = 1e-5
        checkverify_rev(f, (Vector{Float64},))
        check_stack_balance(f, copy(v9))
    end

    # The motivating case from the plan, pinned exactly.
    _, dsum4 = rev_gradient(x -> sum(x .^ 4), [1.0, 2.0])
    @test dsum4 == [4.0, 32.0]
end

@testset "reverse mode: partially-initialised `%new` still bails" begin
    # The gate change only widens the fully-initialised case; a genuine partial `%new` (fewer operands
    # than fields) must still bail, with the more precise message. `Partial2`'s inner constructor
    # supplies only `a`, leaving `b` undef; returning `p` itself (rather than a field read) keeps the
    # `%new` from being optimised away by SROA.
    mutable struct Partial2_9
        a::Float64
        b::Float64
        Partial2_9(x) = new(x)
    end
    f_partial(x) = Partial2_9(x)

    err = try
        rev_gradient(f_partial, 3.0)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("partially-initialised", err.msg)
    @test occursin("possibly-undef fields", err.msg)
    @test occursin("Partial2_9", err.msg)
end

gb_global9 = [1.0, 2.0, 3.0]

@testset "reverse mode: bail quality on a non-const global (not a crash)" begin
    # `gb_global9` is a non-const global reached through a broadcast; not traceable to a function
    # argument, so `Base.broadcasted`'s derived-recursion guard must bail with a located
    # `ErrorException` rather than crash or produce a wrong gradient. Asserts on stable substrings
    # only — full message's SSA numbers/gensym'd closure name shift with unrelated optimizer changes.
    err = try
        rev_gradient(x -> sum(x .* gb_global9), v9)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("non-concrete argument type Any", err.msg)
end
