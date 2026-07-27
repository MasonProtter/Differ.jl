using Test
using Differ
using Differ: Dual, NoTangent, frule!!, gradient, get_tangent_field, MutableTangent

include(joinpath(@__DIR__, "testutils.jl"))

# A dynamic-dispatch callee (read from a non-`const` global) is not statically recursible — must
# bail cleanly, not crash. Must be a real module-level global (not testset-local) — a local would
# instead be captured as a closure field, a different IR shape from what this test exercises.
dyn_g = sin
dyncallee(x) = dyn_g(x)

# Regression: a self-recursive `@noinline` primal has no finite `Tape` type in this design (the
# `in_progress` cycle guard) — must bail with a clean `ErrorException`, not recurse forever /
# stack-overflow. Must be a true top-level function, not testset-local: a *self*-recursive
# function defined in a local scope needs to close over its own (boxed) binding to call itself,
# which gives it a real (non-`NoTangent`) tangent type instead of the plain singleton a top-level
# function has, defeating `gradient`'s zero-tangent seeding for `f`.
@noinline function rec_self(x::Float64, n::Int)
    n <= 0 && return x
    return rec_self(x, n - 1)
end

@testset "reverse mode: recursion into a hand-written rule" begin
    # A surviving high-level call differentiates via the recursive `rrule` support below (`sin`
    # specifically resolves to the hand-written rule in `src/rrules.jl`, not raw recursion into
    # `Base.Math.sin`'s internals — see that file's header).
    plus1(x) = sin(x) + 1
    nest(x)  = sin(cos(x))     # composed hand rules: sin(cos(x))

    _, dx_plus1 = gradient(plus1, 1.3)
    @test dx_plus1 ≈ cos(1.3)

    checkverify_rev(plus1, (Float64,))
    checkverify_rev(nest, (Float64,))
    check_stack_balance(plus1, 1.3)
    check_stack_balance(nest, 0.4)
end

@testset "reverse mode: recursive rrule calls (statically-resolvable only)" begin
    # Tier 5 Part 1: a genuinely separate, non-inlined callee must actually survive as a
    # surviving `:invoke` to exercise recursion, not just fold into ordinary arithmetic — so it's
    # `@noinline`.
    @noinline rec_sq(x) = x * x                         # x^2, d/dx = 2x
    rec_call(x) = rec_sq(x) + x                          # d/dx = 2x + 1
    function rec_branch(x)                               # recursion combined with a branch
        return x > 0.0 ? rec_sq(x) + 1.0 : rec_sq(x) - 1.0
    end
    function rec_loop(x, k)                              # recursion combined with a loop
        s = 0.0
        for i in 1:k
            s += rec_sq(x)
        end
        return s
    end

    _, dx_rec = gradient(rec_call, 3.0)
    @test dx_rec ≈ 2*3.0 + 1
    @test dx_rec ≈ frule!!(Dual(rec_call, NoTangent()), Dual(3.0, 1.0)).dx

    for x in (2.0, -2.0)
        _, dx_rb = gradient(rec_branch, x)
        @test dx_rb ≈ 2x
    end

    _, dx_rl = gradient(rec_loop, 2.0, 3)
    @test dx_rl ≈ 3 * 2 * 2.0
    check_stack_balance(rec_loop, 2.0, 5)

    @test_throws ErrorException gradient(rec_self, 1.0, 3)

    checkverify_rev(rec_call, (Float64,))
    checkverify_rev(rec_branch, (Float64,))
    checkverify_rev(rec_loop, (Float64, Int))
end

@testset "reverse mode: dynamic (non-statically-resolvable) callee bails" begin
    @test_throws ErrorException gradient(dyncallee, 1.0)
    # Asserting the exception is a located `ErrorException` naming the construct
    # (`_static_recursible_call`'s "dynamic (non-statically-resolvable) callee" message,
    # `src/reverse_interp.jl`), not just any `ErrorException` — and explicitly not a `MethodError`
    # or other crash, which a bare `@test_throws ErrorException` would not distinguish from.
    err_dyncall = try
        gradient(dyncallee, 1.0)
        nothing
    catch e
        e
    end
    @test err_dyncall isa ErrorException
    @test !(err_dyncall isa MethodError)
    @test occursin("dynamic (non-statically-resolvable) callee", err_dyncall.msg)
end

@testset "reverse mode: dynamic (non-literal) getfield index" begin
    # Dynamic (non-literal) `getfield` index (Phase B): `for i in 1:2` does not unroll, so `t[i]`
    # reaches the pullback as a genuine dynamic index. Before the fix, the pullback resolved the
    # field to a raw, unresolved `SSAValue` instead of its runtime value, silently degenerating
    # `increment_field!!`'s `Val`-based dispatch into a no-op — the gradient came back
    # `(0.0,0.0)` instead of `(1.0,1.0)`, with no error at all. `d/dt_1 = d/dt_2 = 1`.
    function tupsum_dyn(t::Tuple{Float64,Float64})
        s = 0.0
        for i in 1:2
            s += t[i]
        end
        return s
    end
    function ntupsum_dyn(t::@NamedTuple{a::Float64,b::Float64})
        s = 0.0
        for i in 1:2
            s += t[i]
        end
        return s
    end
    mutable struct MP2; x::Float64; y::Float64; end
    mp2get(m::MP2, i::Int) = Core.getfield(m, i)     # index is a genuine `Argument` (`_3`)
    function mp2sum_dyn(m::MP2)                       # loop index is a genuine `SSAValue` (a phi)
        s = 0.0
        for i in 1:2
            s += Core.getfield(m, i)
        end
        return s
    end
    function setdyn!(m::MP2, i::Int, v::Float64)
        Core.setfield!(m, i, v)
        return m.x + m.y
    end
    # Heterogeneous struct/mutable struct, dynamic `getfield` index: the genuinely hard case
    # (per-field tangent types differ) that must always bail, never miscompile.
    struct Het2; a::Float64; b::Int; end
    hetdyn(h::Het2, i::Int) = Core.getfield(h, i)
    mutable struct MHet2; a::Float64; b::Int; end
    mhetdyn(m::MHet2, i::Int) = Core.getfield(m, i)

    _, dt_dyn = gradient(tupsum_dyn, (3.0, 4.0))
    @test dt_dyn == (1.0, 1.0)
    @test dt_dyn[1] == frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), (1.0, 0.0))).dx
    @test dt_dyn[2] == frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), (0.0, 1.0))).dx
    h = 1e-6
    @test dt_dyn[1] ≈ (tupsum_dyn((3.0 + h, 4.0)) - tupsum_dyn((3.0 - h, 4.0))) / 2h rtol = 1e-5
    @test dt_dyn[2] ≈ (tupsum_dyn((3.0, 4.0 + h)) - tupsum_dyn((3.0, 4.0 - h))) / 2h rtol = 1e-5
    checkverify_rev(tupsum_dyn, (Tuple{Float64,Float64},))

    # same, over a homogeneous NamedTuple.
    _, dnt_dyn = gradient(ntupsum_dyn, (a=3.0, b=4.0))
    @test dnt_dyn == (a=1.0, b=1.0)
    checkverify_rev(ntupsum_dyn, (@NamedTuple{a::Float64,b::Float64},))

    # dynamic getfield index into a homogeneous MUTABLE struct (Part 2b): the field's rdata
    # contribution routes into the object's own `MutableTangent` via the runtime-`Int`
    # `increment_field_rdata!` (not an object-level `RData`, as a mutable struct has none). The
    # gradient w.r.t. the struct is a one-hot `MutableTangent` for a single selected field, and
    # all-ones when summed over both. Index is a genuine `Argument`/`SSAValue`, not const-folded.
    _, dmp2_r, _ = gradient(mp2get, MP2(3.0, 4.0), 2)
    @test get_tangent_field(dmp2_r, :x) == 0.0 && get_tangent_field(dmp2_r, :y) == 1.0
    _, dmp1_r, _ = gradient(mp2get, MP2(3.0, 4.0), 1)
    @test get_tangent_field(dmp1_r, :x) == 1.0 && get_tangent_field(dmp1_r, :y) == 0.0
    _, dms_r = gradient(mp2sum_dyn, MP2(3.0, 4.0))
    @test get_tangent_field(dms_r, :x) == 1.0 && get_tangent_field(dms_r, :y) == 1.0
    checkverify_rev(mp2get, (MP2, Int))
    checkverify_rev(mp2sum_dyn, (MP2,))

    # regression: a dynamic getfield index into a HETEROGENEOUS struct is the genuinely hard case
    # — must bail with a located error, never crash or silently return a wrong/zero gradient.
    @test_throws "dynamic (non-literal) field index" gradient(hetdyn, Het2(1.0, 2), 1)
    err_het = try
        gradient(hetdyn, Het2(1.0, 2), 1)
        nothing
    catch e
        e
    end
    @test err_het isa ErrorException
    @test !(err_het isa MethodError)
    @test occursin("dynamic (non-literal) field index", err_het.msg)

    # regression: the same HETEROGENEOUS case through the mutable-struct branch of `getfield`'s
    # comms rule (`builtin_rrule_comms(::Val{Core.getfield},...)`, `src/builtins_reverse.jl`) —
    # a separate code path from the immutable case above, must bail identically.
    err_mhet = try
        gradient(mhetdyn, MHet2(1.0, 2), 1)
        nothing
    catch e
        e
    end
    @test err_mhet isa ErrorException
    @test !(err_mhet isa MethodError)
    @test occursin("dynamic (non-literal) field index", err_mhet.msg)

    # regression: a dynamic setfield! index — Phase A only, always bails.
    @test_throws "dynamic (non-literal) field index" gradient(setdyn!, MP2(1.0, 2.0), 1, 5.0)
end
