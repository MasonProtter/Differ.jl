using Test
using DifferReverse
using DifferReverse: rev_gradient, value_and_gradient!, increment!!
import DifferentiationInterface as DI
include("testutils.jl")

# Declaring an argument constant: `NoTangent` in a `CoDual`'s shadow slot. Activity propagates from
# there through the primal IR, so anything reached only through a constant is replayed primally
# instead of differentiated.

@testset "activity: scalar arguments" begin
    f(x, y) = x * y + sin(x)
    fcd = zero_fcodual(f)
    x, y = 2.0, 3.0
    dfdx, dfdy = y + cos(x), x

    both = rrule!!(fcd, Ctx(), CoDual(x, NoFData()), CoDual(y, NoFData()))[2](1.0)
    @test both == (NoRData(), dfdx, dfdy)

    ycon = rrule!!(fcd, Ctx(), CoDual(x, NoFData()), const_codual(y))[2](1.0)
    @test ycon == (NoRData(), dfdx, NoRData())

    xcon = rrule!!(fcd, Ctx(), const_codual(x), CoDual(y, NoFData()))[2](1.0)
    @test xcon == (NoRData(), NoRData(), dfdy)

    # Everything constant: nothing is differentiated at all, and the primal value still comes out.
    allcon = rrule!!(fcd, Ctx(), const_codual(x), const_codual(y))
    @test primal(allcon[1]) == f(x, y)
    @test allcon[2](1.0) == (NoRData(), NoRData(), NoRData())

    for inactive in ((), (1,), (2,), (1, 2))
        checkverify_rev(f, (Float64, Float64); inactive)
    end
end

@testset "activity: an inactive argument is not merely zero-seeded" begin
    # A constant argument carries no shadow at all, so a rule that reached for one would fail rather
    # than silently produce a zero. Distinguishes this from passing a zero tangent.
    g(x, y) = x * y
    fcd = zero_fcodual(g)
    ycd = const_codual(3.0)
    @test tangent(ycd) === NoTangent()
    @test rrule!!(fcd, Ctx(), CoDual(2.0, NoFData()), ycd)[2](1.0) == (NoRData(), 3.0, NoRData())
end

@testset "activity: array arguments" begin
    function floop(v, w)
        s = 0.0
        for i in eachindex(v)
            s += v[i] * w[i]
        end
        return s
    end
    v, w = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    at = (Vector{Float64}, Vector{Float64})

    dv = zeros(3)
    _, pb = rrule!!(zero_fcodual(floop), Ctx(), CoDual(v, dv), const_codual(w))
    @test pb(1.0) == (NoRData(), NoRData(), NoRData())
    @test dv == w   # d/dv of sum(v .* w) is w; accumulated in place, no shadow for `w` allocated

    checkverify_rev(floop, at; inactive=(2,))

    # Same answer through the pre-allocated path, and reusable.
    ctx = build_ctx(floop, at; inactive=(2,))
    fcd, vcd, wcd = zero_fcodual(floop), CoDual(v, zeros(3)), const_codual(w)
    y1, g1 = value_and_gradient!(ctx, fcd, vcd, wcd)
    y2, g2 = value_and_gradient!(ctx, fcd, vcd, wcd)
    @test y1 == y2 == floop(v, w)
    @test g1 == g2 == (NoTangent(), w, NoTangent())

    # The constant slot reconstructs to `NoTangent()`, not a zero array.
    @test g1[3] === NoTangent()
end

@testset "activity: matches the full gradient's corresponding slot" begin
    # Whatever is differentiated must agree with the all-active run, for every activity signature.
    h(a, b, c) = a * b + b * c + sin(a * c)
    args = (1.5, 2.5, 0.5)
    full = rev_gradient(h, args...)
    fcd = zero_fcodual(h)
    for inactive in ((1,), (2,), (3,), (1, 2), (1, 3), (2, 3))
        cds = ntuple(k -> k in inactive ? const_codual(args[k]) : CoDual(args[k], NoFData()), 3)
        got = rrule!!(fcd, Ctx(), cds...)[2](1.0)
        @test got[1] === NoRData()
        for k in 1:3
            if k in inactive
                @test got[k + 1] === NoRData()
            else
                @test got[k + 1] ≈ full[k + 1]
            end
        end
        checkverify_rev(h, (Float64, Float64, Float64); inactive)
    end
end

@testset "activity: loops run more than twice" begin
    # A control-flow-replay change can be correct for 0 and 1 iterations and still corrupt gradients
    # from the second onward, so exercise a genuinely repeated ambiguous edge.
    function powloop(x, n)
        s = 1.0
        for _ in 1:n
            s = s * x + 1.0
        end
        return s
    end
    for n in 0:4
        exact = central_diff(t -> powloop(t, n), 1.7)
        got = rrule!!(zero_fcodual(powloop), Ctx(),
                      CoDual(1.7, NoFData()), const_codual(n))[2](1.0)
        @test got[2] ≈ exact rtol = 1e-5
        @test got[3] === NoRData()
    end
    checkverify_rev(powloop, (Float64, Int); inactive=(2,))
end

@testset "activity: constant-only code is replayed, not differentiated" begin
    # `_static_recursible_call`'s gates (concrete argtypes, traceable provenance, resolvable callee)
    # never run for a call reached only through constants — the whole point of the bypass. This
    # primal bails outright when `d` is active.
    lookup(d, k) = d[k]
    f(x, d, k) = x * lookup(d, k)
    d = Dict("a" => 3.0, "b" => 4.0)

    # Treating the dictionary as differentiable fails — here already at `zero_fcodual`, which cannot
    # build a shadow for a `Dict` at all. Asserted only as the contrast to the constant case below.
    @test_throws Exception rrule!!(zero_fcodual(f), Ctx(), CoDual(2.0, NoFData()),
                                   zero_fcodual(d), zero_fcodual("a"))

    got = rrule!!(zero_fcodual(f), Ctx(), CoDual(2.0, NoFData()),
                  const_codual(d), const_codual("a"))
    @test primal(got[1]) == f(2.0, d, "a")
    @test got[2](1.0) == (NoRData(), 3.0, NoRData(), NoRData())
end

@testset "activity: shrinks the tape" begin
    # Correctness tests cannot catch this, and it is the whole point of the feature: a constant
    # argument must remove real per-iteration comms traffic, not just zero out a result.
    function floop(v, w)
        s = 0.0
        for i in eachindex(v)
            s += v[i] * w[i]
        end
        return s
    end
    at = (Vector{Float64}, Vector{Float64})
    full = comms_element_types(tape_type(floop, at))
    cut = comms_element_types(tape_type(floop, at; inactive=(2,)))
    @test sum(fieldcount, full) > sum(fieldcount, cut)
end

@testset "activity: mutating an inactive object with an active value bails" begin
    # Unsound to silently drop: the constant array has no shadow to accumulate into. Must be a
    # located error rather than a wrong number.
    function writeinto!(buf, x)
        buf[1] = x * 2.0
        return sum(buf)
    end
    err = try
        rrule!!(zero_fcodual(writeinto!), Ctx(), const_codual([1.0, 2.0]), CoDual(3.0, NoFData()))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
end

@testset "activity: build_ctx rejects an out-of-range position" begin
    @test_throws ArgumentError build_ctx((x, y) -> x * y, (Float64, Float64); inactive=(3,))
    @test_throws ArgumentError build_ctx((x, y) -> x * y, (Float64, Float64); inactive=(0,))
end

@testset "activity: a constant vararg-tail element bails instead of miscompiling" begin
    # A vararg primal's trailing arguments all land in one packed tail slot, always typed as the
    # active fdata carrier. A constant trailing argument used to sneak a `NoTangent()` value into that
    # slot and produce a `TypeError`; it must now be a located bail.
    vasum(vs...) = sum(vs[1])
    v = [1.0, 2.0, 3.0]

    err = try
        rrule!!(zero_fcodual(vasum), Ctx(), const_codual(v))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("cannot hold a vararg primal's trailing argument constant", err.msg)

    err2 = try
        build_ctx(vasum, (Vector{Float64},); inactive=(1,))
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("cannot hold a vararg primal's trailing argument constant", err2.msg)

    # The active vararg case must still work: not an over-broad bail.
    dv = zeros(3)
    _, pb = rrule!!(zero_fcodual(vasum), Ctx(), CoDual(v, dv))
    @test pb(1.0) == (NoRData(), NoRData())
    @test dv == ones(3)
end

# An inactive source is handled by `memmove`/`memcpy` and by `Core.tuple`/`Core.setfield!`/
# `Base.memoryrefset!`, all gated on `ctx.inactive` rather than `_bi_tracked`.
#
# `sum(v .* w)` itself still bails: `Broadcast.unalias` lowers to a `PhiNode` merging the inactive
# argument with the copied buffer, and `_fdata_tracked`'s `PhiNode` arm has no inactive case. These
# tests reach the four fixed sites without routing through that phi.

@testset "activity: memmove third mode — destination tracked, source inactive" begin
    f(v, w) = sum(v) + sum(copy(w))
    v, w = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    dv = zeros(3)
    y, pb = rrule!!(zero_fcodual(f), Ctx(), CoDual(copy(v), dv), const_codual(w))
    @test primal(y) == f(v, w)
    @test pb(1.0) == (NoRData(), NoRData(), NoRData())
    @test dv == ones(3)   # the constant `w`'s copy contributes nothing; `v`'s own gradient is unaffected

    checkverify_rev(f, (Vector{Float64}, Vector{Float64}); inactive=(2,))
    check_stack_balance(f, copy(v), copy(w))

    # Tape shrink: dropping the source `:fshadow` item is required, not incidental.
    full = comms_element_types(tape_type(f, (Vector{Float64}, Vector{Float64})))
    cut = comms_element_types(tape_type(f, (Vector{Float64}, Vector{Float64}); inactive=(2,)))
    @test sum(fieldcount, full) > sum(fieldcount, cut)

    # Mode A (both-`NoTangent`) gains its first direct test: unaffected by the third mode.
    fA(x) = sum(copy(x))
    y2, pb2 = rrule!!(zero_fcodual(fA), Ctx(), CoDual([1, 2, 3], NoTangent()))
    @test primal(y2) == 6
    @test pb2(1.0) == (NoRData(), NoRData())
    checkverify_rev(fA, (Vector{Int},))

    # An "active but untraceable source must still bail at memmove specifically" case was attempted
    # and dropped, not silently omitted: every construction tried either (a) got optimized away before
    # reaching an untracked state, (b) hit an unrelated bail first (`Vector{Any}`'s element type kills
    # `copy`'s own inlining into a foreigncall before memmove dispatch ever runs; `Core.typeassert` has
    # no reverse rule at all, so it bails at its own point of use before reaching `copy`), or (c) turned
    # out to be *order-dependent* (`Base.inferencebarrier` + a typeassert reaches memmove's bail in
    # isolation, but a different, unrelated bail once other code in the same file has already
    # differentiated through `Base.copy`/`Vector{Float64}` for a different activity signature — an
    # inlining-cost-model artifact, not a property of the fix, and not a reliable regression test).
    # `Vector{Vector{Float64}}` indexing (the obvious "read out of a container" candidate) turned out
    # to be already tracked (the nested-array-read chain), not untracked, so it isn't this case either.
    # The general principle — the gate is an activity test, not a trackedness test — does get a stable,
    # reproducible demonstration below, via `Core.tuple`'s identical gate and the real (documented,
    # structural) phi-merge gap: see "broadcast through a constant array still bails" below.
end

@testset "activity: DI.Constant round trip through memmove's third mode" begin
    f(x, w) = sum(x) + sum(copy(w))
    x, w = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    g = DI.gradient(f, AutoDifferReverse(), x, DI.Constant(w))
    @test g ≈ ones(3)
    @test g ≈ rev_gradient(f, x, w)[2]   # agrees with `w` treated as an ordinary active argument

    # `DI.Cache` stays active — not made inactive by Part 2's `_ctx_codual`.
    fcache(x, c) = sum(x .* c)
    gc = DI.gradient(fcache, AutoDifferReverse(), x, DI.Cache(copy(w)))
    @test gc ≈ w
end

@testset "activity: Core.tuple synthesises a zero for an inactive operand" begin
    # `(v, w)` survives optimization as a literal `Core.tuple` when returned directly (SROA otherwise
    # eliminates an immediately-destructured tuple, which would test nothing).
    tuple_pair(v, w) = (v, w)
    v, w = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    dv = zeros(3)
    y, pb = rrule!!(zero_fcodual(tuple_pair), Ctx(), CoDual(copy(v), dv), const_codual(w))
    @test primal(y) == (v, w)
    # The inactive slot's shadow is a real (synthesised), unaliased zero array — not a crash, not the
    # active slot's own shadow.
    @test tangent(y)[1] === dv
    @test tangent(y)[2] == zeros(3) && tangent(y)[2] !== dv

    increment!!(tangent(y), (ones(3), ones(3)))
    @test dv == ones(3)   # flows through the aliased slot
    @test pb(NoRData()) == (NoRData(), NoRData(), NoRData())
    @test dv == ones(3)   # the inactive slot's increment went nowhere — no leak, no crash

    checkverify_rev(tuple_pair, (Vector{Float64}, Vector{Float64}); inactive=(2,))
    check_stack_balance(tuple_pair, copy(v), copy(w); seed=NoRData())
end

@testset "activity: setfield!/memoryrefset! zero the destination shadow, not skip it" begin
    # Adversarial staleness check: give the destination slot a real, non-zero (aliased) shadow first,
    # then overwrite with a constant, then read again. A "skip the check" implementation would leave
    # the stale alias in place and let the second read's contribution leak into the first value's
    # gradient — 4*ones(3) instead of ones(3). "Zero the shadow" severs it correctly.
    function stale_element(active_arr::Vector{Float64}, const_arr::Vector{Float64})
        buf = Vector{Vector{Float64}}(undef, 1)
        buf[1] = active_arr
        s1 = sum(buf[1])
        buf[1] = const_arr
        s2 = sum(buf[1])
        return s1 + 3 * s2
    end
    av, cv = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    dav = zeros(3)
    y, pb = rrule!!(zero_fcodual(stale_element), Ctx(), CoDual(copy(av), dav), const_codual(cv))
    @test primal(y) == sum(av) + 3 * sum(cv)
    @test pb(1.0) == (NoRData(), NoRData(), NoRData())
    @test dav == ones(3)
    checkverify_rev(stale_element, (Vector{Float64}, Vector{Float64}); inactive=(2,))
    check_stack_balance(stale_element, copy(av), copy(cv))

    mutable struct StaleBox113
        v::Vector{Float64}
    end
    function stale_field(active_arr::Vector{Float64}, const_arr::Vector{Float64})
        b = StaleBox113(zeros(length(active_arr)))
        b.v = active_arr
        s1 = sum(b.v)
        b.v = const_arr
        s2 = sum(b.v)
        return s1 + 3 * s2
    end
    dav2 = zeros(3)
    y2, pb2 = rrule!!(zero_fcodual(stale_field), Ctx(), CoDual(copy(av), dav2), const_codual(cv))
    @test primal(y2) == sum(av) + 3 * sum(cv)
    @test pb2(1.0) == (NoRData(), NoRData(), NoRData())
    @test dav2 == ones(3)
    checkverify_rev(stale_field, (Vector{Float64}, Vector{Float64}); inactive=(2,))
    check_stack_balance(stale_field, copy(av), copy(cv))
end

@testset "activity: broadcast through a constant array still bails (phi gap, not yet fixed)" begin
    # This is the real-world instance of verification point 3 (an active-but-untraceable operand must
    # still bail, gated on activity not trackedness) — no synthetic construction needed. `.`-broadcast's
    # `Broadcast.unalias` builds a `PhiNode` merging one inactive edge (the untouched constant argument)
    # with one active, tracked edge (the memmove-copied buffer); the merged value is therefore ACTIVE
    # (any tracked incoming edge makes a phi active) but `_fdata_tracked`'s `PhiNode` arm still requires
    # *every* edge tracked, so it comes out untracked — exactly the shape this whole feature has to keep
    # bailing on. Had the fix at any of the four sites relaxed on trackedness instead of activity, a
    # case shaped like this one would have silently gone through with a wrong (too-small) gradient
    # instead of raising this error.
    #
    # Regression pin, not a design choice: if this starts passing, `_fdata_tracked`'s `PhiNode` arm
    # (and the fwds carrier's phi-shadow-merge codegen) has been taught about `ctx.inactive` — a
    # deliberate future fix, not a regression. Revisit this test (and its framing above) when that
    # lands; it is not meant to hold as a permanent invariant.
    f(v, w) = sum(v .* w)
    err = try
        rrule!!(zero_fcodual(f), Ctx(), CoDual([1.0, 2.0, 3.0], zeros(3)), const_codual([4.0, 5.0, 6.0]))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
end
