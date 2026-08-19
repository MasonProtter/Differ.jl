using Test
using DifferReverse
using DifferReverse: rev_gradient, value_and_gradient!, increment!!,
                     _intrinsic_needed_operands, intrinsic_rrule_deps,
                     code_reverse_fwds_ircode, rrule!!
using LinearAlgebra: dot
import DifferentiationInterface as DI
include("testutils.jl")

# Declaring an argument constant: `Inactive` in a `CoDual`'s shadow slot. Activity propagates from
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
    @test tangent(ycd) === Inactive()
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
    @test g1 == g2 == (NoTangent(), w, Inactive())

    # The constant slot reconstructs to `Inactive()` — distinct both from a zero array and from
    # the `NoTangent()` a genuinely non-differentiable argument would give.
    @test g1[3] === Inactive()
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

@testset "activity: build_ctx is inferable and steady-state zero-allocation" begin
    # The point of preallocation is a concrete `Ctx{Tape{…}}` at the call site: an abstract `Ctx`
    # makes the adjacent `rrule!!` dispatch dynamically, boxing its argument tuple and its return
    # every call (ISSUES #124). `_inactive_positions` must stay allocation-free, or a literal
    # `inactive` stops const-folding into the `Val` type parameter and this regresses.
    f(x, y) = x * y + sin(x)
    function plain_ctx()
        build_ctx(f, (Float64, Float64))
    end
    function lit_ctx()
        build_ctx(f, (Float64, Float64); inactive=(2,))
    end
    @test isconcretetype(only(Base.return_types(plain_ctx, ())))
    @test isconcretetype(only(Base.return_types(lit_ctx, ())))

    # Steady-state allocation, in the shape the benchmarks use: the `ctx` the setup block builds
    # reaches the timed `rrule!!`/pullback call concretely typed (the closure's capture type is
    # where the type travels), so a warmed round trip allocates nothing.
    ctx = build_ctx(f, (Float64, Float64); inactive=(2,))
    @test ctx isa Ctx{<:DifferReverse.Tape}
    xcd = CoDual(2.0, NoFData())
    ycd = const_codual(3.0)
    measure = () -> (rrule!!(zero_fcodual(f), ctx, xcd, ycd)[2])(1.0)
    measure(); measure()
    @test @allocated(measure()) == 0
end

@testset "activity: the carrier-type and carrier-value build_ctx forms" begin
    # `Tuple{CoDual…}` states everything the tape shape depends on, activity included, as a type
    # parameter, so the result type is a function of that tuple with no const-folding involved.
    # The carrier-value form is the same thing spelled with the carriers themselves.
    f(x, y) = x * y + sin(x)
    fcd = zero_fcodual(f)
    xcd = CoDual(2.0, NoFData())
    ycd = const_codual(3.0)

    ctx_ty = build_ctx(Tuple{typeof(fcd),typeof(xcd),typeof(ycd)})
    ctx_val = build_ctx(fcd, xcd, ycd)
    ctx_pos = build_ctx(f, (Float64, Float64); inactive=(2,))
    # All three describe the same call, so they must land on the same tape.
    @test typeof(ctx_ty) === typeof(ctx_val) === typeof(ctx_pos)
    @test ctx_ty isa Ctx{<:DifferReverse.Tape}

    for ctx in (ctx_ty, ctx_val)
        _, pb = rrule!!(fcd, ctx, xcd, ycd)
        @test pb(1.0) == (NoRData(), 3.0 + cos(2.0), NoRData())
    end

    # Inferable without const-folding: the carriers are locals, so `typeof` is exact.
    function via_type()
        g = (x, y) -> x * y
        build_ctx(Tuple{typeof(zero_fcodual(g)),CoDual{Float64,NoFData},CoDual{Float64,Inactive}})
    end
    function via_values()
        g = (x, y) -> x * y
        build_ctx(zero_fcodual(g), CoDual(2.0, NoFData()), const_codual(3.0))
    end
    @test isconcretetype(only(Base.return_types(via_type, ())))
    @test isconcretetype(only(Base.return_types(via_values, ())))
end

@testset "activity: returning an argument declared constant" begin
    # The returned carrier's type is chosen from the shadow type, so its *value* has to be built to
    # match. A `NoFData` primal used to take the `zero_fcodual` branch, which builds the
    # primal-derived shadow (`NoFData()`) and blew up with a `TypeError` at the `%new`. `verify_ir`
    # passes either way — it does not check declared-vs-actual statement types — so nothing in
    # `checkverify_rev` catches this shape.
    ret_const(x, c) = c
    ctx = build_ctx(ret_const, (Float64, Float64); inactive=(2,))
    ycd, pb = rrule!!(zero_fcodual(ret_const), ctx, zero_fcodual(1.0), const_codual(2.0))
    @test primal(ycd) == 2.0
    @test tangent(ycd) === Inactive()
    @test pb(1.0) == (NoRData(), 0.0, NoRData())

    # The fdata-carrying case routes differently and was already correct; pin it so the two stay
    # in agreement.
    ctx_v = build_ctx(ret_const, (Vector{Float64}, Vector{Float64}); inactive=(2,))
    yv, pbv = rrule!!(zero_fcodual(ret_const), ctx_v,
                      CoDual([1.0, 2.0], zeros(2)), const_codual([3.0, 4.0]))
    @test primal(yv) == [3.0, 4.0]
    @test tangent(yv) === Inactive()

    # An active return alongside an inactive argument keeps the ordinary carrier.
    ret_active(x, c) = x
    ctx_a = build_ctx(ret_active, (Float64, Float64); inactive=(2,))
    ya, pba = rrule!!(zero_fcodual(ret_active), ctx_a, zero_fcodual(5.0), const_codual(2.0))
    @test tangent(ya) === NoFData()
    @test pba(1.0) == (NoRData(), 1.0, NoRData())

    checkverify_rev(ret_const, (Float64, Float64); inactive=(2,))
end

@testset "activity: inactive= accepts an Int or a tuple of them, and nothing else" begin
    # The positions become a `Val` type parameter, so only what is constructible as one is allowed.
    # A `Vector` used to surface as a `TypeError` from `Val` itself, and a range silently produced a
    # tape-less `Ctx{Nothing}` — a pre-allocated context degrading to a per-call allocating one with
    # no error. Both are now refused at the signature, naming the keyword.
    f(x, y) = x * y + sin(x)
    @test typeof(build_ctx(f, (Float64, Float64); inactive=2)) ===
          typeof(build_ctx(f, (Float64, Float64); inactive=(2,)))
    @test_throws TypeError build_ctx(f, (Float64, Float64); inactive=[2])
    @test_throws TypeError build_ctx(f, (Float64, Float64); inactive=2:2)
    # Still inferable in every accepted spelling.
    c_none() = build_ctx(f, (Float64, Float64))
    c_int() = build_ctx(f, (Float64, Float64); inactive=2)
    c_tup() = build_ctx(f, (Float64, Float64); inactive=(2,))
    for c in (c_none, c_int, c_tup)
        @test isconcretetype(only(Base.return_types(c, ())))
    end
end

@testset "activity: build_ctx rejects a malformed carrier tuple" begin
    # A `UnionAll` element is not a usable carrier: it names no shadow, so no tape shape follows
    # from it. Fail rather than guess one.
    fcd = zero_fcodual((x, y) -> x * y)
    @test_throws ErrorException build_ctx(Tuple{typeof(fcd),CoDual{Float64},CoDual{Float64,NoFData}})
    @test_throws ErrorException build_ctx(Tuple{})
end

@testset "activity: a non-constant inactive still differentiates correctly at run time" begin
    # Positions the compiler cannot fold (a generator result): the tape is still built correctly at
    # run time — the generator reads the positions off the `Val`'s runtime type — so the context
    # works; it is just not inferable, and the adjacent `rrule!!` dispatches dynamically.
    f(x, y) = x * y + sin(x)
    inact = Tuple(j for j in 1:2 if j == 2)
    ctx = build_ctx(f, (Float64, Float64); inactive=inact)
    @test ctx isa Ctx{<:DifferReverse.Tape}
    _, pb = rrule!!(zero_fcodual(f), ctx, CoDual(2.0, NoFData()), const_codual(3.0))
    @test pb(1.0) == (NoRData(), 3.0 + cos(2.0), NoRData())
end

@testset "activity: a constant vararg-tail element bails instead of miscompiling" begin
    # A vararg primal's trailing arguments all land in one packed tail slot, always typed as the
    # active fdata carrier. A constant trailing argument used to sneak an `Inactive()` value into that
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

    # Mode A (both-`NoTangent`) gains its first direct test: unaffected by the third mode. Note
    # the *ordinary active* carrier — `Vector{Int}` reaches mode A because its element type has no
    # tangent space, which is a different claim from declaring the argument constant.
    fA(x) = sum(copy(x))
    y2, pb2 = rrule!!(zero_fcodual(fA), Ctx(), zero_fcodual([1, 2, 3]))
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

@testset "activity: Core.tuple builds no shadow for an inactive operand" begin
    # `(v, w)` survives optimization as a literal `Core.tuple` when returned directly (SROA otherwise
    # eliminates an immediately-destructured tuple, which would test nothing).
    tuple_pair(v, w) = (v, w)
    v, w = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    dv = zeros(3)
    y, pb = rrule!!(zero_fcodual(tuple_pair), Ctx(), CoDual(copy(v), dv), const_codual(w))
    @test primal(y) == (v, w)
    # The inactive slot holds `Inactive()`, not a synthesised zero: the shadow tuple's type is
    # narrower than the primal-derived one, so there is no slot to allocate into.
    @test tangent(y)[1] === dv
    @test tangent(y)[2] === Inactive()
    @test typeof(tangent(y)) === Tuple{Vector{Float64},Inactive}

    # Seeding still works even though the caller builds its seed from the primal type and it is
    # therefore wider than the shadow: accumulation is structural, and the inactive slot discards.
    increment!!(tangent(y), (ones(3), ones(3)))
    @test dv == ones(3)   # flows through the aliased slot
    @test pb(NoRData()) == (NoRData(), NoRData(), NoRData())
    @test dv == ones(3)   # the inactive slot's increment went nowhere — no leak, no crash

    # Gone from the carrier, not merely unused.
    ir = code_reverse_fwds_ircode(tuple_pair, (Vector{Float64}, Vector{Float64}); inactive=(2,))[1]
    @test !occursin("_rr_zero_fdata", string(ir))

    # Destructuring a mixed tuple reads the constant slot back out as `Inactive`, so a consumer
    # downstream of the aggregate still types correctly.
    destructure(a, b) = (t = (a, b); sum(t[1]) + sum(t[2]))
    @test rev_gradient(destructure, [1.0, 2.0], [3.0, 4.0]) == (NoTangent(), ones(2), ones(2))
    checkverify_rev(destructure, (Vector{Float64}, Vector{Float64}); inactive=(2,))

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

@testset "activity: broadcast through a constant array" begin
    # `.`-broadcast's `Broadcast.unalias` builds a `PhiNode` merging one inactive edge (the untouched
    # constant argument) with one active, tracked edge (the memmove-copied buffer). The merge is
    # active — any active edge makes it so — and normalises to its own primal-derived shadow type, so
    # the inactive arm is served by a zero hoisted into the entry block rather than by a shadow that
    # was never built.
    f(v, w) = sum(v .* w)
    v, w = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    full = rev_gradient(f, v, w)

    for inact in (1, 2)
        ctx = build_ctx(f, (Vector{Float64}, Vector{Float64}); inactive=(inact,))
        d1 = inact == 1 ? Inactive() : zeros(3)
        d2 = inact == 2 ? Inactive() : zeros(3)
        y, g = value_and_gradient!(ctx, zero_fcodual(f), CoDual(v, d1), CoDual(w, d2))
        @test y == f(v, w)
        @test g[1] === NoTangent()
        # Whatever is differentiated agrees with the all-active run; the constant slot is `Inactive`.
        @test g[inact + 1] === Inactive()
        other = inact == 1 ? 3 : 2
        @test g[other] ≈ full[other]
        checkverify_rev(f, (Vector{Float64}, Vector{Float64}); inactive=(inact,))
    end

    # A loop-carried merge, run well past 2 iterations: a per-edge change to the phi has to survive
    # the block-stack replay, which N=0/1 would not exercise (see ISSUES #52).
    function floop(v, w)
        s = 0.0
        for k in 1:4
            s += sum(v .* w) * k
        end
        return s
    end
    ctx = build_ctx(floop, (Vector{Float64}, Vector{Float64}); inactive=(2,))
    y, g = value_and_gradient!(ctx, zero_fcodual(floop), CoDual(v, zeros(3)), CoDual(w, Inactive()))
    @test y == floop(v, w)
    @test g[2] ≈ rev_gradient(floop, v, w)[2]
    @test g[3] === Inactive()
    checkverify_rev(floop, (Vector{Float64}, Vector{Float64}); inactive=(2,))
end

# Intrinsic operand *primal recording* is per-contribution: an operand whose rdata contribution has
# no sink is not pushed onto the tape.

@testset "activity: mul_float operand recording is per-contribution" begin
    f(x, y) = x * y
    checkverify_rev(f, (Float64, Float64); inactive=(2,))
    x, y = 2.0, 3.0
    _, pb = rrule!!(zero_fcodual(f), Ctx(), CoDual(x, NoFData()), const_codual(y))
    @test pb(1.0) == (NoRData(), y, NoRData())

    # Loop form: `x`'s per-iteration value is an SSA (a phi), not the bare `Argument` that
    # `elide_argument_primal` already elides regardless of activity — this is where the tape shrink
    # is actually observable (the plain two-argument form above never taped either operand's primal
    # to begin with).
    function floop(x, y)
        acc = 0.0
        for _ in 1:3
            acc += x * y
            x += 1.0
        end
        return acc
    end
    checkverify_rev(floop, (Float64, Float64); inactive=(2,))
    full = comms_element_types(tape_type(floop, (Float64, Float64)))
    cut = comms_element_types(tape_type(floop, (Float64, Float64); inactive=(2,)))
    @test full == [Tuple{Float64}]   # `x`'s primal, recorded once per iteration
    @test isempty(cut)                # `y` inactive discards `db`; `da` only needs `y` (free, an argument)

    _, pbloop = rrule!!(zero_fcodual(floop), Ctx(), CoDual(x, NoFData()), const_codual(y))
    @test pbloop(1.0) == (NoRData(), 3y, NoRData())
end

@testset "activity: multiply-by-literal records nothing" begin
    # `x * 2.0` inside a loop, so a per-iteration slot would otherwise be pushed. `da` (routed to
    # `x`) only reads the literal (free); `db` (routed to the literal) is discarded outright since a
    # literal has no rdata sink — so neither contribution ever needs `x`'s own primal.
    function g(x)
        acc = 0.0
        for _ in 1:3
            acc += 3.0 * x
            x += 1.0
        end
        return acc
    end
    checkverify_rev(g, (Float64,))
    ts = comms_element_types(tape_type(g, (Float64,)))
    @test isempty(ts)

    _, pb = rrule!!(zero_fcodual(g), Ctx(), CoDual(2.0, 0.0))
    @test pb(1.0) == (NoRData(), 9.0)
end

@testset "activity: div_float's crossed dependency" begin
    # `da` (routed to the numerator) reads only the denominator; `db` (routed to the denominator)
    # reads both. So an inactive numerator still needs the denominator's primal recorded (already
    # true) *and* keeps the numerator's own primal recorded too, since `db` (wanted, `b` active)
    # reads it — while an inactive denominator drops the numerator's primal entirely, since only
    # `da` stays wanted and `da` never reads it. This is exactly the asymmetry a flat "positions
    # read" declaration could not express.
    function divloop(a, b)
        acc = 0.0
        for _ in 1:3
            acc += a / b
            a += 1.0
        end
        return acc
    end
    full = comms_element_types(tape_type(divloop, (Float64, Float64)))
    num_inactive = comms_element_types(tape_type(divloop, (Float64, Float64); inactive=(1,)))
    den_inactive = comms_element_types(tape_type(divloop, (Float64, Float64); inactive=(2,)))
    @test full == [Tuple{Float64}]      # `a`'s primal; `b` is a bare `Argument`, always elided
    @test num_inactive == full          # keeps both — `db` (wanted) still reads `a`
    @test isempty(den_inactive)         # drops `a` — only `da` (wanted) survives, and it needs only `b`

    checkverify_rev(divloop, (Float64, Float64); inactive=(1,))
    checkverify_rev(divloop, (Float64, Float64); inactive=(2,))

    a0, b0 = 2.0, 5.0
    _, pb_num = rrule!!(zero_fcodual(divloop), Ctx(), const_codual(a0), CoDual(b0, 0.0))
    r_num = pb_num(1.0)
    @test r_num[1] == NoRData() && r_num[2] == NoRData()
    @test r_num[3] ≈ -(3a0 + 3) / b0^2

    _, pb_den = rrule!!(zero_fcodual(divloop), Ctx(), CoDual(a0, 0.0), const_codual(b0))
    r_den = pb_den(1.0)
    @test r_den[1] == NoRData() && r_den[3] == NoRData()
    @test r_den[2] ≈ 3 / b0
end

@testset "activity: _intrinsic_needed_operands pins the deps table" begin
    needed(f, nops, wanted) = _intrinsic_needed_operands(f, nops, wanted)
    mul = Core.Intrinsics.mul_float
    dv = Core.Intrinsics.div_float
    fma = Core.Intrinsics.fma_float

    # mul_float: crossed dependency — `da` (contribution 1, routed to operand 1) reads operand 2,
    # `db` (contribution 2, routed to operand 2) reads operand 1.
    @test needed(mul, 2, j -> true) == BitSet([1, 2])
    @test needed(mul, 2, j -> j == 1) == BitSet([2])
    @test needed(mul, 2, j -> j == 2) == BitSet([1])
    @test needed(mul, 2, j -> false) == BitSet()

    # div_float: asymmetric — `da` needs only the denominator, `db` needs both.
    @test needed(dv, 2, j -> true) == BitSet([1, 2])
    @test needed(dv, 2, j -> j == 1) == BitSet([2])
    @test needed(dv, 2, j -> j == 2) == BitSet([1, 2])
    @test needed(dv, 2, j -> false) == BitSet()

    # fma_float: `da` needs `b`, `db` needs `a`, `dc` needs neither.
    @test needed(fma, 3, j -> true) == BitSet([1, 2])
    @test needed(fma, 3, j -> j == 1) == BitSet([2])
    @test needed(fma, 3, j -> j == 2) == BitSet([1])
    @test needed(fma, 3, j -> j == 3) == BitSet()

    # No declaration, or an arity mismatch against the declared table: conservative `nothing`.
    @test intrinsic_rrule_deps(Val(Core.Intrinsics.not_int)) === nothing
    @test needed(Core.Intrinsics.not_int, 1, j -> true) === nothing
    @test needed(mul, 3, j -> true) === nothing
end

# A nested call into a hand-ruled function with a constant argument keeps the closed-form rule: the
# engine passes a `CoDual{P,Inactive}` through instead of bailing on untraceable provenance.

@testset "activity: nested hand rule with a constant argument" begin
    f(v, w) = dot(v, w)
    v, w = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    dv = zeros(3)
    y, dvout = value_and_gradient!(build_ctx(f, (Vector{Float64}, Vector{Float64}); inactive=(2,)),
                                    zero_fcodual(f), CoDual(v, dv), const_codual(w))
    @test y == dot(v, w)
    @test dvout == (NoTangent(), w, Inactive())
    @test dv == w
    checkverify_rev(f, (Vector{Float64}, Vector{Float64}); inactive=(2,))

    # Without the assertion below, the test above passes for the wrong reason — silently via the
    # derived path instead of the hand rule. Checked on the pre-inlining IR (`build_reverse_fwds_ir`,
    # not `code_reverse_fwds_ircode`): a hand rule's own body is small enough that ordinary inlining
    # now absorbs it, so its `:invoke rrule!!` only survives before `run_ipo_passes!` runs.
    interp = DifferReverse.build_reverse_interp()
    codualtys = Any[DifferReverse.fcodual_type(DifferReverse._typeof(f)),
                    arg_codual_types(f, (Vector{Float64}, Vector{Float64}); inactive=(2,))...]
    impl_tt = Tuple{typeof(DifferReverse.reverse_fwds_impl), codualtys[1], Ctx{Nothing}, codualtys[2:end]...}
    match, _ = Core.Compiler.findsup(impl_tt, Core.Compiler.method_table(interp))
    impl_mi = Base.specialize_method(match.method, match.spec_types, match.sparams)
    ir = DifferReverse.build_reverse_fwds_ir(interp, impl_mi)
    invokes_to_rrule = [stmt for stmt in ir.stmts.stmt
                        if isa(stmt, Expr) && stmt.head === :invoke &&
                           length(stmt.args) >= 2 && stmt.args[2] === rrule!!]
    @test length(invokes_to_rrule) == 1
end

# `reverse_pullback_recursive_ci`'s arity/slot-type check: a wrong-arity hand pullback must produce
# a located bail rather than a `getfield` error inside generated IR. The rule is defined here (before
# anything differentiates `badarity_f`), so the world-age caveat around later rule additions doesn't
# apply.

badarity_f(x, y) = x + y

function DifferReverse.rrule!!(::CoDual{typeof(badarity_f),NoFData}, ::AbstractCtx,
                               xcd::CoDual{Float64,NoFData}, ycd::CoDual{Float64,NoFData})
    z = primal(xcd) + primal(ycd)
    badarity_pullback(dz) = (NoRData(), dz)   # wrong: should be a 3-tuple (own rdata + 2 arguments)
    return CoDual(z, NoFData()), badarity_pullback
end

@testset "activity: recursive-pullback arity check catches a wrong-length hand pullback" begin
    outer_badarity(x, y) = badarity_f(x, y) + 1.0
    err = try
        rev_gradient(outer_badarity, 1.0, 2.0)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    # Pins that the arity check is what fired, not some unrelated error.
    @test occursin("3-element tuple of rdatas", err.msg)
end
