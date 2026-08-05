using Test
using Differ
using Differ: gradient, MutableTangent, rdata_type, tangent_type, NoTangent
using Differ: zero_rdata_from_type, zero_like_rdata_from_type, CannotProduceZeroRDataFromType
using Differ: ZeroRData, NoRData, increment!!

include(joinpath(@__DIR__, "testutils.jl"))

@testset "reverse mode: read-only array indexing" begin
    # Tier 6 Part 2: read-only array indexing.
    function arr_idx3(x::Vector{Float64})                # a fixed-index read
        return x[3]
    end
    function arr_idx_branch(x::Vector{Float64}, pick::Bool)   # index chosen by a branch
        return pick ? x[1] : x[2]
    end
    function arr_sum(x::Vector{Float64})                  # hand-written summation loop
        s = 0.0
        for i in 1:length(x)
            s += x[i]
        end
        return s
    end

    x4 = [1.0, 2.0, 3.0, 4.0]
    _, dx_i3 = gradient(arr_idx3, x4)
    @test dx_i3 == [0.0, 0.0, 1.0, 0.0]

    x2 = [1.0, 2.0]
    _, dx_ib_t, dp_t = gradient(arr_idx_branch, x2, true)
    @test dx_ib_t == [1.0, 0.0]
    @test dp_t === NoTangent()
    _, dx_ib_f, = gradient(arr_idx_branch, x2, false)
    @test dx_ib_f == [0.0, 1.0]

    x3 = [1.0, 2.0, 3.0]
    _, dx_sum = gradient(arr_sum, x3)
    @test dx_sum == ones(3)
    # Cross-check every element individually against central differences.
    for k in eachindex(x3)
        xp = copy(x3); xp[k] += 1e-6
        xm = copy(x3); xm[k] -= 1e-6
        @test dx_sum[k] ≈ (arr_sum(xp) - arr_sum(xm)) / 2e-6 rtol = 1e-5
    end

    checkverify_rev(arr_idx3, (Vector{Float64},))
    checkverify_rev(arr_idx_branch, (Vector{Float64}, Bool))
    checkverify_rev(arr_sum, (Vector{Float64},))
    check_stack_balance(arr_sum, [1.0, 2.0, 3.0])

    # `_collapsible_regions`: a single fixed-index read is nothing but a `@boundscheck` diamond, so
    # it collapses to zero block-stack traffic. `arr_idx_branch`'s merge point has a real `PhiNode`
    # (it selects between two *different* values, `x[1]` vs `x[2]`) and `arr_sum`'s loop has a real
    # back-edge — both must keep paying block-stack traffic, and this locks that in as a regression
    # guard against the optimization over-firing.
    check_block_stack_traffic(arr_idx3, x4; expect_zero=true)
    check_block_stack_traffic(arr_idx_branch, x2, true; expect_zero=false)
    check_block_stack_traffic(arr_sum, x3; expect_zero=false)
end

@testset "reverse mode: array mutation (memoryrefset!)" begin
    # Array mutation (Part 3, `memoryrefset!`): `arr_mutate!(x) = (x[1] = 2*x[1]; x[1])` — the
    # returned value only ever sees the *overwritten* x[1], so d/dx = [2.0, 0.0].
    function arr_mutate!(x::Vector{Float64})
        x[1] = 2.0 * x[1]
        return x[1]
    end

    _, dx_mut = gradient(arr_mutate!, [1.0, 2.0])
    @test dx_mut == [2.0, 0.0]
    h = 1e-6
    xp = [1.0 + h, 2.0]; xm = [1.0 - h, 2.0]
    @test dx_mut[1] ≈ (arr_mutate!(xp) - arr_mutate!(xm)) / 2h rtol = 1e-5

    checkverify_rev(arr_mutate!, (Vector{Float64},))
    check_stack_balance(arr_mutate!, [1.0, 2.0])
    check_block_stack_traffic(arr_mutate!, [1.0, 2.0]; expect_zero=true)
end

@testset "reverse mode: collapsible @boundscheck regions" begin
    # The motivating case: a genuinely straight-line primal (no real branch or loop, just plain
    # array reads/writes) whose `@boundscheck` diamonds — normally CFG-ambiguous, since `merge` has
    # two real static predecessors, the direct "skip the check" edge and the checked "pass" edge —
    # should all collapse away, per `_collapsible_regions` (`src/reverse_interp.jl`).
    function straightline!(v::Vector{Float64}, a::Float64)
        v[1] = a
        v[2] = 2a
        return v[1] + v[2]
    end

    v0 = [0.0, 0.0]
    _, dv, da = gradient(straightline!, v0, 3.0)
    @test dv == [0.0, 0.0]     # both elements overwritten before being read back — no dependence on v0
    @test da == 3.0            # d/da (a + 2a) = 3

    checkverify_rev(straightline!, (Vector{Float64}, Float64))
    check_stack_balance(straightline!, [1.0, 2.0], 4.0)
    check_block_stack_traffic(straightline!, [1.0, 2.0], 4.0; expect_zero=true)

    # A loop over indices is a different source of ambiguity (a genuine back-edge, Phase B/D) and
    # must keep paying block-stack traffic even though every individual access is still a collapsed
    # `@boundscheck` diamond underneath — collapsing the diamonds must not be mistaken for
    # collapsing the loop itself.
    # `check_stack_balance`/`check_block_stack_traffic` seed the pullback with `one(primal(ycd))`,
    # so — unlike `bench/workloads.jl`'s `nothing`-returning version — this one returns a value
    # (ISSUES #51: a `Nothing`-returning primal isn't drivable through those helpers at all).
    function vecloop!(v::Vector{Float64}, x::Float64)
        for i in 1:length(v)
            v[i] = x
        end
        return v[end]
    end
    checkverify_rev(vecloop!, (Vector{Float64}, Float64))
    check_stack_balance(vecloop!, [1.0, 2.0, 3.0], 5.0)
    check_block_stack_traffic(vecloop!, [1.0, 2.0, 3.0], 5.0; expect_zero=false)

    # 2-D indexing: not required to collapse (out of scope for v1 — see `_collapsible_regions`'s
    # docstring), but must still be correct either way.
    function mat_mutate!(A::Matrix{Float64}, a::Float64)
        A[1, 1] = a
        return A[1, 1]
    end
    _, dA, dmat_a = gradient(mat_mutate!, zeros(2, 2), 3.0)
    @test dA == [0.0 0.0; 0.0 0.0]
    @test dmat_a == 1.0
    checkverify_rev(mat_mutate!, (Matrix{Float64}, Float64))
    check_stack_balance(mat_mutate!, zeros(2, 2), 3.0)
end

@testset "reverse mode: dynamic array index re-derivation (no MemoryRef push)" begin
    # `comms-dynamic-index-instead-of-memoryref.md`: a loop-indexed read/write over an
    # argument-rooted array re-derives its `MemoryRef` handle in the pullback from a pushed `Int`
    # index instead of pushing the (16-byte, GC-scanned) handle itself.
    has_memoryref(T) = T <: Tuple && any(F -> F <: MemoryRef, fieldtypes(T))

    # 1. The conversion happened: no argument-rooted access leaves a `MemoryRef` on the tape.
    dynread(x) = (s = 0.0; for i in eachindex(x); s += x[i]; end; s)
    @test !any(has_memoryref, check_tape_size(dynread, (Vector{Float64},)))
    @test !any(has_memoryref, check_tape_size(x -> sum(abs2, x), (Vector{Float64},)))

    # 2. Bounds: `@inbounds` and checked reads both round-trip. The re-derived shadow ref forces
    # `boundscheck=true` regardless, so an out-of-bounds `@inbounds` primal access still throws on
    # the shadow rather than corrupting it (pinned directly for the literal-index case in
    # `test_reverse_mutation_aliasing.jl`; this is the loop-index analogue).
    dynread_inbounds(x) = (s = 0.0; for i in eachindex(x); s += (@inbounds x[i]); end; s)
    x7 = [2.0, 3.0, 5.0, 7.0]
    checkverify_rev(dynread, (Vector{Float64},))
    checkverify_rev(dynread_inbounds, (Vector{Float64},))
    check_stack_balance(dynread, x7)
    check_stack_balance(dynread_inbounds, x7)
    _, ddr = gradient(dynread, x7)
    _, ddri = gradient(dynread_inbounds, x7)
    @test ddr == ones(4)
    @test ddri == ones(4)

    # 3. Two distinct dynamic indices read in the same loop.
    twoidx(x) = (s = 0.0; for i in eachindex(x); s += x[i] * x[end - i + 1]; end; s)
    x8 = [1.0, 2.0, 3.0, 4.0]
    checkverify_rev(twoidx, (Vector{Float64},))
    check_stack_balance(twoidx, x8)
    _, dti = gradient(twoidx, x8)
    for k in eachindex(x8)
        xp = copy(x8); xp[k] += 1e-6
        xm = copy(x8); xm[k] -= 1e-6
        @test dti[k] ≈ (twoidx(xp) - twoidx(xm)) / 2e-6 rtol = 1e-5
    end

    # 4. Two accesses sharing one index SSA (`x[i]`/`y[i]`) — the case that exercises
    # `_scan_block_comms`'s existing per-block dedupe, since both rules independently try to
    # declare `(:primal, i)` when they land in the same block.
    sharedidx(x, y) = (s = 0.0; for i in eachindex(x); s += x[i] * y[i]; end; s)
    x9, y9 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    checkverify_rev(sharedidx, (Vector{Float64}, Vector{Float64}))
    check_stack_balance(sharedidx, x9, y9)
    _, dx9, dy9 = gradient(sharedidx, x9, y9)
    @test dx9 == y9
    @test dy9 == x9

    # 5. Writes: a bulk-saved loop (isbits eltype — the primal is restored via one whole-array
    # copy-back, not per element) and a non-bulk-saved one (non-isbits eltype, so every element's
    # old primal/tangent is still saved individually). The index item is declared outside the
    # `bulk_saved` branch in `builtin_rrule_comms(::Val{Base.memoryrefset!}, ...)`, so both
    # configurations must still push it and still balance.
    bulkwrite!(x) = (for i in eachindex(x); x[i] = 2 * x[i]; end; sum(x))
    xb = [1.0, 2.0, 3.0]
    checkverify_rev(bulkwrite!, (Vector{Float64},))
    check_stack_balance(bulkwrite!, copy(xb))
    _, dxb = gradient(bulkwrite!, copy(xb))
    @test dxb == fill(2.0, 3)
    @test !any(has_memoryref, check_tape_size(bulkwrite!, (Vector{Float64},)))

    nested_loop_write!(x::Vector{Vector{Float64}}, w::Vector{Float64}) =
        (for i in eachindex(x); x[i] = w; end; sum(x[end]))
    xn = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    wn = [7.0, 8.0]
    checkverify_rev(nested_loop_write!, (Vector{Vector{Float64}}, Vector{Float64}))
    check_stack_balance(nested_loop_write!, deepcopy(xn), copy(wn))
    _, dxn, dwn = gradient(nested_loop_write!, deepcopy(xn), copy(wn))
    @test dxn == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]   # every element overwritten before any read
    @test dwn == [1.0, 1.0]                             # only x[end] (== w, aliased) is read
    @test !any(has_memoryref, check_tape_size(nested_loop_write!, (Vector{Vector{Float64}}, Vector{Float64})))

    # 6. Local (non-argument-rooted) array is unaffected: its ref can't be re-derived from
    # `tape.args`, so it keeps pushing its `MemoryRef` handle exactly as before. The point is that
    # this path and the converted argument-rooted path still coexist correctly in one function.
    f2(x) = (y = similar(x); for i in eachindex(x); y[i] = x[i] * x[i]; end; sum(y))
    xf = [1.0, 2.0, 3.0]
    checkverify_rev(f2, (Vector{Float64},))
    check_stack_balance(f2, copy(xf))
    _, dxf = gradient(f2, copy(xf))
    @test dxf == 2 .* xf
    ts6 = check_tape_size(f2, (Vector{Float64},))
    @test any(has_memoryref, ts6)          # y's local ref: untouched
    @test !all(has_memoryref, ts6)         # x's argument-rooted reads: converted
end

@testset "reverse mode: recursive calls with an array argument" begin
    # Tier 7 Part 3: the recursive-call guard allows an array argument through when its identity
    # is traceable back to a function argument, threading the real fdata array through the
    # recursive `:invoke` instead of a detached `NoFData()`. `arr_inner` is a plain composite
    # function (no hand-written rule) taking the array directly, so differentiating a caller of it
    # exercises the *general* engine path, not `sum`'s own hand rule.
    @noinline arr_inner(v::Vector{Float64}) = v[1]^2 + v[2]^2         # d/dv = [2v1, 2v2]
    arr_outer(v::Vector{Float64}) = arr_inner(v)                       # one level of pass-through recursion
    arr_nest_mid(v::Vector{Float64}) = arr_inner(v)
    arr_nest(v::Vector{Float64}) = arr_nest_mid(v)                     # two levels of recursion
    # Aliasing: the same array argument accumulated into by *two* separate recursive calls — the
    # case most likely to expose an accumulation bug, since both inner pullbacks `increment!!` into
    # the same shared fdata array.
    arr_alias(v::Vector{Float64}) = arr_inner(v) + arr_inner(v)

    # `sum(v) do vi ... end` desugars to `sum(f, v)`, which Julia's optimizer inlines down to
    # `Base._mapreduce` — routed via the hand-written `sum`/`sum(f,·)` rules in `src/rrules.jl`
    # (kept off Base's own internals, which are self-recursive above
    # `Base.pairwise_blocksize` elements and would hit the unrelated self-recursion cycle guard).
    f_sumdo(v::Vector{Float64}) = sum(v) do vi
        vi^2 + 2vi + 1
    end

    # `vs[1]` (an inner array read out of an array-of-arrays via ordinary indexing): a
    # `memoryrefget` off a tracked ref is itself a tracked root when its result carries fdata, so
    # `w`'s identity threads straight through into the recursive call below.
    @noinline arr_inner_box(v::Vector{Float64}) = v[1] + v[2]
    function arr_via_box(vs::Vector{Vector{Float64}})
        w = vs[1]
        return arr_inner_box(w)
    end

    # Recursive call with a mutable-struct *argument*: `p` is a genuine function argument, tracked
    # via `_arg_fdata_tracked`, so the recursive-call guard lets it through into the recursive
    # `:invoke`'s `CoDual` — the inner call's rule accumulates straight into the caller's own
    # shared `MutableTangent` in place, no rdata needed back from the call at all.
    mutable struct MPoint; x::Float64; y::Float64; end
    @noinline arr_inner_mut(p::MPoint) = p.x + p.y
    arr_via_mut(p::MPoint) = arr_inner_mut(p)

    x5 = [3.0, 4.0]

    # The general engine path (no hand rule): a plain composite function taking the array
    # directly, one level and two levels of recursion.
    _, dx_outer = gradient(arr_outer, x5)
    @test dx_outer == [2 * x5[1], 2 * x5[2]]
    _, dx_nest = gradient(arr_nest, x5)
    @test dx_nest == [2 * x5[1], 2 * x5[2]]

    _, dx_alias = gradient(arr_alias, x5)
    @test dx_alias == [4 * x5[1], 4 * x5[2]]

    x6 = [1.0, 2.0]
    _, dx_sumdo = gradient(f_sumdo, x6)
    @test dx_sumdo == 2 .* x6 .+ 2
    for k in eachindex(x6)
        xp = copy(x6); xp[k] += 1e-6
        xm = copy(x6); xm[k] -= 1e-6
        @test dx_sumdo[k] ≈ (f_sumdo(xp) - f_sumdo(xm)) / 2e-6 rtol = 1e-5
    end

    # Plain `sum(x)`, also via the hand-written rule.
    x7 = [1.0, 2.0, 3.0, 4.0]
    _, dx_plainsum = gradient(sum, x7)
    @test dx_plainsum == ones(4)

    _, dvs_avb = gradient(arr_via_box, [[1.0, 2.0]])
    @test dvs_avb == [[1.0, 1.0]]
    checkverify_rev(arr_via_box, (Vector{Vector{Float64}},))
    check_stack_balance(arr_via_box, [[1.0, 2.0]])

    _, dp_avm = gradient(arr_via_mut, MPoint(1.0, 2.0))
    @test dp_avm == MutableTangent{@NamedTuple{x::Float64,y::Float64}}((x=1.0, y=1.0))

    checkverify_rev(arr_outer, (Vector{Float64},))
    checkverify_rev(arr_nest, (Vector{Float64},))
    checkverify_rev(arr_alias, (Vector{Float64},))
    checkverify_rev(f_sumdo, (Vector{Float64},))
    checkverify_rev(arr_via_mut, (MPoint,))

    check_stack_balance(arr_outer, [3.0, 4.0])
    check_stack_balance(arr_alias, [3.0, 4.0])
    check_stack_balance(f_sumdo, [1.0, 2.0])
    check_stack_balance(arr_via_mut, MPoint(1.0, 2.0))
end

@testset "zero_like_rdata_from_type for a non-concrete (Union) closure type" begin
    # A pullback that has to produce a zero rdata for a closure type `G` normally sees `G` bound to
    # the closure's concrete runtime type, but the derived recursion glue can resolve a rule via a
    # static call-site type that isn't concrete (`g` reached through an abstractly-typed
    # field/container) — that binds `G` to a non-concrete type. Calling `zero_rdata_from_type(G)`
    # there returns the `CannotProduceZeroRDataFromType()` sentinel for a `G` with real
    # (non-`NoRData`) rdata — and `increment!!` has no method for that, so the pullback crashed with
    # a raw `MethodError`. The fix is `zero_like_rdata_from_type(G)`, which returns `ZeroRData()`
    # instead, and which `increment!!` handles.
    #
    # This was originally written against `sum(f, x)`'s hand rule (`SumMapPullback`), whose `G`
    # parameter is exactly that shape. Those `sum` rules are currently commented out in
    # `src/rrules.jl` (the generic `mapreduce` path covers `sum` now), so the part that constructed
    # a `SumMapPullback` directly is gone; what's asserted below is the underlying tangent-system
    # behaviour, which is what actually regressed and is rule-independent. Restore the
    # `SumMapPullback` construction alongside those rules if they ever come back.
    #
    # `make_sum_map_closures` returns two closures over distinct captured `Float64`s, each with
    # real (non-`NoRData`) rdata, whose common supertype is a non-concrete `Union`. Built inside a
    # function so `a`/`b` are genuine captured closure fields, not global bindings.
    function make_sum_map_closures()
        a = 1.0
        b = 2.0
        return (y -> y * a), (y -> y * b)
    end

    h1, h2 = make_sum_map_closures()
    G2 = Union{typeof(h1),typeof(h2)}
    @test !isconcretetype(G2)
    @test rdata_type(tangent_type(G2)) != NoRData

    # The old call path: confirm it really does produce the sentinel (documenting the bug this
    # guards against, not just the fix) and that `increment!!` chokes on it.
    old_grdata = zero_rdata_from_type(G2)
    @test old_grdata isa CannotProduceZeroRDataFromType
    @test_throws MethodError increment!!(old_grdata, 1.0)

    # The fixed call path.
    new_grdata = zero_like_rdata_from_type(G2)
    @test new_grdata isa ZeroRData
    @test increment!!(new_grdata, 1.0) == 1.0
end
