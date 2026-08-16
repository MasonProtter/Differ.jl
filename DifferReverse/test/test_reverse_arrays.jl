using Test
using DifferReverse
using DifferReverse: rev_gradient, MutableTangent, rdata_type, tangent_type, NoTangent
using DifferReverse: zero_rdata_from_type, zero_like_rdata_from_type, CannotProduceZeroRDataFromType
using DifferReverse: ZeroRData, NoRData, increment!!
using DifferReverse: rrule!!, Ctx, CoDual, primal, tangent
using DifferForwards: Dual, frule!!

include(joinpath(@__DIR__, "testutils.jl"))

@testset "reverse mode: read-only array indexing" begin
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
    _, dx_i3 = rev_gradient(arr_idx3, x4)
    @test dx_i3 == [0.0, 0.0, 1.0, 0.0]

    x2 = [1.0, 2.0]
    _, dx_ib_t, dp_t = rev_gradient(arr_idx_branch, x2, true)
    @test dx_ib_t == [1.0, 0.0]
    @test dp_t === NoTangent()
    _, dx_ib_f, = rev_gradient(arr_idx_branch, x2, false)
    @test dx_ib_f == [0.0, 1.0]

    x3 = [1.0, 2.0, 3.0]
    _, dx_sum = rev_gradient(arr_sum, x3)
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
    # and `arr_sum`'s loop has a real back-edge — both must keep paying traffic, a regression guard
    # against the optimization over-firing.
    check_block_stack_traffic(arr_idx3, x4; expect_zero=true)
    check_block_stack_traffic(arr_idx_branch, x2, true; expect_zero=false)
    check_block_stack_traffic(arr_sum, x3; expect_zero=false)
end

@testset "reverse mode: array mutation (memoryrefset!)" begin
    # `arr_mutate!(x) = (x[1] = 2*x[1]; x[1])` — the returned value only sees the *overwritten*
    # x[1], so d/dx = [2.0, 0.0].
    function arr_mutate!(x::Vector{Float64})
        x[1] = 2.0 * x[1]
        return x[1]
    end

    _, dx_mut = rev_gradient(arr_mutate!, [1.0, 2.0])
    @test dx_mut == [2.0, 0.0]
    h = 1e-6
    xp = [1.0 + h, 2.0]; xm = [1.0 - h, 2.0]
    @test dx_mut[1] ≈ (arr_mutate!(xp) - arr_mutate!(xm)) / 2h rtol = 1e-5

    checkverify_rev(arr_mutate!, (Vector{Float64},))
    check_stack_balance(arr_mutate!, [1.0, 2.0])
    check_block_stack_traffic(arr_mutate!, [1.0, 2.0]; expect_zero=true)
end

@testset "reverse mode: collapsible @boundscheck regions" begin
    # A genuinely straight-line primal (no real branch or loop) whose `@boundscheck` diamonds —
    # normally CFG-ambiguous, since `merge` has two real static predecessors — should all collapse
    # away, per `_collapsible_regions`.
    function straightline!(v::Vector{Float64}, a::Float64)
        v[1] = a
        v[2] = 2a
        return v[1] + v[2]
    end

    v0 = [0.0, 0.0]
    _, dv, da = rev_gradient(straightline!, v0, 3.0)
    @test dv == [0.0, 0.0]     # both elements overwritten before being read back, no dependence on v0
    @test da == 3.0            # d/da (a + 2a) = 3

    checkverify_rev(straightline!, (Vector{Float64}, Float64))
    check_stack_balance(straightline!, [1.0, 2.0], 4.0)
    check_block_stack_traffic(straightline!, [1.0, 2.0], 4.0; expect_zero=true)

    # A loop over indices is a different source of ambiguity (a genuine back-edge) and must keep
    # paying block-stack traffic even though every individual access is still a collapsed
    # `@boundscheck` diamond underneath — collapsing the diamonds must not be mistaken for
    # collapsing the loop itself. Returns a value (not `nothing`): `check_stack_balance`/
    # `check_block_stack_traffic` seed the pullback with `one(primal(ycd))`, which needs one.
    function vecloop!(v::Vector{Float64}, x::Float64)
        for i in 1:length(v)
            v[i] = x
        end
        return v[end]
    end
    checkverify_rev(vecloop!, (Vector{Float64}, Float64))
    check_stack_balance(vecloop!, [1.0, 2.0, 3.0], 5.0)
    check_block_stack_traffic(vecloop!, [1.0, 2.0, 3.0], 5.0; expect_zero=false)

    # 2-D indexing: not required to collapse, but must still be correct either way.
    function mat_mutate!(A::Matrix{Float64}, a::Float64)
        A[1, 1] = a
        return A[1, 1]
    end
    _, dA, dmat_a = rev_gradient(mat_mutate!, zeros(2, 2), 3.0)
    @test dA == [0.0 0.0; 0.0 0.0]
    @test dmat_a == 1.0
    checkverify_rev(mat_mutate!, (Matrix{Float64}, Float64))
    check_stack_balance(mat_mutate!, zeros(2, 2), 3.0)
end

@testset "reverse mode: dynamic array index re-derivation (no MemoryRef push)" begin
    # A loop-indexed read/write over an argument-rooted array re-derives its `MemoryRef` handle in
    # the pullback from a pushed `Int` index instead of pushing the (16-byte, GC-scanned) handle
    # itself.
    has_memoryref(T) = T <: Tuple && any(F -> F <: MemoryRef, fieldtypes(T))

    # 1. The conversion happened: no argument-rooted access leaves a `MemoryRef` on the tape.
    dynread(x) = (s = 0.0; for i in eachindex(x); s += x[i]; end; s)
    @test !any(has_memoryref, check_tape_size(dynread, (Vector{Float64},)))
    @test !any(has_memoryref, check_tape_size(x -> sum(abs2, x), (Vector{Float64},)))

    # 2. Bounds: `@inbounds` and checked reads both round-trip. The re-derived shadow ref forces
    # `boundscheck=true` regardless, so an out-of-bounds `@inbounds` primal access still throws on
    # the shadow instead of corrupting it — the loop-index analogue of the literal-index case
    # pinned in `test_reverse_mutation_aliasing.jl`.
    dynread_inbounds(x) = (s = 0.0; for i in eachindex(x); s += (@inbounds x[i]); end; s)
    x7 = [2.0, 3.0, 5.0, 7.0]
    checkverify_rev(dynread, (Vector{Float64},))
    checkverify_rev(dynread_inbounds, (Vector{Float64},))
    check_stack_balance(dynread, x7)
    check_stack_balance(dynread_inbounds, x7)
    _, ddr = rev_gradient(dynread, x7)
    _, ddri = rev_gradient(dynread_inbounds, x7)
    @test ddr == ones(4)
    @test ddri == ones(4)

    # 3. Two distinct dynamic indices read in the same loop.
    twoidx(x) = (s = 0.0; for i in eachindex(x); s += x[i] * x[end - i + 1]; end; s)
    x8 = [1.0, 2.0, 3.0, 4.0]
    checkverify_rev(twoidx, (Vector{Float64},))
    check_stack_balance(twoidx, x8)
    _, dti = rev_gradient(twoidx, x8)
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
    _, dx9, dy9 = rev_gradient(sharedidx, x9, y9)
    @test dx9 == y9
    @test dy9 == x9

    # 5. Writes: a bulk-saved loop (isbits eltype — the primal is restored via one whole-array
    # copy-back, not per element) and a non-bulk-saved one (non-isbits eltype, so every element's
    # old primal/tangent is still saved individually). The index item is declared outside the
    # `bulk_saved` branch, so both configurations must still push it and still balance.
    bulkwrite!(x) = (for i in eachindex(x); x[i] = 2 * x[i]; end; sum(x))
    xb = [1.0, 2.0, 3.0]
    checkverify_rev(bulkwrite!, (Vector{Float64},))
    check_stack_balance(bulkwrite!, copy(xb))
    _, dxb = rev_gradient(bulkwrite!, copy(xb))
    @test dxb == fill(2.0, 3)
    # Comms fusion drops this from 5 stacks to 4, at an unchanged 48 bytes. The 4th is the nested
    # `mapreduce_impl` tape from the trailing `sum(x)` — a separate inner tape, not a loop-body stack.
    @test !any(has_memoryref, check_tape_size(bulkwrite!, (Vector{Float64},); stacks=4))

    nested_loop_write!(x::Vector{Vector{Float64}}, w::Vector{Float64}) =
        (for i in eachindex(x); x[i] = w; end; sum(x[end]))
    write_only_nested!(x::Vector{Vector{Float64}}, w::Vector{Float64}) =
        (for i in eachindex(x); x[i] = w; end; nothing)
    xn = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    wn = [7.0, 8.0]
    checkverify_rev(nested_loop_write!, (Vector{Vector{Float64}}, Vector{Float64}))
    check_stack_balance(nested_loop_write!, deepcopy(xn), copy(wn))
    _, dxn, dwn = rev_gradient(nested_loop_write!, deepcopy(xn), copy(wn))
    @test dxn == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]   # every element overwritten before any read
    @test dwn == [1.0, 1.0]                             # only x[end] (== w, aliased) is read
    # The write loop itself still converts (matching `bulkwrite!` above): `check_tape_size` on a
    # write-only variant of this same primal is MemoryRef-free. `sum(x[end])` legitimately keeps
    # one, though: `sum`'s hand rule is not in the default build, so this reduction falls through
    # to generic recursion, and `x[end]` is extracted via indexing from the outer argument `x`, not
    # itself a direct function argument, so `_static_ref_derivation` has no `tape.args` entry to
    # re-derive its ref from. Same "untracked provenance keeps its handle" case `ts6` checks below.
    @test !any(has_memoryref, check_tape_size(write_only_nested!, (Vector{Vector{Float64}}, Vector{Float64})))
    @test any(has_memoryref, check_tape_size(nested_loop_write!, (Vector{Vector{Float64}}, Vector{Float64})))

    # 6. Local (non-argument-rooted) array is unaffected: its ref can't be re-derived from
    # `tape.args`, so it keeps pushing its `MemoryRef` handle exactly as before. The point is that
    # this path and the converted argument-rooted path still coexist correctly in one function.
    f2(x) = (y = similar(x); for i in eachindex(x); y[i] = x[i] * x[i]; end; sum(y))
    xf = [1.0, 2.0, 3.0]
    checkverify_rev(f2, (Vector{Float64},))
    check_stack_balance(f2, copy(xf))
    _, dxf = rev_gradient(f2, copy(xf))
    @test dxf == 2 .* xf
    ts6 = check_tape_size(f2, (Vector{Float64},))
    @test any(has_memoryref, ts6)          # y's local ref: untouched
    @test !all(has_memoryref, ts6)         # x's argument-rooted reads: converted
end

@testset "reverse mode: recursive calls with an array argument" begin
    # The recursive-call guard allows an array argument through when its identity is traceable
    # back to a function argument, threading the real fdata array through the recursive `:invoke`
    # instead of a detached `NoFData()`. `arr_inner` is a plain composite function (no hand-written
    # rule) taking the array directly, exercising the general engine path, not `sum`'s hand rule.
    @noinline arr_inner(v::Vector{Float64}) = v[1]^2 + v[2]^2         # d/dv = [2v1, 2v2]
    arr_outer(v::Vector{Float64}) = arr_inner(v)                       # one level of pass-through recursion
    arr_nest_mid(v::Vector{Float64}) = arr_inner(v)
    arr_nest(v::Vector{Float64}) = arr_nest_mid(v)                     # two levels of recursion
    # Aliasing: the same array argument accumulated into by *two* separate recursive calls — the
    # case most likely to expose an accumulation bug, since both inner pullbacks `increment!!` into
    # the same shared fdata array.
    arr_alias(v::Vector{Float64}) = arr_inner(v) + arr_inner(v)

    # `sum(v) do vi ... end` desugars to `sum(f, v)`, routed via the hand-written `sum`/`sum(f,·)`
    # rules (kept off Base's own internals, which are self-recursive above
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
    _, dx_outer = rev_gradient(arr_outer, x5)
    @test dx_outer == [2 * x5[1], 2 * x5[2]]
    _, dx_nest = rev_gradient(arr_nest, x5)
    @test dx_nest == [2 * x5[1], 2 * x5[2]]

    _, dx_alias = rev_gradient(arr_alias, x5)
    @test dx_alias == [4 * x5[1], 4 * x5[2]]

    x6 = [1.0, 2.0]
    _, dx_sumdo = rev_gradient(f_sumdo, x6)
    @test dx_sumdo == 2 .* x6 .+ 2
    for k in eachindex(x6)
        xp = copy(x6); xp[k] += 1e-6
        xm = copy(x6); xm[k] -= 1e-6
        @test dx_sumdo[k] ≈ (f_sumdo(xp) - f_sumdo(xm)) / 2e-6 rtol = 1e-5
    end

    # Plain `sum(x)`, also via the hand-written rule.
    x7 = [1.0, 2.0, 3.0, 4.0]
    _, dx_plainsum = rev_gradient(sum, x7)
    @test dx_plainsum == ones(4)

    _, dvs_avb = rev_gradient(arr_via_box, [[1.0, 2.0]])
    @test dvs_avb == [[1.0, 1.0]]
    checkverify_rev(arr_via_box, (Vector{Vector{Float64}},))
    check_stack_balance(arr_via_box, [[1.0, 2.0]])

    _, dp_avm = rev_gradient(arr_via_mut, MPoint(1.0, 2.0))
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

@testset "reverse mode: recursive call with an array-valued result" begin
    # The engine used to drop a recursive call's own returned shadow on the floor (nowhere to
    # route it), so a caller could never accumulate into it — this only worked when the array-
    # returning call was itself the function's final return. `@noinline` (not `sum(sin.([1,2].*x))`
    # from the array-construction testset above) so this really exercises a recursive `:invoke`
    # rather than getting inlined into a single straight-line block.
    @noinline function vecconstruct(x::Float64)
        return [x, 2x, 3x, 4x, 5x, 6x, 7x, 8x, 9x, 10x]
    end
    f_vecsum(x) = sum(vecconstruct(x))

    x0 = 1.5
    _, dx = rev_gradient(f_vecsum, x0)
    @test dx ≈ central_diff(f_vecsum, x0) rtol = 1e-5
    @test dx == sum(1:10)   # d/dx sum_i(i*x) = sum_i(i)

    # Confirm the call actually survives as a real recursive `:invoke` to `reverse_fwds_impl`
    # specialized on `vecconstruct`, not inlined away — dump the primal IR and look for it, rather
    # than assuming `@noinline` was honored.
    ir = code_reverse_fwds_ircode(f_vecsum, (Float64,))[1]
    @test any(st -> isa(st, Expr) && st.head === :invoke && occursin("vecconstruct", string(st)),
              ir.stmts.stmt)

    checkverify_rev(f_vecsum, (Float64,))
    check_stack_balance(f_vecsum, x0)

    # Aliasing: the shadow the caller (`sum`'s recursion into `mapreduce_impl`) accumulates into
    # must be `===` the object `vecconstruct`'s own pullback reads at pullback time. Proven
    # directly, without going through `sum` at all: seed `vecconstruct`'s own returned shadow by
    # writing into it externally, then confirm its own pullback reads exactly that mutation.
    ctx = Ctx()
    ycd_inner, pb_inner = rrule!!(CoDual(vecconstruct, NoFData()), ctx, CoDual(2.0, NoFData()))
    dv = tangent(ycd_inner)
    @test dv == zeros(10)
    dv .= Float64.(1:10)   # externally accumulate into the exact returned shadow object
    _, dx_inner = pb_inner(NoRData())
    @test dx_inner == sum((1:10) .^ 2)   # element i's coefficient is i, seeded dv[i] = i
end

@testset "reverse mode: dynamic getfield over a homogeneous tuple of arrays" begin
    # `getfield(vs::Tuple{Vector{Float64},...}, i::Int)` with `i` a runtime value (not a literal
    # field index) — the shape `vcat`/`hcat`'s own unrolled-free loop over their vararg tuple
    # compiles down to. Only supported when the tuple is homogeneous (every element the same
    # array type), which is what makes the result type static despite the dynamic index.
    @noinline function tupsum(vs::Tuple{Vector{Float64},Vector{Float64},Vector{Float64}})
        s = 0.0
        for j in 1:3
            v = vs[j]
            for i in eachindex(v)
                s += v[i]
            end
        end
        return s
    end
    x = (Float64.(1:3), Float64.(4:6), Float64.(7:9))

    _, dx = rev_gradient(tupsum, x)
    @test dx == (ones(3), ones(3), ones(3))
    checkverify_rev(tupsum, (Tuple{Vector{Float64},Vector{Float64},Vector{Float64}},))
    check_stack_balance(tupsum, x)
end

@testset "zero_like_rdata_from_type for a non-concrete (Union) closure type" begin
    # A pullback producing a zero rdata for a closure type `G` normally sees `G` bound to the
    # closure's concrete runtime type, but the derived recursion glue can resolve a rule via a
    # static call-site type that isn't concrete (`g` reached through an abstractly-typed
    # field/container), binding `G` to a non-concrete type. `zero_rdata_from_type(G)` there returns
    # the `CannotProduceZeroRDataFromType()` sentinel for a `G` with real (non-`NoRData`) rdata, and
    # `increment!!` has no method for that, so the pullback crashed with a raw `MethodError`. The
    # fix is `zero_like_rdata_from_type(G)`, which returns `ZeroRData()` instead, which
    # `increment!!` handles.
    #
    # `make_sum_map_closures` returns two closures over distinct captured `Float64`s, each with
    # real (non-`NoRData`) rdata, whose common supertype is a non-concrete `Union`. Built inside a
    # function so `a`/`b` are captured closure fields, not global bindings.
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

@testset "reverse mode: array-valued return" begin
    f_sinvec(x) = sin.(x)
    f_id(x) = x
    f_scale2!(x) = (x .*= 2; x)
    f_mixed(x) = (sum(x), 2 .* x)

    x = [1.0, 2.0, 3.0]
    dx = zeros(3)
    ctx = build_ctx(f_sinvec, (Vector{Float64},))
    ycd, pb = rrule!!(zero_fcodual(f_sinvec), ctx, CoDual(x, dx))
    ybar = [0.5, 1.5, -2.0]     # non-uniform, so a wrong-but-plausible scaling can't pass
    increment!!(tangent(ycd), fdata(ybar))
    pb(rdata(ybar))
    @test dx ≈ cos.(x) .* ybar

    # Returning an argument: the result shadow must be that argument's own buffer.
    x2, dx2 = [1.0, 2.0, 3.0], zeros(3)
    ctx2 = build_ctx(f_id, (Vector{Float64},))
    ycd2, pb2 = rrule!!(zero_fcodual(f_id), ctx2, CoDual(x2, dx2))
    @test tangent(ycd2) === dx2
    ybar2 = [1.0, 2.0, 3.0]
    increment!!(tangent(ycd2), fdata(ybar2))
    pb2(rdata(ybar2))
    @test dx2 ≈ ybar2

    x3, dx3 = [1.0, 2.0, 3.0], zeros(3)
    ctx3 = build_ctx(f_scale2!, (Vector{Float64},))
    ycd3, pb3 = rrule!!(zero_fcodual(f_scale2!), ctx3, CoDual(x3, dx3))
    @test tangent(ycd3) === dx3
    ybar3 = [1.0, 1.0, 1.0]
    increment!!(tangent(ycd3), fdata(ybar3))
    pb3(rdata(ybar3))
    @test dx3 ≈ 2 .* ybar3

    # Tuple return mixing a pure-rdata element (`sum`) with an fdata-carrying one (`2 .* x`).
    x4, dx4 = [1.0, 2.0, 3.0], zeros(3)
    ctx4 = build_ctx(f_mixed, (Vector{Float64},))
    ycd4, pb4 = rrule!!(zero_fcodual(f_mixed), ctx4, CoDual(x4, dx4))
    ybar4 = (2.0, [1.0, 2.0, 3.0])
    increment!!(tangent(ycd4), fdata(ybar4))
    pb4(rdata(ybar4))
    @test dx4 ≈ 2.0 .* ones(3) .+ 2 .* ybar4[2]

    checkverify_rev(f_sinvec, (Vector{Float64},))
    checkverify_rev(f_id, (Vector{Float64},))
    checkverify_rev(f_scale2!, (Vector{Float64},))
    checkverify_rev(f_mixed, (Vector{Float64},))
    check_stack_balance(f_sinvec, [1.0, 2.0, 3.0]; seed=NoRData())
end

@testset "reverse mode: array whose element tangent type differs from element type" begin
    # Shadow-side `MemoryRef`/`.ref` statements used to be declared at the primal's type; every
    # case here crashed with an illegal instruction before that was fixed.
    function g_int_read(x, v::Vector{Int})
        return v[1] * x
    end
    function g_int_sum(x, v::Vector{Int})
        return sum(v) * x
    end

    x_i, v_i = 0.5, [3, 4]
    _, dx_ir, dv_ir = rev_gradient(g_int_read, x_i, v_i)
    @test dx_ir == 3.0
    @test dv_ir == [NoTangent(), NoTangent()]
    _, dx_is, dv_is = rev_gradient(g_int_sum, x_i, v_i)
    @test dx_is == 7.0
    @test dv_is == [NoTangent(), NoTangent()]
    checkverify_rev(g_int_read, (Float64, Vector{Int}))
    checkverify_rev(g_int_sum, (Float64, Vector{Int}))
    check_stack_balance(g_int_read, 0.5, [3, 4])
    check_stack_balance(g_int_sum, 0.5, [3, 4])

    function g_bool(x, v::Vector{Bool})
        return v[1] ? 2x : 3x
    end
    _, dx_bt = rev_gradient(g_bool, 0.5, [true, false])
    @test dx_bt == 2.0
    _, dx_bf = rev_gradient(g_bool, 0.5, [false, true])
    @test dx_bf == 3.0
    checkverify_rev(g_bool, (Float64, Vector{Bool}))
    check_stack_balance(g_bool, 0.5, [true, false])

    # `Int`-element array construction and write, scalar-output form.
    f_vecconstruct(x) = sum(sin.([1, 2] .* x))
    x0 = 0.7
    _, dx_vc = rev_gradient(f_vecconstruct, x0)
    @test dx_vc ≈ central_diff(f_vecconstruct, x0) rtol = 1e-5
    @test dx_vc ≈ frule!!(Dual(f_vecconstruct, NoTangent()), Dual(x0, 1.0)).dx
    # Scalar broadcast operand: pullback reflection is a pre-existing gap, the carrier itself is fine.
    checkverify_rev_no_pb_reflection(f_vecconstruct, (Float64,))
    check_stack_balance(f_vecconstruct, x0)

    # A shadow array stores `Tangent{...}` elements, not `RData{...}`; the pullback used to assume
    # otherwise and crashed on every struct or `ComplexF64` element type.
    struct P2; a::Float64; b::Float64; end
    g_p2_read(x, v::Vector{P2}) = v[1].a * x

    v_p2 = [P2(3.0, 4.0), P2(1.0, 2.0)]
    _, dx_p2, dv_p2 = rev_gradient(g_p2_read, 0.5, v_p2)
    @test dx_p2 == v_p2[1].a
    @test dv_p2[1].fields.a == 0.5 && dv_p2[1].fields.b == 0.0
    @test dv_p2[2].fields.a == 0.0 && dv_p2[2].fields.b == 0.0
    checkverify_rev(g_p2_read, (Float64, Vector{P2}))
    check_stack_balance(g_p2_read, 0.5, v_p2)

    # Accumulate into the same element's rdata twice: exercises `increment_rdata!!` on a non-zero
    # starting tangent, not only a fresh zero.
    g_p2_double(x, v::Vector{P2}) = v[1].a * x + v[1].b * x

    _, dx_p2d, dv_p2d = rev_gradient(g_p2_double, 0.5, v_p2)
    @test dx_p2d == v_p2[1].a + v_p2[1].b
    @test dv_p2d[1].fields.a == 0.5 && dv_p2d[1].fields.b == 0.5
    @test dv_p2d[2].fields.a == 0.0 && dv_p2d[2].fields.b == 0.0
    checkverify_rev(g_p2_double, (Float64, Vector{P2}))
    check_stack_balance(g_p2_double, 0.5, v_p2)

    # A dynamic (loop) index routes through the `:shadow_ref` comms item rather than a statically
    # re-derived handle.
    g_p2_loop(x, v::Vector{P2}) = sum(e.a for e in v) * x

    _, dx_p2l, dv_p2l = rev_gradient(g_p2_loop, 0.5, v_p2)
    @test dx_p2l == v_p2[1].a + v_p2[2].a
    @test dv_p2l[1].fields.a == 0.5 && dv_p2l[2].fields.a == 0.5
    checkverify_rev(g_p2_loop, (Float64, Vector{P2}))
    check_stack_balance(g_p2_loop, 0.5, v_p2)

    g_complex_read(x, v::Vector{ComplexF64}) = real(v[1]) * x

    v_c = ComplexF64[3.0+1.0im, 4.0]
    _, dx_c, dv_c = rev_gradient(g_complex_read, 0.5, v_c)
    @test dx_c == real(v_c[1])
    @test dv_c[1].fields.re == 0.5 && dv_c[1].fields.im == 0.0
    @test dv_c[2].fields.re == 0.0 && dv_c[2].fields.im == 0.0
    checkverify_rev(g_complex_read, (Float64, Vector{ComplexF64}))
    check_stack_balance(g_complex_read, 0.5, v_c)

    # Element whose tangent splits across fdata (`v`) and rdata (`s`) — the shape that caught the
    # same defect on the forwards read side.
    struct M; v::Vector{Float64}; s::Float64; end
    g_m_read(x, v::Vector{M}) = v[1].s * x

    v_m = [M([1.0], 2.0), M([3.0], 4.0)]
    _, dx_m, dv_m = rev_gradient(g_m_read, 0.5, v_m)
    @test dx_m == v_m[1].s
    @test dv_m[1].fields.s == 0.5 && dv_m[1].fields.v == [0.0]
    @test dv_m[2].fields.s == 0.0 && dv_m[2].fields.v == [0.0]
    checkverify_rev(g_m_read, (Float64, Vector{M}))
    check_stack_balance(g_m_read, 0.5, v_m)
end
