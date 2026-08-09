using Test
using DifferReverse
using DifferReverse: CoDual, NoRData, rrule!!, AbstractCtx, Ctx, build_ctx, rev_gradient
using DifferReverse: zero_fcodual, primal, tangent

include(joinpath(@__DIR__, "testutils.jl"))

@testset "regression: multi-dim Int indexing already works (native memoryref path)" begin
    # Locks in the pre-existing, engine-native behavior (see `differ-architecture`): scalar/multi-dim
    # Int indexing on a plain `Array` lowers to `memoryrefnew`/`memoryrefget`/`memoryrefset!` and is
    # already fully supported without any rule in this file. This file's own rules (mask/index-vector
    # indexing) never dispatch on this shape, so it's a pure regression lock, not something this
    # file's changes could plausibly affect — kept here anyway per the task's ask.
    read2d(A, i, j) = A[i, j]

    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    _, dx_read = rev_gradient(read2d, A, 2, 3)
    @test dx_read == [0.0 0.0 0.0; 0.0 0.0 1.0]
    checkverify_rev(read2d, (Matrix{Float64}, Int, Int))
    check_stack_balance(read2d, A, 2, 3)

    Am = [1.0 2.0; 3.0 4.0]
    function mutate2d!(A)
        A[1, 1] = 2.0 * A[1, 1]
        return A[1, 1]
    end
    _, dx_mut = rev_gradient(mutate2d!, Am)
    @test dx_mut == [2.0 0.0; 0.0 0.0]
    checkverify_rev(mutate2d!, (Matrix{Float64},))
    check_stack_balance(mutate2d!, Am)
end

@testset "reverse mode: logical (mask) / index-vector getindex (direct rrule!! calls)" begin
    # `rev_gradient` assumes a *scalar*-output primal (it seeds the pullback with `one(y)`), so
    # it can't be used directly on `getindex`, whose result is itself an array. Tested here by
    # calling `rrule!!` directly with an explicit seed, the same style `check_stack_balance` uses
    # elsewhere for array-argument rules.
    #
    # Also only reachable this way (a direct call to `Base.getindex`'s own `rrule!!`), not through a
    # user-defined wrapper (`f(A, m) = A[m]`): see the NOTE above the rules in
    # `src/rules_indexing.jl`. The array-valued getindex result trips the reverse engine's own
    # (unrelated, out-of-scope-for-this-file) "recursive call with a non-trivial-fdata result" bail
    # before a hand rule is even considered.
    v = [1.0, 2.0, 3.0, 4.0]
    mask = [true, false, true, true]

    # The pullback's own `seed` parameter is `getindex`'s *rdata* — always `NoRData()`, since an
    # array-valued result carries its gradient via fdata (mutation of `dy`), not rdata. So seeding
    # is done the way any downstream consumer of `y` would: writing into `dy = tangent(ycd)` before
    # calling the pullback. Seeding `dy` with all-ones simulates a surrounding `sum(y)`.
    ctx = build_ctx(Base.getindex, (Vector{Float64}, Vector{Bool}); prealloc=false)
    fcd, vcd, maskcd = zero_fcodual(Base.getindex), zero_fcodual(v), zero_fcodual(mask)
    ycd, pb = rrule!!(fcd, ctx, vcd, maskcd)
    @test primal(ycd) == [1.0, 3.0, 4.0]
    tangent(ycd) .= 1.0
    r = pb(NoRData())
    @test r == (NoRData(), NoRData(), NoRData())
    # sum(A[mask]) seeded with all-ones: d/dA[i] = 1 at every `true` position, 0 elsewhere.
    @test tangent(vcd) == [1.0, 0.0, 1.0, 1.0]
    for k in eachindex(v)
        expected = central_diff(x -> sum(getindex((v2 = copy(v); v2[k] = x; v2), mask)), v[k])
        @test tangent(vcd)[k] ≈ expected atol=1e-6
    end

    idxvec = [1, 1, 2]
    ctx2 = build_ctx(Base.getindex, (Vector{Float64}, Vector{Int}); prealloc=false)
    fcd2, vcd2, idxcd2 = zero_fcodual(Base.getindex), zero_fcodual(v), zero_fcodual(idxvec)
    ycd2, pb2 = rrule!!(fcd2, ctx2, vcd2, idxcd2)
    @test primal(ycd2) == [1.0, 1.0, 2.0]
    tangent(ycd2) .= 1.0
    r2 = pb2(NoRData())
    @test r2 == (NoRData(), NoRData(), NoRData())
    # repeated index 1 contributes twice: d(sum(A[[1,1,2]]))/dA[1] = 2, /dA[2] = 1
    @test tangent(vcd2) == [2.0, 1.0, 0.0, 0.0]
    for k in eachindex(v)
        expected = central_diff(x -> sum(getindex((v2 = copy(v); v2[k] = x; v2), idxvec)), v[k])
        @test tangent(vcd2)[k] ≈ expected atol=1e-6
    end

    # tape hygiene: a repeated call through a fresh `Ctx()` shouldn't leave anything imbalanced.
    # This hand rule's pullback allocates no stack of its own, but confirm regardless.
    ycd3, pb3 = rrule!!(zero_fcodual(Base.getindex), Ctx(), zero_fcodual(v), zero_fcodual(mask))
    tangent(ycd3) .= 1.0
    pb3(NoRData())
end
