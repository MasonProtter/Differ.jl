using Test
using Differ
using Differ: Dual, CoDual, NoTangent, NoFData, NoRData, frule!!, rrule!!
using Differ: AbstractCtx, Ctx, build_ctx, rev_gradient, zero_fcodual, tangent_type, increment!!
using Differ: primal, tangent, extract

include(joinpath(@__DIR__, "testutils.jl"))

@testset "regression: multi-dim Int indexing already works (native memoryref path)" begin
    # Locks in the pre-existing, engine-native behavior (see `differ-architecture`): scalar/multi-dim
    # Int indexing on a plain `Array` lowers to `memoryrefnew`/`memoryrefget`/`memoryrefset!` and is
    # already fully supported without any rule in this file. This file's own rules (mask/index-vector
    # indexing) never dispatch on this shape, so it's a pure regression lock, not something this
    # file's changes could plausibly affect — kept here anyway per the task's ask.
    read2d(A, i, j) = A[i, j]
    write2d!(A, i, j, x) = (A[i, j] = x; A)

    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    dA = [10.0 20.0 30.0; 40.0 50.0 60.0]
    r = frule!!(Dual(read2d, NoTangent()), Dual(A, dA), Dual(2, NoTangent()), Dual(3, NoTangent()))
    @test r.x == 6.0 && r.dx == 60.0
    checkverify(read2d, (Matrix{Float64}, Int, Int))

    A2, dA2 = copy(A), copy(dA)
    rw = frule!!(Dual(write2d!, NoTangent()), Dual(A2, dA2), Dual(1, NoTangent()), Dual(2, NoTangent()),
                 Dual(9.0, 99.0))
    @test A2[1, 2] == 9.0 && dA2[1, 2] == 99.0
    @test rw.x === A2 && rw.dx === dA2
    checkverify(write2d!, (Matrix{Float64}, Int, Int, Float64))

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

@testset "forward mode: logical (mask) indexing" begin
    getmask(A, mask) = A[mask]

    v = [1.0, 2.0, 3.0, 4.0]
    dv = [10.0, 20.0, 30.0, 40.0]
    mask = [true, false, true, true]
    r = frule!!(Dual(getmask, NoTangent()), Dual(v, dv), Dual(mask, zero_tangent(mask)))
    @test r.x == [1.0, 3.0, 4.0]
    @test r.dx == [10.0, 30.0, 40.0]
    checkverify(getmask, (Vector{Float64}, Vector{Bool}))

    # cross-check against central differences, one component at a time
    for k in eachindex(v)
        dvk = zeros(length(v)); dvk[k] = 1.0
        rk = frule!!(Dual(getmask, NoTangent()), Dual(v, dvk), Dual(mask, zero_tangent(mask)))
        expected = [central_diff(x -> (v2 = copy(v); v2[k] = x; getmask(v2, mask)[j]), v[k])
                    for j in eachindex(rk.x)]
        @test rk.dx ≈ expected atol=1e-6
    end

    # multi-dimensional source array, mask with matching axes
    A = [1.0 2.0; 3.0 4.0]
    dA = [10.0 20.0; 30.0 40.0]
    Amask = [true false; false true]
    rA = frule!!(Dual(getmask, NoTangent()), Dual(A, dA), Dual(Amask, zero_tangent(Amask)))
    @test rA.x == [1.0, 4.0]
    @test rA.dx == [10.0, 40.0]
    checkverify(getmask, (Matrix{Float64}, Matrix{Bool}))

    # mismatched axes: bail with a `DimensionMismatch`, not a silent wrong answer
    badmask = [true, false, true]
    @test_throws DimensionMismatch frule!!(
        Dual(getmask, NoTangent()), Dual([1.0, 2.0], [0.0, 0.0]), Dual(badmask, zero_tangent(badmask)))
end

@testset "forward mode: index-vector indexing (with repeats)" begin
    getidxvec(A, idx) = A[idx]

    v = [1.0, 2.0, 3.0]
    dv = [10.0, 20.0, 30.0]

    idx31 = [3, 1]
    r = frule!!(Dual(getidxvec, NoTangent()), Dual(v, dv), Dual(idx31, zero_tangent(idx31)))
    @test r.x == [3.0, 1.0]
    @test r.dx == [30.0, 10.0]
    checkverify(getidxvec, (Vector{Float64}, Vector{Int}))

    # repeated index: each occurrence independently gathers the same source element/tangent (a
    # forward-mode gather has no accumulation to worry about; that's a reverse-mode concern, tested
    # below).
    idxrep = [1, 1, 2]
    rrep = frule!!(Dual(getidxvec, NoTangent()), Dual(v, dv), Dual(idxrep, zero_tangent(idxrep)))
    @test rrep.x == [1.0, 1.0, 2.0]
    @test rrep.dx == [10.0, 10.0, 20.0]

    # cross-check against central differences
    for k in eachindex(v)
        dvk = zeros(length(v)); dvk[k] = 1.0
        rk = frule!!(Dual(getidxvec, NoTangent()), Dual(v, dvk), Dual(idxrep, zero_tangent(idxrep)))
        expected = [central_diff(x -> (v2 = copy(v); v2[k] = x; getidxvec(v2, idxrep)[j]), v[k])
                    for j in eachindex(rk.x)]
        @test rk.dx ≈ expected atol=1e-6
    end
end

@testset "forward mode: setindex! companions (mask / index-vector)" begin
    setmask!(A, v, mask) = (A[mask] = v; A)
    setidxvec!(A, v, idx) = (A[idx] = v; A)

    A, dA = [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]
    mask = [true, false, true, false]
    r = frule!!(Dual(setmask!, NoTangent()), Dual(A, dA), Dual([9.0, 8.0], [1.0, 2.0]), Dual(mask, zero_tangent(mask)))
    @test r.x == [9.0, 2.0, 8.0, 4.0] && A == [9.0, 2.0, 8.0, 4.0]
    @test r.dx == [1.0, 0.0, 2.0, 0.0] && dA == [1.0, 0.0, 2.0, 0.0]
    checkverify(setmask!, (Vector{Float64}, Vector{Float64}, Vector{Bool}))

    B, dB = [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]
    idx31b = [3, 1]
    rb = frule!!(Dual(setidxvec!, NoTangent()), Dual(B, dB), Dual([7.0, 8.0], [3.0, 4.0]), Dual(idx31b, zero_tangent(idx31b)))
    @test rb.x == [8.0, 2.0, 7.0] && B == [8.0, 2.0, 7.0]
    @test rb.dx == [4.0, 0.0, 3.0] && dB == [4.0, 0.0, 3.0]
    checkverify(setidxvec!, (Vector{Float64}, Vector{Float64}, Vector{Int}))

    badmask2 = [true, true]
    @test_throws DimensionMismatch frule!!(
        Dual(setmask!, NoTangent()), Dual([1.0, 2.0], [0.0, 0.0]), Dual([1.0], [0.0]),
        Dual(badmask2, zero_tangent(badmask2)))
end

@testset "reverse mode: logical (mask) / index-vector getindex (direct rrule!! calls)" begin
    # `Differ.rev_gradient` assumes a *scalar*-output primal (it seeds the pullback with `one(y)`), so
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
