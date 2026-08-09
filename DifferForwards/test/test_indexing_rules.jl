using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, frule!!, zero_tangent

include(joinpath(@__DIR__, "testutils.jl"))

@testset "regression: multi-dim Int indexing already works (native memoryref path)" begin
    # Locks in the pre-existing, engine-native behavior: scalar/multi-dim Int indexing on a plain
    # `Array` lowers to `memoryrefnew`/`memoryrefget`/`memoryrefset!` and is already fully supported
    # without any rule in this file. This file's own rules (mask/index-vector indexing) never
    # dispatch on this shape.
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
    # in DifferReverse/test/test_indexing_rules.jl).
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
