# Tangent/fdata/rdata system tests. Ported from Differ's test_tangent_system.jl, keeping only
# the portion that doesn't need `Dual`/`CoDual` (those carriers, and their own tests, live in
# DifferForwards.jl/DifferReverse.jl — see those packages' test suites).

using Test
using DifferCore
using DifferCore: tangent_type, fdata_type, rdata_type, fdata, rdata, tangent, zero_tangent, increment!!
using DifferCore: Tangent, MutableTangent, NoFData, NoRData, build_tangent
using DifferCore: NoTangent, require_tangent_cache

struct Point; x::Float64; y::Float64; end
mutable struct MPoint; x::Float64; y::Float64; end

@testset "DifferCore.jl" begin
    @testset "tangent_type / fdata_type / rdata_type" begin
        # Matches Mooncake's documented values (see rule_system / fdata_type docstrings).
        @test tangent_type(Int)     === NoTangent
        @test tangent_type(Float64) === Float64
        @test tangent_type(Bool)    === NoTangent
        @test tangent_type(Vector{Float64}) === Vector{Float64}
        @test tangent_type(Tuple{Float64,Vector{Float64},Int}) ===
              Tuple{Float64,Vector{Float64},NoTangent}
        @test tangent_type(Tuple{Int,Int}) === NoTangent          # all-non-diff collapses
        @test tangent_type(Point) === Tangent{@NamedTuple{x::Float64, y::Float64}}
        @test tangent_type(MPoint) === MutableTangent{@NamedTuple{x::Float64, y::Float64}}

        # fdata / rdata split
        @test (fdata_type(Float64), rdata_type(Float64)) === (NoFData, Float64)
        @test (fdata_type(Vector{Float64}), rdata_type(Vector{Float64})) ===
              (Vector{Float64}, NoRData)
        T = tangent_type(Tuple{Float64,Vector{Float64},Int})
        @test fdata_type(T) === Tuple{NoFData,Vector{Float64},NoFData}
        @test rdata_type(T) === Tuple{Float64,NoRData,NoRData}
        # mutable struct: fdata is the whole tangent, no rdata
        @test fdata_type(tangent_type(MPoint)) === tangent_type(MPoint)
        @test rdata_type(tangent_type(MPoint)) === NoRData

        # tangent(fdata(t), rdata(t)) === t round-trips
        for p in Any[5.0, (5.0, [1.0, 2.0], 3), Point(1.0, 2.0), (a=1.0, b=2)]
            t = zero_tangent(p)
            @test tangent(fdata(t), rdata(t)) == t
        end
    end

    @testset "zero_tangent / increment!!" begin
        @test zero_tangent(2.0)  === 0.0
        @test zero_tangent(3)    === NoTangent()          # Int is non-differentiable
        @test zero_tangent(sin)  === NoTangent()          # singleton function
        # Complex{Float64} is a struct in Mooncake, so its tangent is a Tangent (not a Complex)
        @test zero_tangent(1.0 + 2.0im) == Tangent{@NamedTuple{re::Float64,im::Float64}}((re=0.0, im=0.0))
        @test zero_tangent(Point(1.0, 2.0)) == Tangent{@NamedTuple{x::Float64,y::Float64}}((x=0.0, y=0.0))
        @test zero_tangent([1.0, 2.0]) == [0.0, 0.0]
        # increment!! adds tangents; mutates array fdata in place
        @test increment!!(1.0, 2.0) === 3.0
        a = [1.0, 2.0]; @test increment!!(a, [3.0, 4.0]) === a && a == [4.0, 6.0]
        # `increment!!` decides its aliasing cache via `require_tangent_cache` (keyed on the
        # tangent type), the same authority `zero_tangent`/`set_to_zero!!` use — not a cruder
        # `isbitstype`. A `Vector{<:IEEEFloat}` tangent is provably tree-like, so no `IdDict` is
        # built: after warmup the only allocation is the result-copy path, never a cache.
        @test require_tangent_cache(Vector{Float64}) === Val{false}()
        let a = [1.0, 2.0, 3.0], b = [10.0, 20.0, 30.0]
            f(x, y) = increment!!(x, y)
            f(copy(a), b)                                  # warmup
            @test (@allocated f(copy(a), b)) == (@allocated copy(a))   # copy only, no IdDict
        end
    end

    @testset "build_tangent / get_tangent_field / set_tangent_field!" begin
        t = zero_tangent(Point(1.0, 2.0))
        t2 = build_tangent(Point, 3.0, 4.0)
        @test t2 isa Tangent
        @test DifferCore.get_tangent_field(t2, 1) === 3.0
        @test DifferCore.get_tangent_field(t2, 2) === 4.0

        mt = zero_tangent(MPoint(1.0, 2.0))
        DifferCore.set_tangent_field!(mt, 1, 5.0)
        @test DifferCore.get_tangent_field(mt, 1) === 5.0
    end

    @testset "array_tangents" begin
        v = [Point(1.0, 2.0), Point(3.0, 4.0)]
        zt = zero_tangent(v)
        @test zt == [zero_tangent(Point(1.0, 2.0)), zero_tangent(Point(3.0, 4.0))]
    end
end
