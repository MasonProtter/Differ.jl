# Tangent/fdata/rdata system tests. Ported from Differ's test_tangent_system.jl, keeping only
# the portion that doesn't need `Dual`/`CoDual` (those carriers, and their own tests, live in
# DifferForwards.jl/DifferReverse.jl — see those packages' test suites).

using Test
using DifferCore
using DifferCore: tangent_type, fdata_type, rdata_type, fdata, rdata, tangent, zero_tangent, increment!!
using DifferCore: Tangent, MutableTangent, NoFData, NoRData, build_tangent
using DifferCore: NoTangent, require_tangent_cache
using DifferCore: Inactive, shadow_type, isactive, set_to_zero!!, increment_rdata!!, ZeroRData

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

        # A bare `Memory`/`MemoryRef` primal: same-shape, matching its `tangent_type`. Without these
        # arms the generic struct fallback tries to build a `Memory{Float64}` out of its own fields'
        # tangents. Forward mode's activity materialisation asks for exactly this.
        m = Memory{Float64}(undef, 3); fill!(m, 7.0)
        zm = zero_tangent(m)
        @test typeof(zm) === tangent_type(Memory{Float64}) === Memory{Float64}
        @test all(iszero, zm) && length(zm) == 3
        @test typeof(zero_tangent(Memory{Int}(undef, 2))) === Memory{NoTangent}

        r = Core.memoryref(m, 2)
        zr = zero_tangent(r)
        @test typeof(zr) === tangent_type(Core.MemoryRef{Float64}) === Core.MemoryRef{Float64}
        @test Base.memoryrefoffset(zr) == 2
    end

    @testset "Inactive" begin
        @test Base.issingletontype(Inactive) && sizeof(Inactive) == 0

        @test !isactive(Inactive())
        @test isactive(NoFData())        # an active scalar's shadow — not inactivity
        @test isactive(NoTangent())      # "no tangent space" is a different claim

        # `Inactive` survives the fdata/rdata split, which is what `NoTangent` cannot do.
        @test fdata_type(Inactive) === Inactive
        @test rdata_type(Inactive) === Inactive
        @test fdata(Inactive()) === Inactive()
        @test rdata(Inactive()) === Inactive()
        @test tangent(Inactive(), Inactive()) === Inactive()

        @test shadow_type(Vector{Float64}) === Union{Vector{Float64},Inactive}
        @test shadow_type(Float64) === Union{NoFData,Inactive}

        # The mixed-activity tuple arm must not capture homogeneously-typed tuples: those keep the
        # general entry point's aliasing/circular-reference cache. Uncached, a mutable tangent
        # reachable through two slots is counted twice.
        @test which(increment!!, Tuple{Tuple{Float64},Tuple{Float64}}).sig <: Tuple{Any,T,T} where {T<:Tuple}
        let t = zero_tangent(MPoint(1.0, 2.0)), s = zero_tangent(MPoint(1.0, 2.0))
            s.fields = (x = 1.0, y = 2.0)
            @test increment!!((t, t), (s, s))[1].fields == (x = 1.0, y = 2.0)
        end
        let a = zero_tangent(MPoint(1.0, 2.0)), b = zero_tangent(MPoint(1.0, 2.0)),
            s = zero_tangent(MPoint(1.0, 2.0))
            s.fields = (x = 1.0, y = 2.0)
            r = increment!!((a, b), (s, s))
            @test r[1].fields == (x = 1.0, y = 2.0) && r[2].fields == (x = 1.0, y = 2.0)
        end
        # A genuinely mixed-activity tuple still goes structurally, slot by slot.
        @test increment!!((1.0, Inactive()), (10.0, 5.0)) === (11.0, Inactive())
        @test increment!!((Inactive(), 1.0), (5.0, 10.0)) === (Inactive(), 11.0)

        # The mixed-activity arm needs the same aliasing cache as the homogeneous one: a
        # `MutableTangent` reachable through two slots of a mixed-activity tuple must not be
        # incremented twice just because some other slot's type differs between accumulator and
        # contribution.
        let t = zero_tangent(MPoint(1.0, 2.0)), s = zero_tangent(MPoint(1.0, 2.0))
            s.fields = (x = 1.0, y = 2.0)
            r = increment!!((t, t, 3.0), (s, s, Inactive()))
            @test r[1].fields == (x = 1.0, y = 2.0)
            @test r[1] === r[2]
            @test r[3] === 3.0
        end

        # Strong zero: absorbing in the accumulator slot, identity in the contribution slot.
        # Deliberately not commutative — slot 1 owns storage, slot 2 is a contribution.
        @test increment!!(Inactive(), 3.0) === Inactive()
        @test increment!!(3.0, Inactive()) === 3.0
        @test increment!!(Inactive(), Inactive()) === Inactive()
        let a = [1.0, 2.0]
            @test increment!!(a, Inactive()) === a && a == [1.0, 2.0]
            @test increment!!(Inactive(), a) === Inactive() && a == [1.0, 2.0]
        end

        # `ZeroRData` is the additive identity, so it keeps its own accumulator's meaning.
        @test increment!!(Inactive(), ZeroRData()) === Inactive()
        @test increment!!(ZeroRData(), Inactive()) === ZeroRData()

        @test increment_rdata!!(Inactive(), 3.0) === Inactive()
        @test increment_rdata!!(3.0, Inactive()) === 3.0
        @test set_to_zero!!(Inactive()) === Inactive()

        # `NoTangent` stays strict: no absorbing arm, so an analysis bug surfaces as a
        # `MethodError` rather than a silently dropped gradient.
        @test_throws MethodError increment!!(NoTangent(), 3.0)
        @test_throws MethodError increment!!(3.0, NoTangent())

        # Aggregates compose: a mixed tuple is an ordinary concrete tangent, and the
        # `tangent(fdata(t), rdata(t)) === t` round-trip still holds through it.
        let t = (1.0, Inactive())
            @test fdata_type(typeof(t)) === Tuple{NoFData,Inactive}
            @test rdata_type(typeof(t)) === Tuple{Float64,Inactive}
            @test fdata(t) === (NoFData(), Inactive())
            @test rdata(t) === (1.0, Inactive())
            @test tangent(fdata(t), rdata(t)) === t
            @test increment!!(t, (2.0, Inactive())) === (3.0, Inactive())
        end

        # An inactive slot costs nothing in an aggregate.
        @test sizeof(Tuple{Vector{Float64},Inactive}) == sizeof(Tuple{Vector{Float64}})

        # Any shadow is valid fdata/rdata for any primal once it is declared inactive.
        @test DifferCore.verify_fdata_type(Vector{Float64}, Inactive) === nothing
        @test DifferCore.verify_rdata_type(Float64, Inactive) === nothing
    end
end
