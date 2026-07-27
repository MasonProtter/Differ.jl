using Test
using Differ
using Differ: Dual, CoDual, NoTangent, primal, tangent
using Differ: tangent_type, fdata_type, rdata_type, fdata, rdata, zero_tangent, increment!!
using Differ: Tangent, MutableTangent, NoFData, NoRData, build_tangent
using Differ: _dual_primal_type, _dual_tangent_type
using Differ: fcodual_type, codual_type, dual_type

struct Point; x::Float64; y::Float64; end
mutable struct MPoint; x::Float64; y::Float64; end

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

    # `_nondiff_field` decides whether a shadow-`Dual` field carries the primal through: true iff
    # the field's tangent is `NoTangent` but its slot can't hold `NoTangent`. That is *any*
    # non-differentiable field, not only singletons — including concrete non-singletons (`Int`,
    # `Tuple{Int,Int}`) and `Type`-valued fields (which `Base.issingletontype` misreports as
    # non-singleton, a documented Julia quirk). Anything else would drop `NoTangent()` into a
    # slot that rejects it.
    @test Differ._nondiff_field(typeof(sin))     # singleton function
    @test Differ._nondiff_field(Type{Float64})   # regression: Type{P}, missed by issingletontype
    @test Differ._nondiff_field(DataType)
    @test Differ._nondiff_field(Int)             # concrete non-singleton, NoTangent tangent
    @test Differ._nondiff_field(Tuple{Int,Int})  # concrete aggregate that collapses to NoTangent
    @test !Differ._nondiff_field(Float64)        # differentiable, takes its tangent
    @test !Differ._nondiff_field(Point)          # differentiable struct, takes its tangent
    @test !Differ._nondiff_field(NoTangent)      # a NoTangent slot holds NoTangent fine
    @test !Differ._nondiff_field(Integer)        # abstract slot: conservatively left on tangent path
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
    # `increment!!` decides its aliasing cache via `require_tangent_cache` (keyed on the tangent
    # type), the same authority `zero_tangent`/`set_to_zero!!` use — not a cruder `isbitstype`.
    # A `Vector{<:IEEEFloat}` tangent is provably tree-like, so no `IdDict` is built: after warmup
    # the only allocation is the result-copy path, never a cache. (Regression for the old
    # `isbitstype(T) ? … : IdDict` heuristic that disagreed with `zero_tangent`.)
    @test Differ.require_tangent_cache(Vector{Float64}) === Val{false}()
    let a = [1.0, 2.0, 3.0], b = [10.0, 20.0, 30.0]
        f(x, y) = increment!!(x, y)
        f(copy(a), b)                                  # warmup
        @test (@allocated f(copy(a), b)) == (@allocated copy(a))   # copy only, no IdDict
    end
end

@testset "CoDual basics" begin
    cd = CoDual([1.0, 2.0], [0.0, 0.0])
    @test primal(cd) == [1.0, 2.0]
    @test Differ.tangent(cd) == [0.0, 0.0]
    @test Differ.codual_type(Vector{Float64}) === CoDual{Vector{Float64},Vector{Float64}}
    @test Differ.fcodual_type(Float64) === CoDual{Float64,NoFData}
end

@testset "Dual basics" begin
    d = Dual(3.0, 4.0)
    @test d.x === 3.0
    @test d.dx === 4.0
    # getproperty aliases: x/y/z -> primal, dx/dy/dz -> tangent
    @test d.y === 3.0 && d.z === 3.0
    @test d.dy === 4.0 && d.dz === 4.0
    @test primal(d) === 3.0
    # type-level Dual field accessors
    @test _dual_primal_type(typeof(d)) === Float64
    @test _dual_tangent_type(typeof(d)) === Float64
    @test _dual_primal_type(typeof(Dual(sin, NoTangent()))) === typeof(sin)
    @test _dual_tangent_type(typeof(Dual(sin, NoTangent()))) === NoTangent
    # a Dual is its own tangent type (the key to higher-order nesting)
    @test tangent_type(Dual{Float64,Float64}) === Dual{Float64,Float64}
    @test tangent_type(typeof(Dual(sin, NoTangent()))) === typeof(Dual(sin, NoTangent()))
end

@testset "fcodual_type/codual_type on abstract P" begin
    # `fcodual_type`/`codual_type` (`src/codual.jl`) special-case a `UnionAll` with a free type
    # variable to the abstract fallback `CoDual`, but a plain abstract, fully-defined `P` (e.g.
    # `Real`, `Any`) isn't a `UnionAll` at all — it falls through to `_codual_internal`'s final
    # `isconcretetype(P) ? CoDual{P,extractor(P)} : CoDual` ternary instead. This confirms that
    # fallback already produces the correct (if loose) abstract `CoDual` for such a `P`, rather
    # than crashing or silently returning something too specific to be a valid supertype.
    for P in (Real, Any, AbstractFloat, Integer, Number)
        @test fcodual_type(P) === CoDual
        @test codual_type(P) === CoDual
    end
    # A real, concrete `CoDual` instance must be a subtype of the abstract fallback these return.
    inst = CoDual(1.0, NoFData())
    @test inst isa fcodual_type(Real)
    @test inst isa codual_type(Real)

    # The specific pathological case the `@isdefined(P)` guard documents — a `UnionAll` whose body
    # references a `TypeVar` that isn't its own bound variable (constructed directly, since this
    # isn't reachable via ordinary type syntax) — must not crash either.
    Tvar = TypeVar(:T)
    Avar = TypeVar(:A)
    pathological = UnionAll(Avar, AbstractArray{Tvar,Avar})
    @test fcodual_type(pathological) === CoDual
    @test codual_type(pathological) === CoDual
    @test dual_type(pathological) === Dual
end
