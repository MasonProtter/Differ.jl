using Test
using DifferReverse
using DifferReverse: CoDual, primal, tangent, NoFData, NoRData, Ctx, zero_fcodual, rrule!!, rev_gradient

include(joinpath(@__DIR__, "testutils.jl"))

# `map`/`map!` hand rules (`rules_broadcast.jl`), reverse-mode half. Forward-mode tests for the
# same rules live in DifferForwards/test/test_broadcast_rules.jl.
#
# A note on test shape, since it differs from most other rule test files: `map(f, x)` returns an
# array, and `rev_gradient`/`value_and_gradient!` seed the top-level return with `one(y)`, which
# doesn't exist for a `Vector` — so `rev_gradient(map, f, x)` can't be called directly. (Composing
# `map` inside a scalar-returning function, e.g. `f(x) = sum(map(sin, x))`, works fine — see the
# last testset.)
#
# So reverse-mode `map`/`map!` correctness below is tested by calling `rrule!!` directly, following
# the same fdata convention every array-returning value uses: an array's cotangent is supplied by
# writing into its own fdata array directly (`tangent(ycd) .= ...`), not by passing a seed to the
# pullback (whose seed argument is the array's rdata, always `NoRData()`).

@testset "reverse mode: map(f, x) unary" begin
    x = [0.3, 1.2, -0.7]
    fcd = zero_fcodual(map)
    gcd = zero_fcodual(sin)
    xcd = zero_fcodual(x)
    ycd, pb = rrule!!(fcd, Ctx(), gcd, xcd)
    @test primal(ycd) == sin.(x)
    dy = tangent(ycd)
    @test dy == zeros(3)  # fresh fdata, nothing has accumulated into it yet

    dy .= 1.0  # seed: unit cotangent on every output element
    _, gr, xr = pb(NoRData())
    @test gr === NoRData()  # `sin` has no differentiable parameters
    @test xr === NoRData()  # x's gradient flows through its own fdata array, not rdata
    @test tangent(xcd) ≈ cos.(x)

    # Cross-check against central differences on the whole `sum ∘ map` composite (ordinary,
    # undifferentiated Julia, no Differ machinery involved in this comparison function).
    for k in eachindex(x)
        xp = copy(x); xp[k] += 1e-6
        xm = copy(x); xm[k] -= 1e-6
        @test tangent(xcd)[k] ≈ (sum(sin.(xp)) - sum(sin.(xm))) / 2e-6 rtol = 1e-5
    end
end

@testset "reverse mode: map(f, x, y) binary" begin
    x = [0.3, 1.2, -0.7]
    y = [1.0, -2.0, 0.5]
    fcd = zero_fcodual(map)
    gcd = zero_fcodual(+)
    xcd = zero_fcodual(x)
    ycd = zero_fcodual(y)
    outcd, pb = rrule!!(fcd, Ctx(), gcd, xcd, ycd)
    @test primal(outcd) == x .+ y
    tangent(outcd) .= 1.0
    pb(NoRData())
    @test tangent(xcd) ≈ ones(3)
    @test tangent(ycd) ≈ ones(3)

    # `*` binary case, to exercise a non-trivial per-element pullback.
    xcd2 = zero_fcodual(x)
    ycd2 = zero_fcodual(y)
    gcd2 = zero_fcodual(*)
    outcd2, pb2 = rrule!!(fcd, Ctx(), gcd2, xcd2, ycd2)
    @test primal(outcd2) == x .* y
    tangent(outcd2) .= 1.0
    pb2(NoRData())
    @test tangent(xcd2) ≈ y  # d/dx (x*y) = y
    @test tangent(ycd2) ≈ x  # d/dy (x*y) = x
end

@testset "reverse mode: map!(f, dest, x) unary" begin
    x = [0.3, 1.2, -0.7]
    dest = zeros(3)
    fcd = zero_fcodual(map!)
    gcd = zero_fcodual(sin)
    destcd = zero_fcodual(dest)
    xcd = zero_fcodual(x)
    destycd, pb = rrule!!(fcd, Ctx(), gcd, destcd, xcd)
    @test primal(destycd) === dest
    @test dest == sin.(x)
    @test tangent(destcd) == zeros(3)  # freshly zeroed after the call, ready to accumulate

    tangent(destcd) .= 1.0
    pb(NoRData())
    @test tangent(xcd) ≈ cos.(x)

    for k in eachindex(x)
        xp = copy(x); xm = copy(x)
        xp[k] += 1e-6; xm[k] -= 1e-6
        @test tangent(xcd)[k] ≈ (sum(sin.(xp)) - sum(sin.(xm))) / 2e-6 rtol = 1e-5
    end
end

@testset "reverse mode: map!(f, dest, x, y) binary" begin
    x = [0.3, 1.2, -0.7]
    y = [1.0, -2.0, 0.5]
    dest = zeros(3)
    fcd = zero_fcodual(map!)
    gcd = zero_fcodual(+)
    destcd = zero_fcodual(dest)
    xcd = zero_fcodual(x)
    ycd = zero_fcodual(y)
    destycd, pb = rrule!!(fcd, Ctx(), gcd, destcd, xcd, ycd)
    @test dest == x .+ y
    tangent(destcd) .= 1.0
    pb(NoRData())
    @test tangent(xcd) ≈ ones(3)
    @test tangent(ycd) ≈ ones(3)
end

@testset "reverse mode: map/map! guard (Union-typed function argument)" begin
    # `f`'s static type must be concrete: reverse mode has no dynamic dispatch, so the per-element
    # `rrule!!(gcd, Ctx(), ...)` call inside `map`/`map!`'s own rule can't resolve a rule for a
    # non-concrete callee type. `make_map_closures` returns two closures over distinct captured
    # `Float64`s whose common supertype is a genuine `Union`, exactly the shape the derived
    # recursion glue can bind `G` to when `f` is reached through an abstractly-typed
    # field/container (mirrors the identical `SumMapPullback` test in `test_reverse_arrays.jl`).
    function make_map_closures()
        a = 1.0
        b = 2.0
        return (y -> y * a), (y -> y * b)
    end
    h1, h2 = make_map_closures()
    G2 = Union{typeof(h1),typeof(h2)}
    @test !isconcretetype(G2)

    gcd = CoDual{G2,NoFData}(h1, NoFData())
    xcd = zero_fcodual([1.0, 2.0])
    destcd = zero_fcodual([0.0, 0.0])

    err = @test_throws ErrorException rrule!!(zero_fcodual(map), Ctx(), gcd, xcd)
    @test occursin("ISSUES.md #43", err.value.msg)
    @test occursin("map", err.value.msg)

    err2 = @test_throws ErrorException rrule!!(zero_fcodual(map!), Ctx(), gcd, destcd, xcd)
    @test occursin("ISSUES.md #43", err2.value.msg)
    @test occursin("map!", err2.value.msg)
end

@testset "activity: map(f, x) unary, x held constant" begin
    # Nested case: `x` is constant at the top level, `map`'s own call is reached through recursion.
    f(x) = sum(map(sin, x))
    x = [0.3, 1.2, -0.7]
    checkverify_rev(f, (Vector{Float64},); inactive=(1,))

    xcd = const_codual(x)
    ycd, pb = rrule!!(zero_fcodual(map), Ctx(), zero_fcodual(sin), xcd)
    @test primal(ycd) == sin.(x)
    tangent(ycd) .= 1.0
    @test pb(NoRData()) == (NoRData(), NoRData(), NoRData())   # x constant, sin has no params
end

@testset "activity: map(f, x, y) binary, y held constant" begin
    x = [0.3, 1.2, -0.7]
    y = [1.0, -2.0, 0.5]
    dx = zeros(3)
    outcd, pb = rrule!!(zero_fcodual(map), Ctx(), zero_fcodual(*), CoDual(x, dx), const_codual(y))
    @test primal(outcd) == x .* y
    tangent(outcd) .= 1.0
    pb(NoRData())
    @test dx ≈ y   # d/dx (x*y) = y — the still-active argument accumulates normally
    checkverify_rev((x, y) -> sum(map(*, x, y)), (Vector{Float64}, Vector{Float64}); inactive=(2,))
end

@testset "activity: map!(f, dest, x) unary, source held constant" begin
    x = [0.3, 1.2, -0.7]
    dest, ddest = zeros(3), zeros(3)
    destycd, pb = rrule!!(zero_fcodual(map!), Ctx(), zero_fcodual(sin), CoDual(dest, ddest), const_codual(x))
    @test dest == sin.(x)
    tangent(destycd) .= 1.0
    @test pb(NoRData()) == (NoRData(), NoRData(), NoRData(), NoRData())
    checkverify_rev((dest, x) -> map!(sin, dest, x), (Vector{Float64}, Vector{Float64}); inactive=(2,))
end

@testset "activity: map!(f, dest, x) with a constant destination is refused" begin
    # `dest`'s shadow is both the per-element backward seed and the result's shadow, so writing into
    # a constant one would silently zero the source's gradient. A write-only buffer wants a zeroed
    # shadow, not `NoTangent`.
    x = [0.3, 1.2, -0.7]
    dest = zeros(3)
    @test_throws "declared constant" rrule!!(zero_fcodual(map!), Ctx(), zero_fcodual(sin),
                                             const_codual(dest), CoDual(x, zeros(3)))
end

@testset "activity: map!(f, dest, x, y) binary source, y held constant" begin
    x = [0.3, 1.2, -0.7]
    y = [1.0, -2.0, 0.5]
    dest, ddest, dx = zeros(3), zeros(3), zeros(3)
    destycd, pb = rrule!!(zero_fcodual(map!), Ctx(), zero_fcodual(+),
                          CoDual(dest, ddest), CoDual(x, dx), const_codual(y))
    @test dest == x .+ y
    tangent(destycd) .= 1.0
    @test pb(NoRData()) == (NoRData(), NoRData(), NoRData(), NoRData(), NoRData())
    @test dx ≈ ones(3)
    checkverify_rev((dest, x, y) -> map!(+, dest, x, y),
                    (Vector{Float64}, Vector{Float64}, Vector{Float64}); inactive=(3,))
end

@testset "activity: map!(f, dest, x, y) with a constant destination is refused" begin
    x, y = [0.3, 1.2, -0.7], [1.0, -2.0, 0.5]
    dest = zeros(3)
    @test_throws "declared constant" rrule!!(zero_fcodual(map!), Ctx(), zero_fcodual(+),
                                             const_codual(dest), CoDual(x, zeros(3)),
                                             CoDual(y, zeros(3)))
end

@testset "reverse mode: composing map inside rev_gradient" begin
    # `map`'s hand rule returns its shadow for the caller to accumulate into and its own pullback
    # to read back — so `sum`'s recursion into it can now route a real gradient through, even
    # though `map(sin, x)` is not itself the function's final return.
    f(x) = sum(map(sin, x))
    x = [0.3, 1.2, -0.7]
    _, dx = rev_gradient(f, x)
    @test dx ≈ cos.(x)
    for k in eachindex(x)
        xp = copy(x); xp[k] += 1e-6
        xm = copy(x); xm[k] -= 1e-6
        @test dx[k] ≈ (f(xp) - f(xm)) / 2e-6 rtol = 1e-5
    end
    checkverify_rev(f, (Vector{Float64},))
    check_stack_balance(f, x)
end
