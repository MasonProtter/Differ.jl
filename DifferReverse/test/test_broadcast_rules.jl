using Test
using DifferReverse
using DifferReverse: CoDual, primal, tangent, NoFData, NoRData, Ctx, zero_fcodual, rrule!!, rev_gradient

include(joinpath(@__DIR__, "testutils.jl"))

# ===========================================================================
# `map`/`map!` hand rules (`src/rules_broadcast.jl`, ISSUES.md #31), reverse-mode half.
# Forward-mode tests for the same rules live in DifferForwards/test/test_broadcast_rules.jl.
#
# A note on test shape, since it differs from most other rule test files: `map(f, x)` returns an
# array, and Differ's reverse-mode engine has two pre-existing, general (not `map`-specific)
# limitations that this uncovers:
#
#   1. `rev_gradient`/`value_and_gradient!` seed the top-level return with `one(y)` (ISSUES.md #51),
#      which doesn't exist for a `Vector`, so `rev_gradient(map, f, x)` can't be called at all,
#      regardless of `map`'s own rule.
#   2. The general recursive-call dispatcher (`_static_recursible_call` in `reverse_interp.jl`)
#      unconditionally bails on any call whose result carries fdata (an array): "the fwds pass
#      has nowhere to route a result shadow today". This means a composite function that calls
#      `map`/`map!` internally (e.g. `f(x) = sum(map(sin, x))`) cannot be differentiated in reverse
#      mode via `rev_gradient(f, x)` today: the `map(sin, x)` call site bails before the engine ever
#      gets to consult `map`'s hand rule. Confirmed empirically below (last testset): a clean,
#      located `ErrorException`, not a crash. Fixing this is out of scope here; it's a general
#      engine limitation in `reverse_interp.jl`, not specific to `map`/`map!`.
#
# So reverse-mode `map`/`map!` correctness below is tested by calling `rrule!!` directly (exactly
# the pattern `check_stack_balance`/`SumMapPullback`'s own direct-construction test already use),
# following the same fdata convention every array-returning value uses throughout Differ: an
# array's cotangent is supplied by writing into its own fdata array directly (`tangent(ycd) .= ...`),
# not by passing a "seed" to the pullback (whose seed argument is the array's rdata, always
# `NoRData()`).
# ===========================================================================

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

@testset "reverse mode: map/map! #43 guard (Union-typed function argument)" begin
    # `f`'s static type must be concrete: reverse mode has no dynamic dispatch (ISSUES.md #43), so
    # the per-element `rrule!!(gcd, Ctx(), ...)` call inside `map`/`map!`'s own rule can't resolve a
    # rule for a non-concrete callee type. `make_map_closures` returns two closures over distinct
    # captured `Float64`s whose common supertype is a genuine `Union`, exactly the shape the
    # derived recursion glue can bind `G` to when `f` is reached through an abstractly-typed
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

@testset "reverse mode: composing map inside rev_gradient bails cleanly (documented engine gap)" begin
    # See the module-level note: `map`'s hand rule is correct (tested directly above), but a `map`
    # call result is an array with no provenance traceable to a function argument, which the
    # engine has no way to thread a real shadow for. So composing `map` inside a differentiated
    # function still bails, before `map`'s hand rule is ever consulted. This is a `map`-independent
    # engine limitation, out of scope for this rule file; asserted here as a clean, located
    # `ErrorException`, not a crash.
    f(x) = sum(map(sin, x))
    e = try
        rev_gradient(f, [0.3, 1.2, -0.7])
        nothing
    catch e
        e
    end
    @test e isa ErrorException
end
