using Test
using DifferForwards
using DifferForwards: Dual, primal, tangent, NoTangent, frule!!

include(joinpath(@__DIR__, "testutils.jl"))

# `map`/`map!` hand rules (`src/rules_broadcast.jl`), forward-mode half. Reverse-mode tests for
# the same rules live in DifferReverse/test/test_broadcast_rules.jl.
#
# Forward mode has no restriction composing `map`/`map!` inside a larger function (see the
# "composed inside a larger function" testset below) since `frule!!` dispatches on a genuine
# `Dual` value regardless of what it returns — unlike reverse mode, which has two pre-existing,
# general (not `map`-specific) engine limitations documented in DifferReverse's half of this file.

@testset "forward mode: map(f, x) unary" begin
    x = [0.3, 1.2, -0.7]
    dx = [1.0, 0.0, 0.0]
    d = frule!!(Dual(map, NoTangent()), Dual(sin, NoTangent()), Dual(x, dx))
    @test primal(d) == sin.(x)
    @test tangent(d) ≈ [cos(x[1]) * dx[1], 0.0, 0.0]

    # Full-Jacobian directional derivative (identity seed on every coordinate at once, since `map`
    # is elementwise, `dy[i] == cos(x[i])`).
    d2 = frule!!(Dual(map, NoTangent()), Dual(sin, NoTangent()), Dual(x, ones(3)))
    @test tangent(d2) ≈ cos.(x)

    checkverify(x -> map(sin, x), (Vector{Float64},))
end

@testset "forward mode: map(f, x, y) binary" begin
    x = [0.3, 1.2, -0.7]
    y = [1.0, -2.0, 0.5]
    d = frule!!(Dual(map, NoTangent()), Dual(+, NoTangent()), Dual(x, ones(3)), Dual(y, zeros(3)))
    @test primal(d) == x .+ y
    @test tangent(d) ≈ ones(3)

    d2 = frule!!(Dual(map, NoTangent()), Dual(*, NoTangent()), Dual(x, ones(3)), Dual(y, zeros(3)))
    @test primal(d2) == x .* y
    @test tangent(d2) ≈ y  # d/dx (x*y) = y, seeded with dx=1, dy=0

    checkverify((x, y) -> map(+, x, y), (Vector{Float64}, Vector{Float64}))

    @test_throws DimensionMismatch frule!!(
        Dual(map, NoTangent()), Dual(+, NoTangent()), Dual([1.0, 2.0], [0.0, 0.0]),
        Dual([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]))
end

@testset "forward mode: map!(f, dest, x) unary" begin
    x = [0.3, 1.2, -0.7]
    dest = zeros(3)
    ddest = zeros(3)
    d = frule!!(Dual(map!, NoTangent()), Dual(sin, NoTangent()), Dual(dest, ddest), Dual(x, ones(3)))
    @test primal(d) === dest  # map! returns (and mutates) dest in place
    @test dest == sin.(x)
    @test ddest ≈ cos.(x)

    checkverify(
        function (x)
            dest = similar(x)
            map!(sin, dest, x)
            return dest
        end,
        (Vector{Float64},),
    )
end

@testset "forward mode: map!(f, dest, x, y) binary" begin
    x = [0.3, 1.2, -0.7]
    y = [1.0, -2.0, 0.5]
    dest = zeros(3)
    ddest = zeros(3)
    frule!!(Dual(map!, NoTangent()), Dual(+, NoTangent()), Dual(dest, ddest), Dual(x, ones(3)), Dual(y, zeros(3)))
    @test dest == x .+ y
    @test ddest ≈ ones(3)
end

@testset "forward mode: map composed inside a larger function" begin
    # Composing `map`/`map!` with the rest of the engine works fine in forward mode. Unlike
    # reverse mode (see DifferReverse's half of this file), there is no "array-returning call"
    # engine restriction here.
    f(x) = sum(map(sin, x))
    x = [0.3, 1.2, -0.7]
    for k in eachindex(x)
        seed = zeros(3); seed[k] = 1.0
        d = frule!!(Dual(f, NoTangent()), Dual(x, seed))
        @test primal(d) ≈ sum(sin.(x))
        @test tangent(d) ≈ cos(x[k])
    end
    checkverify(f, (Vector{Float64},))

    f2(x) = (dest = similar(x); map!(sin, dest, x); sum(dest))
    for k in eachindex(x)
        seed = zeros(3); seed[k] = 1.0
        d = frule!!(Dual(f2, NoTangent()), Dual(x, seed))
        @test tangent(d) ≈ cos(x[k])
    end
    checkverify(f2, (Vector{Float64},))
end
