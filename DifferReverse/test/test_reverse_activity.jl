using Test
using DifferReverse
using DifferReverse: rev_gradient, value_and_gradient!
include("testutils.jl")

# Declaring an argument constant: `NoTangent` in a `CoDual`'s shadow slot. Activity propagates from
# there through the primal IR, so anything reached only through a constant is replayed primally
# instead of differentiated.

@testset "activity: scalar arguments" begin
    f(x, y) = x * y + sin(x)
    fcd = zero_fcodual(f)
    x, y = 2.0, 3.0
    dfdx, dfdy = y + cos(x), x

    both = rrule!!(fcd, Ctx(), CoDual(x, NoFData()), CoDual(y, NoFData()))[2](1.0)
    @test both == (NoRData(), dfdx, dfdy)

    ycon = rrule!!(fcd, Ctx(), CoDual(x, NoFData()), const_codual(y))[2](1.0)
    @test ycon == (NoRData(), dfdx, NoRData())

    xcon = rrule!!(fcd, Ctx(), const_codual(x), CoDual(y, NoFData()))[2](1.0)
    @test xcon == (NoRData(), NoRData(), dfdy)

    # Everything constant: nothing is differentiated at all, and the primal value still comes out.
    allcon = rrule!!(fcd, Ctx(), const_codual(x), const_codual(y))
    @test primal(allcon[1]) == f(x, y)
    @test allcon[2](1.0) == (NoRData(), NoRData(), NoRData())

    for inactive in ((), (1,), (2,), (1, 2))
        checkverify_rev(f, (Float64, Float64); inactive)
    end
end

@testset "activity: an inactive argument is not merely zero-seeded" begin
    # A constant argument carries no shadow at all, so a rule that reached for one would fail rather
    # than silently produce a zero. Distinguishes this from passing a zero tangent.
    g(x, y) = x * y
    fcd = zero_fcodual(g)
    ycd = const_codual(3.0)
    @test tangent(ycd) === NoTangent()
    @test rrule!!(fcd, Ctx(), CoDual(2.0, NoFData()), ycd)[2](1.0) == (NoRData(), 3.0, NoRData())
end

@testset "activity: array arguments" begin
    function floop(v, w)
        s = 0.0
        for i in eachindex(v)
            s += v[i] * w[i]
        end
        return s
    end
    v, w = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    at = (Vector{Float64}, Vector{Float64})

    dv = zeros(3)
    _, pb = rrule!!(zero_fcodual(floop), Ctx(), CoDual(v, dv), const_codual(w))
    @test pb(1.0) == (NoRData(), NoRData(), NoRData())
    @test dv == w   # d/dv of sum(v .* w) is w; accumulated in place, no shadow for `w` allocated

    checkverify_rev(floop, at; inactive=(2,))

    # Same answer through the pre-allocated path, and reusable.
    ctx = build_ctx(floop, at; inactive=(2,))
    fcd, vcd, wcd = zero_fcodual(floop), CoDual(v, zeros(3)), const_codual(w)
    y1, g1 = value_and_gradient!(ctx, fcd, vcd, wcd)
    y2, g2 = value_and_gradient!(ctx, fcd, vcd, wcd)
    @test y1 == y2 == floop(v, w)
    @test g1 == g2 == (NoTangent(), w, NoTangent())

    # The constant slot reconstructs to `NoTangent()`, not a zero array.
    @test g1[3] === NoTangent()
end

@testset "activity: matches the full gradient's corresponding slot" begin
    # Whatever is differentiated must agree with the all-active run, for every activity signature.
    h(a, b, c) = a * b + b * c + sin(a * c)
    args = (1.5, 2.5, 0.5)
    full = rev_gradient(h, args...)
    fcd = zero_fcodual(h)
    for inactive in ((1,), (2,), (3,), (1, 2), (1, 3), (2, 3))
        cds = ntuple(k -> k in inactive ? const_codual(args[k]) : CoDual(args[k], NoFData()), 3)
        got = rrule!!(fcd, Ctx(), cds...)[2](1.0)
        @test got[1] === NoRData()
        for k in 1:3
            if k in inactive
                @test got[k + 1] === NoRData()
            else
                @test got[k + 1] ≈ full[k + 1]
            end
        end
        checkverify_rev(h, (Float64, Float64, Float64); inactive)
    end
end

@testset "activity: loops run more than twice" begin
    # A control-flow-replay change can be correct for 0 and 1 iterations and still corrupt gradients
    # from the second onward, so exercise a genuinely repeated ambiguous edge.
    function powloop(x, n)
        s = 1.0
        for _ in 1:n
            s = s * x + 1.0
        end
        return s
    end
    for n in 0:4
        exact = central_diff(t -> powloop(t, n), 1.7)
        got = rrule!!(zero_fcodual(powloop), Ctx(),
                      CoDual(1.7, NoFData()), const_codual(n))[2](1.0)
        @test got[2] ≈ exact rtol = 1e-5
        @test got[3] === NoRData()
    end
    checkverify_rev(powloop, (Float64, Int); inactive=(2,))
end

@testset "activity: constant-only code is replayed, not differentiated" begin
    # `_static_recursible_call`'s gates (concrete argtypes, traceable provenance, resolvable callee)
    # never run for a call reached only through constants — the whole point of the bypass. This
    # primal bails outright when `d` is active.
    lookup(d, k) = d[k]
    f(x, d, k) = x * lookup(d, k)
    d = Dict("a" => 3.0, "b" => 4.0)

    # Treating the dictionary as differentiable fails — here already at `zero_fcodual`, which cannot
    # build a shadow for a `Dict` at all. Asserted only as the contrast to the constant case below.
    @test_throws Exception rrule!!(zero_fcodual(f), Ctx(), CoDual(2.0, NoFData()),
                                   zero_fcodual(d), zero_fcodual("a"))

    got = rrule!!(zero_fcodual(f), Ctx(), CoDual(2.0, NoFData()),
                  const_codual(d), const_codual("a"))
    @test primal(got[1]) == f(2.0, d, "a")
    @test got[2](1.0) == (NoRData(), 3.0, NoRData(), NoRData())
end

@testset "activity: shrinks the tape" begin
    # Correctness tests cannot catch this, and it is the whole point of the feature: a constant
    # argument must remove real per-iteration comms traffic, not just zero out a result.
    function floop(v, w)
        s = 0.0
        for i in eachindex(v)
            s += v[i] * w[i]
        end
        return s
    end
    at = (Vector{Float64}, Vector{Float64})
    full = comms_element_types(tape_type(floop, at))
    cut = comms_element_types(tape_type(floop, at; inactive=(2,)))
    @test sum(fieldcount, full) > sum(fieldcount, cut)
end

@testset "activity: mutating an inactive object with an active value bails" begin
    # Unsound to silently drop: the constant array has no shadow to accumulate into. Must be a
    # located error rather than a wrong number.
    function writeinto!(buf, x)
        buf[1] = x * 2.0
        return sum(buf)
    end
    err = try
        rrule!!(zero_fcodual(writeinto!), Ctx(), const_codual([1.0, 2.0]), CoDual(3.0, NoFData()))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
end

@testset "activity: build_ctx rejects an out-of-range position" begin
    @test_throws ArgumentError build_ctx((x, y) -> x * y, (Float64, Float64); inactive=(3,))
    @test_throws ArgumentError build_ctx((x, y) -> x * y, (Float64, Float64); inactive=(0,))
end

@testset "activity: a constant vararg-tail element bails instead of miscompiling" begin
    # A vararg primal's trailing arguments all land in one packed tail slot, always typed as the
    # active fdata carrier. A constant trailing argument used to sneak a `NoTangent()` value into that
    # slot and produce a `TypeError`; it must now be a located bail.
    vasum(vs...) = sum(vs[1])
    v = [1.0, 2.0, 3.0]

    err = try
        rrule!!(zero_fcodual(vasum), Ctx(), const_codual(v))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("cannot hold a vararg primal's trailing argument constant", err.msg)

    err2 = try
        build_ctx(vasum, (Vector{Float64},); inactive=(1,))
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("cannot hold a vararg primal's trailing argument constant", err2.msg)

    # The active vararg case must still work: not an over-broad bail.
    dv = zeros(3)
    _, pb = rrule!!(zero_fcodual(vasum), Ctx(), CoDual(v, dv))
    @test pb(1.0) == (NoRData(), NoRData())
    @test dv == ones(3)
end
