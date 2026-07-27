using Test
using Differ
using Differ: Dual, NoTangent, frule!!, build_tangent

include(joinpath(@__DIR__, "testutils.jl"))

@testset "scalar rules" begin
    x, dx = 0.7, 2.0
    @test frule!!(Dual(sin, NoTangent()), Dual(x, dx)) === Dual(sin(x), cos(x)*dx)
    @test frule!!(Dual(cos, NoTangent()), Dual(x, dx)) === Dual(cos(x), -sin(x)*dx)
    # unary + is identity; unary - negates
    @test frule!!(Dual(+, NoTangent()), Dual(x, dx)) === Dual(x, dx)
    @test frule!!(Dual(-, NoTangent()), Dual(x, dx)) === Dual(-x, -dx)
    # binary +, -, *
    y, dy = 1.5, 3.0
    @test frule!!(Dual(+, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x+y, dx+dy)
    @test frule!!(Dual(-, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x-y, dx-dy)
    @test frule!!(Dual(*, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x*y, x*dy + dx*y)
end

@testset "composite fallback (forward AD)" begin
    plus1(x)   = sin(x) + 1
    nest(x)    = sin(cos(x))
    prod2(x,y) = x*y + sin(x)
    sqr(x)     = x*x
    mul3(x)    = (x*x)*x   # explicit 2-arg grouping; `x*x*x` would be a 3-arg (vararg) `*`, unsupported
    sincosp(x) = sin(x)*cos(x)

    # d/dx sin(x)+1 = cos(x)
    d = frule!!(Dual(plus1, NoTangent()), Dual(1.0, 2.0))
    @test d.x  ≈ sin(1.0) + 1
    @test d.dx ≈ cos(1.0) * 2.0

    # nested: d/dx sin(cos(x)) = cos(cos(x))*(-sin(x))
    dn = frule!!(Dual(nest, NoTangent()), Dual(0.5, 1.0))
    @test dn.x  ≈ sin(cos(0.5))
    @test dn.dx ≈ cos(cos(0.5)) * (-sin(0.5))

    # multi-arg: p(x,y)=x*y+sin(x); ∂/∂x and ∂/∂y via tangent seeding
    px = frule!!(Dual(prod2, NoTangent()), Dual(2.0,1.0), Dual(3.0,0.0))
    @test px.x  ≈ 2.0*3.0 + sin(2.0)
    @test px.dx ≈ 3.0 + cos(2.0)                 # ∂/∂x = y + cos(x)
    py = frule!!(Dual(prod2, NoTangent()), Dual(2.0,0.0), Dual(3.0,1.0))
    @test py.dx ≈ 2.0                            # ∂/∂y = x

    # products: d/dx x^2 = 2x, d/dx x^3 = 3x^2
    @test frule!!(Dual(sqr,  NoTangent()), Dual(3.0,1.0)).dx ≈ 2*3.0
    @test frule!!(Dual(mul3, NoTangent()), Dual(2.0,1.0)).dx ≈ 3*2.0^2

    # d/dx sin(x)cos(x) = cos(2x)
    dsc = frule!!(Dual(sincosp, NoTangent()), Dual(0.9, 1.0))
    @test dsc.dx ≈ cos(2*0.9)

    # derivative matches finite differences, across the whole family above
    for (f, x) in ((plus1, 1.3), (nest, 0.4), (sqr, 2.1), (mul3, -0.7), (sincosp, 0.6))
        got = frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx
        @test got ≈ central_diff(f, x) rtol=1e-5
    end
end

@testset "intrinsic-level rules (no arithmetic frules)" begin
    # Complex arithmetic differentiated via add_float/mul_float/getfield/%new. Under the
    # Mooncake tangent system `Complex{Float64}` is a *struct*, so its tangent is a
    # `Tangent{@NamedTuple{re::Float64, im::Float64}}` (not another `Complex`). `ct` builds such
    # a tangent from a complex "direction"; the shadow reads `re`/`im` via `get_tangent_field`.
    ct(c) = build_tangent(ComplexF64, real(c), imag(c))
    z, w   = 1.0 + 2.0im, 3.0 + 4.0im
    dz, dw = 0.5 + 0.0im, 0.0 + 1.0im
    da = frule!!(Dual(+, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
    @test da.x  == z + w
    @test da.dx == ct(dz + dw)
    dm = frule!!(Dual(*, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
    @test dm.x  == z * w
    @test dm.dx == ct(z*dw + dz*w)                # complex product rule
    ds = frule!!(Dual(-, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
    @test ds.dx == ct(dz - dw)

    # Float32 straight-line composite: d/dx (x^2 + x) = 2x + 1
    poly32(x::Float32) = x*x + x
    d32 = frule!!(Dual(poly32, NoTangent()), Dual(2.0f0, 1.0f0))
    @test d32.x  === 2.0f0^2 + 2.0f0
    @test d32.dx === 2*2.0f0 + 1.0f0              # stays Float32

    # user struct via getfield: d/dv (v.a * v.b). The tangent of a `V2` is a `Tangent`, seeded
    # (da, db) = (1, 0). The shadow reads fields via `get_tangent_field`.
    struct V2; a::Float64; b::Float64; end
    vprod(v::V2) = v.a * v.b
    dv = frule!!(Dual(vprod, NoTangent()), Dual(V2(2.0, 3.0), build_tangent(V2, 1.0, 0.0)))
    @test dv.x  == 6.0
    @test dv.dx == 1.0*3.0 + 2.0*0.0              # = b*da + a*db

    # `sitofp` (Int→Float promotion): the INACTIVE bucket — the result carries a real tangent but
    # the operands (Int value + type) don't. d/dx (x·(1+2+3)) = 6.
    sitofp_ctl(x::Float64) = (s = 0.0; for i in 1:3; s += x*i; end; s)
    dsi = frule!!(Dual(sitofp_ctl, NoTangent()), Dual(2.0, 1.0))
    @test dsi.x == 12.0 && dsi.dx == 6.0
    # `fpext`/`fptrunc` (Float64<->Float32 width conversion): the LINEAR bucket — genuinely
    # differentiable. d/dx (Float64(Float32(x)·Float32(2)) + x) = 2 + 1 = 3.
    mix32_ctl(x::Float64) = Float64(Float32(x) * Float32(2.0)) + x
    dmx = frule!!(Dual(mix32_ctl, NoTangent()), Dual(1.0, 1.0))
    @test dmx.x == 3.0 && dmx.dx == 3.0
    @test dsi.dx ≈ central_diff(sitofp_ctl, 2.0; h=1e-5) rtol=1e-5
    @test dmx.dx ≈ central_diff(mix32_ctl, 1.0; h=1e-5) rtol=1e-2   # loose: FD through Float32 quantization is noisy
end

@testset "local reassignment (straight-line after optimization)" begin
    # p4(x)=x^4 via reassignment; optimization lowers it to straight-line SSA (no phi),
    # so the IRCode engine handles it. derivative 4x^3
    function p4(x)
        r = x*x
        r = r*x
        r = r*x
        r
    end
    d4 = frule!!(Dual(p4, NoTangent()), Dual(2.0, 1.0))
    @test d4.x  ≈ 2.0^4
    @test d4.dx ≈ 4 * 2.0^3
end
