using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, frule!!, build_tangent, zero_tangent

include(joinpath(@__DIR__, "testutils.jl"))

# Vararg-primal fixtures. Top level, not testset-local: `@noinline` is load-bearing for two of these
# (the call must survive into the primal IR as an `:invoke` rather than being inlined away), and a
# local `@noinline` definition is a closure with a non-singleton tangent, which would confuse the
# point being tested.
vsum(x, ys...) = x + sum(ys)
vint(x, ns::Int...) = x * ns[1]
vmix(x, zs...) = x*zs[1] + zs[2]
vfwd(x, ys...) = vsum(x, ys...)          # splat forwarding: the optimizer expands this to getfields
vnone(x, ys...) = x + 1.0                # never touches the (empty) vararg slot
@noinline vtupsum(t::Tuple) = sum(t)
vintsum(x, ns::Int...) = x * vtupsum(ns) # passes the WHOLE all-NoTangent vararg tuple to a live call
@noinline vg(a, bs...) = a + sum(bs)
vouter(x) = vg(x, x, 2x)                 # surviving `:invoke` to a vararg callee

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

@testset "vararg primal methods" begin
    # A vararg method's optimized IR has one slot per *declared* parameter, the last holding its
    # varargs already packed into a tuple (`getfield(_va, j)` in the body), while `frule!!` is always
    # called flat. The dualization prologue re-packs the trailing dual args into that tuple slot.
    fz(f) = Dual(f, NoTangent())

    # 1. Ordinary vararg method: one fixed arg + two varargs. Seed each slot in turn.
    @test frule!!(fz(vsum), Dual(1.0,1.0), Dual(2.0,0.0), Dual(3.0,0.0)) === Dual(6.0, 1.0)
    @test frule!!(fz(vsum), Dual(1.0,0.0), Dual(2.0,1.0), Dual(3.0,0.0)) === Dual(6.0, 1.0)
    @test frule!!(fz(vsum), Dual(1.0,0.0), Dual(2.0,0.0), Dual(3.0,1.0)) === Dual(6.0, 1.0)
    @test frule!!(fz(vsum), Dual(1.0,1.0), Dual(2.0,1.0), Dual(3.0,1.0)) === Dual(6.0, 3.0)
    @test frule!!(fz(vsum), Dual(1.0,1.0), Dual(2.0,0.0), Dual(3.0,0.0)).dx ≈
          central_diff(t -> vsum(t, 2.0, 3.0), 1.0)

    # 2. EMPTY vararg slot. Base's `*(a,b,c,xs...)` has `nargs == 5`, so a 3-arg call leaves the
    # vararg slot empty, and inference types it `Core.Const(())`, a lattice element rather than a
    # bare `Type`. Product rule: d(xyz) = dx·yz + x·dy·z + xy·dz.
    x, y, z = 2.0, 3.0, 4.0
    @test frule!!(fz(*), Dual(x,1.0), Dual(y,0.0), Dual(z,0.0)) === Dual(x*y*z, y*z)
    @test frule!!(fz(*), Dual(x,0.0), Dual(y,1.0), Dual(z,0.0)) === Dual(x*y*z, x*z)
    @test frule!!(fz(*), Dual(x,1.0), Dual(y,1.0), Dual(z,1.0)) === Dual(x*y*z, y*z + x*z + x*y)
    # an empty vararg slot that is genuinely *read* rather than constant-folded away: `sum(())`
    # throws, so inference types the call `Union{}` and the slot survives as a live
    # `invoke sum(_3::Tuple{})`. The derivative reproduces the primal's own error.
    @test_throws ArgumentError vsum(1.0)
    @test_throws ArgumentError frule!!(fz(vsum), Dual(1.0,1.0))
    # …and one where the empty slot is dead, so the packed `Core.tuple()` is DCE'd
    @test frule!!(fz(vnone), Dual(1.0,1.0)) === Dual(2.0, 1.0)

    # 3. One-element vararg slot (`Tuple{Float64}`), which the 4-arg `*` body genuinely reads.
    w = 5.0
    @test frule!!(fz(*), Dual(x,1.0), Dual(y,0.0), Dual(z,0.0), Dual(w,0.0)) ===
          Dual(x*y*z*w, y*z*w)
    @test frule!!(fz(*), Dual(x,0.0), Dual(y,0.0), Dual(z,0.0), Dual(w,1.0)) ===
          Dual(x*y*z*w, x*y*z)

    # 4. All-`NoTangent` vararg: `tangent_type(Tuple{Int,Int})` COLLAPSES to `NoTangent` rather than
    # `Tuple{NoTangent,NoTangent}`, so the packed shadow must be the literal `NoTangent()`. Here the
    # collapsed slot is only ever read element-wise, and each read takes the `NoTangent()` branch.
    @test frule!!(fz(vint), Dual(2.0,1.0), Dual(3,NoTangent()), Dual(4,NoTangent())) === Dual(6.0, 3.0)

    # 5. The collapse trap, directly: `vintsum` hands the *whole* all-`NoTangent` tuple to a surviving
    # call, so `frule_split!` builds `%new(Dual{Tuple{Int,Int},NoTangent}, %packed, <shadow>)`. If the
    # shadow were an emitted `Core.tuple(NoTangent(), NoTangent())` this would `TypeError` at the
    # `%new`. This is the regression test for that rule; cases 4 and 6 never read the slot whole.
    @test frule!!(fz(vintsum), Dual(2.0,1.0), Dual(3,NoTangent()), Dual(4,NoTangent())) ===
          Dual(14.0, 7.0)

    # 6. Mixed vararg (`Tuple{Float64,Int}` → tangent slot `Tuple{Float64,NoTangent}`): per-element
    # `NoTangent` placement inside an otherwise-differentiable packed tangent.
    @test frule!!(fz(vmix), Dual(2.0,1.0), Dual(3.0,0.0), Dual(4,NoTangent())) === Dual(10.0, 3.0)
    @test frule!!(fz(vmix), Dual(2.0,0.0), Dual(3.0,1.0), Dual(4,NoTangent())) === Dual(10.0, 2.0)

    # 7. Splat forwarding (`vfwd(x, ys...) = vsum(x, ys...)`): lowers to `Core._apply_iterate`, which
    # the optimizer fully expands for a statically-known tuple length, so it never reaches the engine.
    @test frule!!(fz(vfwd), Dual(1.0,1.0), Dual(2.0,0.0), Dual(3.0,0.0)) ===
          frule!!(fz(vsum), Dual(1.0,1.0), Dual(2.0,0.0), Dual(3.0,0.0))

    # 8. A *surviving* call to a vararg callee. Julia's compilation-signature heuristic collapses the
    # invoke target's `specTypes` to `Tuple{typeof(vg), Float64, Vararg{Float64}}`; `frule_split!`
    # takes its argument types from the operands (not the callee MI), so the flat `frule!!` call it
    # builds still resolves. vg(x,x,2x) = 4x ⇒ derivative 4.
    @test frule!!(fz(vouter), Dual(1.0,1.0)) === Dual(4.0, 4.0)
    @test frule!!(fz(vouter), Dual(2.5,1.0)).dx ≈ central_diff(vouter, 2.5)

    # 9. A vararg *closure*: `Argument(1)` is a non-singleton captured-field slot alongside the packed
    # vararg slot. d/dx (a·x + Σys) = a; d/da = x.
    vclo = let a = 3.0; (x, ys...) -> a*x + sum(ys); end
    capt(v) = build_tangent(typeof(vclo), v)
    @test frule!!(Dual(vclo, zero_tangent(vclo)), Dual(2.0,1.0), Dual(5.0,0.0)) === Dual(11.0, 3.0)
    @test frule!!(Dual(vclo, capt(1.0)), Dual(2.0,0.0), Dual(5.0,0.0)) === Dual(11.0, 2.0)
    @test frule!!(Dual(vclo, zero_tangent(vclo)), Dual(2.0,0.0), Dual(5.0,1.0)) === Dual(11.0, 1.0)

    # 10. Every shape above produces IR that passes `Core.Compiler.verify_ir`.
    for (f, at) in ((vsum,   (Float64,Float64,Float64)), (vsum,  (Float64,)),
                    (*,      (Float64,Float64,Float64)), (*,     (Float64,Float64,Float64,Float64)),
                    (vnone,  (Float64,)),                (vint,  (Float64,Int,Int)),
                    (vintsum,(Float64,Int,Int)),         (vmix,  (Float64,Float64,Int)),
                    (vfwd,   (Float64,Float64,Float64)), (vouter,(Float64,)),
                    (vclo,   (Float64,Float64)))
        checkverify(f, at)
    end
    @test true   # reached here ⇒ every verify_ir above passed
end
