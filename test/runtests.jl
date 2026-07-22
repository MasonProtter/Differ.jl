using Test
using ADNext
using ADNext: Dual, CoDual, NoTangent, frule, code_dual_ircode
using ADNext: tangent_type, fdata_type, rdata_type, fdata, rdata, tangent, zero_tangent
using ADNext: Tangent, MutableTangent, PossiblyUninitTangent, NoFData, NoRData, FData, RData
using ADNext: build_tangent, primal, increment!!
using ADNext: _dual_primal_type, _dual_tangent_type

# Primal functions used by the forward-AD fallback. Defined at top level so the
# generated `frule` fallback can resolve them via the method table.
plus1(x)   = sin(x) + 1
nest(x)    = sin(cos(x))
prod2(x,y) = x*y + sin(x)
sqr(x)     = x*x
mul3(x)    = (x*x)*x   # explicit 2-arg grouping; `x*x*x` would be a 3-arg (vararg) `*`, unsupported
sincosp(x) = sin(x)*cos(x)

# control flow + local assignments
relu(x)    = x > 0.0 ? x : -x                    # branch (== abs here)
function p4(x)                                    # local reassignment: x^4
    r = x*x
    r = r*x
    r = r*x
    r
end
function sumk(x, k)                               # while loop (backward goto): k*x
    s = x - x                                     # 0, without needing an frule for zero()
    i = 0
    while i < k
        s = s + x
        i = i + 1
    end
    s
end

function sumk2(x, k, m)                           # nested while loops: k*m*x
    s = x - x
    i = 0
    while i < k
        j = 0
        while j < m
            s = s + x
            j = j + 1
        end
        i = i + 1
    end
    s
end

function branch3(x)                               # if/elseif/else: 3-way merge into one phi
    if x > 2.0
        x*x
    elseif x > 0.0
        x + 1.0
    else
        -x
    end
end

function multiret(x)                              # multiple returns from nested branches
    if x > 10.0
        return x*x
    end
    if x > 0.0
        if x > 5.0
            return x + 100.0
        end
        return x + 1.0
    end
    return -x
end

function sinloop(x, k)                            # loop body calls a surviving frule (sin): k*sin(x)
    s = x - x
    i = 0
    while i < k
        s = s + sin(x)
        i = i + 1
    end
    s
end

function sumk_multi(x, y, k)                      # two live loop-carried phis in one block
    s = x - x
    t = y - y
    i = 0
    while i < k
        s = s + x
        t = t + y
        i = i + 1
    end
    s + t
end

trycatch(x) = try sin(x) catch; cos(x) end       # try/catch, non-throwing path: d/dx = cos(x)
# try/catch where the body genuinely throws on one branch (inline throw, M1) and the handler
# differentiates (M2): x>=0 runs the body (x*x), x<0 throws and the catch returns -x.
trythrow(x) = try (x < 0 ? throw(DomainError(x, "neg")) : x*x) catch; -x end

struct Point; x::Float64; y::Float64; end
mutable struct MPoint; x::Float64; y::Float64; end

function pointphi(x)                              # PhiNode merging a Point; one arm a compile-time
    p = x > 0.0 ? Point(1.0, 2.0) : Point(x, x)   # constant, exercising `const_tangent` on structs
    p.x + p.y
end

# Intrinsic-level targets: these differentiate through intrinsics / getfield / %new on the
# post-optimization IRCode, with NO hand-written frule methods for +, -, *, /.
struct V2; a::Float64; b::Float64; end
vprod(v::V2) = v.a * v.b
poly32(x::Float32) = x*x + x

# Closures capturing differentiable data: the *function* carries a tangent (its captured field), so
# one can differentiate w.r.t. the capture as well as the argument. The capture is read via
# `getfield(#self#, :a)` in the primal IR, whose tangent flows from the function-dual's tangent.
mklin(a)  = x -> a*x         # a·x        : d/dx = a,   d/da = x
mkquad(a) = x -> a*(x*x)     # a·x²       : d/dx = 2a·x, d²/dx² = 2a

# Still out of scope (scalar only): array indexing needs `Expr(:boundscheck)` + memoryref builtins
# on the live path. Should bail gracefully with an `ErrorException`, not miscompile.
getidx(v, i) = v[i]

# Throwing error paths: the happy path differentiates while the error path (a `throw` target whose
# block ends in an `unreachable` terminator) is reconstructed primal-only, so the derivative
# reproduces the same throw on the same inputs. `checkdom` uses the `Core.throw` builtin plus an
# inline `DomainError` construction; `checkdom_ni` routes through a `@noinline` `Union{}`-typed
# `:invoke` throw helper (the shape stdlib domain/bounds checks take); `guarded` guards a division.
checkdom(x) = x < 0 ? throw(DomainError(x, "neg")) : x*x
@noinline throwneg(x) = throw(DomainError(x, "neg"))
checkdom_ni(x) = x < 0 ? throwneg(x) : x*x
guarded(x) = x == 0.0 ? throw(ArgumentError("zero")) : 1/x

@testset "ADNext" begin

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
    end

    @testset "CoDual basics" begin
        cd = CoDual([1.0, 2.0], [0.0, 0.0])
        @test primal(cd) == [1.0, 2.0]
        @test ADNext.tangent(cd) == [0.0, 0.0]
        @test ADNext.codual_type(Vector{Float64}) === CoDual{Vector{Float64},Vector{Float64}}
        @test ADNext.fcodual_type(Float64) === CoDual{Float64,NoFData}
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

    @testset "scalar rules" begin
        x, dx = 0.7, 2.0
        @test frule(Dual(sin, NoTangent()), Dual(x, dx)) === Dual(sin(x), cos(x)*dx)
        @test frule(Dual(cos, NoTangent()), Dual(x, dx)) === Dual(cos(x), -sin(x)*dx)
        # unary + is identity; unary - negates
        @test frule(Dual(+, NoTangent()), Dual(x, dx)) === Dual(x, dx)
        @test frule(Dual(-, NoTangent()), Dual(x, dx)) === Dual(-x, -dx)
        # binary +, -, *
        y, dy = 1.5, 3.0
        @test frule(Dual(+, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x+y, dx+dy)
        @test frule(Dual(-, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x-y, dx-dy)
        @test frule(Dual(*, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x*y, x*dy + dx*y)
    end

    @testset "composite fallback (forward AD)" begin
        # d/dx sin(x)+1 = cos(x)
        d = frule(Dual(plus1, NoTangent()), Dual(1.0, 2.0))
        @test d.x  ≈ sin(1.0) + 1
        @test d.dx ≈ cos(1.0) * 2.0

        # nested: d/dx sin(cos(x)) = cos(cos(x))*(-sin(x))
        dn = frule(Dual(nest, NoTangent()), Dual(0.5, 1.0))
        @test dn.x  ≈ sin(cos(0.5))
        @test dn.dx ≈ cos(cos(0.5)) * (-sin(0.5))

        # multi-arg: p(x,y)=x*y+sin(x); ∂/∂x and ∂/∂y via tangent seeding
        px = frule(Dual(prod2, NoTangent()), Dual(2.0,1.0), Dual(3.0,0.0))
        @test px.x  ≈ 2.0*3.0 + sin(2.0)
        @test px.dx ≈ 3.0 + cos(2.0)                 # ∂/∂x = y + cos(x)
        py = frule(Dual(prod2, NoTangent()), Dual(2.0,0.0), Dual(3.0,1.0))
        @test py.dx ≈ 2.0                            # ∂/∂y = x

        # products: d/dx x^2 = 2x, d/dx x^3 = 3x^2
        @test frule(Dual(sqr,  NoTangent()), Dual(3.0,1.0)).dx ≈ 2*3.0
        @test frule(Dual(mul3, NoTangent()), Dual(2.0,1.0)).dx ≈ 3*2.0^2

        # d/dx sin(x)cos(x) = cos(2x)
        dsc = frule(Dual(sincosp, NoTangent()), Dual(0.9, 1.0))
        @test dsc.dx ≈ cos(2*0.9)
    end

    @testset "intrinsic-level rules (no arithmetic frules)" begin
        # Complex arithmetic differentiated via add_float/mul_float/getfield/%new. Under the
        # Mooncake tangent system `Complex{Float64}` is a *struct*, so its tangent is a
        # `Tangent{@NamedTuple{re::Float64, im::Float64}}` (not another `Complex`). `ct` builds such
        # a tangent from a complex "direction"; the shadow reads `re`/`im` via `get_tangent_field`.
        ct(c) = build_tangent(ComplexF64, real(c), imag(c))
        z, w   = 1.0 + 2.0im, 3.0 + 4.0im
        dz, dw = 0.5 + 0.0im, 0.0 + 1.0im
        da = frule(Dual(+, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
        @test da.x  == z + w
        @test da.dx == ct(dz + dw)
        dm = frule(Dual(*, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
        @test dm.x  == z * w
        @test dm.dx == ct(z*dw + dz*w)                # complex product rule
        ds = frule(Dual(-, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
        @test ds.dx == ct(dz - dw)

        # Float32 straight-line composite: d/dx (x^2 + x) = 2x + 1
        d32 = frule(Dual(poly32, NoTangent()), Dual(2.0f0, 1.0f0))
        @test d32.x  === 2.0f0^2 + 2.0f0
        @test d32.dx === 2*2.0f0 + 1.0f0              # stays Float32

        # user struct via getfield: d/dv (v.a * v.b). The tangent of a `V2` is a `Tangent`, seeded
        # (da, db) = (1, 0). The shadow reads fields via `get_tangent_field`.
        dv = frule(Dual(vprod, NoTangent()), Dual(V2(2.0, 3.0), build_tangent(V2, 1.0, 0.0)))
        @test dv.x  == 6.0
        @test dv.dx == 1.0*3.0 + 2.0*0.0              # = b*da + a*db
    end

    @testset "local reassignment (straight-line after optimization)" begin
        # p4(x)=x^4 via reassignment; optimization lowers it to straight-line SSA (no phi),
        # so the IRCode engine handles it. derivative 4x^3
        d4 = frule(Dual(p4, NoTangent()), Dual(2.0, 1.0))
        @test d4.x  ≈ 2.0^4
        @test d4.dx ≈ 4 * 2.0^3
    end

    @testset "control flow: branches and loops" begin
        # Block topology is preserved 1:1 from the primal IR; GotoNode/GotoIfNot/PhiNode are
        # supported by duplicating each PhiNode into a primal phi + a shadow phi.
        function checkverify(f, argtypes)
            ir, _ = code_dual_ircode(f, argtypes)
            Core.Compiler.verify_ir(ir)
        end

        # branch (== abs here): d/dx = sign(x)
        r1 = frule(Dual(relu, NoTangent()), Dual(2.0, 1.0));  @test r1.x ≈ 2.0  && r1.dx ≈ 1.0
        r2 = frule(Dual(relu, NoTangent()), Dual(-2.0, 1.0)); @test r2.x ≈ 2.0  && r2.dx ≈ -1.0
        checkverify(relu, (Float64,))

        # while loop: k*x
        s1 = frule(Dual(sumk, NoTangent()), Dual(3.0, 1.0), Dual(4, 0))
        @test s1.x ≈ 12.0 && s1.dx ≈ 4.0
        checkverify(sumk, (Float64, Int))

        # nested while loops: k*m*x
        n1 = frule(Dual(sumk2, NoTangent()), Dual(2.0, 1.0), Dual(3, 0), Dual(5, 0))
        @test n1.x ≈ 2.0*3*5 && n1.dx ≈ 3.0*5.0
        checkverify(sumk2, (Float64, Int, Int))

        # if/elseif/else 3-way merge
        for (x, expected_x, expected_dx) in ((3.0, 9.0, 6.0), (1.0, 2.0, 1.0), (-1.0, 1.0, -1.0))
            b = frule(Dual(branch3, NoTangent()), Dual(x, 1.0))
            @test b.x ≈ expected_x && b.dx ≈ expected_dx
        end
        checkverify(branch3, (Float64,))

        # multiple returns from nested branches
        for (x, expected_x, expected_dx) in ((20.0, 400.0, 40.0), (7.0, 107.0, 1.0),
                                              (3.0, 4.0, 1.0), (-3.0, 3.0, -1.0))
            m = frule(Dual(multiret, NoTangent()), Dual(x, 1.0))
            @test m.x ≈ expected_x && m.dx ≈ expected_dx
        end
        checkverify(multiret, (Float64,))

        # loop body calling a surviving frule (sin): k*sin(x)
        sl = frule(Dual(sinloop, NoTangent()), Dual(0.6, 1.0), Dual(3, 0))
        @test sl.x ≈ 3*sin(0.6) && sl.dx ≈ 3*cos(0.6)
        checkverify(sinloop, (Float64, Int))

        # two live loop-carried phis in one block: seed x and y independently
        mx = frule(Dual(sumk_multi, NoTangent()), Dual(2.0, 1.0), Dual(3.0, 0.0), Dual(4, 0))
        @test mx.dx ≈ 4.0                                  # ds/dx = k
        my = frule(Dual(sumk_multi, NoTangent()), Dual(2.0, 0.0), Dual(3.0, 1.0), Dual(4, 0))
        @test my.dx ≈ 4.0                                  # dt/dy = k
        checkverify(sumk_multi, (Float64, Float64, Int))

        # PhiNode merging a Point struct, one arm a compile-time constant
        pt = frule(Dual(pointphi, NoTangent()), Dual(1.0, 1.0))
        @test pt.x ≈ 3.0 && pt.dx ≈ 0.0                    # constant arm: d/dx = 0
        pf = frule(Dual(pointphi, NoTangent()), Dual(-1.0, 1.0))
        @test pf.x ≈ -2.0 && pf.dx ≈ 2.0                   # Point(x,x): d/dx (x+x) = 2
        checkverify(pointphi, (Float64,))
    end

    @testset "error paths (throwing)" begin
        # A block ending in an unreachable `ReturnNode` (a throw target) is reconstructed
        # primal-only: the happy path differentiates, and the derivative still throws on inputs
        # that make the primal throw. (Distinct from exception *handling* / try-catch below.)
        checkverify(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at)[1])

        # happy-path derivatives: d/dx x*x = 2x; d/dx 1/x = -1/x^2
        for (f, x, d) in ((checkdom, 3.0, 6.0), (checkdom_ni, 3.0, 6.0), (guarded, 2.0, -0.25))
            @test frule(Dual(f, NoTangent()), Dual(x, 1.0)).dx ≈ d
            checkverify(f, (Float64,))
        end

        # the derivative reproduces the primal's throw on the throwing input
        @test_throws DomainError   frule(Dual(checkdom,    NoTangent()), Dual(-2.0, 1.0))
        @test_throws DomainError   frule(Dual(checkdom_ni, NoTangent()), Dual(-2.0, 1.0))
        @test_throws ArgumentError frule(Dual(guarded,     NoTangent()), Dual(0.0, 1.0))
    end

    @testset "derivative matches finite differences" begin
        fd(f, x; h=1e-6) = (f(x+h) - f(x-h)) / 2h
        for (f, x) in ((plus1, 1.3), (nest, 0.4), (sqr, 2.1), (mul3, -0.7), (sincosp, 0.6))
            got = frule(Dual(f, NoTangent()), Dual(x, 1.0)).dx
            @test got ≈ fd(f, x) rtol=1e-5
        end
    end

    @testset "exception handling (try/catch)" begin
        # try/catch dualizes: UpsilonNode/PhiCNode are duplicated into primal + shadow copies, and
        # EnterNode/:leave/:pop_exception carry over as control markers (block topology preserved
        # 1:1). The optimizer deletes provably-non-throwing try scopes, so `trycatch`/`trythrow`
        # retain a live EnterNode only because their bodies can actually throw.
        checkverify(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at)[1])

        # non-throwing path through a try body calling surviving frules: d/dx sin(x) = cos(x)
        t = frule(Dual(trycatch, NoTangent()), Dual(0.6, 1.0))
        @test t.x ≈ sin(0.6) && t.dx ≈ cos(0.6)
        checkverify(trycatch, (Float64,))

        # body throws on one branch, handler differentiates: x>=0 -> x*x (d=2x); x<0 caught -> -x (d=-1)
        rp = frule(Dual(trythrow, NoTangent()), Dual(3.0, 1.0))
        @test rp.x ≈ 9.0 && rp.dx ≈ 6.0                    # happy path (no throw)
        rc = frule(Dual(trythrow, NoTangent()), Dual(-2.0, 1.0))
        @test rc.x ≈ 2.0 && rc.dx ≈ -1.0                   # catch path (body threw)
        checkverify(trythrow, (Float64,))
    end

    @testset "graceful bail on unsupported IR" begin
        # array indexing (memoryref builtins / boundscheck on the live path) is out of scope
        # (scalar only): should error, not miscompile.
        @test_throws ErrorException frule(Dual(getidx, NoTangent()),
                                          Dual([1.0, 2.0, 3.0], NoTangent()), Dual(2, NoTangent()))
    end

    @testset "function tangents (closures)" begin
        # The function isn't always constant: a closure carries a tangent in its captured field, so
        # the derivative w.r.t. the capture flows from the function-dual's tangent (read via
        # `getfield(#self#, :a)` in the body → `get_tangent_field` on the closure's `Tangent`).
        # Under the Mooncake tangent system a closure's tangent is a `Tangent{@NamedTuple{a::…}}`;
        # `zero_tangent(f)` is the "hold f constant" tangent (all-zero captures).
        f = mklin(2.0)                                     # a·x with a = 2
        capt(v) = build_tangent(typeof(f), v)              # closure tangent with da = v
        # d/dx (a constant): tangent = a·dx = 2
        r = frule(Dual(f, zero_tangent(f)), Dual(3.0, 1.0))
        @test r.x ≈ 6.0 && r.dx ≈ 2.0
        # d/da: seed the *function* tangent (da = 1), hold x fixed (dx = 0); tangent = da·x = 3
        ra = frule(Dual(f, capt(1.0)), Dual(3.0, 0.0))
        @test ra.x ≈ 6.0 && ra.dx ≈ 3.0
        # both directions at once (a=2, da=10, x=3, dx=1): a·dx + da·x = 2 + 30 = 32
        rb = frule(Dual(f, capt(10.0)), Dual(3.0, 1.0))
        @test rb.dx ≈ 32.0

        # a quadratic closure differentiated w.r.t. x, holding the capture constant: g(x)=a·x²,
        # g'(x)=2a·x. verify_ir on the dualized closure body (the capture getfield is dualized).
        g = mkquad(2.0)                                    # 2·x²
        rg = frule(Dual(g, zero_tangent(g)), Dual(1.5, 1.0))
        @test rg.x ≈ 2*1.5^2 && rg.dx ≈ 2*2*1.5
        Core.Compiler.verify_ir(code_dual_ircode(g, (Float64,))[1])
    end

    @testset "higher-order forward mode (uniform nesting)" begin
        # A second derivative differentiates the first-order dualized function itself (Option A —
        # compose the transform): the primal for a nested-dual request is the order-(k-1) dual IR,
        # re-dualized. ALL seeds — the function included — are nested uniformly to the order, with
        # NoTangent at the function's leaves. Then r.x.x = f(x), r.dx.x = r.x.dx = f'(x), r.dx.dx = f''(x).
        fseed2(f) = Dual(Dual(f, NoTangent()), Dual(f, NoTangent()))   # constant fn nested to order 2
        seed2(x)  = Dual(Dual(x, 1.0), Dual(1.0, 0.0))

        # analytic second derivatives: (x²)''=2, (x³)''=6x, (x⁴)''=12x²; and now sin(x)cos(x) too:
        # (sin·cos)'' = d/dx cos(2x) = -2 sin(2x)  (exercises the :new fix + sin/cos rewrite)
        for (f, x, fx, dfx, d2fx) in ((sqr,     2.0, 4.0,  4.0,  2.0),
                                      (mul3,    2.0, 8.0,  12.0, 12.0),
                                      (p4,      2.0, 16.0, 32.0, 48.0),
                                      (sincosp, 0.6, sin(0.6)*cos(0.6), cos(2*0.6), -2*sin(2*0.6)))
            r = frule(fseed2(f), seed2(x))
            @test r.x.x  ≈ fx
            @test r.x.dx ≈ dfx && r.dx.x ≈ dfx      # both first-derivative cross-terms agree
            @test r.dx.dx ≈ d2fx
        end

        # second derivative matches central finite differences of the (AD) first derivative, incl.
        # the transcendental cases now that sin/cos work to higher order
        d1(f, x) = frule(Dual(f, NoTangent()), Dual(x, 1.0)).dx
        d2fd(f, x; h=1e-4) = (d1(f, x+h) - d1(f, x-h)) / 2h
        for (f, x) in ((sqr, 2.1), (mul3, -0.7), (p4, 1.3), (sincosp, 0.6), (nest, 0.4), (plus1, 1.1))
            @test frule(fseed2(f), seed2(x)).dx.dx ≈ d2fd(f, x) rtol=1e-4
        end

        # order-N general: 3rd derivative of x⁴ is 24x (= 48 at x=2) by nesting one level deeper
        # (the function is nested to order 3 as well)
        fz(f) = Dual(f, NoTangent())
        fseed3(f) = Dual(Dual(fz(f), fz(f)), Dual(fz(f), fz(f)))
        s3(x) = Dual(Dual(Dual(x,1.0),Dual(1.0,0.0)), Dual(Dual(1.0,0.0),Dual(0.0,0.0)))
        @test frule(fseed3(p4), s3(2.0)).dx.dx.dx ≈ 24*2.0

        # the 2nd-order transform produces valid IR, including sin/cos and through control flow (the
        # tuple-aware vararg prologue composes with phi/goto re-dualization)
        checkverify2(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at; order=2)[1])
        for (f, at) in ((sqr,(Float64,)), (mul3,(Float64,)), (p4,(Float64,)), (sincosp,(Float64,)),
                        (relu,(Float64,)), (branch3,(Float64,)), (sumk,(Float64,Int)))
            checkverify2(f, at)
        end
        @test true   # reached here ⇒ every verify_ir above passed

        # a non-uniformly-nested seed (function NOT nested at order 2) is no longer valid: the uniform
        # peel can't form the inner carrier, so it bails rather than miscompiling.
        @test_throws Exception frule(Dual(sqr, NoTangent()), seed2(2.0))

        # graceful bail still holds at higher order: array indexing is unsupported → ErrorException.
        @test_throws ErrorException frule(fseed2(getidx),
            Dual(Dual([1.0,2.0], NoTangent()), Dual([0.0,0.0], NoTangent())),
            Dual(Dual(2, NoTangent()), Dual(0, NoTangent())))
    end

    @testset "allocation-free (dualization is fully inlined)" begin
        # The dualized code is real post-optimization IRCode: Duals are built with %new and
        # surviving high-level rules are `:invoke`s to CodeInstances, so a straight-line dual is
        # allocation-free. `sincosp` exercises the surviving `frule(sin)`/`frule(cos)` :invoke path.
        allocs(f, x) = (d = Dual(x, 1.0); df = Dual(f, NoTangent());
                        frule(df, d); @allocated frule(df, d))     # measure warmed
        @test allocs(sqr, 2.0)     == 0        # pure intrinsics
        @test allocs(sincosp, 0.6) == 0        # surviving sin/cos rule :invokes
    end

end
