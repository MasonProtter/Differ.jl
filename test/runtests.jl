using Test
using ADNext
using ADNext: Dual, NoFData, frule, struct_zero, primal_type, tangent_type

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

trycatch(x) = try sin(x) catch; cos(x) end       # exception handling: still unsupported (should bail)

struct Point; x::Float64; y::Float64; end

function pointphi(x)                              # PhiNode merging a Point; one arm a compile-time
    p = x > 0.0 ? Point(1.0, 2.0) : Point(x, x)   # constant, exercising `const_tangent` on structs
    p.x + p.y
end

# Intrinsic-level targets: these differentiate through intrinsics / getfield / %new on the
# post-optimization IRCode, with NO hand-written frule methods for +, -, *, /.
struct V2; a::Float64; b::Float64; end
vprod(v::V2) = v.a * v.b
poly32(x::Float32) = x*x + x

@testset "ADNext" begin

    @testset "struct_zero" begin
        @test struct_zero(2.0) === 0.0
        @test struct_zero(3)   === 0
        @test struct_zero(-1.5) === 0.0
        # singleton / fieldless types (e.g. functions) get NoFData()
        @test struct_zero(sin) === NoFData()
        @test struct_zero(+)   === NoFData()
        # composite non-Number struct: recurse structurally
        @test struct_zero(Point(1.0, 2.0)) === Point(0.0, 0.0)
        # Number subtypes (Complex) go through the Number method
        @test struct_zero(1.0 + 2.0im) === 0.0 + 0.0im
    end

    @testset "Dual basics" begin
        d = Dual(3.0, 4.0)
        @test d.x === 3.0
        @test d.dx === 4.0
        # getproperty aliases: x/y/z -> primal, dx/dy/dz -> tangent
        @test d.y === 3.0 && d.z === 3.0
        @test d.dy === 4.0 && d.dz === 4.0
        @test primal_type(typeof(d)) === Float64
        @test tangent_type(typeof(d)) === Float64
        @test primal_type(typeof(Dual(sin, NoFData()))) === typeof(sin)
        @test tangent_type(typeof(Dual(sin, NoFData()))) === NoFData
    end

    @testset "scalar rules" begin
        x, dx = 0.7, 2.0
        @test frule(Dual(sin, NoFData()), Dual(x, dx)) === Dual(sin(x), cos(x)*dx)
        @test frule(Dual(cos, NoFData()), Dual(x, dx)) === Dual(cos(x), -sin(x)*dx)
        # unary + is identity; unary - negates
        @test frule(Dual(+, NoFData()), Dual(x, dx)) === Dual(x, dx)
        @test frule(Dual(-, NoFData()), Dual(x, dx)) === Dual(-x, -dx)
        # binary +, -, *
        y, dy = 1.5, 3.0
        @test frule(Dual(+, NoFData()), Dual(x,dx), Dual(y,dy)) === Dual(x+y, dx+dy)
        @test frule(Dual(-, NoFData()), Dual(x,dx), Dual(y,dy)) === Dual(x-y, dx-dy)
        @test frule(Dual(*, NoFData()), Dual(x,dx), Dual(y,dy)) === Dual(x*y, x*dy + dx*y)
    end

    @testset "composite fallback (forward AD)" begin
        # d/dx sin(x)+1 = cos(x)
        d = frule(Dual(plus1, NoFData()), Dual(1.0, 2.0))
        @test d.x  ≈ sin(1.0) + 1
        @test d.dx ≈ cos(1.0) * 2.0

        # nested: d/dx sin(cos(x)) = cos(cos(x))*(-sin(x))
        dn = frule(Dual(nest, NoFData()), Dual(0.5, 1.0))
        @test dn.x  ≈ sin(cos(0.5))
        @test dn.dx ≈ cos(cos(0.5)) * (-sin(0.5))

        # multi-arg: p(x,y)=x*y+sin(x); ∂/∂x and ∂/∂y via tangent seeding
        px = frule(Dual(prod2, NoFData()), Dual(2.0,1.0), Dual(3.0,0.0))
        @test px.x  ≈ 2.0*3.0 + sin(2.0)
        @test px.dx ≈ 3.0 + cos(2.0)                 # ∂/∂x = y + cos(x)
        py = frule(Dual(prod2, NoFData()), Dual(2.0,0.0), Dual(3.0,1.0))
        @test py.dx ≈ 2.0                            # ∂/∂y = x

        # products: d/dx x^2 = 2x, d/dx x^3 = 3x^2
        @test frule(Dual(sqr,  NoFData()), Dual(3.0,1.0)).dx ≈ 2*3.0
        @test frule(Dual(mul3, NoFData()), Dual(2.0,1.0)).dx ≈ 3*2.0^2

        # d/dx sin(x)cos(x) = cos(2x)
        dsc = frule(Dual(sincosp, NoFData()), Dual(0.9, 1.0))
        @test dsc.dx ≈ cos(2*0.9)
    end

    @testset "intrinsic-level rules (no arithmetic frules)" begin
        # Complex arithmetic differentiated via add_float/mul_float/getfield/%new.
        z, w   = 1.0 + 2.0im, 3.0 + 4.0im
        dz, dw = 0.5 + 0.0im, 0.0 + 1.0im
        da = frule(Dual(+, NoFData()), Dual(z, dz), Dual(w, dw))
        @test da.x  == z + w
        @test da.dx == dz + dw
        dm = frule(Dual(*, NoFData()), Dual(z, dz), Dual(w, dw))
        @test dm.x  == z * w
        @test dm.dx == z*dw + dz*w                    # complex product rule
        ds = frule(Dual(-, NoFData()), Dual(z, dz), Dual(w, dw))
        @test ds.dx == dz - dw

        # Float32 straight-line composite: d/dx (x^2 + x) = 2x + 1
        d32 = frule(Dual(poly32, NoFData()), Dual(2.0f0, 1.0f0))
        @test d32.x  === 2.0f0^2 + 2.0f0
        @test d32.dx === 2*2.0f0 + 1.0f0              # stays Float32

        # user struct via getfield: d/dv (v.a * v.b), tangent seed (da,db)
        dv = frule(Dual(vprod, NoFData()), Dual(V2(2.0, 3.0), V2(1.0, 0.0)))
        @test dv.x  == 6.0
        @test dv.dx == 1.0*3.0 + 2.0*0.0              # = b*da + a*db
    end

    @testset "local reassignment (straight-line after optimization)" begin
        # p4(x)=x^4 via reassignment; optimization lowers it to straight-line SSA (no phi),
        # so the IRCode engine handles it. derivative 4x^3
        d4 = frule(Dual(p4, NoFData()), Dual(2.0, 1.0))
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
        r1 = frule(Dual(relu, NoFData()), Dual(2.0, 1.0));  @test r1.x ≈ 2.0  && r1.dx ≈ 1.0
        r2 = frule(Dual(relu, NoFData()), Dual(-2.0, 1.0)); @test r2.x ≈ 2.0  && r2.dx ≈ -1.0
        checkverify(relu, (Float64,))

        # while loop: k*x
        s1 = frule(Dual(sumk, NoFData()), Dual(3.0, 1.0), Dual(4, 0))
        @test s1.x ≈ 12.0 && s1.dx ≈ 4.0
        checkverify(sumk, (Float64, Int))

        # nested while loops: k*m*x
        n1 = frule(Dual(sumk2, NoFData()), Dual(2.0, 1.0), Dual(3, 0), Dual(5, 0))
        @test n1.x ≈ 2.0*3*5 && n1.dx ≈ 3.0*5.0
        checkverify(sumk2, (Float64, Int, Int))

        # if/elseif/else 3-way merge
        for (x, expected_x, expected_dx) in ((3.0, 9.0, 6.0), (1.0, 2.0, 1.0), (-1.0, 1.0, -1.0))
            b = frule(Dual(branch3, NoFData()), Dual(x, 1.0))
            @test b.x ≈ expected_x && b.dx ≈ expected_dx
        end
        checkverify(branch3, (Float64,))

        # multiple returns from nested branches
        for (x, expected_x, expected_dx) in ((20.0, 400.0, 40.0), (7.0, 107.0, 1.0),
                                              (3.0, 4.0, 1.0), (-3.0, 3.0, -1.0))
            m = frule(Dual(multiret, NoFData()), Dual(x, 1.0))
            @test m.x ≈ expected_x && m.dx ≈ expected_dx
        end
        checkverify(multiret, (Float64,))

        # loop body calling a surviving frule (sin): k*sin(x)
        sl = frule(Dual(sinloop, NoFData()), Dual(0.6, 1.0), Dual(3, 0))
        @test sl.x ≈ 3*sin(0.6) && sl.dx ≈ 3*cos(0.6)
        checkverify(sinloop, (Float64, Int))

        # two live loop-carried phis in one block: seed x and y independently
        mx = frule(Dual(sumk_multi, NoFData()), Dual(2.0, 1.0), Dual(3.0, 0.0), Dual(4, 0))
        @test mx.dx ≈ 4.0                                  # ds/dx = k
        my = frule(Dual(sumk_multi, NoFData()), Dual(2.0, 0.0), Dual(3.0, 1.0), Dual(4, 0))
        @test my.dx ≈ 4.0                                  # dt/dy = k
        checkverify(sumk_multi, (Float64, Float64, Int))

        # PhiNode merging a Point struct, one arm a compile-time constant
        pt = frule(Dual(pointphi, NoFData()), Dual(1.0, 1.0))
        @test pt.x ≈ 3.0 && pt.dx ≈ 0.0                    # constant arm: d/dx = 0
        pf = frule(Dual(pointphi, NoFData()), Dual(-1.0, 1.0))
        @test pf.x ≈ -2.0 && pf.dx ≈ 2.0                   # Point(x,x): d/dx (x+x) = 2
        checkverify(pointphi, (Float64,))
    end

    @testset "derivative matches finite differences" begin
        fd(f, x; h=1e-6) = (f(x+h) - f(x-h)) / 2h
        for (f, x) in ((plus1, 1.3), (nest, 0.4), (sqr, 2.1), (mul3, -0.7), (sincosp, 0.6))
            got = frule(Dual(f, NoFData()), Dual(x, 1.0)).dx
            @test got ≈ fd(f, x) rtol=1e-5
        end
    end

    @testset "graceful bail on unsupported IR" begin
        # exception handling is not yet supported: should error, not miscompile
        @test_throws ErrorException frule(Dual(trycatch, NoFData()), Dual(1.0, 1.0))
    end

    @testset "allocation-free (dualization is fully inlined)" begin
        # The dualized code is real post-optimization IRCode: Duals are built with %new and
        # surviving high-level rules are `:invoke`s to CodeInstances, so a straight-line dual is
        # allocation-free. `sincosp` exercises the surviving `frule(sin)`/`frule(cos)` :invoke path.
        allocs(f, x) = (d = Dual(x, 1.0); df = Dual(f, NoFData());
                        frule(df, d); @allocated frule(df, d))     # measure warmed
        @test allocs(sqr, 2.0)     == 0        # pure intrinsics
        @test allocs(sincosp, 0.6) == 0        # surviving sin/cos rule :invokes
    end

end
