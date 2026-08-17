using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, frule!!, code_dual_ircode

include(joinpath(@__DIR__, "testutils.jl"))

@testset "control flow: branches and loops" begin
    # Block topology is preserved 1:1 from the primal IR; GotoNode/GotoIfNot/PhiNode are
    # supported by duplicating each PhiNode into a primal phi + a shadow phi.
    relu(x)    = x > 0.0 ? x : -x                    # branch (== abs here)
    function sumk(x, k)                               # while loop (backward goto): k*x
        s = x - x                                     # 0, without needing an frule!! for zero()
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
    function sinloop(x, k)                            # loop body calls a surviving frule!! (sin): k*sin(x)
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
    struct Point; x::Float64; y::Float64; end
    function pointphi(x)                              # PhiNode merging a Point; one arm a compile-time
        p = x > 0.0 ? Point(1.0, 2.0) : Point(x, x)   # constant, exercising `const_tangent` on structs
        p.x + p.y
    end

    # branch (== abs here): d/dx = sign(x)
    r1 = frule!!(Dual(relu, NoTangent()), Dual(2.0, 1.0));  @test r1.x ≈ 2.0  && r1.dx ≈ 1.0
    r2 = frule!!(Dual(relu, NoTangent()), Dual(-2.0, 1.0)); @test r2.x ≈ 2.0  && r2.dx ≈ -1.0
    checkverify(relu, (Float64,))

    # while loop: k*x
    s1 = frule!!(Dual(sumk, NoTangent()), Dual(3.0, 1.0), Dual(4, 0))
    @test s1.x ≈ 12.0 && s1.dx ≈ 4.0
    checkverify(sumk, (Float64, Int))

    # nested while loops: k*m*x
    n1 = frule!!(Dual(sumk2, NoTangent()), Dual(2.0, 1.0), Dual(3, 0), Dual(5, 0))
    @test n1.x ≈ 2.0*3*5 && n1.dx ≈ 3.0*5.0
    checkverify(sumk2, (Float64, Int, Int))

    # if/elseif/else 3-way merge
    for (x, expected_x, expected_dx) in ((3.0, 9.0, 6.0), (1.0, 2.0, 1.0), (-1.0, 1.0, -1.0))
        b = frule!!(Dual(branch3, NoTangent()), Dual(x, 1.0))
        @test b.x ≈ expected_x && b.dx ≈ expected_dx
    end
    checkverify(branch3, (Float64,))

    # multiple returns from nested branches
    for (x, expected_x, expected_dx) in ((20.0, 400.0, 40.0), (7.0, 107.0, 1.0),
                                          (3.0, 4.0, 1.0), (-3.0, 3.0, -1.0))
        m = frule!!(Dual(multiret, NoTangent()), Dual(x, 1.0))
        @test m.x ≈ expected_x && m.dx ≈ expected_dx
    end
    checkverify(multiret, (Float64,))

    # loop body calling a surviving frule!! (sin): k*sin(x)
    sl = frule!!(Dual(sinloop, NoTangent()), Dual(0.6, 1.0), Dual(3, 0))
    @test sl.x ≈ 3*sin(0.6) && sl.dx ≈ 3*cos(0.6)
    checkverify(sinloop, (Float64, Int))

    # two live loop-carried phis in one block: seed x and y independently
    mx = frule!!(Dual(sumk_multi, NoTangent()), Dual(2.0, 1.0), Dual(3.0, 0.0), Dual(4, 0))
    @test mx.dx ≈ 4.0                                  # ds/dx = k
    my = frule!!(Dual(sumk_multi, NoTangent()), Dual(2.0, 0.0), Dual(3.0, 1.0), Dual(4, 0))
    @test my.dx ≈ 4.0                                  # dt/dy = k
    checkverify(sumk_multi, (Float64, Float64, Int))

    # PhiNode merging a Point struct, one arm a compile-time constant
    pt = frule!!(Dual(pointphi, NoTangent()), Dual(1.0, 1.0))
    @test pt.x ≈ 3.0 && pt.dx ≈ 0.0                    # constant arm: d/dx = 0
    pf = frule!!(Dual(pointphi, NoTangent()), Dual(-1.0, 1.0))
    @test pf.x ≈ -2.0 && pf.dx ≈ 2.0                   # Point(x,x): d/dx (x+x) = 2
    checkverify(pointphi, (Float64,))
end

@testset "error paths (throwing)" begin
    # A block ending in an unreachable `ReturnNode` (a throw target) is reconstructed
    # primal-only: the happy path differentiates, and the derivative still throws on inputs
    # that make the primal throw. (Distinct from exception *handling* / try-catch below.)

    # Throwing error paths: the happy path differentiates while the error path (a `throw` target
    # whose block ends in an `unreachable` terminator) is reconstructed primal-only, so the
    # derivative reproduces the same throw on the same inputs. `checkdom` uses the `Core.throw`
    # builtin plus an inline `DomainError` construction; `checkdom_ni` routes through a `@noinline`
    # `Union{}`-typed `:invoke` throw helper (the shape stdlib domain/bounds checks take);
    # `guarded` guards a division.
    checkdom(x) = x < 0 ? throw(DomainError(x, "neg")) : x*x
    @noinline throwneg(x) = throw(DomainError(x, "neg"))
    checkdom_ni(x) = x < 0 ? throwneg(x) : x*x
    guarded(x) = x == 0.0 ? throw(ArgumentError("zero")) : 1/x

    # happy-path derivatives: d/dx x*x = 2x; d/dx 1/x = -1/x^2
    for (f, x, d) in ((checkdom, 3.0, 6.0), (checkdom_ni, 3.0, 6.0), (guarded, 2.0, -0.25))
        @test frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx ≈ d
        checkverify(f, (Float64,))
    end

    # the derivative reproduces the primal's throw on the throwing input
    @test_throws DomainError   frule!!(Dual(checkdom,    NoTangent()), Dual(-2.0, 1.0))
    @test_throws DomainError   frule!!(Dual(checkdom_ni, NoTangent()), Dual(-2.0, 1.0))
    @test_throws ArgumentError frule!!(Dual(guarded,     NoTangent()), Dual(0.0, 1.0))
end

@testset "inactive builtins (isa, <:, nfields, sizeof, typeof, fieldtype)" begin
    # `sum(::Generator)` lowers to `Base._foldl_impl`, whose `_InitialValue` sentinel check is a
    # `Core.isa` builtin on a union-typed value (the invoke's own `Union{_InitialValue,Float64}`
    # return type) — the repro that motivated `@inactive_builtin`.
    gensum(x) = sum(sin(xi) + xi^2 for xi in x)
    x = [0.3, 0.7]
    gd = frule!!(Dual(gensum, NoTangent()), Dual(x, [1.0, 1.0]))
    @test gd.x ≈ gensum(x)
    @test gd.dx ≈ sum(cos.(x) .+ 2 .* x)
    checkverify(gensum, (Vector{Float64},))

    # `isa` against a locally-defined struct type, checked on a value whose static type is a
    # genuine runtime union (an `@noinline` call's `Union{A,B}` return) so the compiler can't fold
    # the check away — this is what exercises `vpresolve` on a value-position `GlobalRef` into this
    # module, as opposed to a `Base`/`Core` type.
    struct IsaA; v::Float64; end
    struct IsaB; v::Float64; end
    @noinline mk_isa(x, flag) = flag ? IsaA(x) : IsaB(x)
    function isa_guard(x, flag)
        r = mk_isa(x, flag)
        r isa IsaA ? r.v : 2*r.v
    end
    ia = frule!!(Dual(isa_guard, NoTangent()), Dual(3.0, 1.0), Dual(true, NoTangent()))
    @test ia.x == 3.0 && ia.dx == 1.0
    ib = frule!!(Dual(isa_guard, NoTangent()), Dual(3.0, 1.0), Dual(false, NoTangent()))
    @test ib.x == 6.0 && ib.dx == 2.0
    checkverify(isa_guard, (Float64, Bool))

    # `<:`/`typeof`/`nfields`/`fieldtype` from ordinary code, on a `Union`-typed argument so none
    # of them fold away at compile time. All four results are inactive (`Type`/`Int`/`Bool`), only
    # the final `Float64(c)` carries a tangent.
    struct ReflA; a::Int; end
    struct ReflB; a::Float64; b::Int; end
    function reflect_guard(y)
        T = typeof(y)
        n = nfields(y)
        isint = fieldtype(T, 1) <: Integer
        c = (T <: ReflA && isint) ? n : -n
        Float64(c)
    end
    checkverify(reflect_guard, (Union{ReflA,ReflB},))

    # `sizeof` on a genuinely runtime `DataType` argument (not folded, unlike `sizeof` of a small
    # concrete `Union`, which the optimizer union-splits into per-arm literals before it ever
    # becomes a `Core.Builtin` call).
    sizeof_guard(T::DataType) = Float64(sizeof(T))
    sd = frule!!(Dual(sizeof_guard, NoTangent()), Dual(Float64, NoTangent()))
    @test sd.x == 8.0 && sd.dx == 0.0
    checkverify(sizeof_guard, (DataType,))
end

@testset "exception handling (try/catch)" begin
    # try/catch dualizes: UpsilonNode/PhiCNode are duplicated into primal + shadow copies, and
    # EnterNode/:leave/:pop_exception carry over as control markers (block topology preserved
    # 1:1). The optimizer deletes provably-non-throwing try scopes, so `trycatch`/`trythrow`
    # retain a live EnterNode only because their bodies can actually throw.
    trycatch(x) = try sin(x) catch; cos(x) end       # try/catch, non-throwing path: d/dx = cos(x)
    # try/catch where the body genuinely throws on one branch (inline throw, M1) and the handler
    # differentiates (M2): x>=0 runs the body (x*x), x<0 throws and the catch returns -x.
    trythrow(x) = try (x < 0 ? throw(DomainError(x, "neg")) : x*x) catch; -x end

    # non-throwing path through a try body calling surviving frules: d/dx sin(x) = cos(x)
    t = frule!!(Dual(trycatch, NoTangent()), Dual(0.6, 1.0))
    @test t.x ≈ sin(0.6) && t.dx ≈ cos(0.6)
    checkverify(trycatch, (Float64,))

    # body throws on one branch, handler differentiates: x>=0 -> x*x (d=2x); x<0 caught -> -x (d=-1)
    rp = frule!!(Dual(trythrow, NoTangent()), Dual(3.0, 1.0))
    @test rp.x ≈ 9.0 && rp.dx ≈ 6.0                    # happy path (no throw)
    rc = frule!!(Dual(trythrow, NoTangent()), Dual(-2.0, 1.0))
    @test rc.x ≈ 2.0 && rc.dx ≈ -1.0                   # catch path (body threw)
    checkverify(trythrow, (Float64,))
end
