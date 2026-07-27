using Test
using Differ
using Differ: Dual, NoTangent, frule!!, build_tangent, zero_tangent, unit_tangent, code_dual_ircode

include(joinpath(@__DIR__, "testutils.jl"))

# Must be a true top-level function: a *self*-recursive function defined in a local scope needs to
# close over its own (boxed) binding to call itself, which gives it a real (non-`NoTangent`)
# tangent type instead of the plain singleton a top-level function has.
@noinline function rec_self(x::Float64, n::Int)
    n <= 0 && return x
    return rec_self(x, n - 1)
end

@testset "function tangents (closures)" begin
    # The function isn't always constant: a closure carries a tangent in its captured field, so
    # the derivative w.r.t. the capture flows from the function-dual's tangent (read via
    # `getfield(#self#, :a)` in the body → `get_tangent_field` on the closure's `Tangent`).
    # Under the Mooncake tangent system a closure's tangent is a `Tangent{@NamedTuple{a::…}}`;
    # `zero_tangent(f)` is the "hold f constant" tangent (all-zero captures).
    mklin(a)  = x -> a*x         # a·x        : d/dx = a,   d/da = x
    mkquad(a) = x -> a*(x*x)     # a·x²       : d/dx = 2a·x, d²/dx² = 2a

    f = mklin(2.0)                                     # a·x with a = 2
    capt(v) = build_tangent(typeof(f), v)              # closure tangent with da = v
    # d/dx (a constant): tangent = a·dx = 2
    r = frule!!(Dual(f, zero_tangent(f)), Dual(3.0, 1.0))
    @test r.x ≈ 6.0 && r.dx ≈ 2.0
    # d/da: seed the *function* tangent (da = 1), hold x fixed (dx = 0); tangent = da·x = 3
    ra = frule!!(Dual(f, capt(1.0)), Dual(3.0, 0.0))
    @test ra.x ≈ 6.0 && ra.dx ≈ 3.0
    # both directions at once (a=2, da=10, x=3, dx=1): a·dx + da·x = 2 + 30 = 32
    rb = frule!!(Dual(f, capt(10.0)), Dual(3.0, 1.0))
    @test rb.dx ≈ 32.0

    # a quadratic closure differentiated w.r.t. x, holding the capture constant: g(x)=a·x²,
    # g'(x)=2a·x. verify_ir on the dualized closure body (the capture getfield is dualized).
    g = mkquad(2.0)                                    # 2·x²
    rg = frule!!(Dual(g, zero_tangent(g)), Dual(1.5, 1.0))
    @test rg.x ≈ 2*1.5^2 && rg.dx ≈ 2*2*1.5
    Core.Compiler.verify_ir(code_dual_ircode(g, (Float64,))[1])
end

@testset "higher-order forward mode (uniform nesting)" begin
    # A second derivative differentiates the first-order dualized function itself (Option A —
    # compose the transform): the primal for a nested-dual request is the order-(k-1) dual IR,
    # re-dualized. ALL seeds — the function included — are nested uniformly to the order, with
    # NoTangent at the function's leaves. Then r.x.x = f(x), r.dx.x = r.x.dx = f'(x), r.dx.dx = f''(x).
    sqr(x)     = x*x
    mul3(x)    = (x*x)*x   # explicit 2-arg grouping; `x*x*x` would be a 3-arg (vararg) `*`, unsupported
    function p4(x)
        r = x*x
        r = r*x
        r = r*x
        r
    end
    sincosp(x) = sin(x)*cos(x)
    nest(x)    = sin(cos(x))
    plus1(x)   = sin(x) + 1
    relu(x)    = x > 0.0 ? x : -x
    function branch3(x)
        if x > 2.0
            x*x
        elseif x > 0.0
            x + 1.0
        else
            -x
        end
    end
    function sumk(x, k)
        s = x - x
        i = 0
        while i < k
            s = s + x
            i = i + 1
        end
        s
    end
    vfun(x, ys...) = x + sum(ys)   # a genuinely vararg-defined primal method — unsupported

    fseed2(f) = Dual(Dual(f, NoTangent()), Dual(f, NoTangent()))   # constant fn nested to order 2
    seed2(x)  = Dual(Dual(x, 1.0), Dual(1.0, 0.0))

    # analytic second derivatives: (x²)''=2, (x³)''=6x, (x⁴)''=12x²; and now sin(x)cos(x) too:
    # (sin·cos)'' = d/dx cos(2x) = -2 sin(2x)  (exercises the :new fix + sin/cos rewrite)
    for (f, x, fx, dfx, d2fx) in ((sqr,     2.0, 4.0,  4.0,  2.0),
                                  (mul3,    2.0, 8.0,  12.0, 12.0),
                                  (p4,      2.0, 16.0, 32.0, 48.0),
                                  (sincosp, 0.6, sin(0.6)*cos(0.6), cos(2*0.6), -2*sin(2*0.6)))
        r = frule!!(fseed2(f), seed2(x))
        @test r.x.x  ≈ fx
        @test r.x.dx ≈ dfx && r.dx.x ≈ dfx      # both first-derivative cross-terms agree
        @test r.dx.dx ≈ d2fx
    end

    # second derivative matches central finite differences of the (AD) first derivative, incl.
    # the transcendental cases now that sin/cos work to higher order
    d1(f, x) = frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx
    d2fd(f, x; h=1e-4) = (d1(f, x+h) - d1(f, x-h)) / 2h
    for (f, x) in ((sqr, 2.1), (mul3, -0.7), (p4, 1.3), (sincosp, 0.6), (nest, 0.4), (plus1, 1.1))
        @test frule!!(fseed2(f), seed2(x)).dx.dx ≈ d2fd(f, x) rtol=1e-4
    end

    # order-N general: 3rd derivative of x⁴ is 24x (= 48 at x=2) by nesting one level deeper
    # (the function is nested to order 3 as well)
    fz(f) = Dual(f, NoTangent())
    fseed3(f) = Dual(Dual(fz(f), fz(f)), Dual(fz(f), fz(f)))
    s3(x) = Dual(Dual(Dual(x,1.0),Dual(1.0,0.0)), Dual(Dual(1.0,0.0),Dual(0.0,0.0)))
    @test frule!!(fseed3(p4), s3(2.0)).dx.dx.dx ≈ 24*2.0

    # the 2nd-order transform produces valid IR, including sin/cos and through control flow (the
    # tuple-aware vararg prologue composes with phi/goto re-dualization)
    for (f, at) in ((sqr,(Float64,)), (mul3,(Float64,)), (p4,(Float64,)), (sincosp,(Float64,)),
                    (relu,(Float64,)), (branch3,(Float64,)), (sumk,(Float64,Int)))
        checkverify2(f, at)
    end
    @test true   # reached here ⇒ every verify_ir above passed

    # a non-uniformly-nested seed (function NOT nested at order 2) is no longer valid: the uniform
    # peel can't form the inner carrier, so it bails rather than miscompiling.
    @test_throws Exception frule!!(Dual(sqr, NoTangent()), seed2(2.0))

    # graceful bail still holds at higher order: a vararg primal is unsupported → ErrorException
    # (unrelated to array support — kept as the "some construct is still unsupported" regression).
    err = try
        frule!!(fseed2(vfun), seed2(1.0), seed2(2.0))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("vararg", err.msg)
end

@testset "higher-order via composed differentiation (nested frule!! / D-of-D)" begin
    # Differentiate a function that itself calls `frule!!`. The inner `frule!!` inlines into the
    # outer closure's primal IR as a surviving `dualized_impl` `:invoke`; the outer dualization
    # pass re-dualizes it (the function slot `Dual{typeof(dualized_impl),NoTangent}` is dropped
    # and the remaining nested value args peel down to the inner order-1 carrier).
    #
    # Higher-order AD written the natural way — a differentiation operator composed with itself —
    # rather than by hand-nesting `Dual` seeds. `Dop(f, x)` is `f'(x)` via one `frule!!`; nesting
    # `Dop` gives `f''` etc.
    Dop(f, x)  = frule!!(Dual(f, zero_tangent(f)), Dual(x, unit_tangent(x))).dx
    scplusx(x) = sin(x) + x
    d1_scpx(x) = Dop(scplusx, x)   # closure-free first derivative: cos(x) + 1

    # The exact form from the design goal: D(f, x) with `do` blocks, second derivative of sin+x.
    r = Dop(10.0) do x
        Dop(x) do x
            sin(x) + x
        end
    end
    @test r ≈ -sin(10.0)                               # d²/dx²(sin x + x) = -sin x

    # Named-function equivalents, checked against the analytic second derivative at several points.
    for x in (0.4, 1.3, -0.7, 2.1)
        @test Dop(scplusx, x) ≈ cos(x) + 1             # first derivative
        @test Dop(d1_scpx, x) ≈ -sin(x)                # second derivative via composed D
        @test Dop(x -> Dop(scplusx, x), x) ≈ -sin(x)   # same, as a closure literal
    end

    # Higher orders by composing D further: d³/dx³(sin x + x) = -cos x, d⁴/dx⁴ = sin x. The
    # `frule!!`-slot compose path (re-dualizing a surviving `frule!!` invoke) recurses cleanly.
    d2_scpx(x) = Dop(d1_scpx, x)
    d3_scpx(x) = Dop(d2_scpx, x)
    @test Dop(d2_scpx, 0.4) ≈ -cos(0.4)
    @test Dop(d3_scpx, 0.4) ≈  sin(0.4)

    # The re-dualized outer closure produces valid IR (verify_ir on the raw dualized IRCode).
    Core.Compiler.verify_ir(code_dual_ircode(d1_scpx, (Float64,))[1])
    @test true

    # Regression: forward-mode dualization of a self-recursive `@noinline` primal must bail
    # cleanly rather than stack-overflow. This exercises the `dualized_impl_in_progress` cycle
    # guard, whose forward-mode twist is that the recursion crosses *fresh* `ADInterpreter`
    # instances via the `frule!!` `@generated` boundary — so the guard is task-local (shared
    # across those instances) rather than a per-`interp` field like reverse mode's `in_progress`.
    # `rec_self` must be a true top-level function, not testset-local: a *self*-recursive function
    # defined in a local scope needs to close over its own (boxed) binding to call itself, which
    # gives it a real (non-`NoTangent`) tangent type instead of the plain singleton a top-level
    # function has — defeating the `Dual(rec_self, NoTangent())` call below.
    @test_throws ErrorException frule!!(Dual(rec_self, NoTangent()), Dual(1.0, 1.0), Dual(3, NoTangent()))

    # Limitation: a closure/struct with *differentiable fields* cannot be differentiated at order
    # ≥2. The self-tangent `Dual` scheme (`tangent_type(Dual{P,T}) == Dual{P,T}`) requires each
    # carried type to be its own tangent type, which fails for such a struct (its tangent is a
    # `Tangent`, not itself). Surfaces as a clear Differ error, not a miscompile. `mkquad(3.0)` is
    # a closure with a `Float64` capture, so nesting D over it lands here (whereas nesting D over
    # the plain-function `scplusx` above is fine). First-order differentiation of the same closure
    # — including w.r.t. its capture — works and is covered by the closures testset above.
    mkquad(a) = x -> a*(x*x)
    @test_throws ErrorException Dop(z -> Dop(mkquad(3.0), z), 1.7)
end

@testset "user function with a hand-written frule!! (world-age callee resolution)" begin
    # Differentiating a caller of a *user* function with a hand rule: the callee must be resolved
    # at the interpreter's inference world, not the stale generation world. First order works even
    # on a clean tree; the crash was at order ≥2 (a `Dual{GlobalRef,…}`). `@noinline` keeps the
    # callee a surviving `:invoke`. This is the exact shape of the original bug report.
    Dop(f, x)  = frule!!(Dual(f, zero_tangent(f)), Dual(x, unit_tangent(x))).dx
    @noinline hr_lin(x) = x + 1     # hand rule below; d/dx = 1,  higher derivatives = 0
    @noinline hr_sqr(x) = x*x       # hand rule below; d/dx = 2x, d²/dx² = 2, d³/dx³ = 0
    call_lin(x) = hr_lin(x)
    call_sqr(x) = hr_sqr(x)
    Differ.frule!!(::Dual{typeof(hr_lin)}, d::Dual) = Dual(d.x + 1, d.dx * one(d.x))
    Differ.frule!!(::Dual{typeof(hr_sqr)}, d::Dual) = Dual(d.x * d.x, d.dx * (2 * d.x))

    @test Dop(call_lin, 1.0) == 1.0                          # d/dx (x+1) = 1

    # The exact original bug report: D-of-D over the wrapper. Second/third derivatives of x+1 = 0.
    @test (Dop(1.0) do x; Dop(call_lin, x) end) == 0.0
    @test (Dop(1.0) do x; Dop(y -> Dop(call_lin, y), x) end) == 0.0

    # Non-linear rule so the second derivative is non-trivial: d/dx(x²)=2x, d²/dx²(x²)=2.
    for x in (0.4, 1.3, -0.7, 2.1)
        @test Dop(call_sqr, x) ≈ 2x
        @test (Dop(z -> Dop(call_sqr, z), x)) ≈ 2.0
    end

    # The dualized caller is valid IR (verify_ir on the raw dualized IRCode).
    Core.Compiler.verify_ir(code_dual_ircode(call_sqr, (Float64,))[1])
    @test true
end

@testset "allocation-free (dualization is fully inlined)" begin
    # The dualized code is real post-optimization IRCode: Duals are built with %new and
    # surviving high-level rules are `:invoke`s to CodeInstances, so a straight-line dual is
    # allocation-free. `sincosp` exercises the surviving `frule!!(sin)`/`frule!!(cos)` :invoke path.
    sqr(x)     = x*x
    sincosp(x) = sin(x)*cos(x)
    allocs(f, x) = (d = Dual(x, 1.0); df = Dual(f, NoTangent());
                    frule!!(df, d); @allocated frule!!(df, d))     # measure warmed
    @test allocs(sqr, 2.0)     == 0        # pure intrinsics
    @test allocs(sincosp, 0.6) == 0        # surviving sin/cos rule :invokes
end
