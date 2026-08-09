using Test
using Differ
using Differ: Dual, NoTangent, frule!!, build_tangent, zero_tangent, unit_tangent, code_dual_ircode

include(joinpath(@__DIR__, "testutils.jl"))

# Must be top-level: a self-recursive function defined in local scope needs to close over its own
# (boxed) binding to call itself, which gives it a real (non-`NoTangent`) tangent type instead of
# the plain singleton a top-level function has.
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

@testset "dynamically-selected closure callee, non-inlined" begin
    # Regression pin for two bugs in shared forward-mode code (`frule_split!` and
    # `frule_codeinstance`, `src/forward_interp.jl`), both only ever exercised before this test via
    # forward-over-reverse (a callee's own tangent flowing through `frule_split!` when the callee is a
    # pullback closure popped off a `Tape`). Reproduced here with no reverse-mode machinery involved: a
    # concretely-typed callee whose value is chosen at run time (by a branch) and carries a real
    # differentiable capture, called as a surviving (non-inlined) call — `frule_split!`'s static path
    # (as opposed to the `dynamic_frule` trampoline, which only fires for a non-concrete
    # callee/argument type).
    #
    # `mkscale` makes a closure over a captured `Float64`; its type is fixed by the method, so both
    # branches below produce the *same* concrete closure type with *different* captured values —
    # `ftype` is concrete (`_conc(ftype)` true), but the callee is a genuine `SSAValue`/`PhiNode`
    # result, not something `_calleeval` can resolve to a compile-time constant. `@noinline` is on
    # `scale`, not `pick_scale`: `pick_scale` is cheap and inlines away, leaving `f` as a two-way
    # phi merging `%new(closureT, a)`/`%new(closureT, b)`; the closure *call* `f(x)` is what must
    # survive as a static `:invoke`.
    function mkscale(a::Float64)
        @noinline scale(x::Float64) = a * x
        return scale
    end
    pick_scale(branch::Bool, a::Float64, b::Float64) = branch ? mkscale(a) : mkscale(b)
    call_dynamic_scale(branch::Bool, a::Float64, b::Float64, x::Float64) =
        pick_scale(branch, a, b)(x)

    # Evidence this hits the intended path rather than some other one: the dualized IR still
    # contains exactly one `:invoke` of `frule!!`, and its callee operand's declared type is
    # `Dual{closureT, Tangent{...}}` — a real capture tangent, not the `NoTangent` bug 1 hardcoded
    # (which would also mismatch the `CodeInstance` bug 2 resolves, crashing at codegen — see below).
    ir, _ = code_dual_ircode(call_dynamic_scale, (Bool, Float64, Float64, Float64))
    Core.Compiler.verify_ir(ir)
    invokes = [stmt for stmt in ir.stmts.stmt
               if isa(stmt, Expr) && stmt.head === :invoke && length(stmt.args) >= 3 &&
                  stmt.args[2] === frule!!]
    @test length(invokes) == 1
    callee_op = only(invokes).args[3]
    @test callee_op isa Core.SSAValue
    callee_dualty = ir.stmts.type[callee_op.id]
    @test callee_dualty <: Dual
    @test Differ._dual_tangent_type(callee_dualty) != NoTangent

    # Numerical correctness: f(x) = a·x if branch, b·x otherwise; only the selected capture's
    # derivative should be nonzero. Actually *running* the call matters for bug 2: it fails at
    # codegen ("Unreachable reached"), which `verify_ir` above does not catch.
    seed(v, dv) = Dual(v, dv)
    r_da = frule!!(Dual(call_dynamic_scale, NoTangent()), seed(true, NoTangent()),
                   seed(2.0, 1.0), seed(3.0, 0.0), seed(5.0, 0.0))
    @test r_da.x ≈ 10.0
    @test r_da.dx ≈ 5.0                 # d/da (a·x) = x, branch picks `a`
    r_db = frule!!(Dual(call_dynamic_scale, NoTangent()), seed(true, NoTangent()),
                   seed(2.0, 0.0), seed(3.0, 1.0), seed(5.0, 0.0))
    @test r_db.dx ≈ 0.0                 # `b` unused when branch picks `a`
    r_dx = frule!!(Dual(call_dynamic_scale, NoTangent()), seed(true, NoTangent()),
                   seed(2.0, 0.0), seed(3.0, 0.0), seed(5.0, 1.0))
    @test r_dx.dx ≈ 2.0                 # d/dx (a·x) = a
    r_false = frule!!(Dual(call_dynamic_scale, NoTangent()), seed(false, NoTangent()),
                       seed(2.0, 0.0), seed(3.0, 1.0), seed(5.0, 0.0))
    @test r_false.x ≈ 15.0
    @test r_false.dx ≈ 5.0              # branch flips: now `b`'s derivative (= x) is live
end

@testset "higher-order forward mode (uniform nesting)" begin
    # A second derivative differentiates the first-order dualized function itself (Option A —
    # compose the transform): the primal for a nested-dual request is the order-(k-1) dual IR,
    # re-dualized. ALL seeds — the function included — are nested uniformly to the order, with
    # NoTangent at the function's leaves. Then r.x.x = f(x), r.dx.x = r.x.dx = f'(x), r.dx.dx = f''(x).
    sqr(x)     = x*x
    mul3(x)    = (x*x)*x   # explicit 2-arg grouping (`x*x*x` inlines to the same two `mul_float`s)
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
    vfun(x, ys...) = x + sum(ys)   # a genuinely vararg-defined primal method

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
    # tuple-aware vararg prologue composes with phi/goto re-dualization), and over a vararg primal
    # (`vfun`, and Base's 3-arg `*(a,b,c,xs...)`): `compose(0)` re-dualizes the order-1 carrier,
    # whose argtypes are already flat, so the primal's vararg-ness is fully absorbed one level down
    # and never reaches the higher-order branch.
    for (f, at) in ((sqr,(Float64,)), (mul3,(Float64,)), (p4,(Float64,)), (sincosp,(Float64,)),
                    (relu,(Float64,)), (branch3,(Float64,)), (sumk,(Float64,Int)),
                    (vfun,(Float64,Float64,Float64)), (*,(Float64,Float64,Float64)))
        checkverify2(f, at)
    end
    @test true   # reached here ⇒ every verify_ir above passed

    # a non-uniformly-nested seed (function NOT nested at order 2) is no longer valid: the uniform
    # peel can't form the inner carrier, so it bails rather than miscompiling.
    @test_throws Exception frule!!(Dual(sqr, NoTangent()), seed2(2.0))

    # second derivative of a vararg primal. d²/dx² (x + y + z) = 0 in each variable, so use a
    # nonlinear one: `vsq(x, ys...) = x*x + sum(ys)` has ∂²/∂x² = 2.
    vsq(x, ys...) = x*x + sum(ys)
    zero2 = Dual(Dual(3.0, 0.0), Dual(0.0, 0.0))     # a non-seeded order-2 argument
    rv = frule!!(fseed2(vsq), seed2(2.0), zero2)
    @test rv.x.x  ≈ 2.0^2 + 3.0
    @test rv.x.dx ≈ 4.0 && rv.dx.x ≈ 4.0
    @test rv.dx.dx ≈ 2.0
    @test frule!!(fseed2(vfun), seed2(1.0), seed2(2.0)).dx.dx ≈ 0.0   # x+y is linear ⇒ f'' = 0

    # graceful bail regression: splatting something whose length is NOT statically known leaves a
    # `Core._apply_iterate` (plus `Core.svec`) in the primal IR, which has no dualization rule.
    # Vararg *methods* are supported; a splat *call site* over a non-tuple is not.
    vsplat(x, v::Vector{Float64}) = vfun(x, v...)
    err = try
        frule!!(Dual(vsplat, NoTangent()), Dual(1.0, 1.0), Dual([2.0, 3.0], [0.0, 0.0]))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("builtin", err.msg)
end

@testset "higher-order via composed differentiation (nested frule!! / D-of-D)" begin
    # Differentiate a function that itself calls `frule!!`. The inner `frule!!` inlines into the
    # outer closure's primal IR as a surviving `dualized_impl` `:invoke`; the outer dualization
    # pass re-dualizes it (the function slot `Dual{typeof(dualized_impl),NoTangent}` is dropped
    # and the remaining nested value args peel down to the inner order-1 carrier).
    #
    # Higher-order AD written the natural way, as a differentiation operator composed with itself,
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

    # Self-recursion, direct correctness check (used to be a `@test_throws` regression — forward mode
    # had no recursion support at all and bailed cleanly via the `dualized_impl_in_progress` cycle
    # guard; ISSUES #82 gave `frule_split!` a resolver that emits a static self-`:invoke` for exactly
    # this shape instead, so this now runs and must give the right answer, not just avoid crashing).
    # `rec_self(x,n) = n<=0 ? x : rec_self(x,n-1)` is the identity in `x` for any `n`, so `d/dx == 1`
    # regardless of recursion depth. `rec_self` must be top-level, not testset-local, for the same
    # reason as the top of this file: a testset-local self-recursive function boxes its own binding,
    # giving it a real (non-`NoTangent`) tangent type that would defeat the `Dual(rec_self,
    # NoTangent())` call below.
    @test frule!!(Dual(rec_self, NoTangent()), Dual(1.0, 1.0), Dual(3, NoTangent())).dx == 1.0

    # Previously a limitation: a closure/struct with differentiable fields could not be
    # differentiated at order ≥2. The blocker was its shadow being built by a `build_tangent` call,
    # an opaque `@generated` function the outer dualization pass couldn't re-dualize, so it bailed.
    # Now that a `Tangent`/`MutableTangent` shadow is emitted as plain `%new`s (which re-dualize like
    # any other struct construction), this composes cleanly. `mkquad(3.0)` is a closure with a
    # `Float64` capture, so nesting D over it exercises exactly this path: d²/dx²[3x²] = 6.
    mkquad(a) = x -> a*(x*x)
    @test Dop(z -> Dop(mkquad(3.0), z), 1.7) ≈ 6.0
    @test Dop(z -> Dop(mkquad(-1.5), z), 0.3) ≈ -3.0     # d²/dx²[-1.5 x²] = -3
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
