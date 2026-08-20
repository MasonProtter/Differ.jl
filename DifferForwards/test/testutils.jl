# Shared test infrastructure: IR-verification wrappers and finite-difference checks reused
# across many test files. Not test fixtures — nothing here is itself differentiated. Reverse-
# mode-specific helpers (checkverify_prealloc/checkverify_rev, tape-balance/traffic/size
# checks) live in DifferReverse.jl/test/testutils.jl instead.

using Test
using DifferForwards
using DifferForwards: code_dual_ircode, Dual, Inactive, NoTangent, frule!!

# Central finite difference, one argument.
central_diff(f, x; h=1e-6) = (f(x + h) - f(x - h)) / 2h

# Central finite difference of a 2-argument function, w.r.t. argument `k` (1 or 2).
central_diff(f, x, y, k::Int; h=1e-6) =
    k == 1 ? (f(x+h, y) - f(x-h, y)) / 2h : (f(x, y+h) - f(x, y-h)) / 2h

# A `Dual` carrier for an argument the caller holds constant.
const_dual(x) = Dual(x, Inactive())

# Forward-mode dualized IR is legal (order-1). `inactive` names argument positions held constant.
checkverify(f, at; inactive=()) =
    Core.Compiler.verify_ir(code_dual_ircode(f, at; inactive)[1])

# The bail message for a function Differ declines to dualize, or `nothing` if it dualizes fine.
# Every graceful bail is supposed to name a *reason*, so tests assert on this rather than just on
# "it threw".
function bail_reason(f, at; inactive=())
    try
        code_dual_ircode(f, at; inactive)
        return nothing
    catch e
        return sprint(showerror, e)
    end
end

# Forward-mode dualized IR is legal at a given nesting order.
checkverify2(f, at; order=2, inactive=()) =
    Core.Compiler.verify_ir(code_dual_ircode(f, at; order, inactive)[1])

# Generic checks for a unary scalar function: forward tangent against central differences, at
# every x in xs, plus an IR-legality check. Reverse mode's counterpart lives in
# DifferReverse/test/testutils.jl.
function check_unary(f, xs; rtol=1e-6)
    for x in xs
        d = frule!!(Dual(f, NoTangent()), Dual(x, 1.0))
        @test d.x ≈ f(x)
        @test d.dx ≈ central_diff(f, x) rtol = rtol
    end
    # `checkverify` dualizes the trivial wrapper rather than `f` directly: passing `f` itself as
    # the top-level dualization target hits an unrelated quirk where Julia's own inliner unfolds
    # `f`'s real Base body into the generic `dualized_impl` wrapper before Differ's call-site
    # hand-rule interception gets a look-in. Wrapping one level deep, as any real caller of `f`
    # would look, sidesteps that and exercises the interception path this test actually cares about.
    wrapped(x) = f(x)
    checkverify(wrapped, (Float64,))
end

# Generic checks for a binary scalar function f(x, y): forward tangent (both argument directions)
# against central differences, plus an IR-legality check.
function check_binary(f, xys; rtol=1e-6)
    for (x, y) in xys
        dx = frule!!(Dual(f, NoTangent()), Dual(x, 1.0), Dual(y, 0.0))
        @test dx.x ≈ f(x, y)
        @test dx.dx ≈ central_diff(f, x, y, 1) rtol = rtol
        dy = frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 1.0))
        @test dy.dx ≈ central_diff(f, x, y, 2) rtol = rtol
    end
    # See the comment in `check_unary` for why `f` is wrapped before verifying.
    wrapped(x, y) = f(x, y)
    checkverify(wrapped, (Float64, Float64))
end

# Central difference of a 3-argument function w.r.t. argument k (1, 2, or 3), mirroring
# `central_diff(f, x, y, k)` above for one more argument.
function central_diff3(f, x, y, z, k::Int; h=1e-6)
    if k == 1
        return (f(x + h, y, z) - f(x - h, y, z)) / 2h
    elseif k == 2
        return (f(x, y + h, z) - f(x, y - h, z)) / 2h
    else
        return (f(x, y, z + h) - f(x, y, z - h)) / 2h
    end
end

# Generic checks for a ternary scalar function f(x, y, z), mirroring `check_binary`.
function check_ternary(f, xyzs; rtol=1e-6)
    for (x, y, z) in xyzs
        dx = frule!!(Dual(f, NoTangent()), Dual(x, 1.0), Dual(y, 0.0), Dual(z, 0.0))
        @test dx.x ≈ f(x, y, z)
        @test dx.dx ≈ central_diff3(f, x, y, z, 1) rtol = rtol
        dy = frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 1.0), Dual(z, 0.0))
        @test dy.dx ≈ central_diff3(f, x, y, z, 2) rtol = rtol
        dz = frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 0.0), Dual(z, 1.0))
        @test dz.dx ≈ central_diff3(f, x, y, z, 3) rtol = rtol
    end
    # See the comment in `check_unary` for why `f` is wrapped before verifying.
    wrapped(x, y, z) = f(x, y, z)
    checkverify(wrapped, (Float64, Float64, Float64))
end
