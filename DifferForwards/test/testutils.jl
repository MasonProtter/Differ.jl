# Shared test infrastructure: IR-verification wrappers and finite-difference checks reused
# across many test files. Not test fixtures — nothing here is itself differentiated. Reverse-
# mode-specific helpers (checkverify_prealloc/checkverify_rev, tape-balance/traffic/size
# checks) live in DifferReverse.jl/test/testutils.jl instead.

using Test
using DifferForwards
using DifferForwards: code_dual_ircode, Dual, Inactive, NoTangent, frule!!, zero_dual

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
#
# Every check runs against both `f` and a trivial wrapper around it. Called directly, `f` resolves
# its hand-written rule at the top level; called through `wrapped`, the rule has to be reached the
# way real code reaches it — a call site the dualizer routes. `wrapped` is also what `checkverify`
# dualizes, because `f` itself as the top-level target hits an unrelated quirk where Julia's own
# inliner unfolds `f`'s real Base body into the generic `dualized_impl` wrapper before Differ's
# call-site hand-rule interception gets a look-in.
function check_unary(f, xs; rtol=1e-6)
    wrapped(x) = f(x)
    for g in (f, wrapped), x in xs
        d = frule!!(zero_dual(g), Dual(x, 1.0))
        @test d.x ≈ f(x)
        @test d.dx ≈ central_diff(f, x) rtol = rtol
    end
    checkverify(wrapped, (Float64,))
end

# Generic checks for a binary scalar function f(x, y): forward tangent (both argument directions)
# against central differences, plus an IR-legality check.
function check_binary(f, xys; rtol=1e-6)
    wrapped(x, y) = f(x, y)
    for g in (f, wrapped), (x, y) in xys
        dx = frule!!(zero_dual(g), Dual(x, 1.0), Dual(y, 0.0))
        @test dx.x ≈ f(x, y)
        @test dx.dx ≈ central_diff(f, x, y, 1) rtol = rtol
        dy = frule!!(zero_dual(g), Dual(x, 0.0), Dual(y, 1.0))
        @test dy.dx ≈ central_diff(f, x, y, 2) rtol = rtol
    end
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
    wrapped(x, y, z) = f(x, y, z)
    for g in (f, wrapped), (x, y, z) in xyzs
        dx = frule!!(zero_dual(g), Dual(x, 1.0), Dual(y, 0.0), Dual(z, 0.0))
        @test dx.x ≈ f(x, y, z)
        @test dx.dx ≈ central_diff3(f, x, y, z, 1) rtol = rtol
        dy = frule!!(zero_dual(g), Dual(x, 0.0), Dual(y, 1.0), Dual(z, 0.0))
        @test dy.dx ≈ central_diff3(f, x, y, z, 2) rtol = rtol
        dz = frule!!(zero_dual(g), Dual(x, 0.0), Dual(y, 0.0), Dual(z, 1.0))
        @test dz.dx ≈ central_diff3(f, x, y, z, 3) rtol = rtol
    end
    checkverify(wrapped, (Float64, Float64, Float64))
end

# `f(ν, x)` with an integer order: `ν` has no tangent space at all, so only the `x` tangent is
# checked, and the shadow the transform would hand the rule for `ν` is `NoTangent()`.
function check_order(f, ν::Integer, xs; rtol=1e-6)
    wrapped(ν, x) = f(ν, x)
    for g in (f, wrapped), x in xs
        d = frule!!(zero_dual(g), Dual(ν, NoTangent()), Dual(x, 1.0))
        @test d.x ≈ f(ν, x)
        @test d.dx ≈ central_diff(t -> f(ν, t), x) rtol = rtol
    end
    checkverify(wrapped, (typeof(ν), Float64))
end

# `f(a, x)` with a real parameter whose derivative is not implemented: `a` is held constant.
function check_param(f, a::Real, xs; rtol=1e-6)
    wrapped(a, x) = f(a, x)
    for g in (f, wrapped), x in xs
        d = frule!!(zero_dual(g), const_dual(a), Dual(x, 1.0))
        @test d.x ≈ f(a, x)
        @test d.dx ≈ central_diff(t -> f(a, t), x) rtol = rtol
    end
    checkverify(wrapped, (typeof(a), Float64); inactive=(1,))
end
