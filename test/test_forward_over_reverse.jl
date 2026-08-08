using Test
using Differ
using Differ: Dual, NoTangent, frule!!, zero_tangent, code_dual_ircode
using Differ: fcodual_type, Ctx, build_ctx, gradient, value_and_gradient!, zero_fcodual

include(joinpath(@__DIR__, "testutils.jl"))

# Forward-over-reverse: `D(x -> gradient(f, x), v)`, i.e. dualizing a call to `gradient` (or
# `value_and_gradient!`/a `Tape` pullback) itself. `gradient`'s generated body is
# `invoke(reverse_fwds_impl, cinst, ...)`/`invoke(reverse_pullback_impl, cinst, ...)` against an
# already-compiled `CodeInstance`, and re-dualizing that surviving invoke needs forward mode to see the
# real reverse-mode-optimized IR for those carriers rather than their generic error-stub bodies (the
# hook in `_build_dual_ir`, `src/forward_interp.jl`). That plus real (non-`NoTangent`) `tangent_type`s
# for `Tape`/`Stack`/`SingletonStack`/`CommsCell` (`src/stack.jl`, `src/reverse_interp.jl`) and hand
# `frule!!`s for the stack runtime's own primitives (`src/rules_ad_runtime.jl`) is what makes this work.
#
# Staging:
#   1. scalar straight-line
#   2. scalar with a loop (block `Stack{Int32}` + real comms `Stack`s)
#   3. `Vector{Float64}`, hand-written loop — array indexing, `MemoryRef` comms items, real
#      block-stack traffic.
#
# `sum(abs2, v)` is explicitly not supported (see the "not `sum`" testset below): it routes through
# `Base._mapreduce`/`mapreduce_impl`, which survive as non-inlined nested calls, each getting a
# recycled inner tape via `_inner_ctx` — a `Ctx{<:Tape}`, which the hook above bails on cleanly (only
# the fresh-tape `Ctx{Nothing}` context is supported). Reaching the `sum` spelling needs recycled-tape
# support, a substantially larger job than array support — deferred.

D(f, x) = frule!!(Dual(f, zero_tangent(f)), Dual(x, one(x))).dx

@testset "staging 1: scalar straight-line" begin
    # g(x) = x² + sin(x), g'(x) = 2x + cos(x), g''(x) = 2 - sin(x).
    g(x) = x*x + sin(x)
    val = D(x -> gradient(g, x)[2], 2.0)
    @test val ≈ 2 - sin(2.0)

    # Cross-check against the independent forward-over-forward oracle: `code_dual_ircode(g, ...;
    # order=2)` differentiates twice via nested `Dual`s, a completely different code path from
    # forward-over-reverse. Agreement here is the strongest evidence F-over-R is actually correct,
    # not just self-consistent.
    dd = Dual(Dual(2.0, 1.0), Dual(1.0, 0.0))
    fof = frule!!(Dual(Dual(g, NoTangent()), Dual(g, NoTangent())), dd)
    @test val ≈ fof.dx.dx

    checkverify(x -> gradient(g, x)[2], (Float64,))
end

@testset "staging 2: scalar with a loop" begin
    # h(x) = 3x² (three iterations of x*x), h'(x) = 6x, h''(x) = 6.
    function h(x)
        s = 0.0
        for i in 1:3
            s += x*x
        end
        return s
    end
    val = D(x -> gradient(h, x)[2], 2.0)
    @test val ≈ 6.0

    dd = Dual(Dual(2.0, 1.0), Dual(1.0, 0.0))
    fof = frule!!(Dual(Dual(h, NoTangent()), Dual(h, NoTangent())), dd)
    @test val ≈ fof.dx.dx

    checkverify(x -> gradient(h, x)[2], (Float64,))
end

# f(v) = Σ vᵢ² has Hessian 2I — constant and diagonal. A forward-over-reverse implementation could
# be wrong in ways this alone can't detect: mishandled off-diagonal coupling, or a term that's
# dropped but happens to be zero here. `nontrivial` below (cubic diagonal + adjacent coupling) has
# a real, non-constant, non-diagonal Hessian and is the test that actually exercises those paths.
sumsq(v) = begin
    s = 0.0
    for i in eachindex(v)
        s += v[i] * v[i]
    end
    return s
end

function nontrivial(v)
    s = 0.0
    n = length(v)
    for i in 1:n
        s += v[i] * v[i] * v[i]
    end
    for i in 1:n-1
        s += v[i] * v[i+1]
    end
    return s
end

@testset "staging 3: Vector{Float64}, hand-written loop" begin
    # IR-level, before any live call: the reverse-mode carrier's own dualized IR must verify on its
    # own, independent of ever being run.
    argtypes = (fcodual_type(typeof(sumsq)), Ctx{Nothing}, fcodual_type(Vector{Float64}))
    ir, _ = code_dual_ircode(Differ.reverse_fwds_impl, argtypes)
    Core.Compiler.verify_ir(ir)

    # The full nested closure (`D`'s target) also verifies on its own.
    checkverify(x -> gradient(sumsq, [x, 2x, 3x])[2], (Float64,))

    # Numerical: gradient(sumsq, v) == 2v, so d/dx gradient(sumsq, [x,2x,3x]) == [2,4,6].
    val = D(1.0) do x
        gradient(sumsq, [x, 2x, 3x])[2]
    end
    @test val ≈ [2.0, 4.0, 6.0]

    # Finite-difference backstop, general-purpose (works for any shape): differentiate
    # `gradient(sumsq, v(x))` w.r.t. x directly via central differences, independent of the closed
    # form above.
    vecgrad(x) = gradient(sumsq, [x, 2x, 3x])[2]
    fd = (vecgrad(1.0 + 1e-6) .- vecgrad(1.0 - 1e-6)) ./ 2e-6
    @test val ≈ fd atol=1e-6
end

@testset "staging 3: non-trivial Hessian (cross terms, non-constant diagonal)" begin
    # nontrivial(v) = Σvᵢ³ + Σvᵢv_{i+1}. Gradient: gₖ = 3vₖ² + [k>1]v_{k-1} + [k<n]v_{k+1} — every
    # off-diagonal Hessian entry adjacent to the coupling term is nonzero, and the diagonal (6vᵢ)
    # is not constant, so a formula that mishandles either would be caught here (unlike `sumsq`'s
    # constant, diagonal 2I).
    val = D(1.0) do x
        gradient(nontrivial, [x, 2x, 3x])[2]
    end

    fd_oracle(x) = begin
        vecgrad(t) = gradient(nontrivial, [t, 2t, 3t])[2]
        h = 1e-6
        (vecgrad(x + h) .- vecgrad(x - h)) ./ 2h
    end
    fd = fd_oracle(1.0)
    @test val ≈ fd atol=1e-6

    # Independently-derived closed form (total derivative of gₖ(v(x)) w.r.t. x, v(x)=[x,2x,3x]):
    #   g₁ = 3x² + 2x        → dg₁/dx = 6x + 2
    #   g₂ = 12x² + 4x       → dg₂/dx = 24x + 4
    #   g₃ = 27x² + 2x       → dg₃/dx = 54x + 2
    # At x=1: [8, 28, 56]. Kept as a secondary check alongside FD, not just a single hand-derived
    # constant.
    @test val ≈ [8.0, 28.0, 56.0]

    checkverify(x -> gradient(nontrivial, [x, 2x, 3x])[2], (Float64,))
end

@testset "not `sum`: sum(abs2, v) bails on a recycled inner tape (Ctx{<:Tape})" begin
    # `sum(abs2, v)` survives dualization as a non-inlined call into `Base._mapreduce`, whose own
    # reverse-mode rule recycles an inner tape via `_inner_ctx` — i.e. a `Ctx{<:Tape}` context supplied
    # to a `reverse_fwds_impl` carrier, which forward-over-reverse's aliasing restriction
    # (`src/forward_interp.jl`) explicitly declines: with no shadow tape of its own threaded in from
    # outside, there's nothing to alias the pullback's shadow tape to. Confirms this is a clean,
    # located bail — not a silent zero and not a crash.
    e = try
        D(1.0) do x
            gradient(v -> sum(abs2, v), [x, 2x, 3x])[2]
        end
        nothing
    catch err
        err
    end
    @test e isa ErrorException
    @test occursin("Ctx", e.msg)
    @test occursin("not yet supported", e.msg)
end

@testset "Ctx{<:Tape} (pre-allocated context) bails cleanly" begin
    # Same underlying restriction as the `sum` case above, triggered directly: a pre-allocated
    # (`build_ctx(...; prealloc=true)`) context's tape is supplied from outside `reverse_fwds_impl`,
    # so forward-over-reverse has no shadow tape to alias it to. `gradient`/`build_ctx(...;
    # prealloc=false)` (what plain `gradient` and the staging tests above use) are unaffected — this
    # is specifically about the pre-allocated path.
    ctx = build_ctx(sumsq, (Vector{Float64},); prealloc=true)
    inner(x) = value_and_gradient!(ctx, zero_fcodual(sumsq), zero_fcodual([x, 2x, 3x]))[2]
    e = try
        D(inner, 1.0)
        nothing
    catch err
        err
    end
    @test e isa ErrorException
    @test occursin("pre-allocated context", e.msg)
end

@testset "reverse-over-forward and reverse-over-reverse bail, not crash or silently zero" begin
    # Both compositions are non-goals: `gradient` of a function that itself calls `frule!!`
    # (reverse-over-forward) or calls `gradient` (reverse-over-reverse). Both must fail cleanly (a
    # located `ErrorException`, not a `MethodError`/segfault, and never a silently-wrong zero
    # derivative) and name the actual composition rather than surfacing an unrelated internal error
    # several frames removed from the real cause. Caught at the point reverse mode resolves a callee's
    # primal function or argument types (`_composition_bail_message`, `src/reverse_interp.jl`).
    rof(x) = frule!!(Dual(sin, NoTangent()), Dual(x, 1.0)).x
    e1 = try
        gradient(rof, 1.0)
        nothing
    catch err
        err
    end
    @test e1 isa ErrorException
    @test !(e1 isa MethodError)
    @test occursin("reverse-over-forward", e1.msg)
    @test occursin("not supported", e1.msg)

    # Reverse-over-reverse: two genuinely different shapes for the *inner* differentiated function,
    # each reaching the composition check from a different chokepoint. `sin` (a hand-ruled primitive)
    # survives dualization as a call to `rrule!!` itself, protected from inlining by
    # `_is_reverse_carrier_mi` — caught by `resolve_reverse_primal`. `y -> y*y` (a composite function,
    # no hand rule — the spelling a user is more likely to actually write) recurses through
    # `_static_recursible_call`/`reverse_fwds_recursive_ci` as an ordinary recursive call whose callee
    # happens to be `rrule!!` — caught by the composition check added directly in
    # `_static_recursible_call`, which otherwise fires its own generic "non-trivial-fdata result"
    # rejection first (real and located, but naming `Tuple{CoDual,Tape}`, not the composition). Only
    # testing one shape is what let the composite one regress silently before.
    for (label, ror) in (
        ("hand-ruled inner function", x -> gradient(sin, x)[2]),
        ("composite inner function", x -> gradient(y -> y * y, x)[2]),
    )
        e2 = try
            gradient(ror, 1.0)
            nothing
        catch err
            err
        end
        @test e2 isa ErrorException
        @test !(e2 isa MethodError)
        @test occursin("reverse-over-reverse", e2.msg)
        @test occursin("not supported", e2.msg)
    end
end
