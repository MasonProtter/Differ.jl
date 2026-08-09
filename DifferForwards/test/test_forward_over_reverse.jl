# Forward-over-reverse composition: `D(x -> rev_gradient(f, x), v)`.
#
# Previously disabled by ISSUES #85 — the whole file hung, because the forward pass's transform runs
# inside `frule!!`'s generator, where dispatch is pinned to that method's `primary_world` and so
# could not see DifferReverse's `tangent_type` overrides or the extension's coupling hooks. Fixed by
# routing every such query through `Contextual.at_world`.
using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, frule!!, zero_tangent, code_dual_ircode
using DifferReverse: DifferReverse, fcodual_type, Ctx, build_ctx, rev_gradient, value_and_gradient!, zero_fcodual

include(joinpath(@__DIR__, "testutils.jl"))

# Forward-over-reverse: `D(x -> rev_gradient(f, x), v)`, i.e. dualizing a call to `rev_gradient` (or
# `value_and_gradient!`/a `Tape` pullback) itself. `rev_gradient`'s generated body is
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
# Also covered below: `sum`/`mapreduce`-family primals (which route through non-inlined nested calls
# to `Base._mapreduce`/`mapreduce_impl`, each getting a recycled inner tape via `_inner_ctx` — a
# `Ctx{<:Tape}` context handed to a `reverse_fwds_impl` carrier), a pre-allocated top-level context
# (`build_ctx(...; prealloc=true)`), and a bulk-saved argument mutation. All three were previously
# blocked by one aliasing gate in `src/forward_interp.jl` that assumed a `Ctx{<:Tape}`'s tape had no
# shadow to alias to — wrong, since `Ctx{P}`'s ordinary struct-derived tangent
# (`Tangent{@NamedTuple{tape::tangent_type(P)}}`) gives it one (ISSUES #79). A directly self-recursive
# primal (`reverse_fwds_recursive_ci`'s `Ctx{own_TapeT}` self-edge) now also works, as a consequence
# of forward mode gaining its own general recursion support (ISSUES #82, `test_forward_recursion.jl`)
# rather than anything specific to forward-over-reverse — see the "self-recursion" testset below.
# *Mutual* recursion (A→B→A) is a different, still-open reverse-mode gap (its tape types would need a
# fixed point solved across the whole SCC) — unaffected by either fix, and bails the same way with or
# without FoR; see its own pinning testset below. Reverse-over-forward and reverse-over-reverse remain
# non-goals — see the bail testset at the bottom of this file.

D(f, x) = frule!!(Dual(f, zero_tangent(f)), Dual(x, one(x))).dx

# Module-level (not testset-local) for the same reason `test_reverse_dispatch_recursion.jl`'s
# recursive test fixtures are: a self-recursive function defined in local scope closes over its own
# boxed binding, which gives it a real (non-`NoTangent`) function tangent instead of the plain
# singleton a top-level function has, defeating `rev_gradient`'s zero-tangent seeding for `f`.
@noinline function recsum(v::Vector{Float64}, i::Int)
    i > length(v) && return 0.0
    return v[i]^3 + recsum(v, i + 1)
end

# Mutually recursive (A→B→A), same module-level requirement as `recsum` above.
@noinline mutA_fr(x::Float64, n::Int) = n <= 0 ? x : 2 * mutB_fr(x, n - 1)
@noinline mutB_fr(x::Float64, n::Int) = n <= 0 ? x : 3 * mutA_fr(x, n - 1)

@testset "staging 1: scalar straight-line" begin
    # g(x) = x² + sin(x), g'(x) = 2x + cos(x), g''(x) = 2 - sin(x).
    g(x) = x*x + sin(x)
    val = D(x -> rev_gradient(g, x)[2], 2.0)
    @test val ≈ 2 - sin(2.0)

    # Cross-check against the independent forward-over-forward oracle: `code_dual_ircode(g, ...;
    # order=2)` differentiates twice via nested `Dual`s, a completely different code path from
    # forward-over-reverse. Agreement here is the strongest evidence F-over-R is actually correct,
    # not just self-consistent.
    dd = Dual(Dual(2.0, 1.0), Dual(1.0, 0.0))
    fof = frule!!(Dual(Dual(g, NoTangent()), Dual(g, NoTangent())), dd)
    @test val ≈ fof.dx.dx

    checkverify(x -> rev_gradient(g, x)[2], (Float64,))
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
    val = D(x -> rev_gradient(h, x)[2], 2.0)
    @test val ≈ 6.0

    dd = Dual(Dual(2.0, 1.0), Dual(1.0, 0.0))
    fof = frule!!(Dual(Dual(h, NoTangent()), Dual(h, NoTangent())), dd)
    @test val ≈ fof.dx.dx

    checkverify(x -> rev_gradient(h, x)[2], (Float64,))
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
    ir, _ = code_dual_ircode(DifferReverse.reverse_fwds_impl, argtypes)
    Core.Compiler.verify_ir(ir)

    # The full nested closure (`D`'s target) also verifies on its own.
    checkverify(x -> rev_gradient(sumsq, [x, 2x, 3x])[2], (Float64,))

    # Numerical: rev_gradient(sumsq, v) == 2v, so d/dx rev_gradient(sumsq, [x,2x,3x]) == [2,4,6].
    val = D(1.0) do x
        rev_gradient(sumsq, [x, 2x, 3x])[2]
    end
    @test val ≈ [2.0, 4.0, 6.0]

    # Finite-difference backstop, general-purpose (works for any shape): differentiate
    # `rev_gradient(sumsq, v(x))` w.r.t. x directly via central differences, independent of the closed
    # form above.
    vecgrad(x) = rev_gradient(sumsq, [x, 2x, 3x])[2]
    fd = (vecgrad(1.0 + 1e-6) .- vecgrad(1.0 - 1e-6)) ./ 2e-6
    @test val ≈ fd atol=1e-6
end

@testset "staging 3: non-trivial Hessian (cross terms, non-constant diagonal)" begin
    # nontrivial(v) = Σvᵢ³ + Σvᵢv_{i+1}. Gradient: gₖ = 3vₖ² + [k>1]v_{k-1} + [k<n]v_{k+1} — every
    # off-diagonal Hessian entry adjacent to the coupling term is nonzero, and the diagonal (6vᵢ)
    # is not constant, so a formula that mishandles either would be caught here (unlike `sumsq`'s
    # constant, diagonal 2I).
    val = D(1.0) do x
        rev_gradient(nontrivial, [x, 2x, 3x])[2]
    end

    fd_oracle(x) = begin
        vecgrad(t) = rev_gradient(nontrivial, [t, 2t, 3t])[2]
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

    checkverify(x -> rev_gradient(nontrivial, [x, 2x, 3x])[2], (Float64,))
end

@testset "sum-family correctness under forward-over-reverse (recycled inner tape, Ctx{<:Tape})" begin
    # `sum(abs2, v)`/`sum(f, v)` survive dualization as a non-inlined call into `Base._mapreduce`,
    # whose own reverse-mode rule recycles an inner tape via `_inner_ctx` — i.e. a `Ctx{<:Tape}`
    # context supplied to a `reverse_fwds_impl` carrier. Each case is checked three ways: against a
    # closed-form/independently-derived value, against the central-difference oracle, and for IR
    # legality.

    # f(v) = Σ vᵢ² — same closed form as `sumsq` above, reached through `sum(abs2, v)` instead of a
    # hand-written loop.
    val1 = D(1.0) do x
        rev_gradient(v -> sum(abs2, v), [x, 2x, 3x])[2]
    end
    @test val1 ≈ [2.0, 4.0, 6.0]
    fd1(x) = rev_gradient(v -> sum(abs2, v), [x, 2x, 3x])[2]
    @test val1 ≈ (fd1(1.0 + 1e-6) .- fd1(1.0 - 1e-6)) ./ 2e-6 atol = 1e-6
    checkverify(x -> rev_gradient(v -> sum(abs2, v), [x, 2x, 3x])[2], (Float64,))

    # nt(v) = (Σvᵢ²)(Σvᵢ) — the cross term between the two sums gives a genuinely non-diagonal,
    # non-constant Hessian (unlike `sumsq`'s constant 2I), so a dropped cross term can't pass.
    nt(v) = sum(abs2, v) * sum(v)
    val2 = D(1.0) do x
        rev_gradient(nt, [x, 2x, 3x])[2]
    end
    @test val2 ≈ [52.0, 76.0, 100.0]
    fd2(x) = rev_gradient(nt, [x, 2x, 3x])[2]
    @test val2 ≈ (fd2(1.0 + 1e-6) .- fd2(1.0 - 1e-6)) ./ 2e-6 atol = 1e-6
    checkverify(x -> rev_gradient(nt, [x, 2x, 3x])[2], (Float64,))

    # sum(sin, v) — a reduction with a function (ISSUES #63b), over a plain array.
    val3 = D(1.0) do x
        rev_gradient(v -> sum(sin, v), [x, 2x, 3x])[2]
    end
    @test val3 ≈ [-0.8414709848078965, -1.8185948536513634, -0.4233600241796016]
    fd3(x) = rev_gradient(v -> sum(sin, v), [x, 2x, 3x])[2]
    @test val3 ≈ (fd3(1.0 + 1e-6) .- fd3(1.0 - 1e-6)) ./ 2e-6 atol = 1e-6
    checkverify(x -> rev_gradient(v -> sum(sin, v), [x, 2x, 3x])[2], (Float64,))
end

@testset "pre-allocated context (Ctx{<:Tape}) correctness under forward-over-reverse" begin
    # IR level, mirroring the `Ctx{Nothing}` check in staging 3 above: the reverse-mode carrier's own
    # dualized IR must verify on its own, given a pre-allocated context type instead of a fresh one.
    pctx = build_ctx(sumsq, (Vector{Float64},); prealloc=true)
    argtypes = (fcodual_type(typeof(sumsq)), typeof(pctx), fcodual_type(Vector{Float64}))
    ir, _ = code_dual_ircode(DifferReverse.reverse_fwds_impl, argtypes)
    Core.Compiler.verify_ir(ir)

    # Runtime, exercising reuse: hold both the ctx *and* the outer `Dual` seed across two calls, so
    # the primal tape *and* the shadow tape are recycled. (Plain `D(inner, 1.0)` called twice would
    # only reuse the primal tape — `zero_tangent(f)` builds a fresh shadow closure every call.) This
    # is the case the `Stack.position` shadow mirroring (`src/builtins.jl`) matters for: without it, a
    # reused shadow tape's stack positions would drift out of step with the primal's.
    ctx = build_ctx(sumsq, (Vector{Float64},); prealloc=true)
    inner = let c = ctx
        x -> value_and_gradient!(c, zero_fcodual(sumsq), zero_fcodual([x, 2x, 3x]))[2]
    end
    t = zero_tangent(inner)
    r1 = frule!!(Dual(inner, t), Dual(1.0, 1.0)).dx
    r2 = frule!!(Dual(inner, t), Dual(1.0, 1.0)).dx
    @test r1 == r2 == (NoTangent(), [2.0, 4.0, 6.0])
end

@testset "self-recursion under forward-over-reverse" begin
    # `recsum(v) = Σ vᵢ³`, gradient `3vᵢ²` — reverse mode alone handles this fine, through
    # `reverse_fwds_recursive_ci`'s closed-form self-edge (`Ctx{own_TapeT}`, the same concrete Tape
    # type at every recursion depth, so no fixed point to solve).
    @test rev_gradient(recsum, [1.0, 2.0, 3.0], 1) == (NoTangent(), [3.0, 12.0, 27.0], NoTangent())

    # Forward-over-reverse of it now works too (closes ISSUES #80), as a special case of the general
    # forward-mode recursion fix (ISSUES #82): dualizing `reverse_fwds_impl`'s IR for this primal
    # walks into the self-edge's `:invoke` back into `reverse_fwds_impl` itself — the *same*
    # `reverse_fwds_impl` specialization currently being dualized one level up — and `frule_split!`'s
    # recursion resolver now recognizes that and emits a static self-`:invoke` against the bare
    # in-progress `MethodInstance` instead of recursing into `build_dual_ir` again (which is what used
    # to trip `dualized_impl_in_progress`'s cycle guard at depth 2). This is also the first test to
    # exercise `_inner_self_ctx`'s hand `frule!!` (`src/rules_ad_runtime.jl`) — previously written but
    # never actually reached by anything, since dualization always bailed before the recursive call
    # was ever emitted.
    #
    # IR level, mirroring the `Ctx{Nothing}` check in staging 3: the reverse-mode carrier's own
    # dualized IR must verify on its own for a self-recursive primal too.
    argtypes = (fcodual_type(typeof(recsum)), Ctx{Nothing}, fcodual_type(Vector{Float64}), fcodual_type(Int))
    ir, _ = code_dual_ircode(DifferReverse.reverse_fwds_impl, argtypes)
    Core.Compiler.verify_ir(ir)

    # Runtime, FD-matched.
    val = D(1.0) do x
        rev_gradient(recsum, [x, 2x, 3x], 1)[2]
    end
    @test val ≈ [6.0, 24.0, 54.0]
    fd(x) = rev_gradient(recsum, [x, 2x, 3x], 1)[2]
    @test val ≈ (fd(1.0 + 1e-6) .- fd(1.0 - 1e-6)) ./ 2e-6 atol = 1e-6
end

@testset "mutual recursion under forward-over-reverse still bails (reverse mode's own SCC gap, not forward mode's)" begin
    # Unlike direct self-recursion, mutual recursion (A→B→A) is a genuine reverse-mode gap: its tape
    # types would need a fixed-point solved across the whole SCC (`reverse_fwds_recursive_ci`'s
    # docstring), which isn't implemented — plain, non-FoR `rev_gradient` on a mutually recursive
    # primal already bails on its own, via `interp.in_progress` (`reverse_interp.jl`'s
    # `build_reverse_fwds_ir`/`build_reverse_pullback_ir`). Forward-mode recursion support (ISSUES
    # #82) doesn't touch this: it fixes forward mode's *own* dualizer, not reverse mode's tape-type
    # machinery. Pin that the bail under FoR still carries reverse mode's message
    # ("recursive reverse-mode forwards-pass build for ...") and not forward mode's ("recursive
    # dualization of ...") — i.e. forward mode gets past its own layer fine and the failure is
    # reverse mode's, propagated through unchanged.
    e_plain = try
        rev_gradient(mutA_fr, 1.0, 3)
        nothing
    catch err
        err
    end
    @test e_plain isa ErrorException
    @test occursin("reverse-mode forwards-pass build", e_plain.msg)

    e_for = try
        D(1.0) do x
            rev_gradient(mutA_fr, x, 3)[1]
        end
        nothing
    catch err
        err
    end
    @test e_for isa ErrorException
    @test occursin("reverse-mode forwards-pass build", e_for.msg)
    @test !occursin("recursive dualization of", e_for.msg)   # forward mode's own (fixed) message
end

@testset "bulk-save under forward-over-reverse (fresh and pre-allocated)" begin
    # A primal that mutates its argument array in a loop routes the write through the bulk
    # save/restore path (`_bulk_save!`/`_bulk_restore!`, `src/rules_ad_runtime.jl`), which reads and
    # writes the tape's own `Vector{Any}` buffer field (`getfield(tape, 3)`) rather than the per-block
    # comms stacks the staging tests above exercise. `bulk_sq!(v) = Σ vᵢ²` in place, then sums —
    # same closed-form gradient as `sumsq`, reached through the bulk-save path instead.
    function bulk_sq!(v::Vector{Float64})
        for i in 1:length(v)
            @inbounds v[i] = v[i] * v[i]
        end
        s = 0.0
        for i in 1:length(v)
            @inbounds s += v[i]
        end
        return s
    end

    # Fresh tape (Ctx{Nothing}).
    val = D(1.0) do x
        rev_gradient(bulk_sq!, [x, 2x, 3x])[2]
    end
    @test val ≈ [2.0, 4.0, 6.0]
    checkverify(x -> rev_gradient(bulk_sq!, [x, 2x, 3x])[2], (Float64,))

    # Pre-allocated (recycled) tape — ctx and outer Dual seed both held across two calls, so the
    # bulk-save buffer is reused, not freshly allocated each time.
    ctx = build_ctx(bulk_sq!, (Vector{Float64},); prealloc=true)
    inner = let c = ctx
        x -> value_and_gradient!(c, zero_fcodual(bulk_sq!), zero_fcodual([x, 2x, 3x]))[2]
    end
    t = zero_tangent(inner)
    r1 = frule!!(Dual(inner, t), Dual(1.0, 1.0)).dx
    r2 = frule!!(Dual(inner, t), Dual(1.0, 1.0)).dx
    @test r1 == r2 == (NoTangent(), [2.0, 4.0, 6.0])
end

@testset "reverse-over-forward and reverse-over-reverse bail, not crash or silently zero" begin
    # Both compositions are non-goals: `rev_gradient` of a function that itself calls `frule!!`
    # (reverse-over-forward) or calls `rev_gradient` (reverse-over-reverse). Both must fail cleanly (a
    # located `ErrorException`, not a `MethodError`/segfault, and never a silently-wrong zero
    # derivative) and name the actual composition rather than surfacing an unrelated internal error
    # several frames removed from the real cause. Caught at the point reverse mode resolves a callee's
    # primal function or argument types (`_composition_bail_message`, `src/reverse_interp.jl`).
    rof(x) = frule!!(Dual(sin, NoTangent()), Dual(x, 1.0)).x
    e1 = try
        rev_gradient(rof, 1.0)
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
        ("hand-ruled inner function", x -> rev_gradient(sin, x)[2]),
        ("composite inner function", x -> rev_gradient(y -> y * y, x)[2]),
    )
        e2 = try
            rev_gradient(ror, 1.0)
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
