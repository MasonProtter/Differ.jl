# Benchmark workloads, as a `BenchmarkGroup`. Two modes, in one suite:
#
#  * **reverse** (`rrule!!` + pullback), keys unprefixed,
#  * **forward** (`frule!!`), keys prefixed `fwd `.
#
# Each workload names what it exercises, because the point of the set is *coverage*, not a single
# headline number: an optimization that helps array mutation should show up here as helping the
# mutation workloads and leaving the rest alone. The `:guard` workloads exist to be unchanged — they
# are how a per-call regression (something added to every tape, say) gets caught.
#
# Where a primal is shared between the two modes (`readonly`, `wrloop`, `structloop`, `scalarcf`,
# `straightline!`) that is deliberate: the same function measured both ways is the only honest way
# to say what one mode costs relative to the other on a given shape.
#
# Two things every *reverse* workload has to get right, both of which have bitten this benchmark
# already:
#
#  * **Reset the shadow in `setup`.** A pullback *accumulates* into the shadow. Reuse one across
#    samples and you are timing an ever-growing accumulation, not a call. The primal needs no such
#    care — the pullback restores it — but the shadow does.
#  * **Reset the shadow in the benchmarked body too, wherever it carries fdata.** BenchmarkTools runs
#    `setup` once per *sample*, not per eval, so `evals>1` reuses one shadow across the whole eval
#    group; without a reset that times an ever-growing accumulation instead of a call. The fix is a
#    `set_to_zero!!(cd.dx)` (or the equivalent `.= 0` for a plain array) as the first statement of the
#    benchmarked expression — allocation-free, so it doesn't distort the timing — which is what lets
#    every workload here pick its own `evals` instead of being pinned to `evals=1`.
#
# Forward mode has neither problem: a shadow is *written*, not accumulated into, and there is no
# tape, so `evals>1` is safe throughout and setup is just "build the `Dual`s and warm the call".
#
# Adding one: give it a `note` saying which part of the transform it stresses, and prefer a shape
# whose cost is dominated by the thing being measured. `N` iterations of a trivial body is usually
# better than a realistic function, whose cost is spread across everything.

using BenchmarkTools
using Differ
using Differ: CoDual, NoRData, rrule!!, build_ctx, zero_fcodual, set_to_zero!!
using Differ: Dual, NoTangent, frule!!, zero_tangent, build_tangent

mutable struct BenchPoint
    x::Float64
    y::Float64
end

# --- primal functions -------------------------------------------------------

memloop!(o::Memory{Float64}, x::Float64, N::Int) =
    (for i in 1:N; @inbounds o[i] = x; end; nothing)

vecloop!(v::Vector{Float64}, x::Float64) =
    (for i in 1:length(v); @inbounds v[i] = x; end; nothing)

function wrloop(v::Vector{Float64}, a::Float64)
    for i in 1:length(v)
        @inbounds v[i] = a * i
    end
    s = 0.0
    for i in 1:length(v)
        @inbounds s += v[i]
    end
    return s
end

straightline!(v::Vector{Float64}, a::Float64) = (v[1] = a; v[2] = 2a; v[1] + v[2])

function structloop(p::BenchPoint, a::Float64, N::Int)
    for _ in 1:N
        p.x = p.x * 0.5 + a
    end
    return p.x
end

function readonly(v::Vector{Float64})
    s = 0.0
    for i in 1:length(v)
        @inbounds s += v[i] * v[i]
    end
    return s
end

function scalarcf(x::Float64, N::Int)
    s = 0.0
    for i in 1:N
        s = s > 1.0 ? s * 0.5 + x : s + x * x
    end
    return s
end

# Forward-only primals.

# Straight-line scalar chain: no branches, so the dualized loop body is nothing but the arithmetic
# rules. The floor for what a `Dual` costs per operation.
polychain(x::Float64, N::Int) = (s = x; for _ in 1:N; s = s * 0.5 + 1.0; end; s)

# Same write loop as `vecloop!`, but returning a scalar. The forward suite now differentiates
# `vecloop!` itself directly (ISSUES.md #53 fixed — a `nothing`-returning primal used to fail
# forward-mode IR verification); this scalar-returning variant is retained as a baseline next to it.
# The returned element is negligible next to the loop.
vecloop_ret!(v::Vector{Float64}, x::Float64) =
    (for i in 1:length(v); @inbounds v[i] = x * i; end; @inbounds v[1])

# Higher-order: the callee is a *value*, so it is carried as a `Dual` and applied through the
# dual calling convention on every iteration rather than being resolved statically.
applyN(g, x::Float64, N::Int) = (s = x; for _ in 1:N; s = g(s); end; s)

# A closure with a differentiable captured field: its tangent is a `Tangent` over the capture, so
# the callee `Dual` carries real tangent data instead of `NoTangent()`.
make_affine(a::Float64) = x -> a * x + a

# Complex arithmetic: under the Mooncake tangent system `ComplexF64` is a *struct*, so the shadow is
# a `Tangent{@NamedTuple{re,im}}` built and torn apart per operation — the non-scalar tangent shape
# that no array workload covers.
cpoly(z::ComplexF64, N::Int) = (s = z; for _ in 1:N; s = s * z + z; end; real(s))

# --- metadata ---------------------------------------------------------------
# Keyed by the same name as the `BenchmarkGroup` entry. `f`/`argtypes` are here so the runner can
# report tape shape (comms bytes, whether the comms stacks are pointer-free) alongside the timing —
# the structural metric these optimizations actually move. Tape shape is reverse-mode-only; the
# forward carrier has no tape, so `mode` is what tells the harness not to report one.

struct WorkloadMeta
    f::Any
    argtypes::Tuple
    kind::Symbol          # :mutation (expected to move) | :guard (expected not to) | :primal
    note::String
    mode::Symbol          # :reverse | :forward
end
WorkloadMeta(f, argtypes, kind, note) = WorkloadMeta(f, argtypes, kind, note, :reverse)

# --- primal baselines -------------------------------------------------------
# Every workload also registers the *same call, undifferentiated*, under `"$k $PRIMAL_SUFFIX"`. That
# is what an AD number has to be read against — an absolute time says nothing on its own, and the
# ratio is the quantity that has to come down. The harness pairs the two keys up and prints them on
# one row; it does not give the primal a row of its own.
#
# Their `setup` is deliberately the *same shape* as the AD workload's (same arrays, same lengths,
# same warm-up call), so the ratio compares two calls on identical state and not, say, a cold array
# against a warm one.
const PRIMAL_SUFFIX = "[primal]"
primal_key(k) = "$k $PRIMAL_SUFFIX"
is_primal_key(k) = endswith(k, PRIMAL_SUFFIX)

# Register the primal baseline for workload `k`, inheriting its `f`/`argtypes`/`mode`.
function primal!(suite::BenchmarkGroup, meta::Dict{String,WorkloadMeta}, k::String, b)
    m = meta[k]
    suite[primal_key(k)] = b
    meta[primal_key(k)] = WorkloadMeta(m.f, m.argtypes, :primal, "undifferentiated `$k`", m.mode)
    return nothing
end

"""
    benchmark_group(; N=1000, mode=:all) -> (suite::BenchmarkGroup, meta::Dict{String,WorkloadMeta})

The benchmark suite. `mode` selects `:reverse`, `:forward`, or `:all` (both). Run with
`run(suite)`, or via `bench/run.jl`.
"""
function benchmark_group(; N::Int=1000, mode::Symbol=:all)
    mode in (:all, :reverse, :forward) || error("mode must be :all, :reverse or :forward")
    suite = BenchmarkGroup()
    meta = Dict{String,WorkloadMeta}()
    mode === :forward || reverse_workloads!(suite, meta, N)
    mode === :reverse || forward_workloads!(suite, meta, N)
    return suite, meta
end

function reverse_workloads!(suite::BenchmarkGroup, meta::Dict{String,WorkloadMeta}, N::Int)
    # `Memory` cannot go through `zero_fcodual`/`gradient` at all (`zero_tangent` has no `Memory`
    # method — ISSUES.md #50), so its coduals are built by hand. This is also the cleanest
    # array-mutation shape there is: the ref chain is rooted directly at the argument via the 1-arg
    # `memoryrefnew`, with no `Array.ref` hop. It returns `nothing`, so it is seeded with `NoRData()`
    # directly rather than through `value_and_gradient!`, which needs a scalar return (ISSUES #51).
    let k = "memloop! Memory[$N] (prealloc)"
        suite[k] = @benchmarkable(
            begin
                ocd.dx .= 0
                y, pb = rrule!!(fcd, ctx, ocd, xcd, ncd)
                pb(NoRData())
            end,
            setup = begin
                o = Memory{Float64}(undef, $N); fill!(o, 0.0)
                d = Memory{Float64}(undef, $N); fill!(d, 0.0)
                ocd = CoDual(o, d)
                fcd = zero_fcodual(memloop!)
                xcd = zero_fcodual(3.0); ncd = zero_fcodual($N)
                ctx = build_ctx(memloop!, (Memory{Float64}, Float64, Int))
                y0, pb0 = rrule!!(fcd, ctx, ocd, xcd, ncd); pb0(NoRData())
            end)
        meta[k] = WorkloadMeta(memloop!, (Memory{Float64}, Float64, Int), :mutation,
                               "argument Memory written in a loop, through a pre-allocated context: " *
                               "the bulk-save best case, and the path that should allocate nothing")
        primal!(suite, meta, k, @benchmarkable(
            memloop!(o, x, n),
            setup = begin
                o = Memory{Float64}(undef, $N); fill!(o, 0.0)
                x = 3.0; n = $N
                memloop!(o, x, n)
            end, evals = 100))
    end

    # The same call through a fresh-tape `Ctx()` — what plain `gradient` uses, and what every
    # recursive inner call uses. It allocates by construction (a tape per call), and bulk save adds
    # one buffer allocation per call on top, so this is the workload that keeps that cost visible
    # instead of hiding it behind the pre-allocated path.
    let k = "memloop! Memory[$N]"
        suite[k] = @benchmarkable(
            begin
                ocd.dx .= 0
                y, pb = rrule!!(fcd, ctx, ocd, xcd, ncd)
                pb(NoRData())
            end,
            setup = begin
                o = Memory{Float64}(undef, $N); fill!(o, 0.0)
                d = Memory{Float64}(undef, $N); fill!(d, 0.0)
                ocd = CoDual(o, d)
                fcd = zero_fcodual(memloop!)
                xcd = zero_fcodual(3.0);
                ncd = zero_fcodual($N)
                ctx = Ctx()
                y0, pb0 = rrule!!(fcd, ctx, ocd, xcd, ncd); pb0(NoRData())
            end)
        meta[k] = WorkloadMeta(memloop!, (Memory{Float64}, Float64, Int), :mutation,
                               "same call on a fresh `Ctx()` tape: allocates by construction; keeps " *
                               "the per-call cost of bulk save's buffer visible")
        # Same primal as the pre-allocated row above, measured again rather than shared: the two
        # figures landing on the same number is a free check on the run's noise floor.
        primal!(suite, meta, k, @benchmarkable(
            memloop!(o, x, n),
            setup = begin
                o = Memory{Float64}(undef, $N); fill!(o, 0.0)
                x = 3.0; n = $N
                memloop!(o, x, n)
            end, evals = 100))
    end


    let k = "vecloop! Vector[$N]"
        suite[k] = @benchmarkable(
            begin
                vcd.dx .= 0
                y, pb = rrule!!(fcd, ctx, vcd, xcd)
                pb(NoRData())
            end,
            setup = begin
                v = zeros($N); dv = zeros($N)
                vcd = CoDual(v, dv)
                fcd = zero_fcodual(vecloop!); xcd = zero_fcodual(3.0)
                ctx = build_ctx(vecloop!, (Vector{Float64}, Float64))
                y0, pb0 = rrule!!(fcd, ctx, vcd, xcd); pb0(NoRData())
            end)
        meta[k] = WorkloadMeta(vecloop!, (Vector{Float64}, Float64), :mutation,
                               "same, through an Array: adds the `getfield(a, :ref)` hop")
        primal!(suite, meta, k, @benchmarkable(
            vecloop!(v, x),
            setup = begin
                v = zeros($N); x = 3.0
                vecloop!(v, x)
            end, evals = 100))
    end

    let k = "wrloop Vector[$N]"
        suite[k] = @benchmarkable(
            begin
                vcd.dx .= 0
                y, pb = rrule!!(fcd, ctx, vcd, acd)
                pb(1.0)
            end,
            setup = begin
                v = zeros($N); dv = zeros($N)
                vcd = CoDual(v, dv)
                fcd = zero_fcodual(wrloop); acd = zero_fcodual(3.0)
                ctx = build_ctx(wrloop, (Vector{Float64}, Float64))
                y0, pb0 = rrule!!(fcd, ctx, vcd, acd); pb0(1.0)
            end)
        meta[k] = WorkloadMeta(wrloop, (Vector{Float64}, Float64), :mutation,
                               "write loop + read-back loop; diluted by reads nothing here optimizes")
        primal!(suite, meta, k, @benchmarkable(
            wrloop(v, a),
            setup = begin
                v = zeros($N); a = 3.0
                wrloop(v, a)
            end, evals = 100))
    end

    # ~30 ns: far too fast to time at `evals=1`, so the eval group shares one shadow. Safe here —
    # the accumulation is a handful of float adds into an existing buffer and cannot change the
    # shape of the work — but it is the reason this one is `evals=200` and the others are not.
    let k = "straightline! 2 stores"
        suite[k] = @benchmarkable(
            begin
                y, pb = rrule!!(fcd, ctx, vcd, acd)
                pb(1.0)
            end,
            setup = begin
                v = [1.0, 2.0]; dv = zeros(2)
                vcd = CoDual(v, dv)
                fcd = zero_fcodual(straightline!); acd = zero_fcodual(4.0)
                ctx = build_ctx(straightline!, (Vector{Float64}, Float64))
                y0, pb0 = rrule!!(fcd, ctx, vcd, acd); pb0(1.0)
            end, evals = 200)
        meta[k] = WorkloadMeta(straightline!, (Vector{Float64}, Float64), :mutation,
                               "no loop, so bulk save must NOT fire: isolates per-store pullback cost")
        # A couple of stores and an add: at this size the eval count has to be large enough that the
        # timer's own resolution is not the measurement.
        primal!(suite, meta, k, @benchmarkable(
            straightline!(v, a),
            setup = begin
                v = [1.0, 2.0]; a = 4.0
                straightline!(v, a)
            end, evals = 2000))
    end

    let k = "structloop $N iters"
        suite[k] = @benchmarkable(
            begin
                set_to_zero!!(pcd.dx)
                y, pb = rrule!!(fcd, ctx, pcd, acd, ncd)
                pb(1.0)
            end,
            setup = begin
                pcd = zero_fcodual(BenchPoint(1.0, 2.0))
                fcd = zero_fcodual(structloop)
                acd = zero_fcodual(0.5); ncd = zero_fcodual($N)
                ctx = build_ctx(structloop, (BenchPoint, Float64, Int))
                y0, pb0 = rrule!!(fcd, ctx, pcd, acd, ncd); pb0(1.0)
            end)
        meta[k] = WorkloadMeta(structloop, (BenchPoint, Float64, Int), :mutation,
                               "mutable-struct setfield! in a loop; dominated by the tangent-field " *
                               "accessor invokes, which nothing has optimized yet")
        primal!(suite, meta, k, @benchmarkable(
            structloop(p, a, n),
            setup = begin
                p = BenchPoint(1.0, 2.0); a = 0.5; n = $N
                structloop(p, a, n)
            end, evals = 100))
    end

    let k = "readonly Vector[$N]"
        suite[k] = @benchmarkable(
            begin
                vcd.dx .= 0
                y, pb = rrule!!(fcd, ctx, vcd)
                pb(1.0)
            end,
            setup = begin
                v = rand($N); dv = zeros($N)
                vcd = CoDual(v, dv)
                fcd = zero_fcodual(readonly)
                ctx = build_ctx(readonly, (Vector{Float64},))
                y0, pb0 = rrule!!(fcd, ctx, vcd); pb0(1.0)
            end)
        meta[k] = WorkloadMeta(readonly, (Vector{Float64},), :guard,
                               "read-only array reduction: no mutation machinery at all")
        primal!(suite, meta, k, @benchmarkable(
            readonly(v),
            setup = begin
                v = rand($N)
                readonly(v)
            end, evals = 100))
    end

    let k = "scalarcf $N iters"
        suite[k] = @benchmarkable(
            begin
                y, pb = rrule!!(fcd, ctx, xcd, ncd)
                pb(1.0)
            end,
            setup = begin
                fcd = zero_fcodual(scalarcf)
                xcd = zero_fcodual(1.5); ncd = zero_fcodual($N)
                ctx = build_ctx(scalarcf, (Float64, Int))
                y0, pb0 = rrule!!(fcd, ctx, xcd, ncd); pb0(1.0)
            end)
        meta[k] = WorkloadMeta(scalarcf, (Float64, Int), :guard,
                               "scalar arithmetic + a branch in a loop: block stack and rdata routing only")
        primal!(suite, meta, k, @benchmarkable(
            scalarcf(x, n),
            setup = begin
                x = 1.5; n = $N
                scalarcf(x, n)
            end, evals = 100))
    end

    return nothing
end

# --- forward mode -----------------------------------------------------------
# `frule!!(Dual(f, df), Dual(x, dx), …) -> Dual`: one pass, no tape, no context, nothing to
# pre-allocate — so unlike reverse there is only ever one variant of a workload, and **allocs should
# be 0 everywhere**. A nonzero figure here means a `Dual` that failed to stay in registers (a
# non-concrete carrier, or a tangent that had to be heap-built), which is the forward-mode analogue
# of the thing the reverse suite watches the tape for.

# The forward analogue of `zero_fcodual`: how a non-seeded argument is passed (a function, a loop
# count, an array that is only written to). For a singleton or an `Int` this is `NoTangent()`.
fdual(x) = Dual(x, zero_tangent(x))

# Uniform order-2 nesting, as the higher-order transform requires: *every* seed is nested to the
# order, the function and the non-differentiable arguments included (a non-uniform seed is rejected).
nest2(x) = Dual(Dual(x, NoTangent()), Dual(x, NoTangent()))
seed2(x::Float64) = Dual(Dual(x, 1.0), Dual(1.0, 0.0))

function forward_workloads!(suite::BenchmarkGroup, meta::Dict{String,WorkloadMeta}, N::Int)

    # The floor: a scalar chain with no branches and no memory traffic, so the whole cost is the
    # dualized arithmetic. Every other forward number should be read against this one.
    let k = "fwd polychain $N iters"
        suite[k] = @benchmarkable(
            frule!!(fd, xd, nd),
            setup = begin
                fd = fdual(polychain); xd = Dual(2.0, 1.0); nd = fdual($N)
                frule!!(fd, xd, nd)
            end)
        meta[k] = WorkloadMeta(polychain, (Float64, Int), :guard,
                               "branch-free scalar chain: pure `Dual` arithmetic, the per-operation floor",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            polychain(x, n),
            setup = begin
                x = 2.0; n = $N
                polychain(x, n)
            end, evals = 100))
    end

    # Same primal as the reverse `scalarcf`, so the pair is directly comparable. Forward has no block
    # stack — a branch costs only what the primal's branch costs — which is exactly what the
    # comparison is meant to show.
    let k = "fwd scalarcf $N iters"
        suite[k] = @benchmarkable(
            frule!!(fd, xd, nd),
            setup = begin
                fd = fdual(scalarcf); xd = Dual(1.5, 1.0); nd = fdual($N)
                frule!!(fd, xd, nd)
            end)
        meta[k] = WorkloadMeta(scalarcf, (Float64, Int), :guard,
                               "scalar arithmetic + a branch in a loop; the reverse twin of this key",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            scalarcf(x, n),
            setup = begin
                x = 1.5; n = $N
                scalarcf(x, n)
            end, evals = 100))
    end

    let k = "fwd readonly Vector[$N]"
        suite[k] = @benchmarkable(
            frule!!(fd, vd),
            setup = begin
                fd = fdual(readonly)
                vd = Dual(rand($N), ones($N))
                frule!!(fd, vd)
            end)
        meta[k] = WorkloadMeta(readonly, (Vector{Float64},), :guard,
                               "read-only array reduction: one shadow `memoryrefget` per element",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            readonly(v),
            setup = begin
                v = rand($N)
                readonly(v)
            end, evals = 100))
    end

    # Writes: the shadow is a real same-shape `Array`, so a store mirrors the primal's
    # `memoryrefset!` onto it. No tape, so this is as cheap as forward array mutation gets.
    let k = "fwd vecloop! Vector[$N]"
        suite[k] = @benchmarkable(
            frule!!(fd, vd, xd),
            setup = begin
                fd = fdual(vecloop!)
                vd = Dual(zeros($N), zeros($N)); xd = Dual(3.0, 1.0)
                frule!!(fd, vd, xd)
            end)
        meta[k] = WorkloadMeta(vecloop!, (Vector{Float64}, Float64), :mutation,
                               "write loop onto the shadow array, `nothing`-returning (ISSUES #53 fixed)",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            vecloop!(v, x),
            setup = begin
                v = zeros($N); x = 3.0
                vecloop!(v, x)
            end, evals = 100))
    end

    # Retained baseline: the same write loop returning a scalar instead of `nothing`. Kept as an
    # existing row now that `fwd vecloop!` above differentiates the `nothing`-returning form directly
    # (ISSUES #53 fixed); the two should cost essentially the same.
    let k = "fwd vecloop_ret! Vector[$N]"
        suite[k] = @benchmarkable(
            frule!!(fd, vd, xd),
            setup = begin
                fd = fdual(vecloop_ret!)
                vd = Dual(zeros($N), zeros($N)); xd = Dual(3.0, 1.0)
                frule!!(fd, vd, xd)
            end)
        meta[k] = WorkloadMeta(vecloop_ret!, (Vector{Float64}, Float64), :mutation,
                               "write loop returning a scalar; retained baseline next to `fwd vecloop!`",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            vecloop_ret!(v, x),
            setup = begin
                v = zeros($N); x = 3.0
                vecloop_ret!(v, x)
            end, evals = 100))
    end

    let k = "fwd wrloop Vector[$N]"
        suite[k] = @benchmarkable(
            frule!!(fd, vd, ad),
            setup = begin
                fd = fdual(wrloop)
                vd = Dual(zeros($N), zeros($N)); ad = Dual(3.0, 1.0)
                frule!!(fd, vd, ad)
            end)
        meta[k] = WorkloadMeta(wrloop, (Vector{Float64}, Float64), :mutation,
                               "write loop + read-back loop; the reverse twin of this key",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            wrloop(v, a),
            setup = begin
                v = zeros($N); a = 3.0
                wrloop(v, a)
            end, evals = 100))
    end

    let k = "fwd structloop $N iters"
        suite[k] = @benchmarkable(
            frule!!(fd, pd, ad, nd),
            setup = begin
                p = BenchPoint(1.0, 2.0)
                fd = fdual(structloop)
                pd = Dual(p, zero_tangent(p)); ad = Dual(0.5, 1.0); nd = fdual($N)
                frule!!(fd, pd, ad, nd)
            end)
        meta[k] = WorkloadMeta(structloop, (BenchPoint, Float64, Int), :mutation,
                               "mutable-struct `setfield!` in a loop: the shadow is a `MutableTangent`, " *
                               "so each store goes through the tangent-field accessors",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            structloop(p, a, n),
            setup = begin
                p = BenchPoint(1.0, 2.0); a = 0.5; n = $N
                structloop(p, a, n)
            end, evals = 100))
    end

    # ~ns: too fast to time at `evals=1`. Sharing setup state across an eval group is harmless in
    # forward mode (a shadow is overwritten, not accumulated into), so the count is free to be high.
    let k = "fwd straightline! 2 stores"
        suite[k] = @benchmarkable(
            frule!!(fd, vd, ad),
            setup = begin
                fd = fdual(straightline!)
                vd = Dual([1.0, 2.0], zeros(2)); ad = Dual(4.0, 1.0)
                frule!!(fd, vd, ad)
            end, evals = 500)
        meta[k] = WorkloadMeta(straightline!, (Vector{Float64}, Float64), :mutation,
                               "no loop: isolates the per-call cost of entering the dual carrier",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            straightline!(v, a),
            setup = begin
                v = [1.0, 2.0]; a = 4.0
                straightline!(v, a)
            end, evals = 2000))
    end

    # The callee is a value, not a statically-resolved name, so it is carried as a `Dual` and applied
    # through the dual calling convention every iteration — and here it is a closure with a
    # differentiable capture, so that `Dual` carries a real `Tangent`, not `NoTangent()`.
    let k = "fwd applyN closure $N iters"
        suite[k] = @benchmarkable(
            frule!!(fd, gd, xd, nd),
            setup = begin
                g = make_affine(1.0)
                fd = fdual(applyN); gd = fdual(g)
                xd = Dual(2.0, 1.0); nd = fdual($N)
                frule!!(fd, gd, xd, nd)
            end)
        meta[k] = WorkloadMeta(applyN, (Any, Float64, Int), :guard,
                               "higher-order: closure-with-capture applied per iteration",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            applyN(g, x, n),
            setup = begin
                g = make_affine(1.0); x = 2.0; n = $N
                applyN(g, x, n)
            end, evals = 100))
    end

    # `z` has modulus < 1, so the recurrence converges to `z/(1-z)` — no overflow to `Inf` and no
    # drift into subnormals, either of which would time something other than the arithmetic.
    let k = "fwd cpoly ComplexF64 $N iters"
        suite[k] = @benchmarkable(
            frule!!(fd, zd, nd),
            setup = begin
                fd = fdual(cpoly)
                zd = Dual(0.5 + 0.25im, build_tangent(ComplexF64, 1.0, 0.0)); nd = fdual($N)
                frule!!(fd, zd, nd)
            end)
        meta[k] = WorkloadMeta(cpoly, (ComplexF64, Int), :guard,
                               "struct-shaped tangent (`Tangent{@NamedTuple{re,im}}`) built and " *
                               "destructured per operation, with no array involved",
                               :forward)
        primal!(suite, meta, k, @benchmarkable(
            cpoly(z, n),
            setup = begin
                z = 0.5 + 0.25im; n = $N
                cpoly(z, n)
            end, evals = 100))
    end

    # Second order, which is the transform applied to its own output: the primal being dualized here
    # is the order-1 dual IR of `polychain`. Against `fwd polychain` it says what one extra level of
    # nesting costs at run time (the compile-time cost is not measured here).
    let k = "fwd polychain order-2 $N iters"
        suite[k] = @benchmarkable(
            frule!!(fd, xd, nd),
            setup = begin
                fd = nest2(polychain); xd = seed2(2.0); nd = nest2($N)
                frule!!(fd, xd, nd)
            end)
        meta[k] = WorkloadMeta(polychain, (Float64, Int), :guard,
                               "order-2 nested duals: the composed transform's run-time cost",
                               :forward)
        # Same primal as `fwd polychain`: the ratio on this row is order-2-vs-primal, so it is the
        # square-ish figure, not the marginal cost of the second level (compare the two AD times for
        # that).
        primal!(suite, meta, k, @benchmarkable(
            polychain(x, n),
            setup = begin
                x = 2.0; n = $N
                polychain(x, n)
            end, evals = 100))
    end

    return nothing
end
