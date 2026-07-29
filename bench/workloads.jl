# Reverse-mode benchmark workloads, as a `BenchmarkGroup`.
#
# Each workload names what it exercises, because the point of the set is *coverage*, not a single
# headline number: an optimization that helps array mutation should show up here as helping the
# mutation workloads and leaving the rest alone. The `:guard` workloads exist to be unchanged — they
# are how a per-call regression (something added to every tape, say) gets caught.
#
# Two things every workload has to get right, both of which have bitten this benchmark already:
#
#  * **Reset the shadow in `setup`.** A pullback *accumulates* into the shadow. Reuse one across
#    samples and you are timing an ever-growing accumulation, not a call. The primal needs no such
#    care — the pullback restores it — but the shadow does.
#  * **`evals=1` wherever setup state must not be shared.** BenchmarkTools runs `setup` once per
#    *sample*, not per eval, so `evals>1` reuses one shadow across the whole eval group. It is only
#    safe where the accumulation cannot affect the timing (pure float adds into an existing buffer),
#    and it is needed there — a ~30 ns call can't be timed at `evals=1`.
#
# Adding one: give it a `note` saying which part of the transform it stresses, and prefer a shape
# whose cost is dominated by the thing being measured. `N` iterations of a trivial body is usually
# better than a realistic function, whose cost is spread across everything.

using BenchmarkTools
using Differ
using Differ: CoDual, NoRData, rrule!!, build_ctx, zero_fcodual

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

# --- metadata ---------------------------------------------------------------
# Keyed by the same name as the `BenchmarkGroup` entry. `f`/`argtypes` are here so the runner can
# report tape shape (comms bytes, whether the comms stacks are pointer-free) alongside the timing —
# the structural metric these optimizations actually move.

struct WorkloadMeta
    f::Any
    argtypes::Tuple
    kind::Symbol          # :mutation (expected to move) | :guard (expected not to)
    note::String
end

"""
    benchmark_group(; N=1000) -> (suite::BenchmarkGroup, meta::Dict{String,WorkloadMeta})

The reverse-mode benchmark suite. Run with `run(suite)`, or via `bench/run.jl`.
"""
function benchmark_group(; N::Int=1000)
    suite = BenchmarkGroup()
    meta = Dict{String,WorkloadMeta}()

    # `Memory` cannot go through `zero_fcodual`/`gradient` at all (`zero_tangent` has no `Memory`
    # method — ISSUES.md #50), so its coduals are built by hand. This is also the cleanest
    # array-mutation shape there is: the ref chain is rooted directly at the argument via the 1-arg
    # `memoryrefnew`, with no `Array.ref` hop. It returns `nothing`, so it is seeded with `NoRData()`
    # directly rather than through `value_and_gradient!`, which needs a scalar return (ISSUES #51).
    let k = "memloop! Memory[$N] (prealloc)"
        suite[k] = @benchmarkable(
            begin
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
            end, evals = 1)
        meta[k] = WorkloadMeta(memloop!, (Memory{Float64}, Float64, Int), :mutation,
                               "argument Memory written in a loop, through a pre-allocated context: " *
                               "the bulk-save best case, and the path that should allocate nothing")
    end

    # The same call through a fresh-tape `Ctx()` — what plain `gradient` uses, and what every
    # recursive inner call uses. It allocates by construction (a tape per call), and bulk save adds
    # one buffer allocation per call on top, so this is the workload that keeps that cost visible
    # instead of hiding it behind the pre-allocated path.
    let k = "memloop! Memory[$N]"
        suite[k] = @benchmarkable(
            begin
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
            end, evals = 1)
        meta[k] = WorkloadMeta(memloop!, (Memory{Float64}, Float64, Int), :mutation,
                               "same call on a fresh `Ctx()` tape: allocates by construction; keeps " *
                               "the per-call cost of bulk save's buffer visible")
    end
    

    let k = "vecloop! Vector[$N]"
        suite[k] = @benchmarkable(
            begin
                y, pb = rrule!!(fcd, ctx, vcd, xcd)
                pb(NoRData())
            end,
            setup = begin
                v = zeros($N); dv = zeros($N)
                vcd = CoDual(v, dv)
                fcd = zero_fcodual(vecloop!); xcd = zero_fcodual(3.0)
                ctx = build_ctx(vecloop!, (Vector{Float64}, Float64))
                y0, pb0 = rrule!!(fcd, ctx, vcd, xcd); pb0(NoRData())
            end, evals = 1)
        meta[k] = WorkloadMeta(vecloop!, (Vector{Float64}, Float64), :mutation,
                               "same, through an Array: adds the `getfield(a, :ref)` hop")
    end

    let k = "wrloop Vector[$N]"
        suite[k] = @benchmarkable(
            begin
                y, pb = rrule!!(fcd, ctx, vcd, acd)
                pb(1.0)
            end,
            setup = begin
                v = zeros($N); dv = zeros($N)
                vcd = CoDual(v, dv)
                fcd = zero_fcodual(wrloop); acd = zero_fcodual(3.0)
                ctx = build_ctx(wrloop, (Vector{Float64}, Float64))
                y0, pb0 = rrule!!(fcd, ctx, vcd, acd); pb0(1.0)
            end, evals = 1)
        meta[k] = WorkloadMeta(wrloop, (Vector{Float64}, Float64), :mutation,
                               "write loop + read-back loop; diluted by reads nothing here optimizes")
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
    end

    let k = "structloop $N iters"
        suite[k] = @benchmarkable(
            begin
                y, pb = rrule!!(fcd, ctx, pcd, acd, ncd)
                pb(1.0)
            end,
            setup = begin
                pcd = zero_fcodual(BenchPoint(1.0, 2.0))
                fcd = zero_fcodual(structloop)
                acd = zero_fcodual(0.5); ncd = zero_fcodual($N)
                ctx = build_ctx(structloop, (BenchPoint, Float64, Int))
                y0, pb0 = rrule!!(fcd, ctx, pcd, acd, ncd); pb0(1.0)
            end, evals = 1)
        meta[k] = WorkloadMeta(structloop, (BenchPoint, Float64, Int), :mutation,
                               "mutable-struct setfield! in a loop; dominated by the tangent-field " *
                               "accessor invokes, which nothing has optimized yet")
    end

    let k = "readonly Vector[$N]"
        suite[k] = @benchmarkable(
            begin
                y, pb = rrule!!(fcd, ctx, vcd)
                pb(1.0)
            end,
            setup = begin
                v = rand($N); dv = zeros($N)
                vcd = CoDual(v, dv)
                fcd = zero_fcodual(readonly)
                ctx = build_ctx(readonly, (Vector{Float64},))
                y0, pb0 = rrule!!(fcd, ctx, vcd); pb0(1.0)
            end, evals = 1)
        meta[k] = WorkloadMeta(readonly, (Vector{Float64},), :guard,
                               "read-only array reduction: no mutation machinery at all")
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
            end, evals = 1)
        meta[k] = WorkloadMeta(scalarcf, (Float64, Int), :guard,
                               "scalar arithmetic + a branch in a loop: block stack and rdata routing only")
    end

    return suite, meta
end
