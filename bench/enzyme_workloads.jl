# Enzyme timings for the "core" workload set shared with workloads.jl — same primal functions
# (memloop!, vecloop!, wrloop, straightline!, structloop, readonly, loopdot, scalarcf, polychain),
# same keys, so bench/run.jl --vs-enzyme can pair Differ's and Enzyme's results up by name.
#
# Not covered: cpoly (ComplexF64 struct tangent), applyN (closure-valued callee), and polychain
# order-2 (nested duals) — each needs its own Enzyme activity-annotation investigation, and order-2
# especially has no direct Enzyme equivalent to Differ's nested-`Dual` approach. See README.
#
# Assumes workloads.jl has already been `include`d in this session — this file reuses BenchPoint,
# memloop!, vecloop!, wrloop, straightline!, structloop, readonly, scalarcf, polychain directly rather
# than redefining them.
#
# Same shadow-accumulation trap as Differ's own reverse workloads (see workloads.jl's header):
# Enzyme's `Duplicated` shadow is accumulated into, not overwritten, and `@benchmarkable`'s `setup`
# runs once per *sample*, not per `eval`. Every reverse workload whose shadow is a mutable buffer
# resets it as the first statement of the timed body. `scalarcf` needs no reset — an `Active` scalar's
# adjoint is returned, not accumulated in place. `straightline!` needs no reset either, following
# Differ's own precedent for that workload: accumulating a couple of floats doesn't change the shape
# of the work.

using Enzyme

"""
    enzyme_benchmark_group(; N=1000, mode=:all) -> suite::BenchmarkGroup

The Enzyme half of the comparison. `mode` selects `:reverse`, `:forward`, or `:all` (both), matching
`benchmark_group`. Keys match `workloads.jl`'s exactly (the reverse `memloop!` key matches Differ's
`(prealloc)` variant specifically — Enzyme has no separate fresh-tape/prealloc distinction, so its
natural steady-state call is the fair comparison against Differ's pre-allocated-context path).
"""
function enzyme_benchmark_group(; N::Int=1000, mode::Symbol=:all)
    mode in (:all, :reverse, :forward) || error("mode must be :all, :reverse or :forward")
    suite = BenchmarkGroup()
    mode === :forward || enzyme_reverse_workloads!(suite, N)
    mode === :reverse || enzyme_forward_workloads!(suite, N)
    return suite
end

function enzyme_reverse_workloads!(suite::BenchmarkGroup, N::Int)
    suite["memloop! Memory[$N]"] = @benchmarkable(
        begin
            d .= 0
            Enzyme.autodiff(Reverse, memloop!, Const, Duplicated(o, d), Active(3.0), Const($N))
        end,
        setup = begin
            o = Memory{Float64}(undef, $N); fill!(o, 0.0)
            d = Memory{Float64}(undef, $N); fill!(d, 0.0)
        end)

    suite["vecloop! Vector[$N]"] = @benchmarkable(
        begin
            dv .= 0
            Enzyme.autodiff(Reverse, vecloop!, Const, Duplicated(v, dv), Active(3.0))
        end,
        setup = begin
            v = zeros($N); dv = zeros($N)
        end)

    suite["wrloop Vector[$N]"] = @benchmarkable(
        begin
            dv .= 0
            Enzyme.autodiff(Reverse, wrloop, Active, Duplicated(v, dv), Active(3.0))
        end,
        setup = begin
            v = zeros($N); dv = zeros($N)
        end)

    suite["straightline! 2 stores"] = @benchmarkable(
        Enzyme.autodiff(Reverse, straightline!, Active, Duplicated(v, dv), Active(4.0)),
        setup = begin
            v = [1.0, 2.0]; dv = zeros(2)
        end, evals = 200)

    suite["structloop $N iters"] = @benchmarkable(
        begin
            dp.x = 0.0; dp.y = 0.0
            Enzyme.autodiff(Reverse, structloop, Active, Duplicated(p, dp), Active(0.5), Const($N))
        end,
        setup = begin
            p = BenchPoint(1.0, 2.0); dp = BenchPoint(0.0, 0.0)
        end)

    suite["readonly Vector[$N]"] = @benchmarkable(
        begin
            dv .= 0
            Enzyme.autodiff(Reverse, readonly, Active, Duplicated(v, dv))
        end,
        setup = begin
            v = rand($N); dv = zeros($N)
        end)

    suite["loopdot Vector[$N]"] = @benchmarkable(
        begin
            dv .= 0
            Enzyme.autodiff(Reverse, loopdot, Active, Active(2.0), Duplicated(v, dv))
        end,
        setup = begin
            v = rand($N); dv = zeros($N)
        end)

    suite["scalarcf $N iters"] = @benchmarkable(
        Enzyme.autodiff(Reverse, scalarcf, Active, Active(1.5), Const($N)))

    return nothing
end

function enzyme_forward_workloads!(suite::BenchmarkGroup, N::Int)
    suite["fwd polychain $N iters"] = @benchmarkable(
        Enzyme.autodiff(Forward, polychain, Duplicated(2.0, 1.0), Const($N)))

    suite["fwd scalarcf $N iters"] = @benchmarkable(
        Enzyme.autodiff(Forward, scalarcf, Duplicated(1.5, 1.0), Const($N)))

    suite["fwd readonly Vector[$N]"] = @benchmarkable(
        Enzyme.autodiff(Forward, readonly, Duplicated(v, dv)),
        setup = begin
            v = rand($N); dv = ones($N)
        end)

    suite["fwd vecloop! Vector[$N]"] = @benchmarkable(
        Enzyme.autodiff(Forward, vecloop!, Duplicated(v, dv), Duplicated(3.0, 1.0)),
        setup = begin
            v = zeros($N); dv = zeros($N)
        end)

    suite["fwd wrloop Vector[$N]"] = @benchmarkable(
        Enzyme.autodiff(Forward, wrloop, Duplicated(v, dv), Duplicated(3.0, 1.0)),
        setup = begin
            v = zeros($N); dv = zeros($N)
        end)

    suite["fwd structloop $N iters"] = @benchmarkable(
        Enzyme.autodiff(Forward, structloop, Duplicated(p, dp), Duplicated(0.5, 1.0), Const($N)),
        setup = begin
            p = BenchPoint(1.0, 2.0); dp = BenchPoint(0.0, 0.0)
        end)

    suite["fwd straightline! 2 stores"] = @benchmarkable(
        Enzyme.autodiff(Forward, straightline!, Duplicated(v, dv), Duplicated(4.0, 1.0)),
        setup = begin
            v = [1.0, 2.0]; dv = zeros(2)
        end, evals = 500)

    return nothing
end
