# Mooncake timings for the "core" workload set shared with workloads.jl — same primal functions
# (memloop!, vecloop!, wrloop, straightline!, structloop, readonly, loopdot, scalarcf, polychain),
# same keys, so bench/run.jl --vs-mooncake can pair Differ's and Mooncake's results up by name.
#
# Differ's tangent system (Tangent/FData/RData, CoDual/Dual, rrule!!/frule!!) is a direct port of
# Mooncake's, so this file calls Mooncake at the same level Differ's own workloads.jl calls Differ:
# `Mooncake.build_rrule`/`build_frule` once per workload (the `Ctx`/`build_ctx` analogue), then
# `Mooncake.CoDual`/`Dual` + `zero_fcodual`/`zero_tangent` + the rule call + pullback, every sample.
# That symmetry is also why every Mooncake name below is written `Mooncake.foo` rather than pulled in
# with `using Mooncake: foo` — workloads.jl already brought Differ's `CoDual`, `Dual`, `NoRData`,
# `NoTangent`, `rrule!!`, `frule!!`, `zero_tangent`, `zero_fcodual` into scope unqualified, and
# Mooncake uses the identical names for the identical concepts.
#
# Not covered: cpoly (Mooncake builds a `ComplexF64` tangent differently — `build_tangent` expects a
# `Tangent`-shaped NamedTuple constructor Complex doesn't have) and polychain order-2 (no direct
# nested-`Dual` equivalent to ask Mooncake for). Matches enzyme_workloads.jl's exclusions, and for
# similar reasons — see README.
#
# Assumes workloads.jl has already been `include`d in this session — this file reuses BenchPoint,
# memloop!, vecloop!, wrloop, straightline!, structloop, readonly, loopdot, scalarcf, polychain
# directly rather than redefining them.
#
# Same shadow-accumulation trap as Differ's own reverse workloads (see workloads.jl's header):
# Mooncake's pullback accumulates into the fdata array passed in via `CoDual`, and `@benchmarkable`'s
# `setup` runs once per *sample*, not per `eval`. Every reverse workload whose shadow is a mutable
# buffer resets it as the first statement of the timed body, exactly like the Differ workload it pairs
# with. `scalarcf` needs no reset — a scalar's adjoint is returned, not accumulated in place.
# `straightline!` needs no reset either, following Differ's own precedent for that workload.
#
# `Mooncake.build_rrule`/`build_frule` do real compilation, so — like Differ's `build_ctx` in
# workloads.jl — they belong in `setup`, not the timed body: `setup` reruns once per sample, but is
# not itself timed.

using Mooncake

"""
    mooncake_benchmark_group(; N=1000, mode=:all) -> suite::BenchmarkGroup

The Mooncake half of the comparison. `mode` selects `:reverse`, `:forward`, or `:all` (both),
matching `benchmark_group`. Keys match `workloads.jl`'s exactly (the reverse `memloop!` key matches
Differ's `(prealloc)` variant specifically — Mooncake has no separate fresh-tape/prealloc distinction,
so its natural steady-state call, rule built once and reused, is the fair comparison against Differ's
pre-allocated-context path).
"""
function mooncake_benchmark_group(; N::Int=1000, mode::Symbol=:all)
    mode in (:all, :reverse, :forward) || error("mode must be :all, :reverse or :forward")
    suite = BenchmarkGroup()
    mode === :forward || mooncake_reverse_workloads!(suite, N)
    mode === :reverse || mooncake_forward_workloads!(suite, N)
    return suite
end

function mooncake_reverse_workloads!(suite::BenchmarkGroup, N::Int)
    suite["memloop! Memory[$N]"] = @benchmarkable(
        begin
            fill!(ocd.dx, 0.0)
            y, pb = rule(fcd, ocd, xcd, ncd)
            pb(Mooncake.NoRData())
        end,
        setup = begin
            o = Memory{Float64}(undef, $N); fill!(o, 0.0)
            d = Memory{Float64}(undef, $N); fill!(d, 0.0)
            ocd = Mooncake.CoDual(o, d)
            fcd = Mooncake.zero_fcodual(memloop!)
            xcd = Mooncake.zero_fcodual(3.0)
            ncd = Mooncake.zero_fcodual($N)
            rule = Mooncake.build_rrule(memloop!, o, 3.0, $N)
            y0, pb0 = rule(fcd, ocd, xcd, ncd); pb0(Mooncake.NoRData())
        end)

    suite["vecloop! Vector[$N]"] = @benchmarkable(
        begin
            fill!(vcd.dx, 0.0)
            y, pb = rule(fcd, vcd, xcd)
            pb(Mooncake.NoRData())
        end,
        setup = begin
            v = zeros($N); dv = zeros($N)
            vcd = Mooncake.CoDual(v, dv)
            fcd = Mooncake.zero_fcodual(vecloop!); xcd = Mooncake.zero_fcodual(3.0)
            rule = Mooncake.build_rrule(vecloop!, v, 3.0)
            y0, pb0 = rule(fcd, vcd, xcd); pb0(Mooncake.NoRData())
        end)

    suite["wrloop Vector[$N]"] = @benchmarkable(
        begin
            fill!(vcd.dx, 0.0)
            y, pb = rule(fcd, vcd, acd)
            pb(1.0)
        end,
        setup = begin
            v = zeros($N); dv = zeros($N)
            vcd = Mooncake.CoDual(v, dv)
            fcd = Mooncake.zero_fcodual(wrloop); acd = Mooncake.zero_fcodual(3.0)
            rule = Mooncake.build_rrule(wrloop, v, 3.0)
            y0, pb0 = rule(fcd, vcd, acd); pb0(1.0)
        end)

    suite["straightline! 2 stores"] = @benchmarkable(
        begin
            y, pb = rule(fcd, vcd, acd)
            pb(1.0)
        end,
        setup = begin
            v = [1.0, 2.0]; dv = zeros(2)
            vcd = Mooncake.CoDual(v, dv)
            fcd = Mooncake.zero_fcodual(straightline!); acd = Mooncake.zero_fcodual(4.0)
            rule = Mooncake.build_rrule(straightline!, v, 4.0)
            y0, pb0 = rule(fcd, vcd, acd); pb0(1.0)
        end, evals = 200)

    suite["structloop $N iters"] = @benchmarkable(
        begin
            Mooncake.set_to_zero!!(pcd.dx)
            y, pb = rule(fcd, pcd, acd, ncd)
            pb(1.0)
        end,
        setup = begin
            p = BenchPoint(1.0, 2.0)
            pcd = Mooncake.CoDual(p, Mooncake.zero_tangent(p))
            fcd = Mooncake.zero_fcodual(structloop)
            acd = Mooncake.zero_fcodual(0.5); ncd = Mooncake.zero_fcodual($N)
            rule = Mooncake.build_rrule(structloop, p, 0.5, $N)
            y0, pb0 = rule(fcd, pcd, acd, ncd); pb0(1.0)
        end)

    suite["readonly Vector[$N]"] = @benchmarkable(
        begin
            fill!(vcd.dx, 0.0)
            y, pb = rule(fcd, vcd)
            pb(1.0)
        end,
        setup = begin
            v = rand($N); dv = zeros($N)
            vcd = Mooncake.CoDual(v, dv)
            fcd = Mooncake.zero_fcodual(readonly)
            rule = Mooncake.build_rrule(readonly, v)
            y0, pb0 = rule(fcd, vcd); pb0(1.0)
        end)

    suite["loopdot Vector[$N]"] = @benchmarkable(
        begin
            fill!(vcd.dx, 0.0)
            y, pb = rule(fcd, xcd, vcd)
            pb(1.0)
        end,
        setup = begin
            v = rand($N); dv = zeros($N)
            vcd = Mooncake.CoDual(v, dv)
            xcd = Mooncake.zero_fcodual(2.0)
            fcd = Mooncake.zero_fcodual(loopdot)
            rule = Mooncake.build_rrule(loopdot, 2.0, v)
            y0, pb0 = rule(fcd, xcd, vcd); pb0(1.0)
        end)

    suite["scalarcf $N iters"] = @benchmarkable(
        begin
            y, pb = rule(fcd, xcd, ncd)
            pb(1.0)
        end,
        setup = begin
            fcd = Mooncake.zero_fcodual(scalarcf)
            xcd = Mooncake.zero_fcodual(1.5); ncd = Mooncake.zero_fcodual($N)
            rule = Mooncake.build_rrule(scalarcf, 1.5, $N)
            y0, pb0 = rule(fcd, xcd, ncd); pb0(1.0)
        end)

    return nothing
end

# `frule_(Dual(f,df), Dual(x,dx), …)`: one pass, no separate pullback, matching Differ's own
# `frule!!` calling convention exactly. Mooncake writes into the shadow in place rather than
# accumulating, same as Differ, so — like the forward half of workloads.jl — there is no reset to do
# between samples.
mdual(x) = Mooncake.Dual(x, Mooncake.zero_tangent(x))

function mooncake_forward_workloads!(suite::BenchmarkGroup, N::Int)
    suite["fwd polychain $N iters"] = @benchmarkable(
        frule_(fd, xd, nd),
        setup = begin
            fd = mdual(polychain); xd = Mooncake.Dual(2.0, 1.0); nd = mdual($N)
            frule_ = Mooncake.build_frule(polychain, 2.0, $N)
            frule_(fd, xd, nd)
        end)

    suite["fwd scalarcf $N iters"] = @benchmarkable(
        frule_(fd, xd, nd),
        setup = begin
            fd = mdual(scalarcf); xd = Mooncake.Dual(1.5, 1.0); nd = mdual($N)
            frule_ = Mooncake.build_frule(scalarcf, 1.5, $N)
            frule_(fd, xd, nd)
        end)

    suite["fwd readonly Vector[$N]"] = @benchmarkable(
        frule_(fd, vd),
        setup = begin
            fd = mdual(readonly)
            vd = Mooncake.Dual(rand($N), ones($N))
            frule_ = Mooncake.build_frule(readonly, rand($N))
            frule_(fd, vd)
        end)

    suite["fwd vecloop! Vector[$N]"] = @benchmarkable(
        frule_(fd, vd, xd),
        setup = begin
            fd = mdual(vecloop!)
            vd = Mooncake.Dual(zeros($N), zeros($N)); xd = Mooncake.Dual(3.0, 1.0)
            frule_ = Mooncake.build_frule(vecloop!, zeros($N), 3.0)
            frule_(fd, vd, xd)
        end)

    suite["fwd wrloop Vector[$N]"] = @benchmarkable(
        frule_(fd, vd, ad),
        setup = begin
            fd = mdual(wrloop)
            vd = Mooncake.Dual(zeros($N), zeros($N)); ad = Mooncake.Dual(3.0, 1.0)
            frule_ = Mooncake.build_frule(wrloop, zeros($N), 3.0)
            frule_(fd, vd, ad)
        end)

    suite["fwd structloop $N iters"] = @benchmarkable(
        frule_(fd, pd, ad, nd),
        setup = begin
            p = BenchPoint(1.0, 2.0)
            fd = mdual(structloop)
            pd = Mooncake.Dual(p, Mooncake.zero_tangent(p)); ad = Mooncake.Dual(0.5, 1.0); nd = mdual($N)
            frule_ = Mooncake.build_frule(structloop, p, 0.5, $N)
            frule_(fd, pd, ad, nd)
        end)

    suite["fwd straightline! 2 stores"] = @benchmarkable(
        frule_(fd, vd, ad),
        setup = begin
            fd = mdual(straightline!)
            vd = Mooncake.Dual([1.0, 2.0], zeros(2)); ad = Mooncake.Dual(4.0, 1.0)
            frule_ = Mooncake.build_frule(straightline!, [1.0, 2.0], 4.0)
            frule_(fd, vd, ad)
        end, evals = 500)

    return nothing
end
