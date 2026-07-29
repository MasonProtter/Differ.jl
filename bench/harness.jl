# Reporting on top of BenchmarkTools.
#
# BenchmarkTools does the measurement: it tunes, warms up, keeps the full sample distribution, and —
# the part that matters most here — re-runs each workload's `setup` per sample, which is what keeps
# a pullback's shadow accumulation from being folded into the timing.
#
# What this file adds is the two things the suite is actually judged on:
#
#  * **Tape shape** alongside the timing. Total comms bytes, and whether every comms stack is
#    pointer-free — a `MemoryRef` in a comms tuple makes the backing buffer GC-tracked, costing a
#    write barrier per push. That is the structural quantity these optimizations move; the timing is
#    the consequence.
#  * **A `:guard` distinction** in comparisons. Workloads that exercise no mutation machinery should
#    not move; when one does, it is a per-call regression and gets called out rather than averaged in.
#
# Compare **minimum** times. On this workload the medians swing ±7% run to run while the minima are
# stable to ~1%.

using BenchmarkTools
using PrettyTables
using Crayons
using Differ

# Tape shape, when the running Differ exposes it. Guarded with `isdefined` so the same harness runs
# against revisions predating `tape_type`/`comms_element_types` — which is what `compare.jl` does
# whenever the baseline is old enough.
function tape_shape(f, argtypes)
    (isdefined(Differ, :tape_type) && isdefined(Differ, :comms_element_types)) || return (-1, false)
    try
        ts = Differ.comms_element_types(Differ.tape_type(f, argtypes))
        return (sum(sizeof, ts; init=0), all(isbitstype, ts))
    catch
        return (-1, false)   # a workload whose tape can't be built for introspection; timing stands
    end
end

_dash(x) = x < 0 ? "—" : string(x)

function print_table(results::BenchmarkGroup, meta::Dict{String,WorkloadMeta}; io::IO=stdout)
    names = sort!(collect(keys(results)))
    data = Matrix{Any}(undef, length(names), 6)
    for (i, k) in enumerate(names)
        t = minimum(results[k])
        m = get(meta, k, nothing)
        cb, ci = m === nothing ? (-1, false) : tape_shape(m.f, m.argtypes)
        data[i, :] = Any[k, BenchmarkTools.prettytime(time(t)), memory(t),
                         _dash(cb), cb < 0 ? "—" : (ci ? "yes" : "no"),
                         m === nothing ? "" : String(m.kind)]
    end
    # Nonzero allocations deserve attention but are not always a fault: the fresh-tape (`Ctx()`)
    # workloads are *expected* to allocate — that is what they exist to measure.
    hl_alloc = TextHighlighter((d, i, j) -> j == 3 && d[i, j] isa Integer && d[i, j] > 0,
                               crayon"yellow")
    # A GC-tracked comms tuple costs a write barrier on every push.
    hl_ptr = TextHighlighter((d, i, j) -> j == 5 && d[i, j] == "no", crayon"yellow")
    hl_guard = TextHighlighter((d, i, j) -> j == 6 && d[i, j] == "guard", crayon"dark_gray")
    pretty_table(io, data;
                 column_labels=["workload", "min", "allocs", "comms B", "isbits", "kind"],
                 alignment=[:l, :r, :r, :r, :r, :l],
                 highlighters=[hl_alloc, hl_ptr, hl_guard],
                 display_size=(-1, -1))    # never crop: these tables get pasted into results/
    return nothing
end

# Side-by-side on minimum time. `tolerance` is the fraction below which a difference is treated as
# noise — 5% is about twice the run-to-run spread of the minima here, but see the README: on a noisy
# machine the guard workloads can swing further, and the fix is to raise this, not to ignore the flag.
function print_comparison(before::BenchmarkGroup, after::BenchmarkGroup,
                          meta::Dict{String,WorkloadMeta}; io::IO=stdout, tolerance::Float64=0.05)
    names = sort!(collect(keys(after)))
    rows = Any[]
    moved_guard = String[]
    for k in names
        if !haskey(before, k)
            push!(rows, Any[k, "—", BenchmarkTools.prettytime(time(minimum(after[k]))), NaN, "new"])
            continue
        end
        tb, ta = time(minimum(before[k])), time(minimum(after[k]))
        d = (ta - tb) / tb * 100
        kind = haskey(meta, k) ? meta[k].kind : :unknown
        verdict = abs(d) < tolerance * 100 ? "noise" : d < 0 ? "faster" : "SLOWER"
        if kind === :guard && abs(d) >= tolerance * 100
            verdict = "GUARD MOVED"
            push!(moved_guard, k)
        end
        push!(rows, Any[k, BenchmarkTools.prettytime(tb), BenchmarkTools.prettytime(ta), d, verdict])
    end
    data = permutedims(reduce(hcat, rows))

    hl_faster = TextHighlighter((d, i, j) -> d[i, 5] == "faster", crayon"green")
    hl_slower = TextHighlighter((d, i, j) -> d[i, 5] == "SLOWER", crayon"red bold")
    hl_guard = TextHighlighter((d, i, j) -> d[i, 5] == "GUARD MOVED", crayon"yellow bold")
    hl_noise = TextHighlighter((d, i, j) -> d[i, 5] == "noise", crayon"dark_gray")
    pretty_table(io, data;
                 column_labels=["workload", "before", "after", "Δ", "verdict"],
                 alignment=[:l, :r, :r, :r, :l],
                 formatters=[(v, i, j) -> j == 4 ? (v isa Real && isnan(v) ? "—" :
                                                    string(v > 0 ? "+" : "", round(v; digits=1), "%")) : v],
                 highlighters=[hl_slower, hl_guard, hl_faster, hl_noise],
                 display_size=(-1, -1))

    println(io, "Minima compared; `noise` is |Δ| < $(round(Int, tolerance * 100))%.")
    println(io, "`guard` workloads exercise no mutation machinery and should not move — one that")
    println(io, "does is a per-call regression. Check the sign is stable across runs before")
    println(io, "believing it; if it is not, raise --tolerance.")
    isempty(moved_guard) || println(io, "\nGUARDS MOVED: ", join(moved_guard, ", "))
    return nothing
end
