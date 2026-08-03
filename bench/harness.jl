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

# `×` — how much more expensive the AD call is than the primal it differentiates. The only figure in
# the table that is comparable across workloads: an absolute time is mostly a statement about how
# much work the primal does.
_ratio(t_ad, t_primal) = t_primal > 0 ? string(round(t_ad / t_primal; digits=1), "×") : "—"

function print_table(results::BenchmarkGroup, meta::Dict{String,WorkloadMeta}; io::IO=stdout)
    # Primal baselines are folded into their workload's row rather than given one of their own.
    names = sort!([k for k in keys(results) if !is_primal_key(k)])
    data = Matrix{Any}(undef, length(names), 10)
    for (i, k) in enumerate(names)
        t = minimum(results[k])
        m = get(meta, k, nothing)
        # Tape shape is reverse-mode-only: the forward carrier has no tape, and asking for one would
        # report the shape of a *different* transform's output next to a forward timing.
        cb, ci = (m === nothing || m.mode === :forward) ? (-1, false) : tape_shape(m.f, m.argtypes)
        pk = primal_key(k)
        tp = haskey(results, pk) ? minimum(results[pk]) : nothing
        data[i, :] = Any[k, m === nothing ? "" : (m.mode === :forward ? "fwd" : "rev"),
                         BenchmarkTools.prettytime(time(t)), memory(t),
                         tp === nothing ? "—" : BenchmarkTools.prettytime(time(tp)),
                         tp === nothing ? "—" : memory(tp),
                         tp === nothing ? "—" : _ratio(time(t), time(tp)),
                         _dash(cb), cb < 0 ? "—" : (ci ? "yes" : "no"),
                         m === nothing ? "" : String(m.kind)]
    end
    # Nonzero allocations deserve attention but are not always a fault: the fresh-tape (`Ctx()`)
    # workloads are *expected* to allocate — that is what they exist to measure. In forward mode
    # there is no such excuse: anything nonzero there is a `Dual` that failed to stay in registers.
    hl_alloc = TextHighlighter((d, i, j) -> j == 4 && d[i, j] isa Integer && d[i, j] > 0,
                               crayon"yellow")
    # A primal that allocates makes its workload's `allocs` column unreadable — the AD figure then
    # includes allocation the transform is not responsible for. None of the current primals do.
    hl_palloc = TextHighlighter((d, i, j) -> j == 6 && d[i, j] isa Integer && d[i, j] > 0,
                                crayon"yellow")
    # A GC-tracked comms tuple costs a write barrier on every push.
    hl_ptr = TextHighlighter((d, i, j) -> j == 9 && d[i, j] == "no", crayon"yellow")
    hl_guard = TextHighlighter((d, i, j) -> j == 10 && d[i, j] == "guard", crayon"dark_gray")
    pretty_table(io, data;
                 column_labels=["workload", "mode", "min", "allocs",
                                "primal", "primal allocs", "×", "comms B", "isbits", "kind"],
                 alignment=[:l, :l, :r, :r, :r, :r, :r, :r, :r, :l],
                 highlighters=[hl_alloc, hl_palloc, hl_ptr, hl_guard],
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
        if kind === :primal
            # Identical code on both sides — no revision can change it. Whatever Δ shows up here is
            # the machine's noise floor for this run, which is the number to judge the rest against.
            verdict = "(machine)"
        elseif kind === :guard && abs(d) >= tolerance * 100
            verdict = "GUARD MOVED"
            push!(moved_guard, k)
        end
        push!(rows, Any[k, BenchmarkTools.prettytime(tb), BenchmarkTools.prettytime(ta), d, verdict])
    end
    data = permutedims(reduce(hcat, rows))

    hl_faster = TextHighlighter((d, i, j) -> d[i, 5] == "faster", crayon"green")
    hl_slower = TextHighlighter((d, i, j) -> d[i, 5] == "SLOWER", crayon"red bold")
    hl_guard = TextHighlighter((d, i, j) -> d[i, 5] == "GUARD MOVED", crayon"yellow bold")
    hl_noise = TextHighlighter((d, i, j) -> d[i, 5] in ("noise", "(machine)"), crayon"dark_gray")
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
    println(io, "`[primal]` rows are the same undifferentiated call on both sides, so their Δ is this")
    println(io, "run's noise floor — read every other Δ against it.")
    isempty(moved_guard) || println(io, "\nGUARDS MOVED: ", join(moved_guard, ", "))
    return nothing
end

# Differ vs Enzyme, on the core workload set (see enzyme_workloads.jl). Unlike `print_comparison`
# this isn't a regression check against history — it's a snapshot of where Differ stands against an
# established IR-based AD tool on the same primals — so there's no noise tolerance or guard-movement
# tracking, just the two numbers and their ratio. Iterates `keys(enzyme_results)` rather than every
# Differ key: only the core subset has an Enzyme twin.
function print_vs_enzyme(differ_results::BenchmarkGroup, enzyme_results::BenchmarkGroup,
                         meta::Dict{String,WorkloadMeta}; io::IO=stdout)
    names = sort!(collect(keys(enzyme_results)))
    data = Matrix{Any}(undef, length(names), 8)
    for (i, k) in enumerate(names)
        td, te = minimum(differ_results[k]), minimum(enzyme_results[k])
        m = get(meta, k, nothing)
        data[i, :] = Any[k, m === nothing ? "" : (m.mode === :forward ? "fwd" : "rev"),
                         BenchmarkTools.prettytime(time(td)), memory(td),
                         BenchmarkTools.prettytime(time(te)), memory(te),
                         _ratio(time(td), time(te)),
                         m === nothing ? "" : String(m.kind)]
    end
    hl_alloc_d = TextHighlighter((d, i, j) -> j == 4 && d[i, j] isa Integer && d[i, j] > 0,
                                 crayon"yellow")
    hl_alloc_e = TextHighlighter((d, i, j) -> j == 6 && d[i, j] isa Integer && d[i, j] > 0,
                                 crayon"yellow")
    pretty_table(io, data;
                 column_labels=["workload", "mode", "Differ min", "Differ allocs",
                               "Enzyme min", "Enzyme allocs", "Differ/Enzyme", "kind"],
                 alignment=[:l, :l, :r, :r, :r, :r, :r, :l],
                 highlighters=[hl_alloc_d, hl_alloc_e],
                 display_size=(-1, -1))
    println(io, "Minima compared. `Differ/Enzyme` > 1× means Differ is slower on that workload.")
    return nothing
end
