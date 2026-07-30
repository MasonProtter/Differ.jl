# Run the benchmark suite against the checked-out Differ.
#
#   julia +1.13 --project=bench bench/run.jl
#   julia +1.13 --project=bench bench/run.jl --n=10000 --seconds=2
#   julia +1.13 --project=bench bench/run.jl --mode=forward
#   julia +1.13 --project=bench bench/run.jl --json=/tmp/after.json
#
# `--mode=` selects `reverse`, `forward` or `all` (default). `--json` writes BenchmarkTools results
# for `compare.jl` (or a later `BenchmarkTools.load`) to diff.

using BenchmarkTools
include(joinpath(@__DIR__, "workloads.jl"))
include(joinpath(@__DIR__, "harness.jl"))

function argval(args, key, default)
    for a in args
        startswith(a, "--$key=") && return split(a, '='; limit=2)[2]
    end
    return default
end

function runner(; N, seconds, json, mode::Symbol=:all, verbose=false)
    suite, meta = benchmark_group(; N, mode)
    for (k, b) in suite
        # Primal baselines get a shorter budget: they run at a high eval count and their minima are
        # stable within a few percent, so spending the full budget on them would roughly double the
        # suite's wall time for no extra resolution where it matters.
        b.params.seconds = is_primal_key(k) ? min(seconds, 0.5) : seconds
    end
    results = run(suite; verbose)
    print_table(results, meta)
    if json !== nothing
        BenchmarkTools.save(json, results)
        println("\n\nwrote ", json)
    end
    return results
end

function (@main)(args)
    N = parse(Int, argval(args, "n", "1000"))
    seconds = parse(Float64, argval(args, "seconds", "1.0"))
    json = argval(args, "json", nothing)
    mode = Symbol(argval(args, "mode", "all"))
    results = runner(; N, seconds, json, mode)
    nothing
end
