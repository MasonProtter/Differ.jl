# Run the reverse-mode benchmark suite against the checked-out Differ.
#
#   julia +1.13 --project=bench bench/run.jl
#   julia +1.13 --project=bench bench/run.jl --n=10000 --seconds=2
#   julia +1.13 --project=bench bench/run.jl --json=/tmp/after.json
#
# `--json` writes BenchmarkTools results for `compare.jl` (or a later `BenchmarkTools.load`) to diff.

using BenchmarkTools
include(joinpath(@__DIR__, "workloads.jl"))
include(joinpath(@__DIR__, "harness.jl"))

function argval(args, key, default)
    for a in args
        startswith(a, "--$key=") && return split(a, '='; limit=2)[2]
    end
    return default
end

function runner(; N, seconds, json, verbose=false)
    suite, meta = benchmark_group(; N)
    for b in values(suite)
        b.params.seconds = seconds
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
    results = runner(; N, seconds, json)
    nothing
end
