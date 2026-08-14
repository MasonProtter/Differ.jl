# Run the benchmark suite against the checked-out Differ.
#
#   julia +1.13 --project=bench bench/run.jl
#   julia +1.13 --project=bench bench/run.jl --n=10000 --seconds=2
#   julia +1.13 --project=bench bench/run.jl --mode=forward
#   julia +1.13 --project=bench bench/run.jl --json=/tmp/after.json
#   julia +1.13 --project=bench bench/run.jl --vs-enzyme
#   julia +1.13 --project=bench bench/run.jl --vs-mooncake
#
# `--mode=` selects `reverse`, `forward` or `all` (default). `--json` writes BenchmarkTools results
# for `compare.jl` (or a later `BenchmarkTools.load`) to diff. `--vs-enzyme` additionally times the
# core workload set through Enzyme.jl and prints a side-by-side table (see enzyme_workloads.jl for
# what's covered); Enzyme is only `using`d when this flag is passed, so plain runs never pay for it.
# `--vs-mooncake` does the same against Mooncake.jl (see mooncake_workloads.jl); the two flags may be
# combined.

using BenchmarkTools
include(joinpath(@__DIR__, "workloads.jl"))
include(joinpath(@__DIR__, "harness.jl"))
# Included at top level, conditionally, so `using Enzyme`/`using Mooncake` (and their first-time
# precompile — Mooncake's runs to ~80s) is only ever paid when the matching flag is passed. Doing this
# inside `run_vs_enzyme`/`run_vs_mooncake` instead would hit Julia's world-age check: a function can't
# call a method `include`d after the function itself was compiled without `invokelatest`, and top
# level is the natural place to dodge that.
const VS_ENZYME = "--vs-enzyme" in ARGS
const VS_MOONCAKE = "--vs-mooncake" in ARGS
VS_ENZYME && include(joinpath(@__DIR__, "enzyme_workloads.jl"))
VS_MOONCAKE && include(joinpath(@__DIR__, "mooncake_workloads.jl"))

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

function run_vs_enzyme(differ_results, meta; N, seconds, mode::Symbol=:all, verbose=false)
    suite = enzyme_benchmark_group(; N, mode)
    for (_, b) in suite
        b.params.seconds = seconds
    end
    results = run(suite; verbose)
    println("\n=== vs Enzyme ===")
    print_vs_enzyme(differ_results, results, meta)
    return results
end

function run_vs_mooncake(differ_results, meta; N, seconds, mode::Symbol=:all, verbose=false)
    suite = mooncake_benchmark_group(; N, mode)
    for (_, b) in suite
        b.params.seconds = seconds
    end
    results = run(suite; verbose)
    println("\n=== vs Mooncake ===")
    print_vs_mooncake(differ_results, results, meta)
    return results
end

function (@main)(args)
    N = parse(Int, argval(args, "n", "1000"))
    seconds = parse(Float64, argval(args, "seconds", "1.0"))
    json = argval(args, "json", nothing)
    mode = Symbol(argval(args, "mode", "all"))
    results = runner(; N, seconds, json, mode)
    if VS_ENZYME || VS_MOONCAKE
        _, meta = benchmark_group(; N, mode)
        VS_ENZYME && run_vs_enzyme(results, meta; N, seconds, mode)
        VS_MOONCAKE && run_vs_mooncake(results, meta; N, seconds, mode)
    end
    nothing
end
