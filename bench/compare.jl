# A/B the working tree against another git revision.
#
#   julia +1.13 --project=bench bench/compare.jl HEAD
#   julia +1.13 --project=bench bench/compare.jl master --n=10000 --seconds=2 --tolerance=0.08
#
# Checks the baseline revision out into a throwaway `git worktree` and runs the *current* benchmark
# scripts against that revision's `src/`. Copying this `bench/` directory across, rather than using
# whatever the baseline shipped, is what makes the comparison meaningful — both sides must run
# identical measurement code, and the baseline generally predates it.
#
# `tape_shape` guards its introspection with `isdefined`, so a baseline lacking
# `tape_type`/`comms_element_types` reports `-` in those columns instead of failing.

using BenchmarkTools
include(joinpath(@__DIR__, "run.jl"))   # pulls in workloads.jl + harness.jl; its `main` is reused

const PKG = dirname(@__DIR__)

function bench_rev(rev::AbstractString, extra::Vector{String})
    tmp = mktempdir()
    wt = joinpath(tmp, "baseline")
    out = joinpath(tmp, "before.json")
    run(`git -C $PKG worktree add --detach $wt $rev`)
    try
        cp(joinpath(PKG, "bench"), joinpath(wt, "bench"); force=true)
        # Drop the copied Manifest: it pins `Differ` to *this* checkout, and would otherwise either
        # be stale against the copied Project or quietly point the baseline run at our own src/.
        # Resolving fresh in the worktree is what makes `[sources] Differ = {path = ".."}` bind to
        # the baseline checkout, which is the whole trick.
        rm(joinpath(wt, "bench", "Manifest.toml"); force=true)
        run(`julia +1.13 --project=$(joinpath(wt, "bench")) -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"`)
        run(`julia +1.13 --project=$(joinpath(wt, "bench")) $(joinpath(wt, "bench", "run.jl")) --json=$out $extra`)
        return BenchmarkTools.load(out)[1]
    finally
        run(`git -C $PKG worktree remove --force $wt`)
    end
end

function (@main)(args)
    isempty(args) && error("usage: julia --project=bench bench/compare.jl <git-rev> " *
                           "[--n=…] [--seconds=…] [--tolerance=…]")
    rev = args[1]
    extra = String[a for a in args[2:end] if startswith(a, "--")]
    N = parse(Int, argval(extra, "n", "1000"))
    seconds = parse(Float64, argval(extra, "seconds", "1.0"))
    tol = parse(Float64, argval(extra, "tolerance", "0.05"))
    _, meta = benchmark_group(; N)

    println("=== baseline: $rev ===")
    before = bench_rev(rev, extra)

    println("\n=== working tree ===")
    after = runner(; N, seconds, json=nothing)

    println("\n=== comparison ===")
    print_comparison(before, after, meta; tolerance=tol)
    return nothing
end
