# Differ reverse-mode benchmarks

A small suite for tracking the cost of reverse-mode AD, built on
[BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl).

```bash
# measure the working tree
julia +1.13 --project=bench bench/run.jl

# A/B the working tree against a git revision (checks it out into a temporary worktree)
julia +1.13 --project=bench bench/compare.jl HEAD
julia +1.13 --project=bench bench/compare.jl master --n=10000 --seconds=2
```

Flags: `--n=` array length / loop trip count (default 1000), `--seconds=` per-workload budget
(default 1.0), `--json=` write results for a later `BenchmarkTools.load`, `--tolerance=` the
below-which-it's-noise fraction for `compare.jl` (default 0.05).

**Use `--seconds=2` when the result matters.** At the 1 s default the sub-microsecond workload
(`straightline!`) has measured anywhere from −14% to −39% for the same change; it settles by 2 s.
On a noisy machine the `guard` workloads can swing ±6%, in which case raise `--tolerance`
rather than ignoring the flag — and check whether the sign is stable across runs before believing
any single number.

## Layout

| file | what it is |
|---|---|
| `workloads.jl` | the `BenchmarkGroup` — one `@benchmarkable` per workload, plus metadata |
| `harness.jl` | reporting: the table, and the before/after comparison |
| `run.jl` | entry point for a single measurement |
| `compare.jl` | entry point for an A/B against a git revision |
| `results/` | recorded runs, with the change they measured |

## Reading the table

```
┌──────────────────────────────────┬───────────┬────────┬─────────┬────────┬──────────┐
│ workload                         │       min │ allocs │ comms B │ isbits │ kind     │
├──────────────────────────────────┼───────────┼────────┼─────────┼────────┼──────────┤
│ memloop! Memory[1000]            │ 16.341 μs │  79624 │      24 │     no │ mutation │
│ memloop! Memory[1000] (prealloc) │  4.358 μs │      0 │      24 │     no │ mutation │
└──────────────────────────────────┴───────────┴────────┴─────────┴────────┴──────────┘
```

- **min** — minimum sample. Compare minima, not medians: on this workload the medians swing ±7% run
  to run while the minima are stable to ~1%. `compare.jl` treats anything under 5% as noise.
- **allocs** — bytes allocated per call. Through a `build_ctx(...; prealloc=true)` context this
  should be **0**, and anything else is worth chasing. The fresh-tape (`Ctx()`) workloads allocate
  by construction — that is what they exist to measure, so they are highlighted for attention
  rather than flagged as faults.
- **commsB** / **isbits** — tape shape: total bytes across the tape's comms stacks, and whether they
  are all pointer-free. A `MemoryRef` in a comms tuple makes the backing buffer GC-tracked, costing a
  write barrier on every push. Note `commsB` is a **whole-function sum over all blocks**, not a
  per-iteration figure — for a loop workload the loop block dominates it, but for a straight-line one
  it counts blocks that run once.
- **kind** — `mutation` workloads are expected to move when mutation costs change. **`guard`
  workloads are expected not to move at all**: they exercise no mutation machinery, so they are how
  a per-call regression (something added to every tape, say) gets caught. `compare.jl` flags a guard
  that moves.

### Pre-allocated vs fresh tape

`memloop!` appears twice on purpose. The `(prealloc)` variant goes through
`build_ctx(...; prealloc=true)`, reusing one tape across calls; the other builds a fresh `Ctx()` tape
per call — which is what plain `gradient` does, and what *every recursive inner call* does. Costs
that a pre-allocated context amortizes away are paid in full on the fresh-tape path, so an
optimization that trades an allocation for less per-iteration work will look much better on one than
the other. Quote both.

## Writing a workload

Two traps, both of which have already produced wrong numbers in this repo:

**Reset the shadow in `setup`.** A pullback *accumulates* into the shadow. Reuse one across samples
and you are timing an ever-growing accumulation rather than a call. The primal needs no such care —
the pullback restores it — but the shadow does. `setup` also runs a full warm-up round trip, so the
first sample isn't paying for compilation.

**Use `evals=1` unless sharing setup state is provably harmless.** BenchmarkTools runs `setup` once
per *sample*, not per eval, so `evals>1` shares one shadow across the whole eval group. Only
`straightline!` uses a higher count, because at ~40 ns it cannot be timed at `evals=1` and its
accumulation is a couple of float adds that cannot change the shape of the work.

Two entry-point gaps force some workloads to build coduals by hand and seed `rrule!!` directly
rather than going through `gradient!`:

- `zero_tangent` has no `Memory` method (`ISSUES.md` #50), so a `Memory` argument cannot go through
  `zero_fcodual`/`gradient` at all.
- `gradient!`/`value_and_gradient!` seed with `one(y)` and so require a scalar return (`ISSUES.md`
  #51), which a `nothing`-returning mutating function doesn't have.

If either is fixed, the affected workloads can be simplified to the ordinary `gradient!` form.

## How `compare.jl` works

It checks the baseline revision out into a throwaway `git worktree` and copies *this* `bench/`
directory over the one in the worktree, so both sides run identical measurement code — the baseline
generally predates it. The copied `Project.toml` points `Differ` at `..`, which inside the worktree
is the baseline checkout, so it loads that revision's `src/`. Tape-shape introspection is guarded
with `isdefined`, so a baseline lacking `tape_type`/`comms_element_types` reports `-` in those
columns rather than failing.
