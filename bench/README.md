# Differ benchmarks

A small suite for tracking the cost of AD in both modes — reverse (`rrule!!` + pullback) and forward
(`frule!!`) — built on [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl).

```bash
# measure the working tree (both modes)
julia +1.13 --project=bench bench/run.jl

# one mode only
julia +1.13 --project=bench bench/run.jl --mode=forward

# A/B the working tree against a git revision (checks it out into a temporary worktree)
julia +1.13 --project=bench bench/compare.jl HEAD
julia +1.13 --project=bench bench/compare.jl master --n=10000 --seconds=2
```

Flags: `--n=` array length / loop trip count (default 1000), `--seconds=` per-workload budget
(default 1.0; primal baselines are capped at 0.5), `--mode=` `reverse` | `forward` | `all` (default
`all`), `--json=` write results for a later `BenchmarkTools.load`, `--tolerance=` the
below-which-it's-noise fraction for `compare.jl` (default 0.05).

Forward-mode keys are prefixed `fwd `; reverse-mode keys are bare. Five primals
(`readonly`, `wrloop`, `structloop`, `scalarcf`, `straightline!`) appear in **both** modes on
purpose — the same function measured both ways is the only honest way to say what one mode costs
relative to the other on a given shape.

**Use `--seconds=2` when the result matters.** At the 1 s default the sub-microsecond workload
(`straightline!`) has measured anywhere from −14% to −39% for the same change; it settles by 2 s.
On a noisy machine any workload can swing a few percent run to run, in which case raise
`--tolerance` rather than ignoring the flag — and check whether the sign is stable across runs
before believing any single number.

## Layout

| file | what it is |
|---|---|
| `workloads.jl` | the `BenchmarkGroup` — one `@benchmarkable` per workload, plus metadata |
| `harness.jl` | reporting: the table, and the before/after comparison |
| `run.jl` | entry point for a single measurement |
| `compare.jl` | entry point for an A/B against a git revision |
| `enzyme_workloads.jl` | the core workload set, timed through Enzyme.jl (`--vs-enzyme`) |
| `mooncake_workloads.jl` | the core workload set, timed through Mooncake.jl (`--vs-mooncake`) |
| `results/` | recorded runs, with the change they measured |

## Reading the table

```
┌──────────────────────────────────┬──────┬───────────┬────────┬───────────┬───────────────┬────────┬─────────┬────────┐
│ workload                         │ mode │       min │ allocs │    primal │ primal allocs │      × │ comms B │ isbits │
├──────────────────────────────────┼──────┼───────────┼────────┼───────────┼───────────────┼────────┼─────────┼────────┤
│ fwd wrloop Vector[1000]          │ fwd  │  1.282 μs │      0 │ 671.560ns │             0 │   1.9× │       — │      — │
│ memloop! Memory[1000]            │ rev  │ 16.501 μs │  79624 │ 52.990 ns │             0 │ 311.4× │      24 │     no │
│ memloop! Memory[1000] (prealloc) │ rev  │  4.509 μs │      0 │ 52.690 ns │             0 │  85.6× │      24 │     no │
└──────────────────────────────────┴──────┴───────────┴────────┴───────────┴───────────────┴────────┴─────────┴────────┘
```

- **mode** — `rev` (`rrule!!` + pullback) or `fwd` (`frule!!`).
- **min** — minimum sample. Compare minima, not medians: on this workload the medians swing ±7% run
  to run while the minima are stable to ~1%. `compare.jl` treats anything under 5% as noise.
- **primal** / **primal allocs** / **×** — the same call, undifferentiated, and the ratio. Every
  workload registers one (as a separate `"… [primal]"` benchmark, folded into this row rather than
  given its own). **The ratio is the only figure comparable across workloads** — an absolute time
  mostly says how much work the primal does. Two cautions when reading it:
  - A primal the compiler vectorizes inflates the ratio, and that is real but not all the
    transform's doing: `memloop!` fills 1000 elements in 53 ns because it becomes a `memset`, and
    nothing dualized will match that. Compare ratios across *revisions* of the same workload freely;
    compare them across workloads only with the primal's shape in mind.
  - A nonzero **primal allocs** would mean the workload's own `allocs` includes allocation the
    transform is not responsible for. None of the current primals allocate; the column is
    highlighted if one starts to.

  In `compare.jl` the primal rows appear on their own, marked `(machine)`: the same undifferentiated
  code runs on both sides, so their Δ *is* the run's noise floor. Most sit under 1%. The exception is
  `straightline! [primal]` at ~1.7 ns, which is close enough to the timer's resolution to swing ±25%;
  treat that one as a resolution artifact, not a measurement.
- **allocs** — bytes allocated per call. Through a `build_ctx(...; prealloc=true)` context this
  should be **0**, and anything else is worth chasing. The fresh-tape (`Ctx()`) workloads allocate
  by construction — that is what they exist to measure, so they are highlighted for attention
  rather than flagged as faults. **Forward mode has no such excuse**: there is no tape and no
  context, so a nonzero figure is a `Dual` that failed to stay in registers — see the forward
  section below.
- **commsB** / **isbits** — tape shape: total bytes across the tape's comms stacks, and whether they
  are all pointer-free. A `MemoryRef` in a comms tuple makes the backing buffer GC-tracked, costing a
  write barrier on every push. Note `commsB` is a **whole-function sum over all blocks**, not a
  per-iteration figure — for a loop workload the loop block dominates it, but for a straight-line one
  it counts blocks that run once. Blank (`—`) on forward rows: the forward carrier has no tape.

### Pre-allocated vs fresh tape

`memloop!` appears twice on purpose. The `(prealloc)` variant goes through
`build_ctx(...; prealloc=true)`, reusing one tape across calls; the other builds a fresh `Ctx()` tape
per call — which is what plain `gradient` does, and what *every recursive inner call* does. Costs
that a pre-allocated context amortizes away are paid in full on the fresh-tape path, so an
optimization that trades an allocation for less per-iteration work will look much better on one than
the other. Quote both.

## The forward workloads

There is no tape, no context and nothing to pre-allocate, so a forward workload is just
`frule!!(Dual(f, df), Dual(x, dx), …)` — one variant per primal, and **allocs should be 0**. What the
set covers, beyond the five primals shared with reverse:

| workload | what it stresses |
|---|---|
| `fwd polychain` | branch-free scalar chain: the per-operation floor for a `Dual` |
| `fwd vecloop_ret!` | shadow array writes with no read-back |
| `fwd applyN closure` | higher-order: a closure *value* applied through the dual convention each iteration |
| `fwd cpoly ComplexF64` | a struct-shaped tangent (`Tangent{@NamedTuple{re,im}}`), no array involved |
| `fwd polychain order-2` | nested duals: the transform applied to its own output |

Order-2 seeds must be **uniform** — the function and the non-differentiable arguments are nested to
the order too (`nest2`), not just the value being differentiated; a non-uniform seed is rejected.

Three forward paths currently allocate per iteration, which the table makes obvious and which
`ISSUES.md` #54 records with numbers: `structloop` (mutable-struct shadow), `applyN closure`
(closure-valued callee), and `cpoly` (struct tangent — ~19 allocations for a *single* complex
multiply). `structloop` is the sharp one: forward is ~14× *slower* than reverse on the same primal,
which is backwards for a one-input function.

`vecloop!` and `memloop!` have no forward twin: a `nothing`-returning primal fails forward-mode IR
verification (`ISSUES.md` #53), so `vecloop_ret!` returns an element instead.

The three allocating workloads are also the **noisy** ones — GC lands in the samples, so their minima
need a longer budget than the rest. At `--seconds=0.3`, `fwd applyN closure` measured −11% against an
identical tree, purely from noise; across runs at 1–2 s it holds to ~2%. Use `--seconds=2` before
believing any movement in those three rows.

## Writing a workload

Every workload registers **two** benchmarkables: the AD call, and the same call undifferentiated
under `"$k [primal]"` (`primal!` in `workloads.jl`). Give the primal the same setup shape as the AD
workload — same arrays, same lengths, same warm-up — so the ratio compares two calls on identical
state. Primals may use a high `evals` freely; there is no shadow to share.

Two traps, both **reverse-mode only** — forward mode has neither, since a shadow there is written
rather than accumulated into and there is no tape, so `evals>1` is always safe. Both have already
produced wrong numbers in this repo:

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

## Comparing against Enzyme

```bash
julia +1.13 --project=bench bench/run.jl --vs-enzyme
julia +1.13 --project=bench bench/run.jl --vs-enzyme --mode=forward --n=10000 --seconds=2
```

Prints the usual Differ table, then a second one pairing each workload against the same primal run
through [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) (`Enzyme.autodiff(Reverse, ...)` /
`Enzyme.autodiff(Forward, ...)`, called directly — no thunk/compile-once machinery needed, since a
repeated plain `autodiff` call is already 0-alloc). `--vs-enzyme` is the only thing that pulls Enzyme
in: it's `using`d from `enzyme_workloads.jl`, which is only `include`d when the flag is passed, so a
plain `run.jl` invocation never pays Enzyme's first-run precompile (~60-90s the first time its
manifest entry is instantiated).

The comparison covers the **core workload set** — the 8 primals whose shape maps directly onto
Enzyme's activity annotations (`readonly`, `vecloop!`, `wrloop`, `straightline!`, `structloop`,
`scalarcf`, `memloop!`, `loopdot`), in both modes where Differ has a forward twin (`memloop!` and
`loopdot` are reverse-only, like the Differ side), plus `fwd polychain`. **Not covered**, and not
planned without further investigation:

- `cpoly` — Enzyme's activity annotations for a `ComplexF64`-typed argument need their own look; this
  isn't the same as an `isbits` struct field-by-field.
- `applyN` (closure-valued callee) — Enzyme differentiates closures differently from Differ's
  `Dual`-carrying-a-value convention; wiring the two up to compare fairly is unstarted work.
- `polychain order-2` — Enzyme's higher-order story doesn't mirror Differ's transform-applied-to-its-
  own-output approach; there's no obvious equivalent to ask for the same thing.

`memloop!` pairs against Differ's `(prealloc)` key specifically, not the fresh-`Ctx()` one: Enzyme has
no tape and no comparable fresh-vs-reused-context distinction, so its natural steady-state call is the
fair comparison against Differ's pre-allocated-context path.

The `Differ/Enzyme` column is minimum-time ratio; above 1× means Differ is slower on that workload.
Unlike `compare.jl` this isn't a regression check against history, so there's no noise tolerance —
just the two numbers next to each other.

## Comparing against Mooncake

```bash
julia +1.13 --project=bench bench/run.jl --vs-mooncake
julia +1.13 --project=bench bench/run.jl --vs-mooncake --mode=forward --n=10000 --seconds=2
julia +1.13 --project=bench bench/run.jl --vs-enzyme --vs-mooncake   # both, one run
```

Same shape as `--vs-enzyme`: a second table pairing each workload against the same primal run through
[Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl), only `using`d (from `mooncake_workloads.jl`)
when the flag is passed, so a plain `run.jl` invocation never pays Mooncake's first-run precompile
(~80s the first time its manifest entry is instantiated). Mooncake comes from the General registry —
not a `path` source to `../Mooncake.jl` — so `compare.jl`'s baseline worktree (which lives outside
this checkout) can still resolve it.

Mooncake is the closest comparison of the three: Differ's tangent system (`Tangent`/`MutableTangent`,
`FData`/`RData`, `CoDual`/`Dual`, `rrule!!`/`frule!!`) is a direct port of Mooncake's, so
`mooncake_workloads.jl` calls Mooncake at the same level Differ's own workloads call Differ — build a
rule once (`Mooncake.build_rrule`/`build_frule`, the `build_ctx` analogue) in `setup`, then in the
timed body call the rule and, for reverse, its pullback — rather than through Mooncake's higher-level
`prepare_gradient_cache`/`value_and_gradient!!` convenience API. Every Mooncake name is written
`Mooncake.foo` rather than pulled in unqualified: workloads.jl already has Differ's `CoDual`, `Dual`,
`NoRData`, `rrule!!`, `frule!!`, etc. in scope, and Mooncake uses the identical names for the identical
concepts.

The comparison covers the same **core workload set** as `--vs-enzyme` (`readonly`, `vecloop!`,
`wrloop`, `straightline!`, `structloop`, `scalarcf`, `memloop!`, `loopdot`, plus `fwd polychain`).
**Not covered**: `cpoly` — Mooncake's `build_tangent(ComplexF64, ...)` doesn't accept the
`(re, im)` pair the way Differ's does on this Mooncake version, and `applyN`/`polychain order-2` for
the same reasons `--vs-enzyme` excludes them (no direct equivalent to ask Mooncake for).

`memloop!` pairs against Differ's `(prealloc)` key, matching `--vs-enzyme`'s convention: a rule built
once and reused is Mooncake's natural steady-state call, the fair comparison against Differ's
pre-allocated-context path.

The `Differ/Mooncake` column is minimum-time ratio; above 1× means Differ is slower on that workload.
