---
name: adnext-architecture
description: Orients Claude to ADNext's overall design before making changes — the custom AbstractInterpreter plugin architecture that hooks Julia's compiler pipeline, the Dual/frule calling convention, and how the pieces in src/ fit together. Use this whenever working on ADNext (adding features, fixing bugs, reviewing code, explaining how it works), especially before touching contextual.jl, forward_interp.jl, frules.jl, or reflection.jl for the first time in a session.
---

# ADNext architecture

ADNext is a small, experimental **post-optimization IR-based automatic differentiator** for
Julia. Instead of transforming source code or a macro-expanded expression, it transforms a
function's *already fully-optimized* `Core.Compiler.IRCode` — the same IR Julia's own optimizer
produces right before codegen — and splices the transformed IR back into the compiler pipeline so
it gets treated as an ordinary compiled method. There is no separate "AD runtime": the derivative
code IS a normal `CodeInstance`, inlined and optimized exactly like anything else Julia compiles.

## File map (`src/`)

- **`ADNext.jl`** — module entry point; just `include`s the other files in dependency order
  (`frules.jl`, `contextual.jl`, `forward_interp.jl`, `reflection.jl`) and exports the public API.
- **`frules.jl`** — the `Dual{T,U}` struct, `NoFData` (the "no forward data" zero-tangent
  singleton), `struct_zero` (structural zero-tangent constructor), and hand-written `frule`
  methods for `sin`/`cos` (the only two — see below for why more aren't needed).
- **`contextual.jl`** — the *mode-agnostic* compiler-plugin machinery: `ADInterpreter{M<:ADMode}`
  (a custom `AbstractInterpreter`) and the two pipeline seams it overrides, `finishinfer!` and
  `optimize`. Nothing forward-mode-specific lives here — a future `Reverse` mode would reuse this
  file unchanged.
- **`forward_interp.jl`** — all forward-mode-specific glue (`build_contextual_ir`,
  `is_dualized_impl`, `primal_of_impl`, `build_dual_ir`, `frule_codeinstance`, the `dualized_impl`
  carrier stub, the `@generated frule` fallback) **and** the dualization engine itself
  (`dualize_to_ircode`). See the `adnext-ircode-dualization` skill for a deep dive on that engine —
  this file is the one you'll touch most often, and it's dense.
- **`reflection.jl`** — `code_dual_ircode`/`@code_dual_ircode`, a debugging/reflection entry point
  that reproduces exactly what the compiler-plugin seam installs, without going through the full
  `@generated` machinery. Use this (not raw `frule` calls) whenever you need to *see* the dualized
  IR — e.g. `code_dual_ircode(f, (Float64,))` or `@code_dual_ircode f(1.0)`.
- **`test/runtests.jl`** — the test suite; also doubles as documentation-by-example of what's
  currently supported (each primal test function has a comment explaining what IR feature it
  exercises).

## The core trick: two pipeline seams, no CodeInfo, no rettype patching

`ADInterpreter{M}` is a normal `AbstractInterpreter` in every respect except two overridden hooks:

1. **`finishinfer!`** — the point where Julia's inference machinery is about to freeze a method's
   return type into its `CodeInstance`. ADNext calls the mode's `build_contextual_ir(interp, mi)`
   hook here, and if it returns a real `IRCode` (not `nothing`), stashes it in
   `interp.transformed_ir[mi]` and sets `me.bestguess = compute_ir_rettype(ir)` — so the *ordinary*
   codepath writes the correct return type, once, with no post-hoc patching.
2. **`optimize`** — where Julia would normally run its own optimization passes on inferred
   `CodeInfo`. ADNext instead installs the already-built `IRCode` from step 1, runs the IPO-safe
   passes on it (`run_ipo_passes!`: `compact!` → `ssa_inlining_pass!` → `compact!` → `sroa_pass!` →
   `adce_pass!` → `compact!`), and calls `finishopt!` directly.

**This is a hard project constraint, not a style preference**: an earlier attempt built a second
engine that dualized lowered `CodeInfo` and patched the `CodeInstance` return type after the fact.
That was explicitly rejected as "a mess" — never resurrect that shape. Transform IRCode only,
never CodeInfo, never patch rettype after the fact.

For forward mode specifically, `build_contextual_ir` recognizes a *carrier* `MethodInstance`
(`dualized_impl(dualargs::Dual...)`, whose `specTypes` encodes the dual signature) and asks
`build_dual_ir` to transform the corresponding *primal* method's already-optimized `IRCode` via
`dualize_to_ircode`. Any `MethodInstance` that isn't a `dualized_impl` specialization — or any
input `dualize_to_ircode` can't handle — flows through the ordinary, unmodified pipeline.

## The Dual/frule calling convention

- `Dual{T,U}` wraps a primal value `x::T` and a tangent `dx::U` where `U <: Union{T, NoFData}`.
  `NoFData()` is the tangent of non-differentiable things (functions, singletons, `Bool`, …).
  `d.x`/`d.y`/`d.z` all alias the primal field; `d.dx`/`d.dy`/`d.dz` all alias the tangent — pick
  whichever reads best at the call site.
- To differentiate `f` at arguments `x, y, …` with tangents `dx, dy, …`, call
  `frule(Dual(f, NoFData()), Dual(x, dx), Dual(y, dy), …)`. The *function itself* is wrapped in a
  `Dual` too — that's what lets `frule` dispatch on `Dual{typeof(f)}` to find hand-written rules.
- Only `sin`/`cos` have hand-written `frule` methods in `frules.jl`. Deliberately **no** rules
  exist for `+`, `-`, `*`, `/`, or comparisons: those inline down to intrinsics
  (`add_float`/`mul_float`/`lt_float`/…), which the dualization engine differentiates directly at
  the intrinsic level. This means arithmetic works generically for *any* type (`Float32`,
  `Complex`, a user struct via `getfield`) without a per-type rule — don't add arithmetic `frule`s;
  add intrinsic cases in `dualize_to_ircode` instead if something's missing.
- `frule(dualargs::Dual...)` itself is `@generated`: for a composite primal with no hand-written
  rule, the generator compiles a dualized version of the primal's body under
  `ADInterpreter{Forward}` and returns a trivial body that `invoke`s the resulting `CodeInstance`.

## Running things

- Tests: `julia +1.13 --project=ADNext -e 'using Pkg; Pkg.test()'` — this project targets Julia
  1.13, invoked via the `+1.13` juliaup channel selector, not whatever `julia` resolves to by
  default.
- Inspect dualized IR: `code_dual_ircode(f, (Float64,))` or `@code_dual_ircode f(1.0)` from
  `reflection.jl` — always prefer this over trying to reason about the transform from source
  alone when debugging.

## Current support status

Straight-line code, branches, and loops (`GotoNode`/`GotoIfNot`/`PhiNode`) are all supported.
Exception handling (`try`/`catch`, i.e. `EnterNode`/`PhiCNode`/`UpsilonNode`) is not yet
supported and bails gracefully (throws a clear `ErrorException` rather than miscompiling). See the
`adnext-ircode-dualization` skill for how the supported constructs are actually handled, and the
`adnext-extending-ir-support` skill before adding support for anything new.
