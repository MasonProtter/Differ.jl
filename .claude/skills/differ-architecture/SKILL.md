---
name: differ-architecture
description: Orients Claude to Differ's overall design before making changes — the 4-package monorepo split (Contextual/DifferCore/DifferForwards/DifferReverse), the generic ContextualInterpreter compiler-plugin hooks, the Dual/frule!! and CoDual/rrule!! calling conventions, and where to go for a deeper dive on any one piece. Use this whenever working on Differ (adding features, fixing bugs, reviewing code, explaining how it works), especially at the start of a session before touching any specific package for the first time.
---

# Differ architecture

Differ is a small, experimental **post-optimization IR-based automatic differentiator** for
Julia. Instead of transforming source code or a macro-expanded expression, it transforms a
function's *already fully-optimized* `Core.Compiler.IRCode` — the same IR Julia's own optimizer
produces right before codegen — and splices the transformed IR back into the compiler pipeline so
it gets treated as an ordinary compiled method. There is no separate "AD runtime": the derivative
code IS a normal `CodeInstance`, inlined and optimized exactly like anything else Julia compiles.

This is a workspace/monorepo of four independently-installable packages plus a thin re-export
meta-package. Read this skill first, then follow the pointers below for whichever piece you're
actually touching — each has its own dense, precise skill, and this one deliberately stays at
orientation level rather than duplicating them.

## Package map

| Package | Role | Deep-dive skill |
|---|---|---|
| `Contextual/` | Mode-agnostic compiler-plugin machinery: `ContextualInterpreter{T,S}` and the two pipeline hooks it overrides. No AD logic of its own. | (covered below, no dedicated skill needed) |
| `DifferCore/` | The tangent/fdata/rdata type system, keyed on the *primal* type. Zero forward/reverse-specific logic. Both engines depend on it. | `differ-tangent-system` |
| `DifferForwards/` | Forward mode: the `Dual` carrier, `frule!!`, and the split-shadow `dualize_to_ircode` engine. Depends on `Contextual` + `DifferCore`. | `differ-forward-dualization` (engine internals), `differ-extending-ir-support` (playbook for adding IR support) |
| `DifferReverse/` | Reverse mode: the `CoDual` carrier, `rrule!!`, and the two-carrier `Tape`/pullback engine. Depends on `Contextual` + `DifferCore`. | `differ-reverse-engine` (engine internals), `differ-extending-reverse-support` (playbook for adding IR support) |
| `Differ/` (repo root, `src/Differ.jl`) | Thin re-export meta-package: `using Differ` pulls in all four sub-packages, reproducing the pre-split monolith's combined namespace. No logic of its own. | — |
| `ext/`, `DifferForwards/ext/`, `DifferReverse/ext/` | Package extensions: `DifferentiationInterface.jl` integration (`AutoDifferForwards`/`AutoDifferReverse`) for both modes, plus `DifferForwards/ext/DifferForwardsOverReverseExt.jl` (forward-over-reverse composition hooks — see "Known limitations" below) and its `rules_ad_runtime.jl`, plus a `SpecialFunctions.jl` rule extension per mode (`Differ{Forwards,Reverse}SpecialFunctionsExt.jl`). | — |

Each sub-package has its own `Project.toml`/`test/Project.toml` and can be tested standalone.
`Differ`'s root `Project.toml` uses Julia's workspace mechanism (`[workspace] projects = [...]`,
`[sources]` pointing at each sub-directory) to develop all of them together.

`DifferForwards`/`DifferReverse` never depend on each other in `[deps]` — they're connected only
through `DifferForwards/ext/DifferForwardsOverReverseExt.jl` (a weak-dep extension, loaded only
when both packages are present), which is what lets forward mode differentiate *through* a call to
reverse mode. See "Known limitations" below — this composition currently doesn't work end to end.

## The core trick: two pipeline hooks, no CodeInfo, no rettype patching

`ContextualInterpreter{T,S}` (`Contextual/src/Contextual.jl`) is a normal `AbstractInterpreter` in
every respect except two overridden hooks. `T` is the plugin's immutable "owner" type — it IS the
`cache_owner` partition key directly, so it must stay portable/immutable (`DifferForwards` uses a
fieldless `Forward` singleton; `DifferReverse` uses `Reverse(nested_forward::Bool)`, one config bit
needed to give a forward-over-reverse build a distinct cache partition). `S` is whatever mutable,
per-session scratch state the plugin needs (an in-progress-call set, a bail-reason cache, …),
deliberately kept out of `T`/`cache_owner`'s reach.

1. **`finishinfer!`** — the point where Julia's inference machinery is about to freeze a method's
   return type into its `CodeInstance`. `Contextual` calls the plugin's `build_contextual_ir(interp,
   mi)` hook here, and if it returns a real `IRCode` (not `nothing`), stashes it in
   `interp.transformed_ir[mi]`, sets `me.bestguess` to its return type, and folds any backedges the
   plugin discovered into `me.src.edges` — so the *ordinary* codepath writes the correct return type
   and real backedges, once, with no post-hoc patching.
2. **`optimize`** — where Julia would normally run its own optimization passes on inferred
   `CodeInfo`. `Contextual` instead installs the already-built `IRCode` from step 1, runs the
   IPO-safe passes on it (`run_ipo_passes!`: `compact!` → `ssa_inlining_pass!` → `compact!` →
   `sroa_pass!` → `adce_pass!` → `compact!`), and calls `finishopt!` directly.

`build_contextual_ir(::ContextualInterpreter, ::MethodInstance) = nothing` is the default — any
`MethodInstance` a plugin doesn't recognize (not one of its own carrier shapes, or a construct its
transform can't handle) flows through the ordinary, unmodified pipeline. When a plugin's transform
bails, the carrier's throwing stub body compiles normally and raises when invoked — a graceful
failure, never a miscompile.

**This is a hard project constraint, not a style preference**: an earlier attempt built a second
engine that dualized lowered `CodeInfo` and patched the `CodeInstance` return type after the fact.
That was explicitly rejected as unworkable — never resurrect that shape. Transform `IRCode` only,
never `CodeInfo`, never patch rettype after the fact.

`DifferForwards`/`DifferReverse` each implement `build_contextual_ir` for their own carrier shape:
forward mode recognizes a `dualized_impl(dualargs::Dual...)` carrier and asks `dualize_to_ircode` to
transform the primal's `IRCode`; reverse mode recognizes `reverse_fwds_impl`/`reverse_pullback_impl`
carriers and asks `reverse_fwds_to_ircode`/`reverse_pullback_to_ircode`. Both are single-source-pass
transforms on the primal's already-optimized `IRCode` — see `differ-forward-dualization` and
`differ-reverse-engine` for how each actually works; they are genuinely different designs, not a
mirror image of each other (forward mode preserves block topology 1:1 and tracks one split-shadow
value pair per SSA statement; reverse mode builds two separately-compiled carriers around an
explicit `Tape` value, and only the forwards carrier keeps the primal's topology).

## The tangent system (`DifferCore`)

Both carriers are built on a primal-type-keyed tangent system ported from Mooncake.jl —
`tangent_type`/`fdata_type`/`rdata_type`, `Tangent`/`MutableTangent`/`PossiblyUninitTangent`,
`FData`/`RData`, `zero_tangent`/`increment!!`. This lives entirely in `DifferCore`, with zero
forward/reverse-specific logic — see `differ-tangent-system` for the full reference. The short
version: every primal type `P` has exactly one tangent type `tangent_type(P)` (`NoTangent` for
non-differentiable/singleton types, itself for a scalar, `Tangent{...}`/`MutableTangent{...}` for a
struct, a per-field tuple for a `Tuple`/`NamedTuple`, a same-shape `Array`/`MemoryRef`/`Memory` for
an array type), and that type splits further into an **fdata** half (address-identified, threaded
forward and incremented in place) and an **rdata** half (value-identified, only accumulable during
a reverse pass) — the fdata/rdata split is what `CoDual` uses that `Dual` doesn't need.

## The `Dual`/`frule!!` calling convention (forward mode)

- `Dual{P,T}` (`DifferForwards/src/dual.jl`) wraps a primal value `primal::P` and a tangent
  `tangent::T`, with the invariant `T == tangent_type(P)`. `d.x`/`d.dx` alias `primal`/`tangent`;
  `primal(d)`/`tangent(d)` also work. `tangent_type(::Type{<:Dual}) = itself` — a `Dual` is its own
  tangent type, which is what makes higher-order nesting compose cleanly.
- To differentiate `f` at arguments `x, y, …` with tangents `dx, dy, …`, call
  `frule!!(Dual(f, tf), Dual(x, dx), Dual(y, dy), …)`. The function itself is wrapped in a `Dual`
  too, carrying a real tangent `tf` (not a forced zero) — that's what lets differentiating a closure
  w.r.t. its captures work. Use `zero_tangent(f)` to hold `f` constant.
- `frule!!(dualargs::Dual...)` is `@generated`: for a composite primal with no hand-written rule,
  the generator compiles a dualized version of the primal's body under `ContextualInterpreter{Forward}`
  and returns a trivial body that `invoke`s the resulting `CodeInstance`. Only `sin`/`cos`/a handful
  of `Base`/stdlib functions have hand-written `frule!!` rules (`DifferForwards/src/frules.jl`,
  `rules_math.jl`/`rules_reductions.jl`/`rules_broadcast.jl`/`rules_indexing.jl`/`rules_linalg.jl`);
  ordinary arithmetic (`+`,`-`,`*`,`/`) inlines down to intrinsics handled directly by
  `src/intrinsics.jl`'s `Val`-dispatch layer, not routed through `frule!!`.
- User-facing entry points: `code_dual_ircode`/`@code_dual_ircode` (`reflection.jl`) for
  inspecting the dualized IR directly — always prefer this over reasoning about the transform from
  source alone. See `differ-forward-dualization` for the engine itself.

## The `CoDual`/`rrule!!` calling convention (reverse mode)

- `CoDual{Tx,Tdx}` (`DifferReverse/src/codual.jl`) pairs a primal `Tx` with its **fdata** half
  `Tdx` (not a full tangent — reverse mode reconstructs a full tangent only where needed, via
  `tangent(fdata, rdata)`).
- `rrule!!(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) -> (ycd, pullback)` is a single
  generic function: hand-written primitives (`DifferReverse/src/rrules.jl`, plus the
  `rules_*.jl` files) and an `@generated` derived fallback are both methods of it. **The pullback
  closure returned by the derived path *is* the tape** — `reverse_pullback_impl(tape::Tape, seed)`.
  A hand rule's pullback can be anything (e.g. `SinPullback` is a plain struct holding the saved
  input); only the derived path's pullback is a `Tape`.
- `ctx::AbstractCtx` holds the `Tape` *between calls* so it can be reused instead of reallocated —
  it is not a second tape. A hand rule must declare its `ctx` slot `::AbstractCtx` (never a
  concrete subtype), which is what keeps hand-rule-vs-fallback dispatch unambiguous (dispatch
  specificity is decided entirely by the `fcd`/arg slots). `build_ctx(f, argtypes)` returns a `Ctx`
  wrapping a tape allocated once and reused per call (the pre-allocated fast path, not
  reentrant/thread-safe — one per task); a bare `Ctx()` is a fresh-tape
  `Ctx()` per call, used by every recursive inner call and by plain `gradient`.
- User-facing entry points: `rev_gradient(f, args...)` (allocates everything) and
  `rev_gradient!`/`value_and_gradient!(ctx, fcd, argcds...)` (caller supplies the context and each
  argument's shadow; a steady-state call allocates only the returned tuple). `rev_gradient`/
  `rev_gradient!` are `public`, not `export`ed — `DifferentiationInterface.jl` is the primary
  user-facing entry point now. `code_reverse_fwds_ircode`/`code_reverse_pullback_ircode`
  (`reflection.jl`) are the debugging entry points, mirroring `code_dual_ircode`. See
  `differ-reverse-engine` for the engine itself.

## Running things

Each sub-package is independently testable:

```sh
julia +1.13 --project=<Pkg> -e 'using Pkg; Pkg.test()'   # <Pkg> = Contextual, DifferCore, DifferForwards, or DifferReverse
```

or equivalently `julia +1.13 --project=<Pkg>/test <Pkg>/test/runtests.jl`. Both work post-split
(each sub-package has a real, named `test/Project.toml`) — `Pkg.test()` is the CI-realistic path and
is preferred, since its stricter sandbox (test `Project.toml`'s own `[deps]` only) catches real
missing-dependency gaps direct invocation misses. The repo-root `test/runtests.jl`
(`--project=test`) holds only genuine cross-package integration tests that don't belong to any one
sub-package: backedge/invalidation behavior spanning both `frule!!` and `rrule!!`, and a
no-method-ambiguities check across the whole re-exported `Differ` namespace.

**One caveat, not a reason to avoid `Pkg.test()`**: `Pkg.test()` runs under `--check-bounds=yes`,
which can mask a bug where this project's own emitted bounds check was supposed to substitute for
one inherited from the primal but doesn't actually fire — `--check-bounds=yes` raises a clean
`BoundsError` in that case where the real default (`--check-bounds=auto`) would silently corrupt
memory or segfault. When touching anything boundscheck-related, also run once under the default
(`julia +1.13 --project=<Pkg>/test <Pkg>/test/runtests.jl`, no `--check-bounds` override) — see
`differ-forward-dualization`'s gotcha list for the concrete case this bit.

Inspect dualized/reverse IR directly rather than reasoning from source alone:
`code_dual_ircode(f, (Float64,))` / `@code_dual_ircode f(1.0)`, and
`code_reverse_fwds_ircode`/`code_reverse_pullback_ircode` (both in each package's `reflection.jl`).

## Current support status (summary — see the deep-dive skills for specifics)

**Forward mode** (`DifferForwards`) supports: straight-line code, branches/loops, try/catch, array
element read/write and mutation, mutable-struct field mutation, array allocation, tuples/closures
(differentiate w.r.t. captures), `GC.@preserve` + raw pointer arithmetic, `Expr(:foreigncall)`
(`ccall`, registered per-target — `memmove`/`memcpy` today) + `Expr(:loopinfo)` (`@simd`), dynamic
dispatch (via a runtime `dynamic_frule` dispatcher), higher-order differentiation (both hand-nested
`Dual` seeds and composed `D`-of-`D`), and both self- and mutual recursion. See
`differ-forward-dualization` for the engine and `differ-extending-ir-support` for exactly what's
still unsupported and how to close a gap.

**Reverse mode** (`DifferReverse`) supports: the same core (branches/loops, array indexing/mutation
via a shadow-chain comms scheme, mutable-struct field mutation, `Core.tuple`, `Core.ifelse`,
`@simd`/`:loopinfo`), direct self-recursion (closed-form `Tape` type) and argument-position callees,
but not yet: mutual recursion, dynamic dispatch (no `dynamic_rrule` equivalent), vararg primal
methods, or `Core.Box`/abstract-field `setfield!`. See `differ-reverse-engine` for the engine and
`differ-extending-reverse-support` for the exact gap list.

**Known limitations, documented in `ISSUES.md`, not being actively chased right now:**
- **ISSUES #84**: a pre-existing (not split-caused) reentrancy bug in `typeinf_ext_toplevel` called
  from inside a `@generated` function, at ≥3 levels of nested dualization — see
  `differ-forward-dualization`'s gotcha list.

## World-age hygiene (the `at_world` contract)

`jl_call_staged` pins a `@generated` generator body's task world age to the generated method's
`Method.primary_world`, fixed at definition. Every ordinary function the transform calls therefore
dispatches at that pin, so **methods added by any later-loaded package — a sibling package, and always
a package extension — are invisible to it**. That is what broke forward-over-reverse across the package
split (ISSUES #85): `DifferReverse`'s `tangent_type` overrides and every `DifferForwardsOverReverseExt`
coupling hook silently resolved to `DifferCore`'s generic fallback and to their own inert defaults.

The contract, stated in `Contextual`'s module header:

> Pass code running at generator time may reach another package only through `at_world`, or through a
> call emitted into the IR (compiled later, at the real world). Never by direct dispatch, and never
> through mutable global state. Every such lookup must record an `mt_edge!`.

`at_world(interp_or_world, f, args...)` wraps `Core._call_in_world_total`, which — unlike
`Base.invoke_in_world`, a silent no-op inside a generator — is not `in_pure_callback`-guarded, and
whose world switch covers nested dispatch inside the callee too (`tangent_type(Stack{T})` recurses into
`tangent_type(T)`, so a switch covering only the outer frame would fix nothing).

In practice both engines funnel their queries rather than converting every call site: `tt`/`dualt`/`zt`
and the `fsel_*` hook wrappers in `dualize_to_ircode` (handed to `builtins.jl`/`intrinsics.jl`/
`foreigncalls.jl` via the `*_ctx` bundles), and `_tt`/`_fcdtype`/`rdtype`/`fdtype` in
`reverse_interp.jl` (handed to `builtins_reverse.jl` the same way). Adding a bare `tangent_type(T)` call
to transform code silently reintroduces the bug — use the funnel.

The invariant is asserted directly in `Contextual/test/runtests.jl`, and the load-order case the
ordinary suite can't reach is covered by `DifferForwards/test/test_late_extension_load.jl`.

## Where to go next

- `differ-tangent-system` — `DifferCore`'s tangent/fdata/rdata system.
- `differ-forward-dualization` — `DifferForwards`' `dualize_to_ircode` engine internals.
- `differ-reverse-engine` — `DifferReverse`'s `reverse_fwds_impl`/`reverse_pullback_impl`/`Tape`
  engine internals.
- `differ-extending-ir-support` / `differ-extending-reverse-support` — playbooks for adding support
  for a new Julia IR construct, forward- and reverse-mode respectively.
