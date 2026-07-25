Differ package is a small, experimental post-optimization IR-based automatic differentiator.

This is implemented with reference to julia v1.13, always run code using `julia +1.13` as your engine, 
and use `../julia/Compiler` as your reference for anything related to the compiler.

The tangent representation is a port of Mooncake.jl's tangent / fdata / rdata type system
(`tangent_type`/`fdata_type`/`rdata_type` keyed on the *primal* type, `Tangent`/`MutableTangent`/
`PossiblyUninitTangent`, `FData`/`RData`, `zero_tangent`/`increment!!`, and the Mooncake-shaped
`Dual`/`CoDual` carriers). Forward mode is built on this `Dual`, reverse mode on this `CoDual`.
`../Mooncake.jl` is the reference for anything tangent-system-related.

## Rule interfaces

Both modes follow Mooncake's naming: `frule!!` (forward) and `rrule!!` (reverse) are the
hand-written *primitive* rules; a composite function gets a *derived* rule built by transforming its
IR.

- **Forward**: `frule!!(Dual(f, tf), Dual(x, dx), …) -> Dual`. `frule!!` has a `@generated`
  composite fallback, so it works on anything.
- **Reverse**: `rrule!!(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) -> (ycd, pullback)`, where
  **the pullback closure *is* the tape**. `rrule!!` is a single stateless generic function under one
  uniform convention: hand-written primitives (`src/rrules.jl`) and a `@generated` derived fallback
  (for any composite function) are both methods of it. The tape lives not in the rule but in `ctx`,
  threaded through — so `rrule!!` itself is reentrant, and it is the *`Ctx`* that is single-use per
  task. A hand rule ignores `ctx` but **must** declare that slot `::AbstractCtx` (never a concrete
  subtype): that dispatch-neutrality is what keeps hand-rule-vs-fallback dispatch unambiguous.
- **Context**: `build_ctx(f, argtypes)` returns a `Ctx` wrapping a tape allocated once and reused
  (stacks reset) per call — the pre-allocated fast path (not reentrant/thread-safe, one per task).
  `build_ctx(…; prealloc=false)` returns `Ctx()`, which allocates a fresh tape per call (also what
  every recursive inner call and plain `gradient` use).
- User-facing reverse entry points: `gradient(f, args...)` (allocates everything) and
  `gradient!`/`value_and_gradient!(ctx, fcd, argcds...)` (caller supplies both the context and each
  argument's shadow; a steady-state call allocates only the returned tuple).

## Differ skills

`Differ/.claude/skills/` has three skills describing the structure and meaning of Differ's code in
detail. They should trigger automatically when relevant, but can also be invoked directly (e.g.
`/differ-architecture`):

- **`differ-architecture`** — orientation: the custom `AbstractInterpreter` compiler-plugin design,
  the `Dual`/`frule` calling convention, the file map, how to run tests and inspect dualized IR.
  Start here.
- **`differ-ircode-dualization`** — deep-dive internals of the split-shadow dualization engine
  (`dualize_to_ircode` in `src/forward_interp.jl`): how control flow is handled, the `PhiNode`
  forward-reference mechanism, and known `Core.Compiler.verify_ir` gotchas.
- **`differ-extending-ir-support`** — playbook for adding support for a new Julia IR construct
  (e.g. the next milestone, exception handling for `try`/`catch`), including what's already known
  about that specific follow-up. 

