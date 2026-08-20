Differ package is a small, experimental post-optimization IR-based automatic differentiator.

This is implemented with reference to julia v1.13, always run code using `julia +1.13` as your engine, 
and use `../julia/Compiler` as your reference for anything related to the compiler.

The tangent representation is a port of Mooncake.jl's tangent / fdata / rdata type system
(`tangent_type`/`fdata_type`/`rdata_type` keyed on the *primal* type, `Tangent`/`MutableTangent`/
`PossiblyUninitTangent`, `FData`/`RData`, `zero_tangent`/`increment!!`, and the Mooncake-shaped
`Dual`/`CoDual` carriers). Forward mode is built on this `Dual`, reverse mode on this `CoDual`.
`../Mooncake.jl` is the reference for anything tangent-system-related.

One addition of Differ's own: `Inactive` (`DifferCore/src/inactive.jl`) marks a value the caller
holds constant, and is a legal inhabitant of a shadow slot alongside the primal-derived one.
`NoTangent` cannot do that job — it already means "this *type* has no tangent space", and its fdata
is `NoFData`, which is also an active `Float64`'s — so constancy needs its own inhabitant, one that
`fdata`/`rdata` preserve and that composes inside aggregates. Both carriers admit it:
`fdata_shadow_type(P)` gives the legal set for a `CoDual`'s shadow,
`Union{fdata_type(tangent_type(P)), Inactive}`, and `tangent_shadow_type(P)` the set for a `Dual`'s,
`Union{tangent_type(P), Inactive}` (`shadow_type` is a deprecated alias for the fdata one). Each is a
**validity predicate, never a declaration**: every declared slot, field, comms item and SSA type
stays concrete, so no union reaches a hot path.
`Inactive` is a strong zero — `increment!!(Inactive(), y) === Inactive()` (accumulating into a
constant discards) and `increment!!(x, Inactive()) === x` (a constant contributes nothing), which
makes `increment!!` deliberately non-commutative: slot 1 is the accumulator that owns storage, slot 2
is a contribution. `NoTangent` keeps *no* absorbing arm, so a mis-analysed active value still raises a
`MethodError` instead of dropping a gradient.

When writing comments, be terse and too the point (though not opaque). Don't use Claude-jargon/metaphors like calling a function a "seam", or "belt-and-suspenders", "brace", etc.


## Rule interfaces

Both modes follow Mooncake's naming: `frule!!` (forward) and `rrule!!` (reverse) are the
hand-written *primitive* rules; a composite function gets a *derived* rule built by transforming its
IR.

- **Forward**: `frule!!(Dual(f, tf), Dual(x, dx), …) -> Dual`. `frule!!` has a `@generated`
  composite fallback, so it works on anything. A hand rule must accept `Inactive()` in any
  differentiable slot (the transform routes a constant operand straight to it, which is what keeps
  the strong zero) and must never *return* it — `frule_split!` declares a nested call's result
  tangent at `tangent_type(R)` before the callee is compiled. Forward signatures leave the tangent
  parameter free, so an unwidened rule still matches and fails inside its body;
  `DifferForwards/test/test_forward_rule_activity.jl` audits every method under every activity mask.
- **Reverse**: `rrule!!(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) -> (ycd, pullback)`, where
  **the pullback closure *is* the tape** — the derived path's `pullback` is literally a `Tape` value.
  `ctx` just holds that same `Tape` object *between calls* so it can be reused instead of reallocated
  (`Ctx(nothing)` allocates fresh each call; `build_ctx` resets and hands back the
  cached one) — not a second tape. That's why `rrule!!` itself is stateless/reentrant while `Ctx` is
  single-use per task. `rrule!!` is a single generic function: hand-written primitives
  (`src/rrules.jl`) and a `@generated` derived fallback are both methods of it. A hand rule ignores
  `ctx` but **must** declare that slot `::AbstractCtx` (never a concrete subtype), which is what keeps
  hand-rule-vs-fallback dispatch unambiguous.
- **Context**: `build_ctx(f, argtypes)` returns a `Ctx` wrapping a tape allocated once and reused
  (stacks reset) per call — the pre-allocated fast path (not reentrant/thread-safe, one per task).
  A bare `Ctx()` allocates a fresh tape per call instead (also what every recursive inner call and
  plain `gradient` use). `build_ctx` also accepts the carriers directly — `build_ctx(fcd, argcds...)`
  — or their types as one tuple, `build_ctx(Tuple{typeof(fcd),typeof.(argcds)...})`; both state
  activity through the carriers' own shadow slots rather than an `inactive=` position list.
- User-facing reverse entry points: `gradient(f, args...)` (allocates everything) and
  `gradient!`/`value_and_gradient!(ctx, fcd, argcds...)` (caller supplies both the context and each
  argument's shadow; a steady-state call allocates only the returned tuple).

## Differ skills

`Differ/.claude/skills/` has six skills describing the structure and meaning of Differ's code in
detail, matching the package split (`Contextual`/`DifferCore`/`DifferForwards`/`DifferReverse`).
They should trigger automatically when relevant, but can also be invoked directly (e.g.
`/differ-architecture`):

- **`differ-architecture`** — orientation: the package map, the `Contextual` compiler-plugin hooks,
  the `Dual`/`frule!!` and `CoDual`/`rrule!!` calling conventions, how to run tests and inspect
  dualized/reverse IR. Start here.
- **`differ-tangent-system`** — deep-dive on `DifferCore`, the tangent/fdata/rdata type system
  shared by both AD modes.
- **`differ-forward-dualization`** — deep-dive internals of `DifferForwards`' split-shadow
  dualization engine (`dualize_to_ircode` in `DifferForwards/src/forward_interp.jl`): how control
  flow is handled, the `PhiNode` forward-reference mechanism, and known `Core.Compiler.verify_ir`
  gotchas.
- **`differ-reverse-engine`** — deep-dive internals of `DifferReverse`'s two-carrier `Tape`/
  pullback engine (`DifferReverse/src/reverse_interp.jl`): the block-stack control-flow-replay
  scheme, the mutation comms scheme, recursion, and reverse-specific `verify_ir` gotchas.
- **`differ-extending-ir-support`** — playbook for adding support for a new Julia IR construct in
  `DifferForwards`, including what's already known about outstanding forward-mode gaps.
- **`differ-extending-reverse-support`** — the same playbook applied to `DifferReverse`, including
  what's already known about outstanding reverse-mode gaps. 

