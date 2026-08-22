---
name: differ-extending-reverse-support
description: Playbook for extending DifferReverse's own IR-transformation engine (reverse_interp.jl, builtins_reverse.jl) to support a new Julia construct or widen an existing one — array/struct mutation, Core.tuple, recursion (direct self-recursion, argument-position callees, mutual recursion), and the still-deferred gaps (dynamic dispatch, Core.Box, per-SSA-value shadow threading). Cross-references differ-extending-ir-support for the generic 8-step methodology and differ-reverse-engine for how the engine currently works; this skill is specifically about growing DifferReverse further. Use this when asked to make reverse-mode handle more Julia constructs, when the reverse-mode transform bails on something new, or when comparing what forward mode supports against what reverse mode still doesn't.
---

# Extending DifferReverse's engine to a new construct

This is `DifferReverse`'s counterpart to `differ-extending-ir-support` (forward mode's playbook).
Read that skill first for the generic 8-step methodology (get the real IR shape, read
`Core.Compiler.verify_ir`'s actual checks, narrow the bail list, decide the duplication scheme,
handle forward references, preserve/rebuild CFG bookkeeping, call `verify_ir` and let it throw, add
tests in both dimensions) — it applies to reverse mode too, and isn't repeated here. Read
`differ-reverse-engine` for how `reverse_fwds_impl`/`reverse_pullback_impl`/`Tape`/the comms scheme
work today; this skill only covers the *process* of extending that engine further.

## What's genuinely different about reverse mode

Forward mode's `dualize_to_ircode` preserves the primal's block topology 1:1 — control flow is a
pure duplication problem, so the generic methodology's steps 4-6 (duplication scheme, forward
references, CFG bookkeeping) apply almost unchanged. Reverse mode's pullback pass does not preserve
topology: it walks the primal's blocks in reverse and has to insert phi-routing blocks and lower
multi-way dispatches into `GotoIfNot` chains (`cfg_ir.jl`'s own header comment states this
explicitly), so block *identity* has to survive insertion/reordering — which is exactly why
`cfg_ir.jl` exists at all (an `ID`/`CFGBlock` working-IR layer, ported from Mooncake's
`ir_utils`/`reverse_mode`, position-independent by design). Two carriers exist for this reason:

- `reverse_fwds_impl` replays the primal forward, block-topology-preserving like forward mode, but
  additionally instruments ambiguous control flow: it pushes the current block number onto a shared
  `Stack{Int32}` (the block stack) and pushes any forward-computed values a rule will need later onto
  a per-block "comms" `Stack` (`_scan_block_comms` decides what each block communicates, statically,
  before either carrier is emitted).
- `reverse_pullback_impl` takes the `Tape` those pushes built and walks the primal's blocks in
  reverse order using the `ID`-based CFG, popping the block stack to know which predecessor to
  route to and popping each block's comms stack to recover that visit's forward values.

Practical consequence for step 4 of the generic methodology ("does this construct carry a value that
needs primal + shadow duplication, or is it a pure control marker?"): for reverse mode you also have
to ask a *third* question — does this construct change what a block needs to communicate to its own
pullback? A pure control marker in forward mode (e.g. `GotoIfNot`) already answers two separate
questions in reverse mode: how does the fwds carrier replay it (unchanged), and does the block it
lives in need a stack push declared for it in `_scan_block_comms` (a new question forward mode never
asks). Any new construct's `_scan_block_comms` arm, fwds-emission arm, and pullback-emission arm are
three separate functions in `reverse_interp.jl`/`builtins_reverse.jl` that must agree by construction
on what a block's comms tuple contains — see the "single-source-of-truth invariant" documented at the
top of `builtins_reverse.jl` (comms declares an item, emission must resolve or save a value for it,
or `emit_epilogue!` raises an internal error rather than silently mistyping the tuple). A `Stack`
never pushes to (forwards) / pops from (pullback) unless a block's predecessor identity is genuinely
ambiguous (`_unique_predecessor_info`), so straight-line code costs zero block-stack traffic — only
branches/loops pay, and only at genuinely ambiguous joins.

Also worth internalizing before touching the engine: `Base.code_ircode`-style inspection (step 1 of
the generic methodology) has its own reverse-mode entry points — `code_reverse_fwds_ircode(f,
argtypes)` and `code_reverse_pullback_ircode(f, argtypes)` (`DifferReverse/src/reflection.jl`), the
counterparts to forward mode's `code_dual_ircode`. `tape_type`/`comms_element_types` are useful for
asserting on tape/comms shape in tests without hardcoding block numbers (which shift with any
unrelated optimizer change).

## Array indexing/mutation and mutable-struct `setfield!`

Reverse mode's array/struct-mutation story is independent of forward mode's — built on the general
`Val{f}`-dispatch builtin-rule layer in `builtins_reverse.jl` (`builtin_rrule_comms`/
`apply_builtin_rrule_fwds!`/`apply_builtin_rrule!`, mirroring `intrinsics_reverse.jl`'s pattern) and
the `_fdata_tracked` provenance scan (`reverse_interp.jl`): a `BitVector`, indexed by SSA id, of
which values have a statically-known fdata (shadow) traceable back to a function argument, through
two chains — the array-index chain (`getfield(x, :ref)` → `memoryrefnew`, with an optional `PiNode`
alias) and the general-struct chain (`Core.getfield(x, fld)` off a tracked object, or a fresh
`%new` of a mutable struct as a provenance root in its own right).

Supported:
- Read-only array indexing traceable to a function argument.
- Recursive calls (`_static_recursible_call`) into another primal whose fdata-carrying argument —
  array, mutable struct, or an immutable struct/`Tuple`/`NamedTuple` wrapping either — is likewise
  traceable to a function argument. See the Recursion section below.
- Array mutation via `Base.memoryrefset!` (`x[i] = v`).
- Mutable-struct `Core.getfield`/`Core.setfield!` (`p.x = v`).

The last two use a "shadow chain" comms scheme: three new comms-item kinds beyond the pre-existing
`:primal`/`:subtape`/`:shadow_ref` —
- `(:fshadow, obj_node)` — `obj_node`'s fdata handle (`MutableTangent`/shadow array), resolved by
  `sresolve` the same way `:shadow_ref` is (needed because `reverse_pullback_impl` has no access to
  the argument coduals — every fdata handle must arrive via the tape).
- `(:old_primal, SSAValue(i))` / `(:old_tangent, SSAValue(i))` — the field/element value a mutating
  statement overwrote, keyed by the mutating statement itself. Unlike every other comms kind these
  are computed by the fwds-emission function's own emitted statements rather than resolved from an
  existing node, so it must return them in a `saved` dict for `emit_epilogue!` to find.
  For `memoryrefset!` these are declared at `_rr_saved_slot_type(T)`: `T` for an `isbits` element,
  `Union{Nothing,T}` otherwise, since a fresh array's `Core.memorynew` leaves every slot undefined
  and the first write has nothing to read back (ISSUES #111).

Mechanically: the fwds pass zeroes the fdata slot and writes through; the pullback increments the
rdata accumulator with the *old* tangent's rdata and restores both the old primal value and old
tangent in place, so nested mutations on the same object thread correctly through the block-reversed
tape (the pullback runs blocks in the opposite order from the fwds pass, so "restore what was
overwritten" has to happen exactly once per mutation, in exact reverse order). Both rules scope-guard
on the field/element's tangent type having a nontrivial fdata component (`fdtype(...) !== NoFData`)
and bail with a located reason otherwise, never a crash — see `builtin_rrule_comms`'s guards for
`Core.getfield`/`Core.setfield!`/`Base.memoryrefset!` in `builtins_reverse.jl` for the exact current
message text (search for `"has no differentiable provenance traceable"`, `"does not support a
dynamic (non-literal) field"`, `"carries fdata"`).

`Core.getfield`'s dynamic-(non-literal-)index path is more restrictive than the literal-index path:
it's only allowed when the object is a homogeneous `Tuple`/`NamedTuple`/mutable struct whose fields
all share one pure-rdata tangent type (mirroring Mooncake's own `is_homogeneous_and_immutable`
restriction) — a heterogeneous object's dynamic-index `getfield` bails with a located reason rather
than guessing, since the field-type-per-index has no single answer to unroll to.

Still unsupported (verified against current source, not just the old draft): a recursive call's own
*result* carrying an array/struct shadow — `_static_recursible_call`'s third guard rejects any call
whose result type has non-trivial fdata, since the fwds pass has nowhere to route a result shadow
today.

**Correction to a stale claim**: an earlier draft of this material said recursion into a callee whose
argument is a genuinely mutable struct was still unsupported, then (ISSUES #105) that any
fdata-carrying argument other than an `Array`/mutable struct was unsupported. Neither is true anymore
— `_static_recursible_call`'s argument guard no longer restricts fdata *kind* at all, only
*provenance*: whatever `P` is, if its tangent carries fdata, the only requirement is that the actual
argument's identity is traceable to a function argument (`tracked_here`). fdata is the
identity-carrying half of a tangent for any `P` — for an immutable aggregate it's a value wrapper
whose leaves are the caller's own shared shadow arrays/`MutableTangent`s, so a callee accumulating
into it accumulates into the caller's real buffers exactly as for a bare `Array`; the value half
(rdata) comes back via `route!` regardless. A mutable-struct argument is the one special case needing
no rdata back from the inner call at all: its rule accumulates into the shared `MutableTangent` in
place, so the gradient is already there once the call returns.
`test_reverse_mutation_aliasing.jl`'s `"%new of a mutable struct + recursion"` testset
(`newmut_recursive_mutate`) exercises the mutable-struct case; `test_reverse_dispatch_recursion.jl`'s
"recursion into an fdata-carrying immutable argument" testset exercises a struct/`Tuple`/`NamedTuple`
wrapping a tracked array, including a case with a derived rdata field; `test_reverse_arrays.jl`'s
"hcat/vcat" testset is the motivating end-to-end case (`hcat`'s `collect(::Generator)` recursion
passes an immutable closure). If you're relying on old notes (or an LLM's memory of them) claiming a
narrower restriction, verify against `_static_recursible_call` directly before assuming a bail.

## `Core.tuple`'s reverse rule (ISSUES #78, fixed 2026-08-08)

Was one of the few builtins with no reverse rule at all — any tuple carrying a tangent bailed with
the generic "no reverse rule" message. Covers a multi-value return, `[a, b]` array literals
(`Base.vect`'s `X::Tuple` capture), and reverse-over-reverse's own `(ycd, pullback)` return.

A tuple's rdata is a bare `Tuple`, not an `RData{NamedTuple}` wrapper, so the pullback mirrors the
immutable-`%new` case minus the `NamedTuple` unwrap step. Being immutable, it needs **no comms items
at all** — the rdata accumulator is allocated generically by the pullback prologue, and a tracked
shadow tuple is only ever consumed via `sresolve` by a later `getfield`. Covers both:
- A pure-rdata tuple (every field scalar/non-differentiable).
- An fdata-carrying tuple (a field holding an array/mutable struct), gated on every fdata-carrying
  operand's provenance being independently tracked (`_bi_tracked`).

Residual limits, all declining with a located reason rather than crashing:
- A non-concrete or `Vararg`-tailed tuple type (inference couldn't pin the shape down).
- An fdata-carrying operand whose provenance `_fdata_tracked` can't trace to a function argument
  (e.g. a tuple built from a non-`const` global read).
- A *dynamic*-index `getfield` into an fdata-carrying tuple field — this is a separate, still-open
  limitation of `getfield`'s own dynamic-index rule (its homogeneous-*pure-rdata* restriction, above),
  not something `Core.tuple`'s rule can address on its own. This is why `[a, b]` array literals
  differentiate only for scalar (pure-rdata) elements today.

## Recursion

Three distinct cases, with different scope:

**Direct self-recursion** (ISSUES #65, fixed) — a recursive call whose callee resolves to the exact
primal `MethodInstance` currently being differentiated. Has a finite, closed-form `Tape` type after
all: `_scan_block_comms` declares the cyclic subtape comms slot with the bare, unparameterized `Tape`
UnionAll rather than trying to solve `TapeT_f = Tape{...,TapeT_f}` to a fixed point (a
`Tuple{Stack{Tuple{Float64,Tape}}}`-shaped comms tuple is perfectly well-defined even though one of
its own element types is the abstract `Tape`). `reverse_fwds_recursive_ci`/
`reverse_pullback_recursive_ci` resolve the recursive edge to a static `Expr(:invoke, mi, ...)`
against a bare `MethodInstance` (no `CodeInstance`, no eager compile) whenever the target is literally
the mi currently being compiled — exactly the condition under which codegen's own `mi == ctx.linfo`
self-recursion fast path fires (a direct, unboxed specsig self-call). ISSUES #68 (fixed) closed a
follow-up cost bug where that one comms slot stayed abstractly `Tape`-typed even after this fix,
costing an allocation per read; `_inner_self_ctx` (`stack.jl`) and a dedicated `Tape.subtapes` field
now give it a concrete, recycled type like every other nested call.

**Mutual recursion** (A→B→A) — still unsupported, and bails cleanly via
`interp.custom_state.in_progress` (`contextual.jl`), keyed by *carrier* mi (not primal mi — see the
long comment above `build_reverse_fwds_ir` in `reverse_interp.jl` for why: the fwds carrier alone has
two independent specializations per primal, `Ctx{Nothing}` and `Ctx{<:Tape}`, and a primal-mi-keyed
guard would wrongly flag the second's one-off nested compile of the first as a cycle). Fixing this for
real needs a tape-type pre-pass across the whole strongly-connected component, not a local fix —
that's a project of its own, not a follow-up patch.

**Argument-position callees** (`_static_recursible_call`, ISSUES #65) — a callee reached through an
`Argument`/`SSAValue` (e.g. `mapreduce_impl(f, op, A, ...)`'s `f`) has no compile-time *value*
(`_calleeval` returns `nothing`), but the recursion machinery only ever needed the callee's *type*:
`reverse_fwds_recursive_ci` takes `ftype`, never `fval`, and `fval` is used exactly once, to build the
callee's `CoDual` at emission — where `fval === nothing` now means "resolve the operand there instead"
(`presolve`). This also covers a closure whose only captures are non-differentiable (a non-singleton
`DataType` with `tangent_type(ftype) === NoTangent`), not just literal singletons like `typeof(sin)`.
Combined with reverse mode's `Expr(:loopinfo)` support (below), this is what makes `sum(sin, v)`,
`sum(abs2, v)`, and `sum(transpose(M))` all compose through the generic path, including past
`Base.pairwise_blocksize`'s self-recursive split.

**Callees with differentiable captures** (a closure calling a closure — ISSUES #134, 2026-08-22).
`_static_recursible_call` used to reject any callee whose `tangent_type` wasn't `NoTangent`. It now
admits one on exactly the terms an fdata-carrying *argument* already gets: the fdata threads through
as the caller's real shadow (provenance traceable to a function argument, same check, same reason),
and the callee's own rdata — slot 1 of the inner rdatas tuple, previously asserted `NoRData` and
discarded — routes to the callee operand via `route!` like any argument's. A captured scalar
therefore works, provided the operand has a sink to route to; if it hasn't, the call still bails.
This is what unblocked `Threads.@threads`, whose worker is exactly this shape (a default/keyword
argument splits `threadsfor_fun` into a wrapper holding a body function holding the loop's captures),
but it is a general gain, not a threading-specific one.

**Four sites have to agree on the callee's carrier type**, and this is the trap: the comms scan
(`_scan_block_comms`), `reverse_fwds_recursive_ci`'s three signatures (`tt`, `tt_self`, `tt2`, plus
the one `hand_reverse_rule_match` builds), and the fwds emission site. All now take the type
`_static_recursible_call` returns as its fifth element. A scan/emission mismatch does not fail
`verify_ir` — it surfaces much later as an `UndefRefError` out of `sresolve`, because the callee's
build was resolved with `NoFData` and so never populated `farg` for the slot the getfield chain then
reads. Same lesson as ISSUES #52's two-sided push/pop: if you change what one side declares about a
recursive edge, change every side in the same pass.

A hand-written `rrule!!` (`src/rrules.jl`) always takes priority over raw recursion into a callee,
mirroring `frules.jl`'s treatment of `frule!!` — check `hand_reverse_rule_match` before assuming a
call needs the recursion path at all.

`Expr(:loopinfo)` (`@simd`'s marker) is supported in reverse mode too (ISSUES #65) — a pure
control-marker passthrough, same as forward mode, with one deliberate difference: `julia.ivdep` is
filtered out rather than copied through, since it asserts no loop-carried memory dependence and the
reverse carrier's own per-block comms/block-stack pushes (`emit_epilogue!`) violate exactly that
(forward mode's shadow mirrors the primal's access pattern one-for-one, so keeping `ivdep` there is
sound; reverse mode's added accesses are not). This is what unblocked reverse-mode recursion into
`Base.mapreduce_impl`'s `@simd`-annotated base case — the other half of the fix that made `sum`/
reduction-style recursion work generically (see the argument-position-callee paragraph above).
Forward mode's own recursion support (ISSUES #82) is simpler and more general than reverse mode's —
it covers mutual recursion too, not just self-recursion — precisely because `Dual{P,tangent_type(P)}`
never needs a closed-form-type trick: it doesn't grow with recursion depth, so there's no fixed point
to solve. That asymmetry is inherent to the two designs (a `Tape`'s type genuinely depends on which
primal it's for), not a gap reverse mode is expected to close the same way.

## Threading (`Threads.@threads`, 2026-08-22)

Supported, via a hand `rrule!!` on `Base.Threads.threading_run` (`src/rules_threads.jl`) rather than
any engine change: a `@threads` loop leaves exactly one non-inlined `invoke` in optimized IR, with the
whole body inside its closure argument, so ruling that call runs the scheduler as primal code and
differentiates only the worker. One `Ctx` per worker (a `Tape`'s stacks are unguarded
read-modify-writes), and the workers' pullbacks replayed **sequentially** in reverse worker order —
the pullback read-modify-writes shadow slots and restores saved primal values, so a shared *read* in
the primal is a shared *write* in reverse and a parallel replay would race even on a race-free
primal. Sequential replay is order-insensitive (rdata comes back by value and folds with
`increment!!`; fdata lands in per-worker-disjoint slots), which is what makes it sound.

The contract, which nothing checks and nothing can: the parallel region's result must not depend on
how the workers interleave. Each worker's pullback replays that worker's operations in that worker's
own reverse order; the cross-worker interleaving is never recorded. A lock-protected
*non-commutative* shared update gets a wrong gradient from a primal that runs fine.

`Threads.@spawn`/`@async`/`@sync`, bare `Task`/`fetch`, `@threads :greedy` and `Threads.Atomic`
remain unsupported. See ISSUES' "`Threads.@threads` support, both modes" entry for the full picture,
including the `threadpoolsize`→`cglobal` trap that only appears once the worker body itself is
transformed.

## Three explicitly-deferred gaps

Identified during a Phase 3 audit pass, not implemented — real candidates for future work, not
just bail messages to leave alone:

- **Reverse-mode dynamic dispatch.** Forward mode has `dynamic_frule(f, ff, primals, fdatas)`
  (`DifferForwards/src/forward_interp.jl`) — a runtime dispatcher for a genuinely dynamic
  (`apply_generic`-style) call, rebuilding concrete `Dual`s from runtime values. Reverse mode has no
  equivalent; any call whose callee `_static_recursible_call` can't statically resolve bails cleanly.
  A `dynamic_rrule(f, ff, primals, fdatas)` mirroring `dynamic_frule` would need its returned pullback
  pushed onto the current block's comms stack as an `Any`-typed `:subtape` item and called dynamically
  from the pullback side — the `:subtape` machinery already has the right shape, it just lacks a
  non-statically-typed variant.
- **`Core.Box`/abstract-field `setfield!`.** The guard in the `Core.setfield!` reverse rule
  (`builtins_reverse.jl`, search for the message containing `"carries fdata"`) keys on the *declared*
  field type, which for a `Core.Box` (a reassigned captured closure variable) is always `Any`, so it
  always bails — confirmed still doing so cleanly by the "boxed captured variable" regression test in
  `test_reverse_closures_globals.jl`. Fixing this needs a genuine *runtime* check of
  `fdata_type(typeof(v))` for the assigned value, which only makes sense once dynamic dispatch (above)
  exists, since routing an unknown-until-runtime type's shadow is the same underlying problem.
- **Threading a shadow for every SSA value** (full coduals, Mooncake-style) instead of the current
  `_fdata_tracked` provenance-scan approach — would remove the whole "not traceable to a function
  argument" family of bails. Roughly free for concretely-inferred scalar code (no allocation either
  way), but would force a genuine zero-shadow allocation for *every* array-valued intermediate that
  today gets none — a real tradeoff, not a strict improvement, and the reason the provenance-scan
  approach was chosen instead of just doing this from the start.

## `verify_ir` gotchas specific to hand-built reflection IR

Two incidental fixes were needed to get `code_reverse_fwds_ircode`/`code_reverse_pullback_ircode`
past `verify_ir` once mutable-struct/array-mutation rules started emitting more complex carrier IR.
Both are general lessons, applicable to any future reverse-mode rule that calls a small helper
function — not specific to the feature that surfaced them:

1. `presolve` didn't resolve a bare `GlobalRef` operand (e.g. `GlobalRef(Main, :nothing)` as a
   `ReturnNode.val`) before embedding it as a literal argument.
2. Small `@noinline`-free helper functions with a bare, unqualified `getfield`/`setfield!` call in
   their body could get inlined by `CC.ssa_inlining_pass!` into the hand-built carrier IR, embedding a
   bare `GlobalRef(DifferReverse, :getfield)` that `verify_ir` rejects the same way it rejects any
   non-`Core`/`Base`-module `GlobalRef` in value position. Fixed by routing every builtin rule's helper
   calls through `@noinline` wrapper functions — `_rr_get_tangent_field`, `_rr_set_tangent_field!`,
   `_rr_get_fdata_field`, `_rr_increment_field_rdata!`, `_rr_rdata`, `_rr_zero_tangent2` in
   `builtins_reverse.jl` (note: **not** `_rr_zero_tangent` — an earlier draft of this material had the
   name without the trailing `2`; verify current names by grepping the file rather than trusting notes)
   — the same technique the pre-existing `__pop_blk_stack!`/`__switch_case` control-flow helpers
   already used for this exact reason. `stack.jl`'s own header comments document the identical hazard
   for `Base.push!`/`Base.pop!`/`Base.copyto!`/`Base.similar`/`Base.length`/`Base.isassigned` — a bare
   unqualified name resolving as `GlobalRef(DifferReverse, :push!)` (an implicit `using Base` binding)
   rather than a genuinely bound cross-module reference. If you add a new helper that gets inlined into
   carrier IR and calls anything from `Base`/`Core` without a `Base.`/`Core.` qualifier, expect this.

## Narrowing the bail list, reverse-mode specifics

The generic methodology's step 3 ("narrow the bail list, don't broaden a catch-all") applies with a
reverse-mode-specific wrinkle: bails live in *three* places that must all agree, not one — the comms
scan (`_scan_block_comms`, sets `reason[]`), the builtin-rule comms function
(`builtin_rrule_comms`, sets `ctx.reason[]` and returns `false`), and the top-level per-statement
dispatch in `reverse_fwds_to_ircode`/`reverse_pullback_to_ircode` (the `"unsupported statement kind"`/
`"reverse mode does not support builtin"` fallthroughs). Before adding a new arm, grep the exact
current message text in `reverse_interp.jl`/`builtins_reverse.jl` rather than trusting a summary
(including this one) — messages and line numbers drift as the file grows (currently ~3600 lines).
Known-current, verified-open bail/crash points worth knowing about before you start:

- Reverse mode still bails on any vararg primal method (ISSUES #59, open) — the carrier-side
  flat→packed mapping exists (`_impl_argtypes`), but the pullback additionally needs an rdata
  *scatter* back across the flat per-argument accumulators, which hasn't been built.
- `Core.ifelse` has a reverse rule (ISSUES #66, fixed) — routes incoming rdata to whichever operand
  the tape-recorded primal condition selected, zero to the other; declines with its own located
  reason (not this rule) for an fdata-carrying result. A `pow_body`-class bit-twiddling kernel
  (`Base.Math.pow_body`, reached from `^` with a non-literal integer exponent) gets a hand rule
  instead of relying on this — see `rules_math.jl`'s `^(x::Union{Float32,Float64}, n::Integer)`,
  which keeps `^` un-inlined so `pow_body` never gets dualized, matching the forward-mode
  `bitcast`/`reinterpret` rationale (`DifferForwards/src/intrinsics.jl:254-259`).
- `memoryrefset!` with a dynamic index *outside* a loop crashes rather than bailing (ISSUES #67,
  open, unreproduced-as-fixed) — `x[i] = a` for a genuine runtime `i` segfaults with "Unreachable
  reached" inside `Stack`/`CommsCell` when the write isn't inside a loop; the identical write inside a
  loop, or a literal-index write outside a loop, both work. Not investigated further; a real crash to
  be aware of if you're touching array-mutation comms, not just a documented bail.

## Cross-references

- `differ-architecture` — overall design and file map.
- `differ-extending-ir-support` — forward mode's sibling playbook; the generic 8-step methodology
  lives there.
- `differ-reverse-engine` — deep dive on how `DifferReverse` currently works (`Tape`, the comms
  scheme, the two-carrier design) — read this before this skill if you haven't touched
  `reverse_interp.jl` before.
- `differ-tangent-system` — the `tangent_type`/`fdata_type`/`rdata_type`/`MutableTangent` system both
  modes build on (`DifferCore`).
