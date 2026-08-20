---
name: differ-reverse-engine
description: Deep-dive reference for DifferReverse's IR-transformation engine (reverse_fwds_to_ircode / reverse_pullback_to_ircode in DifferReverse/src/reverse_interp.jl) — the CoDual/rrule!!/Tape/pullback-closure-is-the-tape design, the block-stack control-flow-replay scheme, the :fshadow/:old_primal/:old_tangent mutation comms scheme, direct self-recursion, and known verify_ir gotchas specific to reverse mode. Use this before modifying reverse_interp.jl/builtins_reverse.jl/stack.jl/cfg_ir.jl, debugging a reverse-mode bug, adding support for a new construct in DifferReverse, or investigating a Core.Compiler.verify_ir failure in reverse-mode carrier IR.
---

# DifferReverse's IR-transformation engine

Internals reference for reverse mode's compiler-plugin body, split across `DifferReverse/src/`:
`reverse_interp.jl` (the two carriers and their builders), `cfg_ir.jl` (the `ID`/`CFGBlock` working-IR
layer the pullback pass restructures), `builtins_reverse.jl` / `intrinsics_reverse.jl` /
`foreigncalls_reverse.jl` (the per-callee rule dispatch layer), `stack.jl` (the tape's storage
primitives), `codual.jl` (the `CoDual` carrier), `rrules.jl` (hand-written primitives), `reflection.jl`
(debugging entry points). Read
`differ-architecture` first for orientation and `differ-tangent-system` for the `tangent_type`/`FData`/
`RData` machinery this engine builds on; this skill assumes both and goes straight to *how reverse mode
itself works*. See `differ-forward-dualization` for the forward-mode analog — it's a genuinely
different design, not a mirror-image: forward mode tracks one split-shadow value pair per SSA
statement and replays the primal's block topology 1:1; reverse mode builds two separately-compiled
carriers around an explicit `Tape` value, and only the forwards carrier keeps the primal's topology —
the pullback carrier has its own, restructured CFG.

## The two-carrier design

Unlike Mooncake (two `OpaqueClosure`s sharing captured state via `SharedDataPairs`), reverse mode here
uses two ordinary `CodeInstance`s and one explicit `Tape` value passed between them:

- **`reverse_fwds_impl(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) -> (result::CoDual, tape::Tape)`**
  — replays the primal computation forward (ordinary value recomputation, no shadow arithmetic) and,
  wherever control flow or mutation is ambiguous, instruments it: pushes the current block number onto
  a shared `Stack{Int32}` (the block stack) and pushes whatever forward-computed values that block's
  rules will need onto that block's own "comms" `Stack`.
- **`reverse_pullback_impl(tape::Tape, seed) -> rdata_tuple`** — walks the primal's blocks in reverse
  (over a *different* CFG than the primal's, built via `cfg_ir.jl`'s `ID`/`CFGBlock` layer, since the
  pullback inserts phi-routing blocks and lowers multi-way dispatches into `GotoIfNot` chains), popping
  the block stack to learn which predecessor fired and popping each block's comms stack to recover that
  visit's forward values, accumulating rdata into per-SSA/per-argument `Ref`s.

**Pullback-seed convention for an fdata-carrying return** (ISSUES #99): the returned `CoDual`'s own
shadow is not implicitly seeded — the caller must increment it before calling the pullback, and pass
only the rdata part as the pullback's seed argument: `increment!!(tangent(ycd), fdata(seed));
pb(rdata(seed))`. For a scalar return this collapses to today's `pb(seed)` (`fdata(seed) ===
NoFData()`), which is why the DI extension's array-return path was the first to need the split. Mirrors
Mooncake's `__value_and_pullback!!` (`Mooncake.jl/src/interface.jl`).

Both are recognized as *carrier* `MethodInstance`s by `is_reverse_fwds_impl`/`is_reverse_pullback_impl`
and built by `build_contextual_ir(interp::ContextualInterpreter{Reverse}, mi)` — the same
`finishinfer!`/`optimize` hook forward mode uses (`Contextual.jl`, mode-agnostic). `Reverse` is the
plugin-owner struct (`DifferReverse.jl`) with one config bit, `nested_forward::Bool`, set only when this
interpreter is compiling a reverse carrier on behalf of an outer forward-mode dualization
(forward-over-reverse); `REVERSE_BAIL_REASONS` is a shared `IdDict{MethodInstance,String}` so a later
caller can retrieve *why* a given carrier bailed without recompiling it. `build_reverse_interp` builds
the `ContextualInterpreter{Reverse}`.

`rrule!!` (`reverse_interp.jl`) is a single generic function, one uniform convention, two kinds of
method:
- hand-written primitives (`rrules.jl`) — e.g. `rrule!!(::CoDual{typeof(sin),NoFData}, ::AbstractCtx,
  xcd::CoDual{Float64,NoFData})` returning `(CoDual(sin(x), NoFData()), SinPullback(x))`;
- the `@generated` derived fallback (`is_generated_reverse_fwds_fallback` recognizes it by exact
  signature `Tuple{typeof(rrule!!),CoDual,AbstractCtx,Vararg{CoDual}}`), which resolves/compiles
  `reverse_fwds_impl` and emits a static `:invoke` to it.

A hand rule's `ctx` slot must stay `::AbstractCtx` — never a concrete `Ctx{P}` — exactly for the reason
`CLAUDE.md`'s "Rule interfaces" section gives: dispatch specificity is decided entirely by the `fcd`/arg
slots, so the fallback (least specific) never wins over a hand rule, and `hand_reverse_rule_match`
(query against `Ctx{Nothing}`) can tell "a hand rule matched" from "only the fallback resolved."

**A hand rule's pullback can be anything** — `SinPullback`/`CosPullback` are plain immutable structs
holding just the saved primal `x`. Only the *derived* path's pullback is a `Tape`. The recursion glue
(`reverse_fwds_recursive_ci`) never inspects a pullback's internals, just threads it through opaquely.

## `Ctx`/`build_ctx`: where the tape lives

```julia
abstract type AbstractCtx end
struct Ctx{P} <: AbstractCtx
    tape::P
end
Ctx() = Ctx(nothing)
```

`Ctx{Nothing}` means "allocate a fresh tape every call" (`build_ctx(...; prealloc=false)`, every
recursive inner call to a non-hand-ruled, non-self-recursive callee's *first* resolution, and plain
`rev_gradient`). Any other `P` is a tape whose stacks are reset and reused per call — either the
top-level `build_ctx(f, argtypes)` (default `prealloc=true`) or an inner call's *recycled* tape (see
"Nested-tape recycling" below). A `Ctx` is not reentrant/thread-safe — one per task.

User-facing entry points (bottom of `reverse_interp.jl`): `rev_gradient(f, args...)` (allocates
everything — shadows via `zero_fcodual`, a fresh `Ctx()`) and the pre-allocated pair
`rev_gradient!`/`value_and_gradient!(ctx::AbstractCtx, fcd::CoDual, argcds::CoDual...)` (caller supplies
context and every argument's own shadow `CoDual`; steady state allocates only the returned tuple). Note
these are named `rev_gradient`/`rev_gradient!`, not `gradient`/`gradient!` — `CLAUDE.md`'s "Rule
interfaces" section predates the post-split rename (see git history: "gradient rename").

## `Tape{ArgsTT,CS}`

```julia
mutable struct Tape{ArgsTT<:Tuple,CS<:Tuple}
    const block_stack::Stack{Int32}
    const comms::CS
    const bufs::Vector{Any}
    const subtapes::Stack{Tape{ArgsTT,CS}}
    const count_stack::Stack{Int64}   # counted-loop trip counts, one push per loop-exit event
    args::ArgsTT   # left undef by the shorter constructor; assigned once per call
end
```

The pullback closure *is* the tape — there's no separate captured-state object. `ArgsTT` is the tuple
of the primal's `CoDual` argument types (so a `Tape` can recover which primal it belongs to, mirroring
how the forwards carrier's own `specTypes` names it). `CS` is a tuple of per-primal-block comms-stack
types: `Stack{T}` for a block with something to communicate, `SingletonStack{Tuple{}}` for a block with
nothing (a comms-free block costs no storage — mirrors Mooncake's `SharedDataPairs` singleton-type
optimization).

- `bufs::Vector{Any}` — reusable buffers for bulk primal save/restore (`_bulk_save_args`, see
  "Mutation" below), one slot per bulk-saved argument. `const`, only its contents change.
- `subtapes::Stack{Tape{ArgsTT,CS}}` — dedicated storage for a **direct self-recursive** call's own
  inner tape. `Tape` names itself as its own field's element type — ordinary recursive-struct
  self-reference (same shape as `next::Union{Nothing,Node{T}}` inside `Node{T}`), not an attempt to
  solve a fixed point for `CS`. One shared stack serves every self-recursive call site in a primal,
  pushed in fwds program order and popped in the pullback's exact reverse order — the same
  global forward/reverse ordering the rest of the engine already relies on. Always constructed (even
  for a primal with no self-recursion) — a fresh-`Tape`-construction cost, not a steady-state one.
- `count_stack::Stack{Int64}` — counted-loop trip counts (see "Counted loops" under "Control-flow
  replay" below): one `Int64` pushed per loop-*exit* event, replacing that loop's per-iteration
  block-stack pushes. A dedicated stack, not the block stack, so counts never contend with block-id
  pushes for one stack's LIFO order, and `Int64` because compression is exactly what makes >2^31
  iterations affordable. Sits before `args` so the shorter constructor can leave only `args` undef
  — which is why `args` is **field 6** (the generated carriers address tape fields by index:
  block_stack=1, comms=2, bufs=3, subtapes=4, count_stack=5, args=6).

**Why `Tape` needs a hand-written `tangent_type`.** `subtapes`'s element type is
`Tape{ArgsTT,CS}` itself, so the generic per-field tangent derivation (re-deriving `Tape`'s tangent by
re-deriving its own fields) does not terminate. `tangent_type(::Type{Tape{ArgsTT,CS}})` instead maps
type parameters directly: `Tape{_tuple_tangent_types(ArgsTT), _tuple_tangent_types(CS)}`, terminating
the same way `Tape`'s own struct definition does (the self-reference is untouched by the mapping). The
payoff: the shadow of a `Tape` *is* a `Tape`, so `zero_tangent_internal(x::Tape, ...)` can build a fresh
shadow with the same `_alloc_tape` code that builds a fresh primal one — no separate construction path.
`Stack{T}`/`SingletonStack{T}`/`CommsCell{T}` (`stack.jl`) get matching hand-written, non-recursive
`tangent_type` overrides for the identical reason, plus a "self-typed collapse" rule: when
`tangent_type(T) === NoTangent`, the shadow is `Stack{T}` itself (same `T`), not `Stack{NoTangent}` —
what makes `Tape`'s hardcoded `block_stack::Stack{Int32}` field type-check as its own shadow's field
type regardless of `Tape`'s parameters.

This self-reference is what **ISSUES #85** (fixed 2026-08-09) surfaced through: forward-over-reverse
hung because `tangent_type(Stack{Tape{X}})` resolved to `DifferCore`'s generic struct-derivation
fallback instead of this override, which then did not terminate. The cause was world age, not
dispatch — a `@generated` generator body dispatches at the generated method's `Method.primary_world`,
so `DifferReverse`'s overrides (defined at a later world than `DifferForwards`' `frule!!`) were
invisible to the forward transform. Fixed by routing every transform-time `tangent_type` query through
`Contextual.at_world`. The consequence worth knowing when touching this code: transform-time queries
must go through the funnels (`_tt`/`_fcdtype`/`rdtype`/`fdtype` here, `tt`/`dualt`/`zt` in forward
mode), never a bare `tangent_type` call. Note there is no automatic guard against a *new* self-
referential type added without its own `tangent_type` method — that still hangs. A static check was
tried and reverted: structural self-reference doesn't imply non-termination (`GlobalRef` cycles
through `Core.Binding` and derives fine, converging to `NoTangent`), and telling the convergent case
from `Tape`'s non-convergent one means computing the fixpoint inference already computes. See
ISSUES #85.

## Stack storage kinds (`stack.jl`)

| Type | Chosen when | Storage |
|---|---|---|
| `Stack{T}` | block executes >1 time, or comms tuple isn't `isbits` | growable `Vector{T}` + `position`; never deallocates |
| `SingletonStack{T}` | block's comms type is a singleton (nothing to actually store) | none — `push!`/`pop!` are no-ops, `pop!` materializes `T.instance` |
| `CommsCell{T}` | block not in any loop (`_loop_blocks`) and comms tuple `isbits` | one `val::T` field, read/write emitted directly (`setfield!`/`getfield`), no `position`, no bounds check |

`Base.push!`/`pop!`/`copyto!`/`similar`/`length`/`isassigned` are fully-qualified inside `stack.jl`
(never bare) because these bodies get inlined into hand-built carrier IR, where a bare name resolves
against *this module* and re-embeds as `GlobalRef(DifferReverse, :push!)` — an unbound/partitioned
`GlobalRef` `Core.Compiler.verify_ir` rejects in value position once re-embedded. Same hazard as the
`@noinline` barriers in `builtins_reverse.jl` (below) — this file's version of the same problem.

**Nested-tape recycling** (`_inner_ctx`/`_inner_self_ctx`, `stack.jl`): a `Stack` never deallocates, so
after a block's first execution the slot the next push lands in already holds a structurally identical
inner `Tape` from the previous call. `_inner_ctx` peeks that slot (`st.position + 1`) and hands the
callee that recycled tape via `Ctx{ConcreteTapeT}` instead of allocating fresh — what makes a
steady-state nested/recursive inner call allocation-free. `_inner_self_ctx` is the direct-self-recursion
variant, reading `Tape.subtapes` directly (no per-site multiplexing needed, since every self-recursive
call site in one primal shares that one field).

## Control-flow replay: the block stack

`emit_epilogue!`'s push precedes a block's terminator statement except when that terminator is a
`PhiNode` or a non-control-transfer, value-producing statement (e.g. a fallthrough `invoke`) that owns
its own comms item — then the push follows it instead (ISSUES #98).

`_unique_predecessor_info` (Phase D, mirrors Mooncake's `_characterise_unique_predecessor_blocks`)
computes, per primal block number:

- `is_unique_pred[b]` — is `b` the sole predecessor of every one of its successors? If so nothing
  downstream of `b` is ever ambiguous about where it came from, so `b` needs no disambiguation.
  A lone reachable exit block also counts (the sole way control leaves the function).
- `pred_is_unique_pred[b]` — does `b` need to *pop* the block stack on arrival? Current formula:
  `length(preds[b]) <= 1` — a single-predecessor block is never itself ambiguous, so its sole
  predecessor never pushes on the edge into it, and it must not pop either.

**Per-edge, not per-block (ISSUES #52, fixed, two-sided).** The push/pop granularity is decided per
*edge*, not per source block: for edge `b -> s`, `b` pushes iff `needs_push(b, s)` — i.e.
`length(preds[s]) > 1` and `(b, s)` is not a counted loop's header edge (`counted_edges`, below) —
and `s` pops under the same predicate — one push per ambiguous arrival is matched by exactly one
pop. This matters
because a `GotoIfNot` with one unambiguous arm and one ambiguous arm previously pushed unconditionally
on *both* arms (a block-granularity push always fires unless *every* successor is unambiguous),
wasting a push on the arm that's never disambiguated by it. Two coupled pieces, and a first, forwards-
only attempt at this was reverted before landing correctly (see below):

- **Pullback side** — `pred_is_unique_pred[b] = length(preds[b]) <= 1` (the old formula additionally
  required `is_unique_pred[only(preds[b])]`, a "balance" clause needed only because the forwards push
  used to be per-block).
- **Forwards side** — `_split_ambiguous_block_pushes` (`reverse_interp.jl`), a post-processing pass
  over the already-built forwards-carrier `IRCode`. It round-trips through the `ID`/`CFGBlock` layer
  (`cfg_ir.jl`): finds every `GotoIfNot`-terminated block with exactly one ambiguous successor, splices
  a new relay block onto that arm holding just the relocated push + a `goto`, and patches any
  `PhiNode` at the retargeted block whose `edges` named the original block (its real predecessor is now
  the relay). Wired in via one line in `reverse_fwds_to_ircode` right before `CC.verify_ir(ir)`.

**Why the first attempt broke gradients, and why it matters for future work here**: an isolated version
of `_split_ambiguous_block_pushes` (touching only the forwards carrier) built `verify_ir`-clean IR and
passed its own unit tests, but silently corrupted gradients for any loop iterating ≥2 times. Root cause:
`_emit_switch!`'s single-predecessor pullback path popped the block stack unconditionally "for
balance," which is exactly the balance a per-edge push breaks — the pullback kept popping on a
successor the forwards side had stopped pushing for. **The lesson, not just the historical note**: a
push/pop scheme's two sides can't be changed independently even when only one carrier's IR changes;
verify both together, and don't trust "it passed `verify_ir` and its own tests" as proof of a
control-flow-replay change's correctness — write a gradient-correctness regression across at least 2
loop iterations (N=0/1 alone won't exercise the ambiguous edge).

**`_emit_switch!`** (used both for exit routing and ordinary block-arrival routing) emits either a
plain `goto` (`skip_pop=true`, always true today for a single predecessor under the per-edge formula) or
`pop!(block_stack)` followed by a `Switch` pseudo-node comparing the popped id against each candidate
predecessor. Its `length(preds) == 1` non-`skip_pop` branch is now dead code under the current formula,
kept as a defensive no-op rather than removed.

**Counted loops** (`_counted_loops`/`CountedLoop`, beside `_collapsible_regions`): a reducible
single-latch natural loop's remaining per-iteration block-stack traffic at the *header* is fully
determined by the trip count — in reverse, the latch fired `C` times, then the preheader once — so
it compresses to one integer. Eligibility: exactly one back edge; header preds exactly
`{preheader, latch}`; exactly one exiting edge `(x, e)` toward code that can still return, from a
2-successor `GotoIfNot` block (any body block — a real `for` loop exits mid-body from the
iterate-end check, only a `while` loop exits from its header); all six characteristic blocks
reachable, `e != 1`; `x`/`e` outside any collapsible region. Two deliberate widenings, both needed
for any bounds-checked array-reading loop (`mysum`-shaped) to qualify at all: an edge into
throw-only code (`can_return` — backwards reachability from every valued return, transitive so
multi-block throw paths count whole) is **not** an exit, since a call that takes it never runs the
pullback; and `h`/`s_in` **may** sit in a collapsible region's quiet set (the optimizer routinely
puts the bounds compare right in the header, making it the diamond's entry) — the region forcing
only concerns that block's outgoing pushes, the counted scheme only `h`'s incoming edges, and
`s_in` is shape-sanity only. Three coupled pieces, all recomputed from identical inputs at both
builders:

- `_unique_predecessor_info` takes the loop's two header edges (`_counted_edges`) out of
  `needs_push`, so nothing pushes block ids on them any more.
- Forwards (`_split_ambiguous_block_pushes`, extended): a synthetic `Int64` counter
  `c = φ(preheader ⇒ 0, latch ⇒ c+1)` + increment spliced into the header (after its leading
  phis), and `push!(count_stack, c)` — the *phi* value, i.e. the back-edge count at exit — on a
  relay along the `x -> e` arm (sharing the relay with `x`'s relocated disc push when both exist;
  order between them is immaterial, separate stacks). Its phi-fixup pass runs over the *current*
  block versions, so one loop's exit relay landing on another loop's header edge re-edges that
  header's counter phi too (nested/adjacent counted loops compose; no exclusion needed).
- Pullback (`reverse_pullback_to_ircode`): one `Ref{Int64}` countdown cell per loop in the entry
  block (a `Ref`, not a phi — all pullback state routes through `Ref`s SROA scalarizes, and a
  missed scalarization costs a load/store, never correctness); the reverse arm routing into `x`
  (`counted_exit`) pops the count into the cell — that arm is where the reverse walk enters the
  loop region, re-run once per entry, which is what makes nesting work (inner loops push one count
  per outer iteration, popped LIFO); and the header's reverse block (`counted_header`) replaces its
  `_emit_switch!` pop with `emit_pred_dispatch!`'s countdown test — take the latch arm while
  `c != 0`, decrementing unconditionally (a stale final store is harmless; re-entry re-pops), the
  preheader arm at zero.

Result: a `while`-shaped loop (`whilesum`, bounds-checked `mysum_w`) drops to **zero** block-stack
traffic; a `for`-shaped loop (`loopsum`/`memloop!`/`mysum`/`mydot`) drops `2N+3 -> N+3`, and the
implied-merge scheme (next) then removes the residual N — its per-iteration pushes were the
iterate-end merge's. `sum(v)` itself recurses into `mapreduce_impl`'s pairwise
scheme, so its traffic lives on nested per-call-site tapes and the recursion machinery dominates
regardless. Tests: `test_reverse_counted_loops.jl` (eligibility on real primal IR, gradients
across N=0..50 for eligible and ineligible shapes, nested/`continue`/`break`/`@goto`/
bounds-checked-read fixtures, exact traffic through the live pipeline — the raw-IR analysis alone
can't see the regions/quiet interplay), plus the flat `memloop!` assertion in
`test_reverse_block_stack_split.jl`.

**Implied merges** (`_implied_merges`/`ImpliedMerge`, ISSUES #130): a 2-predecessor merge `m`
whose *following branch* is a pure function of which predecessor fired — the `iterate`-end diamond
every `for` loop lowers to, whose `GotoIfNot` condition is `not_int(φ(#a ⇒ true, #b ⇒ false))` or
`φ(tuple, nothing) === nothing` — keeps **no** block-stack traffic: which successor the branch
block `d` takes already identifies `m`'s predecessor, and the pullback's reverse walk knows that
direction for free (its arms into `d`'s reverse code are per-successor). `d` may be `m` itself
(unit-range loops) or sit a few blocks downstream through a single-pred/single-succ chain
(iterator-protocol lowering, `eachsum`/`zipsum`); a narrow evaluator (`_branch_cond_eval`:
φ-at-`m` leaves, chain phis, `PiNode`, `not_int`, `===` by egal's type-disjointness/singleton
rules) must produce a known, differing `Bool` under each predecessor assumption, so a genuinely
data-dependent condition disqualifies naturally. Three coupled pieces, mirroring the counted
scheme's structure but with **no forwards instrumentation at all** — the fwds carrier just stops
pushing on the two edges (`_implied_edges` joins `_counted_edges` in the shared `served_edges`
exclusion of `needs_push`, in `_unique_predecessor_info` and `_split_ambiguous_block_pushes`
alike); the pullback allocates one `Ref{Int32}` cell per merge in its entry block (`imp_ref_id`,
beside the countdown cells), each reverse arm into `d`'s reverse block stores the predecessor id
that arrival direction implies (`store_pred!`, decorating the same routing blocks `count_pop!`
does — the two compose when `d` is also a counted loop's exiting block), and `m`'s reverse
dispatch reads the cell instead of popping (`_emit_switch!`'s `prev_id` override). Correctness
window: between the arm's store and `m`'s read only that visit's own branch/chain reverse code
runs, and only arms into `d`'s reverse code store that cell. Excluded: collapsible-region blocks
(a region entry's reverse code is entered from the region merge unconditionally, erasing the arm
information) and counted headers (the dispatch kinds must stay mutually exclusive). Result: `for`
loops match `while` loops at O(1) block-stack traffic (`forsum`/`memloop!`/`mysum` flat at ~2
per call, `eachsum` flat); `diamondloop`'s data-dependent ternary merge keeps exactly one push
per iteration, and `zipsum` keeps one — zip's combined done-check merge is 3-predecessor with
only a *partial* implication (the singleton class is the per-iteration pred), a tracked
generalization (ISSUES #133). Tests: `test_reverse_implied_merges.jl`.

**Collapsible regions** (`_collapsible_regions`) extend the unique-predecessor optimization from a
single edge to a whole comms-free sub-region: the fixed diamond shape `@boundscheck` lowering produces
around every `getindex`/`setindex!` (entry `br` branching to `{merge, chk}`, `chk` a comms-free block
branching to `{throw dead-end, onward}`, `onward` reaching `merge` directly or via a linear comms-free
chain) needs no block-stack traffic at all — the pullback always routes `merge`'s reverse replay through
`br` directly, since neither arm the primal actually took changes what the checked-vs-direct edge
means downstream. Deliberately narrow pattern matching (not a general SESE/dominance analysis) — a
region that doesn't match byte-for-byte falls through to the ordinary (correct, just not free)
unique-predecessor handling.

**Comms fusion** (`resolve_host!`, `_scan_block_comms`): if `b`'s only successor is `c` and `c`'s
only predecessor is `b`, the two run the same number of times in a fixed order, so `b`'s comms items
can ride along on `c`'s stack instead of pushing their own — one push per visit instead of two.
`block_fused_refs[b]` records, per moved item, which block's popped tuple (and which slot in it) to
read from instead of `b`'s own (now-empty) stack. This is a distinct optimization from block
collapsing above (it doesn't skip block-stack traffic, just merges two blocks' pushes into one) and
runs after it, since fusion empties `nodes[b]` and would otherwise make `b` misread as a
collapsible-region interior.

Not purely a pullback-side concern (ISSUES #107): the inner-tape recycling lookup for a `:subtape`
comms item (the `_inner_ctx` call at a recursive-call site) reads the comms-item block's own stack
directly rather than through `presolve`, so if that block's only item got fused onto a successor's
stack, the lookup has to know where it landed — `reverse_fwds_to_ircode` binds `block_fused_refs`
from the scan for exactly this. `block_hoisted_refs` (the loop-invariant-hoist optimization above)
needs no equivalent handling: it only ever relocates a `:primal` item, never a `:subtape` one.

## `cfg_ir.jl`: the `ID`/`CFGBlock` working-IR layer

Forward mode's `dualize_to_ircode` never needs this — it preserves the primal's block topology 1:1, so
plain block numbers suffice. The reverse pullback pass does **not** preserve topology (it inserts
phi-routing blocks and lowers `Switch` dispatches into `GotoIfNot` chains), so block identity must
survive insertion/reordering — hence `ID` (a globally-unique `Int32` wrapper) instead of position
numbers, and `IDPhiNode`/`IDGotoNode`/`IDGotoIfNot` (`ID`-addressed analogs of the real nodes) plus
`Switch` (a pseudo multi-way-branch node, not real Julia IR — lowered to a `GotoIfNot` chain by
`lower_cfg_blocks_to_ir` before the result becomes a real `IRCode`). `CFGBlock` is immutable — build a
new one rather than editing in place. Ported from Mooncake's `src/interpreter/{ir_utils,reverse_mode}.jl`
generic `ID`/`CFGBlock` layer only, none of its AD-specific `rrule!!` logic.

Both the pullback carrier's own construction and `_split_ambiguous_block_pushes`'s forwards-carrier
post-processing round-trip through this layer: `_ircode_to_cfg_blocks` -> mutate -> `_cfg_lower_switch_statements`
-> `_cfg_remove_double_edges` (Julia's IR forbids two edges between the same block pair — a `GotoIfNot`
whose `dest` happens to be the fallthrough block becomes an unconditional `IDGotoNode`) ->
`lower_cfg_blocks_to_ir`.

## Mutation: the shadow-chain comms scheme

Array-element writes (`Base.memoryrefset!`) and mutable-struct field writes (`Core.setfield!`) share
one scheme, both in `builtins_reverse.jl`: on the forwards pass, zero the fdata slot being overwritten
and write through; on the pullback, increment the rdata accumulator with the **old** tangent's rdata,
then restore both the old primal value and old tangent in place — so nested mutations of the same
object thread correctly through the block-reversed tape (the pullback undoes writes in exactly the
reverse order the forwards pass made them).

**Invariant: a shadow-side statement or comms item is declared at `fdata_type(tangent_type(primal))`
(`ctx.fdtype(...)`/`fdtype(iworld, ...)`), never at the primal type** (ISSUES #100). A `MemoryRef`/
`.ref` access on a shadow array, a `memoryrefnew`/`memoryrefget` mirrored onto a shadow ref, and a
`:shadow_ref` comms item's declared type must all use the shadow's own type, not the primal's. The two
coincide only for a `Float64`-like element (`fdata_type(tangent_type(Float64)) === Float64`), which is
why declaring these at the primal type went unnoticed for so long: it's a silent type lie that Julia
miscompiles into an illegal instruction (`Unreachable reached`, signal 4) the moment element and shadow
element types diverge — `Vector{Int}`, `Vector{Bool}`, `Vector{ComplexF64}`, any `Vector{<:Struct}`.
Forward mode is unaffected (single split-shadow value pair, no separately-typed carrier IR to lie in).

**Companion invariant: a shadow *element* (what a `MemoryRef` into the shadow array actually points
at) is stored at `tangent_type(elt)`, not at its rdata or fdata type** (ISSUES #102). Reading it back
out is `rdata(...)`/`fdata(...)`/the `_rr_rdata`/`_rr_fdata` barriers as appropriate; folding an rdata
contribution back in is `increment_rdata!!` (`_rr_increment_rdata!!` barrier), never a bare
`increment!!` of two rdatas. `tangent_type(elt)` coincides with its own rdata type only for a
bits-scalar element (`Float64`); it coincides with its own fdata type only when the element carries no
rdata at all. `memoryrefget`'s pullback (the read side) and its fwds-carrier mirror (the shadow read
that feeds `sresolve`) both used to load/store at the rdata/fdata type directly, which is exactly the
`Vector{ComplexF64}`/`Vector{<:Struct-of-Float64s}` and split-fdata/rdata-element crashes #102 covers.
`memoryrefset!`'s pullback already got this right from the start.

New comms item kinds beyond the pre-existing `:primal`/`:subtape`/`:shadow_ref`:
- **`(:fshadow, obj_node)`** — `obj_node`'s fdata handle (`MutableTangent` or shadow `Array`),
  resolved the same way as `:shadow_ref`. Needed because the pullback carrier has no access to the
  original argument `CoDual`s — every fdata handle must arrive via the tape.
- **`(:old_primal, SSAValue(i))`** / **`(:old_tangent, SSAValue(i))`** — the field/element value a
  mutating statement overwrote, keyed by the mutating statement itself. Unlike every other comms kind
  these are *computed* by the forwards emission's own statements (not resolved from an existing node),
  so the forwards side must return them in a `saved::Dict` for the epilogue to find, or `emit_epilogue!`
  raises an internal error rather than silently mis-typing the comms tuple.

Two field/element shapes, both routed through the same comms/pullback shape:
- **pure rdata** (a scalar, or an immutable-scalar struct) — the new tangent is a fresh zero; the
  assigned value's gradient flows back purely through the pullback's rdata return.
- **fdata-carrying** (an array- or mutable-struct-valued field/element) — the new tangent *aliases* the
  assigned value's own shadow (`zero_tangent(p, f)` with `f` the value's own fdata, not `NoFData()`), so
  a later in-place accumulation reached through that field already lands in the assigned value's own
  gradient. The pullback's rdata return for it is `NoRData()` in the common all-array case.

`Core.getfield` accumulates an **immutable** struct's field rdata via the object's own rdata `Ref`
(`ref_for` + `increment_field!!`, unchanged since before mutation support existed); a **mutable**
struct's field rdata instead increments its `MutableTangent` in place
(`increment_field_rdata!`/`_rr_increment_field_rdata!`), since a mutable struct's whole tangent lives in
fdata, not rdata. A **dynamic** (non-literal) field index is supported only for a homogeneous
Tuple/NamedTuple/mutable struct whose fields share one pure-rdata tangent type
(`_bi_homog_tangent_type`) — deliberately mirrors Mooncake's `is_homogeneous_and_immutable`
restriction, not an unfinished TODO. `Core.setfield!`'s dynamic-index case is not supported at all
(Phase A only): unlike a read, the write needs a statically-known field to place the value into the
right slot type.

## `_fdata_tracked`: provenance gating

An array-element or struct-field access is only safe to route through the shadow-chain scheme when its
fdata handle is *statically traceable to a function argument* — otherwise the fwds pass would have
nothing real to communicate. `_fdata_tracked(pir, iworld, n, codualparams)` (`reverse_interp.jl`) walks
several chains, returning a `BitVector` indexed by SSA id (arguments are checked inline via
`_arg_fdata_tracked`, a separate `BitVector` of length `n`):

- **array-index chain**: `Base.getfield(x, :ref)` -> `Base.memoryrefnew(ref, i, bc)`, with an optional
  `PiNode` alias — exactly what Julia 1.13 lowers `x[i]` to.
- **general-struct chain**: `Core.getfield(x, fld)` off a tracked object whenever the result's own fdata
  is non-trivial (a nested array or mutable-struct field) — resolved uniformly via `_get_fdata_field`
  (covers `FData`-wrapped immutable structs and raw `MutableTangent`s alike). `Array`, `MemoryRef`, and
  `Memory` objects are excluded from this chain (their own shadow is a raw `.ref`-chain `MemoryRef`,
  handled by the array-index chain above, not by `_get_fdata_field`) — `MemoryRef`/`Memory` specifically
  because `fdata_type(tangent_type(Ptr{Nothing})) === Ptr{NoTangent}`, non-trivial, so
  `getfield(::MemoryRef, :ptr_or_offset)` would otherwise be marked tracked with no `_get_fdata_field`
  method to serve it (ISSUES #87).
- **`Core.PhiNode`**: tracked iff `fdtype(iworld, Ti) !== NoFData` (the phi's own inferred type carries
  fdata) and every edge value is assigned and either itself tracked **or** an inactive bare
  `Core.Argument` (`phi_inactive_edge`). A merge with one constant edge is active — any active edge
  makes it so — and normalises to its own primal-derived shadow type, so the constant arm is served by
  a zero the fwds builder hoists into the entry block, keyed `(phi, edge)`: per *edge*, since
  `sresolve` still serves every other one, and hoisted because a phi must lead its block and a
  loop-carried merge would otherwise rebuild the zero every iteration. The restriction to a bare
  `Core.Argument` is what keeps this declaration and that emission in sync by construction. This is the
  shape `Broadcast.unalias` produces for a constant array operand, so `sum(v .* w)` with either operand
  held constant works (ISSUES #117). A branch-merged or loop-carried array/
  mutable-struct binding is exactly this shape — the construct that blocked `sum(v .+ v)` before ISSUES
  #86 (the broadcast loop reads its accumulator through a phi).
- **`%new` of an immutable struct**: tracked (not a root by itself, unlike the mutable case below) iff
  `T` is concrete, `fdtype(T) !== NoFData`, and every fdata-carrying field operand is itself tracked —
  same gate shape as the `Core.tuple` arm below (ISSUES #87; the motivating case is `Base.Broadcast`'s
  own `Extruded`/`Broadcasted` structs, both immutable wrappers around a differentiable array/tuple
  field).

A local `%new` of a **mutable** struct with differentiable content, and `Core.memorynew`, remain
provenance *roots* (no ancestor needed) — the fwds pass always builds them a real shadow. `Core.tuple`
is tracked too, gated on every fdata-carrying operand's own provenance being independently tracked
(mirrors its own `builtin_rrule_comms` scope gate). Anything reachable any other way (nested in a
returned value, read out of a container by ordinary indexing) is untracked, and untracked-but-
differentiable is a real, located bail at the point of use — never silently mishandled.

**The scan runs to a least fixpoint, not a single forward pass.** `tracked` is monotone — every arm
above only ever flips an entry `false -> true`, deriving its own value from other entries via a helper
that is itself monotone — so a loop back-edge phi that reads a not-yet-computed value is handled by
just re-running the whole statement loop until nothing changes, rather than needing a separate
forward-reference pre-pass. Least fixpoint matches the scan's existing conservatism: anything left
untracked after convergence still bails at its point of use instead of being silently mishandled.

**Known caveat (ISSUES #91, open): a *self-referential* loop phi can never go tracked.**
`φ(entry => arg, loop => itself)` — a binding whose back-edge value is the phi's own SSA id — computes
`tracked[i] = arg_tracked && tracked[i]` on the phi's own arm, seeded `tracked[i] = false`; that's
`arg_tracked && false`, i.e. `false`, on every iteration of the fixpoint regardless of `arg_tracked`.
Fails safe (a located bail, not a wrong gradient) but is a real gap — a phi that genuinely *mixes*
provenance across loop iterations, not just a value that never changes, is exactly what the fixpoint
was added to handle, and this specific shape of it still isn't reachable.

## `_activity`: constant arguments

`Inactive` in a `CoDual`'s shadow slot means the caller declared that value **constant**:
`CoDual(x, Inactive())`. This is a third flavour of the carrier alongside `fcodual_type`'s fdata form
and `codual_type`'s full-tangent form. `CoDual` has no inner constructor to relax; `Dual`'s did have
to be (it enforced `tangent_type(P) == T`) — forward mode now carries `Inactive` too, on a
deliberately simpler scheme described in `differ-forward-dualization`.
`isactive(dx) = !isa(dx, Inactive)` and `@ifactive(dx, expr)` (`DifferCore/src/inactive.jl`) are the
rule-author predicates, and unlike the old `NoTangent` encoding **`isactive` is decidable from the
shadow type in every position** — see `differ-tangent-system` for why `NoTangent` could not do this
job (its fdata is `NoFData`, which is also an active `Float64`'s).

Activity is **type-level at function boundaries, a dataflow analysis inside a function**. Because the
`@generated` `rrule!!` fallback keys on the `CoDual` argument types, each activity signature gets its
own carrier, `Tape` type and `CodeInstance` for free; `Tape{ArgsTT,CS}`'s `ArgsTT` is what lets the
pullback carrier recover the same seed the fwds carrier used, so the two agree with no extra channel.
Mixed-activity aggregates are expressible because `Inactive` composes: a `Core.tuple` built from one
active and one constant operand has shadow type `Tuple{Vector{Float64},Inactive}` — concrete, one word
narrower than the primal-derived type, and with no slot to allocate a zero into. `_shadow_types(pir,
iworld, n, arg_active, codualparams)` (beside `_activity`) computes the type each SSA's shadow is
*actually* declared at, and `_shadow_type_of` answers the same for an arbitrary operand node (an
argument's is exactly its codual's fdata type). **Emission must use `ctx.sty`, never `ctx.fdtype`,
wherever it types a shadow read off an operand** — today that is `Core.tuple`'s construction,
`Core.getfield`'s two shadow reads, and the returned `CoDual`'s own type. A single forward pass, not a
fixpoint: a non-phi operand dominates its use, and a `PhiNode` is pinned to its own primal-derived
type, so no mixed type is ever carried around a loop.

Inside a function there are no intermediate `CoDual`s, so `_activity(pir, iworld, n, codualparams)`
returns a `BitVector` over SSA ids — a monotone least-fixpoint scan of exactly the same shape as
`_fdata_tracked`, and for the same reason (loop-carried phis). Its conservatism runs the *other* way,
though: it grows "may be active", so an unrecognised value-producing statement must default to
**active**. It is computed at all three sites (`_scan_block_comms` and both builders) from identical
inputs, exactly as `_fdata_tracked` already is.

`_arg_fdata_tracked` is gated on activity, which untracks the whole derivation chain off an inactive
argument — that is what routes those reads to primal replay rather than a bail, and `_fdata_tracked`
itself skips inactive statements so it can't promise a shadow the fwds pass won't build.

**The main payoff is coverage, not tape size.** An all-inactive `:call`/`:invoke` is replayed primally
*before* `_static_recursible_call` runs, so its concrete-argtype / traceable-provenance /
resolvable-callee gates never fire on code the derivative doesn't depend on — logging, `Dict`
bookkeeping, string handling. Secondarily, an inactive statement gets no rdata `Ref`, no comms item,
and no accumulate, and an inactive argument's slot in the pullback's return tuple is a `NoRData()`
literal.

**Four rules that are load-bearing, three of them learned the hard way:**

- **"Result type has no tangent space ⇒ inactive" is valid only for a pure value producer.** A generic
  call routinely returns `Nothing` while writing through an argument — `push!` lowers to
  `Base._growend_internal!`, exactly this shape — so applying the gate there marks a mutation inactive
  and turns a clean bail into a runtime `BoundsError`. Generic calls are decided by their operands
  alone. A rule-less `Core.Builtin`/intrinsic *does* keep the shortcut: without it `x === y` on active
  operands would have no rule and bail.
- **A locally allocated mutable object is an activity root**, not a function of its initialiser
  operands (`%new` of a mutable type, `Core.memorynew`) — an active value may be written into it later.
  The same roots `_fdata_tracked` treats as provenance roots.
- **`:foreigncall` is unconditionally active.** Native code can write through any pointer it is handed,
  and keeping it active is what lets the comms scan, fwds pass and pullback stay uniformly
  activity-gated without an exemption that would desynchronise a push/pop pair.
- **The packed vararg tail always keeps its rdata accumulator** (`_has_rdata_sink`'s `nfixed+1`
  exemption), even when some trailing arguments are constant — the scatter at the pullback's return
  now gates *per flat position* (`ret_inactive`, a `NoRData()` literal for a constant one) but still
  reads the one accumulator for the rest, and an all-`NoTangent` tail (`Tuple{}`, an `Int` vararg) is
  inactive by type. Per-element tail activity itself is ISSUES #116: a constant trailing argument
  gives the packed slot a mixed per-element shadow type (`Inactive` in the constant slots, via
  `_packed_tail_shadow_type`), and a literal-index read of a constant slot is inactive
  (`_inactive_arg_root`'s mixed-tuple-argument arm, consulted by `_activity`). A *dynamic* index into
  a genuinely mixed tail bails (not statically decidable which slot — the `getfield` comms rule, via
  the scan ctx's `sty`); an all-constant tail's dynamic read is fine. Passing the whole mixed-shadow
  tail to a resolved nested call bails too (`_static_recursible_call`'s `mixed_shadow` guard): the
  recursion glue declares unmasked argument coduals at the uniformly-active fdata type.

**Aliasing is a user obligation**, as in Enzyme: an inactive value must not share memory with an active
one, and that is not checkable here. The one statically detectable case — writing an active value into
an inactive container — is a located bail, never a silent zero. `Inactive` absorbs by definition
(`increment!!(Inactive(), y) === Inactive()`), but **`NoTangent` keeps no absorbing arm**: its absence
is what turns an analysis bug into a `MethodError` instead of a dropped gradient, and a mis-analysed
active value still lands on `NoTangent`, not on `Inactive`. Do not add one as a convenience.

Entry points take `inactive=(positions...)` (1-based, arguments only, not `f`): `build_ctx`,
`tape_type`, `code_reverse_fwds_ircode`, `code_reverse_pullback_ircode`. See ISSUES #112, and
#113-#118 for the follow-ups (all fixed).

**`ctx.inactive`/`_inactive_arg_root`** (all five ctx bundles: the three `:foreigncall` ones and two
builtin ones) ask a different question than `_bi_tracked`: not "can a shadow be resolved back to an
argument" but "does this node provably alias an argument the caller declared constant" — the same
transparent-view chain `_fc_ptr_origin` walks toward a raw pointer, walked here toward a bare
`Core.Argument` instead (`PiNode`, `Ptr->Ptr` bitcast, `Core.getfield(x, :ref)`,
`Base.memoryrefnew`). Never through a `PhiNode` (a phi merging an inactive edge with an active one is
genuinely active) and never through anything else (a global read, a generic call result) — only an
argument carries the no-aliasing promise. This is the gate `memmove`/`Core.tuple`/`Core.setfield!`/
`Base.memoryrefset!`'s third modes use to decide when a shadow can be zeroed instead of routed
(ISSUES #113); using `_bi_tracked` there instead would be unsound, since it's false both for a
genuinely constant value and for one that's active but merely untraceable.

## The builtin-rule dispatch layer

`builtins_reverse.jl` mirrors `intrinsics_reverse.jl`'s `Val(f)`-dispatch trick but three-sided, since
three separate passes over the same statement must agree on its shape:

```
(a) builtin_rrule_comms(::Val{F}, actual, Ti, ctx)       -> Vector{Tuple{item,type}} | false | nothing
(b) apply_builtin_rrule_fwds!(::Val{F}, actual, Ti, ctx)  -> (primal_ssa, shadow_ssa|nothing, saved::Dict) | nothing
(c) apply_builtin_rrule!(::Val{F}, actual, Ti, ctx)       -> Tuple (one entry per operand, nothing = no contribution) | nothing
```

`nothing` means "no rule registered — try something else"; `false` (only `(a)` can return it) means
"a rule *is* registered but this call is out of scope" (wrong types, untracked provenance, ...),
signaled by setting `ctx.reason[]`. Only the comms-scan function `(a)` needs the `false` case: it's the
one place scope is decided from types + static provenance alone, so by the time `(b)`/`(c)` run for the
same statement the rule is already known to apply. `ctx` is a different `NamedTuple` shape per phase —
`(optype, ssa, tracked, arg_tracked, reason)` for `(a)`, `(emit!, icall!, presolve, sresolve, optype,
tracked, ssa)` for `(b)`, `(emit!, icall, fetch_shadow, fetch_primal, fetch_saved, pb_presolve,
deref_and_zero!, optype, ssa, ref_for)` for `(c)`.

Currently covers `Core.getfield`/`Core.setfield!`, `Core.tuple`, `Core.memorynew`,
`Base.memoryrefnew`/`memoryrefget`/`memoryrefset!`. `intrinsics_reverse.jl` covers the differentiable
float intrinsics (`add/sub/neg/mul/div_float[_fast]`, `sitofp`/`uitofp` as inactive-operand,
`fpext`/`fptrunc`) via the analogous two-function `apply_intrinsic_rrule!`/`intrinsic_rrule_deps`
pair — the latter declares, per contribution (one per operand), which operand primals that
contribution's computation reads, so `_scan_block_comms` and the pullback can skip recording an
operand no *wanted* contribution ever reads (`_has_rdata_sink` decides "wanted": whether that
operand's own contribution has anywhere to route to). A linear rule like `add_float` needs none of
its operands regardless (the whole point of linearity); `mul_float`/`div_float`/`fma_float` have a
crossed dependency (e.g. `mul_float`'s `da` reads `b`, not `a`), so an inactive or literal operand
can drop the *other* operand's recording instead of its own (ISSUES #114, fixed).

**`Core.ifelse`** has a reverse rule (ISSUES #66): in scope only when both branches share the result
type's own concrete, fdata-free shape, routing the accumulated rdata branchlessly to whichever branch
the tape-recorded primal condition selected (a `(:primal, cond)` comms item), zero to the other. An
fdata-carrying result (array/mutable-struct select) is a deliberate, located bail — soundly
expressible via shadow aliasing, not implemented, since no real primal exercises it. Note
`sum(x -> x^n, v)` for non-literal integer `n` doesn't actually reach this rule at all any more:
`^(x::Union{Float32,Float64}, n::Integer)` (`rules_math.jl`) has its own hand rule now, which keeps
`^` un-inlined (a hand rule blocks inlining) so `Base.Math.pow_body`'s branchless fast-path selection
— the `Core.ifelse` call that originally motivated this rule — never gets dualized in the first
place. Same category as the `bitcast`/`reinterpret` rationale (`DifferForwards/src/intrinsics.jl:254-
259`): a bit-twiddling kernel gets a hand rule for the affected function rather than teaching the
engine to dualize the bit-level machinery generically.

## `:gc_preserve_begin`/`:gc_preserve_end`

`GC.@preserve` lowers to these two heads (Base's `Broadcast.materialize`/`copyto!` use it internally,
so this became load-bearing for broadcast support, ISSUES #86-90). The fwds carrier's main loop and its
throw-only/unreachable-block arm both root the primal operand and, when the operand is an
`SSAValue`/`Argument` with a tracked shadow, the shadow too — mirroring forward mode's treatment
(`forward_interp.jl` lines 1348-1364) — so a raw shadow address taken inside the region can't be
collected out from under it. `:gc_preserve_end` references its matching `:gc_preserve_begin` by
`SSAValue`. The pullback statement loop treats both heads as pure markers (`continue`, next to
`:boundscheck`/`:loopinfo`) — always `NoRData`, nothing to route back.

## The `:foreigncall` dispatch layer (`foreigncalls_reverse.jl`)

Mirrors `builtins_reverse.jl`'s three-sided `Val`-dispatch contract exactly, since the same three passes
over one statement (comms scan, fwds emission, pullback emission, all in `reverse_interp.jl`) must agree
on its shape:

```
foreigncall_rrule_comms(::Val{name}, fc, Ti, ctx)       -> Vector{Tuple{item,type}} | false | nothing
apply_foreigncall_rrule_fwds!(::Val{name}, fc, Ti, ctx) -> (primal_ssa, shadow|nothing, saved::Dict) | nothing
apply_foreigncall_rrule!(::Val{name}, fc, Ti, ctx)      -> Tuple | nothing
```

`fc` is `_fc_parse(s)`'s result (`nothing` for a runtime function pointer target, handled before
dispatch ever runs). Unlike `builtins_reverse.jl`, `nothing` here is **always** a bail, never "replay
primally": native code can write through any pointer it's handed, so silently doing nothing is not
sound. The mode-agnostic parsing/provenance helpers this rule builds on — `_fc_parse`, `_fc_stmt`,
`_fc_ptr_origin`, `_fc_same_stride`, `_fc_check_extent`, `_fc_copy_sig_ok`, `_FC_COPY_ATS` — live in
`DifferCore/src/shared_ir_helpers.jl`, shared unchanged with `DifferForwards/src/foreigncalls.jl` (they
already took every interpreter-specific piece through a `ctx` NamedTuple of closures).

**`memmove`/`memcpy`, the one registered target**: reverse semantics are **not** forward mode's mirrored
copy. Forward mode mirrors the copy onto the shadow (`dst_shadow = src_shadow`); reverse mode's fwds
pass **saves** the destination shadow's current `nelem` elements into a fresh `Memory{Td}` (returned via
the `saved::Dict` epilogue mechanism as an `(:old_tangent, ssa)` comms item) and then **zeroes** that
destination range — a copy's tangent starts at zero right after the copy, since nothing has contributed
to it yet. The pullback does `src[i] = increment!!(src[i], dst[i])` over the range, then restores the
saved old destination tangent — the same old-tangent-restore discipline `memoryrefset!`'s rule and
`MapBangPullback` already use, so a buffer mutated more than once still threads correctly through the
reversed tape. All loop bodies are `@noinline` helpers taking shadow `MemoryRef`s, never raw pointers —
the usual re-embedded-`GlobalRef` reason (see "`verify_ir` gotchas" below).

Three modes, decided by `ctx.inactive(src_ref)` (never `_bi_tracked` — see "`_activity`" below for
why): both buffers provenance-tracked (`_fdata_tracked`) as above; both-sides-`NoTangent`
(`copy(::Vector{Int})`, a `Bool` mask), which emits the primal call alone with no shadow work; and
destination tracked with source inactive (a constant array reaching `copy`/`.`-broadcast's
`unalias`), which skips the source's own checks and `:fshadow` item and drops the destination's
accumulated cotangent on the floor via a restore-only pullback helper (`_fc_restore!`, beside
`_fc_accum_restore!`). A mixed active-but-untracked pair still bails. The symmetric "source tracked,
destination inactive" mode stays a deliberate bail, the same shape `memoryrefset!`/`setfield!` already
make one. Deliberately deferred: offset pointers (`add_ptr`/`sub_ptr` — `_fc_ptr_origin` already ends
its walk there) and any target other than `memmove`/`memcpy` (ISSUES #94).

## Recursion

Three distinct shapes, resolved by `_static_recursible_call` (comms-scan-time static analysis, no
compilation) then `reverse_fwds_recursive_ci`/`reverse_pullback_recursive_ci` (resolve-and-compile):

- **Direct self-recursion** — the callee's own primal `MethodInstance` is *exactly* the primal
  currently being differentiated. Has a finite, closed-form `Tape` type after all: rather than solving
  `TapeT_f = Tape{...,TapeT_f}` to a fixed point, `_scan_block_comms` declares the cyclic slot with the
  bare, unparameterized `Tape` UnionAll marker (a concretely-typed comms tuple can have one element
  declared with an abstract type). `reverse_fwds_recursive_ci` then resolves the edge to a static
  `Expr(:invoke, mi, ...)` against a bare `MethodInstance` (no eager `CodeInstance` compile — `:invoke`
  accepts either) whenever the target is literally the `impl_mi` currently being compiled, which is
  exactly when codegen's `mi == ctx.linfo` self-recursion fast path fires (a direct unboxed specsig
  self-call). The pre-allocated `Ctx{<:Tape}` carrier's self-edge always targets the `Ctx{Nothing}`
  sibling instead of itself — resolved via one bounded nested compile (that sibling's own self-edge
  *is* literal identity, so it terminates). ISSUES #65/#68.

  Because the callee is the build itself, both carriers must state the recursive call's declared
  result type in **closed form** — there is no `CodeInstance` to read a `rettype` off. Both therefore
  have to derive it from the very expression the build's own `return` uses, activity substitution
  included: the pullback's self-`:invoke` and its returned tuple share one `own_RdatasT`
  (`ret_rt`, which maps an inactive argument to `NoRData`), and the fwds self-`:invoke`'s carrier
  comes from a pre-pass over `exit_blocks` running the same `ret_carrier` the `ReturnNode` arm does.
  Recomputing either independently is how ISSUES #127 arose — the pullback's half as a SIGILL, the
  fwds' half as a silently reinterpreted shadow, and neither visible to `verify_ir`, which does not
  check declared statement types. A self-recursive build whose exits disagree on their shadow type
  needs a genuine fixpoint and bails (ISSUES #128).
- **Mutual recursion** (A -> B -> A) — still unsupported; would need a tape-type pre-pass across the
  whole strongly-connected component. Bails cleanly via `interp.custom_state.in_progress`
  (`build_reverse_fwds_ir`/`build_reverse_pullback_ir`, keyed by *carrier* `MethodInstance` — not primal
  `MethodInstance`, because the `Ctx{Nothing}` fresh-tape carrier's self-edge legitimately triggers one
  bounded nested compile of its `Ctx{<:Tape}` sibling, which a primal-keyed guard would wrongly flag as
  "already in progress").
- **Argument-position callees** (`_static_recursible_call`) — a callee reached through an
  `Argument`/`SSAValue` operand (e.g. `mapreduce_impl(f, op, A, ...)`'s `f`) has no compile-time
  *value* (`_calleeval` returns `nothing`), but the recursion machinery only ever needs the callee's
  *type* (`reverse_fwds_recursive_ci` takes `ftype`, never `fval`); the one place `fval` mattered
  (building the callee's `CoDual` at emission) now falls back to `presolve`-ing the operand instead.
  Covers `sum(sin, v)`, `sum(abs2, v)`, and any closure whose only captures are non-differentiable.

A recursive call is only eligible at all when: the callee is a concrete, non-tangent-carrying value/type
(no differentiable captures); every argument type is concrete, and any argument whose tangent carries
fdata is a real array or mutable struct whose identity is provenance-traceable
(`arg_tracked`/`fdata_tracked`); and the call's own result carries no fdata. This is narrower than "a
function can't return fdata" (ISSUES #99 fixed that for an ordinary `ReturnNode`, via the tracked
shadow the fwds pass already built) — a *recursive call*'s result, consumed by more IR in the caller
rather than immediately returned, still has nowhere to route a shadow to: `reverse_fwds_recursive_ci`
builds no shadow for it, so a caller trying to use that array/struct value downstream is unsupported
regardless of #99. A hand-written `rrule!!` for the callee always takes priority over raw recursion
(checked first in `reverse_fwds_recursive_ci` via `hand_reverse_rule_match`).

**Reverse-over-forward and reverse-over-reverse are both explicitly rejected**, not silently
miscompiled: `_composition_bail_message` fires as soon as a callee's primal function type is known
(`has_hand_reverse_rule`, `resolve_reverse_primal`) — before a foreign carrier type would otherwise
reach `fcodual_type` and crash with an unrelated "Unhandled type" error. `_is_foreign_forward_carrier`/
`_forward_entry_name`/`_nested_forward_protects_frule` are coupling-point hooks, default-inert in
standalone `DifferReverse` and overridden by `DifferForwardsOverReverseExt` once both `DifferForwards`
and `DifferReverse` are loaded (`DifferReverse` cannot reference `DifferForwards`' `Dual`/`dualized_impl`
directly — no such dependency).

## `_hand_rule_ftype_candidate`: guarding `has_hand_reverse_rule` from `src_inlining_policy`

`has_hand_reverse_rule` (used by `src_inlining_policy` to keep a call from being inlined away before it
reaches recursion dispatch) is called on **every** callee Julia's optimizer considers inlining while
optimizing a primal — not just calls that survive into the final IR. Its strict check builds a probe
signature (`argcodualtys`, via `_fcdtype`/`tangent_type` on every one of the callee's argument types)
and hands it to `hand_reverse_rule_match`, which just assembles the `Tuple` and queries the method
table — fine for a call that actually survives into the primal (concrete, primal-relevant types) but
not safe to run unconditionally on any inlining candidate: `tangent_type`'s
generic struct fallback has no termination guarantee against an arbitrary self-referential type
(`Base.ImmutableDict`'s literal self-reference is a real, reachable example — its `parent` field is the
same type, and unlike `GlobalRef`'s cycle it doesn't converge, ISSUES #92), and Julia's inference
explores plenty of unrelated Base-internal callees this compiler never needs an opinion on. Broadcast's
real primal IR pulls in enough Base internals to reach exactly this: `IOContext{IOBuffer}`'s
`tangent_type` reaches `Base.ImmutableDict`, and because `tangent_type` is `@foldable`, inference
concrete-evaluates it inside its own call stack — so the non-termination surfaced as a **compile-time
`StackOverflowError`**, not a runtime one (ISSUES #90).

`_hand_rule_ftype_candidate(interp, ftype)` is the fix: a cheap pre-filter answering "could *any*
hand-written `rrule!!` method match this callee, for *some* choice of argument types?", decided from
`ftype` alone via an abstract method-table lookup (`Tuple{typeof(rrule!!), CoDual{ftype,NoFData},
AbstractCtx, Vararg{Any}}` — `CoDual{ftype,NoFData}` is already concrete, so this never touches
`tangent_type` on anything argument-related). A negative answer is always sound: the loose probe is a
superset of every concrete `tt` the strict check could ever build for this `ftype`, so if only the
generated fallback matches the loose probe, the strict check would find nothing either.
`has_hand_reverse_rule` checks this first and returns `false` immediately on failure, before it builds
`argcodualtys` (its own per-argument `tangent_type` calls) and calls `hand_reverse_rule_match`.
`reverse_fwds_recursive_ci` builds its own `argcodualtys` the same way, except a masked (inactive)
position gets `CoDual{P,Inactive}` in place of `_fcdtype(P)` — see ISSUES #115 below.

**Hazard for any future change here**: `tangent_type` is `@assume_effects :foldable` — no mutable
state, no depth counters, nothing that would falsify `:consistent`/`:effect_free`. A global recursion-
depth counter inside `tangent_type` itself was tried as a generic fix for this class of bug and
rejected/reverted for exactly that reason. The correct fix for a *specific* non-convergent
self-referential type is always its own hand-written `tangent_type` method (`Tape`/`Stack`, ISSUES
#85; `Base.ImmutableDict` still needs one, ISSUES #92) — never a change to the generic derivation's
termination behavior, and never anything that makes `tangent_type` stateful.

## `Expr(:loopinfo)` and `julia.ivdep`

Reverse mode passes `:loopinfo` (the `@simd` marker) through as a pure control marker, same treatment as
forward mode, with **one difference**: `julia.ivdep` is filtered out rather than copied through
(`filter(a -> a !== Symbol("julia.ivdep"), s.args)`, both in `reverse_fwds_to_ircode`'s main loop and its
recursive-call-argument-tuple arm). `ivdep` asserts no loop-carried memory dependence; forward mode's
shadow mirrors the primal's access pattern one-for-one so that assertion stays true there, but the
reverse carrier's own epilogue (`emit_epilogue!`, block-stack/comms pushes on every iteration) genuinely
does add loop-carried memory traffic the primal didn't have — keeping `ivdep` would be a silent
miscompile under vectorization, not just a missed optimization. `julia.simdloop` and anything else pass
through unchanged. This (plus argument-position callees, above) is what unblocks generic reverse-mode
recursion into `Base.mapreduce_impl`'s `@simd`-annotated pairwise base case (`sum`, `sum(f, ·)`,
`sum(transpose(M))`).

## `verify_ir` gotchas specific to reverse-mode carrier IR

- **Bare names re-embed as the wrong module's `GlobalRef`.** Any small helper whose body contains an
  unqualified `getfield`/`setfield!`/`push!`/`pop!`/`copyto!`/`similar`/`length`/`isassigned` risks
  `CC.ssa_inlining_pass!` inlining it straight into hand-built carrier IR, embedding e.g.
  `GlobalRef(DifferReverse, :getfield)` — an implicit `using Core`/`using Base` binding, not a directly-
  named one — which `Core.Compiler.verify_ir` rejects as unbound/partitioned in value position. Fixed
  two ways depending on the situation:
  - **`@noinline` wrapper barriers** for small `Tangent`-system accessors threaded through carrier IR:
    `_rr_get_tangent_field`, `_rr_set_tangent_field!`, `_rr_get_fdata_field`,
    `_rr_increment_field_rdata!`, `_rr_rdata`, `_rr_zero_tangent2`, `_rr_build_tangent`
    (`builtins_reverse.jl`), and the pre-existing `__pop_blk_stack!`/`__switch_case`
    (`reverse_interp.jl`) for the same reason in control-flow helpers. `@noinline` keeps each a genuine
    `:invoke`, so its internals are never re-embedded.
  - **Fully-qualified names** (`Base.push!`, `Core` GlobalRefs via `_getfieldg`/`_setfieldg`/`_ctupleg`)
    where the call site itself is what gets inlined into carrier IR (`stack.jl`'s `Base.push!`/`pop!`,
    `_bulk_save!`/`_bulk_restore!`'s `Base.copyto!`/`Base.similar`).
  - Direct-emission fast paths (`_emit_gtf!`/`_emit_stf!`/`_emit_rdata!`, `builtins_reverse.jl`) avoid
    the barrier's call overhead entirely for the common concrete-struct-field case by emitting the
    equivalent `getfield`/`setfield!` statements with qualified `Core` GlobalRefs directly, falling back
    to the `_rr_*` barrier only for a `PossiblyUninitTangent` slot or a genuine `Tangent` fdata/rdata
    split.
- **Every recursive carrier `:invoke` must stay a genuine `:invoke`, never inlined**, regardless of
  apparent cost — `_is_reverse_carrier_mi` + `CC.src_inlining_policy` block inlining any call whose
  callee is `reverse_fwds_impl`/`reverse_pullback_impl`/`rrule!!` itself, or has a hand-written
  `rrule!!` (`has_hand_reverse_rule`). Inlining one back into its caller was confirmed empirically to
  re-embed a `GlobalRef` resolved relative to the *callee's own* defining module (a `sin(x)` call
  inside a hand rule inlines as `GlobalRef(DifferReverse, :sin)`, not `GlobalRef(Base, :sin)`). A
  hand-written *pullback* has no common supertype to test this way (it's a method on the rule author's
  own type, e.g. `SinPullback`), so it's blocked at the emission site instead: the recursive
  pullback-invoke carries `CC.IR_FLAG_NOINLINE`.
- **Two fixes specific to the hand-built reflection IR** (`code_reverse_fwds_ircode`/
  `code_reverse_pullback_ircode`, `reflection.jl`), needed once mutable-struct/array-mutation rules
  started emitting more complex carrier IR: `presolve` didn't resolve a bare `GlobalRef` operand (e.g.
  `GlobalRef(Main, :nothing)` as a `ReturnNode.val`) before embedding it as a literal; and the
  `@noinline` barrier fix above. **A third instance of the same class is open** (ISSUES #95): any
  primal containing a broadcast with at least one scalar (non-array) operand —
  `sum(v .+ 1.0)`, `2.0 .* v`, a fused chain like `v .* w .+ 2.0 .* v` — makes
  `code_reverse_pullback_ircode` build IR that fails its own `verify_ir` with the identical "Unbound or
  partitioned GlobalRef not allowed in value position" message. Confirmed reflection-only: the real
  optimized carrier (what `rev_gradient`/`gradient!` actually run — check with `checkverify_prealloc`,
  `DifferReverse/test/testutils.jl`) and `code_reverse_fwds_ircode` both pass for the same primal, and
  the gradient is correct. Not yet bisected to the specific unresolved operand.
- **A statement's inferred type can be a lattice element**, not a bare `Type` — `_widen` (shared with
  `DifferCore`) must wrap it wherever it's used as a type parameter. Two call sites only became
  reachable once `Core.tuple` got a reverse rule (ISSUES #78): the `ReturnNode` handling in
  `reverse_fwds_to_ircode` (`fcodual_type(R)` on an unwidened `R`) and the pullback's exit-route seeding
  in `reverse_pullback_to_ircode` (`zero_like_rdata_type(_optype(...))`).
  ISSUES #110 moved the widening to the boundary: statement-type reads go through `_stype(stmts, i)`
  (`DifferCore`) and all six `ctx.optype` closures widen, so **a rule's `Ti` and `ctx.optype(x)` are
  always bare `Type`s**. `_optype` itself still returns the IR's exact lattice element, deliberately.
  The widening must stay consistent across `_scan_block_comms` and both builders, or they disagree
  about the tape's type.

## Debugging entry points (`reflection.jl`)

`code_reverse_fwds_ircode(f, argtypes)` / `code_reverse_pullback_ircode(f, argtypes; seedtype=Float64)`
reproduce exactly what `CC.optimize` installs for each carrier, without going through the throwing
carrier stub — the `@code_reverse_fwds_ircode f(args...)` / `@code_reverse_pullback_ircode f(args...)`
macros are the convenience wrappers. `tape_type(f, argtypes)` recovers the concrete `Tape` type both
carriers agree on; `comms_element_types(TapeT)` lists each non-singleton per-block comms stack's element
type in block order — useful for asserting on tape layout/size in tests without hardcoding a block
index (block numbering shifts with unrelated optimizer changes).

## Known gaps

- **No dynamic-dispatch equivalent.** Forward mode has `dynamic_frule(f, ff, primals, fdatas)` for a
  genuinely dynamic (`apply_generic`-style) call. Reverse mode has nothing analogous — any call whose
  callee `_static_recursible_call` can't statically resolve bails cleanly (located `ErrorException`,
  not a crash). Would need a `dynamic_rrule` pushing its returned pullback onto the block's comms stack
  as an `Any`-typed `:subtape` item, called dynamically on the pullback side.
- **`Core.Box` / abstract-field `setfield!`** — the `Core.setfield!` reverse rule's fdata guard keys on
  the field's *declared* type, which for a `Core.Box` (reassigned captured closure variable) is always
  `Any`, so it always bails. Needs a genuine runtime `fdata_type(typeof(v))` check, which only makes
  sense once dynamic dispatch (above) exists.
- **Vararg primal methods** (ISSUES #59) — the carrier-side flat<->packed helper (`_impl_argtypes`)
  already exists, but the pullback additionally needs an rdata *scatter*: it allocates one accumulator
  per flat codual and returns them flat, while a vararg primal accumulates against one packed tuple
  slot.
- **Full per-SSA shadow threading** (every value gets a real `CoDual`, Mooncake-style) instead of the
  current `_fdata_tracked` provenance-scan restriction — would remove the whole "not traceable to a
  function argument" bail family, but would force a real zero-shadow allocation for every array-valued
  intermediate that today gets none. A genuine tradeoff, not a strict improvement — explicitly deferred,
  not scheduled. ISSUES #86/#87 (phi and immutable-`%new` provenance roots) narrow this bail family —
  `.`-broadcast no longer falls into it — but don't remove it.
- **A dynamic (non-literal) index outside a loop crashes, both read and write.** `memoryrefset!`
  (ISSUES #67) and `memoryrefget` (ISSUES #93, the read-side companion, found while testing broadcast
  support) both segfault with `Unreachable reached` inside `Stack.push!` (`stack.jl:37`) for this
  specific combination. The identical access *inside* a loop, or a *literal*-index access outside a
  loop, both work in either direction — it's "dynamic index" ∩ "non-loop" that crashes, for reads and
  writes alike. Neither case investigated further yet.
- **`ctx.optype` types some operands by node shape, not resolved value** (ISSUES #64, partly fixed by
  #110) — the lattice-element half is gone (the closures widen now), but a `getfield` on a `const`
  global with a dynamic index is still typed as `GlobalRef` rather than its true (possibly mutable
  struct) type; fixing that means switching all six `optype=` closures to `_optype_w` together (comms
  scan and both builders must agree) plus teaching `builtins_reverse.jl`'s `getfield` rules that a
  non-`SSAValue`/non-`Argument` object operand is a compile-time constant.
- **ISSUES #85** (fixed) — forward-over-reverse `tangent_type` non-termination; see "`Tape{ArgsTT,CS}`" above.
  Accepted as a known limitation, not currently being chased.
- **A self-referential loop phi can't go tracked** (ISSUES #91) — see the caveat in "`_fdata_tracked`:
  provenance gating" above.
- **`Base.ImmutableDict` still has no converging `tangent_type`** (ISSUES #92) — ISSUES #90's pre-filter
  means nothing reaches it through the inlining-policy path any more, but the type itself still needs
  its own `tangent_type` method (see "`_hand_rule_ftype_candidate`" above) before anything can
  legitimately differentiate through one.
- **`code_reverse_pullback_ircode`'s reflection tool fails `verify_ir` on a scalar-operand broadcast**
  (ISSUES #95) — see the "`verify_ir` gotchas" entry above. Reflection-only; the real carrier and
  computed gradients are unaffected.
- **An fdata-carrying recursive-call result still has no shadow to route to** (ISSUES #99 fixed the
  same gap for an ordinary `ReturnNode`, not this) — see "Recursion" above.
- **ISSUES #102** (fixed) — `memoryrefget`'s pullback and fwds-carrier read side both loaded/stored the
  shadow element at its rdata/fdata type instead of its own tangent type; see the "companion invariant"
  in "Mutation: the shadow-chain comms scheme" above.

- **Activity follow-ups**: ISSUES #113 fixed the `memmove`/`Core.tuple`/`Core.setfield!`/
  `Base.memoryrefset!` third modes (`ctx.inactive`/`_inactive_arg_root`, above). ISSUES #117 (fixed):
  a `PhiNode` merging an inactive bare-`Argument` edge with an active, tracked one is tracked
  (`phi_inactive_edge` + the entry-block zero hoisted per `(phi, edge)` — the `PhiNode` arm in
  "`_fdata_tracked`: provenance gating" above), so `sum(v .* w)` works with either operand held
  constant. ISSUES #118 (fixed): `Core.tuple`'s inactive-operand slot no longer synthesises a zero at
  all — `Inactive` is a legal shadow-slot inhabitant, so a mixed aggregate's shadow carries
  `Inactive()` in the constant slots (`_shadow_types`/`ctx.sty`, above); the earlier "one allocation
  per argument shape hung off the `Tape`" idea was dropped as treating the symptom, and a shared
  absorbing sink stays rejected (`NoTangent` keeps no absorbing arm — a mis-analysed active value is a
  `MethodError`, not a dropped gradient). ISSUES #114 (fixed): intrinsic operand primal recording is now per-contribution
  (`intrinsic_rrule_deps`/`_intrinsic_needed_operands`/`_has_rdata_sink`) rather than a flat
  "positions read" set, so `mul_float`/`div_float`/`fma_float`'s crossed dependencies drop the
  correct operand (not necessarily the inactive/literal one itself) instead of always keeping both.
  ISSUES #115 (fixed): hand-written *multi-argument* rules now match an inactive argument, both at
  the top level and through a nested recursive call. Rules-side, the affected slots widen to
  `CoDual{P,<:Union{NoFData,Inactive}}` (scalar) or `CoDual{X,<:Union{X,Inactive}}` (fdata-carrying)
  — originally spelled with `NoTangent` as the constancy marker; #118 migrated the whole encoding to
  `Inactive` — plus `@ifactive`/an `isactive(dx)` loop guard; unary rules need nothing, since an
  inactive sole argument makes the whole callsite inactive and primal replay fires before dispatch.
  Engine-side, `_static_recursible_call` computes a per-operand `mask` (`_has_rdata_sink` on the
  operand, restricted to `SSAValue`/`Argument` so a literal/`GlobalRef` operand stays active) and
  threads it through `reverse_fwds_recursive_ci`/`hand_reverse_rule_match` (which now takes prebuilt
  codual types, not argtypes) and both emission loops, so a masked operand is passed as
  `CoDual{P,Inactive}` instead of tripping the fdata-provenance guard that used to bail the nested
  case outright. The same change added a pullback recursion arity/slot-type check
  (`reverse_pullback_recursive_ci`): a wrong-arity or wrong-slot-type hand pullback is now a located
  bail instead of a `getfield` error in generated IR.
  An **overwriting** rule (`mul!`/`map!`) with an inactive *destination* is a located `error` in the
  rule itself, by design (ISSUES #123): the destination's shadow is both the backward seed and the
  result's shadow, so a no-op fast path would silently zero the sources' gradients — pass a zeroed
  shadow buffer (what `DI.Cache` produces) instead of declaring the destination constant. A *nested*
  recursive call into that shape hits `reverse_fwds_recursive_ci`'s own-result-fdata check first and
  bails with a worse message than the rule's own located refusal — cosmetic, not tracked separately. ISSUES #116 (fixed):
  an inactive element in a vararg primal's packed tail is supported — per-element tail activity via
  `_packed_tail_shadow_type` + `_inactive_arg_root`'s mixed-tuple-argument arm + `ret_inactive`'s
  per-flat-position scatter gating (see "the packed vararg tail always keeps its rdata accumulator"
  above); a dynamic index into a genuinely *mixed* tail and passing the whole mixed-shadow tail to a
  nested call are the two remaining located bails.

Growable-array mutation (`push!`/`resize!`), non-bits array elements, and any `Core.Builtin` with no
registered rule remain out of scope for both modes (`differ-extending-ir-support`).

## Cross-references

- `differ-architecture` — overall design, file map, running tests.
- `differ-tangent-system` (`DifferCore`) — `tangent_type`/`FData`/`RData`/`MutableTangent` machinery
  this engine's comms/mutation scheme is built on.
- `differ-forward-dualization` — the forward-mode analog (`dualize_to_ircode`); a genuinely different
  design (split-shadow, 1:1 block topology), not a mirror image of this one.
- `differ-extending-reverse-support` — playbook for extending this engine to a new construct.
- ISSUES.md — search `## 🟢 Reverse-Mode Tape Cost`, `#52`, `#65`, `#68`, `#78`, and `#85` for the
  design history behind the block-stack scheme, self-recursion, `Core.tuple` support, and the
  forward-over-reverse limitation, respectively. Search `## 🟢 Reverse-mode` and `#86` through `#95`
  for `.`-broadcast support: `PhiNode`/immutable-`%new` provenance and the fixpoint scan (#86/#87),
  `:gc_preserve_*` (#88), the `:foreigncall` dispatch layer (#89), the `has_hand_reverse_rule`
  ftype pre-filter (#90, same self-referential-`tangent_type` family as #85), and the open follow-ups
  (#91-#95). Search `#99` for the fdata-carrying-return fix and its pullback-seed convention. Search
  `#100`-`#102` for the shadow-declared-at-primal-type crash family, the `memoryrefset!`
  `NoTangent`-element bail, and the `memoryrefget` non-collapsed-element pullback/fwds-read fix.
  Search `#105`-`#108` for `hcat`'s own fix (the recursion argument-fdata-kind restriction was the
  sole blocker, not `collect(::Generator)`) and the two latent bugs it exposed: comms fusion vs.
  `:subtape` inner-tape recycling (#107), and the immutable-`%new` pullback's bare-`Tuple`/`NamedTuple`
  rdata crash (#108). Search `## 🟢 Reverse-mode activity analysis` and `#112`-`#120` for constant
  arguments (#112), the `memmove`/`Core.tuple`/`Core.setfield!`/`Base.memoryrefset!` third modes and
  `ctx.inactive`/`_inactive_arg_root` (#113), and the follow-ups #114-#118 (all fixed:
  per-contribution intrinsic recording, hand-rule inactive matching, per-element vararg-tail
  activity, the `PhiNode` inactive edge, and `Inactive` as a shadow-slot inhabitant); #119/#120 (a
  stale-doc note and an unexplained once-per-test-run print) are the section's only open entries.
  Search `#123` for why an inactive *destination* is refused by design.
