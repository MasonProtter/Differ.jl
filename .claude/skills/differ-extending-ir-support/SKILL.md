---
name: differ-extending-ir-support
description: "Playbook for extending DifferForwards' dualization engine to support a new Julia IR construct — the methodology used to add control flow (branches/loops), exception handling (try/catch), array indexing/mutation + mutable-struct setfield!, array allocation, GC.@preserve + raw pointer loads/stores, and foreigncall/ccall + @simd's loopinfo (all implemented), plus what's known about the remaining forward-mode gaps (unregistered ccall targets, non-bits array elements, `Core._apply_iterate`). Forward-mode scoped: the sibling `differ-extending-reverse-support` skill covers the same methodology applied to DifferReverse. Use this when asked to make DifferForwards handle more Julia language features, or when dualize_to_ircode bails on something new."
---

# Extending DifferForwards' dualization engine to a new IR construct

`dualize_to_ircode` (`DifferForwards/src/forward_interp.jl`) bails — returns `nothing`, which
surfaces as a clear `ErrorException` rather than a miscompile — on any IR construct it doesn't
handle yet. This is a playbook for closing one of those gaps, distilled from actually doing it for
control flow (branches/loops). Read the `differ-forward-dualization` skill first for how the engine
currently works; this skill is about the *process* of growing it further.

## Methodology

1. **Get the real IR shape before designing anything.** Write a minimal Julia function that
   exercises the construct and dump its fully-optimized `IRCode` with
   `Base.code_ircode(f, argtypes)[1]` (or `first`). Don't design against documentation or memory
   alone — invariants here are subtle and genuinely surprising even when you think you know the
   rules. Concrete example from the control-flow work: a plain nested-`while`-loop function turned
   out to contain a block whose *only* statement is a bare `nothing` placeholder (a trivial
   fallthrough Julia's optimizer leaves between adjacent loops) — nothing in the docs predicts
   that, but it broke the transform until handled explicitly (see the empty-block gotcha in the
   `differ-forward-dualization` skill).

2. **Read `Core.Compiler.verify_ir`'s actual checks** for the construct, in
   `Compiler/src/ssair/verify.jl` (this checkout has it at `../julia/Compiler/src/ssair/verify.jl`
   relative to `Differ/`). This is the authoritative source of legality invariants — e.g. exactly
   which basic-block numbering convention a field uses, what "the top of a block" means for
   placement rules, whether forward references are legal and how dominance is checked for them.
   Design the transform to satisfy these checks from the start rather than discovering them by
   trial and error.

3. **Narrow the bail list.** Bails are `return nothing` in the main `if`/`elseif` chain of
   `dualize_to_ircode` (`DifferForwards/src/forward_interp.jl`), which the caller turns into a
   clear `ErrorException`. The live ones today are: any un-ruled `Core.Builtin` (the
   `elseif isa(f, Core.Builtin)` arm — this is where array/`memoryref` indexing currently lands),
   plus the `anything else` fallthrough. Add a dedicated branch for the construct you're
   supporting *before* those catch-alls; don't broaden a catch-all to swallow more than the specific
   node/builtin you're prepared to handle in every branch below. (Historical note: `PhiCNode`/
   `UpsilonNode`/`EnterNode` used to be rejected by a pre-scan; that pre-scan is gone now that
   try/catch is handled inline.)

4. **Decide the duplication scheme.** Ask: does this construct carry a *value* that needs both a
   primal and a shadow copy (like `PhiNode` → primal-phi + shadow-phi), or is it a pure *control
   marker* that can be copied through unchanged because it only encodes block-numbered targets
   (like `GotoNode`/`GotoIfNot`/`EnterNode.catch_dest`)? Most new constructs will be one or the
   other — figure out which before writing code.

5. **Handle forward references if the construct has them.** Loop-carried `PhiNode` values are the
   existing example — a back-edge operand can reference an SSA index not yet visited in the linear
   walk. The `pending` dict pattern (documented in `differ-forward-dualization`) is the general
   solution: don't invent a second mechanism if this one applies.

6. **Preserve or rebuild `StmtRange`/CFG bookkeeping as appropriate.** If your construct (like
   control flow) doesn't change block topology, reuse the existing `block_start_new` tracking and
   remember the empty-block backfill (see the gotchas in `differ-forward-dualization`). If it *does*
   need new blocks or edges that don't exist in the primal, that's a bigger design decision — don't
   assume the 1:1-preservation trick still applies without checking.

7. **Call `Core.Compiler.verify_ir` and let it throw.** It's already wired in unconditionally at
   the end of `dualize_to_ircode` — don't remove or weaken that. Treat any failure while developing
   as a bug in your new code, not a reason to catch-and-bail.

8. **Add tests in both dimensions**: (a) real functions exercising the construct, checked against
   an independent reference (finite differences, or a hand-computed analytic derivative) *and*
   against `Core.Compiler.verify_ir` not throwing on the raw dualized IR (via `code_dual_ircode`);
   (b) a regression test that constructs still outside scope keep bailing with a clear
   `ErrorException` rather than silently miscompiling. New tests go in `DifferForwards/test/`, wired
   into `DifferForwards/test/runtests.jl` as a `@safetestset`; run the whole suite with
   `julia +1.13 --project=DifferForwards -e 'using Pkg; Pkg.test()'`.

## Exception handling (try/catch) — implemented; kept here as a worked reference

`try`/`catch` is **supported and tested** (it was the milestone this playbook was first written
for). The IR-shape notes below are how it works today; they're a good concrete example of applying
the methodology above. Array indexing/mutation, mutable-struct `setfield!`, and array allocation are
all implemented too (see the following sections) — the current gap beyond those is non-bits/
undef-checked array element access and `Core._apply_iterate` (left behind by splatting a
runtime-length container). Growable-array mutation (`push!`/`resize!`/…) is handled, but through
hand rules on Base's six growth helpers (`rules_growable.jl`) rather than by teaching the transform
`Core.memoryrefoffset`: the transform decides the capacity/realloc branch on the primal alone, so
mirroring the length change onto a shadow with a tighter `Memory` would write past its end. Rules
let each carrier resize through its own layout instead.

`try`/`catch`/`finally` lowers to `EnterNode` (marks the start of a protected region;
`catch_dest` names the handler block, `0` meaning "no handler needed" once the optimizer proves
nothing throws; may carry an optional `scope`), `Expr(:leave, ...)` (pops one or more enter scopes,
referencing the `EnterNode`(s) by `SSAValue`), `Expr(:the_exception)` (retrieves the caught
exception inside the handler), and `Expr(:pop_exception, ...)` (unwinds the handler frame).

Values that are live going into a handler use **`UpsilonNode`/`PhiCNode`** instead of ordinary
`PhiNode`s: an `UpsilonNode` is inserted at each point a live value could reach the handler (a
def, or right before the `:enter`), and a `PhiCNode` at the top of the handler collects the
matching `UpsilonNode`s by `SSAValue` reference. The reason ordinary `PhiNode`s don't work here is
exactly the reason this is harder than branches/loops: a `PhiNode` operand's legality is checked
against *dominance from a specific predecessor edge*, but an exception can be thrown from *any*
instruction inside the protected region — there's no single predecessor edge to hang a normal phi
off of. The "does this need a primal-copy + shadow-copy" question from step 4 above applies to
`UpsilonNode`/`PhiCNode` the same way it did to `PhiNode`, and that's exactly how it was
implemented: a shadow `UpsilonNode` at each capture point (typed `tangent_type(Ti)`), feeding a
shadow `PhiCNode` in the handler, reusing the same `pending` forward-ref mechanism.

The exception object itself (`Expr(:the_exception)`) has no meaningful tangent — its shadow is the
zero tangent of its type (`zero_shadow`, i.e. `NoTangent()` / a literal `zero` / a runtime
`zero_tangent`), the same treatment non-differentiable intrinsic results get.

Also worth knowing: the optimizer already eliminates `try`/`catch` scopes it can prove unreachable,
so post-optimization IR handed to `dualize_to_ircode` may have simpler exception structure than the
source suggests — don't assume every source-level `try` shows up as a live `EnterNode`.

## Array indexing/mutation and mutable-struct `setfield!` (forward mode) — implemented

`v[i]`/`v[i]=x` used to bail: on Julia 1.13 they lower to `memoryrefnew`/`memoryrefget`/
`memoryrefset!` builtins plus `Expr(:boundscheck)` and (for ≥2-D indexing) `Core.tuple` on the live
path, which hit the "un-ruled `Core.Builtin` → `return nothing`" arm; `setfield!` (mutable-struct
field mutation) hit the same arm. The tangent *type* system already covered arrays
(`tangent_type(Array{Float64}) == Array{Float64}`, and `DifferCore/src/array_tangents.jl` has the
value ops), so the work was entirely in the engine: `Core.getfield`/`Core.setfield!`/
`Core.memoryrefnew`/`Core.memoryrefget`/`Core.memoryrefset!`/`Core.:(===)`/`Core.tuple` each got a
dualization branch in `dualize_to_ircode` (`DifferForwards/src/forward_interp.jl`), plus a new
`tangent_type(MemoryRef{P})`/`tangent_type(Memory{P})` rule (`DifferCore/src/tangents.jl`) and
registering the `bitcast` intrinsic (needed for the `UInt`-cast in bounds-check comparisons —
`DifferForwards/src/intrinsics.jl`). Mooncake's own `src/rules/memory.jl` was consulted for how the
`Memory`/`MemoryRef` layer behaves, though its approach is fused with the reverse-mode rule system
rather than the split-shadow IR transform Differ uses here.

The key insight: an array's shadow (its tangent) is a real, same-shape `Array{tangent_type(P),N}`
object, so mirroring the *identical* `memoryrefnew`/`memoryrefget`/`memoryrefset!`/`getfield(:ref)`
call onto the shadow array directly yields the correct tangent value/mutation — no `Tangent`-style
wrapper is needed, unlike a general struct's field access (`get_tangent_field`/`set_tangent_field!`).
A `setfield!` on a genuinely mutable struct is the mutation-side counterpart of the already-supported
`getfield`/`get_tangent_field` read path, via the new `set_tangent_field!` call.

**Safety note worth remembering if you touch this again**: `Dual`'s constructor only checks
`tangent_type(P) == T`, never that a user-supplied tangent array is the same *length* as its primal.
The shadow `memoryrefnew` therefore always forces its own boundscheck flag to `true` regardless of
what the primal's bounds check decided, so a mismatched-length tangent raises a catchable
`BoundsError` instead of corrupting memory via an unchecked out-of-bounds `MemoryRef`.

## Array allocation (forward mode) — implemented

Allocating a new array *inside* a differentiated function (`zeros(n)`, `similar(v)`,
`Vector{T}(undef,n)`, array comprehensions) used to bail: they all lower to the identical
**4-statement sequence** (only the length-computation prologue differs) —

```
%1 = builtin Core.memorynew(Memory{Float64}, n)::Memory{Float64}   # 2 args: literal Memory{P} type, length
%2 = builtin Core.memoryrefnew(%1)::MemoryRef{Float64}              # 1-arg "fresh whole-memory ref" form
%3 = builtin Core.tuple(n)::Tuple{Int64}                            # size tuple
%4 = %new(Vector{Float64}, %2, %3)::Vector{Float64}                 # exactly 2 fields: :ref, :size
```

— and `Core.memorynew`, a `Core.Builtin` with no dualization rule, hit the "un-ruled builtin" bail
arm. `Core.memoryrefnew`'s 1-arg form and `Core.tuple` already had working arms from the array
indexing/mutation work above; only two new pieces were needed: a `Core.memorynew` dispatch arm, and
a `T <: Array` branch in the `Expr(:new, ...)` handling (an array's `%new` used to incorrectly fall
into the generic `build_tangent` struct path). Both mirror Mooncake's reference rules
(`Mooncake.jl/src/rules/memory.jl`): the shadow allocates a same-length, uninitialized
`Memory{tangent_type(P)}` (safe because every element the primal ever *reads* was necessarily
*written* first, by an already-dualized `memoryrefset!`), and the shadow `%new` uses the shadow ref
but the *primal's own* size tuple (array shape is structural, not a differentiable quantity — the
same reasoning `Core.tuple`'s own shadow already applies, collapsing to `NoTangent()`).

**Gotcha worth remembering**: `tt` (the primal-type → tangent-type helper, `tt(T) =
tangent_type(_widen(T))` in `DifferForwards/src/forward_interp.jl`) needs its input widened before
calling `tangent_type`, because a statement whose result the primal's own const-prop narrowed (e.g.
`Core.memorynew` called with a literal length) can carry a `Core.PartialStruct`/`Core.Const` lattice
element instead of a bare `Type` in `pstmts[i][:type]` — `tangent_type` errors on those. `_widen`
(`DifferCore/src/shared_ir_helpers.jl`, `_widen(@nospecialize T) = T isa Type ? T :
CC.widenconst(T)`) is the shared fix, used by every site that pulls a type off inferred IR — not
something specific to array allocation. `apply_builtin_frule!(::Val{Core.getfield})` and
`::Val{Core.setfield!}` route their object-operand type through the same helper for the same reason.

## `GC.@preserve` and raw pointers (forward mode) — implemented

Added together because they arrive together: `GC.@preserve v unsafe_store!(pointer(v), y)`
is the whole point of the construct. Worked example of the methodology, since it shows how much of
"support construct X" is usually *not* construct X:

- **The two IR heads are the easy half.** `:gc_preserve_begin` emits one rooting statement covering
  **both** the primal and the shadow of each operand (the dualized code holds interior pointers into
  both, so rooting only the primal is a use-after-free waiting to happen); `:gc_preserve_end` just
  `presolve`s the token. Both also went into the unreachable-block chain. That's ~15 lines.
- **The other four fifths were the pointer path underneath.** `pointer(v)` reaches its address through
  `getfield(::MemoryRef, :ptr_or_offset)` + `bitcast`, and the store is a `pointerset` intrinsic — so
  the change also needed a `MemoryRef`/`Memory` branch in the `getfield` builtin rule and hand rules
  for `bitcast`/`pointerref`/`pointerset`/`add_ptr`/`sub_ptr` (`DifferForwards/src/intrinsics.jl`).
  All mirrors, on `tangent_type(Ptr{P}) === Ptr{tangent_type(P)}`.
- **New generalizable lesson: a *position* must mean the same thing on both sides.** An element index
  always does; a byte offset only does when `aligned_sizeof(P) == aligned_sizeof(tangent_type(P))`;
  and a `MemoryRef`'s data-pointer field is only an address at all in the inline-bits layout regime,
  which the primal and shadow buffers can disagree about (`Vector{Int}`'s shadow is a
  `Memory{NoTangent}`, where the field holds the ref's 0-based *index*, not an address). Each of
  those got an explicit gate; see the pointer-layout gotchas in `differ-forward-dualization`. The
  no-tangent case is no longer a bail: it hands back the `NULL_SHADOW_PTR` sentinel
  (`DifferForwards/src/intrinsics.jl`), which every pointer rule must recognise and refuse to
  dereference.
- **New plumbing worth reusing:** `intrinsic_ctx`/`builtin_ctx` now carry `reason`, so a *registered*
  rule can decline with its own explanation instead of the caller's misleading "no rule registered".
  Any future rule with preconditions should use it.
- Tests: `DifferForwards/test/test_forward_pointers.jl` (37, including one per bail reason).

Deliberately **not** included at the time: the atomic pointer intrinsics,
`unsafe_convert(Ptr{T}, ::Ref)` (routes through `:foreigncall`, and a `MutableTangent`'s layout doesn't
match its primal's anyway), and `unsafe_wrap` (Mooncake has a 3-line frule — a natural follow-up).
`Expr(:foreigncall)` was also excluded then; it is the subject of the next section.

## `Expr(:foreigncall)` (`ccall`) and `Expr(:loopinfo)` — implemented (ISSUES #62)

The most-requested gap, and a good example of the methodology's step 3 (*narrow* the bail list rather
than broadening a catch-all). Three things are worth carrying forward from it:

**1. "Compute the primal, give it a zero tangent" is not a safe default for an opaque *call*, even
though it is for an opaque *intrinsic*.** Native code can write through any pointer it is handed. A
`memmove` handled that way would leave the destination's tangent **stale** — silently, and only in the
tangent, so every primal value still looks right. So there is no `@inactive_foreigncall` analogue of
`@inactive_intrinsic` and no fallthrough: `apply_foreigncall_frule!`'s fallback returns `nothing` and
the target bails with a located reason naming the target and pointing at
`DifferForwards/src/foreigncalls.jl`. Any future target has to be understood individually before it
gets a rule.

**2. Empirically the gap was almost entirely one target.** Scanning optimized IR across ~30 typical
numeric functions, the only foreigncall on a differentiable path is `:memmove`, inside `copyto!` /
`copy` / `unsafe_copyto!` — i.e. broadcast. `:memset` looks like an obvious companion but is *provably
unreachable*: every Base caller (`base/cmem.jl`'s wrapper, from `fill!` on `UInt8`/`Int8` buffers,
`IdDict`, `String`) has `tangent_type(P) === NoTangent`, so the shadow buffer is a
`Memory{NoTangent}`. (That regime used to bail one statement earlier in the data-pointer layout gate;
it now takes the `NULL_SHADOW_PTR` path, where a `memset` would need a rule of its own — still
unwritten, still unreachable from Base.) Check reachability before writing a rule *and* its guard
predicate.

**3. The mirror's preconditions are where the work is.** `memmove`'s rule is four lines of emission
and about eighty of gating: the full ccall signature (a bare target symbol says nothing about arity,
so operands 6/7/8 could be mis-assigned); a **pointer provenance walk** (`_fc_ptr_origin`) back
through `PiNode`/Ptr→Ptr `bitcast` to the originating `getfield(::MemoryRef, :ptr_or_offset)`,
recovering the buffer element type `P`; equal primal/tangent strides keyed on that `P`; and an
emitted extent check (`_fc_check_extent`) per shadow buffer. Two details generalize:

- **The stride check keys on the *provenance* element type, deliberately the opposite of `add_ptr`'s
  gate** (which keys on the pointer's own declared `Ptr{P}` parameter). That asymmetry is what makes
  order ≥2 work — there the shadow pointer is a `Ptr{NoTangent}` (stride 0) reached by `bitcast` from
  a `MemoryRef{Float64}`, and it is the `Float64` that governs. Don't "harmonise" the two gates.
- **Emit your own length check; don't inherit one.** Julia's own `@boundscheck` guard in
  `unsafe_copyto!` is elided under the default `--check-bounds=auto`, and the unguarded mirror
  segfaults on a short destination tangent. See the `differ-forward-dualization` skill for the
  general `verify_ir` gotcha this runs into (an inline branch would split a basic block, so the
  check is emitted as a `@noinline` `:invoke` instead).

`Expr(:loopinfo)` (`@simd`'s marker) came along because `sin.(x)`'s IR contains one right after the
memmove. It's a pure control marker copied through unchanged — and the reason nothing is `presolve`d
is not "its args are only Symbols" but that `:loopinfo` isn't in the compiler's `is_relevant_expr`,
so `userefs` never traverses its operands at all.

One incidental fix was needed to reach the memmove at all: `apply_builtin_frule!(::Val{Core.tuple})`
called `fieldtype(Ti, j)` on the raw statement type, which `TypeError`s on a `Core.PartialStruct`.
That is what unlocked the array-with-scalar broadcast forms (`x .* 2.0`).

Tests: `DifferForwards/test/test_forward_foreigncall.jl` (47). Result: unmodified `.`-syntax
dualizes — single array, array-with-scalar, and two-array broadcast `x .* y` and fused chains, with
no further construct needed.

## Threading (`Threads.@threads`, 2026-08-22)

Supported by a hand `frule!!` on `Base.Threads.threading_run` (`src/rules_threads.jl`), no engine
change: a `@threads` loop leaves exactly one non-inlined `invoke` in optimized IR with the whole body
inside its closure argument, so ruling that call runs the scheduler as primal code and dualizes only
the worker. Forward mode needs no per-worker state at all — dualized IR is an ordinary
`CodeInstance` with no per-call state, so one `Dual` closure is safely invoked from every worker at
once. (Mooncake copies its rule per thread; there is nothing here to copy.)

Two traps worth knowing. `Threads.threadpoolsize()` is called from *inside* every worker body
(`default_func`'s `divrem(lenr, threadpoolsize())`) and inlines to
`cglobal(:jl_n_threads_per_pool, Ptr{Cint})` — an unregistered *intrinsic*, not a foreigncall, and it
only appears once the worker body itself is dualized, so ruling `threading_run` alone is not enough.
And `@threads :static` emits a bare `ccall(:jl_in_threaded_region, Cint, ())` in the *enclosing*
function, out of reach of any Julia-level rule; it is registered as an inactive foreigncall in
`foreigncalls.jl` (as is `jl_set_task_tid`, `@spawnat`'s pinning call).

## Tasks (`Threads.@spawn`/`@async`/`@sync`/`Task`/`fetch`, 2026-08-22)

`@spawn` expands *inline* — `Task(thunk)`, `task.sticky = false`, `_spawn_set_thrpool`, an
optional `put!` into `@sync`'s channel, `schedule` — so `Task` itself is the interception point
(the rule is also what keeps `jl_new_task` out of dualized IR). Hand rules in `rules_threads.jl`
for `Task`/`schedule`/`wait`/`fetch`/`istask*`/`_spawn_set_thrpool`/`yield`/`current_task` and
`@sync`'s `Channel`/`put!`/`take!`/`sync_end` (a `Channel` has no tangent space; moving a
differentiable value through one is refused loudly). The spawned thunk runs dualized; its `Dual`
result is parked in the task's own task-local storage and the task's *primal* result is the
primal half, so by-value `fetch` transports the tangent and an escaped task still fetches an
ordinary value. Engine changes this needed: the inactive-`%new` replay renumbers a
runtime-computed type operand (kwargs `NamedTuple`s) instead of embedding it raw, and a
`Core.typeassert` builtin rule narrows the shadow with the asserted primal (`fetch(t)::Float64`).
(An effect-bit activity shortcut was tried alongside and reverted — it silently zeroed
forward-over-reverse Hessians; see the ISSUES entry.) OhMyThreads works
downstream through StableTasks with **no extension** (`test_forward_ohmythreads.jl`); its
`tmap`/`tmap!` still bail on a heterogeneous-capture dynamic `getfield`, and `StaticScheduler` on
the `@warn` logging (`invokelatest`) in StableTasks' pinning retry loop. `@threads :greedy` and
`Threads.Atomic` remain unsupported. See ISSUES' two threading entries.

## Remaining gaps (forward mode)

Non-bits/undef-checked array element access (`Vector{Any}`/`Vector{String}`), `splice!` (its
`Core.typeassert`, plus a `Vector{Any}` default argument reverse mode cannot trace), any `ccall`
target with no registered rule (BLAS, libm, the runtime C API — see the
section above for why bailing is the right default), `Base`-library array functions (`sum`,
`map`, `copyto!`, broadcast) beyond a hand-written index loop, `Core._apply_iterate` (left behind by
splatting something whose length isn't statically known, e.g. `f(xs...)` with `xs::Vector`; a tuple
splat of known length is fully expanded by the optimizer and never reaches the engine), and
atomic/non-default `GenericMemoryRef` kinds (falls through to a generic-derivation `TypeError` rather
than a clean bail — a known sharp edge, not yet guarded against; the `MemoryRef`/`Memory` `getfield`
branch above deliberately gates on the non-atomic aliases so as not to widen that edge).

Vararg *methods* (`f(x, ys...)`) are **no longer a gap in forward mode** — see ISSUES #58 and the
"Vararg primal methods" note in the `differ-forward-dualization` skill.

**Forward mode's own recursion support (ISSUES #82)**: `Dual{P,tangent_type(P)}` never needs a
closed-form-type trick — it doesn't grow with recursion depth, so there's no fixed point to solve,
only somewhere to point the `:invoke`. `frule_split!` resolves a call's `dualized_impl` carrier
(`dual_recursive_impl_mi`) before ever calling `frule_codeinstance` (`typeinf_ext_toplevel` there is
exactly what would recurse into the cycle). Self-recursion (callee resolves to the exact `impl_mi`
currently being dualized — checked *first*, since `impl_mi` is itself already in
`dualized_impl_in_progress()`) emits a static `Expr(:invoke, impl_mi, dualized_impl, fd, duals...)`
against the bare, uncompiled `MethodInstance`, tagged `IR_FLAG_NOINLINE` (a third argument to
`emit!`) so the inliner — which processes an `:invoke`'s target regardless of `CallInfo` — doesn't
try to inline the self-call into itself. This is legal and fast for exactly this case because
`julia/src/codegen.cpp`'s `mi == ctx.linfo` self-recursion fast path emits a direct specsig
self-call with no `CodeInstance` needed; using this trick for a non-self target degrades to a boxed
`jl_invoke` resolving against the *native* method cache and silently runs `dualized_impl`'s throwing
stub instead of the derivative, so the self-edge check must run before the mutual-recursion check.
Mutual recursion (callee's carrier is mid-compile elsewhere on this task) falls back to the
pre-existing dynamic `Expr(:call, frule!!, fd, duals...)` form, breaking the cycle the same way an
ordinary synthesized `:call` already does (no `CallInfo`, so `ssa_inlining_pass!` can't inline it) —
one boxed dispatch per SCC back-edge per recursion level, everything else static. This also gives
forward-over-reverse a self-recursive primal for free (ISSUES #80): the reverse carrier's own
self-`:invoke` is just another surviving call from this resolver's point of view. A bug caught only
by actually running the reported case, not by `verify_ir` or any static check: the self-`:invoke`
must carry the callee's own dual `fd` as its first argument (`dualized_impl(dualargs::Dual...)`'s
convention), same as the dynamic form — omitting it passed `verify_ir` clean and crashed at run time
with a low-level "Unreachable reached" codegen error, an arity mismatch invisible until codegen
builds the direct specsig call.

## Cross-references

- `differ-architecture` — overall design and file map.
- `differ-forward-dualization` — how the currently-supported forward constructs are implemented, and
  the `verify_ir` gotchas you're likely to rediscover.
- `differ-tangent-system` — DifferCore's tangent/fdata/rdata type system (`tangent_type`,
  `Tangent`/`MutableTangent`, `zero_tangent`, the array-tangent helpers) that every forward-mode rule
  above builds on.
- `differ-extending-reverse-support` — the same methodology applied to DifferReverse; read that one
  instead if the construct you're adding needs a reverse-mode (pullback-side) rule.
