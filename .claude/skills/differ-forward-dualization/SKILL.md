---
name: differ-forward-dualization
description: Deep-dive reference for DifferForwards' split-shadow IRCode dualization engine (dualize_to_ircode in DifferForwards/src/forward_interp.jl) — how primal/shadow SSA values are tracked, how control flow is handled by preserving block topology 1:1, the PhiNode forward-reference (pending-patch) mechanism, and known Core.Compiler.verify_ir gotchas that have already cost real debugging time. Use this before modifying dualize_to_ircode, debugging a forward-mode dualization bug, adding support for a new IR construct in DifferForwards, or investigating any Core.Compiler.verify_ir / IR-shape failure on the forward-mode path.
---

# DifferForwards' IRCode dualization engine

This is the internals reference for `dualize_to_ircode` in `DifferForwards/src/forward_interp.jl` —
the function that turns a primal method's fully-optimized `IRCode` into a dualized one. Read the
`differ-architecture` skill first if you haven't already; this one assumes you know *why* the
transform exists and jumps straight into *how* it works. For reverse mode's analogous engine, see
`differ-reverse-engine`; for the tangent type system both modes build on, see
`differ-tangent-system`.

**Package layout.** `dualize_to_ircode` and everything else described here lives in
`DifferForwards/src/forward_interp.jl`. It's spliced into Julia's typeinf pipeline through
`Contextual/src/Contextual.jl`'s generic `ContextualInterpreter{T,S}` (a plugin-based
`AbstractInterpreter`: `owner::T` is the plugin identity used as the `cache_owner` partition key,
`custom_state::S` is whatever extra bookkeeping the owner wants). `Contextual.jl`'s
`finishinfer!`/`optimize` overrides call the plugin hook `build_contextual_ir(interp, mi)` to get a
transformed `IRCode`, or `nothing` to leave `mi` to the ordinary pipeline. `DifferForwards` supplies
`Forward` (a plain singleton struct, `custom_state` always `nothing`) as its owner and implements
`build_contextual_ir(interp::ContextualInterpreter{Forward}, mi::MethodInstance)` in
`forward_interp.jl` itself — there is no separate `contextual.jl` file in this package; that generic
machinery is `Contextual.jl`'s job, and this skill only needs to point at it, not re-derive it (see
`differ-architecture` for the full plugin-mechanism writeup). Tangent-system primitives
(`tangent_type`, `Tangent`/`MutableTangent`, `zero_tangent`, `build_tangent`, `get_tangent_field`,
`set_tangent_field!`, …) come from `DifferCore`. A handful of IR helpers shared with reverse mode
(`_calleeval`, `_globalref_val`/`_globalref_isconst`, `_optype`/`_optype_w`, `_stmt_str`,
`_bi_literal_index`, `_bi_homog_tangent_type`, `_tangent_field_slot`, `_widen`, and the `getfield`/
`setfield!`/`tuple` `GlobalRef` constants `_getfieldg`/`_setfieldg`/`_ctupleg`) now live in
`DifferCore/src/shared_ir_helpers.jl` rather than being private to this file.

## The split-shadow scheme

For every primal SSA statement `i`, the engine computes **two** new values: `primal[i]` (the
reconstructed primal computation) and `shadow[i]` (the parallel tangent computation). At a
`ReturnNode`, the two are packed into one `Dual`. When the returned value's type `R` is **concrete**,
this is an allocation-free `%new(Dual{R, tangent_type(R)}, …)` (the test suite guards this fast path
directly — do not turn it into a dynamic `Dual(...)` call). When `R` is **non-concrete** (a
dynamic-dispatch result, e.g. `Any`, or a `Union`), a `%new` would freeze the over-wide declared type
into the value (`Dual{Any,Any}(9.0, 6.0)` — boxed, and un-composable back into `frule!!`); instead the
`Dual` constructor is called *dynamically* (`dyn_dual!`) so the runtime infers the concrete leaf type,
annotated with the abstract `dual_type(R)` (a genuine supertype of every runtime leaf — sound for
`Any` and `Union` alike, unlike the invariant `Dual{Any,Any}`). Allocation-freedom is moot there: a
non-concrete result means the function already dynamic-dispatched.

Two small resolver closures do all the SSA-reference translation:
- `presolve(x)` — the *new* primal value corresponding to an *old* operand `x` (an `SSAValue`
  looks up `primal[x.id]`, an `Argument` looks up `parg[x.n]`, anything else passes through as a
  literal).
- `tresolve(x)` — the tangent counterpart (`shadow[x.id]` / `targ[x.n]` / `const_tangent(x)` for a
  genuine constant).
- `optype(x)` — the operand's declared *primal* type (`argty[x.n]` for an `Argument`, otherwise
  `_optype(pir, x)`). Always use this, never `_optype(pir, …)` directly, for an operand that could be
  an `Argument`: `pir.argtypes` holds *lattice elements* (`Core.Const(f)` for a singleton function
  slot, `Core.Const(())` for an empty vararg slot), and the results are consumed as genuine **type
  parameters** (`Dual{P,tt(P)}`, `Tuple{ptys...}`, `dual_type(R)`, `Pobj <: Tuple` in the builtin
  rules) — each a hard `TypeError`/`MethodError` on a non-`Type`, not a graceful bail. `argty` also
  carries the *reconstructed* tuple type for a vararg primal's packed slot (below). `_optype` itself
  stays lattice-faithful because both modes share it (`DifferCore/src/shared_ir_helpers.jl`);
  `_optype_w(pir, world, x)`, next to it, is the wrapper that answers with the type of the *value* an
  operand denotes — a `const` `GlobalRef` resolved at `world` (`Any` otherwise), a `QuoteNode`'s value
  type, everything else widened to a bare `Type`. Reverse mode's `_static_recursible_call` uses it.

Function arguments are unpacked once up front into `parg`/`targ`/`argty` (via `getfield` on the
incoming `Dual`s), so `presolve`/`tresolve`/`optype` can treat `Argument`s uniformly with
`SSAValue`s.

**Vararg primal methods.** A vararg method (`f(x, ys...)`) has `m.nargs` argument slots in its
optimized IR, the last holding its varargs **already packed into a tuple** (read as
`getfield(_va, j)`), while `frule!!` is always called **flat**. `_build_dual_ir` therefore passes
`primal_nfixed = Int(m.nargs) - 1` (counts `#self#`, excludes the vararg slot), and the prologue
re-packs dual args `nfixed+1..n` into a `Core.tuple` at slot `nfixed+1` — after which the rest of the
transform sees exactly the shape `pir` was compiled with and needs no vararg awareness. Two traps:

- The reconstructed tuple type is compared against `CC.widenconst(pir.argtypes[nfixed+1])` with `<:`,
  not `===`: the reconstruction is legitimately *sharper* when an element is `Type`-valued
  (`Core.Const((Float64,))` widens to `Tuple{DataType}` where we rebuild `Tuple{Type{Float64}}`).
  A mismatch is a graceful `reason[]` bail, not an `@assert` — the shape is user-driven, and an
  `AssertionError` out of the `@generated frule!!` body aborts compilation instead of producing an
  `error_ircode` carrier that can report why.
- **`tangent_type` collapses an all-`NoTangent` tuple to plain `NoTangent`**
  (`tangent_type(Tuple{Int,Int}) === NoTangent`, likewise `Tuple{}` — the empty-vararg case), and
  `Dual{P,T}` requires `T == tangent_type(P)`. So a collapsed vararg slot's shadow must be the
  **literal `NoTangent()`**; an emitted `Core.tuple(NoTangent(), NoTangent())` is a
  `Tuple{NoTangent,NoTangent}` and `TypeError`s at the `%new(Dual{P,NoTangent}, …)` `frule_split!`
  builds when the whole tuple reaches a live call. Same rule as `Core.tuple`'s own shadow.

**Types are read directly off the primal IR, never guessed.** The primal statement keeps its
declared type `Ti` (`pstmts[i][:type]`); the **shadow statement is typed `tangent_type(Ti)`** (the
local helper `tt(T) = tangent_type(T)`). For scalars this is a no-op (`tangent_type(Float64) ==
Float64`), so scalar shadows look exactly as they did under the old same-typed scheme; for a struct
it's a `Tangent`/`MutableTangent`, for a tuple a per-field tangent tuple, and for a `Dual` carrier
(higher order) the `Dual` itself. `_optype`/`_valtype` pull primal types off SSA values/arguments/
globals the same way. The result is a fully-typed `IRCode` whose return type `finishinfer!` reads
straight off via `compute_ir_rettype` — no re-inference step to get wrong.

## Per-construct handling (the big `if`/`elseif` chain)

| Primal statement | What happens |
|---|---|
| `ReturnNode` | Pack `presolve(val)`/`tresolve(val)` into a `Dual`, wrapped in a new `ReturnNode`. Concrete `R`: allocation-free `%new(Dual{R,tt(R)}, …)` (`dual!`). Non-concrete `R` (e.g. `Any`): a dynamic `Dual(p,t)` call (`dyn_dual!`) so the runtime value is a concrete leaf, annotated `dual_type(R)`. |
| `PiNode` | Pure alias: `primal[i] = presolve(val)`, no new instruction emitted. |
| `Expr(:new, T, args...)` | Primal is a `%new(T, presolved…)`. Shadow **dispatches on `T`**: `T<:Dual` → same-typed `%new(T, …)` (a `Dual` is its own tangent; non-diff *singleton* fields carry the primal value); a foreign self-similar-shadow type (see below) → per-field `%new` via the `_foreign_selfsim_shadow_field` hook; `T<:Tuple`/`NamedTuple` → `%new(tangent_type(T), …)` with `NoTangent()` in non-diff slots; `T<:Array` → `%new(tangent_type(T), tresolved-ref, presolved-size)` (array construction, step 4/4 of the allocation sequence — `:ref` is differentiable, `:size` is structural and taken from the primal); a general struct → build the backing `NamedTuple` and wrap it directly as `%new(NT, tresolved…)` then `%new(Tangent{NT}/MutableTangent{NT}, …)` — **no `build_tangent` call** (that would be a `CallInfo`-less `:call` — see perf gotcha below — and can't be re-dualized at higher order; the direct `%new`s can). Falls back to `build_tangent(T, tresolved…)` only when a slot is `PossiblyUninitTangent` or `%new` supplies fewer args than the struct has fields. (If `tangent_type(T)==NoTangent`, the shadow is just `NoTangent()`.) |
| any intrinsic call | `apply_intrinsic_frule!(Val(f), actual, Ti, ctx)` (`DifferForwards/src/intrinsics.jl`) dispatches straight to a per-intrinsic method that emits the primal + shadow IR *directly* into the instruction stream — no `Dual` boxing, no `frule!!` dispatch, no `CodeInstance` resolution (that machinery is reserved for genuinely surviving calls like `sin`/`cos`; every intrinsic gets this cheaper path since arithmetic is far more common). `Val(f)` dispatch works because an intrinsic value is a valid type parameter even though every intrinsic shares the single type `Core.IntrinsicFunction`. `ctx` bundles the closures the rule needs: `opf`/`emit!` to emit IR, `presolve`/`tresolve` to resolve an operand, `optype`/`tt` for primal-type introspection, `zero_shadow` for a computed non-diff result's zero tangent, and `reason` (the `Ref{String}`) for a *registered* rule that nonetheless declines — the pointer rules do, and without setting it the caller would report the flatly wrong "no rule registered". Set it unlocated; the caller appends `at %i: …`. **Differentiable** intrinsics (`add_float`/`sub_float`/`neg_float`/`mul_float`/`div_float`, `_fast` variants, plus `sqrt_llvm`/`abs_float`/`max_float`/`min_float`/`fma_float`/`muladd_float`/`copysign_float`/`fpext`/`fptrunc`) have hand-written rules (product/quotient rule for `mul`/`div`; `max`/`min` use a branchless `Core.ifelse` select — see below); **non-differentiable** ones (comparisons, integer/bit ops, rounding, int↔float conversions) are registered via `@inactive_intrinsic`, whose generic body computes the primal and gives it a zero tangent via `zero_shadow`. Handling is **explicit**: the fallback method returns `nothing`, so an *unregistered* intrinsic bails gracefully with a located reason (`"unsupported intrinsic \`atomic_pointerref\` at %i: …"`) instead of a silent wrong zero. |
| `Core.getfield` | Primal is `getfield` (extra trailing operands — an atomic ordering and/or boundscheck bool — forwarded verbatim). Shadow dispatches on the object's primal type: `Dual`/`Tuple`/`NamedTuple`/`Array` → `getfield` on the (same-shape) shadow (an array's shadow is a real same-shape `Array{tangent_type(P),N}`, so its own `:ref` field mirrors directly); a foreign self-similar-shadow type (e.g. another AD-mode package's own bookkeeping struct — see below) → the same per-field hook the `:new` case uses; a general struct → read the field out of the `Tangent`/`MutableTangent`'s `fields` `NamedTuple` **directly**, as two builtin `getfield`s (`getfield(shadow, :fields)` then `getfield(_, i)` with `i` the literal field index) — **not** a `get_tangent_field` call (that would be a `CallInfo`-less `:call` — see perf gotcha below). Falls back to `get_tangent_field` (emitted as a static `:invoke` via `emit_invoke!`) for a dynamic index or a `PossiblyUninitTangent` slot. (`NoTangent()` if the field's `tangent_type` is `NoTangent` — this is what makes an array's `:size` field exit early rather than reaching the `Array` case.) |
| `Core.setfield!` | The mutation-side counterpart of `getfield` above: primal is `setfield!` (mutates in place); shadow rebuilds the `MutableTangent`'s `fields` `NamedTuple` with the target slot replaced and `setfield!`s it back **directly** (what `set_tangent_field!` compiles to, minus the `CallInfo`-less `:call`) — falling back to a `set_tangent_field!` `:invoke` for a `PossiblyUninitTangent` target slot. Only ever differentiable on a genuinely mutable primal (none of `Dual`/`Tuple`/`NamedTuple`/`Array` are mutable in ordinary user code). |
| `Core.memorynew` | Array allocation, step 1/4 of the sequence completed by the `Expr(:new, ::Array, …)` row above (`zeros`/`similar`/`Vector{T}(undef,n)`/comprehensions all lower to `memorynew -> memoryrefnew -> Core.tuple -> %new`). Shadow allocates a same-length, uninitialized `Memory{tangent_type(P)}` via the identical builtin — safe because every element the primal ever *reads* was necessarily *written* first, by an already-dualized `memoryrefset!`. The length argument is `presolve`d (structural, not differentiable), mirroring `Core.tuple`'s own `NoTangent()` shadow below. |
| `Core.memoryrefnew` / `Core.memoryrefget` / `Core.memoryrefset!` | Array indexing's `MemoryRef` layer (`v[i]`/`v[i]=x` lower to `getfield(:ref)` + these three). Each mirrors the *identical* builtin call onto the shadow `MemoryRef` (itself typed `MemoryRef{tangent_type(P)}` — see `tangent_type(MemoryRef{P})` in `DifferCore/src/tangents.jl`) — no wrapper needed, since the shadow array genuinely holds tangent data at the same offsets. `memoryrefnew`'s shadow **always forces its own boundscheck to `true`**, independent of the primal's: `Dual`'s constructor never checks a tangent array is the same *length* as its primal, so a mismatched-length tangent must raise a catchable `BoundsError`, not corrupt memory via an unchecked out-of-bounds ref. |
| `Core.getfield` on a `MemoryRef`/`Memory` | Same-shape like `Array` (`tangent_type(MemoryRef{P}) === MemoryRef{tangent_type(P)}`) but with its own branch, because the shadow's field types don't all match the primal's: the read is typed by the *shadow object's* field type and then reconciled with the required `tangent_type(Ti)` by a no-op `bitcast` when they differ (only ever for a `Ptr` field — see the `Ptr{Nothing}` gotcha below). Gated on the `MemoryRef`/`Memory` aliases specifically, **not** `GenericMemoryRef`/`GenericMemory`: an atomic/other-addrspace one has no same-shape `tangent_type` method (its tangent is an ordinary `Tangent`), so mirroring a `getfield` onto it would fail at run time. The data-pointer field (`:ptr_or_offset`/`:ptr`) additionally requires the *inline-bits* element layout on **both** sides — see the layout gotcha below. |
| `Core.:(===)` | Identity/egal, pervasive in `eachindex`/iterate-protocol loops. Primal passed through; shadow always `NoTangent()` (result is always `Bool`). |
| `Core.isdefined` | Field-definedness check (the boxed-capture `throw_undef_if_not` guard, or a user `isdefined(x,:f)`). Primal passed through; shadow always `NoTangent()` (result is always `Bool`). |
| `Core.tuple` | Same-shape tangent tuple (mirrors the `Expr(:new, ::Tuple, …)` case). Needed on the live path for ≥2-D array indexing, where the `BoundsError` index tuple is hoisted into a reachable block rather than living only in the unreachable throw block the 1-D case keeps it in. |
| `Core.ifelse` | A branchless select: primal is `ifelse` on the presolved operands; shadow is the same select applied to the (same-shape) shadow operands, indexed by the same (non-differentiable) condition — `NoTangent()` if the result's tangent type is trivial. Handled inline (like `getfield`) rather than bailing as an unrecognized builtin, so it stays dualizable if this IR is itself re-dualized at a higher order — needed for `max_float`/`min_float`'s rule above, which emits one to pick the surviving operand's tangent without splitting the block (a real Julia `?:` branch would). |
| any other `Core.Builtin` | **bail** (`return nothing`) — e.g. `Core.memoryrefoffset` (used by `push!`/`resize!`) or non-bits/undef-checked array element access. |
| surviving `:call`/`:invoke` (e.g. `sin`, `cos`, or any composite function with no intrinsic-level path), callee + args + result all concrete | `frule_split!`: wrap the callee (`Dual{ftype,NoTangent}`) and each arg (`Dual{P,tangent_type(P)}`) in a fresh `Dual` via `%new`, then emit a static `:invoke` to a *compiled `CodeInstance`* (via `frule_codeinstance`) when resolvable, else a dynamic `:call` to `frule!!`. Result is `Dual{R,tangent_type(R)}`. Self-/mutually-recursive callees are detected first (`dual_recursive_impl_mi`) and routed to a bare-`MethodInstance` self-`:invoke` or an uninlined dynamic `:call` respectively — see "Recursion" below. |
| surviving call with a non-concrete callee or argument type (a dynamic `apply_generic` dispatch) | `frule_split!` emits a static call to the runtime `dynamic_frule` dispatcher (callee tangent + primals + tangents passed as `Core.tuple`s), which rebuilds concrete `Dual`s and dispatches `frule!!` at run time — see the "Dynamic dispatch" section below. A call with concrete callee+args but an abstract *result* type stays on the static `:invoke` path, typed `dual_type(R)`. |
| `GotoNode` / `GotoIfNot` | Passed through basically unchanged — see "Control flow" below. |
| `PhiNode` | Duplicated into a primal-phi (typed `Ti`) and a shadow-phi (typed `tangent_type(Ti)`) — see "Control flow" below. |
| `UpsilonNode` / `PhiCNode` (try/catch value capture) | Duplicated into primal + shadow copies, exactly like `PhiNode` (shadow typed `tangent_type(Ti)`), reusing the same `pending` forward-ref mechanism. |
| `EnterNode` / `Expr(:leave` / `:pop_exception` / `:the_exception)` | Control markers: carried through unchanged (block-numbered `catch_dest`, `SSAValue` refs remapped); `:the_exception`'s shadow is a zero tangent. |
| `Expr(:gc_preserve_begin, args...)` / `:gc_preserve_end` | `GC.@preserve`. One `gc_preserve_begin` (typed `Any`) rooting **both** the primal and the shadow of every operand — the dualized code holds interior pointers into both, so rooting only the primal would let the shadow array be freed while a live shadow `Ptr` still points into it. A tangent that isn't an `SSAValue`/`Argument` is skipped (a literal has nothing of ours to root; codegen ignores constants here anyway). `gc_preserve_end` just `presolve`s the token — `verify_ir` deliberately skips its usual dominance check for that head, since a token may span try/catch blocks. Both are also handled in the unreachable-block chain. |
| `bitcast` / `pointerref` / `pointerset` / `add_ptr` / `sub_ptr` (raw pointers) | Mirrors, resting on `tangent_type(Ptr{P}) === Ptr{tangent_type(P)}`: a shadow pointer addresses tangent storage at the position its primal addresses. `bitcast` mirrors only *within* `Ptr` space (that's `pointer(v)`'s tail: `getfield(ref, :ptr_or_offset)` then `bitcast(Ptr{P}, _)`) and keeps its old zero-tangent behaviour for non-pointer results; `pointerref`/`pointerset` mirror the *element index*, which is what makes a differing tangent stride correct (the load scales by the shadow pointer's own element type); `add_ptr`/`sub_ptr` mirror a *byte* offset and so demand equal strides. Anything unmirrorable **declines with its own reason** (see below): a `Ptr` built from an integer has no tangent storage and `Ptr` has no zero tangent, an abstract `Ptr` has no shadow pointer, and a mismatched stride would land mid-element. |
| `Expr(:foreigncall, ...)` | `ccall`. Parsed by `_fc_parse` (`DifferForwards/src/foreigncalls.jl`) and dispatched per *target symbol* to `apply_foreigncall_frule!(Val(name), fc, Ti, ctx)`, mirroring the intrinsic/builtin dispatch. An unregistered target **bails** — there is deliberately no "primal + zero tangent" fallback, since native code can write through any pointer it is handed (a `memmove` given that treatment leaves the destination's tangent *stale*, not zero). A non-literal target (a runtime function pointer) bails before dispatch. Registered: `memmove`/`memcpy` — see the section below. |
| `Expr(:loopinfo, ...)` | `@simd`'s marker. Copied through verbatim, no shadow of its own. Nothing is `presolve`d, and the reason is stronger than "its args are only Symbols": `:loopinfo` isn't in the compiler's `is_relevant_expr`, so `userefs` never traverses its operands at all — resolving them would be actively wrong. Two invariants: codegen consumes the marker positionally from the block terminator backwards, so nothing may be emitted *between* it and the terminator (this pass emits shadow statements before it — fine); and a `julia.ivdep` marker carries into the dualized loop, where it now also asserts non-aliasing for the shadow accesses (true whenever it was true of the primal, since the shadow mirrors the primal's access pattern one-for-one). |
| `Expr(:boundscheck, ...)` | Array/tuple indexing's compile-time-decided bounds-check mode marker, consulted downstream by a `GotoIfNot`/a `getfield`/`memoryref*` boundscheck argument. Emitted through unchanged; its own result is a non-differentiable `Bool` (zero shadow). |
| `Expr(:throw_undef_if_not, name, cond)` | Undef-var/boxed-capture guard. `name` is a bare Symbol/GlobalRef, copied through verbatim (never resolved/dualized); `cond` is resolved uniformly with everything else. No shadow-bearing value (zero tangent). |
| `GlobalRef` / any other non-`Expr` | Alias via `presolve`/`tresolve`, no new instruction — **except** a bare `GlobalRef` *statement* (a non-`const` global load), which must be `emit!`ted as a real instruction; see gotcha #4 below. |
| anything else | **bail**. |

## Control flow: preserve block topology 1:1

This is the key design decision that makes branches/loops tractable: **the transform never
splits, merges, reorders, or adds/removes basic blocks** — it only expands each *original*
statement into more instructions. Block count, order, predecessors, and successors are therefore
identical between the primal and the dualized IR.

Consequently, in post-optimization `IRCode`, `GotoNode.label`, `GotoIfNot.dest`, `PhiNode.edges`
entries, and `EnterNode.catch_dest` are already **basic-block numbers**, not statement indices
(this is a `slot2ssa!` rewrite — earlier, pre-SSA `CodeInfo` uses statement indices for the same
fields, which is a common source of confusion if you go reading unrelated compiler code). Since
block numbers never change here, these fields are copied through **completely unchanged** — no
remapping table needed. Only each block's `StmtRange` changes (because instruction *counts*
within a block grow), and that's tracked live during the single pass over primal statements via
`block_start_new[bidx]` bookkeeping — see the `while bidx < nblocks && i > ...` loop near the top
of the per-statement loop.

`PhiNode` becomes **two** phis (a primal-phi and a shadow-phi) sharing the primal's `edges`
verbatim, each built by `presolve`/`tresolve`-ing every edge value. The one new mechanism this
requires: a loop back-edge's phi operand can reference an SSA index *not yet processed* in the
linear walk (its definition comes later, inside the loop body). This is resolved with a
`pending::Dict{Int,Vector{Tuple{Vector{Any},Int,Bool}}}` table keyed by the referenced *original*
SSA index — each entry records `(target_values_vector, slot, want_primal)`. When that original
index is eventually processed, a uniform flush (`if haskey(pending, i) ... end`, run for *every*
statement kind, not just phis) patches the phi's `values` vector in place. This works with no
two-pass renumbering because `PhiNode.values` is a plain mutable `Vector` even though `PhiNode`
itself is immutable. `PhiCNode` (try/catch capture collection) reuses the identical
`resolve_phi_values`/`pending` mechanism.

## Activity: constant arguments (`_activity` / `_materialized`)

`Inactive()` in a `Dual`'s tangent slot is the caller declaring that argument **constant**:
`frule!!(fd, Dual(x, dx), Dual(w, Inactive()))`. `Dual`'s inner constructors were relaxed to admit it
(they used to enforce `tangent_type(P) == T` flatly); `tangent_shadow_type(P) =
Union{tangent_type(P), Inactive}` is the validity predicate, and `code_dual_ircode(f, at;
inactive=(2,))` seeds it for reflection. See `differ-tangent-system` for `Inactive` itself and
`differ-reverse-engine` for reverse mode's much larger version of this.

Three module-level functions, all called once at the top of `dualize_to_ircode`:

- **`_arg_active(dualparams, nfixed, nslots, tt)`** — which argument slots carry a derivative, in the
  *packed* space the primal IR's `Core.Argument` numbering lives in. `Inactive` in the tangent slot
  says constant; `tangent_type(P) === NoTangent` says the same from the type alone. A vararg
  primal's packed tail slot is active if *any* trailing element is.
- **`_activity(pir, iworld, tt, nslots, arg_active)`** — a monotone least fixpoint over the primal
  IR (a loop-carried `PhiNode` reads a back-edge value not yet computed). Conservatism grows "may be
  active", so an unrecognised value-producing `Expr` defaults to **active**.
- **`_materialized(pir, active, nslots, arg_active)`** — which *inactive* values nonetheless need a
  real zero, because something active reads them.

An inactive statement is **replayed primally** — the primal reconstructed faithfully, no shadow
emitted, `shadow[i] = Inactive()`. An inactive `PhiNode`/`PhiCNode`/`UpsilonNode` emits only its
primal half, so a loop-carried constant costs no shadow phi at all. `_act_replayable`/`_act_phi_like`
decide which arms these are; control-flow nodes and the marker `Expr` heads stay on their existing
arms.

**Where an inactive value read by an active one goes** is `_materialized`'s decision, and it has two
answers. A call the main loop routes through `frule_split!`'s *static* path takes the operand as
`Inactive()` — wrapped `Dual{P,Inactive}`, keying its own carrier specialisation — because both a
hand `frule!!` and the `@generated` fallback accept one. Every other consumer gets a real zero from
`zero_shadow(Ti, primal[i])`, emitted at the value's *definition*, which dominates every use
including a phi edge: intrinsic, builtin and `:foreigncall` rules, phi-likes, `%new`, and the
`ReturnNode`. `_act_tolerates_inactive` draws that line, repeating every condition the dispatch arm
and `frule_split!` test; anything it cannot resolve answers "materialise", because a missed
tolerance costs a zero while a wrong one is a `MethodError` inside generated IR.

Routing to the rule rather than zeroing is what recovers `Inactive`'s structural strong zero: a
widened `dot` skips its `dot(x, dy)` term outright, where multiplying by a materialised zero turns a
constant holding `Inf` into a `NaN`. (Reverse's #117 needed a per-`(phi, edge)` zero hoisted into the
entry block precisely because a phi must lead its block; defining-point materialisation dissolves
that case.) The fixpoint in `_materialized` exists because a phi-like statement builds its shadow
from its *operands* rather than from `zero_shadow` — materialising one materialises those too.

**`inactive_shadow` decides `Inactive()` vs. a zero for the value itself.** Two kinds never carry
`Inactive`: a type with no tangent space (nothing to decline, and `NoTangent()` is a free literal),
and a primal that is itself a `Dual`, i.e. the self-tangent nesting used at order ≥ 2, where a shadow
must be a `Dual` of that same type. Handing `Inactive` to the latter breaks the higher-order
composition lookup in `_build_dual_ir` with "could not find an inner carrier method to compose".

**The payoff is mostly coverage.** An inactive `:call` never reaches `frule_split!`, so a callee the
transform could not have dualized at all (a `Dict` lookup, logging, string handling) no longer bails
the whole build. Elided shadow arithmetic, the skipped `zero_tangent` allocation for a constant
array argument, and the strong zero above are the secondary wins.

**Hand rules accept `Inactive` and must never return it.** All 41 `frule!!` methods destructure their
shadows and guard on `isactive` (`_inert` where a `NoTangent` slot needs the same treatment); a
mutating rule refuses an inactive *destination* via `_require_active_dest`, since the destination's
shadow is also the result's. A rule always returns a real zero tangent, because `frule_split!`
declares every nested call's result slot at `tangent_type(R)` before the callee is compiled — the
same entanglement #138 describes. Forward signatures leave the tangent parameter free, so a rule
that has *not* been widened still matches and fails inside its body; `test_forward_rule_activity.jl`
audits every method under every activity mask, including that no rule has been added without a
fixture. A fixture can also name a `frozen` slot — a parameter with no implemented derivative, such
as a Bessel order — which the rule must refuse when handed a nonzero tangent. `ext/rules_ad_runtime.jl`'s reverse-runtime rules are in the same table: their value slots
materialise a zero rather than skipping, because a push must pair with a pop whatever the pushed
value's activity.

**Four rules that are load-bearing:**

- **"Result type has no tangent space ⇒ inactive" gates only the alias/merge/`%new` arms, never a
  call.** A generic call routinely returns `Nothing` while writing through an argument
  (`Base._growend_internal!`, `copyto!`, `mul!`), so calls go by their operands alone.
- **Forward mode does *not* copy reverse's exemption for a rule-less `Core.Builtin`/intrinsic.**
  Reverse needs it (`x === y` on active operands would otherwise have no rule and bail); forward
  already has rules for `===`/`isdefined` and the `@inactive_builtin` group, and here a rule-less
  builtin's bail is load-bearing — it is how growable-array mutation (`push!`, via
  `Core.memoryrefoffset`) is refused. Marking one inactive replays it primally and walks on into code
  the transform cannot handle. This was caught by `test_forward_arrays.jl`'s growable-array bail.
- **A raw pointer is an activity root** — any statement whose type is `<: Ptr`, plus
  `pointerref`/`pointerset`/their atomic forms (`_act_ptr_deref`). Same rationale as `:foreigncall`:
  what a pointer addresses is outside the analysis, so how the *address* was computed does not bound
  what reading through it depends on. Without it, `unsafe_load(Ptr{Float64}(u))` for a `u::UInt` (no
  tangent space, hence inactive) is silently zeroed where forward mode deliberately bails — caught by
  `test_forward_pointers.jl`'s "graceful bails" testset.
- **A locally allocated mutable object is an activity root**, not a function of its initialiser
  operands (`%new` of a mutable type with a non-trivial tangent, `Core.memorynew`): an active value
  may be written into it later. `:foreigncall` is unconditionally active for the same reason.

**Vararg tails and order ≥ 2 degrade rather than bail.** A constant trailing element gets a
materialised zero in the prologue instead of an `Inactive()` slot, which keeps the packed tangent
tuple at its primal-derived type `Tva` — so nothing downstream needs vararg activity awareness, and
`vttys` is reconstructed from the materialised types for the existing shape check. Same for an
inactive seed on the higher-order (`pir_is_vararg`) path: correct, with the elision happening one
order down. Reverse mode instead models per-element tail activity with a mixed `Tuple{…,Inactive,…}`
shadow type (ISSUES #116); forward mode deliberately does not.

**Known limitations**, all deferred deliberately (see ISSUES):

- A hand rule called *directly* with an `Inactive` argument still throws. A signature like
  `Dual{Vector{Float64}}` leaves the tangent parameter free, so it matches `Dual{…,Inactive}` and
  then fails in the body — nothing materialises at the `frule!!` boundary itself. Only the derived
  path supports a constant argument. Widening the rules is the other half of this feature.
- Materialising forfeits `Inactive`'s *structural* strong zero: `Inf * 0.0` is `NaN` where reverse's
  `@ifactive` short-circuits before the arithmetic. Rule widening is what would recover it.
- A materialised zero inside a loop is recomputed per iteration (free for scalars, an allocation for
  an array). Reverse hoists loop-invariant ones; forward does not yet.
- The returned carrier is always `Dual{R, tangent_type(R)}` — an inactive result is materialised at
  the `ReturnNode` rather than returned as `Dual{R,Inactive}`, which would entangle with
  `frule_split!`'s declared result type the way reverse's #126/#127 did.

## Recursion

A surviving call's *derived* (non-hand-ruled) carrier can resolve back to the very `impl_mi` this
pass is currently building (direct self-recursion) or to some other carrier that's mid-compile on
this task (mutual recursion, A→B→A). `frule_split!` checks this *before* ever calling
`frule_codeinstance` (which would `typeinf_ext_toplevel` straight into the cycle), via
`dual_recursive_impl_mi` and the task-local `dualized_impl_in_progress()` guard:

- **Self-recursion** — `impl_mi` is the bare, uncompiled `MethodInstance` currently being dualized:
  emit a static `Expr(:invoke, impl_mi, dualimplg, fd, duals...)` against the *bare* `MethodInstance`
  (`IR_FLAG_NOINLINE`, so the inliner doesn't try to inline the self-call into itself). This is legal
  and fast **only** for this exact case — codegen's `mi == ctx.linfo` self-recursion fast path emits a
  direct specsig call with no `CodeInstance` needed. Never do this for a non-self target: a bare
  non-self `MethodInstance` `:invoke` degrades to a boxed `jl_invoke` against the *native* method
  cache, silently running `dualized_impl`'s throwing stub instead of the derivative.
- **Mutual recursion** — the callee's own carrier is genuinely mid-compile (in
  `dualized_impl_in_progress()`) but isn't this build's own `impl_mi`: fall back to an ordinary
  dynamic `:call` to `frule!!` (uninlined, since it carries no `CallInfo`), resolved at run time
  against whatever `CodeInstance` the in-progress build eventually installs. This is what breaks the
  compile-time cycle — any SCC stops recursing at the first back-edge into an already-in-progress
  carrier.

`dualized_impl_in_progress()` (task-local storage, not a plain global — concurrent compilation on
another thread must not corrupt it) is a backstop for anything that reaches `build_dual_ir` some
other way; it is not the primary mechanism, since the two paths above resolve recursion at the call
site before ever recursing into `build_dual_ir` for the same `impl_mi`.

## Higher-order composition and the vararg prologue's `pir_arg_offset`

Higher-order requests are handled in `build_dual_ir` (`DifferForwards/src/forward_interp.jl`) by
**composing the transform** (Option A): the primal for an order-k carrier is the order-(k-1) dual
`IRCode`, re-dualized. `build_dual_ir`'s local `compose(offset)` peels one `Dual` level off
`dualparams[1+offset:end]` to form the inner carrier signature, gets its optimized dual IR
(`optimized_dual_ir`), and calls back into `dualize_to_ircode` with `pir_is_vararg=true,
pir_arg_offset=offset`. Two shapes reach it:

- **Uniform seeds** (`code_dual_ircode order≥2`, or a hand-built `frule!!(fseed_k, seed_k)`): every dual
  arg *including the function* is nested one level, so the whole `dualparams` list peels — `offset=0`.
- **Composed differentiation** (`D`-of-`D`, nesting a `frule!!`-based derivative operator): when the
  outer pass dualizes a closure whose body called `frule!!`, that inner call survives in the primal IR
  as a `dualized_impl` `:invoke` (inlined generated `frule!!`) or a `frule!!` `:invoke`. `frule_split!`
  re-wraps its callee as a **non-nested function slot** (`Dual{typeof(dualized_impl),NoTangent}` or
  `Dual{typeof(frule!!),NoTangent}`) at `dualparams[1]`, naming the function being re-differentiated
  rather than a value arg — so it is dropped (`offset=1`) and only the trailing nested args peel. The
  `frule!!`-slot case is disambiguated from the base-case hand rule by whether `primal_of_impl` resolves
  to the generated (vararg) `frule!!` (compose) or a concrete hand rule (base case). It recurses, so
  `D∘D∘D…` works to arbitrary order.

`pir_arg_offset` only affects the **vararg argument-reconstruction prologue**: the tuple index reads
`getfield(Argument(2), j+offset)` (skipping the offset function slots in the outer carrier's vararg
tuple) while the reconstructed inner tuple is offset-free and must equal `pir.argtypes[2]` (asserted).
The rest of the engine is offset-agnostic.

A `Dual` is its own tangent type only when both its fields can live in a same-typed shadow: each
field's tangent type is either itself or `NoTangent` (the slot then carries the primal through).
`tangent_type(::Type{Dual{P,T}})` (`DifferCore/src/dual.jl`) tests that with `_dual_selfsim_field`;
a `Dual` carrying a struct/closure with differentiable fields fails it and gets the ordinary
two-field `Tangent{@NamedTuple{primal, tangent}}` instead. Consumers that mirror a `Dual` shadow
field-by-field must check the same thing before doing so — the `%new` arm of `dualize_to_ircode`
guards on `tt(T) === T`, `getfield`'s same-shape branch on `ctx.tt(Pobj) === Pobj` — or they emit a
`%new` that `TypeError`s at run time on a `Tangent` in a primal-typed slot.

## Known `Core.Compiler.verify_ir` gotchas (cost real debugging time — read before you hit them again)

`Compiler.verify_ir(ir)` is called unconditionally at the end of `dualize_to_ircode` as a
correctness safety net. **A `verify_ir` failure means a bug in this transform, not unsupported
input** — it's allowed to throw plainly rather than being caught and turned into a graceful bail.
Each entry below is a real bug that has already been hit, none of them exotic — expect variations
again. Items 1-4 are genuine `verify_ir` rejections; 5-7 pass `verify_ir` clean and bite later (a
dynamic dispatch, a run-time `TypeError`, a silent miscompile respectively); 10 is a distinct
failure mode entirely (reentrant `@generated`-function compilation, not this transform's own IR
shape).

1. **`CFG.index` off-by-one.** `Core.Compiler.CFG.index[b]` must be `stop_of_block_b + 1` — one
   statement **past** the block's end — not the block's last statement index itself. This mirrors
   `compute_basic_blocks` in `Compiler/src/ssair/ir.jl` (`basic_block_index[i] = last`, where
   block `i`'s own range stops at `last - 1`). Getting this wrong produces the error `End of BB $n
   ($x) is not one less than CFG index ($x)`.
2. **Bare `GlobalRef`s to non-`Core`/`Base` modules can't be embedded as values.** A `GlobalRef`
   used directly as an *operand* (not a call's callee — as a literal value, e.g. inside a `%new`
   or as the display-callee of an `:invoke`) is rejected by `verify_ir` with `Unbound or
   partitioned GlobalRef not allowed in value position`, unless its module is `Core`/`Base` or its
   binding is proven constant across the IR's valid worlds. `GlobalRef(DifferForwards, :frule!!)`
   and a primal callee like `GlobalRef(Main, :sin)` both trip this. Fix: resolve to the actual
   (stable, singleton) value first — `_calleeval(fpos, world)` for a primal callee, or reference the
   helper functions directly as Julia values (the engine binds `fruleg = frule!!`, `zerotang_g =
   zero_tangent`, `buildtang_g = build_tangent`, `gettfield_g = get_tangent_field` for exactly this
   reason) — and embed *that*, not the `GlobalRef`.
   **For an ordinary operand, use `vpresolve`, not `presolve`** (`presolve` passes a `GlobalRef`
   through unchanged). It routes through `gref_operand!`, which embeds a defined *constant*
   binding's value as a literal and otherwise emits a real global-load instruction and uses its
   `SSAValue`; `gref_optype` is the matching declared type. This is what ISSUES #60 was: the `:new`
   arm used plain `presolve` for its field operands *and* tested `T <: Dual` on an unresolved
   `GlobalRef` type argument (a raw `TypeError`, since a struct defined at module level lowers to
   `%new(Main.S, …)`). Call/invoke *argument* positions don't need it — `frule_split!` and the
   `Core.Const` arm already embed a statically-known operand's value via `_calleeval` — but any new
   arm that puts an operand in value position does. Two related traps live in `optype`, both silent
   miscompiles rather than verify failures (ISSUES #63): `_optype`'s literal fallback reports
   `typeof(node) === GlobalRef`, the type of the *node* rather than of the value the binding names
   (which sent `getfield(Main.CONST_VEC, :ref)` down the `getfield` rule's general-struct branch and
   `MethodError`d at run time), and constness must be asked at the *inference* world via
   `_globalref_isconst` — `Base.isconst(mod, name)` answers for the current task world, which inside
   the generated `frule!!` body predates the user's `const` declaration and reports `false` for a
   genuinely constant binding.
3. **Empty blocks corrupt the following block's range.** A block whose every original statement is
   a pure alias (`PiNode`, or a bare `nothing` placeholder statement — Julia leaves these between
   adjacent loops, e.g. in `sumk2`'s primal IR) emits zero new instructions. Left unhandled, this
   makes that block's `StmtRange` empty and misattributes the *next* block's terminator to the
   wrong block. Fix: backfill a `nothing` placeholder (matching Julia's own convention) whenever a
   block would otherwise end up with no emitted instructions — done at both the mid-loop
   block-boundary crossing and, separately, for the final block after the main loop ends.
4. **A bare `GlobalRef` *statement* is a load, not a pure alias.** A non-`const` global read (e.g.
   `y[]` where `y = Ref{Any}(1.0)` at module scope) shows up in the primal IR as its own statement
   whose `:stmt` is literally a `GlobalRef` (`%1 = Main.y::Any`), *not* nested inside another
   expression. Treating it like a `PiNode` (`primal[i] = presolve(s)`, no new instruction) aliases
   the raw `GlobalRef` itself forward; once a later statement uses that alias as an operand, it trips
   the same "value position" `verify_ir` rejection as gotcha #2 above (a non-`const` global's binding
   isn't proven constant). Fix: `emit!(s, Ti)` — the statement must be *emitted* as a real
   instruction so later references go through a legitimate `SSAValue`, not the bare node; its shadow
   is the zero tangent (`zero_shadow`), same treatment as any other non-differentiable source.
5. **A synthesized `Expr(:call, helper, …)` carries no `CallInfo` — it survives as a dynamic
   dispatch, not an inlined call (a *perf* trap that passes `verify_ir` clean).** IR the engine emits
   by hand has no inference metadata attached, so `CC.ssa_inlining_pass!` has no method-match info and
   cannot inline it — the call runs as a genuine runtime dynamic dispatch, boxing its result, even
   though the hand-written `::T` result annotation makes the IR *look* type-stable. This is what made
   the `get_tangent_field`/`set_tangent_field!`/`build_tangent` tangent-helper calls allocate per
   iteration (ISSUES #54: 209×/227×/757× the primal). Two fixes, applied together: emit the operation
   *directly* as builtins the engine already lowers (`getfield`/`setfield!`/`%new` on the `fields`
   `NamedTuple` — see the `getfield`/`setfield!`/`:new` table rows), and for the residual cold
   fallbacks route through `emit_invoke!`, which resolves a `CodeInstance` (via `static_codeinstance`,
   modelled on `frule_codeinstance`) and emits a static `Expr(:invoke, ci, …)` instead. **Only invoke
   when every argument type is concrete** — a non-concrete argtype (e.g. `Any`, a dynamic-dispatch
   result) means dispatch is genuinely runtime, and freezing an `:invoke` to the over-general method
   `findsup` returns both defeats dispatch and can hit an unbound-static-param path; a bare `:call`
   there is correct (it dispatches on the concrete runtime value), which is the fallback.
6. **A shadow statement's *declared* type must be `tangent_type(Ti)` even when the mirrored operation
   produces something else (`verify_ir` won't catch the lie; `%new` will, at run time).** The case
   that forced this: `MemoryRef{Float64}`'s `:ptr_or_offset` is a `Ptr{Nothing}` on *both* primal and
   shadow, but `tangent_type(Ptr{Nothing}) === Ptr{NoTangent}`, so mirroring the `getfield` and
   declaring it `tangent_type(Ti)` would annotate a `Ptr{Nothing}` value as `Ptr{NoTangent}`. That
   passes `verify_ir` (it doesn't type-check operands) and even runs, since both are same-sized
   pointer bitstypes — until the value reaches a `%new(Dual{P,tangent_type(P)}, …)`, which *does*
   type-check its fields and `TypeError`s. Fix: declare the mirrored read with the shadow object's
   *real* field type, then reconcile with a no-op `bitcast` to `tangent_type(Ti)` (which is what
   `Base.convert(::Type{Ptr{T}}, ::Ptr)` compiles to anyway — legal in both `verify_ir` and codegen,
   and a `DataType` literal in operand position is already precedent from `memorynew`'s shadow).
7. **`MemoryRef.ptr_or_offset`/`Memory.ptr` is only an *address* in the inline-bits layout regime, and
   the two sides can be in different regimes (a silent-miscompile trap, not a verify trap).** Base's
   own `unsafe_convert(::Type{Ptr{Cvoid}}, ::GenericMemoryRef)` branches on `arrayelem == isunion ||
   elsz == 0`, where the field holds an *offset* instead. A shadow buffer's element type is
   `tangent_type(P)`, so it can fall on the other side of that branch: a `Vector{Int}`'s shadow is a
   `Memory{NoTangent}` (zero-size elements). **Do not believe the older claim that the field is then
   `Ptr(0x0)`** — for `layoutsize == 0` a `MemoryRef` stores its *0-based index* there. Measured:
   element 3 of a `Memory{NoTangent}` reads back `Ptr{Nothing}(0x2)`, and the `Memory`'s own `:ptr`
   is a genuine (zero-size) heap address. So mirroring the read yields a small bogus address (or,
   for a bits-union, an offset scaled by the *primal's* element size) that the Ptr→Ptr `bitcast`
   rule would happily launder into a genuine dereference — never a recognisable null.
   `_bi_mem_ptr_field_regime` therefore classifies rather than mirrors: `:address` when
   `datatype_arrayelem == 0 && datatype_layoutsize != 0` on **both** buffers, `:null` when the shadow
   elements are `NoTangent`, a bail otherwise. In the `:null` regime the shadow is the *synthesised*
   `NULL_SHADOW_PTR = Ptr{NoTangent}(0)` sentinel (`DifferForwards/src/intrinsics.jl`), which is how
   `copy(::Vector{Int})`, `Bool` masks and `collect(1:n)` dualize at all (ISSUES #62a).
   **If you add a rule that consumes a shadow `Ptr`, decide statically what it does with the
   sentinel** — recognise it with `=== NULL_SHADOW_PTR` on the resolved shadow operand (it is always
   a compile-time literal, and `Ptr{NoTangent}` only ever arises as a tangent type). Existing rules:
   `bitcast` carries it through in `Ptr{NoTangent}` space and **bails** when asked to relabel it into
   a differentiable pointer; `add_ptr`/`sub_ptr` carry it through *without* applying the offset;
   `pointerref`/`pointerset` skip the shadow operation when the element's tangent is `NoTangent` and
   bail otherwise; `memmove`/`memcpy` emits the primal call alone (and bails when the two buffers are
   in different regimes). `const_tangent` also special-cases it, because it is a `Ptr` literal in the
   emitted IR and `zero_tangent(::Ptr)` throws by design — a null shadow's own shadow is again null,
   which is what keeps such IR re-dualizable at order ≥ 2.
   The same "a position must mean the same thing on both sides" reasoning is what makes `add_ptr`
   demand equal element strides for a *real* shadow pointer
   (`aligned_sizeof(P) == aligned_sizeof(tangent_type(P)) > 0` — the `> 0` is load-bearing, since
   `aligned_sizeof` is `0` for both `Nothing` and `NoTangent`).
8. **Julia's own `@boundscheck` guards are *not* a length check you can lean on — they're elided under
   the default `--check-bounds=auto` (a segfault trap, and a testing trap on top of it).** Found while
   adding the `memmove` mirror. `Base.unsafe_copyto!` contains exactly the guard you'd want
   (`memoryrefnew(ref, n, true)`), and this pass faithfully mirrors it onto the shadow — but it sits
   inside `_copyto_impl!`'s `@inbounds` block, and `bounds_check_enabled` (`julia/src/cgutils.cpp`)
   returns 0 there under the default. Measured with the mirror but no explicit guard: a short
   **destination** tangent segfaults (`signal 11`), a short **source** tangent silently returns garbage
   read from uninitialised heap. The nasty part is that both raise a clean `BoundsError` under
   `--check-bounds=yes`, which is what `Pkg.test()` runs under by default — so a suite can "confirm"
   a guard that isn't there. Two rules follow: emit your *own* check (the `_fc_check_extent`
   `@noinline` helper, forcing `boundscheck=true`) rather than relying on one inherited from the
   primal; and when touching anything boundscheck-related, run the suite at least once under the
   real default too (`julia +1.13 --project=DifferForwards/test DifferForwards/test/runtests.jl`,
   no `--check-bounds` override), not just via `Pkg.test()` — `Pkg.test()` itself is fine for
   everyday iteration (see `differ-architecture`'s "Running things" section), it just isn't
   sufficient on its own to catch this specific bug class.
9. **A `Core.PartialStruct` can reach `fieldtype`, not just `tangent_type`.** Gotcha-adjacent to `tt`'s
   own widening note: `apply_builtin_frule!(::Val{Core.tuple})` called `fieldtype(Ti, j)` on the raw
   statement type and `TypeError`d as soon as inference partly pinned the tuple down (the `Broadcasted`
   argument tuple in `x .+ 1.0`). Any place that consumes `pstmts[i][:type]` as a *structure* — not
   just as a type parameter — needs `CC.widenconst` first (`_widen`, `DifferCore/src/shared_ir_helpers.jl`).
10. **OPEN (ISSUES #84): reentrant `typeinf_ext_toplevel` inside a `@generated` function can produce
    invalid IR or crash at sufficient nesting depth.** Not a `dualize_to_ircode` IR-shape bug on its
    own — it's `const_tangent`'s handling of a `PhiNode` operand that's a raw literal constant (not an
    `SSAValue`) combined with the fragility of calling `typeinf_ext_toplevel` reentrantly from inside a
    `@generated` function's own generator body. Concretely: `exp`'s optimized primal IR has a `PhiNode`
    operand that's a raw literal (`Base.Math.Inf`, on its overflow branch); computing that literal's
    shadow (`zero_tangent(Inf)`, since `tangent_type(Float64) = Float64 ≠ NoTangent`) requires emitting
    a real `invoke` instruction, but a `PhiNode`'s per-edge value normally comes pre-computed from a
    dominating predecessor block — a literal embedded directly in `PhiNode.values` has no such block —
    so `const_tangent` lands the `invoke` in the phi's *own* block, ahead of the phi, which `verify_ir`
    correctly rejects with `"φ node ... is not at the beginning of the basic block"`. This only
    surfaces through **three levels of nested/reentrant `typeinf_ext_toplevel`**
    (`frule_codeinstance` → compiling the callee's generated `frule!!` method → that method's own
    generator issuing a third `typeinf_ext_toplevel` to dualize the callee's body) — a direct
    (non-nested) call to the same primal only needs one level and succeeds. Confirmed with a bare
    `frule!!` call reentering three deep on a composite (non-hand-ruled) function whose primal IR has
    this exact `PhiNode` shape; whether the *outer* symptom is a clean `TypeError` (`frule_codeinstance`
    gets `nothing` back instead of a `CodeInstance`) or a stack-overflow-shaped crash depends on
    incidental `CodeInstance`-caching/warmup state, not anything principled — so don't trust "it didn't
    crash this time" as evidence it's fixed. Two separable problems, either alone needing real engine
    work: (1) `const_tangent`'s literal-`PhiNode`-operand codegen should emit into the correct
    predecessor block, not the phi's own block; (2) `typeinf_ext_toplevel` reentrancy from inside a
    `@generated` generator is fragile in general, independent of (1). Fix (1) first — it's
    self-contained — before reasoning about (2). No regression test currently exercises this (the one
    test that originally hit it stopped doing so once `exp` got a hand-written `frule!!`, which
    resolves via `frule_codeinstance` directly and never reaches `exp`'s own composite-fallback body).
    Any composite function reached through this depth of nested dualization, whose primal IR has a
    raw-literal `PhiNode` operand needing a non-`NoTangent` shadow, is a plausible future trigger — a
    likely candidate class is other branch-heavy transcendental implementations (`log`, `pow`, …) that
    don't yet have a hand rule.

If you need to see the raw IR to debug a `verify_ir` failure, don't fight `@verify_error`'s silent
string-only branch — call `code_dual_ircode(f, argtypes)` from `DifferForwards/src/reflection.jl`
and `println` the resulting `IRCode` yourself (or temporarily add a debug print right before the
`verify_ir` call in `dualize_to_ircode`), rather than trying to coax a dump out of `verify_ir` itself.

## Dynamic dispatch (`apply_generic`) — supported via the runtime `dynamic_frule` dispatcher

A call whose callee or an argument has a non-concrete declared type (e.g. `x + y[]` where `y =
Ref{Any}(...)`, always inferred `Any` regardless of what it holds) is a genuine dynamic dispatch: the
method actually invoked depends on runtime types unknowable at dualization time. It can't be wrapped
statically — a `%new` of a `Dual` would freeze the abstract declared type into the object, and
`frule!!`'s `@generated` dispatch, keyed on exactly those type parameters, could never resolve a
concrete primal from `Dual{Any,…}`. So `frule_split!` (`DifferForwards/src/forward_interp.jl`)
detects the case and **defers it to run time**:

- It emits a *static* `Expr(:call, dynamic_frule, fcallee, ftang, ptuple, ttuple)`, where
  `ptuple`/`ttuple` are `Core.tuple` calls collecting the argument primals and tangents and `ftang`
  is the callee's own tangent. The result is typed `Any`; the primal/tangent are pulled back out with
  `getfield(_, 1)` / `getfield(_, 2)` (typed `R` / `tt(R)` — `tt` widens to `Any` for abstract `R`).
- `dynamic_frule` is an **ordinary function** (no `@noinline`, no `@constprop :none`, no
  `invokelatest` — none of those are load-bearing) that calls `frule!!(Dual(f, tf), map(Dual, primals,
  tangents)...)`. The `map` is the whole trick: at run time the values are concrete, so `map(Dual, …)`
  builds concrete `Dual`s (params inferred from the actual values) and `frule!!` resolves the real
  primal method and dualizes it. It carries the callee's *real* tangent `tf`, not a forced zero, so a
  dynamically-dispatched closure's capture derivatives still propagate.

Note: a statically-known operand (a `GlobalRef`/`QuoteNode`, e.g. the `^` passed as an argument to
`literal_pow`) must be embedded as its *resolved value* with its concrete type in the tuple, not the
raw node (a bare non-Core/Base `GlobalRef` in value position is a `verify_ir` reject).

## The invariant-`Dual` typing rule (why the static path uses `dual_type(R)`, not `Dual{R,tt(R)}`)

`Dual` is **invariant**, so `Dual{Float64,Float64}` is a subtype of the abstract `Dual` (the
UnionAll) but **not** of the concrete `Dual{Any,Any}`. This matters wherever the engine declares the
type of a `Dual`-producing statement for a *non-concrete* primal type `R`:

- The static `frule_split!` path declares its `:invoke`/`:call` result as `dual_type(R)` — which is
  `Dual{R,tangent_type(R)}` for concrete `R` (exact) and the **abstract `Dual`** when `R` is
  non-concrete. The compiled `frule!!` `CodeInstance` returns a concrete `Dual{Rc,Tc}`, which is `<:
  Dual` (sound) but would *not* be `<: Dual{Any,Any}`.
- The `ReturnNode` packs its result the same way *in effect* but by a different mechanism: for
  concrete `R` an exact `%new(Dual{R,tt(R)}, …)`; for non-concrete `R` a **dynamic `Dual(p,t)` call**
  (`dyn_dual!`) annotated `dual_type(R)`. A `%new` there *cannot* simply be re-annotated to a
  supertype: `%new` stamps its exact declared type onto the value, so `%new(Dual{Union{Float64,Int},…})`
  produces a `Dual{Union{…},…}` that is **not** `<: dual_type(R)` (the `Union` of *leaf* `Dual`s) —
  the dynamic call is what makes the runtime value a concrete leaf that the annotation covers.

This is the real fix for a case that once looked like it needed special handling: a call with concrete
callee+args but an inference-widened abstract result (e.g. `_2 * _2 :: Any`, left after a `Ref{Any}`
was SROA'd away — this is what `dynbox` in the tests exercises). It takes the ordinary **static invoke
path**; only its result annotation widens to the abstract `Dual`. Declaring the invariant
`Dual{Any,Any}` instead is unsound and miscompiles to a runtime "Unreachable reached" crash — that
crash was *only* ever the bad type annotation, nothing subtler.

**The `%new`-vs-widen distinction is the crux.** A `%new`'s declared type must equal what it
constructs (widening it is unsound — the `Union` case above). Widening to a supertype is sound *only*
for a value whose type is fixed *elsewhere* and is already a concrete leaf `<:` the supertype — an
`:invoke` result (fixed by the callee `CodeInstance`), or a dynamic `Dual(p,t)` call (fixed at run
time from the actual values). That is exactly why the two non-concrete paths (`frule_split!`'s
`:invoke`, the `ReturnNode`'s `dyn_dual!`) can carry the abstract `dual_type(R)` while a `%new` cannot.

## World-age inside the generated `frule!!` body (resolve bindings at the *inference* world)

Everything reached transitively from the `@generated frule!!` body — `frule_body` →
`typeinf_ext_toplevel(interp, …)` → `build_contextual_ir` → the whole dualization engine — runs at a
**stale generation world** that can *predate a user's (`Main`) function definition*. So **never resolve
a global binding, do a method lookup, or build an interpreter using the *ambient* world**:
`Base.get_world_counter()`, `Core.Compiler.tls_world_age()`, a bare `getglobal`/`isdefined`, or
`invoke_in_world` for a binding (which reparameterizes *dispatch* but **not** global-binding lookup —
its `UndefVarError` still reports the stale world). Always thread the interpreter's inference world
`CC.get_inference_world(interp)` and use world-parameterized primitives:

- `CC.method_table(interp)` for `findsup` (already the case in `primal_of_impl`/`compose`/`frule_codeinstance`);
- `CC.NativeInterpreter(world)` / `retrieve_code_info(mi, world)` for sub-inference (already in `build_dual_ir`/`optimized_dual_ir`);
- **`Base.getglobalref(gr, world)`** (`ccall :jl_eval_globalref`) to read a `GlobalRef`'s value — this
  is what `_calleeval(x, world)` uses, and why its `world` argument is **mandatory (no default)**: a
  silent `get_world_counter()` default is exactly the trap.

`Base` functions accidentally survive a wrong (ambient) world because they were defined at an early
world; **user** functions are the ones that expose the bug. The classic symptom is a `GlobalRef`
leaking into the dual IR — e.g. a callee that couldn't be resolved degrades to a raw `GlobalRef`,
producing `%new(Dual{GlobalRef,…}, foo, …)` and a runtime `TypeError(:new, …, GlobalRef, foo)` at
order ≥2 (the misresolved callee is invisible at first order until re-processing/const-prop reaches
it). This was the root cause of the "differentiate a user function with a hand-written `frule!!`" crash.

**Known open issue (not yet fixed):** the carrier's installed `CodeInstance` does not record real
backedges to the dependencies discovered while building the dual IR (the primal `MethodInstance` from
`typeinf_ircode`, and `frule_codeinstance` `:invoke` targets) — `finishinfer!` derives edges from the
`dualized_impl` *stub* body, and the raw `CC.findsup` lookups aren't edge-tracked. Redefining a primal
or adding/changing a hand `frule!!` therefore does **not** invalidate an already-compiled dual (hence the
manual `refresh_frule()` escape hatch). Systemic (reverse mode shares it). Fixing it means wiring those
dependencies into `me.edges`/`opt.src.edges`.

## Where to go next

- General orientation / why any of this exists, the `ContextualInterpreter` plugin mechanism, the
  `Dual`/`frule!!` calling convention, the file map, and how to run tests: the `differ-architecture`
  skill.
- The tangent type system (`Tangent`/`MutableTangent`/`FData`/`RData`, `tangent_type`, `zero_tangent`,
  …) shared by both modes: the `differ-tangent-system` skill.
- Reverse mode's analogous `rrule!!`-derived dualization engine: the `differ-reverse-engine` skill.
- Adding support for a brand-new IR construct in forward mode: the `differ-extending-ir-support`
  skill.
