---
name: adnext-ircode-dualization
description: Deep-dive reference for ADNext's split-shadow IRCode dualization engine (dualize_to_ircode in src/forward_interp.jl) — how primal/shadow SSA values are tracked, how control flow is handled by preserving block topology 1:1, the PhiNode forward-reference (pending-patch) mechanism, and known Core.Compiler.verify_ir gotchas that have already cost real debugging time. Use this before modifying dualize_to_ircode, debugging a dualization bug, adding support for a new IR construct, or investigating any Core.Compiler.verify_ir / IR-shape failure.
---

# ADNext's IRCode dualization engine

This is the internals reference for `dualize_to_ircode` in `src/forward_interp.jl` — the function
that turns a primal method's fully-optimized `IRCode` into a dualized one. Read the
`adnext-architecture` skill first if you haven't already; this one assumes you know *why* the
transform exists and jumps straight into *how* it works.

## The split-shadow scheme

For every primal SSA statement `i`, the engine computes **two** new values: `primal[i]` (the
reconstructed primal computation) and `shadow[i]` (the parallel tangent computation). At a
`ReturnNode`, the two are packed into one `Dual`. When the returned value's type `R` is **concrete**,
this is an allocation-free `%new(Dual{R, tangent_type(R)}, …)` (the test suite guards this fast path
directly — do not turn it into a dynamic `Dual(...)` call). When `R` is **non-concrete** (a
dynamic-dispatch result, e.g. `Any`, or a `Union`), a `%new` would freeze the over-wide declared type
into the value (`Dual{Any,Any}(9.0, 6.0)` — boxed, and un-composable back into `frule`); instead the
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

Function arguments are unpacked once up front into `parg`/`targ` (via `getfield` on the incoming
`Dual`s), so `presolve`/`tresolve` can treat `Argument`s uniformly with `SSAValue`s.

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
| `Expr(:new, T, args...)` | Primal is a `%new(T, presolved…)`. Shadow **dispatches on `T`**: `T<:Dual` → same-typed `%new(T, …)` (a `Dual` is its own tangent; non-diff *singleton* fields carry the primal value); `T<:Tuple`/`NamedTuple` → `%new(tangent_type(T), …)` with `NoTangent()` in non-diff slots; a general struct → `build_tangent(T, tresolved…)` yielding a `Tangent`/`MutableTangent`. (If `tangent_type(T)==NoTangent`, the shadow is just `NoTangent()`.) |
| intrinsic call (`add_float`/`sub_float`/`neg_float`/`mul_float`/`div_float`, `_fast` variants) | Hand-coded differentiation rule per intrinsic (product/quotient rule for `mul`/`div`). |
| any other intrinsic (int arithmetic, comparisons, conversions) | Primal computed normally; tangent is the *zero tangent* of the result via `zero_shadow(Ti, primal[i])` — `NoTangent()` when `tangent_type(Ti)==NoTangent` (e.g. `Int`/`Bool`), a literal `zero(Ti)` for a concrete `Number`, else a runtime `zero_tangent` on the primal. |
| `Core.getfield` | Primal is `getfield`. Shadow dispatches on the object's primal type: `Dual`/`Tuple`/`NamedTuple` → `getfield` on the (same-shape) shadow; a general struct → `get_tangent_field` on the `Tangent`/`MutableTangent`. (`NoTangent()` if the field's `tangent_type` is `NoTangent`.) |
| any other `Core.Builtin` | **bail** (`return nothing`) — e.g. array/`memoryref` indexing. |
| surviving `:call`/`:invoke` (e.g. `sin`, `cos`, or any composite function with no intrinsic-level path), callee + args + result all concrete | `frule_split!`: wrap the callee (`Dual{ftype,NoTangent}`) and each arg (`Dual{P,tangent_type(P)}`) in a fresh `Dual` via `%new`, then emit a static `:invoke` to a *compiled `CodeInstance`* (via `frule_codeinstance`) when resolvable, else a dynamic `:call` to `frule`. Result is `Dual{R,tangent_type(R)}`. |
| surviving call with a non-concrete callee or argument type (a dynamic `apply_generic` dispatch) | `frule_split!` emits a static call to the runtime `dynamic_frule` dispatcher (callee tangent + primals + tangents passed as `Core.tuple`s), which rebuilds concrete `Dual`s and dispatches `frule` at run time — see the "Dynamic dispatch" section below. A call with concrete callee+args but an abstract *result* type stays on the static `:invoke` path, typed `dual_type(R)`. |
| `GotoNode` / `GotoIfNot` | Passed through basically unchanged — see "Control flow" below. |
| `PhiNode` | Duplicated into a primal-phi (typed `Ti`) and a shadow-phi (typed `tangent_type(Ti)`) — see "Control flow" below. |
| `UpsilonNode` / `PhiCNode` (try/catch value capture) | Duplicated into primal + shadow copies, exactly like `PhiNode` (shadow typed `tangent_type(Ti)`), reusing the same `pending` forward-ref mechanism. |
| `EnterNode` / `Expr(:leave` / `:pop_exception` / `:the_exception)` | Control markers: carried through unchanged (block-numbered `catch_dest`, `SSAValue` refs remapped); `:the_exception`'s shadow is a zero tangent. |
| `GlobalRef` / any other non-`Expr` | Alias via `presolve`/`tresolve`, no new instruction. |
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
itself is immutable.

## Higher-order composition and the vararg prologue's `pir_arg_offset`

Higher-order requests are handled in `build_dual_ir` by **composing the transform** (Option A): the
primal for an order-k carrier is the order-(k-1) dual `IRCode`, re-dualized. `build_dual_ir`'s local
`compose(offset)` peels one `Dual` level off `dualparams[1+offset:end]` to form the inner carrier
signature, gets its optimized dual IR (`optimized_dual_ir`), and calls back into `dualize_to_ircode`
with `pir_is_vararg=true, pir_arg_offset=offset`. Two shapes reach it:

- **Uniform seeds** (`code_dual_ircode order≥2`, or a hand-built `frule(fseed_k, seed_k)`): every dual
  arg *including the function* is nested one level, so the whole `dualparams` list peels — `offset=0`.
- **Composed differentiation** (`D`-of-`D`, nesting a `frule`-based derivative operator): when the
  outer pass dualizes a closure whose body called `frule`, that inner call survives in the primal IR
  as a `dualized_impl` `:invoke` (inlined generated `frule`) or a `frule` `:invoke`. `frule_split!`
  re-wraps its callee as a **non-nested function slot** (`Dual{typeof(dualized_impl),NoTangent}` or
  `Dual{typeof(frule),NoTangent}`) at `dualparams[1]`, naming the function being re-differentiated
  rather than a value arg — so it is dropped (`offset=1`) and only the trailing nested args peel. The
  `frule`-slot case is disambiguated from the base-case hand rule by whether `primal_of_impl` resolves
  to the generated (vararg) `frule` (compose) or a concrete hand rule (base case). It recurses, so
  `D∘D∘D…` works to arbitrary order.

`pir_arg_offset` only affects the **vararg argument-reconstruction prologue**: the tuple index reads
`getfield(Argument(2), j+offset)` (skipping the offset function slots in the outer carrier's vararg
tuple) while the reconstructed inner tuple is offset-free and must equal `pir.argtypes[2]` (asserted).
The rest of the engine is offset-agnostic. **Limit:** a `Dual` carrying a struct/closure with
differentiable fields has no same-typed zero tangent (the self-tangent scheme needs `tangent_type(P)
=== P`), so composing `D` over such a value at order ≥2 errors clearly in `_carrier_zero` (`dual.jl`)
rather than miscompiling.

## Known `Core.Compiler.verify_ir` gotchas (cost real debugging time — read before you hit them again)

`Compiler.verify_ir(ir)` is called unconditionally at the end of `dualize_to_ircode` as a
correctness safety net. **A `verify_ir` failure means a bug in this transform, not unsupported
input** — it's allowed to throw plainly rather than being caught and turned into a graceful bail.
Three real bugs surfaced this way, none of them exotic — expect to hit variations of these again:

1. **`CFG.index` off-by-one.** `Core.Compiler.CFG.index[b]` must be `stop_of_block_b + 1` — one
   statement **past** the block's end — not the block's last statement index itself. This mirrors
   `compute_basic_blocks` in `Compiler/src/ssair/ir.jl` (`basic_block_index[i] = last`, where
   block `i`'s own range stops at `last - 1`). Getting this wrong produces the error `End of BB $n
   ($x) is not one less than CFG index ($x)`.
2. **Bare `GlobalRef`s to non-`Core`/`Base` modules can't be embedded as values.** A `GlobalRef`
   used directly as an *operand* (not a call's callee — as a literal value, e.g. inside a `%new`
   or as the display-callee of an `:invoke`) is rejected by `verify_ir` with `Unbound or
   partitioned GlobalRef not allowed in value position`, unless its module is `Core`/`Base` or its
   binding is proven constant across the IR's valid worlds. `GlobalRef(ADNext, :frule)` and a
   primal callee like `GlobalRef(Main, :sin)` both trip this. Fix: resolve to the actual (stable,
   singleton) value first — `_calleeval(fpos)` for a primal callee, or reference the helper
   functions directly as Julia values (the engine binds `fruleg = frule`, `zerotang_g =
   zero_tangent`, `buildtang_g = build_tangent`, `gettfield_g = get_tangent_field` for exactly this
   reason) — and embed *that*, not the `GlobalRef`.
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

If you need to see the raw IR to debug a `verify_ir` failure, don't fight `@verify_error`'s silent
string-only branch — call `code_dual_ircode(f, argtypes)` from `reflection.jl` and `println` the
resulting `IRCode` yourself (or temporarily add a debug print right before the `verify_ir` call in
`dualize_to_ircode`), rather than trying to coax a dump out of `verify_ir` itself.

## Dynamic dispatch (`apply_generic`) — supported via the runtime `dynamic_frule` dispatcher

A call whose callee or an argument has a non-concrete declared type (e.g. `x + y[]` where `y =
Ref{Any}(...)`, always inferred `Any` regardless of what it holds) is a genuine dynamic dispatch: the
method actually invoked depends on runtime types unknowable at dualization time. It can't be wrapped
statically — a `%new` of a `Dual` would freeze the abstract declared type into the object, and
`frule`'s `@generated` dispatch, keyed on exactly those type parameters, could never resolve a
concrete primal from `Dual{Any,…}`. So `frule_split!` (`forward_interp.jl`) detects the case and
**defers it to run time**:

- It emits a *static* `Expr(:call, dynamic_frule, fcallee, ftang, ptuple, ttuple)`, where
  `ptuple`/`ttuple` are `Core.tuple` calls collecting the argument primals and tangents and `ftang`
  is the callee's own tangent. The result is typed `Any`; the primal/tangent are pulled back out with
  `getfield(_, 1)` / `getfield(_, 2)` (typed `R` / `tt(R)` — `tt` widens to `Any` for abstract `R`).
- `dynamic_frule` is an **ordinary function** (no `@noinline`, no `@constprop :none`, no
  `invokelatest` — none of those are load-bearing) that calls `frule(Dual(f, tf), map(Dual, primals,
  tangents)...)`. The `map` is the whole trick: at run time the values are concrete, so `map(Dual, …)`
  builds concrete `Dual`s (params inferred from the actual values) and `frule` resolves the real
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
  non-concrete. The compiled `frule` `CodeInstance` returns a concrete `Dual{Rc,Tc}`, which is `<:
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

## Where to go next

- General orientation / why any of this exists: the `adnext-architecture` skill.
- Adding support for a brand-new IR construct (the next one being exception handling): the
  `adnext-extending-ir-support` skill.
