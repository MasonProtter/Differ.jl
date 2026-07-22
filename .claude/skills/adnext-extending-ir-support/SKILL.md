---
name: adnext-extending-ir-support
description: Playbook for extending ADNext's dualization engine to support a new Julia IR construct — the methodology used to add control flow (branches/loops) and exception handling (try/catch), and what's known about the next gap, array / memoryref indexing. Use this when asked to make ADNext handle more Julia language features, when dualize_to_ircode bails on something new, or when planning array-indexing support.
---

# Extending ADNext's dualization engine to a new IR construct

`dualize_to_ircode` (`src/forward_interp.jl`) bails — returns `nothing`, which surfaces as a clear
`ErrorException` rather than a miscompile — on any IR construct it doesn't handle yet. This is a
playbook for closing one of those gaps, distilled from actually doing it for control flow
(branches/loops). Read the `adnext-ircode-dualization` skill first for how the engine currently
works; this skill is about the *process* of growing it further.

## Methodology

1. **Get the real IR shape before designing anything.** Write a minimal Julia function that
   exercises the construct and dump its fully-optimized `IRCode` with
   `Base.code_ircode(f, argtypes)[1]` (or `first`). Don't design against documentation or memory
   alone — invariants here are subtle and genuinely surprising even when you think you know the
   rules. Concrete example from the control-flow work: a plain nested-`while`-loop function turned
   out to contain a block whose *only* statement is a bare `nothing` placeholder (a trivial
   fallthrough Julia's optimizer leaves between adjacent loops) — nothing in the docs predicts
   that, but it broke the transform until handled explicitly (see the empty-block gotcha in the
   `adnext-ircode-dualization` skill).

2. **Read `Core.Compiler.verify_ir`'s actual checks** for the construct, in
   `Compiler/src/ssair/verify.jl` (this checkout has it at `julia/Compiler/src/ssair/verify.jl`).
   This is the authoritative source of legality invariants — e.g. exactly which basic-block
   numbering convention a field uses, what "the top of a block" means for placement rules, whether
   forward references are legal and how dominance is checked for them. Design the transform to
   satisfy these checks from the start rather than discovering them by trial and error.

3. **Narrow the bail list.** Bails are `return nothing` in the main `if`/`elseif` chain (which the
   caller turns into a clear `ErrorException`). The live ones today are: any un-ruled `Core.Builtin`
   (the `elseif isa(f, Core.Builtin)` arm — this is where array/`memoryref` indexing currently
   lands), plus the `anything else` fallthrough. Add a dedicated branch for the construct you're
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
   walk. The `pending` dict pattern (documented in `adnext-ircode-dualization`) is the general
   solution: don't invent a second mechanism if this one applies.

6. **Preserve or rebuild `StmtRange`/CFG bookkeeping as appropriate.** If your construct (like
   control flow) doesn't change block topology, reuse the existing `block_start_new` tracking and
   remember the empty-block backfill (see gotcha #3 in `adnext-ircode-dualization`). If it *does*
   need new blocks or edges that don't exist in the primal (exception handling's implicit
   handler-entry edges are a candidate for this), that's a bigger design decision — don't assume
   the 1:1-preservation trick still applies without checking.

7. **Call `Core.Compiler.verify_ir` and let it throw.** It's already wired in unconditionally at
   the end of `dualize_to_ircode` — don't remove or weaken that. Treat any failure while developing
   as a bug in your new code, not a reason to catch-and-bail.

8. **Add tests in both dimensions**: (a) real functions exercising the construct, checked against
   an independent reference (finite differences, or a hand-computed analytic derivative) *and*
   against `Core.Compiler.verify_ir` not throwing on the raw dualized IR (via `code_dual_ircode`);
   (b) a regression test that constructs still outside scope keep bailing with a clear
   `ErrorException` rather than silently miscompiling.

## Exception handling (try/catch) — implemented; kept here as a worked reference

`try`/`catch` is now **supported and tested** (it was the milestone this playbook was first written
for). The IR-shape notes below are how it works today; they're a good concrete example of applying
the methodology above. The *next* gap is array / `memoryref` indexing (see the final section).

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
off of. Expect the "does this need a primal-copy + shadow-copy" question from step 4 above to
applies to `UpsilonNode`/`PhiCNode` the same way it did to `PhiNode`, and that's exactly how it was
implemented: a shadow `UpsilonNode` at each capture point (typed `tangent_type(Ti)`), feeding a
shadow `PhiCNode` in the handler, reusing the same `pending` forward-ref mechanism.

The exception object itself (`Expr(:the_exception)`) has no meaningful tangent — its shadow is the
zero tangent of its type (`zero_shadow`, i.e. `NoTangent()` / a literal `zero` / a runtime
`zero_tangent`), the same treatment non-differentiable intrinsic results get.

Also worth knowing: the optimizer already eliminates `try`/`catch` scopes it can prove unreachable
(in `Compiler/src/optimize.jl`), so post-optimization IR handed to `dualize_to_ircode` may have
simpler exception structure than the source suggests — don't assume every source-level `try` shows
up as a live `EnterNode`.

## The next gap: array / memoryref indexing

`v[i]` (and mutating a container) currently **bails**: on Julia 1.13 it lowers to `memoryref` /
`memoryrefget` / `memoryrefset!` builtins plus `Expr(:boundscheck)` on the live path, which hit the
"un-ruled `Core.Builtin` → `return nothing`" arm. The tangent *type* system already covers arrays
(`tangent_type(Array{Float64}) == Array{Float64}`, and `array_tangents.jl` has the value ops), so
the work is in the engine: give those builtins dualization branches (an indexed read differentiates
like a per-element `getfield`; a write propagates into the shadow array). Mooncake's own
`src/rules/memory.jl` is the reference for how the `Memory`/`MemoryRef` layer behaves, though its
approach is fused with the reverse-mode rule system rather than the split-shadow IR transform.

## Cross-references

- `adnext-architecture` — overall design and file map.
- `adnext-ircode-dualization` — how the currently-supported constructs are implemented, and the
  `verify_ir` gotchas you're likely to rediscover.
