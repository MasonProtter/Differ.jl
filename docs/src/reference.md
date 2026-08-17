# API Reference

This page documents Differ's public API, grouped by topic rather than dumped alphabetically.
`using Differ` re-exports everything below; the individual sub-packages
(`Contextual`/`DifferCore`/`DifferForwards`/`DifferReverse`) also export it directly if you only
depend on one of them.

Each package has plenty of internal, non-exported functions with their own docstrings (accessible
via `?somename` in the REPL) that aren't listed here — they're implementation detail, not part of
the API surface. See `Differ/.claude/skills/` (`differ-architecture` and friends) for a guided tour
of those internals.

```@docs
Differ
```

## Tangents, fdata, and rdata (`DifferCore`)

The primal-type-keyed tangent system shared by both AD modes. Every primal type `P` has exactly
one tangent type `tangent_type(P)`, which splits further into an **fdata** half (threaded forward
and incremented in place) and an **rdata** half (only accumulable during a reverse pass) —
`DifferReverse`'s `CoDual` carries the fdata half; `DifferForwards`'s `Dual` carries the whole
tangent, since forward mode has no need for the split.

```@docs
NoTangent
Tangent
MutableTangent
PossiblyUninitTangent
NoFData
FData
NoRData
RData
LazyZeroRData
tangent_type
fdata_type
rdata_type
zero_tangent
zero_rdata
randn_tangent
fdata
rdata
tangent
build_tangent
get_tangent_field
set_tangent_field!
increment!!
set_to_zero!!
as_tangent
unit_tangent
```

## Forward mode (`DifferForwards`)

`Dual` pairs a primal value with its full tangent; `frule!!` is the forward-mode rule, dispatched
on `Dual` arguments, with an `@generated` fallback that derives a rule for any composite function
from its IR.

```@docs
Dual
primal
frule!!
code_dual_ircode
@code_dual_ircode
```

## Reverse mode (`DifferReverse`)

`CoDual` pairs a primal value with its fdata; `rrule!!` is the reverse-mode rule — hand-written
primitives and an `@generated` derived fallback are both methods of the same generic function.
`Ctx` (built via `build_ctx`) carries the pullback `Tape` between calls so it can be reused instead
of reallocated. `rev_gradient`/`rev_gradient!` are `public` rather than `export`ed — reach them as
`DifferReverse.rev_gradient(...)`, since DifferentiationInterface.jl is the primary user-facing
entry point now.

```@docs
CoDual
rrule!!
AbstractCtx
Ctx
build_ctx
value_and_gradient!
DifferReverse.rev_gradient
DifferReverse.rev_gradient!
zero_fcodual
tape_type
comms_element_types
code_reverse_fwds_ircode
@code_reverse_fwds_ircode
code_reverse_pullback_ircode
@code_reverse_pullback_ircode
```

## DifferentiationInterface.jl integration

Dispatch structs selecting Differ's forward/reverse mode through
[DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) — the
recommended way to call into Differ; see [the Home page](index.md#Using-via-DifferentiationInterface.jl)
for usage examples.

```@docs
AutoDifferForwards
AutoDifferReverse
```

## Compiler-plugin machinery (`Contextual`)

Mode-agnostic machinery both AD engines are built on top of: a generic `AbstractInterpreter` that
lets a plugin swap in a transformed `IRCode` for a method during compilation, with no AD logic of
its own. Relevant if you're extending Differ's engines or building a similar IR-transforming
compiler plugin; not needed to just use forward/reverse mode.

```@docs
ContextualInterpreter
build_contextual_ir
run_ipo_passes!
expr_to_codeinfo
at_world
mt_edge!
```
