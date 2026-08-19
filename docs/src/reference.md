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

## Activity marking (`DifferCore`)

`Inactive` marks a value the caller holds constant: no derivative is propagated to or from it. It
is a legal inhabitant of a shadow slot alongside the primal-derived one — `fdata_shadow_type(P)`
gives that legal set for reverse mode's `CoDual`, `tangent_shadow_type(P)` for forward mode's `Dual`
(which carries the whole tangent rather than the fdata half). `shadow_type` is a deprecated alias for
the former. `Inactive` is distinct from `NoTangent`, which says a *type* has no tangent space rather
than that a particular value was declared constant; see the `Inactive` docstring for why the two
can't be merged.

Both modes carry it. In reverse mode a `CoDual`'s shadow can be `Inactive`, and [`build_ctx`](@ref)
accepts it both via its `inactive=` positional-index form and by reading it straight off `CoDual`
carriers. In forward mode a `Dual`'s tangent slot can be `Inactive` — `frule!!(…, Dual(w,
Inactive()))` — and [`code_dual_ircode`](@ref) takes the same `inactive=` position list. Everything
reachable only through constants is replayed primally with no tangent computed at all; a constant
read by an active computation gets a zero materialised at its definition, so hand-written rules see
an ordinary tangent. Widening the forward hand rules to take `Inactive` directly is future work.

```@docs
Inactive
fdata_shadow_type
tangent_shadow_type
shadow_type
isactive
@ifactive
```

## Forward mode (`DifferForwards`)

`Dual` pairs a primal value with its full tangent — or with [`Inactive`](@ref) for an argument the
caller holds constant; see [Activity marking](#Activity-marking-(DifferCore)). `frule!!` is the
forward-mode rule, dispatched on `Dual` arguments, with an `@generated` fallback that derives a rule
for any composite function from its IR.

```@docs
Dual
primal
frule!!
code_dual_ircode
@code_dual_ircode
```

## Reverse mode (`DifferReverse`)

`CoDual` pairs a primal value with its fdata — or with [`Inactive`](@ref) for an argument the
caller holds constant; see [Activity marking](#Activity-marking-(DifferCore)). `rrule!!` is the
reverse-mode rule — hand-written primitives and an `@generated` derived fallback are both methods
of the same generic function. `Ctx` (built via `build_ctx`) carries the pullback `Tape` between
calls so it can be reused instead of reallocated. `rev_gradient`/`rev_gradient!`/`fcodual_type`/
`codual_type` are `public` rather than `export`ed — reach them as `DifferReverse.rev_gradient(...)`,
etc. — since DifferentiationInterface.jl is the primary user-facing entry point now.

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
DifferReverse.fcodual_type
DifferReverse.codual_type
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
