---
name: differ-tangent-system
description: Deep-dive reference for DifferCore, the primal-type-keyed tangent/fdata/rdata type system shared by DifferForwards and DifferReverse — tangent_type/fdata_type/rdata_type, Tangent/MutableTangent/PossiblyUninitTangent, FData/RData, zero_tangent/increment!!/set_to_zero!!, build_tangent/get_tangent_field/set_tangent_field!, and the array-tangent and shared IR-inspection helpers that live alongside them. Use this before modifying anything in DifferCore/src/, when a tangent_type/fdata_type/rdata_type result looks wrong, when adding tangent support for a new primal type, or when tracing how Dual/CoDual get their tangent representation.
---

# DifferCore: the tangent/fdata/rdata system

DifferCore is the primal-type-keyed tangent/fdata/rdata type system shared by both AD modes. It
has zero forward- or reverse-mode-specific logic: no `Dual`, no `CoDual`, no `frule!!`/`rrule!!`,
no IR transformation. `DifferForwards` (`Dual`) and `DifferReverse` (`CoDual`) both depend on
`DifferCore` and build their carriers on top of it. Ported from Mooncake.jl — `../Mooncake.jl` is
the reference for anything that looks underspecified here.

Start with the `differ-architecture` skill for overall orientation; this skill is the deeper dive
on the tangent system specifically. The deep dives on how each engine consumes this system belong
in `differ-forward-dualization` and `differ-reverse-engine` (not yet written).

## File map (`DifferCore/src/`)

Include order, from `DifferCore.jl`: `tangent_utils.jl` → `tangents.jl` → `fwds_rvs_data.jl` →
`inactive.jl` → `array_tangents.jl` → `shared_ir_helpers.jl`.

| File | Contents |
|---|---|
| `tangent_utils.jl` | Generic helpers the port needs: `@foldable`, `_typeof`, `tuple_map`, `tuple_fill`, `_findall`, `stable_all`, `_map`, `_map_if_assigned!`, `always_initialised`/`is_always_initialised`/`is_always_fully_initialised`, `_new_`, boxed-error printing, base `_copy` methods. |
| `tangents.jl` | `NoTangent`, `PossiblyUninitTangent`, `Tangent`, `MutableTangent`, `tangent_type(P)`, `zero_tangent`/`randn_tangent`/`increment!!`/`set_to_zero!!`/`_scale`/`_dot`/`_add_to_primal`, `build_tangent`/`get_tangent_field`/`set_tangent_field!`, `as_tangent`/`unit_tangent`, `tangent_to_primal!!`/`primal_to_tangent!!`, `require_tangent_cache`. |
| `fwds_rvs_data.jl` | The fdata/rdata split: `NoFData`/`FData`/`NoRData`/`RData`, `fdata_type`/`rdata_type`, `fdata`/`rdata`/`tangent(f,r)`, `zero_rdata`/`LazyZeroRData`/`ZeroRData`. |
| `inactive.jl` | `Inactive` (a value held constant), `shadow_type`, `isactive`/`@ifactive`, and the `Inactive` arms of every operation above. Included after the two files it extends. |
| `array_tangents.jl` | Element-wise `Array` tangent *value* ops (`zero_tangent_internal`, `increment_internal!!`, etc. specialised to `Array`). |
| `shared_ir_helpers.jl` | Small, mode-agnostic IR-inspection helpers used by both `DifferForwards` and `DifferReverse`'s own dualization/pullback engines. No tangent-system logic of its own. |

`DifferCore.jl` also declares `function primal end` — a bare stub with no methods. `DifferForwards`
adds a `primal(::Dual)` method and `DifferReverse` adds `primal(::CoDual)`; `tangent` works the
same way (the `tangent(f, r)` merge function lives here, `tangent(::Dual)`/`tangent(::CoDual)` are
added downstream). This is how one generic function name serves both carriers without either mode
package depending on the other.

Exports: `tangent_type`, `fdata_type`, `rdata_type`; `Tangent`, `MutableTangent`,
`PossiblyUninitTangent`, `NoTangent`; `NoFData`, `NoRData`, `FData`, `RData`; `fdata`, `rdata`,
`tangent`, `primal`, `zero_tangent`, `zero_rdata`, `randn_tangent`; `increment!!`, `set_to_zero!!`;
`build_tangent`, `get_tangent_field`, `set_tangent_field!`; `as_tangent`, `unit_tangent`;
`LazyZeroRData`; `Inactive`, `shadow_type`, `isactive`, `@ifactive`.

## `Inactive`: a value held constant

`tangent_type(P)` is one type per primal type, but *activity* is a property of a value, so a shadow
slot has one more legal inhabitant:

```julia
shadow_type(P) = Union{fdata_type(tangent_type(P)), Inactive}
```

**A validity predicate, never a declaration.** Use it in `<:` checks and rule-signature constraints
only. Every declared slot, field, comms item and SSA type stays concrete — the engine picks the
concrete alternative from its own per-value activity (`_shadow_types`, `differ-reverse-engine`), so
no union reaches a hot path.

**Why not `NoTangent`.** `NoTangent` already means "this *type* has no tangent space", and it is
closed under the split the wrong way: `fdata_type(NoTangent) === NoFData`, which is also an active
`Float64`'s fdata. So `NoTangent` in a shadow slot cannot say "constant" once a value is nested, and
`isactive` would not be decidable from the type. `Inactive` is preserved by both halves —
`fdata(Inactive()) === rdata(Inactive()) === Inactive()` — so it survives into an aggregate's shadow.

**Bare singleton, no type parameter**, matching every other marker here (`NoTangent`, `NoFData`,
`NoRData`, `ZeroRData`). The one primal-keyed type, `LazyZeroRData{P,Tdata}`, is keyed because it
must *reconstruct* `zero_rdata_from_type(P)` with no value in hand; `Inactive` never reconstructs
anything, and anywhere you would materialise you have the primal from the carrier (which carries the
size a type cannot).

**Strong zero, asymmetrically.** Both arms follow the general rule that `increment!!` returns a value
of the accumulator's (first argument's) type:

```julia
increment_internal!!(::IncCache, ::Inactive, _)          = Inactive()   # accumulate into a constant → discard
increment_internal!!(::IncCache, x, ::Inactive)          = x            # a constant contributes nothing
increment_internal!!(::IncCache, ::Inactive, ::Inactive) = Inactive()   # disambiguator, else ambiguous
```

So **`increment!!` is not commutative** in the presence of `Inactive`: slot 1 is the accumulator that
owns storage, slot 2 is a contribution. Any site that swaps argument order would silently drop a live
gradient (all current call sites are accumulator-first — audited).

The strong-zero property is *structural, not numerical*: `Inactive` short-circuits before any
arithmetic, so `dx += y * dz` never evaluates with an infinite `y`. A materialised zero buffer would
give `NaN` there, which is a correctness argument for this representation and not only a performance
one.

`NoTangent` deliberately keeps **no** absorbing arm, so a mis-analysed active value still raises a
`MethodError` rather than dropping a gradient. Do not add one.

Two consequences worth knowing:

- The `increment!!`/`increment_internal!!` entry points are homogeneously typed
  (`increment!!(x::T, y::T)`), so the `Inactive` arms are *heterogeneous* methods — a deliberate
  loosening of an invariant the rest of the system enforces.
- `increment_internal!!(c, x::Tuple, y::Tuple)` accumulates two tuples of *different* types
  elementwise. A caller builds its seed from the primal type, so the seed is wider than a
  mixed-activity aggregate's shadow; accumulation across aggregates has to be structural rather than
  type-equality-gated.

`@ifactive(dx, expr)` returns `NoRData()` (not `Inactive()`) on the inactive arm: activity has to be
visible in the *fdata* half, where a primal-derived declaration would otherwise be a type lie; an
inactive argument's rdata slot carries nothing either way, and `NoRData()` is what the derived path
emits, so hand rules and derived rules return the same shape. `tangent(Inactive(), r)` is `Inactive()`
whatever `r` is.

## `tangent_type(P)`: the central function

Each primal type `P` has exactly one tangent type, `tangent_type(P)`. This is a hard invariant the
whole system relies on (`Dual{P,T}`'s inner constructor checks `tangent_type(P) == T`). Shapes,
keyed on what kind of primal `P` is:

| Primal shape | `tangent_type(P)` |
|---|---|
| Non-differentiable / singleton (`Int`, `Bool`, `Char`, `Symbol`, `String`, `Module`, `Nothing`, `Core.Builtin`, compiler-internal types like `Core.MethodInstance`) | `NoTangent` |
| `P <: Base.IEEEFloat` (`Float32`/`Float64`/`Float16`) | `P` itself (self-tangent) |
| `Ptr{P}` | `Ptr{tangent_type(P)}`; bare/abstract `Ptr` → `NoTangent` |
| `Array{P,N}` | `Array{tangent_type(P),N}`; non-concrete `Array{P,N} where P` → bare `Array` |
| `MemoryRef{P}` / `Memory{P}` | `MemoryRef{tangent_type(P)}` / `Memory{tangent_type(P)}` — only for the default `:not_atomic`, CPU `GenericMemoryRef`/`GenericMemory` aliases (see caveat below) |
| `Tuple{...}` / `NamedTuple` | Per-field tuple of tangent types, recursively; collapses to `NoTangent` if every field is non-differentiable; splits into a `Union` if exactly one field type is itself a `Union` |
| Immutable `struct` | `Tangent{NamedTuple{names,Tfields}}`, one field per primal field (a maybe-undefined field's tangent type is wrapped in `PossiblyUninitTangent`); collapses to `NoTangent` if all fields are non-differentiable |
| `mutable struct` | `MutableTangent{NamedTuple{names,Tfields}}`, same shape as `Tangent` but mutable |
| Abstract or non-concrete `P` | `Any` (tangent type not knowable until runtime) |

Sharp edge (documented inline in `tangents.jl`, not elsewhere): a `GenericMemoryRef`/
`GenericMemory` with a non-default `Kind`/`AddrSpace` (e.g. atomic memory) does **not** match the
`MemoryRef{P}`/`Memory{P}` methods above and falls through to the generic struct-derivation method
instead, which is wrong for these types. This is a known gap, not a designed fallback.

`backing_type(P)` gives the `NamedTuple` type used to store a struct's per-field tangents.
`build_tangent(::Type{P}, fields...)` constructs a full `Tangent`/`MutableTangent` for `P` from
positional field-tangent values, wrapping `PossiblyUninitTangent` around any field that needs it.
`get_tangent_field(t, i_or_sym)` / `set_tangent_field!(t, i_or_sym, x)` are the `getfield`/
`setfield!` equivalents for `Tangent`/`MutableTangent`, transparently unwrapping/wrapping
`PossiblyUninitTangent`.

## The fdata/rdata split (`fwds_rvs_data.jl`)

Rules don't operate on a full tangent directly. Instead every tangent type splits into two halves:

- **fdata** (forwards-pass data) — the part of a tangent identified by *address* (a
  `MutableTangent`, an `Array`/`MemoryRef`/`Memory`, a `Ptr`). This is threaded through the
  forward pass and incremented **in place** on the reverse pass, because address-identified
  data doesn't need reverse-pass accumulation machinery — it's just mutated where it lives.
- **rdata** (reverse-pass data) — the part identified by *value* (a `Float64`, a field of an
  immutable struct). This has no fixed address, so it can only be accumulated during the
  reverse pass proper.

`fdata_type(T)`/`rdata_type(T)` take a *tangent* type (usually `tangent_type(P)`) and return the
corresponding fdata/rdata type. `NoFData`/`NoRData` are the "nothing to propagate" singletons.
`FData{T<:NamedTuple}`/`RData{T<:NamedTuple}` wrap the non-trivial per-field case for an immutable
struct's tangent — structurally identical to `Tangent`, just used in a different context. Concrete
splits:

| Tangent | fdata | rdata |
|---|---|---|
| `NoTangent` | `NoFData` | `NoRData` |
| `Float64` (scalar) | `NoFData` | `Float64` |
| `Vector{Float64}` (array) | `Vector{Float64}` | `NoRData` |
| `MutableTangent{...}` | itself | `NoRData` (all data is address-identified) |
| `Tangent{NamedTuple{...}}` | `FData{...}` (or `NoFData` if every field is rdata-only) | `RData{...}` (or `NoRData` if every field is fdata-only) |

`fdata(t)`/`rdata(t)` extract the two halves from a tangent instance; `tangent(f, r)` is the
inverse — `tangent(fdata(t), rdata(t)) === t` must hold for every valid tangent `t`. This package
also has a second, unrelated overload of `tangent_type`: `tangent_type(F, R)` reconstructs a
tangent type from an fdata type and rdata type (the inverse direction), needed where DifferReverse
builds a tangent type from a fdata/rdata pair without having the original primal type handy.

`zero_rdata(p)` computes the zero rdata element directly from a primal value, without building a
full tangent first. `can_produce_zero_rdata_from_type(P)`/`zero_rdata_from_type(P)` ask whether
that zero element is derivable from the *type* `P` alone (no runtime value needed) — this matters
for reverse mode, which wants to skip computing a zero rdata value at trace time whenever the type
determines it. `LazyZeroRData{P,Tdata}` is the corresponding lazy placeholder: it defers actually
building the zero rdata until `instantiate` is called on the reverse pass, storing only what it
must (nothing at all for a `Float64`, since its zero is `0.0` unconditionally — `lazy_zero_rdata`
picks the minimal representation via `lazy_zero_rdata_type`). `ZeroRData`/`zero_like_rdata_type`/
`zero_like_rdata_from_type` are a further fallback used only when a value's runtime type isn't
fully known (an abstract field or non-const global) at the point a zero rdata is needed.

`verify_fdata_type`/`verify_fdata_value` and `verify_rdata_type`/`verify_rdata_value` are runtime
consistency checks — not on any hot path, used for debugging/assertions — that raise
`InvalidFDataException`/`InvalidRDataException` on mismatch.

## Mutation / construction API both engines call into

| Function | Purpose |
|---|---|
| `zero_tangent(x)` | The unique zero element of `x`'s tangent space. Handles circular references/aliasing via an `IdDict` cache, but only when `require_tangent_cache(typeof(x))` says one is needed (not a blanket `isbitstype` check — see below). `zero_tangent(x::Ptr)` throws; use the two-arg `zero_tangent(primal, fdata)` form instead (safe because it derives the rdata half via `zero_rdata`, whose `Ptr` answer is always `NoRData`). |
| `randn_tangent(rng, x)` | Same shape as `zero_tangent`, randomly-chosen elements. Testing only. |
| `increment!!(x, y)` | Add tangent `y` into `x`; mutates in place when `T` is mutable (`increment!!(x,y) === x`). Same cache-decision logic as `zero_tangent`. |
| `set_to_zero!!(x)` | Zero a tangent in place. Uses a more permissive, purely-perf cache decision than `require_tangent_cache` (zeroing is idempotent, so the cache is never needed for correctness — see `_set_to_zero_cache`). |
| `_scale(a::Float64, t)` | Multiply a tangent by a scalar — every tangent type is a vector space. Testing/gradient-checking use. |
| `_dot(t, s)::Float64` | Inner product between two tangents of the same type. Testing/gradient-checking use. |
| `_add_to_primal(p, t, unsafe=false)` | Add tangent `t` to primal `p`, returning a new `P`. `unsafe=true` bypasses `P`'s constructor via the `:new` instruction directly — needed when `P` has no default constructor matching its field values, at the cost of skipping any invariants the real constructor enforces. |
| `build_tangent(::Type{P}, fields...)` | Construct a `Tangent`/`MutableTangent` for `P` from field tangent values. |
| `get_tangent_field`/`set_tangent_field!` | `getfield`/`setfield!` equivalents for `Tangent`/`MutableTangent`. |
| `increment_field!!(x, y, f)` / `increment_field_rdata!(dx, dy_rdata, f)` | Field-level increment helpers for getfield-like rules; `Val`-dispatch for a static field, plain `Int` for a dynamic runtime index into a homogeneously-typed aggregate. |
| `as_tangent(x)` / `unit_tangent(x)` | Project `x` (resp. `oneunit(x)`) into its own tangent space, via `primal_to_tangent!!`. |

**`require_tangent_cache(::Type{P})`** is the single authority both `zero_tangent` and
`increment!!` (and, more permissively, `set_to_zero!!`) consult to decide whether an aliasing/
circular-reference cache is needed at all. It is deliberately *not* a bare `isbitstype(P)` check:
that would allocate an `IdDict` for every non-bits tangent, including provably tree-like ones like
`Vector{<:IEEEFloat}`, which was a measured allocation on every `rev_gradient` call. The default is
the conservative `Val{!isbitstype(P)}()`; `Array{P}` gets a more precise override (`Val{false}`
whenever `tangent_type(P) === NoTangent`, since an array of non-differentiable elements can't
alias or cycle through its tangent either). Only override this after proving a tangent type's
memory layout is tree-like — the docstring in `tangents.jl` has worked examples of both a
circular-reference case (`Ref{Any}`) and an aliasing case (two tuple fields pointing at the same
mutable object) that require the cache.

**`tangent_to_primal!!(primal, tangent)`/`primal_to_tangent!!(tangent, primal)`** convert between
a primal value and its tangent in place where possible. Note: `tangent_to_primal!!`'s own
docstring says it "will be removed in the next breaking release (0.6)... retained solely for
backward compatibility with downstream packages" — it is present and still used internally (by
`primal_to_tangent!!`, which backs `as_tangent`/`unit_tangent`), but is a known future-removal
candidate, not a stable long-term API.

**Correction versus the old `differ-architecture` file-map bullets**: those bullets describe a
`FriendlyTangentCache`/`tangent_to_friendly!!` user-facing machinery living in `tangents.jl`. No
such names exist anywhere in `DifferCore` (verified by grep — this is a stale carryover, likely
from an early Mooncake port draft or a renamed API that never shipped in this form). The actual
current surface for primal↔tangent conversion is `as_tangent`/`unit_tangent` plus
`tangent_to_primal!!`/`primal_to_tangent!!`, all described above.

## `array_tangents.jl` scope

Element-wise `Array` tangent *value* operations only: `zero_tangent_internal`,
`randn_tangent_internal`, `increment_internal!!`, `set_to_zero_internal!!`, `_scale_internal`,
`_dot_internal`, `_add_to_primal_internal`, and `tangent_to_primal_internal!!`/
`primal_to_tangent_internal!!`, all specialised to `Array{P,N}`. These build a plain
`Array{tangent_type(P),N}` tangent element-wise, aliasing/circular-reference-aware via the same
cache convention as the generic scalar/struct implementations in `tangents.jl`.

Deliberately **not** ported from Mooncake, still true as of this split:

- Mooncake's `Memory`/`MemoryRef`-internals array path (its `src/rules/memory.jl`), which is fused
  with Mooncake's reverse-mode primitive-rule system (`frule!!`/`rrule!!`/`@is_primitive`) — out of
  scope for `DifferCore`, which has zero rule-system logic of its own. `tangent_type(MemoryRef{P})`/
  `tangent_type(Memory{P})` (type-level bookkeeping only, needed so `DifferForwards`'s array
  builtins can type their shadow SSAs) live in `tangents.jl` instead, next to the `Array`/`Ptr`
  methods — they don't conflict with this file's scope, they're just type-level rather than
  value-level.
- Mooncake's `AbstractDict`/`AsPrimal` "friendly" tangent path — confirmed absent (no `AsPrimal` or
  `AbstractDict` reference anywhere in `DifferCore`).

## `shared_ir_helpers.jl`

Genuinely mode-agnostic IR-inspection helpers used by both `DifferForwards`' and `DifferReverse`'s
own dualization/pullback engines — moved here specifically because both packages need them and
neither owns them. No tangent/fdata/rdata logic; it's a grab-bag of small IR utilities:

- `_globalref_val`/`_globalref_isconst` — resolve a `GlobalRef` to its bound value / ask whether
  its binding is a defined constant, both at an explicit inference `world` (never the ambient task
  world, since dualization runs against IR compiled at a specific, possibly-past world).
- `_calleeval` — resolve an IR operand to its statically-known value (`GlobalRef`/`QuoteNode`
  literal), or `nothing` for an `SSAValue`/`Argument`.
- `_optype`/`_optype_w` — the primal IR's declared type for an operand, exact (`_optype`) or
  widened to a bare `Type` (`_optype_w`, resolving `GlobalRef`/`QuoteNode` to the value's type).
- `_stmt_str` — compact single-line rendering of an IR statement, used in bail-reason error
  messages so a user sees *what* IR construct wasn't handled.
- `_bi_literal_index` — whether a `getfield`/`setfield!` name/index operand is already a
  compile-time literal.
- `_bi_homog_tangent_type` — the common tangent type shared by every field of a concrete struct
  type, or `nothing` if the fields don't all agree; used to allow a dynamic (runtime-computed)
  field index into a homogeneous aggregate.
- `_tangent_field_slot` — for a literal-named field access, the tangent-backing `NamedTuple` type
  and 1-based field index to emit direct field access against (or `nothing` if the direct-emission
  preconditions don't hold, e.g. non-concrete primal or a `PossiblyUninitTangent` slot).
- `_widen` — widen a lattice element (`Core.Const`/`Core.PartialStruct`/...) to a real `Type`.
- `_getfieldg`/`_setfieldg`/`_ctupleg` — shared `GlobalRef` constants for `Core.getfield`/
  `Core.setfield!`/`Core.tuple`, the builtins both engines special-case when reconstructing IR.

This matches the module-header comment in `DifferCore.jl` almost verbatim — it is accurate as of
this split, not stale.

## How the two AD engines use this

`DifferForwards` builds its `Dual{P,T}` carrier directly on `tangent_type`: the invariant
`T == tangent_type(P)` is enforced in `Dual`'s own constructor, and forward mode's dualization
engine calls `zero_tangent`/`increment!!`/`get_tangent_field`/`set_tangent_field!` etc. as it
mirrors primal IR onto shadow IR. `DifferReverse`'s `CoDual{Tx,Tdx}` carries a primal alongside its
**fdata** half (`Tdx` is `fdata_type(tangent_type(Tx))`, not the full tangent) — reverse mode
threads fdata forward through the trace and only reconstructs full tangents (via `tangent(f, r)`)
or extracts rdata contributions (via `rdata`/`increment_rdata!!`) on the pullback side, which is
exactly the fdata/rdata split's reason for existing. Neither engine reimplements any of this
logic; both are pure consumers of the `DifferCore` API described above.
