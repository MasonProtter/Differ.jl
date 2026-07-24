---
name: adnext-architecture
description: Orients Claude to ADNext's overall design before making changes — the custom AbstractInterpreter plugin architecture that hooks Julia's compiler pipeline, the Dual/frule calling convention, and how the pieces in src/ fit together. Use this whenever working on ADNext (adding features, fixing bugs, reviewing code, explaining how it works), especially before touching contextual.jl, forward_interp.jl, frules.jl, or reflection.jl for the first time in a session.
---

# ADNext architecture

ADNext is a small, experimental **post-optimization IR-based automatic differentiator** for
Julia. Instead of transforming source code or a macro-expanded expression, it transforms a
function's *already fully-optimized* `Core.Compiler.IRCode` — the same IR Julia's own optimizer
produces right before codegen — and splices the transformed IR back into the compiler pipeline so
it gets treated as an ordinary compiled method. There is no separate "AD runtime": the derivative
code IS a normal `CodeInstance`, inlined and optimized exactly like anything else Julia compiles.

## File map (`src/`)

- **`ADNext.jl`** — module entry point; just `include`s the other files in dependency order
  (`tangent_utils.jl`, `tangents.jl`, `fwds_rvs_data.jl`, `array_tangents.jl`, `dual.jl`,
  `codual.jl`, `frules.jl`, `contextual.jl`, `forward_interp.jl`, `reflection.jl`) and exports the
  public API. (`reverse_interp.jl` — a WIP barebones reverse engine — is intentionally *not*
  included; its mutable `CoDual` is incompatible with the ported immutable one and reverse AD is
  out of scope.)
- **Tangent / fdata / rdata type system** (ported from Mooncake.jl — see `../Mooncake.jl` and its
  `understanding_mooncake/rule_system` docs for the authoritative reference):
  - **`tangent_utils.jl`** — helpers the port needs (`tuple_map`, `always_initialised`, `_new_`,
    `_copy`, boxed-error printing), plus no-op `@unstable`/`@stable` and the real `@foldable`.
  - **`tangents.jl`** — `NoTangent`, `PossiblyUninitTangent`, `Tangent` (immutable-struct tangent),
    `MutableTangent` (mutable-struct tangent), `tangent_type(P)` (keyed on the *primal* type),
    `zero_tangent`/`randn_tangent`/`increment!!`/`set_to_zero!!`/`_scale`/`_dot`/`_add_to_primal`,
    `build_tangent`/`get_tangent_field`/`set_tangent_field!`, and the user-facing
    `FriendlyTangentCache`/`tangent_to_friendly!!`/`tangent_to_primal!!` machinery.
  - **`fwds_rvs_data.jl`** — the fdata/rdata split: `NoFData`/`FData`/`NoRData`/`RData`,
    `fdata_type`/`rdata_type`, `fdata`/`rdata`/`tangent(f,r)`, `zero_rdata`/`LazyZeroRData`.
  - **`array_tangents.jl`** — element-wise `Array` tangent value ops (`zero_tangent` on arrays,
    etc.). (Mooncake's `Memory`/`MemoryRef` internals and the `AbstractDict`/`AsPrimal` friendly
    path live in its rules layer and were *not* ported.)
  - **`dual.jl`** — the forward carrier `Dual{P,T}` (`T == tangent_type(P)`), `dual_type`,
    `zero_dual`, the `d.x`/`d.dx` property aliases, the type-level field accessors
    `_dual_primal_type`/`_dual_tangent_type`, and the key `tangent_type(::Type{<:Dual}) = itself`.
  - **`codual.jl`** — the reverse carrier `CoDual{Tx,Tdx}` (`Tdx` is fdata), `codual_type`/
    `fcodual_type`, `zero_codual`/`zero_fcodual`, `NoPullback`. (Type infrastructure only — no
    reverse-mode rules yet.)
- **`frules.jl`** — hand-written `frule` methods for `sin`/`cos` only (see below for why no more).
  The `Dual`/`NoTangent`/`tangent_type`/zero-tangent machinery now lives in the tangent-system
  files above, not here.
- **`contextual.jl`** — the *mode-agnostic* compiler-plugin machinery: `ADInterpreter{M<:ADMode}`
  (a custom `AbstractInterpreter`) and the two pipeline seams it overrides, `finishinfer!` and
  `optimize`. Nothing forward-mode-specific lives here — a future `Reverse` mode would reuse this
  file unchanged.
- **`forward_interp.jl`** — all forward-mode-specific glue (`build_contextual_ir`,
  `is_dualized_impl`, `primal_of_impl`, `build_dual_ir`, `frule_codeinstance`, the `dualized_impl`
  carrier stub, the `@generated frule` fallback) **and** the dualization engine itself
  (`dualize_to_ircode`). See the `adnext-ircode-dualization` skill for a deep dive on that engine —
  this file is the one you'll touch most often, and it's dense.
- **`reflection.jl`** — `code_dual_ircode`/`@code_dual_ircode`, a debugging/reflection entry point
  that reproduces exactly what the compiler-plugin seam installs, without going through the full
  `@generated` machinery. Use this (not raw `frule` calls) whenever you need to *see* the dualized
  IR — e.g. `code_dual_ircode(f, (Float64,))` or `@code_dual_ircode f(1.0)`.
- **`test/runtests.jl`** — the test suite; also doubles as documentation-by-example of what's
  currently supported (each primal test function has a comment explaining what IR feature it
  exercises).

## The core trick: two pipeline seams, no CodeInfo, no rettype patching

`ADInterpreter{M}` is a normal `AbstractInterpreter` in every respect except two overridden hooks:

1. **`finishinfer!`** — the point where Julia's inference machinery is about to freeze a method's
   return type into its `CodeInstance`. ADNext calls the mode's `build_contextual_ir(interp, mi)`
   hook here, and if it returns a real `IRCode` (not `nothing`), stashes it in
   `interp.transformed_ir[mi]` and sets `me.bestguess = compute_ir_rettype(ir)` — so the *ordinary*
   codepath writes the correct return type, once, with no post-hoc patching.
2. **`optimize`** — where Julia would normally run its own optimization passes on inferred
   `CodeInfo`. ADNext instead installs the already-built `IRCode` from step 1, runs the IPO-safe
   passes on it (`run_ipo_passes!`: `compact!` → `ssa_inlining_pass!` → `compact!` → `sroa_pass!` →
   `adce_pass!` → `compact!`), and calls `finishopt!` directly.

**This is a hard project constraint, not a style preference**: an earlier attempt built a second
engine that dualized lowered `CodeInfo` and patched the `CodeInstance` return type after the fact.
That was explicitly rejected as "a mess" — never resurrect that shape. Transform IRCode only,
never CodeInfo, never patch rettype after the fact.

For forward mode specifically, `build_contextual_ir` recognizes a *carrier* `MethodInstance`
(`dualized_impl(dualargs::Dual...)`, whose `specTypes` encodes the dual signature) and asks
`build_dual_ir` to transform the corresponding *primal* method's already-optimized `IRCode` via
`dualize_to_ircode`. Any `MethodInstance` that isn't a `dualized_impl` specialization — or any
input `dualize_to_ircode` can't handle — flows through the ordinary, unmodified pipeline.

## The Dual/frule calling convention

- `Dual{P,T}` wraps a primal value `primal::P` and a tangent `tangent::T`, with the Mooncake
  invariant `T == tangent_type(P)`. So the tangent of a `Float64` is a `Float64`; of an `Int`/`Bool`/
  function/singleton is `NoTangent()`; of an immutable struct is a `Tangent{…}`; of a mutable struct
  a `MutableTangent{…}`; of a `Tuple` a per-field tangent tuple; of an `Array{Float64}` an
  `Array{Float64}`. (Note `Complex{Float64}` is a *struct* in this system, so its tangent is a
  `Tangent{@NamedTuple{re,im}}`, not another `Complex`.) `d.x`/`d.y`/`d.z` alias `primal`;
  `d.dx`/`d.dy`/`d.dz` alias `tangent` — pick whichever reads best. `primal(d)`/`tangent(d)` also work.
- To differentiate `f` at arguments `x, y, …` with tangents `dx, dy, …`, call
  `frule(Dual(f, tf), Dual(x, dx), Dual(y, dy), …)`. The *function itself* is wrapped in a `Dual`
  too — that's what lets `frule` dispatch on `Dual{typeof(f)}` to find hand-written rules.
- The **function carries a real tangent** `tf`, not a forced zero. Use `zero_tangent(f)` to hold `f`
  constant — that's `NoTangent()` for a plain/singleton function, but a `Tangent{…}` over the
  captures for a **closure**, letting you differentiate w.r.t. captured data (the capture is read as
  `getfield(#self#, field)`, whose tangent flows out of `tf` via `get_tangent_field`). Seed a field
  of that `Tangent` (e.g. `build_tangent(typeof(f), 1.0)`) to get ∂/∂capture.
- **Higher-order** = nested `Dual`s, applied *uniformly to every argument including the function*
  (Option A — an order-k request re-dualizes the order-(k-1) dualized function; `build_dual_ir`
  peels one `Dual` level off each arg). This is preserved cleanly because **a `Dual` is its own
  tangent type**: `tangent_type(::Type{<:Dual}) = itself` (defined in `dual.jl`), so after peeling a
  primal `Dual` the tangent is again a `Dual`. Order-2 seeds: value `Dual(Dual(x,dx),Dual(ddx,…))`;
  plain function `Dual(Dual(f,NoTangent()),Dual(f,NoTangent()))` (these two shapes are unchanged
  from before). `code_dual_ircode(f, argtypes; order)` builds these seeds (leaf `Dual{T,
  tangent_type(T)}`) for inspection/`verify_ir`.
- **Higher-order the natural way — composed differentiation (`D`-of-`D`)**: you can also write
  `D(f,x) = frule(Dual(f,zero_tangent(f)), Dual(x,one(x))).dx` and *nest* it (e.g. `D(x -> D(g, x), y)`)
  to get second and higher derivatives, instead of hand-building nested `Dual` seeds. This works
  because when the outer pass dualizes a closure whose body calls `frule`, that inner call survives in
  the primal IR as a `dualized_impl`/`frule` `:invoke`; `frule_split!` re-dualizes it, and
  `build_dual_ir`'s `compose(offset)` drops the leading non-nested *function slot*
  (`Dual{typeof(dualized_impl),NoTangent}` / `Dual{typeof(frule),NoTangent}`) and peels the remaining
  nested value args to the inner carrier. It recurses, so `D∘D∘D…` works to arbitrary order. **Limit:**
  a closure/struct with *differentiable fields* can't be differentiated at order ≥2 (the self-tangent
  `Dual` scheme needs each carried type to equal its own tangent type — true for scalars/arrays, false
  for such a struct whose tangent is a `Tangent`); it errors clearly rather than miscompiling.
- Only `sin`/`cos` have hand-written `frule` methods in `frules.jl`. Deliberately **no** rules
  exist for `+`, `-`, `*`, `/`, or comparisons: those inline down to intrinsics
  (`add_float`/`mul_float`/`lt_float`/…), which are handled by **dispatch** via `src/intrinsics.jl`.
  Each supported intrinsic has a thin wrapper function + an `frule`; the engine rewrites each
  intrinsic call to its wrapper and routes it through `frule_split!` like any surviving call.
  Registration is **explicit** — `translate` (intrinsic value → wrapper) has *no* identity fallback,
  so an unregistered intrinsic bails with a clear located error instead of silently miscompiling.
  Two macros register:
  - `@intrinsic name` + a hand-written `frule(::Dual{typeof(name)}, …)` for a **differentiable**
    intrinsic. Registered: `add/sub/neg/mul/div_float`, `sqrt_llvm`, `abs_float`, `max/min_float`,
    `fma_float`, `muladd_float`, `copysign_float`, `fpext`/`fptrunc` (float-width conversions), plus
    `_fast` variants.
  - `@inactive_intrinsic name` for a **non-differentiable** one — auto-generates a variadic `frule`
    computing the primal with a zero tangent. Registered: int/float comparisons, integer arithmetic
    (+ checked), bit/boolean ops, rounding (`floor/ceil/trunc/rint_llvm`), and int↔float / width
    conversions (`sitofp`, `fptosi`, …) whose leading *type* argument is carried through as a
    `Dual{DataType}`.

  This means arithmetic works for *any* type (`Float32`, `Complex`, a user struct via
  `getfield`/`%new`) without a per-type rule — don't add arithmetic `frule`s. To support a new
  intrinsic, add the appropriate line in `src/intrinsics.jl` (no engine change needed); a
  differentiable `frule` body must call the *wrappers* so it re-dualizes at higher order. The goal is
  for *every* intrinsic to be registered; the few still unregistered are exotic (pointer/atomic ops,
  `llvmcall`, `cglobal`, `bitcast`) and error loudly if hit rather than miscompiling.
- `frule(dualargs::Dual...)` itself is `@generated`: for a composite primal with no hand-written
  rule, the generator compiles a dualized version of the primal's body under
  `ADInterpreter{Forward}` and returns a trivial body that `invoke`s the resulting `CodeInstance`.

## Running things

- Tests: `julia +1.13 --project=ADNext -e 'using Pkg; Pkg.test()'` — this project targets Julia
  1.13, invoked via the `+1.13` juliaup channel selector, not whatever `julia` resolves to by
  default.
- Inspect dualized IR: `code_dual_ircode(f, (Float64,))` or `@code_dual_ircode f(1.0)` from
  `reflection.jl` — always prefer this over trying to reason about the transform from source
  alone when debugging.

## Current support status

Supported: straight-line code, branches and loops (`GotoNode`/`GotoIfNot`/`PhiNode`), throwing
error paths (a block ending in an unreachable `ReturnNode` is reconstructed primal-only), and
exception handling (`try`/`catch` via `EnterNode`/`PhiCNode`/`UpsilonNode`/`:leave`/
`:pop_exception`/`:the_exception`). Data-wise: scalars, immutable structs (→ `Tangent`), mutable
structs (→ `MutableTangent`), tuples/namedtuples, and closures (differentiate w.r.t. captures).
Higher-order works two ways: hand-nested `Dual` seeds (uniform nesting) and composed differentiation
(`D`-of-`D`, i.e. nesting a `frule`-based derivative operator) to arbitrary order — see the
higher-order bullets above. **Dynamic dispatch** (`apply_generic`) is also supported: a surviving
call whose callee or an argument has a non-concrete declared type — e.g. a value read from a
non-`const` global or an `Any`-typed field/`Ref`, which Julia infers as `Any` regardless of what it
holds — is deferred at run time to the `dynamic_frule` dispatcher (in `forward_interp.jl`), which
rebuilds concrete `Dual`s from the actual runtime values (that runtime `map(Dual, …)` is what recovers
concrete type parameters for `frule`'s generator) and dispatches `frule` dynamically. A call with
concrete callee+args but an inference-widened *abstract result* type stays on the ordinary static
`:invoke` path — only its result annotation widens to the abstract `Dual` (via `dual_type(R)`, since
`Dual` is invariant). See the "Dynamic dispatch" and "invariant-`Dual` typing rule" sections of the
`adnext-ircode-dualization` skill.

Not yet supported (bails gracefully with a clear `ErrorException` rather than miscompiling): array /
`memoryref` indexing (arrayref/boundscheck builtins on the live path) and other `Core.Builtin`s with
no rule, and vararg calls. Also unsupported: differentiating a closure/struct
with *differentiable fields* at order ≥2 (a representational limit of the self-tangent `Dual`
scheme — errors clearly). Every bail now records a specific, human-
readable *reason* (threaded via a `reason::Ref{String}` through `dualize_to_ircode`/`build_dual_ir`/
`primal_of_impl`) that names the offending IR construct — including a compact rendering of the actual
statement (`_stmt_str`) and its `%i` index, so the user sees *what* showed up in the IR that couldn't
be handled (e.g. the literal `:boundscheck` an array index emits) — surfaced by installing a small
error-raising `IRCode` (`error_ircode`) through the
same `finishinfer!`/`optimize` seam rather than falling through to the carrier stub's generic
message — see `adnext-architecture`'s "hard project constraint" above; this keeps that constraint
intact (still transform-IRCode-only, no post-hoc patching). See the `adnext-ircode-dualization` skill
for how supported constructs are handled, and `adnext-extending-ir-support` before adding support for
anything new.
