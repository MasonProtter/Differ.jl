ADNext package is a small, experimental post-optimization IR-based automatic differentiator.

This is implemented with reference to julia v1.13, always run code using `julia +1.13` as your engine, 
and use `../julia/Compiler` as your reference for anything related to the compiler.

The tangent representation is a port of Mooncake.jl's tangent / fdata / rdata type system
(`tangent_type`/`fdata_type`/`rdata_type` keyed on the *primal* type, `Tangent`/`MutableTangent`/
`PossiblyUninitTangent`, `FData`/`RData`, `zero_tangent`/`increment!!`, and the Mooncake-shaped
`Dual`/`CoDual` carriers). Forward mode is built on this `Dual`. `../Mooncake.jl` is the reference
for anything tangent-system-related. Reverse-mode AD is not implemented yet (only the type system).

## ADNext skills

`ADNext/.claude/skills/` has three skills describing the structure and meaning of ADNext's code in
detail. They should trigger automatically when relevant, but can also be invoked directly (e.g.
`/adnext-architecture`):

- **`adnext-architecture`** — orientation: the custom `AbstractInterpreter` compiler-plugin design,
  the `Dual`/`frule` calling convention, the file map, how to run tests and inspect dualized IR.
  Start here.
- **`adnext-ircode-dualization`** — deep-dive internals of the split-shadow dualization engine
  (`dualize_to_ircode` in `src/forward_interp.jl`): how control flow is handled, the `PhiNode`
  forward-reference mechanism, and known `Core.Compiler.verify_ir` gotchas.
- **`adnext-extending-ir-support`** — playbook for adding support for a new Julia IR construct
  (e.g. the next milestone, exception handling for `try`/`catch`), including what's already known
  about that specific follow-up. 

