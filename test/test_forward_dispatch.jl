using Test
using Differ
using Differ: Dual, NoTangent, frule!!, primal, tangent

# Dynamic dispatch (`apply_generic`): reading a non-`const` global always infers as `Any`
# (regardless of the concrete type of the value it holds), so any call whose argument flows
# through it — here `getindex` on the `Ref`, then `+` — is a genuine `apply_generic`-style
# dynamic dispatch. These are handled by deferring the surviving call to the runtime
# `dynamic_frule` dispatcher, which rebuilds concrete `Dual`s from the runtime values and
# dispatches `frule!!` dynamically. Must be real module-level globals (not testset-local
# variables) — a local would instead be captured as a closure field, a different IR shape
# entirely from what this test exercises.
dyn_ref = Ref{Any}(1.0)
dyncall(x) = x + dyn_ref[]         # `dyncall` holds the `Ref` constant, so d/dx (x + c) = 1
dyn_g = sin
dyncallee(x) = dyn_g(x)            # callee itself is dynamic (read from a non-const global)
# A dynamic value that *carries a tangent*: box `x` in a `Ref{Any}`, read it back, and use it —
# the tangent must propagate, so d/dx (r[] * x) = 2x. SROA proves `r[] === x` and folds the read
# away, leaving `*(x, x)` with concrete args but an already-widened `::Any` result — that stays on
# the static `:invoke` path (result annotated `dual_type(R)` = abstract `Dual`), exercising the
# invariant-`Dual` typing rule rather than the `dynamic_frule` trampoline.
dynbox(x) = (r = Ref{Any}(x); r[] * x)
# A `Union`-typed return (a single `ReturnNode` whose value is a `PhiNode` typed
# `Union{Float64,Int}`): the packed `Dual` must be a concrete leaf (`Dual{Float64,Float64}` on
# this input), not the frozen `Dual{Union{Float64,Int},…}` a `%new` would build — which is *not*
# `<: dual_type(Union{…})`.
dynret(x) = (x > 0 ? x*x : 1)

@testset "dynamic dispatch (apply_generic)" begin
    # d/dx (x + const) = 1 (the `Ref` is held constant).
    @test frule!!(Dual(dyncall, NoTangent()), Dual(1.0, 1.0)).dx ≈ 1.0
    # dynamically-resolved callee read from a global: d/dx sin(x) = cos(x).
    @test frule!!(Dual(dyncallee, NoTangent()), Dual(0.5, 1.0)).dx ≈ cos(0.5)
    # tangent must propagate *through* the dynamic (Any-typed) value: d/dx (r[]*x) = 2x.
    @test frule!!(Dual(dynbox, NoTangent()), Dual(3.0, 1.0)).dx ≈ 6.0
    # matches finite differences on a nonlinear composition through a dynamic value.
    fdyn(x) = sin(x + dyn_ref[])
    h = 1e-6
    @test frule!!(Dual(fdyn, NoTangent()), Dual(0.7, 1.0)).dx ≈ (fdyn(0.7+h) - fdyn(0.7-h))/2h atol=1e-6

    # A non-concrete return type must pack the result as a *concrete leaf* `Dual` (built via a
    # runtime `Dual(p,t)` call), not a frozen `Dual{Any,Any}`/`Dual{Union,…}`, so the result stays
    # a well-typed dual (composable back into `frule!!`). `dynbox` returns `Any`; `dynret` a `Union`.
    @test frule!!(Dual(dynbox, NoTangent()), Dual(3.0, 1.0)) isa Dual{Float64,Float64}
    @test frule!!(Dual(dynret, NoTangent()), Dual(3.0, 1.0)) isa Dual{Float64,Float64}
end

# A `const` global `Ref`, unlike `dyn_ref` above, resolves to a concrete type at compile time, so
# reading it (`Core.getfield` on a bare `GlobalRef` in value position) stays on the static
# per-statement dualization path instead of falling to `dynamic_frule` — this used to crash with
# `MethodError: get_tangent_field(::NoTangent, ::Symbol)`, because the tangent of the *value* the
# global names was computed as the tangent of the `GlobalRef` struct itself (always `NoTangent`)
# rather than the tangent of the `Ref`. Exercised for both a concrete-eltype and an `Any`-eltype
# `Ref`, since the bug isn't a type-instability issue — both go through the identical code path.
# Must be real `const` module-level globals for the same reason as `dyn_ref` above.
const constref_float = Ref(2.0)
constref_float_use(x) = x * constref_float[]
const constref_any = Ref{Any}(2.0)
constref_any_use(x) = x * constref_any[]

@testset "GlobalRef operand in value position (const global Ref)" begin
    # `x -> x * G[]` for a `const` global `Ref` used to crash forward mode for both `Ref{Float64}`
    # and `Ref{Any}` globals. d/dx (x*c) = c.
    d1 = frule!!(Dual(constref_float_use, NoTangent()), Dual(3.0, 1.0))
    @test primal(d1) ≈ 3.0 * constref_float[]
    @test tangent(d1) ≈ constref_float[]

    d2 = frule!!(Dual(constref_any_use, NoTangent()), Dual(3.0, 1.0))
    @test primal(d2) ≈ 3.0 * constref_any[]
    @test tangent(d2) ≈ constref_any[]

    # Two calls with different primal inputs must not observe a stale/aliased tangent object
    # frozen from an earlier call (the fix emits a runtime `zero_tangent` call rather than
    # splicing a constructed tangent as a compile-time literal specifically to avoid this).
    d3 = frule!!(Dual(constref_float_use, NoTangent()), Dual(5.0, 1.0))
    @test primal(d3) ≈ 5.0 * constref_float[]
    @test tangent(d3) ≈ constref_float[]
end
