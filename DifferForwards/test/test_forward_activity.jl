using Test
using DifferForwards
using DifferForwards: Dual, Inactive, NoTangent, NoFData, frule!!, zero_tangent, isactive,
    code_dual_ircode, primal, tangent, tangent_type, fdata_type, Tangent
using LinearAlgebra: dot
import DifferentiationInterface as DI
using DifferForwards: AutoDifferForwards

include(joinpath(@__DIR__, "testutils.jl"))

# `Inactive()` in a `Dual`'s tangent slot declares the value constant. Inside the dualized body,
# everything reachable only through constants is replayed primally with no shadow at all. A constant
# read by something *active* travels on as `Inactive()` to a call routed through `frule!!`, and is
# materialised as a real zero at its definition for every other consumer — intrinsic and builtin
# rules, phi-likes, `%new`, the return. `test_forward_rule_activity.jl` audits the rule side.
#
# Fixtures are top level where `@noinline` matters (the call must survive into the primal IR rather
# than be inlined away).

sxy(x, y) = x*y + sin(x)
sxyz(x, y, z) = x*y + y*z + sin(x*z)
loopdot(v, w) = (s = 0.0; for i in eachindex(v, w); s += v[i]*w[i]; end; s)
nested_dot(v, w) = dot(v, w)
# Splatting a runtime-length container (`Core._apply_iterate`) is a construct forward mode cannot
# dualize at all — the canonical bail. Written inline so it lands in `tagged`'s own IR: behind an
# `@noinline` call the bail would surface at run time from the callee's own carrier instead of at
# `code_dual_ircode`. The `push!` alongside it is supported, and is what the replay assertion below
# observes.
tagged(x, v) = x * 2.0 + (push!(v, 1.0); max(v...))
vtail(x, ys...) = x * ys[1] + ys[2]
loopdots(v, w, n) = (s = 0.0; for _ in 1:n; s += dot(v, w); end; s)
mapsin(v, w) = sum(map((a, b) -> a * sin(b), v, w))

# A function barrier allocating a fresh buffer from constant-only arguments, then written into with
# an active value: the buffer must still be an activity root, since a store never propagates
# activity backward into where the buffer was allocated.
@noinline mkbuf(n) = fill(0.0, n)
storebuf(x, n) = (v = mkbuf(n); v[1] = x; v[1] * 1.0)
# The counter-case: a barrier returning a bits value from constant arguments stays inactive.
@noinline mkbits(n) = n + 1
usebits(x, n) = (m = mkbits(n); x * m)
# A barrier returning an immutable struct that wraps a mutable `Vector` field: still an activity
# root, since the fdata-carrying part of its tangent is the nested `Vector`, not the struct itself.
struct VecWrap
    v::Vector{Float64}
end
@noinline mkwrap(n) = VecWrap(fill(0.0, n))
storewrap(x, n) = (w = mkwrap(n); w.v[1] = x; w.v[1] * 1.0)

@testset "activity: an inactive argument is not merely zero-seeded" begin
    d = const_dual(3.0)
    @test tangent(d) === Inactive()
    @test !isactive(tangent(d))
    @test typeof(d) === Dual{Float64,Inactive}
    # Distinct from both a zero tangent and `NoTangent()` — either of those says something else.
    @test tangent(Dual(3.0, 0.0)) === 0.0
    @test tangent(d) !== NoTangent()
    @test typeof(const_dual([1.0, 2.0])) === Dual{Vector{Float64},Inactive}
end

@testset "activity: scalar arguments, every signature" begin
    x, y = 1.3, 2.1
    fd = Dual(sxy, NoTangent())
    # Directional derivative with a unit seed on each active slot, checked against finite differences
    # and against the corresponding slot of the fully active run.
    @test frule!!(fd, Dual(x, 1.0), const_dual(y)).dx ≈ central_diff(sxy, x, y, 1)
    @test frule!!(fd, const_dual(x), Dual(y, 1.0)).dx ≈ central_diff(sxy, x, y, 2)
    @test frule!!(fd, Dual(x, 1.0), const_dual(y)).dx ≈ frule!!(fd, Dual(x, 1.0), Dual(y, 0.0)).dx
    @test frule!!(fd, const_dual(x), Dual(y, 1.0)).dx ≈ frule!!(fd, Dual(x, 0.0), Dual(y, 1.0)).dx
    # Primal is unaffected by any activity signature, including all-constant.
    for a in (Dual(x, 1.0), const_dual(x)), b in (Dual(y, 1.0), const_dual(y))
        @test frule!!(fd, a, b).x ≈ sxy(x, y)
    end
    for inact in ((), (1,), (2,), (1, 2))
        checkverify(sxy, (Float64, Float64); inactive=inact)
    end
end

@testset "activity: matches the full run slot for slot" begin
    x, y, z = 0.7, 1.9, 2.3
    fd = Dual(sxyz, NoTangent())
    full = (frule!!(fd, Dual(x, 1.0), Dual(y, 0.0), Dual(z, 0.0)).dx,
            frule!!(fd, Dual(x, 0.0), Dual(y, 1.0), Dual(z, 0.0)).dx,
            frule!!(fd, Dual(x, 0.0), Dual(y, 0.0), Dual(z, 1.0)).dx)
    seeds = (x, y, z)
    for k in 1:3
        args = ntuple(j -> j == k ? Dual(seeds[j], 1.0) : const_dual(seeds[j]), 3)
        @test frule!!(fd, args...).dx ≈ full[k]
    end
    for inact in ((), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))
        checkverify(sxyz, (Float64, Float64, Float64); inactive=inact)
    end
end

@testset "activity: array arguments" begin
    v, w = [1.0, 2.0, 3.0], [0.5, 1.5, 2.5]
    fd = Dual(loopdot, NoTangent())
    dv = [1.0, 0.0, 0.0]
    @test frule!!(fd, Dual(v, dv), const_dual(w)).dx ≈ w[1]
    @test frule!!(fd, Dual(v, dv), const_dual(w)).dx ≈
          frule!!(fd, Dual(v, dv), Dual(w, zeros(3))).dx
    @test frule!!(fd, const_dual(v), Dual(w, [0.0, 1.0, 0.0])).dx ≈ v[2]
    @test frule!!(fd, const_dual(v), const_dual(w)).x ≈ loopdot(v, w)
    for inact in ((), (1,), (2,), (1, 2))
        checkverify(loopdot, (Vector{Float64}, Vector{Float64}); inactive=inact)
    end
end

@testset "activity: loops run more than twice" begin
    # ISSUES #52's lesson: a dataflow change touching phis can be right for 0 and 1 iterations and
    # corrupt everything from the second onward. Run a range of trip counts.
    for n in 0:4
        v, w = collect(1.0:n), collect(0.5:1.0:(n - 0.5))
        for k in 1:n
            dv = zeros(n); dv[k] = 1.0
            @test frule!!(Dual(loopdot, NoTangent()), Dual(v, dv), const_dual(w)).dx ≈ w[k]
        end
        @test frule!!(Dual(loopdot, NoTangent()), const_dual(v), const_dual(w)).x ≈ loopdot(v, w)
    end

    # Same range, but with a *hand rule* in the loop body: the constant operand is handed to `dot`
    # as `Inactive()` once per iteration rather than zeroed once at its definition.
    v, w = [1.0, 2.0, 3.0], [0.5, 1.5, 2.5]
    fd = Dual(loopdots, NoTangent())
    for n in 0:4
        @test frule!!(fd, Dual(v, [1.0, 0.0, 0.0]), const_dual(w), Dual(n, NoTangent())).dx ≈ n*w[1]
        @test frule!!(fd, const_dual(v), Dual(w, [0.0, 0.0, 1.0]), Dual(n, NoTangent())).dx ≈ n*v[3]
        @test frule!!(fd, const_dual(v), const_dual(w), Dual(n, NoTangent())).x ≈ loopdots(v, w, n)
    end
    checkverify(loopdots, (Vector{Float64}, Vector{Float64}, Int); inactive=(2,))
end

@testset "activity: a constant array through map's per-element carriers" begin
    # `map`'s rule builds an inner `Dual` per element; a constant outer array has to reach the inner
    # `frule!!` as inactive too, not as a zero it would then multiply by.
    v, w = [1.0, 2.0, 3.0], [0.5, 1.5, 2.5]
    fd = Dual(mapsin, NoTangent())
    @test frule!!(fd, Dual(v, [1.0, 0.0, 0.0]), const_dual(w)).dx ≈ sin(w[1])
    @test frule!!(fd, const_dual(v), Dual(w, [0.0, 0.0, 1.0])).dx ≈ v[3]*cos(w[3])
    @test frule!!(fd, const_dual(v), const_dual(w)).x ≈ mapsin(v, w)
    checkverify(mapsin, (Vector{Float64}, Vector{Float64}); inactive=(2,))
end

@testset "activity: constant-only code is replayed, not differentiated" begin
    # The coverage payoff, and the reason this feature is worth more than the elided shadow
    # arithmetic: `growread` is code the transform cannot dualize at all. Active, it bails; constant,
    # it is replayed primally and never reaches `frule_split!`.
    r_active = bail_reason(tagged, (Float64, Vector{Float64}))
    @test r_active !== nothing
    @test occursin("_apply_iterate", r_active)
    @test bail_reason(tagged, (Float64, Vector{Float64}); inactive=(2,)) === nothing

    v = [4.0]
    r = frule!!(Dual(tagged, NoTangent()), Dual(2.0, 1.0), const_dual(v))
    @test r.x ≈ tagged(2.0, [4.0])          # the primal still runs, mutation and all
    @test r.dx ≈ 2.0                        # only `x * 2.0` contributes
    @test length(v) == 2                    # `push!` was genuinely replayed
    checkverify(tagged, (Float64, Vector{Float64}); inactive=(2,))
end

@testset "activity: a fresh container behind a function barrier is an activity root" begin
    # `mkbuf`'s only argument (`n`) is inactive, so operand-based activity alone would classify the
    # allocated buffer constant; the store of `x` into it would then have nowhere active to go.
    r = frule!!(Dual(storebuf, NoTangent()), Dual(2.0, 1.0), Dual(3, NoTangent()))
    @test r.x ≈ storebuf(2.0, 3)
    @test r.dx ≈ 1.0
    checkverify(storebuf, (Float64, Int))

    # Counter-case: a barrier returning a bits value from constant arguments has nothing to write a
    # derivative into, so it stays inactive — no shadow statement for the `mkbits` call.
    ir, _ = code_dual_ircode(usebits, (Float64, Int))
    stmts = [sprint(show, ir.stmts[i][:stmt]) for i in 1:length(ir.stmts)]
    @test !any(s -> occursin("mkbits", s) && occursin("Dual", s), stmts)
    r2 = frule!!(Dual(usebits, NoTangent()), Dual(2.0, 1.0), Dual(3, NoTangent()))
    @test r2.x ≈ usebits(2.0, 3)
    @test r2.dx ≈ 4.0                       # d/dx(x*(n+1)) = n+1
    checkverify(usebits, (Float64, Int))

    # An immutable wrapper around a mutable field, returned through a barrier from constant
    # arguments: still a root, since fdata-non-triviality is checked on the tangent recursively.
    r3 = frule!!(Dual(storewrap, NoTangent()), Dual(2.0, 1.0), Dual(3, NoTangent()))
    @test r3.x ≈ storewrap(2.0, 3)
    @test r3.dx ≈ 1.0
    checkverify(storewrap, (Float64, Int))
end

@testset "activity: no shadow is emitted for an inactive subgraph" begin
    # The elision is real, not merely unused: the constant operand's shadow work is gone from the
    # carrier, so the inactive dualization is strictly smaller than the fully active one.
    at = (Vector{Float64}, Vector{Float64})
    n_full = length(code_dual_ircode(loopdot, at)[1].stmts)
    n_cut  = length(code_dual_ircode(loopdot, at; inactive=(2,))[1].stmts)
    @test n_cut < n_full
    # With nothing active at all, no tangent is read out of any carrier and no zero is materialised
    # — the whole body is primal replay.
    stmts = [string(s) for s in code_dual_ircode(loopdot, at; inactive=(1, 2))[1].stmts.stmt]
    @test !any(s -> occursin("zero_tangent", s), stmts)
    @test length(stmts) < n_cut
end

@testset "activity: an inactive operand feeding an active hand rule" begin
    # Nested, deliberately: the routing decision happens inside `dualize_to_ircode`, so this is the
    # path that hands the `dot` rule an `Inactive()` through `frule_split!`.
    v, w = [1.0, 2.0, 3.0], [0.5, 1.5, 2.5]
    fd = Dual(nested_dot, NoTangent())
    @test frule!!(fd, Dual(v, [1.0, 0.0, 0.0]), const_dual(w)).dx ≈ w[1]
    @test frule!!(fd, const_dual(v), Dual(w, [0.0, 0.0, 1.0])).dx ≈ v[3]
    @test frule!!(fd, const_dual(v), const_dual(w)).x ≈ dot(v, w)
    checkverify(nested_dot, (Vector{Float64}, Vector{Float64}); inactive=(2,))

    # The constant operand reaches the rule as `Inactive()`, not as a materialised zero: no
    # `zero_tangent` call survives for it, and the `frule!!` invoke names `Dual{…,Inactive}`.
    ir, _ = code_dual_ircode(nested_dot, (Vector{Float64}, Vector{Float64}); inactive=(2,))
    stmts = [sprint(show, ir.stmts[i][:stmt]) for i in 1:length(ir.stmts)]
    @test !any(s -> occursin("zero_tangent", s), stmts)
    @test any(s -> occursin("Inactive", s), stmts)
end

@testset "activity: a hand rule takes an Inactive shadow directly" begin
    # A rule signature like `Dual{Vector{Float64}}` leaves the tangent parameter free, so it matches
    # `Dual{…,Inactive}`; the bodies now handle what they match rather than throwing inside.
    v, w = [1.0, 2.0, 3.0], [0.5, 1.5, 2.5]
    fd = Dual(dot, NoTangent())
    @test frule!!(fd, Dual(v, [1.0, 0.0, 0.0]), const_dual(w)).dx ≈ w[1]
    @test frule!!(fd, const_dual(v), Dual(w, [0.0, 0.0, 1.0])).dx ≈ v[3]
    @test frule!!(fd, const_dual(v), const_dual(w)) === Dual(dot(v, w), 0.0)

    # The structural strong zero the materialised zero forfeited: skipping the `dot(x, dy)` term is
    # not the same as multiplying by zeros, when `x` holds an `Inf`.
    vinf, dvinf = [Inf, 1.0, 1.0], [0.0, 1.0, 0.0]
    @test frule!!(fd, Dual(vinf, dvinf), const_dual(w)).dx == w[2]
    @test isnan(dot(dvinf, w) + dot(vinf, zero_tangent(w)))     # what a materialised zero gives
end

setidx1!(x, v) = (v[1] = x; v[1] * 1.0)
mutable struct ActMS
    a::Float64
end
setfld1!(x, m) = (m.a = x; m.a * 1.0)
setidxmask!(x, v) = (setindex!(v, [x], [true, false]); v[1] * 1.0)

@testset "activity: an inlined store refuses an Inactive destination fed an active value" begin
    # `v[1] = x` has no hand rule for a scalar `Int` index — it always inlines to raw
    # `memoryrefset!`. A constant `v` (`Inactive` shadow) materialises a fresh, disconnected zero
    # `MemoryRef` wherever the ref is needed; writing an active `x` through it would silently drop
    # the derivative instead of surfacing an error.
    @test_throws ErrorException frule!!(Dual(setidx1!, NoTangent()), Dual(2.0, 1.0), const_dual([0.0]))
    @test_throws ErrorException frule!!(Dual(setfld1!, NoTangent()), Dual(2.0, 1.0), const_dual(ActMS(0.0)))

    # Dest active, value active: unchanged.
    @test frule!!(Dual(setidx1!, NoTangent()), Dual(2.0, 1.0), Dual([0.0], [0.0])).dx ≈ 1.0
    @test frule!!(Dual(setfld1!, NoTangent()), Dual(2.0, 1.0), Dual(ActMS(0.0), zero_tangent(ActMS(0.0)))).dx ≈ 1.0

    # Dest active, value Inactive: the existing strong zero, unchanged.
    @test frule!!(Dual(setidx1!, NoTangent()), const_dual(2.0), Dual([0.0], [0.0])).dx == 0.0
    @test frule!!(Dual(setfld1!, NoTangent()), const_dual(2.0), Dual(ActMS(0.0), zero_tangent(ActMS(0.0)))).dx == 0.0

    # Dest Inactive, value Inactive too: nothing active is lost, so no error.
    @test frule!!(Dual(setidx1!, NoTangent()), const_dual(2.0), const_dual([0.0])).dx == 0.0
    @test frule!!(Dual(setfld1!, NoTangent()), const_dual(2.0), const_dual(ActMS(0.0))).dx == 0.0

    # The dualized IR itself is still legal in every case — the refusal is a runtime error inside
    # otherwise-verifiable IR, not a compile-time bail.
    checkverify(setidx1!, (Float64, Vector{Float64}); inactive=(2,))
    checkverify(setfld1!, (Float64, ActMS); inactive=(2,))

    # The non-inlined hand-rule path (`setindex!` with a mask, which does have a rule) agrees: an
    # Inactive destination is refused there too.
    @test_throws ErrorException frule!!(Dual(setidxmask!, NoTangent()), Dual(2.0, 1.0), const_dual([0.0, 0.0]))
end

@testset "activity: vararg tails and higher order degrade rather than bail" begin
    # A constant trailing element is materialised in the prologue, so the packed tangent tuple keeps
    # its primal-derived type and the rest of the transform needs no vararg awareness.
    fd = Dual(vtail, NoTangent())
    @test frule!!(fd, Dual(2.0, 1.0), const_dual(3.0), const_dual(4.0)).x ≈ 10.0
    @test frule!!(fd, Dual(2.0, 1.0), const_dual(3.0), const_dual(4.0)).dx ≈ 3.0
    @test frule!!(fd, const_dual(2.0), Dual(3.0, 1.0), const_dual(4.0)).dx ≈ 2.0
    checkverify(vtail, (Float64, Float64, Float64); inactive=(2, 3))

    # Order >= 2 with a constant seed: correct, with the elision happening one order down.
    sq(x, y) = x*x*y
    r = frule!!(Dual(Dual(sq, NoTangent()), Dual(sq, NoTangent())),
                Dual(Dual(1.5, 1.0), Dual(1.0, 0.0)), const_dual(Dual(2.0, 0.0)))
    @test r.x.x ≈ sq(1.5, 2.0)
    checkverify2(sq, (Float64, Float64); order=2, inactive=(2,))
end

@testset "activity: DI.Constant maps to Inactive, DI.Cache stays active" begin
    f(x, w) = sum(x .* w)
    x, w = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    tx = ([1.0, 0.0, 0.0],)
    # A `DI.Constant` context is seeded `Dual(w, Inactive())` by `_ctx_dual`, so the pushforward
    # matches the one where `w` is an ordinary argument with a zero tangent.
    ty = DI.pushforward(f, AutoDifferForwards(), x, tx, DI.Constant(w))
    @test only(ty) ≈ w[1]
    # `DI.Cache` stays active — deliberately not mapped to `Inactive`.
    fcache(x, c) = sum(x .* c)
    @test only(DI.pushforward(fcache, AutoDifferForwards(), x, tx, DI.Cache(copy(w)))) ≈ w[1]
end

@testset "activity: out-of-range inactive positions are rejected" begin
    @test_throws ArgumentError code_dual_ircode(sxy, (Float64, Float64); inactive=(3,))
    @test_throws ArgumentError code_dual_ircode(sxy, (Float64, Float64); inactive=(0,))
    @test_throws MethodError code_dual_ircode(sxy, (Float64, Float64); inactive=[2])
    checkverify(sxy, (Float64, Float64); inactive=2)   # a bare Int is accepted
end

# A statically-known operand whose tangent has no fdata (a float literal, a `const` global scalar)
# is minted `Dual{P,Inactive}` at a call, exactly like a caller-declared constant argument. A
# demanding consumer — an intrinsic, a builtin, a `%new` — still gets a materialised zero, since it
# reads the tangent structurally rather than dispatching on it.
@noinline litmix(a, b) = a*b + a/b            # no hand rule: reached through the derived fallback
struct LitPair
    a::Float64
    b::Float64
end
@noinline litnew(x) = LitPair(x, 2.5)

lit_hand(x)  = atan(x, 2.5)                   # two-arg `atan` has a hand rule
lit_derived(x) = litmix(x, 2.5)
lit_intr(x)  = x*2.5 + 1.5                    # both literals consumed by intrinsics
lit_field(x) = (p = litnew(x); p.a * p.b)

# Types print module-qualified inside a `SafeTestset`, so match around the qualifiers.
const inactive_lit_new = r"%new\((\w+\.)?Dual\{Float64, (\w+\.)?Inactive\}, 2\.5, (\w+\.)?Inactive\(\)\)"

@testset "activity: a float literal is minted Inactive at a call" begin
    x = 1.25
    # (i) a hand rule.
    d = frule!!(Dual(lit_hand, NoTangent()), Dual(x, 1.0))
    @test d.x ≈ lit_hand(x)
    @test d.dx ≈ 2.5/(x^2 + 2.5^2)
    ir, _ = code_dual_ircode(lit_hand, (Float64,))
    stmts = [sprint(show, ir.stmts[i][:stmt]) for i in 1:length(ir.stmts)]
    @test any(s -> occursin(inactive_lit_new, s), stmts)
    @test !any(s -> occursin("zero_tangent", s), stmts)
    checkverify(lit_hand, (Float64,))

    # (ii) the derived fallback of a composite callee.
    d = frule!!(Dual(lit_derived, NoTangent()), Dual(x, 1.0))
    @test d.x ≈ lit_derived(x)
    @test d.dx ≈ 2.5 + 1/2.5
    ir, _ = code_dual_ircode(lit_derived, (Float64,))
    stmts = [sprint(show, ir.stmts[i][:stmt]) for i in 1:length(ir.stmts)]
    @test any(s -> occursin(inactive_lit_new, s), stmts)
    checkverify(lit_derived, (Float64,))

    # (iii) demanding consumers keep their zero: an intrinsic reads the operand's tangent directly,
    # and so does the `%new` building a struct's `Tangent`.
    d = frule!!(Dual(lit_intr, NoTangent()), Dual(x, 1.0))
    @test d.x ≈ lit_intr(x)
    @test d.dx ≈ 2.5
    ir, _ = code_dual_ircode(lit_intr, (Float64,))
    @test !any(occursin("Inactive", sprint(show, ir.stmts[i][:stmt])) for i in 1:length(ir.stmts))

    d = frule!!(Dual(lit_field, NoTangent()), Dual(x, 1.0))
    @test d.x ≈ lit_field(x)
    @test d.dx ≈ 2.5
    checkverify(lit_field, (Float64,))
end


# A code constant whose tangent holds an inner-pass `Dual`: the shape DifferentiationInterface's
# prep object has at order 2, where it keeps the callee's `Dual` in a field. `@noinline` keeps the
# constant as an operand of a surviving call, which is where the mint-`Inactive` decision is made.
struct DualHolder{D}
    fdual::D
end
const DUAL_HOLDER = DualHolder(Dual(sin, NoTangent()))
@noinline holder_scale(h::DualHolder, x::Float64) = primal(h.fdual)(x)
holder_apply(x::Float64) = holder_scale(DUAL_HOLDER, x) * x

@testset "activity: a constant whose tangent holds a Dual" begin
    # A `Dual` slot that carries a primal through contributes no forward-pass storage, so the
    # holder's tangent is fdata-free and the constant qualifies to be minted `Inactive()`. Without
    # a `fdata_type` method for `Dual` this query errors and takes the compilation with it.
    P = typeof(DUAL_HOLDER)
    @test tangent_type(P) === Tangent{@NamedTuple{fdual::Dual{typeof(sin),NoTangent}}}
    @test fdata_type(tangent_type(P)) === NoFData
    @test DifferForwards.DifferCore.inactive_constant_type(P)

    x = 0.75
    d = frule!!(Dual(holder_apply, NoTangent()), Dual(x, 1.0))
    @test d.x ≈ sin(x) * x
    @test d.dx ≈ cos(x) * x + sin(x)
    checkverify(holder_apply, (Float64,))
end
