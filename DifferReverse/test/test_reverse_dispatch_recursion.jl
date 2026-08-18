using Test
using DifferReverse
using DifferReverse: NoTangent, rev_gradient, value_and_gradient!, get_tangent_field,
                     MutableTangent
# `Dual`/`frule!!` here are DifferForwards' forward-mode carrier, used purely as an independent
# numerical oracle.
using DifferForwards: Dual, frule!!

include(joinpath(@__DIR__, "testutils.jl"))

# A dynamic-dispatch callee (read from a non-const global) isn't statically recursible; must bail
# cleanly, not crash. Must be module-level, not testset-local: a local would be captured as a
# closure field instead, a different IR shape.
dyn_g = sin
dyncallee(x) = dyn_g(x)

# A self-recursive @noinline primal has no finite Tape type without the closed-form-Tape handling
# (the in_progress cycle guard would otherwise bail). Must be module-level: a local self-recursive
# function closes over its own boxed binding, giving it a real (non-NoTangent) tangent type instead
# of the plain singleton a top-level function has, defeating rev_gradient's zero-tangent seeding.
@noinline function rec_self(x::Float64, n::Int)
    n <= 0 && return x
    return rec_self(x, n - 1)
end

# Same module-level requirement as rec_self. Used across several tests below: plain recursion,
# under a branch, under a loop, and through both carrier specializations.
@noinline function rec_sum(v::Vector{Float64}, i::Int)
    i > length(v) && return 0.0
    return v[i] + rec_sum(v, i + 1)
end

# Genuine mutual recursion (A -> B -> A), also module-level. Must still bail cleanly: it needs a
# tape-type pre-pass across the whole SCC, out of scope for direct self-recursion.
@noinline mutA(x::Float64, n::Int) = n <= 0 ? x : mutB(x, n - 1)
@noinline mutB(x::Float64, n::Int) = mutA(x, n - 1)

# Self-recursive with two distinct static call sites in two different, mutually-exclusive blocks,
# unlike rec_sum / the Base.mapreduce_impl case (one shared block).
@noinline function evenodd(n::Int, x::Float64)
    n <= 0 && return x
    if iseven(n)
        return evenodd(n - 1, x + 1.0) * 2.0
    else
        return evenodd(n - 1, x * 2.0) + 1.0
    end
end

# Self-recursion combined with a constant argument. Both carriers declare a self-`:invoke`'s result
# type in closed form (the callee is the build itself, so there is no `CodeInstance` to read it off),
# and both have to apply the same inactive substitution their own return does. Module-level per the
# `rec_self` comment above.
@noinline function rec_scale(x::Float64, c::Float64, n::Int)
    n <= 0 && return x * c
    return rec_scale(x + 1.0, c, n - 1)
end

# The constant argument's rdata type is a non-trivial `RData`, so its slot cannot pass by looking
# like the neighbouring `Float64` one.
@noinline function rec_tupc(x::Float64, c::Tuple{Float64,Float64}, n::Int)
    n <= 0 && return x * c[1] + c[2]
    return rec_tupc(x + 1.0, c, n - 1)
end

# The constant argument carries fdata (an array), threaded through the recursive call.
@noinline function rec_arrc(v::Vector{Float64}, w::Vector{Float64}, i::Int)
    i > length(v) && return 0.0
    return v[i] * w[i] + rec_arrc(v, w, i + 1)
end

# Vararg self-recursion with a constant in the fixed part (`Inactive` in the packed tail is refused
# outright), so the returned rdatas tuple mixes the substitution with the tail scatter.
@noinline function rec_va(c::Float64, n::Int, xs::Float64...)
    n <= 0 && return c * sum(xs)
    return rec_va(c, n - 1, xs...)
end

# Exits disagreeing on their shadow type: the base case returns the constant itself
# (`CoDual{Float64,Inactive}`), the recursive one returns the call's result (`NoFData`). Resolving
# that needs a fixpoint over the return shadow type, so it must bail rather than declare one of the
# two over a call that returns the other.
@noinline function rec_ret_c(x::Float64, c::Float64, n::Int)
    n <= 0 && return c
    return rec_ret_c(x + 1.0, c, n - 1)
end

# A const global holding a function, passed as an operand of a surviving call: the operand is a
# GlobalRef in the optimized IR, so its type must come from the binding, not the node.
const CONST_G = sin
usecg(v) = sum(CONST_G, v)

# The non-const twin. Its operand survives as its own `%k = Main.dyn_g` load statement typed Any,
# so this must bail (reverse) or dispatch dynamically (forward).
nonconst_g = sin
usencg(v) = sum(nonconst_g, v)

vasum(x...) = sum(x)

# fdata-carrying immutable arguments to a recursive call (struct/tuple/NamedTuple wrapping a
# tracked array) — module-level per the file's convention.
struct RecW
    v::Vector{Float64}
end
@noinline recw_inner(w::RecW) = sum(w.v)
recw_outer(x) = recw_inner(RecW(x))

# Mixed fdata (b) + rdata (a) fields: exercises the returned-rdata routing on top of the fdata path.
struct RecM
    a::Float64
    b::Vector{Float64}
end
@noinline recm_inner(m::RecM) = m.a * sum(m.b)
recm_outer(x) = recm_inner(RecM(x[1], x))

@noinline rectup_inner(t) = sum(t[1]) + 2sum(t[2])
rectup_outer(x) = rectup_inner((x, 2 .* x))

@noinline recnt_inner(nt) = sum(nt.p) * nt.q
recnt_outer(x) = recnt_inner((p = x, q = x[1] * 3))

# Struct built from a non-const global. Nothing in the call reaches `x`, so activity analysis marks
# it inactive and the whole call is replayed primally rather than differentiated — no provenance
# question arises.
global_recw_v::Vector{Float64} = [10.0, 20.0, 30.0]
recw_untraced_outer(x) = recw_inner(RecW(global_recw_v)) + sum(x)

# Negative case: a struct mixing a traceable field with an untraceable one. The call *is* active (it
# reaches `x`), so it goes through recursion resolution and the provenance check fires. Explicitly
# typed so the argument type stays concrete (unlike `dyncallee` above) and the bail comes from the
# provenance check, not an earlier non-concrete-type guard.
struct RecMix
    a::Vector{Float64}
    b::Vector{Float64}
end
@noinline recmix_inner(m::RecMix) = sum(m.a) + 2 * sum(m.b)
recmix_untraced_outer(x) = recmix_inner(RecMix(x, global_recw_v))

@testset "reverse mode: recursion into a hand-written rule" begin
    # A surviving high-level call differentiates via the recursive rrule support below (sin
    # resolves to the hand-written rule in rrules.jl, not raw recursion into its internals).
    plus1(x) = sin(x) + 1
    nest(x)  = sin(cos(x))

    _, dx_plus1 = rev_gradient(plus1, 1.3)
    @test dx_plus1 ≈ cos(1.3)

    checkverify_rev(plus1, (Float64,))
    checkverify_rev(nest, (Float64,))
    check_stack_balance(plus1, 1.3)
    check_stack_balance(nest, 0.4)
end

@testset "reverse mode: recursive rrule calls (statically-resolvable only)" begin
    # A genuinely separate, non-inlined callee must survive as an :invoke to exercise recursion,
    # not fold into ordinary arithmetic, hence @noinline.
    @noinline rec_sq(x) = x * x                         # x^2, d/dx = 2x
    rec_call(x) = rec_sq(x) + x                          # d/dx = 2x + 1
    function rec_branch(x)                               # recursion combined with a branch
        return x > 0.0 ? rec_sq(x) + 1.0 : rec_sq(x) - 1.0
    end
    function rec_loop(x, k)                              # recursion combined with a loop
        s = 0.0
        for i in 1:k
            s += rec_sq(x)
        end
        return s
    end

    _, dx_rec = rev_gradient(rec_call, 3.0)
    @test dx_rec ≈ 2*3.0 + 1
    @test dx_rec ≈ frule!!(Dual(rec_call, NoTangent()), Dual(3.0, 1.0)).dx

    for x in (2.0, -2.0)
        _, dx_rb = rev_gradient(rec_branch, x)
        @test dx_rb ≈ 2x
    end

    _, dx_rl = rev_gradient(rec_loop, 2.0, 3)
    @test dx_rl ≈ 3 * 2 * 2.0
    check_stack_balance(rec_loop, 2.0, 5)

    checkverify_rev(rec_call, (Float64,))
    checkverify_rev(rec_branch, (Float64,))
    checkverify_rev(rec_loop, (Float64, Int))
end

@testset "reverse mode: direct self-recursion" begin
    # rec_self is genuinely self-recursive: no hand rrule!!, and the recursive call resolves to the
    # exact primal currently being differentiated. d(rec_self(x,n))/dx = 1 for every n >= 0.
    _, dx_self, dn_self = rev_gradient(rec_self, 1.0, 3)
    @test dx_self == 1.0
    @test dn_self isa NoTangent
    for n in 0:4
        _, dx_n, _ = rev_gradient(rec_self, 2.7, n)
        @test dx_n == 1.0
    end
    checkverify_rev(rec_self, (Float64, Int))
    # Exercises both the tape-allocating (Ctx{Nothing}) and pre-allocated (Ctx{<:Tape}) carrier
    # specializations. The self-edge always targets the pre-allocated one, even when the outer call
    # is the tape-allocating carrier — a self-recursive primal's tape-allocating carrier must
    # actually compile the pre-allocated sibling it recurses into.
    check_stack_balance(rec_self, 1.0, 3)

    # Accumulating self-recursion: each level contributes a distinct addend, so a wrong tape layout
    # or a stale/misrouted comms value would show up as a wrong (not just uniformly-1) gradient.
    v = [1.0, 2.0, 3.0, 4.0]
    _, dv, _ = rev_gradient(rec_sum, v, 1)
    @test dv == ones(4)
    h = 1e-6
    for k in 1:4
        vp = copy(v); vp[k] += h
        vm = copy(v); vm[k] -= h
        @test dv[k] ≈ (rec_sum(vp, 1) - rec_sum(vm, 1)) / 2h rtol = 1e-6
    end
    checkverify_rev(rec_sum, (Vector{Float64}, Int))
    check_stack_balance(rec_sum, [1.0, 2.0, 3.0], 1)

    # Self-recursion reached under a branch: the recursive call site itself lives in one arm only.
    function rec_sum_branch(v::Vector{Float64}, flag::Bool)
        return flag ? rec_sum(v, 1) : 0.0
    end
    _, dvb, _ = rev_gradient(rec_sum_branch, v, true)
    @test dvb == ones(4)
    _, dvb0, _ = rev_gradient(rec_sum_branch, v, false)
    @test dvb0 == zeros(4)
    checkverify_rev(rec_sum_branch, (Vector{Float64}, Bool))
    check_stack_balance(rec_sum_branch, v, true)

    # Self-recursion invoked repeatedly from inside a loop.
    function rec_sum_loop(v::Vector{Float64}, k::Int)
        s = 0.0
        for _ in 1:k
            s += rec_sum(v, 1)
        end
        return s
    end
    _, dvl, _ = rev_gradient(rec_sum_loop, v, 3)
    @test dvl == fill(3.0, 4)
    checkverify_rev(rec_sum_loop, (Vector{Float64}, Int))
    check_stack_balance(rec_sum_loop, v, 3)

    # Mutual recursion (A -> B -> A) is explicitly out of scope (needs a tape-type pre-pass across
    # the whole SCC) and must still bail cleanly: never hang, never crash on verify_ir.
    err_mut = try
        rev_gradient(mutA, 1.0, 4)
        nothing
    catch e
        e
    end
    @test err_mut isa ErrorException
    @test occursin("self- or mutually-recursive primal", err_mut.msg)
end

@testset "reverse mode: direct self-recursion with a constant argument" begin
    # A self-`:invoke`'s result type is closed-form on both carriers — its callee is the build
    # itself — so both have to apply the same inactive substitution their own return does. Checked
    # at run time by necessity: `verify_ir` does not look at declared statement types, and the
    # pullback's half failed as a SIGILL rather than an exception.
    at = (Float64, Float64, Int)
    fcd, xcd, ncd = zero_fcodual(rec_scale), CoDual(2.0, NoFData()), zero_fcodual(2)
    all_active = rrule!!(fcd, Ctx(), xcd, CoDual(3.0, NoFData()), ncd)[2](1.0)

    ycd, pb = rrule!!(fcd, Ctx(), xcd, const_codual(3.0), ncd)
    @test primal(ycd) == rec_scale(2.0, 3.0, 2)
    rd = pb(1.0)
    @test rd == (NoRData(), all_active[2], NoRData(), NoRData())
    @test rd[2] ≈ central_diff(x -> rec_scale(x, 3.0, 2), 2.0) rtol = 1e-6
    _assert_tape_balanced(pb)
    checkverify_rev(rec_scale, at; inactive=(2,))

    # Pre-allocated context: the self-edge resolves against an already-compiled sibling there rather
    # than the mid-compile identity, and reusing the tape must not move the answer.
    ctx = build_ctx(rec_scale, at; inactive=(2,))
    y1, g1 = value_and_gradient!(ctx, fcd, xcd, const_codual(3.0), ncd)
    y2, g2 = value_and_gradient!(ctx, fcd, xcd, const_codual(3.0), ncd)
    @test y1 == y2 == rec_scale(2.0, 3.0, 2)
    @test g1 == g2 == (NoTangent(), all_active[2], Inactive(), NoTangent())

    # A constant whose rdata type isn't `Float64`, so its slot can't pass by resembling its
    # neighbour's.
    yt, pbt = rrule!!(zero_fcodual(rec_tupc), Ctx(), xcd, const_codual((3.0, 4.0)), ncd)
    @test primal(yt) == rec_tupc(2.0, (3.0, 4.0), 2)
    @test pbt(1.0) == (NoRData(), 3.0, NoRData(), NoRData())
    checkverify_rev(rec_tupc, (Float64, Tuple{Float64,Float64}, Int); inactive=(2,))

    # A constant carrying fdata, threaded through the recursive call: the active argument still
    # accumulates into the caller's own buffer, and no shadow is allocated for the constant.
    v, w, dv = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], zeros(3)
    ya, pba = rrule!!(zero_fcodual(rec_arrc), Ctx(), CoDual(v, dv), const_codual(w), zero_fcodual(1))
    @test primal(ya) == rec_arrc(v, w, 1)
    @test pba(1.0) == (NoRData(), NoRData(), NoRData(), NoRData())
    @test dv == w
    checkverify_rev(rec_arrc, (Vector{Float64}, Vector{Float64}, Int); inactive=(2,))

    # Vararg, constant in the fixed part: the returned rdatas tuple mixes the substitution with the
    # packed tail's scatter.
    yv, pbv = rrule!!(zero_fcodual(rec_va), Ctx(), const_codual(2.0), zero_fcodual(1),
                      CoDual(3.0, NoFData()), CoDual(4.0, NoFData()))
    @test primal(yv) == rec_va(2.0, 1, 3.0, 4.0)
    @test pbv(1.0) == (NoRData(), NoRData(), NoRData(), 2.0, 2.0)

    # Exits disagreeing on their shadow type must bail — never crash, and never hand back the
    # primal-derived shadow where the constant's belongs.
    err = try
        rrule!!(zero_fcodual(rec_ret_c), Ctx(), xcd, const_codual(3.0), ncd)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("disagree on their shadow type", err.msg)
    # The same primal with no constant argument is unaffected.
    @test rev_gradient(rec_ret_c, 2.0, 3.0, 2)[3] == 1.0
end

@testset "operand types come from the value a node names, not from the node" begin
    # `_optype_w` is the single place that answers this.
    interp = DifferReverse.build_reverse_interp()
    world = Core.Compiler.get_inference_world(interp)
    @test DifferReverse._optype_w(nothing, world, GlobalRef(@__MODULE__, :CONST_G)) === typeof(sin)
    @test DifferReverse._optype_w(nothing, world, GlobalRef(@__MODULE__, :nonconst_g)) === Any
    @test DifferReverse._optype_w(nothing, world, QuoteNode(:a)) === Symbol
    @test DifferReverse._optype_w(nothing, world, 1.5) === Float64
    # A `Core.Const`-narrowed argument type widens to a bare `Type`: the callee guard tests
    # `isconcretetype`, which a lattice element fails. `_optype` only reads `argtypes`/`stmts` off
    # `pir`, so a stand-in carrying just `argtypes` exercises the path exactly.
    @test DifferReverse._optype_w((; argtypes=Any[Core.Const(3)]), world, Core.Argument(1)) === Int

    # End to end: a `const` global function operand differentiates.
    v = [0.4, -1.1, 2.5]
    _, dv = rev_gradient(usecg, v)
    @test dv ≈ cos.(v)
    checkverify_rev(usecg, (Vector{Float64},))
    check_stack_balance(usecg, v)
end

@testset "a UnionAll specTypes must not crash the compile" begin
    # Regression: reverse mode bails cleanly (the operand is `Any`); forward mode dispatches
    # dynamically and gets a real answer. Both paths previously died inside `finishinfer!` with
    # `FieldError: type UnionAll has no field parameters` (a MethodInstance with free typevars).
    v = [0.4, -1.1, 2.5]

    err = try
        rev_gradient(usencg, v)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test !(err isa FieldError)
    @test occursin("non-concrete argument type", err.msg)

    d = frule!!(Dual(usencg, NoTangent()), Dual(v, ones(3)))
    @test d.x ≈ sum(sin, v)
    @test d.dx ≈ sum(cos, v)
    checkverify(usencg, (Vector{Float64},))
end

@testset "reverse mode: dynamic (non-statically-resolvable) callee bails" begin
    @test_throws ErrorException rev_gradient(dyncallee, 1.0)
    # Assert the exception is specifically not a MethodError or other crash, which a bare
    # @test_throws ErrorException wouldn't distinguish.
    #
    # dyn_g is a non-const global, so its read has no statically-known value. `_static_recursible_call`
    # falls back to the operand's type instead of bailing immediately (what lets an argument-position
    # callee like sum(sin, v)'s f recurse) — here that fallback type is itself `Any`, so the call
    # still bails, just on the next guard down: "not a concrete DataType".
    err_dyncall = try
        rev_gradient(dyncallee, 1.0)
        nothing
    catch e
        e
    end
    @test err_dyncall isa ErrorException
    @test !(err_dyncall isa MethodError)
    @test occursin("is not a concrete DataType", err_dyncall.msg)
end

@testset "reverse mode: build_ctx reports the recorded bail reason" begin
    # `dyncallee` (dynamic-dispatch callee, above) reliably bails; a vararg primal (`vasum`) used to
    # be this test's vehicle but no longer bails at all now that reverse mode supports vararg
    # primals — see the vararg testset below.
    err = try
        build_ctx(dyncallee, (Float64,))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("is not a concrete DataType", err.msg)
    @test occursin("dyncallee", err.msg)
end

@testset "reverse mode: vararg primal" begin
    # `_static_recursible_call`'s guards apply to a vararg call the same as any other; `vasum`
    # itself, and the flat<->packed argument-space split (`resolve_reverse_primal`, the fwds/
    # pullback prologues), are what let this build at all.
    _, dx2, dx3 = rev_gradient(vasum, 2.0, 3.0)
    @test dx2 == 1.0 && dx3 == 1.0
    checkverify_rev(vasum, (Float64, Float64))
    check_stack_balance(vasum, 2.0, 3.0)

    # 0 trailing arguments: the packed tail is `Tuple{}`, the `NoTangent`/`NoFData` collapse case.
    # (`vasum` itself can't be called with zero args: `sum(())` throws in plain Julia too.)
    vasum0(x, ys...) = x * sum(ys; init = 1.0)
    _, dx0 = rev_gradient(vasum0, 3.0)
    @test dx0 == 1.0
    checkverify_rev(vasum0, (Float64,))
    check_stack_balance(vasum0, 3.0)

    # All-`NoTangent` trailing arguments (an `Int` vararg): every trailing gradient is `NoTangent`.
    viasum(x, ys::Int...) = x * sum(ys; init = 0)
    _, dxv, dy1, dy2 = rev_gradient(viasum, 1.5, 2, 3)
    @test dxv == 5.0   # sum(ys) == 5
    @test dy1 === NoTangent() && dy2 === NoTangent()
    checkverify_rev(viasum, (Float64, Int, Int))
    check_stack_balance(viasum, 1.5, 2, 3)

    # fdata-carrying trailing arguments (an array vararg): the scatter case — one packed rdata
    # accumulator (trivial here, arrays have no rdata) plus the per-argument array shadows. Loops
    # over the packed tuple by dynamic index, the same shape `vcat`'s own body uses — `sum(f,
    # ::Tuple)` recurses into `Base.afoldl` (itself vararg) and currently hits a separate, unrelated
    # `verify_ir` gap; not this feature's concern.
    @noinline function vvsum(x::Float64, vs::Vector{Float64}...)
        s = 0.0
        for j in 1:length(vs)
            v = vs[j]
            for i in eachindex(v)
                s += v[i]
            end
        end
        return x * s
    end
    v1, v2 = [1.0, 2.0], [3.0, 4.0]
    _, dxvv, dv1, dv2 = rev_gradient(vvsum, 2.0, v1, v2)
    @test dxvv == sum(v1) + sum(v2)
    @test dv1 == fill(2.0, 2) && dv2 == fill(2.0, 2)
    for k in eachindex(v1)
        v1p = copy(v1); v1p[k] += 1e-6
        v1m = copy(v1); v1m[k] -= 1e-6
        @test dv1[k] ≈ (vvsum(2.0, v1p, v2) - vvsum(2.0, v1m, v2)) / 2e-6 rtol = 1e-5
    end
    checkverify_rev(vvsum, (Float64, Vector{Float64}, Vector{Float64}))
    check_stack_balance(vvsum, 2.0, v1, v2)
end

@testset "reverse mode: dynamic (non-literal) getfield index" begin
    # `for i in 1:2` doesn't unroll, so t[i] reaches the pullback as a genuine dynamic index.
    # Regression: resolving the field to a raw, unresolved SSAValue instead of its runtime value
    # used to silently degenerate `increment_field!!`'s Val-based dispatch into a no-op — gradient
    # (0.0,0.0) instead of (1.0,1.0), with no error at all. d/dt_1 = d/dt_2 = 1.
    function tupsum_dyn(t::Tuple{Float64,Float64})
        s = 0.0
        for i in 1:2
            s += t[i]
        end
        return s
    end
    function ntupsum_dyn(t::@NamedTuple{a::Float64,b::Float64})
        s = 0.0
        for i in 1:2
            s += t[i]
        end
        return s
    end
    mutable struct MP2; x::Float64; y::Float64; end
    mp2get(m::MP2, i::Int) = Core.getfield(m, i)     # index is a genuine `Argument` (`_3`)
    function mp2sum_dyn(m::MP2)                       # loop index is a genuine `SSAValue` (a phi)
        s = 0.0
        for i in 1:2
            s += Core.getfield(m, i)
        end
        return s
    end
    function setdyn!(m::MP2, i::Int, v::Float64)
        Core.setfield!(m, i, v)
        return m.x + m.y
    end
    # Heterogeneous struct/mutable struct, dynamic `getfield` index: the genuinely hard case
    # (per-field tangent types differ) that must always bail, never miscompile.
    struct Het2; a::Float64; b::Int; end
    hetdyn(h::Het2, i::Int) = Core.getfield(h, i)
    mutable struct MHet2; a::Float64; b::Int; end
    mhetdyn(m::MHet2, i::Int) = Core.getfield(m, i)

    _, dt_dyn = rev_gradient(tupsum_dyn, (3.0, 4.0))
    @test dt_dyn == (1.0, 1.0)
    @test dt_dyn[1] == frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), (1.0, 0.0))).dx
    @test dt_dyn[2] == frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), (0.0, 1.0))).dx
    h = 1e-6
    @test dt_dyn[1] ≈ (tupsum_dyn((3.0 + h, 4.0)) - tupsum_dyn((3.0 - h, 4.0))) / 2h rtol = 1e-5
    @test dt_dyn[2] ≈ (tupsum_dyn((3.0, 4.0 + h)) - tupsum_dyn((3.0, 4.0 - h))) / 2h rtol = 1e-5
    checkverify_rev(tupsum_dyn, (Tuple{Float64,Float64},))

    # same, over a homogeneous NamedTuple.
    _, dnt_dyn = rev_gradient(ntupsum_dyn, (a=3.0, b=4.0))
    @test dnt_dyn == (a=1.0, b=1.0)
    checkverify_rev(ntupsum_dyn, (@NamedTuple{a::Float64,b::Float64},))

    # dynamic getfield index into a homogeneous mutable struct: the field's rdata contribution
    # routes into the object's own `MutableTangent` via the runtime-`Int` `increment_field_rdata!`
    # (a mutable struct has no object-level `RData`). The gradient w.r.t. the struct is a one-hot
    # `MutableTangent` for a single selected field, all-ones when summed over both.
    _, dmp2_r, _ = rev_gradient(mp2get, MP2(3.0, 4.0), 2)
    @test get_tangent_field(dmp2_r, :x) == 0.0 && get_tangent_field(dmp2_r, :y) == 1.0
    _, dmp1_r, _ = rev_gradient(mp2get, MP2(3.0, 4.0), 1)
    @test get_tangent_field(dmp1_r, :x) == 1.0 && get_tangent_field(dmp1_r, :y) == 0.0
    _, dms_r = rev_gradient(mp2sum_dyn, MP2(3.0, 4.0))
    @test get_tangent_field(dms_r, :x) == 1.0 && get_tangent_field(dms_r, :y) == 1.0
    checkverify_rev(mp2get, (MP2, Int))
    checkverify_rev(mp2sum_dyn, (MP2,))

    # regression: a dynamic getfield index into a HETEROGENEOUS struct is the genuinely hard case.
    # Must bail with a located error, never crash or silently return a wrong/zero gradient.
    @test_throws "dynamic (non-literal) field index" rev_gradient(hetdyn, Het2(1.0, 2), 1)
    err_het = try
        rev_gradient(hetdyn, Het2(1.0, 2), 1)
        nothing
    catch e
        e
    end
    @test err_het isa ErrorException
    @test !(err_het isa MethodError)
    @test occursin("dynamic (non-literal) field index", err_het.msg)

    # Same heterogeneous case through the mutable-struct branch of getfield's comms rule — a
    # separate code path from the immutable case above, must bail identically.
    err_mhet = try
        rev_gradient(mhetdyn, MHet2(1.0, 2), 1)
        nothing
    catch e
        e
    end
    @test err_mhet isa ErrorException
    @test !(err_mhet isa MethodError)
    @test occursin("dynamic (non-literal) field index", err_mhet.msg)

    # regression: a dynamic setfield! index always bails (only a literal index is supported).
    @test_throws "dynamic (non-literal) field index" rev_gradient(setdyn!, MP2(1.0, 2.0), 1, 5.0)
end

# ===========================================================================
# Nested-tape recycling: a non-inlined/recursive inner call's tape is recycled from the slot in the
# caller's own comms Stack that its next :subtape push will land in (`_inner_ctx`/`_alloc_tape`),
# instead of a fresh Ctx() allocating one every call.
# ===========================================================================

@testset "reverse mode: nested-tape recycling — steady-state allocation" begin
    # Ordinary, non-self-recursive inner calls: once a pre-allocated context's comms slots are
    # warm, a round trip through a non-inlined callee allocates nothing. n=5 keeps
    # Base.mapreduce_impl (which sum / sum(abs2, ·) fall through to, since this test file never
    # loads rules_perf_backstop.jl) below Base.pairwise_blocksize, so self-recursion (tested
    # separately below) isn't exercised here.
    @noinline sq_steady(x::Float64) = x * x
    callshelper_steady(x::Float64) = sq_steady(x) + sq_steady(x + 1.0)

    for (f, args) in ((x -> sum(abs2, x), (rand(5),)),
                      (x -> sum(x), (rand(5),)),
                      (callshelper_steady, (1.3,)))
        ctx = build_ctx(f, map(DifferReverse._typeof, args))
        fcd = zero_fcodual(f)
        argcds = map(zero_fcodual, args)
        rev_gradient!(ctx, fcd, argcds...)   # warm the slots
        rev_gradient!(ctx, fcd, argcds...)
        @test (@allocated rev_gradient!(ctx, fcd, argcds...)) == 0
    end
end

@testset "reverse mode: nested-tape recycling — distinct tapes per iteration" begin
    # Regression for the peek-position invariant recycling rests on: `_inner_ctx` reads from
    # `stack.position + 1`, which advances with every execution of the block, so N executions of a
    # loop body calling a non-inlined helper must land in N distinct comms slots, never aliasing
    # the same inner tape across iterations within one call.
    @noinline addone_dtc(x::Float64) = x + 1.0
    function loopcall_dtc(v::Vector{Float64})
        s = 0.0
        for i in eachindex(v)
            s += addone_dtc(v[i])
        end
        return s
    end
    N = 6
    v = collect(1.0:N)
    pctx = build_ctx(loopcall_dtc, (Vector{Float64},))
    rev_gradient!(pctx, zero_fcodual(loopcall_dtc), zero_fcodual(v))
    rev_gradient!(pctx, zero_fcodual(loopcall_dtc), zero_fcodual(v))   # second call: slots now recycled

    # Locate the block's comms Stack (its element type is a 1-tuple of addone_dtc's own tape type)
    # and pull out the N tapes the last forward pass used. A Stack never shrinks, so they're still
    # sitting in memory[1:N] even though position is back at 0.
    is_subtape_stack(s) = s isa DifferReverse.Stack && eltype(s.memory) <: Tuple &&
                          any(F -> F <: DifferReverse.Tape, fieldtypes(eltype(s.memory)))
    matches = filter(is_subtape_stack, collect(pctx.tape.comms))
    @test length(matches) == 1
    subtape_stack = only(matches)
    tapes = [only(subtape_stack.memory[i]) for i in 1:N]
    @test allunique(tapes)
end

@testset "reverse mode: nested-tape recycling — slot growth" begin
    # A longer call through the same context must grow into slots this context has never used
    # before (the fresh-allocation arm of `_inner_ctx`) rather than reuse/alias a shorter call's
    # stale slot, and a subsequent short call must still get the right answer once grown.
    @noinline addone_sg(x::Float64) = x + 1.0
    function loopcall_sg(v::Vector{Float64})
        s = 0.0
        for i in eachindex(v)
            s += addone_sg(v[i])
        end
        return s
    end
    v_short = [1.0, 2.0]
    v_long = collect(1.0:10.0)
    ctx = build_ctx(loopcall_sg, (Vector{Float64},))

    g_short = rev_gradient!(ctx, zero_fcodual(loopcall_sg), zero_fcodual(v_short))
    @test g_short[2] == rev_gradient(loopcall_sg, v_short)[2]
    g_long = rev_gradient!(ctx, zero_fcodual(loopcall_sg), zero_fcodual(v_long))
    @test g_long[2] == rev_gradient(loopcall_sg, v_long)[2]
    g_short2 = rev_gradient!(ctx, zero_fcodual(loopcall_sg), zero_fcodual(v_short))
    @test g_short2[2] == rev_gradient(loopcall_sg, v_short)[2]
end

@testset "reverse mode: nested-tape recycling — self-recursion" begin
    # Base.mapreduce_impl splits pairwise above Base.pairwise_blocksize, so sum(abs2, x) at n=2000
    # reaches its self-recursive branch, the case `own_TapeT` self-edge retargeting
    # (`reverse_fwds_recursive_ci`) is for.
    #
    # Steady-state allocation is exactly 0 (vs. 111056 B / 87 allocs at this size before tape
    # recycling existed).
    f_self = x -> sum(abs2, x)
    n = 2000
    @assert n > Base.pairwise_blocksize(abs2, Base.add_sum)
    x = rand(n)

    _, dx = rev_gradient(f_self, x)
    @test dx ≈ 2 .* x

    ctx = build_ctx(f_self, (Vector{Float64},))
    fcd, xcd = zero_fcodual(f_self), zero_fcodual(x)
    rev_gradient!(ctx, fcd, xcd)   # warm
    rev_gradient!(ctx, fcd, xcd)
    bytes = @allocated rev_gradient!(ctx, fcd, xcd)
    @test bytes == 0

    # Not vacuous: confirm the mechanism was genuinely exercised, not zero because nothing pushed to
    # subtapes at all. Locate the innermost Base.mapreduce_impl tape (self-recursive primal) and
    # assert its own subtapes stack actually holds recycled entries.
    function find_self_recursive_tape(tape, seen=Base.IdSet{Any}())
        tape in seen && return nothing
        push!(seen, tape)
        length(tape.subtapes.memory) > 0 && return tape
        for s in tape.comms
            s isa DifferReverse.Stack || continue
            for i in eachindex(s.memory)
                isassigned(s.memory, i) || continue
                for v in s.memory[i]
                    v isa DifferReverse.Tape || continue
                    found = find_self_recursive_tape(v, seen)
                    found === nothing || return found
                end
            end
        end
        return nothing
    end
    self_tape = find_self_recursive_tape(ctx.tape)
    @test self_tape !== nothing
    @test length(self_tape.subtapes.memory) > 0
end

@testset "reverse mode: nested-tape recycling — self-recursion across distinct blocks" begin
    # mapreduce_impl's two self-recursive call sites share one block. This is the complementary
    # case: two self-recursive call sites in different, mutually-exclusive blocks of one primal,
    # sharing the same global, Tape-wide `subtapes` stack.
    checkverify_rev(evenodd, (Int, Float64))
    check_stack_balance(evenodd, 9, 1.0)

    ctx = build_ctx(evenodd, (Int, Float64))
    g1 = rev_gradient!(ctx, zero_fcodual(evenodd), zero_fcodual(9), zero_fcodual(1.0))
    g2 = rev_gradient!(ctx, zero_fcodual(evenodd), zero_fcodual(9), zero_fcodual(1.0))
    @test g1 == g2
    @test g1 == rev_gradient(evenodd, 9, 1.0)
end

@testset "reverse mode: nested-tape recycling — hand rule callee still gets a fresh Ctx()" begin
    # A callee with a hand-written rrule!! (its pullback is whatever's cheapest to remember, not a
    # Tape, so there is no tape to pre-allocate) must still receive a fresh Ctx() at its call site,
    # never a recycled one from _inner_ctx.
    plus1_hr(x) = sin(x) + 1
    ir = code_reverse_fwds_ircode(plus1_hr, (Float64,))[1]
    invokes_to_rrule = [stmt for stmt in ir.stmts.stmt
                        if isa(stmt, Expr) && stmt.head === :invoke &&
                           length(stmt.args) >= 4 && stmt.args[2] === rrule!!]
    @test length(invokes_to_rrule) == 1
    ctx_arg = only(invokes_to_rrule).args[4]
    @test ctx_arg isa DifferReverse.Ctx{Nothing}
end

@testset "reverse mode: recursion into an fdata-carrying immutable argument" begin
    v = [1.0, 2.0, 3.0]

    _, dw = rev_gradient(recw_outer, v)
    @test dw ≈ [1.0, 1.0, 1.0]
    checkverify_rev(recw_outer, (Vector{Float64},))
    check_stack_balance(recw_outer, v)

    _, dm = rev_gradient(recm_outer, v)
    @test dm ≈ [7.0, 1.0, 1.0]
    checkverify_rev(recm_outer, (Vector{Float64},))
    check_stack_balance(recm_outer, v)

    _, dt = rev_gradient(rectup_outer, v)
    @test dt ≈ [5.0, 5.0, 5.0]
    checkverify_rev(rectup_outer, (Vector{Float64},))
    check_stack_balance(rectup_outer, v)

    _, dnt = rev_gradient(recnt_outer, v)
    @test dnt ≈ [21.0, 3.0, 3.0]
    checkverify_rev(recnt_outer, (Vector{Float64},))
    check_stack_balance(recnt_outer, v)

    # Reached only through a global, so inactive: replayed primally, and the gradient comes from
    # `sum(x)` alone. The replayed call still computes the real primal value.
    ctx = build_ctx(recw_untraced_outer, (Vector{Float64},))
    y, gs = value_and_gradient!(ctx, zero_fcodual(recw_untraced_outer), zero_fcodual(v))
    @test y ≈ recw_untraced_outer(v)
    @test gs[2] ≈ [1.0, 1.0, 1.0]
    check_stack_balance(recw_untraced_outer, v)

    err = try
        build_ctx(recmix_untraced_outer, (Vector{Float64},))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("provenance is not traceable", err.msg)
end
