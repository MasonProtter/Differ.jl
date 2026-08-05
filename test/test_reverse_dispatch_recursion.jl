using Test
using Differ
using Differ: Dual, NoTangent, frule!!, gradient, get_tangent_field, MutableTangent

include(joinpath(@__DIR__, "testutils.jl"))

# A dynamic-dispatch callee (read from a non-`const` global) is not statically recursible — must
# bail cleanly, not crash. Must be a real module-level global (not testset-local) — a local would
# instead be captured as a closure field, a different IR shape from what this test exercises.
dyn_g = sin
dyncallee(x) = dyn_g(x)

# Regression: a self-recursive `@noinline` primal has no finite `Tape` type in this design (the
# `in_progress` cycle guard) — must bail with a clean `ErrorException`, not recurse forever /
# stack-overflow. Must be a true top-level function, not testset-local: a *self*-recursive
# function defined in a local scope needs to close over its own (boxed) binding to call itself,
# which gives it a real (non-`NoTangent`) tangent type instead of the plain singleton a top-level
# function has, defeating `gradient`'s zero-tangent seeding for `f`.
@noinline function rec_self(x::Float64, n::Int)
    n <= 0 && return x
    return rec_self(x, n - 1)
end

# Same requirement as `rec_self` above (must be module-level, not testset-local): a genuinely
# self-recursive primal used across several ISSUES #65 tests below (plain recursion, under a branch,
# under a loop, and through both the tape-allocating and pre-allocated carrier specializations).
@noinline function rec_sum(v::Vector{Float64}, i::Int)
    i > length(v) && return 0.0
    return v[i] + rec_sum(v, i + 1)
end

# Genuine mutual recursion (A -> B -> A), also module-level for the same reason. Must still bail
# cleanly — it needs a tape-type pre-pass across the whole SCC, out of scope for direct self-recursion.
@noinline mutA(x::Float64, n::Int) = n <= 0 ? x : mutB(x, n - 1)
@noinline mutB(x::Float64, n::Int) = mutA(x, n - 1)

# A `const` global holding a function, passed as an *operand* of a surviving call: the operand is a
# `GlobalRef` in the optimized IR, so its type has to come from the binding, not from the node (the
# reverse-mode half of ISSUES #63). Module-level `const` for the same reason `dyn_g` is module-level.
const CONST_G = sin
usecg(v) = sum(CONST_G, v)

# The non-`const` twin. Its operand survives as its own `%k = Main.dyn_g` load statement typed
# `Any`, so this must bail (reverse) or dispatch dynamically (forward) — and, before the
# `isa(specTypes, DataType)` guard in `is_reverse_fwds_impl`/`is_dualized_impl`, it crashed the
# compile outright with `FieldError: type UnionAll has no field parameters` from inside
# `finishinfer!`, because inference reaches a MethodInstance whose signature has free typevars.
nonconst_g = sin
usencg(v) = sum(nonconst_g, v)

@testset "reverse mode: recursion into a hand-written rule" begin
    # A surviving high-level call differentiates via the recursive `rrule` support below (`sin`
    # specifically resolves to the hand-written rule in `src/rrules.jl`, not raw recursion into
    # `Base.Math.sin`'s internals — see that file's header).
    plus1(x) = sin(x) + 1
    nest(x)  = sin(cos(x))     # composed hand rules: sin(cos(x))

    _, dx_plus1 = gradient(plus1, 1.3)
    @test dx_plus1 ≈ cos(1.3)

    checkverify_rev(plus1, (Float64,))
    checkverify_rev(nest, (Float64,))
    check_stack_balance(plus1, 1.3)
    check_stack_balance(nest, 0.4)
end

@testset "reverse mode: recursive rrule calls (statically-resolvable only)" begin
    # Tier 5 Part 1: a genuinely separate, non-inlined callee must actually survive as a
    # surviving `:invoke` to exercise recursion, not just fold into ordinary arithmetic — so it's
    # `@noinline`.
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

    _, dx_rec = gradient(rec_call, 3.0)
    @test dx_rec ≈ 2*3.0 + 1
    @test dx_rec ≈ frule!!(Dual(rec_call, NoTangent()), Dual(3.0, 1.0)).dx

    for x in (2.0, -2.0)
        _, dx_rb = gradient(rec_branch, x)
        @test dx_rb ≈ 2x
    end

    _, dx_rl = gradient(rec_loop, 2.0, 3)
    @test dx_rl ≈ 3 * 2 * 2.0
    check_stack_balance(rec_loop, 2.0, 5)

    checkverify_rev(rec_call, (Float64,))
    checkverify_rev(rec_branch, (Float64,))
    checkverify_rev(rec_loop, (Float64, Int))
end

@testset "reverse mode: direct self-recursion (ISSUES #65)" begin
    # `rec_self` is genuinely self-recursive: no hand `rrule!!`, and the recursive call resolves to
    # the exact primal currently being differentiated — the case the `in_progress` cycle guard used
    # to bail on unconditionally. `d(rec_self(x,n))/dx = 1` for every `n >= 0`.
    _, dx_self, dn_self = gradient(rec_self, 1.0, 3)
    @test dx_self == 1.0
    @test dn_self isa NoTangent
    for n in 0:4
        _, dx_n, _ = gradient(rec_self, 2.7, n)
        @test dx_n == 1.0
    end
    checkverify_rev(rec_self, (Float64, Int))
    # Exercises both the tape-allocating (`Ctx{Nothing}`) and pre-allocated (`Ctx{<:Tape}`) carrier
    # specializations — the self-edge always targets the *pre-allocated* one (nested-tape-recycling
    # plan, Stage 2: recycling the inner tape needs a recycled-typed ctx on both ends), even when the
    # outer call is the tape-allocating carrier, which is what makes this the discriminating
    # regression: a self-recursive primal's tape-allocating carrier must actually compile the
    # pre-allocated sibling it recurses into, not assume it's the literal carrier being compiled (only
    # true when the outer carrier is *already* the pre-allocated one).
    check_stack_balance(rec_self, 1.0, 3)

    # Accumulating self-recursion: each level contributes a distinct addend, so a wrong tape layout
    # or a stale/misrouted comms value would show up as a wrong (not just uniformly-1) gradient.
    v = [1.0, 2.0, 3.0, 4.0]
    _, dv, _ = gradient(rec_sum, v, 1)
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
    _, dvb, _ = gradient(rec_sum_branch, v, true)
    @test dvb == ones(4)
    _, dvb0, _ = gradient(rec_sum_branch, v, false)
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
    _, dvl, _ = gradient(rec_sum_loop, v, 3)
    @test dvl == fill(3.0, 4)
    checkverify_rev(rec_sum_loop, (Vector{Float64}, Int))
    check_stack_balance(rec_sum_loop, v, 3)

    # Mutual recursion (A -> B -> A) is explicitly out of scope (needs a tape-type pre-pass across
    # the whole SCC) and must still bail cleanly — never hang, never crash on `verify_ir`.
    err_mut = try
        gradient(mutA, 1.0, 4)
        nothing
    catch e
        e
    end
    @test err_mut isa ErrorException
    @test occursin("self- or mutually-recursive primal", err_mut.msg)
end

@testset "operand types come from the value a node names, not from the node" begin
    # ISSUES #63, reverse-mode half. `_optype_w` is the single place that answers this.
    interp = Differ.ADInterpreter{Differ.Reverse}()
    world = Core.Compiler.get_inference_world(interp)
    @test Differ._optype_w(nothing, world, GlobalRef(@__MODULE__, :CONST_G)) === typeof(sin)
    @test Differ._optype_w(nothing, world, GlobalRef(@__MODULE__, :nonconst_g)) === Any
    @test Differ._optype_w(nothing, world, QuoteNode(:a)) === Symbol
    @test Differ._optype_w(nothing, world, 1.5) === Float64
    # A `Core.Const`-narrowed argument type widens to a bare `Type` (ISSUES #57): the callee guard
    # tests `isconcretetype`, which a lattice element fails. `_optype` only reads `argtypes`/`stmts`
    # off `pir`, so a stand-in carrying just `argtypes` exercises the path exactly.
    @test Differ._optype_w((; argtypes=Any[Core.Const(3)]), world, Core.Argument(1)) === Int

    # End to end: a `const` global function operand differentiates.
    v = [0.4, -1.1, 2.5]
    _, dv = gradient(usecg, v)
    @test dv ≈ cos.(v)
    checkverify_rev(usecg, (Vector{Float64},))
    check_stack_balance(usecg, v)
end

@testset "a UnionAll specTypes must not crash the compile" begin
    # Regression for the `FieldError: type UnionAll has no field parameters` escape described above.
    # Reverse mode bails cleanly (the operand is `Any`); forward mode dispatches dynamically and
    # gets a real answer. Both paths previously died inside `finishinfer!` instead.
    v = [0.4, -1.1, 2.5]

    err = try
        gradient(usencg, v)
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
    @test_throws ErrorException gradient(dyncallee, 1.0)
    # Asserting the exception is a located `ErrorException` naming the construct, not just any
    # `ErrorException` — and explicitly not a `MethodError` or other crash, which a bare
    # `@test_throws ErrorException` would not distinguish from.
    #
    # `dyn_g` is a non-`const` global, so its read has no statically-known *value* (`_calleeval`
    # returns `nothing`) — but `_static_recursible_call` (`src/reverse_interp.jl`) now falls back to
    # the operand's *type* instead of bailing immediately on that (this is what lets an
    # argument-position callee like `sum(sin, v)`'s `f` recurse). Here that fallback type is itself
    # `Any` (an unannotated mutable global has no static type either), so the call still bails, just
    # on the next guard down: "not a concrete DataType", rather than "dynamic callee" directly.
    err_dyncall = try
        gradient(dyncallee, 1.0)
        nothing
    catch e
        e
    end
    @test err_dyncall isa ErrorException
    @test !(err_dyncall isa MethodError)
    @test occursin("is not a concrete DataType", err_dyncall.msg)
end

@testset "reverse mode: dynamic (non-literal) getfield index" begin
    # Dynamic (non-literal) `getfield` index (Phase B): `for i in 1:2` does not unroll, so `t[i]`
    # reaches the pullback as a genuine dynamic index. Before the fix, the pullback resolved the
    # field to a raw, unresolved `SSAValue` instead of its runtime value, silently degenerating
    # `increment_field!!`'s `Val`-based dispatch into a no-op — the gradient came back
    # `(0.0,0.0)` instead of `(1.0,1.0)`, with no error at all. `d/dt_1 = d/dt_2 = 1`.
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

    _, dt_dyn = gradient(tupsum_dyn, (3.0, 4.0))
    @test dt_dyn == (1.0, 1.0)
    @test dt_dyn[1] == frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), (1.0, 0.0))).dx
    @test dt_dyn[2] == frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), (0.0, 1.0))).dx
    h = 1e-6
    @test dt_dyn[1] ≈ (tupsum_dyn((3.0 + h, 4.0)) - tupsum_dyn((3.0 - h, 4.0))) / 2h rtol = 1e-5
    @test dt_dyn[2] ≈ (tupsum_dyn((3.0, 4.0 + h)) - tupsum_dyn((3.0, 4.0 - h))) / 2h rtol = 1e-5
    checkverify_rev(tupsum_dyn, (Tuple{Float64,Float64},))

    # same, over a homogeneous NamedTuple.
    _, dnt_dyn = gradient(ntupsum_dyn, (a=3.0, b=4.0))
    @test dnt_dyn == (a=1.0, b=1.0)
    checkverify_rev(ntupsum_dyn, (@NamedTuple{a::Float64,b::Float64},))

    # dynamic getfield index into a homogeneous MUTABLE struct (Part 2b): the field's rdata
    # contribution routes into the object's own `MutableTangent` via the runtime-`Int`
    # `increment_field_rdata!` (not an object-level `RData`, as a mutable struct has none). The
    # gradient w.r.t. the struct is a one-hot `MutableTangent` for a single selected field, and
    # all-ones when summed over both. Index is a genuine `Argument`/`SSAValue`, not const-folded.
    _, dmp2_r, _ = gradient(mp2get, MP2(3.0, 4.0), 2)
    @test get_tangent_field(dmp2_r, :x) == 0.0 && get_tangent_field(dmp2_r, :y) == 1.0
    _, dmp1_r, _ = gradient(mp2get, MP2(3.0, 4.0), 1)
    @test get_tangent_field(dmp1_r, :x) == 1.0 && get_tangent_field(dmp1_r, :y) == 0.0
    _, dms_r = gradient(mp2sum_dyn, MP2(3.0, 4.0))
    @test get_tangent_field(dms_r, :x) == 1.0 && get_tangent_field(dms_r, :y) == 1.0
    checkverify_rev(mp2get, (MP2, Int))
    checkverify_rev(mp2sum_dyn, (MP2,))

    # regression: a dynamic getfield index into a HETEROGENEOUS struct is the genuinely hard case
    # — must bail with a located error, never crash or silently return a wrong/zero gradient.
    @test_throws "dynamic (non-literal) field index" gradient(hetdyn, Het2(1.0, 2), 1)
    err_het = try
        gradient(hetdyn, Het2(1.0, 2), 1)
        nothing
    catch e
        e
    end
    @test err_het isa ErrorException
    @test !(err_het isa MethodError)
    @test occursin("dynamic (non-literal) field index", err_het.msg)

    # regression: the same HETEROGENEOUS case through the mutable-struct branch of `getfield`'s
    # comms rule (`builtin_rrule_comms(::Val{Core.getfield},...)`, `src/builtins_reverse.jl`) —
    # a separate code path from the immutable case above, must bail identically.
    err_mhet = try
        gradient(mhetdyn, MHet2(1.0, 2), 1)
        nothing
    catch e
        e
    end
    @test err_mhet isa ErrorException
    @test !(err_mhet isa MethodError)
    @test occursin("dynamic (non-literal) field index", err_mhet.msg)

    # regression: a dynamic setfield! index — Phase A only, always bails.
    @test_throws "dynamic (non-literal) field index" gradient(setdyn!, MP2(1.0, 2.0), 1, 5.0)
end

# ===========================================================================
# Nested-tape recycling (Stages 1-2): a non-inlined/recursive inner call's tape is recycled from the
# slot in the caller's own comms `Stack` its next `:subtape` push will land in
# (`_inner_ctx`/`_alloc_tape`, `src/stack.jl`+`src/reverse_interp.jl`), instead of a fresh `Ctx()`
# allocating one every call. See that plan for the full writeup; these are its verification tests.
# ===========================================================================

@testset "reverse mode: nested-tape recycling — steady-state allocation" begin
    # Stage 1 (ordinary, non-self-recursive inner calls): once a pre-allocated context's comms slots
    # are warm, a round trip through a non-inlined callee allocates nothing. `n=5` keeps
    # `Base.mapreduce_impl` (which `sum`/`sum(abs2, ·)` fall through to — this test file never loads
    # `src/rules_perf_backstop.jl`, so neither has a hand rule) below `Base.pairwise_blocksize`, so
    # only Stage 1's machinery is exercised here; self-recursion is Stage 2, tested separately below.
    @noinline sq_steady(x::Float64) = x * x
    callshelper_steady(x::Float64) = sq_steady(x) + sq_steady(x + 1.0)

    for (f, args) in ((x -> sum(abs2, x), (rand(5),)),
                      (x -> sum(x), (rand(5),)),
                      (callshelper_steady, (1.3,)))
        ctx = build_ctx(f, map(Differ._typeof, args))
        fcd = zero_fcodual(f)
        argcds = map(zero_fcodual, args)
        gradient!(ctx, fcd, argcds...)   # warm the slots
        gradient!(ctx, fcd, argcds...)
        @test (@allocated gradient!(ctx, fcd, argcds...)) == 0
    end
end

@testset "reverse mode: nested-tape recycling — distinct tapes per iteration" begin
    # Regression for the peek-position argument the recycling design rests on: `_inner_ctx` reads
    # from `stack.position + 1`, which advances with every execution of the block — so N executions
    # of a loop body calling a non-inlined helper must land in N distinct comms slots, never aliasing
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
    gradient!(pctx, zero_fcodual(loopcall_dtc), zero_fcodual(v))
    gradient!(pctx, zero_fcodual(loopcall_dtc), zero_fcodual(v))   # second call: slots now recycled

    # Locate the block's comms `Stack` (its element type is a 1-tuple of `addone_dtc`'s own tape
    # type) and pull out the N tapes the last forward pass used — a `Stack` never shrinks, so they're
    # still sitting in `memory[1:N]` even though `position` is back at 0 (`check_stack_balance`
    # covers that balance separately).
    is_subtape_stack(s) = s isa Differ.Stack && eltype(s.memory) <: Tuple &&
                          any(F -> F <: Differ.Tape, fieldtypes(eltype(s.memory)))
    matches = filter(is_subtape_stack, collect(pctx.tape.comms))
    @test length(matches) == 1
    subtape_stack = only(matches)
    tapes = [only(subtape_stack.memory[i]) for i in 1:N]
    @test allunique(tapes)
end

@testset "reverse mode: nested-tape recycling — slot growth" begin
    # A longer call through the same context must grow into slots this context has never used
    # before (the fresh-allocation arm of `_inner_ctx`) rather than reuse/alias a shorter call's
    # stale slot — and a subsequent short call must still get the right answer once the context has
    # grown.
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

    g_short = gradient!(ctx, zero_fcodual(loopcall_sg), zero_fcodual(v_short))
    @test g_short[2] == gradient(loopcall_sg, v_short)[2]
    g_long = gradient!(ctx, zero_fcodual(loopcall_sg), zero_fcodual(v_long))
    @test g_long[2] == gradient(loopcall_sg, v_long)[2]
    g_short2 = gradient!(ctx, zero_fcodual(loopcall_sg), zero_fcodual(v_short))
    @test g_short2[2] == gradient(loopcall_sg, v_short)[2]
end

@testset "reverse mode: nested-tape recycling — self-recursion (Stage 2)" begin
    # `Base.mapreduce_impl` splits pairwise above `Base.pairwise_blocksize`, so `sum(abs2, x)` at
    # n=2000 reaches its self-recursive branch — the case Stage 2's `own_TapeT` self-edge retargeting
    # (`reverse_fwds_recursive_ci`, `src/reverse_interp.jl`) is for.
    #
    # Gradient correctness is the hard requirement. Allocation drops by ~3 orders of magnitude
    # relative to the fresh-`Ctx()` baseline this plan measured (111056 B / 87 allocs at this size
    # before Stages 1-2) but is NOT literally 0, and this is not a bug: reading a recycled tape back
    # out of a *self-recursive* comms slot still costs a small, roughly recursion-depth-proportional
    # allocation, because that slot's declared element type is necessarily the *abstract* bare `Tape`
    # UnionAll (never a concrete type — see `reverse_fwds_recursive_ci`'s docstring on why no fixed
    # point is solved for a self-edge), and reading a value out of an abstractly-typed comms `Stack`
    # allocates even when the underlying `Tape` object is genuinely being reused (confirmed directly:
    # a `Stack{Tuple{ConcreteType,ConcreteType}}` read allocates nothing, a `Stack{Tuple{Tape,Tape}}`
    # read allocates 16 bytes, for the exact same recycled objects). That is a distinct, narrower
    # cost than "a tape escaping recycling" — see the plan writeup for the follow-up this motivates.
    f_self = x -> sum(abs2, x)
    n = 2000
    @assert n > Base.pairwise_blocksize(abs2, Base.add_sum)
    x = rand(n)

    _, dx = gradient(f_self, x)
    @test dx ≈ 2 .* x

    ctx = build_ctx(f_self, (Vector{Float64},))
    fcd, xcd = zero_fcodual(f_self), zero_fcodual(x)
    gradient!(ctx, fcd, xcd)   # warm
    gradient!(ctx, fcd, xcd)
    bytes = @allocated gradient!(ctx, fcd, xcd)
    @test 0 < bytes < 2_000   # was 111056 B pre-Stages-1-2 — a residual, documented above, not a regression
end

@testset "reverse mode: nested-tape recycling — hand rule callee still gets a fresh Ctx()" begin
    # A callee with a hand-written `rrule!!` (its pullback is whatever's cheapest to remember, not a
    # `Tape` — there is no tape to pre-allocate) must still receive a fresh `Ctx()` at its call site,
    # never a recycled one from `_inner_ctx`.
    plus1_hr(x) = sin(x) + 1
    ir = code_reverse_fwds_ircode(plus1_hr, (Float64,))[1]
    invokes_to_rrule = [stmt for stmt in ir.stmts.stmt
                        if isa(stmt, Expr) && stmt.head === :invoke &&
                           length(stmt.args) >= 4 && stmt.args[2] === rrule!!]
    @test length(invokes_to_rrule) == 1
    ctx_arg = only(invokes_to_rrule).args[4]
    @test ctx_arg isa Differ.Ctx{Nothing}
end
