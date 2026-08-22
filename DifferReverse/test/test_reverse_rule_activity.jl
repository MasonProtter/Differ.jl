using Test
using DifferReverse
using DifferReverse: CoDual, Ctx, Inactive, NoFData, NoRData, NoTangent, rrule!!,
    primal, tangent, isactive, tangent_type, rdata_type, zero_rdata, zero_fcodual
using LinearAlgebra: dot, norm, tr, mul!
using SpecialFunctions

include(joinpath(@__DIR__, "testutils.jl"))

# Every hand-written `rrule!!` must accept an `Inactive()` in any differentiable slot, because the
# transform routes a constant operand straight to the rule rather than materialising a zero for it.
# A rule whose signature was never widened does not error — it silently stops matching and the call
# falls through to the `@generated` derived transform, which then re-derives a body (a BLAS `ccall`,
# an openspecfun `ccall`) it cannot see through. The dispatch assertion below is what makes that
# visible.
#
# The other half is what the pullback hands back. A slot the caller declared constant gets
# `NoRData()`, the same thing the derived path emits, so hand rules and derived rules return the
# same shape; an active slot gets rdata at exactly `rdata_type(tangent_type(P))`, which is the type
# the transform declared for it before the rule was compiled.

# A fixture is a thunk (mutating rules need fresh buffers per mask), the callee, and the two slots a
# rule treats specially: `dest`, a destination it must refuse when declared *constant*, and
# `frozen`, a parameter with no implemented derivative, which the rule refuses when it is *active*.
struct Fixture
    name::String
    f::Any
    args::Function
    dest::Union{Int,Nothing}
    frozen::Union{Int,Nothing}
end
Fixture(name, f, args; dest=nothing, frozen=nothing) = Fixture(name, f, args, dest, frozen)

const V = [1.0, 2.0, 3.0]
const W = [0.5, 1.5, 2.5]
const M = [1.0 2.0; 3.0 4.0]
const MASK = [true, false, true]
const IDX = [1, 3]

unary(f, x) = Fixture(string(f), f, () -> (x,))

# `Threads.@threads`'s worker, reproduced without the macro: a closure with a default positional
# argument, which Julia splits into a wrapper holding a body function that holds the loop's
# captures. `threading_run`'s rule is handed exactly this shape.
# A lock that is actually held, for `unlock`'s fixture.
held_lock() = (l = ReentrantLock(); lock(l); l)

# A completed task, for the fixtures whose rule would otherwise block on a fresh one.
done_task() = (t = Task(() -> nothing); schedule(t); wait(t); t)

# A zero-argument thunk with differentiable captures — the shape `Task`'s rule is handed.
audit_task_thunk(x, y) = () -> (y[1] = 2.0 * x[1]; nothing)

function audit_worker(x::Vector{Float64}, y::Vector{Float64})
    range = eachindex(y, x)
    function threadsfor_fun(tid=1; onethread=false)
        r = range
        len, rem = onethread ? (length(r), 0) : divrem(length(r), Threads.threadpoolsize())
        if len == 0
            tid > rem && return nothing
            len, rem = 1, 0
        end
        f = firstindex(r) + ((tid - 1) * len)
        l = f + len - 1
        if rem > 0
            if tid <= rem
                f += tid - 1
                l += tid
            else
                f += rem
                l += rem
            end
        end
        for i in f:l
            j = @inbounds r[i]
            y[j] = sin(x[j])
        end
        return nothing
    end
    return threadsfor_fun
end

const FIXTURES = Fixture[
    # rrules.jl
    unary(sin, 0.7), unary(cos, 0.7),
    # rules_math.jl — arguments kept inside each function's real domain
    unary(exp, 0.7), unary(log, 0.7), unary(log1p, 0.7), unary(expm1, 0.7),
    unary(log2, 0.7), unary(log10, 0.7), unary(exp2, 0.7), unary(exp10, 0.7),
    unary(sinh, 0.7), unary(cosh, 0.7), unary(tanh, 0.7),
    unary(asinh, 0.7), unary(acosh, 2.0), unary(atanh, 0.7),
    unary(asin, 0.7), unary(acos, 0.7), unary(atan, 0.7), unary(cbrt, 0.7),
    Fixture("atan(y,x)", atan, () -> (1.0, 2.0)),
    Fixture("^(x,y)", ^, () -> (2.0, 3.0)),
    Fixture("^(x,n::Int)", ^, () -> (2.0, 3)),
    Fixture("hypot", hypot, () -> (3.0, 4.0)),
    Fixture("sqrt(::Complex)", sqrt, () -> (1.0 + 2.0im,)),
    unary(abs, 0.7), unary(sign, 0.7), unary(abs2, 0.7), unary(inv, 0.7),
    Fixture("max", max, () -> (1.0, 2.0)),
    Fixture("min", min, () -> (1.0, 2.0)),
    Fixture("copysign", copysign, () -> (1.0, -2.0)),
    Fixture("fma", fma, () -> (1.0, 2.0, 3.0)),
    Fixture("muladd", muladd, () -> (1.0, 2.0, 3.0)),
    # rules_indexing.jl
    Fixture("getindex(A,mask)", getindex, () -> (copy(V), copy(MASK))),
    Fixture("getindex(A,idx)", getindex, () -> (copy(V), copy(IDX))),
    # rules_growable.jl — structure-only mutation, so an inactive array shadow is a legal no-op
    # rather than a refusal: no `dest` slot. `sizehint!` returns the array, so a constant one has its
    # zero tangent materialised rather than refused — there are no sources a fresh zero could discard.
    Fixture("_growend!", Base._growend!, () -> (copy(V), 2)),
    Fixture("_growbeg!", Base._growbeg!, () -> (copy(V), 2)),
    Fixture("_growat!", Base._growat!, () -> (copy(V), 2, 1)),
    Fixture("_deleteend!", Base._deleteend!, () -> (copy(V), 1)),
    Fixture("_deletebeg!", Base._deletebeg!, () -> (copy(V), 1)),
    Fixture("_deleteat!", Base._deleteat!, () -> (copy(V), 2, 1)),
    Fixture("sizehint!", sizehint!, () -> (copy(V), 8)),
    # rules_linalg.jl
    Fixture("dot", dot, () -> (copy(V), copy(W))),
    Fixture("norm", norm, () -> (copy(V),)),
    Fixture("tr", tr, () -> (copy(M),)),
    Fixture("*(M,v)", *, () -> (copy(M), [1.0, 2.0])),
    Fixture("*(M,M)", *, () -> (copy(M), copy(M))),
    Fixture("mul!(v,M,v)", mul!, () -> (zeros(2), copy(M), [1.0, 2.0]); dest=1),
    Fixture("mul!(M,M,M)", mul!, () -> (zeros(2, 2), copy(M), copy(M)); dest=1),
    # rules_reductions.jl
    Fixture("cumsum", cumsum, () -> (copy(V),)),
    Fixture("extrema", extrema, () -> (copy(V),)),
    # rules_broadcast.jl
    Fixture("map(f,x)", map, () -> (sin, copy(V))),
    Fixture("map(f,x,y)", map, () -> (atan, copy(V), copy(W))),
    Fixture("map!(f,d,x)", map!, () -> (sin, zeros(3), copy(V)); dest=2),
    Fixture("map!(f,d,x,y)", map!, () -> (atan, zeros(3), copy(V), copy(W)); dest=2),
    # DifferCoreSpecialFunctionsExt — arguments kept inside each function's real domain.
    unary(airyai, 0.7), unary(airyaix, 0.7), unary(airyaiprime, 0.7), unary(airyaiprimex, 0.7),
    unary(airybi, 0.7), unary(airybiprime, 0.7),
    unary(besselj0, 0.7), unary(besselj1, 0.7), unary(bessely0, 0.7), unary(bessely1, 0.7),
    Fixture("besselj(ν::Int,x)", besselj, () -> (2, 0.7)),
    Fixture("besseli(ν::Int,x)", besseli, () -> (2, 0.7)),
    Fixture("bessely(ν::Int,x)", bessely, () -> (2, 0.7)),
    Fixture("besselk(ν::Int,x)", besselk, () -> (2, 0.7)),
    Fixture("besselkx(ν::Int,x)", besselkx, () -> (2, 0.7)),
    Fixture("besselix(ν::Int,x)", besselix, () -> (2, 0.7)),
    Fixture("besseljx(ν::Int,x)", besseljx, () -> (2, 0.7)),
    Fixture("besselyx(ν::Int,x)", besselyx, () -> (2, 0.7)),
    # A real order has a tangent space, and no derivative with respect to it is implemented.
    Fixture("besselj(ν::Float64,x)", besselj, () -> (1.5, 0.7); frozen=1),
    unary(dawson, 0.7),
    unary(gamma, 0.7), unary(loggamma, 0.7), unary(digamma, 0.7), unary(trigamma, 0.7),
    unary(invdigamma, 0.7),
    Fixture("polygamma(m,x)", polygamma, () -> (2, 0.7)),
    Fixture("beta", beta, () -> (1.5, 2.5)),
    Fixture("logbeta", logbeta, () -> (1.5, 2.5)),
    Fixture("gamma(a,x)", gamma, () -> (1.5, 0.7); frozen=1),
    Fixture("loggamma(a,x)", loggamma, () -> (1.5, 0.7); frozen=1),
    unary(logabsgamma, 0.7),
    Fixture("gamma_inc(a,x,IND)", gamma_inc, () -> (1.5, 0.7, 0); frozen=1),
    unary(erf, 0.7), unary(erfc, 0.7), unary(logerfc, 0.7), unary(erfcx, 0.7),
    unary(logerfcx, 0.7), unary(erfi, 0.7), unary(erfinv, 0.7), unary(erfcinv, 0.7),
    Fixture("erf(x,y)", erf, () -> (0.3, 1.2)),
    unary(expint, 0.7), unary(expintx, 0.7), unary(expinti, 0.7),
    unary(sinint, 0.7), unary(cosint, 0.7),
    Fixture("expint(ν,x)", expint, () -> (2, 0.7)),
    Fixture("expintx(ν,x)", expintx, () -> (2, 0.7)),
    unary(ellipk, 0.7), unary(ellipe, 0.7),
    # rules_threads.jl — the parallel-region rule plus the thread/lock queries. A lock has no
    # tangent space at all (`tangent_type(ReentrantLock) === NoTangent`), so the lock slots carry no
    # activity mask; `unlock` needs a lock that is actually held.
    Fixture("threading_run", Base.Threads.threading_run,
            () -> (audit_worker(copy(V), zeros(3)), false)),
    Fixture("threadid", Threads.threadid, () -> ()),
    Fixture("threadid(::Task)", Threads.threadid, () -> (Task(() -> 1),)),
    Fixture("nthreads", Threads.nthreads, () -> ()),
    Fixture("nthreads(pool)", Threads.nthreads, () -> (:default,)),
    Fixture("threadpoolsize", Threads.threadpoolsize, () -> ()),
    Fixture("threadpoolsize(pool)", Threads.threadpoolsize, () -> (:default,)),
    Fixture("threadpool", Threads.threadpool, () -> ()),
    Fixture("threadpool(tid)", Threads.threadpool, () -> (Threads.threadid(),)),
    Fixture("maxthreadid", Threads.maxthreadid, () -> ()),
    Fixture("nthreadpools", Threads.nthreadpools, () -> ()),
    Fixture("lock", lock, () -> (ReentrantLock(),)),
    Fixture("unlock", unlock, () -> (held_lock(),)),
    Fixture("trylock", trylock, () -> (ReentrantLock(),)),
    Fixture("islocked", islocked, () -> (ReentrantLock(),)),
    Fixture("lock(f,l)", lock, () -> (() -> 1.0, ReentrantLock())),
    # rules_threads.jl — tasks & the `@sync` plumbing. `Task`'s thunk is the only differentiable
    # slot in the family; the fixtures' tasks are pre-completed wherever the rule would otherwise
    # block, and channel payloads carry no tangent (a differentiable one is refused by the rule).
    Fixture("Task(thunk)", Task, () -> (audit_task_thunk(copy(V), zeros(3)),)),
    Fixture("schedule", schedule, () -> (Task(() -> nothing),)),
    Fixture("wait(::Task)", wait, () -> (done_task(),)),
    Fixture("fetch(::Task)", fetch, () -> (done_task(),)),
    Fixture("istaskdone", istaskdone, () -> (done_task(),)),
    Fixture("istaskstarted", istaskstarted, () -> (done_task(),)),
    Fixture("istaskfailed", istaskfailed, () -> (done_task(),)),
    Fixture("_spawn_set_thrpool", Base.Threads._spawn_set_thrpool,
            () -> (Task(() -> nothing), :default)),
    Fixture("yield()", yield, () -> ()),
    Fixture("yield(::Task)", yield, () -> (Task(() -> nothing),)),
    Fixture("current_task", current_task, () -> ()),
    Fixture("Channel(sz)", Channel, () -> (2,)),
    Fixture("Channel{T}(sz)", Channel{Any}, () -> (2,)),
    Fixture("put!(::Channel)", put!, () -> (Channel(2), Task(() -> nothing))),
    Fixture("take!(::Channel)", take!, () -> ((c = Channel(2); put!(c, 1); c),)),
    Fixture("sync_end", Base.sync_end, () -> (Channel(Inf),)),
]

# Slots with a tangent space are the ones an activity mask ranges over; the rest can only ever carry
# `NoFData()`, which every rule already handled.
diff_slots(args) = [i for i in eachindex(args) if tangent_type(typeof(args[i])) !== NoTangent]

# The seed handed to a pullback: a unit seed where the output's rdata carries one, the type's own
# zero elsewhere (an array's rdata is `NoRData` — its gradient flows through fdata instead).
pb_seed(y::Float64) = 1.0
pb_seed(y::Tuple) = map(pb_seed, y)
pb_seed(y) = zero_rdata(y)

covered = Set{Method}()

@testset "rule activity: $(fx.name)" for fx in FIXTURES
    slots = diff_slots(fx.args())
    for bits in 0:(2^length(slots) - 1)
        inactive = Set(slots[i] for i in eachindex(slots) if (bits >> (i - 1)) & 1 == 1)
        args = fx.args()
        argcds = Any[i in inactive ? CoDual(args[i], Inactive()) : zero_fcodual(args[i])
                     for i in eachindex(args)]
        fcd = zero_fcodual(fx.f)

        # A hand rule must keep matching under every mask — never quietly fall through to the
        # derived transform, which cannot see through the `ccall` most of these bottom out in.
        m = rrule_method(fx.f, argcds)
        @test !is_derived_rrule(m)
        push!(covered, m)

        if fx.dest !== nothing && fx.dest in inactive
            # A mutating rule overwrites its destination, so the destination's shadow is both the
            # backward seed and the result's shadow; treating a constant one as a no-op would
            # silently drop the sources' gradients. Refused, not supported.
            @test_throws ErrorException rrule!!(fcd, Ctx(), argcds...)
            continue
        end

        if fx.frozen !== nothing && !(fx.frozen in inactive)
            # No closed form exists for this parameter's derivative. Constants arrive `Inactive`, so
            # a live shadow here is a genuine request for it — refused at the forwards call rather
            # than answered with a poisoned rdata.
            @test_throws ErrorException rrule!!(fcd, Ctx(), argcds...)
            continue
        end

        ycd, pb = rrule!!(fcd, Ctx(), argcds...)
        @test ycd isa CoDual
        # Never `Inactive()`: the transform declares a nested call's shadow slot at
        # `fdata_type(tangent_type(R))` before the callee is compiled.
        @test isactive(tangent(ycd))

        rds = pb(pb_seed(primal(ycd)))
        @test rds isa Tuple
        @test length(rds) == length(args) + 1
        # Slot 1 is the callee, which none of these rules carries tangent state for.
        @test rds[1] === NoRData()
        for i in eachindex(args)
            rd = rds[i + 1]
            if i in inactive
                @test rd === NoRData()
            else
                @test typeof(rd) === rdata_type(tangent_type(typeof(args[i])))
            end
        end
    end
end

@testset "rule activity: every hand rule has a fixture" begin
    # The completeness half: a newly added `rrule!!` with no fixture here fails this test rather
    # than silently going un-audited.
    #
    # Package-owned rules only. Under the full `runtests.jl` every file shares one process, and
    # several define test-local `rrule!!` methods of their own (the world-age and dispatch tests);
    # those are not part of the rule table this audits.
    sfmod = Base.get_extension(DifferCore, :DifferCoreSpecialFunctionsExt)
    owned(m) = m.module === DifferReverse || m.module === sfmod
    declared = Set(m for m in methods(rrule!!) if owned(m) && !is_derived_rrule(m))
    @test setdiff(declared, covered) == Set{Method}()
end
