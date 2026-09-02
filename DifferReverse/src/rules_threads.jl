# Hand-written `rrule!!`s for Julia's threading constructs. Mirrors
# `DifferForwards/src/rules_threads.jl`; see that file for why `Base.Threads.threading_run` is the
# right interception point and why the thread/lock queries need rules of their own.
#
# Reverse mode's extra requirement is the tape. `Stack`'s `push!`/`pop!` (`stack.jl`) are unguarded
# read-modify-writes on `position`, so a tape shared across workers is a data race — and even a
# locked one would interleave the block stack and destroy the replay order. Each worker therefore
# gets its own `Ctx`. `Ctx()` rather than `build_ctx`: a hand rule must ignore its own `ctx` slot, so
# there is nowhere to keep pre-allocated tapes between calls.
#
# The pullbacks are replayed **sequentially**, in reverse worker order. Not an optimization gap —
# the pullback read-modify-writes shadow slots non-atomically and restores saved primal values back
# into the primal array, so a shared *read* in the primal is a shared *write* in reverse. A parallel
# replay would race even on a perfectly race-free primal.
#
# Sequential replay is order-insensitive, which is what makes it sound: rdata comes back by value
# and folds with `increment!!`, and fdata lands in slots that are disjoint across workers under the
# contract below.
#
# The contract: the parallel region's result may not depend on how the workers interleave. Each
# worker's pullback replays *that worker's* operations in *that worker's* reverse order; the
# cross-worker interleaving is neither recorded nor reconstructed. Disjoint writes and
# commutative-associative shared updates satisfy it — a lock-protected non-commutative update
# (`s[] = s[] * x[i]`) does not, and gets a wrong gradient from a primal that runs fine.

struct ThreadingRunPullback{V,R}
    pbs::V
    dfun::R      # zero rdata for the worker closure — the accumulator's seed and its declared type
end

# The worker pullback's concrete type, so the replay loop below is statically dispatched and its
# accumulated rdata has the type the recursion glue declares for this call's `fun` slot. A widened
# answer costs a dynamic dispatch per worker, never correctness.
@inline function _worker_pullback_type(::Type{FC}) where {FC<:CoDual}
    RT = Base.promote_op(rrule!!, FC, Ctx{Nothing}, CoDual{Int,NoFData})
    return (RT isa DataType && RT <: Tuple && length(RT.parameters) == 2) ? RT.parameters[2] : Any
end

# The accumulator's seed, which also fixes what the pullback hands back for the `fun` slot. A worker
# the caller declared constant contributes nothing and gets `NoRData()` — what every rule returns for
# a constant slot, and what the recursion glue declares for one. Seeding its zero rdata instead would
# be a `MethodError` against the `NoRData()` each worker's pullback then returns, and the wrong shape
# even if it weren't.
_worker_seed_rdata(fun::CoDual) = zero_rdata(primal(fun))
_worker_seed_rdata(::CoDual{<:Any,Inactive}) = NoRData()

# One rdata per `rrule!!` slot: `threading_run` itself, the worker closure, `static`. Only the
# closure carries anything — a captured scalar read by every worker accumulates here, which is the
# case that distinguishes this from replaying the workers independently.
function (pb::ThreadingRunPullback)(::NoRData)
    dfun = pb.dfun
    for i in length(pb.pbs):-1:1
        dfun = increment!!(dfun, pb.pbs[i](NoRData())[1])
    end
    return NoRData(), dfun, NoRData()
end

function rrule!!(::CoDual{typeof(Base.Threads.threading_run)}, ::AbstractCtx,
                 fun::CoDual, static::CoDual{Bool})
    n = Threads.threadpoolsize()
    ctxs = [Ctx() for _ in 1:n]
    pbs = Vector{_worker_pullback_type(typeof(fun))}(undef, n)
    Base.Threads.threading_run(primal(static)) do tid
        1 <= tid <= n || throw(ArgumentError("unexpected thread id $tid"))
        ycd, pb = rrule!!(fun, ctxs[tid], zero_fcodual(tid))
        # `@threads` workers return `nothing` on every schedule, which is what makes `NoRData()` the
        # right seed in the pullback above. Anything else means this rule is being used for a shape
        # it can't seed, so say so here rather than silently seeding a zero.
        rdata_type(tangent_type(typeof(primal(ycd)))) === NoRData ||
            throw(ArgumentError("`threading_run`'s worker returned a `$(typeof(primal(ycd)))`, " *
                                "whose rdata is not `NoRData` — Differ can only differentiate a " *
                                "parallel region whose workers return a non-differentiable value"))
        pbs[tid] = pb
        nothing
    end
    return zero_fcodual(nothing), ThreadingRunPullback(pbs, _worker_seed_rdata(fun))
end

# Thread/pool queries: primal value, no contribution to anything.
for (f, argtys) in ((:(Threads.threadid), (Tuple{}, Tuple{Task})),
                    (:(Threads.nthreads), (Tuple{}, Tuple{Symbol})),
                    (:(Threads.threadpoolsize), (Tuple{}, Tuple{Symbol})),
                    (:(Threads.threadpool), (Tuple{}, Tuple{Int})),
                    (:(Threads.maxthreadid), (Tuple{},)),
                    (:(Threads.nthreadpools), (Tuple{},)))
    for tt in argtys
        args = [Symbol(:a, i) for i in 1:length(tt.parameters)]
        sig = [:($a::CoDual{$T}) for (a, T) in zip(args, tt.parameters)]
        @eval function rrule!!(fcd::CoDual{typeof($f)}, ::AbstractCtx, $(sig...))
            y = $f($((:(primal($a)) for a in args)...))
            return zero_fcodual(y), NoPullback(fcd, $(args...))
        end
    end
end

const _Lock = Union{ReentrantLock,Base.Threads.SpinLock}

for f in (:lock, :unlock, :trylock, :islocked)
    @eval function rrule!!(fcd::CoDual{typeof(Base.$f)}, ::AbstractCtx, ld::CoDual{<:_Lock})
        y = Base.$f(primal(ld))
        return zero_fcodual(y), NoPullback(fcd, ld)
    end
end

# `lock(f, l)` do-block form. The lock is held for the primal call only: the pullback runs later,
# from whichever task owns the enclosing tape, and by then the region is over.
struct LockPullback{P}
    pb::P
end
(pb::LockPullback)(dy) = (NoRData(), only(pb.pb(dy)), NoRData())

function rrule!!(::CoDual{typeof(Base.lock)}, ctx::AbstractCtx, fcd::CoDual, ld::CoDual{<:_Lock})
    l = primal(ld)
    Base.lock(l)
    try
        ycd, pb = rrule!!(fcd, ctx)
        return ycd, LockPullback(pb)
    finally
        Base.unlock(l)
    end
end

# ---------------------------------------------------------------------------
# Tasks: `Threads.@spawn` / `@async` / bare `Task` + `schedule`/`wait`/`fetch`, and the
# `Channel`/`put!`/`sync_end` trio `@sync` expands to.
#
# `@spawn` expands *inline* in the enclosing function — `Task(thunk)`, `task.sticky = false`,
# `_spawn_set_thrpool(task, tp)`, an optional `put!` into `@sync`'s channel, `schedule(task)` — so
# ruling `Task` itself is the interception point (it also keeps `jl_new_task` out of carrier IR);
# `task.sticky = false` is fielded by `setfield!`'s `NoTangent`-object arm in the engine.
#
# Gradient routing: the task's pullback replays at the *spawn site's* reverse position, not at
# `fetch`'s. Spawn dominates every use of the task, so its reverse turn runs after every fetch's —
# and after the caller has finished accumulating into any shared shadow the thunk's captures
# alias. What `fetch` moves *by value* has no rdata channel back to the spawn site (the `Task`
# carrier has no tangent), so a differentiable fetched value is refused loudly; a captured `Ref`
# written inside the task — the StableTasks shape — is the supported transport, flowing through
# shared fdata like any other mutation.
#
# The interleaving contract is `threading_run`'s: the region's result may not depend on how the
# task interleaves with the code between spawn and join.
# ---------------------------------------------------------------------------

const _TASK_RESULT_KEY = :__differ_task_result__

struct _RRTaskThunk{F<:CoDual}
    thunk::F
end
function (w::_RRTaskThunk)()
    ycd, pb = rrule!!(w.thunk, Ctx())
    task_local_storage(_TASK_RESULT_KEY, (ycd, pb))
    return primal(ycd)
end

_task_record(t::Task) =
    (s = t.storage; s !== nothing && haskey(s, _TASK_RESULT_KEY) ? s[_TASK_RESULT_KEY] : nothing)

struct TaskPullback{R}
    t::Task
    dthunk_seed::R   # zero rdata for the thunk — the accumulator's seed and its declared type
end

function (pb::TaskPullback{R})(::NoRData) where {R}
    t = pb.t
    # Never scheduled: the thunk never ran, so it contributed nothing.
    istaskstarted(t) || return NoRData(), pb.dthunk_seed
    Base.wait(t)   # a task the primal never joined; a failed one rethrows here, loudly
    rec = _task_record(t)
    rec === nothing && error("Differ-spawned task finished without recording its pullback")
    ycd, inner_pb = rec
    rdata_type(tangent_type(typeof(primal(ycd)))) === NoRData ||
        throw(ArgumentError("a task returning a `$(typeof(primal(ycd)))`, whose rdata is not " *
                            "`NoRData`, cannot be replayed — route the value through a captured " *
                            "`Ref` instead of the task's return slot"))
    dthunk = increment!!(pb.dthunk_seed, first(inner_pb(NoRData())))::R
    return NoRData(), dthunk
end

function rrule!!(::CoDual{Type{Task}}, ::AbstractCtx, thunk::CoDual)
    t = Task(_RRTaskThunk(thunk))
    return zero_fcodual(t), TaskPullback(t, _worker_seed_rdata(thunk))
end

function rrule!!(fcd::CoDual{typeof(Base.fetch)}, ::AbstractCtx, tcd::CoDual{Task})
    t = primal(tcd)
    y = Base.fetch(t)
    rec = _task_record(t)
    rec !== nothing && (y = primal(rec[1]))
    # By-value transport out of a task has no rdata channel back to the spawn site; a result with
    # no tangent space is the only thing `fetch` can hand back. This guard is what makes the
    # `NoPullback` sound: a `NoTangent`-typed result has no rdata for it to swallow.
    tangent_type(typeof(y)) === NoTangent ||
        throw(ArgumentError("reverse mode cannot transport a differentiable `$(typeof(y))` out " *
                            "of `fetch(::Task)` — write it to a captured `Ref` inside the task " *
                            "(the StableTasks.jl shape) instead"))
    # `CoDual{Any,Any}`: the call's inferred result is `Any`, so its fdata slot is declared `Any`
    # too; the recursion emission requires the resolved and inferred fdata types to agree.
    return CoDual{Any,Any}(y, NoFData()), NoPullback(fcd, tcd)
end

function rrule!!(fcd::CoDual{typeof(Base.schedule)}, ::AbstractCtx, tcd::CoDual{Task})
    return zero_fcodual(Base.schedule(primal(tcd))), NoPullback(fcd, tcd)
end

function rrule!!(fcd::CoDual{typeof(Base.wait)}, ::AbstractCtx, tcd::CoDual{Task})
    Base.wait(primal(tcd))
    return zero_fcodual(nothing), NoPullback(fcd, tcd)
end

for f in (:istaskdone, :istaskstarted, :istaskfailed)
    @eval function rrule!!(fcd::CoDual{typeof(Base.$f)}, ::AbstractCtx, tcd::CoDual{Task})
        return zero_fcodual(Base.$f(primal(tcd))), NoPullback(fcd, tcd)
    end
end

function rrule!!(fcd::CoDual{typeof(Base.Threads._spawn_set_thrpool)}, ::AbstractCtx,
                 tcd::CoDual{Task}, tp::CoDual{Symbol})
    Base.Threads._spawn_set_thrpool(primal(tcd), primal(tp))
    return zero_fcodual(nothing), NoPullback(fcd, tcd, tp)
end

rrule!!(fcd::CoDual{typeof(Base.yield)}, ::AbstractCtx) =
    (Base.yield(); (zero_fcodual(nothing), NoPullback(fcd)))
rrule!!(fcd::CoDual{typeof(Base.yield)}, ::AbstractCtx, tcd::CoDual{Task}) =
    (Base.yield(primal(tcd)); (zero_fcodual(nothing), NoPullback(fcd, tcd)))
rrule!!(fcd::CoDual{typeof(Base.current_task)}, ::AbstractCtx) =
    (zero_fcodual(Base.current_task()), NoPullback(fcd))

# `@sync`'s trio. A `Channel` has no tangent space, so a value moved through one is a constant;
# `put!`/`take!` refuse a differentiable value loudly rather than zeroing it. The ruled `Channel`
# constructor also keeps its internals (a `Vector{Any}` allocation) out of carrier IR.
rrule!!(fcd::CoDual{Type{Channel}}, ::AbstractCtx, sz::CoDual) =
    (zero_fcodual(Channel(primal(sz))), NoPullback(fcd, sz))
rrule!!(fcd::CoDual{Type{Channel{T}}}, ::AbstractCtx, sz::CoDual) where {T} =
    (zero_fcodual(Channel{T}(primal(sz))), NoPullback(fcd, sz))

function rrule!!(fcd::CoDual{typeof(Base.put!)}, ::AbstractCtx, cd::CoDual{<:Channel}, vd::CoDual)
    v = primal(vd)
    tangent_type(typeof(v)) === NoTangent ||
        error("`put!` of a differentiable value (`$(typeof(v))`) into a `Channel` is not supported")
    Base.put!(primal(cd), v)
    return zero_fcodual(v), NoPullback(fcd, cd, vd)
end

function rrule!!(fcd::CoDual{typeof(Base.take!)}, ::AbstractCtx, cd::CoDual{<:Channel})
    y = Base.take!(primal(cd))
    tangent_type(typeof(y)) === NoTangent ||
        error("`take!` of a differentiable value (`$(typeof(y))`) from a `Channel` is not supported")
    return zero_fcodual(y), NoPullback(fcd, cd)
end

function rrule!!(fcd::CoDual{typeof(Base.sync_end)}, ::AbstractCtx, cd::CoDual{<:Channel})
    Base.sync_end(primal(cd))
    return zero_fcodual(nothing), NoPullback(fcd, cd)
end
