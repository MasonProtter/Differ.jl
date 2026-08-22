# Hand-written `frule!!`s for Julia's threading constructs.
#
# `Threads.@threads` expands to a single non-inlined `invoke` of `Base.Threads.threading_run`, with
# the whole loop body inside a closure argument. Ruling that one call runs the scheduler as ordinary
# primal code and dualizes only the worker body, which is the only tractable split: `threading_run`
# itself is `jl_enter_threaded_region`, `jl_new_task`, `jl_set_task_tid`, `jl_wakeup_thread`, and a
# `try`/`finally` around them.
#
# Nothing here needs a per-thread copy of anything. Dualized IR is an ordinary `CodeInstance` with no
# per-call state, so one `Dual` closure is safely invoked from every worker at once.
#
# The remaining rules are thread/lock queries. They carry no derivative, but they must be ruled
# rather than dualized: each bottoms out in a `ccall` or a `cglobal` the engine has no rule for, and
# `Threads.threadpoolsize` in particular is called from *inside* every `@threads` worker body (the
# `divrem(lenr, threadpoolsize())` that computes a task's slice). A hand rule also keeps
# `src_inlining_policy` from inlining the callee away before the transform sees it, which is what
# keeps those `ccall`s out of dualized IR entirely.
#
# What the caller must guarantee: the parallel region's result may not depend on how the tasks
# interleave. Writes disjoint across tasks, or shared updates commutative and associative. That is
# already what makes the primal loop deterministic, and forward mode adds nothing to it — a tangent
# is written exactly where its primal is.

function frule!!(::Dual{typeof(Base.Threads.threading_run)}, fun::Dual, static::Dual{Bool})
    n = Threads.threadpoolsize()
    Base.Threads.threading_run(primal(static)) do tid
        1 <= tid <= n || throw(ArgumentError("unexpected thread id $tid"))
        frule!!(fun, Dual(tid, NoTangent()))
        nothing
    end
    return Dual(nothing, NoTangent())
end

# Thread/pool queries: reconstruct the primal from the argument primals, zero tangent on the result.
for (f, argtys) in ((:(Threads.threadid), (Tuple{}, Tuple{Task})),
                    (:(Threads.nthreads), (Tuple{}, Tuple{Symbol})),
                    (:(Threads.threadpoolsize), (Tuple{}, Tuple{Symbol})),
                    (:(Threads.threadpool), (Tuple{}, Tuple{Int})),
                    (:(Threads.maxthreadid), (Tuple{},)),
                    (:(Threads.nthreadpools), (Tuple{},)))
    for tt in argtys
        args = [Symbol(:a, i) for i in 1:length(tt.parameters)]
        sig = [:($a::Dual{$T}) for (a, T) in zip(args, tt.parameters)]
        @eval function frule!!(::Dual{typeof($f)}, $(sig...))
            y = $f($((:(primal($a)) for a in args)...))
            return Dual(y, NoTangent())
        end
    end
end

const _Lock = Union{ReentrantLock,Base.Threads.SpinLock}

# Locks move no values, so they are pure pass-throughs. Note this only makes a lock-protected body
# *differentiable*, not a lock-protected reduction *correct*: a shared update that isn't commutative
# and associative violates the interleaving contract above, whatever the lock does.
for f in (:lock, :unlock, :trylock, :islocked)
    @eval function frule!!(::Dual{typeof(Base.$f)}, ld::Dual{<:_Lock})
        y = Base.$f(primal(ld))
        return Dual(y, NoTangent())
    end
end

# `lock(f, l)` do-block form: the body is differentiated, the locking is not.
function frule!!(::Dual{typeof(Base.lock)}, fd::Dual, ld::Dual{<:_Lock})
    l = primal(ld)
    Base.lock(l)
    try
        return frule!!(fd)
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
# there is no single function to rule the way `threading_run` was. Ruling `Task` itself is what
# keeps `jl_new_task` out of dualized IR; the other steps get the rules below.
#
# The spawned thunk runs dualized. Its `Dual` result is parked in the task's own task-local
# storage, and the task's *primal* result is the primal half — so a task that escapes the
# differentiated region still fetches an ordinary value.
# ---------------------------------------------------------------------------

const _TASK_RESULT_KEY = :__differ_task_result__

struct _DualTaskThunk{F<:Dual}
    thunk::F
end
function (w::_DualTaskThunk)()
    d = frule!!(w.thunk)
    task_local_storage(_TASK_RESULT_KEY, d)
    return primal(d)
end

frule!!(::Dual{Type{Task}}, thunk::Dual) = Dual(Task(_DualTaskThunk(thunk)), NoTangent())

function frule!!(::Dual{typeof(Base.schedule)}, td::Dual{Task})
    return Dual(Base.schedule(primal(td)), NoTangent())
end

function frule!!(::Dual{typeof(Base.wait)}, td::Dual{Task})
    Base.wait(primal(td))
    return Dual(nothing, NoTangent())
end

function frule!!(::Dual{typeof(Base.fetch)}, td::Dual{Task})
    t = primal(td)
    y = Base.fetch(t)
    store = t.storage
    if store !== nothing && haskey(store, _TASK_RESULT_KEY)
        return store[_TASK_RESULT_KEY]::Dual
    end
    # A task created outside the differentiated region carries no recorded `Dual`; only a result
    # with no tangent space is safe to hand back as a constant.
    tangent_type(typeof(y)) === NoTangent && return Dual(y, NoTangent())
    error("`fetch` of a `Task` not spawned inside the differentiated region would drop the " *
          "derivative of its `$(typeof(y))` result")
end

for f in (:istaskdone, :istaskstarted, :istaskfailed)
    @eval frule!!(::Dual{typeof(Base.$f)}, td::Dual{Task}) = Dual(Base.$f(primal(td)), NoTangent())
end

function frule!!(::Dual{typeof(Base.Threads._spawn_set_thrpool)}, td::Dual{Task}, tp::Dual{Symbol})
    Base.Threads._spawn_set_thrpool(primal(td), primal(tp))
    return Dual(nothing, NoTangent())
end

frule!!(::Dual{typeof(Base.yield)}) = (Base.yield(); Dual(nothing, NoTangent()))
frule!!(::Dual{typeof(Base.yield)}, td::Dual{Task}) = (Base.yield(primal(td)); Dual(nothing, NoTangent()))
frule!!(::Dual{typeof(Base.current_task)}) = Dual(Base.current_task(), NoTangent())

# `@sync` expands to `Channel(Inf)` + `put!(chan, task)` per `@spawn`/`@async` + `sync_end(chan)`.
# A `Channel` has no tangent space (`DifferCore/src/tangents.jl`), so a value moved through one is
# a constant; `put!`/`take!` refuse a differentiable value loudly rather than zeroing it.
frule!!(::Dual{Type{Channel}}, sz::Dual) = Dual(Channel(primal(sz)), NoTangent())
frule!!(::Dual{Type{Channel{T}}}, sz::Dual) where {T} = Dual(Channel{T}(primal(sz)), NoTangent())

function frule!!(::Dual{typeof(Base.put!)}, cd::Dual{<:Channel}, vd::Dual)
    v = primal(vd)
    tangent_type(typeof(v)) === NoTangent ||
        error("`put!` of a differentiable value (`$(typeof(v))`) into a `Channel` is not supported")
    Base.put!(primal(cd), v)
    return Dual(v, NoTangent())
end

function frule!!(::Dual{typeof(Base.take!)}, cd::Dual{<:Channel})
    y = Base.take!(primal(cd))
    tangent_type(typeof(y)) === NoTangent ||
        error("`take!` of a differentiable value (`$(typeof(y))`) from a `Channel` is not supported")
    return Dual(y, NoTangent())
end

function frule!!(::Dual{typeof(Base.sync_end)}, cd::Dual{<:Channel})
    Base.sync_end(primal(cd))
    return Dual(nothing, NoTangent())
end
