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
