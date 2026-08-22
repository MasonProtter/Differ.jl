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
