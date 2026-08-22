# Reverse mode through `Threads.@threads`. Every case is checked twice: once by calling the
# `threading_run` rule directly, once through an enclosing function that writes `Threads.@threads`.
#
# The design these pin down: one `Tape` per worker (a `Tape`'s stacks are unguarded
# read-modify-writes, so a shared one is a data race), and the workers' pullbacks replayed
# sequentially in reverse worker order (the pullback read-modify-writes shadow slots and restores
# saved primal values, so a shared read in the primal is a shared write in reverse).
#
# The contract these do *not* check, because it can't be: the parallel region's result must not
# depend on how the workers interleave. Each worker's pullback replays that worker's operations in
# that worker's reverse order; the cross-worker interleaving is never recorded.
#
# Meaningless on one thread, so a single-threaded run fails loudly rather than passing vacuously.

using Test
using DifferReverse
using DifferReverse: CoDual, Ctx, Inactive, NoFData, NoRData, NoTangent, rrule!!, rev_gradient,
    primal, tangent, zero_fcodual, fdata, zero_tangent, build_tangent, tangent_type, rdata_type
include("testutils.jl")

@testset "thread pool is large enough for these tests to mean anything" begin
    @test Threads.threadpoolsize() > 1
end

const X8 = collect(1.0:8.0)

# --- The primals, each paired with a serial twin to cross-check against ------------------------

function threaded_map(x::Vector{Float64})
    y = similar(x)
    Threads.@threads for i in eachindex(y, x)
        y[i] = sin(x[i])
    end
    s = 0.0
    for i in eachindex(y)
        s += y[i]
    end
    return s
end

function threaded_map_static(x::Vector{Float64})
    y = similar(x)
    Threads.@threads :static for i in eachindex(y, x)
        y[i] = sin(x[i])
    end
    s = 0.0
    for i in eachindex(y)
        s += y[i]
    end
    return s
end

function serial_map(x::Vector{Float64})
    s = 0.0
    for i in eachindex(x)
        s += sin(x[i])
    end
    return s
end

# The case that distinguishes a working sequential replay from a broken one: `a` is read by every
# worker, so its gradient is the sum of per-worker contributions. Replaying the workers
# independently gets `dx` right and `da` wrong.
function threaded_scaled(a::Float64, x::Vector{Float64})
    y = similar(x)
    Threads.@threads for i in eachindex(y, x)
        y[i] = a * sin(x[i])
    end
    s = 0.0
    for i in eachindex(y)
        s += y[i]
    end
    return s
end

function serial_scaled(a::Float64, x::Vector{Float64})
    s = 0.0
    for i in eachindex(x)
        s += a * sin(x[i])
    end
    return s
end

function threaded_by_tid(x::Vector{Float64})
    y = zeros(Threads.maxthreadid())
    Threads.@threads for i in eachindex(x)
        y[Threads.threadid()] += sin(x[i])
    end
    return sum(y)
end

function threaded_locked(x::Vector{Float64})
    s = Ref(0.0)
    l = ReentrantLock()
    Threads.@threads for i in eachindex(x)
        v = sin(x[i])
        lock(l)
        s[] += v
        unlock(l)
    end
    return s[]
end

function threaded_locked_do(x::Vector{Float64})
    s = Ref(0.0)
    l = ReentrantLock()
    Threads.@threads for i in eachindex(x)
        v = sin(x[i])
        lock(l) do
            s[] += v
        end
    end
    return s[]
end

# The queries a worker body may make of the thread pool, all `ccall`/`cglobal`-backed. Each worker
# writes its own slot: an unsynchronized `s += ...` here would be a race in the primal, and no AD
# contract covers that.
function threaded_queries(x::Vector{Float64})
    y = similar(x)
    Threads.@threads for i in eachindex(y, x)
        n = Threads.nthreads() + Threads.threadpoolsize() + Threads.maxthreadid() +
            Threads.threadid() + Threads.nthreadpools()
        y[i] = sin(x[i]) * (n > 0)
    end
    s = 0.0
    for i in eachindex(y)
        s += y[i]
    end
    return s
end

# --- Gradients, through the `@threads` spelling -----------------------------------------------

@testset "single-argument loops" begin
    for f in (threaded_map, threaded_map_static, threaded_by_tid, threaded_locked,
              threaded_locked_do, threaded_queries)
        _, dx = rev_gradient(f, copy(X8))
        @test dx ≈ cos.(X8)
    end
end

@testset "cross-check against the serial loop" begin
    _, dx = rev_gradient(threaded_map, copy(X8))
    _, dx_serial = rev_gradient(serial_map, copy(X8))
    @test dx ≈ dx_serial
end

@testset "shared captured scalar accumulates across workers" begin
    a = 3.0
    _, da, dx = rev_gradient(threaded_scaled, a, copy(X8))
    _, da_serial, dx_serial = rev_gradient(serial_scaled, a, copy(X8))
    @test da ≈ sum(sin.(X8))
    @test dx ≈ a .* cos.(X8)
    @test da ≈ da_serial
    @test dx ≈ dx_serial
    # A per-worker-independent replay would land near `da / nworkers` on the chunk that ran last,
    # so pin the magnitude rather than just the shape.
    @test !(da ≈ sum(sin.(X8)) / Threads.threadpoolsize())
end

@testset "loop lengths around the thread count" begin
    # 0 and 1 exercise `len == 0`/`tid > rem`; a length not divisible by the pool size exercises
    # the `rem > 0` slicing branches.
    for n in (0, 1, 2, 5, 7, 13, 64)
        xs = collect(1.0:n)
        _, dx = rev_gradient(threaded_map, xs)
        @test dx ≈ cos.(xs)
    end
end

@testset "repeated gradients are deterministic" begin
    want = nothing
    for _ in 1:20
        _, dx = rev_gradient(threaded_map, copy(X8))
        want === nothing && (want = copy(dx))
        @test dx == want
    end
end

# --- The `threading_run` rule, called directly -------------------------------------------------
#
# `@threads`'s worker is a closure with a default positional argument, so Julia splits it into a
# wrapper holding a body function that holds the loop's captures. Reproduced here so the rule is
# exercised without the macro in the way.

function make_worker(x::Vector{Float64}, y::Vector{Float64})
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

# Same shape, but with a captured scalar, so the worker's own rdata is a real `RData` rather than
# `NoRData`. Used for the constant-worker case below.
function make_scaled_worker(a::Float64, x::Vector{Float64}, y::Vector{Float64})
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
            y[j] = a * sin(x[j])
        end
        return nothing
    end
    return threadsfor_fun
end

# The worker's `CoDual`, with `dx`/`dy` as the captured arrays' shadows.
function worker_codual(x, dx, y, dy)
    w = make_worker(x, y)
    W = typeof(w)
    B = fieldtype(W, 1)
    return CoDual(w, fdata(build_tangent(W, build_tangent(B, dx, dy, zero_tangent(eachindex(y, x))))))
end

@testset "threading_run rule, called directly" begin
    for static in (false, true)
        x, dx = copy(X8), zeros(8)
        y, dy = zeros(8), zeros(8)
        ycd, pb = rrule!!(zero_fcodual(Base.Threads.threading_run), Ctx(),
                          worker_codual(x, dx, y, dy), zero_fcodual(static))
        @test primal(ycd) === nothing
        @test y ≈ sin.(x)
        dy .= 1.0
        rds = pb(NoRData())
        @test length(rds) == 3          # threading_run, fun, static
        @test rds[1] === NoRData()
        @test rds[3] === NoRData()
        @test dx ≈ cos.(x)
    end
end

@testset "threading_run rule with a constant worker" begin
    # An `Inactive()` worker shadow must reach the derived rule untouched (strong zero), and the
    # rule must never hand an `Inactive` back.
    x, y = copy(X8), zeros(8)
    w = make_worker(x, y)
    ycd, pb = rrule!!(zero_fcodual(Base.Threads.threading_run), Ctx(),
                      CoDual(w, Inactive()), zero_fcodual(false))
    @test y ≈ sin.(x)
    @test !isa(tangent(ycd), Inactive)
    @test pb(NoRData())[1] === NoRData()
end

@testset "constant worker with a captured scalar hands back NoRData" begin
    # The worker's own rdata is a real `RData` here, but the caller declared it constant, so the
    # slot must still come back `NoRData()` — what the recursion glue declares for a constant slot,
    # and what each worker's own pullback returns. Seeding the closure's zero rdata instead is a
    # `MethodError` against that `NoRData()`, not a silently wrong number.
    x, y = copy(X8), zeros(8)
    w = make_scaled_worker(2.0, x, y)
    @test rdata_type(tangent_type(typeof(w))) !== NoRData
    _, pb = rrule!!(zero_fcodual(Base.Threads.threading_run), Ctx(),
                    CoDual(w, Inactive()), zero_fcodual(false))
    @test y ≈ 2.0 .* sin.(x)
    @test pb(NoRData())[2] === NoRData()
end

@testset "each worker gets its own tape" begin
    # A shared tape would interleave block-stack pushes across workers and misreplay; distinct
    # contexts are what the rule relies on. Pinned structurally: the pullback holds one entry per
    # worker, and they are distinct objects.
    x, dx = copy(X8), zeros(8)
    y, dy = zeros(8), zeros(8)
    _, pb = rrule!!(zero_fcodual(Base.Threads.threading_run), Ctx(),
                    worker_codual(x, dx, y, dy), zero_fcodual(false))
    @test length(pb.pbs) == Threads.threadpoolsize()
    @test length(unique(objectid.(pb.pbs))) == Threads.threadpoolsize()
end

@testset "Task and lock types are non-differentiable" begin
    @test tangent_type(Task) === NoTangent
    @test tangent_type(ReentrantLock) === NoTangent
    @test tangent_type(Base.Threads.SpinLock) === NoTangent
    @test tangent_type(Channel{Any}) === NoTangent
end

# --- `Threads.@spawn` / `@async` / `@sync` / bare `Task` ----------------------------------------
#
# A task's pullback replays at the *spawn site's* reverse position: spawn dominates every use, so
# its reverse turn runs after every fetch's and after the caller finished accumulating into any
# shared shadow the thunk's captures alias. What `fetch` moves *by value* has no rdata channel
# back to the spawn site (the `Task` carrier has no tangent), so a differentiable fetched value is
# refused loudly; a captured `Ref` written inside the task is the supported transport.

function spawn_ref_out(x)
    out = Ref(0.0)
    t = Threads.@spawn begin
        out[] = sum(abs2, x)
        nothing
    end
    wait(t)
    return out[]
end

function sync_two_spawns(x)
    y = zeros(2)
    Base.@sync begin
        Threads.@spawn (y[1] = 2.0 * x[1]; nothing)
        Threads.@spawn (y[2] = 3.0 * x[2]; nothing)
    end
    return y[1] + y[2]
end

# A scalar read by both tasks: its gradient is the sum of per-task contributions, routed through
# each spawn's own pullback — the case that distinguishes real cross-task accumulation.
function spawn_shared_scalar(a, x)
    o1, o2 = Ref(0.0), Ref(0.0)
    t1 = Threads.@spawn (o1[] = a * x[1]; nothing)
    t2 = Threads.@spawn (o2[] = a * x[2]; nothing)
    wait(t1); wait(t2)
    return o1[] + o2[]
end

@testset "spawn with a captured Ref" begin
    x = collect(1.0:3.0)
    _, dx = rev_gradient(spawn_ref_out, copy(x))
    @test dx ≈ 2.0 .* x
end

@testset "@sync with two spawns" begin
    x = collect(1.0:3.0)
    _, dx = rev_gradient(sync_two_spawns, copy(x))
    @test dx ≈ [2.0, 3.0, 0.0]
end

@testset "shared captured scalar accumulates across tasks" begin
    x = collect(1.0:3.0)
    _, da, dx = rev_gradient(spawn_shared_scalar, 3.0, copy(x))
    @test da ≈ x[1] + x[2]
    @test dx ≈ [3.0, 3.0, 0.0]
end

@testset "fetch of a differentiable value is refused" begin
    fetch_val(x) = fetch(Threads.@spawn sum(abs2, x))::Float64
    @test_throws Exception rev_gradient(fetch_val, collect(1.0:3.0))
end

@testset "put! of a differentiable value is refused" begin
    putfloat(x) = (c = Channel(2); put!(c, x); take!(c))
    @test_throws Exception rev_gradient(putfloat, 1.0)
end

@testset "Task rule handles a never-scheduled task" begin
    # The thunk never ran, so its pullback hands back the zero seed rather than waiting forever.
    x, dx = collect(1.0:3.0), zeros(3)
    w = () -> (x[1] * 2.0)
    wcd = CoDual(w, fdata(zero_tangent(w)))
    tcd, pb = rrule!!(zero_fcodual(Task), Ctx(), wcd)
    @test primal(tcd) isa Task
    rds = pb(NoRData())
    @test rds[1] === NoRData()
end

# --- Engine regressions the task work needed --------------------------------------------------

mutable struct ARef{T}
    @atomic x::T
    ARef{T}() where {T} = new{T}()
    ARef(x::T) where {T} = new{T}(x)
end

# Atomic field write + read: the primal ops keep an atomic ordering, the shadow ops stay plain.
function atomic_rw(x)
    r = ARef(0.0)
    @atomic r.x = 2.0 * x
    return (@atomic r.x) * 3.0
end

# Partially-initialised `%new` of a tangent-carrying mutable struct, then an atomic write into the
# still-uninitialised slot — the StableTasks `AtomicRef{T}()` shape.
function undef_new_rw(x)
    r = ARef{Float64}()
    @atomic r.x = x
    return (@atomic r.x) * 2.0
end

@testset "atomic mutable-struct fields" begin
    _, dx = rev_gradient(atomic_rw, 1.5)
    @test dx ≈ 6.0
    _, dx = rev_gradient(undef_new_rw, 1.5)
    @test dx ≈ 2.0
end

# --- Concurrent first-call compilation ---------------------------------------------------------

@testset "distinct derived rules compile concurrently" begin
    # `ID()`'s counter and the bail-reason table are reached during a build, and a build can start
    # inside a worker task. Distinct primals so each really does compile rather than hitting a
    # cached `CodeInstance`.
    fs = (x -> sum(sin, x), x -> sum(cos, x), x -> sum(exp, x), x -> sum(abs2, x),
          x -> sum(tanh, x), x -> sum(atan, x), x -> sum(log, x), x -> sum(log1p, x))
    xs = [abs.(copy(X8)) .+ 0.5 for _ in fs]
    out = Vector{Any}(undef, length(fs))
    Threads.@threads for k in eachindex(fs)
        out[k] = rev_gradient(fs[k], xs[k])[2]
    end
    for (k, f) in pairs(fs)
        @test out[k] ≈ rev_gradient(f, xs[k])[2]
    end
end
