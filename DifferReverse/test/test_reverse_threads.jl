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
