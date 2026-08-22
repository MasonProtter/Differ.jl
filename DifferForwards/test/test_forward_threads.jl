# Forward mode through `Threads.@threads`. Every case is checked twice: once by calling the
# `threading_run` rule directly, once through an enclosing function that writes `Threads.@threads`.
#
# These tests are meaningless on one thread — the whole point is that one `Dual` closure is invoked
# concurrently from several workers — so a single-threaded run fails loudly rather than passing
# vacuously.

using Test
using DifferForwards
using DifferForwards: Dual, Inactive, NoTangent, frule!!, primal, tangent, zero_dual, zero_tangent,
    build_tangent
include("testutils.jl")

@testset "thread pool is large enough for these tests to mean anything" begin
    @test Threads.threadpoolsize() > 1
end

const X8 = collect(1.0:8.0)

# --- The primals -----------------------------------------------------------------------------

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

# A scalar read by every worker: its tangent must reach every chunk, not just one.
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

# `threadid()` reaches a `ccall(:jl_threadid)`; the per-thread accumulator is the shape it's for.
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

# --- Dualized IR is legal --------------------------------------------------------------------

@testset "dualized IR verifies" begin
    for f in (threaded_map, threaded_map_static, threaded_by_tid, threaded_locked,
              threaded_locked_do, threaded_queries)
        @test (checkverify(f, (Vector{Float64},)); true)
    end
    @test (checkverify(threaded_scaled, (Float64, Vector{Float64})); true)
end

# --- Derivatives, through the `@threads` spelling ---------------------------------------------

@testset "single-argument loops" begin
    for f in (threaded_map, threaded_map_static, threaded_by_tid, threaded_locked,
              threaded_locked_do)
        d = frule!!(zero_dual(f), Dual(copy(X8), ones(8)))
        @test primal(d) ≈ sum(sin.(X8))
        @test tangent(d) ≈ sum(cos.(X8))
    end
end

@testset "shared captured scalar" begin
    a = 3.0
    # seed only `a`
    d = frule!!(zero_dual(threaded_scaled), Dual(a, 1.0), Dual(copy(X8), zeros(8)))
    @test primal(d) ≈ a * sum(sin.(X8))
    @test tangent(d) ≈ sum(sin.(X8))
    # seed only `x`
    d = frule!!(zero_dual(threaded_scaled), Dual(a, 0.0), Dual(copy(X8), ones(8)))
    @test tangent(d) ≈ a * sum(cos.(X8))
end

@testset "thread-pool queries carry no derivative" begin
    d = frule!!(zero_dual(threaded_queries), Dual(copy(X8), ones(8)))
    @test tangent(d) ≈ sum(cos.(X8))
end

# --- Derivatives, calling the `threading_run` rule directly -----------------------------------
#
# `@threads`'s worker is a closure with a default positional argument, so Julia splits it into a
# wrapper holding a body function that holds the loop's captures. Reproduced here so the rule is
# exercised without the macro in the way.

function make_worker(x::Vector{Float64}, y::Vector{Float64})
    range = eachindex(y, x)
    function threadsfor_fun(tid=1; onethread=false)
        r = range
        lenr = length(r)
        if onethread
            tid = 1
            len, rem = lenr, 0
        else
            len, rem = divrem(lenr, Threads.threadpoolsize())
        end
        if len == 0
            tid > rem && return nothing
            len, rem = 1, 0
        end
        f = firstindex(r) + ((tid - 1) * len)
        l = f + len - 1
        if rem > 0
            if tid <= rem
                f = f + (tid - 1)
                l = l + tid
            else
                f = f + rem
                l = l + rem
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

# The worker's tangent, with `dx`/`dy` as the captured arrays' shadows.
function worker_dual(x, dx, y, dy)
    w = make_worker(x, y)
    W = typeof(w)
    B = fieldtype(W, 1)
    return Dual(w, build_tangent(W, build_tangent(B, dx, dy, zero_tangent(eachindex(y, x)))))
end

@testset "threading_run rule, called directly" begin
    x, dx = copy(X8), ones(8)
    y, dy = zeros(8), zeros(8)
    d = frule!!(zero_dual(Base.Threads.threading_run), worker_dual(x, dx, y, dy),
                zero_dual(false))
    @test primal(d) === nothing
    @test tangent(d) === NoTangent()
    @test y ≈ sin.(x)
    @test dy ≈ cos.(x)
end

@testset "threading_run rule with a constant worker" begin
    # An `Inactive()` worker tangent must reach the derived rule untouched (strong zero), and the
    # rule must never hand an `Inactive` back.
    x, y = copy(X8), zeros(8)
    w = make_worker(x, y)
    d = frule!!(zero_dual(Base.Threads.threading_run), Dual(w, Inactive()), zero_dual(false))
    @test y ≈ sin.(x)
    @test tangent(d) === NoTangent()
    @test !isa(tangent(d), Inactive)
end

@testset ":static schedule dispatches through jl_in_threaded_region" begin
    x, dx = copy(X8), ones(8)
    y, dy = zeros(8), zeros(8)
    d = frule!!(zero_dual(Base.Threads.threading_run), worker_dual(x, dx, y, dy),
                zero_dual(true))
    @test dy ≈ cos.(x)
end

# --- Loop-shape edge cases -------------------------------------------------------------------

@testset "loop lengths around the thread count" begin
    # 0 and 1 exercise `len == 0`/`tid > rem`; a length not divisible by the pool size exercises
    # the `rem > 0` slicing branches.
    for n in (0, 1, 2, 5, 7, 13, 64)
        xs = collect(1.0:n)
        d = frule!!(zero_dual(threaded_map), Dual(xs, ones(n)))
        @test primal(d) ≈ sum(sin.(xs))
        @test tangent(d) ≈ sum(cos.(xs))
    end
end

@testset "repeated calls are deterministic" begin
    want = nothing
    for _ in 1:20
        d = frule!!(zero_dual(threaded_map), Dual(copy(X8), ones(8)))
        want === nothing && (want = tangent(d))
        @test tangent(d) === want
    end
end

@testset "Task and lock types are non-differentiable" begin
    @test tangent_type(Task) === NoTangent
    @test tangent_type(ReentrantLock) === NoTangent
    @test tangent_type(Base.Threads.SpinLock) === NoTangent
    @test tangent_type(Channel{Any}) === NoTangent
end

# --- `Threads.@spawn` / `@async` / `@sync` / bare `Task` + `fetch` ------------------------------
#
# The spawned thunk runs dualized; its `Dual` result is parked in the task's own storage and
# `fetch`'s rule hands it back, so a value fetched by value carries its tangent. Effects through
# captures (a `Ref`/array written inside the task) flow through the shared shadow as in any other
# mutation. The interleaving contract is `@threads`'s: the region's result may not depend on how
# the task interleaves with the code between spawn and join.

function spawn_fetch_value(x)
    t = Threads.@spawn sum(abs2, x)
    return 2.0 * (fetch(t)::Float64)
end

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

function async_fetch(x)
    t = @async sum(abs2, x)
    return fetch(t)::Float64
end

@testset "spawn/fetch value transport" begin
    x = collect(1.0:3.0)
    d = frule!!(zero_dual(spawn_fetch_value), Dual(copy(x), [1.0, 0.0, 0.0]))
    @test primal(d) ≈ 2.0 * sum(abs2, x)
    @test tangent(d) ≈ 2.0 * 2.0 * x[1]
end

@testset "spawn with a captured Ref" begin
    x = collect(1.0:3.0)
    d = frule!!(zero_dual(spawn_ref_out), Dual(copy(x), [1.0, 0.0, 0.0]))
    @test primal(d) ≈ sum(abs2, x)
    @test tangent(d) ≈ 2.0 * x[1]
end

@testset "@sync with two spawns" begin
    x = collect(1.0:3.0)
    d = frule!!(zero_dual(sync_two_spawns), Dual(copy(x), ones(3)))
    @test primal(d) ≈ 2.0 * x[1] + 3.0 * x[2]
    @test tangent(d) ≈ 5.0
end

@testset "@async" begin
    x = collect(1.0:3.0)
    d = frule!!(zero_dual(async_fetch), Dual(copy(x), ones(3)))
    @test tangent(d) ≈ sum(2.0 .* x)
end

@testset "fetch of a foreign task with a differentiable result is refused" begin
    # A task created outside the differentiated region has no recorded `Dual`; handing back its
    # `Float64` as a constant would silently drop the derivative.
    t = Task(() -> 1.0)
    schedule(t); wait(t)
    fetch_foreign(tk) = 2.0 * (fetch(tk)::Float64)
    @test_throws ErrorException frule!!(zero_dual(fetch_foreign), Dual(t, NoTangent()))
    # A non-differentiable result from the same shape is fine.
    t2 = Task(() -> 41); schedule(t2); wait(t2)
    fetch_int(tk) = fetch(tk)::Int + 1
    d = frule!!(zero_dual(fetch_int), Dual(t2, NoTangent()))
    @test primal(d) === 42
end

@testset "put! of a differentiable value is refused" begin
    putfloat(x) = (c = Channel(2); put!(c, x); take!(c))
    @test_throws ErrorException frule!!(zero_dual(putfloat), Dual(1.0, 1.0))
end
