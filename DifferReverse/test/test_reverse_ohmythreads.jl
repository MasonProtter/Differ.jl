# Reverse mode through OhMyThreads.jl, exercised as an ordinary downstream package — no extension,
# no OhMyThreads-specific rules. StableTasks' value transport is exactly the shape reverse mode
# supports: the spawned thunk writes its result into a captured `AtomicRef{T}`, the caller reads it
# back after `fetch(t.t)` synchronizes, and each spawn site's pullback replays its task's tape.
#
# The interleaving contract is `@threads`'s. What `fetch(::Task)` moves *by value* stays refused
# (no rdata channel back to the spawn site) — StableTasks never relies on it.

using Test
using DifferReverse
using DifferReverse: rev_gradient
using OhMyThreads, StableTasks
include("testutils.jl")

@testset "thread pool is large enough for these tests to mean anything" begin
    @test Threads.threadpoolsize() > 1
end

const X8 = collect(1.0:8.0)

@testset "StableTasks: @spawn + fetch" begin
    f = x -> fetch(StableTasks.@spawn sum(abs2, x))
    _, dx = rev_gradient(f, copy(X8))
    @test dx ≈ 2.0 .* X8
end

@testset "tmapreduce" begin
    for f in (x -> tmapreduce(i -> sin(x[i]), +, eachindex(x)),                      # default (chunked)
              x -> tmapreduce(i -> sin(x[i]), +, 1:length(x); nchunks=2),
              x -> tmapreduce(sin, +, x; scheduler=DynamicScheduler(; chunking=false)))
        _, dx = rev_gradient(f, copy(X8))
        @test dx ≈ cos.(X8)
    end
end

@testset "tforeach" begin
    function tf(x)
        y = similar(x)
        tforeach(eachindex(x)) do i
            y[i] = sin(x[i])
        end
        return sum(y)
    end
    _, dx = rev_gradient(tf, copy(X8))
    @test dx ≈ cos.(X8)
end

@testset "tforeach with heterogeneous captures" begin
    function tfh(a, x)
        y = similar(x)
        tforeach(eachindex(x)) do i
            y[i] = a * sin(x[i])
        end
        return sum(y)
    end
    _, da, dx = rev_gradient(tfh, 3.0, copy(X8))
    @test da ≈ sum(sin.(X8))
    @test dx ≈ 3.0 .* cos.(X8)
end

@testset "repeated gradients are deterministic" begin
    f = x -> tmapreduce(i -> sin(x[i]), +, eachindex(x))
    want = nothing
    for _ in 1:10
        _, dx = rev_gradient(f, copy(X8))
        want === nothing && (want = copy(dx))
        @test dx == want
    end
end

@testset "known gaps stay loud" begin
    # The array-form chunked path re-splats OhMyThreads' `ChunkingArgs` into a `Union`-field-typed
    # kwargs `NamedTuple` and hits a genuinely dynamic `Core.kwcall` — reverse mode's
    # dynamic-dispatch gap, not a task-machinery one. `tmap` additionally collects its output
    # through `BangBang.append!!` into a fresh, untracked buffer. Errors, never silent.
    @test_throws Exception rev_gradient(x -> tmapreduce(sin, +, x), copy(X8))
    @test_throws Exception rev_gradient(x -> sum(tmap(sin, x)), copy(X8))
end
