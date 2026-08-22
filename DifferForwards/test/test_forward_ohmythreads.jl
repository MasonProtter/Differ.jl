# Forward mode through OhMyThreads.jl, exercised as an ordinary downstream package — no extension,
# no OhMyThreads-specific rules. Everything flows through the generic engine plus the
# `Task`/`fetch`/`Channel` rules in `rules_threads.jl`: StableTasks' `@spawn` expands to
# `Task(wrapper)` + `_spawn_set_thrpool` + `schedule`, its value transport is an `AtomicRef{T}`
# written inside the task, and `fetch(::StableTask)` is plain composite code the engine recurses
# into.

using Test
using DifferForwards
using DifferForwards: Dual, frule!!, primal, tangent, zero_dual
using OhMyThreads, StableTasks
include("testutils.jl")

@testset "thread pool is large enough for these tests to mean anything" begin
    @test Threads.threadpoolsize() > 1
end

const X8 = collect(1.0:8.0)

@testset "StableTasks: @spawn + fetch" begin
    f = x -> fetch(StableTasks.@spawn sum(abs2, x))
    d = frule!!(zero_dual(f), Dual(copy(X8), ones(8)))
    @test primal(d) ≈ sum(abs2, X8)
    @test tangent(d) ≈ sum(2.0 .* X8)
end

@testset "tmapreduce" begin
    for f in (x -> tmapreduce(sin, +, x),                                            # default (chunked)
              x -> tmapreduce(i -> sin(x[i]), +, eachindex(x)),                      # index form
              x -> tmapreduce(sin, +, x; scheduler=DynamicScheduler(; chunking=false)),
              x -> tmapreduce(i -> sin(x[i]), +, 1:length(x); nchunks=2))
        d = frule!!(zero_dual(f), Dual(copy(X8), ones(8)))
        @test primal(d) ≈ sum(sin.(X8))
        @test tangent(d) ≈ sum(cos.(X8))
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
    d = frule!!(zero_dual(tf), Dual(copy(X8), ones(8)))
    @test tangent(d) ≈ sum(cos.(X8))
end

@testset "tforeach with heterogeneous captures" begin
    function tfh(a, x)
        y = similar(x)
        tforeach(eachindex(x)) do i
            y[i] = a * sin(x[i])
        end
        return sum(y)
    end
    # seed only `a`
    d = frule!!(zero_dual(tfh), Dual(3.0, 1.0), Dual(copy(X8), zeros(8)))
    @test tangent(d) ≈ sum(sin.(X8))
    # seed only `x`
    d = frule!!(zero_dual(tfh), Dual(3.0, 0.0), Dual(copy(X8), ones(8)))
    @test tangent(d) ≈ 3.0 * sum(cos.(X8))
end

@testset "known gaps stay loud" begin
    # `tmap`/`tmap!` collect their output through `BangBang.append!!`/heterogeneous-capture
    # reflection the engines don't support yet — an error, never a silent wrong tangent.
    @test_throws Exception frule!!(zero_dual(x -> sum(tmap(sin, x))), Dual(copy(X8), ones(8)))
end
