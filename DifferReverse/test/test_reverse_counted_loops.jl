# Counted-loop trip-count compression (`_counted_loops`, `reverse_interp.jl`): a single-latch
# natural loop's per-iteration block-stack pushes are replaced by one `Int64` trip count pushed on
# the loop-exit edge (onto the tape's dedicated `count_stack`) and consumed by a countdown in the
# pullback. This is a two-sided push/pop change of exactly the class that silently corrupted
# gradients for >= 2 iterations when first attempted one-sided, so the tests here lean on gradient
# correctness across N in (0,1,2,3,5,50) — N=0/1 never exercise the back edge — plus stack balance
# and exact traffic scaling, for eligible and ineligible loop shapes alike.

using Test
using DifferReverse
using DifferReverse: rev_gradient, rrule!!, zero_fcodual, Ctx, CountedLoop, _counted_loops,
                     _unreachable_blocks

include(joinpath(@__DIR__, "testutils.jl"))

const CC = Core.Compiler

# `while`-shaped: the exiting block is the header itself; fully compressible (zero block-stack
# traffic — the only ambiguity was the header arrival).
whilesum(x, N) = (s = 0.0; i = 1; while i <= N; s += x; i += 1; end; s)

# `for`-shaped: exits from a mid-body block (the iterate-end check), whose merge keeps one
# genuinely data-dependent push per iteration.
forsum(x, N) = (s = 0.0; for i in 1:N; s = s + x; end; s)

# Body-first (do-while): the body runs before the exit check, so N=0 still runs once.
function dowhilesum(x, N)
    s = 0.0
    i = 0
    while true
        s += x
        i += 1
        i < N || break
    end
    return s
end

# A data-dependent branch inside the body: its merge's per-iteration pushes must survive
# compression of the header.
diamondloop(x, N) = (s = 0.0; for i in 1:N; s += (i % 2 == 0 ? x * x : x); end; s)

# `continue` into the loop's update path: eligible regardless of how many predecessors the latch
# ends up with, since the count stack is separate from the block stack.
contloop(x, N) = (s = 0.0; for i in 1:N; i % 2 == 0 && continue; s += x; end; s)

# Nested loops, inner trip count varying with the outer index: counts push per exit event, so the
# inner loop records one count per outer iteration and LIFO order matches the reverse walk.
function nestedsum(x, N)
    s = 0.0
    for i in 1:N
        for j in 1:i
            s += x
        end
    end
    return s
end

# `break` gives the loop a second exit edge — ineligible, must keep the ordinary per-edge scheme.
function breaksum(x, N)
    s = 0.0
    for i in 1:N
        s += x
        s > 10.0 && break
    end
    return s
end

# Two `@goto` back edges to one label. Depending on lowering this is either a multi-latch loop
# (ineligible) or a merged single latch (eligible); either way the gradient must be right.
function twoback(x, N)
    s = 0.0
    i = 0
    @label top
    i += 1
    s += x
    if i < N
        if i % 2 == 0
            @goto top
        end
        @goto top
    end
    return s
end

_counted_of(f, at) = begin
    pir = first(only(Base.code_ircode(f, at)))
    _counted_loops(pir, _unreachable_blocks(pir), Dict{Int,Int}(), Set{Int}())
end

@testset "_counted_loops: eligibility analysis on real primal IR" begin
    for (f, at) in ((whilesum, (Float64, Int)), (forsum, (Float64, Int)))
        pir = first(only(Base.code_ircode(f, at)))
        loops = _counted_loops(pir, _unreachable_blocks(pir), Dict{Int,Int}(), Set{Int}())
        @test length(loops) == 1
        cl = only(loops)
        preds(b) = filter(!=(0), pir.cfg.blocks[b].preds)
        succs(b) = pir.cfg.blocks[b].succs
        @test Set(preds(cl.header)) == Set((cl.preheader, cl.latch))
        @test cl.header in succs(cl.latch)
        @test Set(succs(cl.exiting)) == Set((cl.exit_target, cl.inloop_succ))
        # while-shape exits from the header itself; for-shape from a mid-body block.
        f === whilesum && @test cl.exiting == cl.header
        f === forsum && @test cl.exiting != cl.header
    end
    @test length(_counted_of(nestedsum, (Float64, Int))) == 2
    # `break` adds a second exit edge: the whole loop must fall back to the per-edge scheme.
    @test isempty(_counted_of(breaksum, (Float64, Int)))
end

@testset "gradient correctness across the compression boundary (N=0,1,2,3,5,50)" begin
    x = 3.0
    for N in (0, 1, 2, 3, 5, 50)
        for f in (whilesum, forsum, dowhilesum, diamondloop, contloop, nestedsum, breaksum, twoback)
            _, dx = rev_gradient(f, x, N)
            @test dx ≈ central_diff(z -> f(z, N), x) rtol = 1e-5
        end
    end
end

@testset "IR verification and stack balance (prealloc reuse included)" begin
    for f in (whilesum, forsum, dowhilesum, diamondloop, contloop, nestedsum, breaksum, twoback)
        checkverify_rev(f, (Float64, Int))
        for N in (0, 1, 2, 3, 5)
            check_stack_balance(f, 3.0, N)
        end
    end
end

# Peak stack usage of one fresh round trip: (block-stack slots, count-stack slots).
function _loop_traffic(f, args...)
    ctx = Ctx()
    fcd, argcds = zero_fcodual(f), map(zero_fcodual, args)
    ycd, pb = rrule!!(fcd, ctx, argcds...)
    pb(one(DifferReverse.primal(ycd)))
    @test pb.block_stack.position == 0
    @test pb.count_stack.position == 0
    return length(pb.block_stack.memory), length(pb.count_stack.memory)
end

@testset "traffic: compressible shapes are flat, data-dependent pushes survive" begin
    # while-shape: the header arrival was the only ambiguity — zero block-stack traffic at any N,
    # one recorded count.
    for N in (0, 2, 5, 100, 10_000)
        @test _loop_traffic(whilesum, 3.0, N) == (0, 1)
    end
    # for-shape: the iterate-end merge's push per iteration remains; the header's is gone.
    b2, c2 = _loop_traffic(forsum, 3.0, 2)
    b100, c100 = _loop_traffic(forsum, 3.0, 100)
    @test c2 == c100 == 1
    @test b100 - b2 == 100 - 2            # exactly one push per additional iteration
    # data-dependent diamond: still one push per iteration for the merge, none for the header.
    d2, _ = _loop_traffic(diamondloop, 3.0, 2)
    d100, _ = _loop_traffic(diamondloop, 3.0, 100)
    @test d100 - d2 >= 100 - 2
    # nested: the inner loop records one count per outer iteration, plus the outer loop's own.
    _, cn = _loop_traffic(nestedsum, 3.0, 7)
    @test cn == 7 + 1
    # ineligible `break` loop: per-iteration block-stack traffic, no counts recorded. `s > 10`
    # first trips at N >= 5 for x = 3.0, so N=3 runs the loop to completion.
    bb, cb = _loop_traffic(breaksum, 3.0, 3)
    @test cb == 0
    @test bb > 0
end
