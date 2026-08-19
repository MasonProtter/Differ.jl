# Implied merges (`_implied_merges`, `reverse_interp.jl`): a 2-predecessor merge whose following
# branch direction is a pure function of which predecessor fired — the `iterate`-end diamond every
# `for` loop lowers to — keeps no block-stack traffic at all: the pullback recovers the
# predecessor from which arm of its own reverse walk it arrives through, via a per-merge
# `Ref{Int32}` cell stored on the arms into the branch block's reverse code. Like the counted-loop
# scheme this is a two-sided push/pop change of exactly the class that silently corrupted
# gradients for >= 2 iterations when first attempted one-sided (see
# test_reverse_block_stack_split.jl), so the tests lean on gradient correctness across
# N in (0,1,2,3,5,50) — N=0/1 never exercise the replaced per-iteration pop — plus stack balance
# and exact traffic scaling, for eligible and ineligible shapes alike.

using Test
using DifferReverse
using DifferReverse: rev_gradient, rrule!!, zero_fcodual, Ctx, ImpliedMerge, _implied_merges,
                     _counted_loops, _unreachable_blocks

include(joinpath(@__DIR__, "testutils.jl"))

const CC = Core.Compiler

# Merge and branch in the same block: the unit-range `for` loop's iterate-end check is
# `not_int(φ(#a ⇒ true, #b ⇒ false))` feeding the `GotoIfNot` right in the merge block.
forsum(x, N) = (s = 0.0; for i in 1:N; s = s + x; end; s)

# Merge and branch in different blocks: `for x in v` lowers the constant-`φ` merge and its
# `GotoIfNot` one or two single-pred/single-succ blocks apart.
eachsum(v) = (s = 0.0; for x in v; s += x; end; s)

# Two chained iterate-end diamonds per iteration (one per zipped array). zip's combined "either
# side done" merge is 3-predecessor — two arms share a successor, so only a partial implication
# exists and the merge stays on the per-edge scheme (one push per iteration); the eight
# 2-predecessor merges around it all compress.
zipsum(v, w) = (s = 0.0; for (a, b) in zip(v, w); s += a * b; end; s)

# Bounds-checked reads: the loop body's collapsible `@boundscheck` regions must neither disqualify
# the iterate-end merge nor be disturbed by it.
mysum(v) = (s = 0.0; for i in 1:length(v); s += v[i]; end; s)

# A genuinely data-dependent diamond in the body: its merge's following branch is the iterate-end
# check, whose condition doesn't depend on which ternary arm ran — must stay per-iteration.
diamondloop(x, N) = (s = 0.0; for i in 1:N; s += (i % 2 == 0 ? x * x : x); end; s)

# A merge that feeds a `return`, not a branch: no downstream direction to read — ineligible.
mergereturn(x) = x < 0.0 ? -2.0 * x : x

# More loop shapes whose gradients must survive the new routing unchanged.
whilesum(x, N) = (s = 0.0; i = 1; while i <= N; s += x; i += 1; end; s)
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
contloop(x, N) = (s = 0.0; for i in 1:N; i % 2 == 0 && continue; s += x; end; s)
function nestedsum(x, N)
    s = 0.0
    for i in 1:N
        for j in 1:i
            s += x
        end
    end
    return s
end
function breaksum(x, N)
    s = 0.0
    for i in 1:N
        s += x
        s > 10.0 && break
    end
    return s
end

_implied_of(f, at) = begin
    pir = first(only(Base.code_ircode(f, at)))
    counted = _counted_loops(pir, _unreachable_blocks(pir), Dict{Int,Int}(), Set{Int}())
    pir, _implied_merges(pir, Base.get_world_counter(), _unreachable_blocks(pir),
                         Dict{Int,Int}(), Set{Int}(), counted)
end

@testset "_implied_merges: eligibility analysis on real primal IR" begin
    # Structural invariants every entry must satisfy, on whatever block numbers the optimizer
    # produced: the two mapped preds are exactly the merge's preds, the branch block ends in a
    # 2-way `GotoIfNot`, and the merge reaches the branch through single-pred/single-succ blocks.
    function check_invariants(pir, ims)
        preds(b) = filter(!=(0), pir.cfg.blocks[b].preds)
        succs(b) = pir.cfg.blocks[b].succs
        for im in ims
            @test Set(preds(im.merge)) == Set((im.dest_pred, im.fall_pred))
            term = pir.stmts[pir.cfg.blocks[im.branch].stmts.stop][:stmt]
            @test term isa Core.GotoIfNot
            @test length(unique(succs(im.branch))) == 2
            b = im.merge
            n = 0
            while b != im.branch
                @test length(succs(b)) == 1
                b = only(succs(b))
                @test length(preds(b)) == 1
                @test (n += 1) <= length(pir.cfg.blocks)
            end
        end
    end
    # Unit-range `for`: two eligible merges (the `1:N` non-empty check's true/false merge and the
    # iterate-end merge), both with merge == branch. The `1:N` construction's own `sle_int`
    # diamond is data-dependent and must not appear.
    pir, ims = _implied_of(forsum, (Float64, Int))
    @test length(ims) == 2
    @test all(im -> im.merge == im.branch, ims)
    check_invariants(pir, ims)
    # Iterator protocol: same two merges, but each `GotoIfNot` sits downstream of its merge.
    pir, ims = _implied_of(eachsum, (Vector{Float64},))
    @test length(ims) == 2
    @test all(im -> im.merge != im.branch, ims)
    check_invariants(pir, ims)
    # zip: per-array iterate-end and per-side done-flag diamonds, entry and in-loop — eight 2-pred
    # merges in all; the 3-pred combined "either side done" merge is not among them.
    pir, ims = _implied_of(zipsum, (Vector{Float64}, Vector{Float64}))
    @test length(ims) == 8
    check_invariants(pir, ims)
    # The data-dependent ternary merge must not qualify — only the range check and iterate-end do.
    pir, ims = _implied_of(diamondloop, (Float64, Int))
    @test length(ims) == 2
    check_invariants(pir, ims)
    # A merge flowing into `return` has no downstream branch to read.
    _, ims = _implied_of(mergereturn, (Float64,))
    @test isempty(ims)
end

@testset "gradient correctness across the routing boundary (N=0,1,2,3,5,50)" begin
    x = 3.0
    for N in (0, 1, 2, 3, 5, 50)
        for f in (forsum, whilesum, dowhilesum, contloop, nestedsum, breaksum, diamondloop)
            _, dx = rev_gradient(f, x, N)
            @test dx ≈ central_diff(z -> f(z, N), x) rtol = 1e-5
        end
        v, w = randn(N), randn(N)
        @test rev_gradient(mysum, v)[2] ≈ ones(N)
        @test rev_gradient(eachsum, v)[2] ≈ ones(N)
        dv, dw = rev_gradient(zipsum, v, w)[2:3]
        @test dv ≈ w && dw ≈ v
    end
    _, dm = rev_gradient(mergereturn, -1.5)
    @test dm == -2.0
    @test rev_gradient(mergereturn, 1.5)[2] == 1.0
end

@testset "IR verification and stack balance (prealloc reuse included)" begin
    for f in (forsum, dowhilesum, contloop, nestedsum, breaksum, diamondloop)
        checkverify_rev(f, (Float64, Int))
        for N in (0, 1, 2, 3, 5)
            check_stack_balance(f, 3.0, N)
        end
    end
    for f in (eachsum, mysum)
        checkverify_rev(f, (Vector{Float64},))
        for N in (0, 1, 3, 5)
            check_stack_balance(f, randn(N))
        end
    end
    checkverify_rev(zipsum, (Vector{Float64}, Vector{Float64}))
    check_stack_balance(zipsum, randn(4), randn(4))
    checkverify_rev(mergereturn, (Float64,))
    check_stack_balance(mergereturn, -1.5)
end

@testset "traffic: implied merges are free, data-dependent ones still pay" begin
    # Flat at any N: what's left is the `1:N` empty-range diamond (data-dependent) and the
    # loop-exit -> exit-merge edge; the iterator-protocol shapes have no range diamond at all.
    b2, c2 = loop_traffic(forsum, 3.0, 2)
    b10k, c10k = loop_traffic(forsum, 3.0, 10_000)
    @test c2 == c10k == 1
    @test b2 == b10k
    for N in (2, 100)
        @test loop_traffic(eachsum, randn(N)) == loop_traffic(eachsum, randn(2))
        @test loop_traffic(mysum, randn(N)) == loop_traffic(mysum, randn(2))
    end
    # zip's 3-pred combined done-merge keeps exactly one push per iteration (a partial
    # implication only — see the fixture comment); everything else around it is compressed.
    z2, zc2 = loop_traffic(zipsum, randn(2), randn(2))
    z100, zc100 = loop_traffic(zipsum, randn(100), randn(100))
    @test zc2 == zc100 == 1
    @test z100 - z2 == 100 - 2
    # The ternary merge's push survives — exactly one per iteration, no more.
    d2, _ = loop_traffic(diamondloop, 3.0, 2)
    d100, _ = loop_traffic(diamondloop, 3.0, 100)
    @test d100 - d2 == 100 - 2
end
