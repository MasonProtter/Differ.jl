# ISSUES #52: the reverse-mode forwards carrier used to push a block's number onto the block stack
# unconditionally whenever the block wasn't a unique predecessor of *all* its successors, even though
# the push is only ever popped on the one successor edge that's genuinely ambiguous. `loopsum` below
# is the validated fixture: a `for` loop whose per-iteration "continue vs. exit" check is a
# `GotoIfNot` with one unambiguous arm (loop body, single predecessor) and one ambiguous arm (the
# shared loop-exit merge, reached both from here and from the "zero-trip" skip check) — exactly the
# loop-exit-diamond shape ISSUES #49 measured as the largest remaining per-iteration reverse-mode
# cost.
#
# STATUS: FIXED. `_split_ambiguous_block_pushes` (`src/reverse_interp.jl`) is wired into
# `reverse_fwds_to_ircode` (the one-line call before `CC.verify_ir(ir)`), paired with the per-edge
# `pred_is_unique_pred` formula (`length(preds[b]) <= 1`) in `_unique_predecessor_info` that stops the
# pullback's single-predecessor balance-pop. The two changes are coupled; neither is correct alone
# (see ISSUES.md #52). The direct unit tests below cover the surgery itself; the `gradient`-level
# testsets exercise the live fix across the disambiguation boundary (N=0,1,2,3,5,50), and the
# `memloop!` traffic test asserts the 3N+2 → 2N+3 reduction (the remaining 2/iter are irreducible).
#
# A second, *fallthrough*-ambiguous shape (the mirror insertion case: the ambiguous arm is the
# implicit fallthrough rather than the explicit `dest`) is exercised too, but not via a real Julia
# primal — see the comment above `_raw_mixed_candidates` for why. Every structured-control-flow
# shape tried (if/else in every polarity, `||`/ternary, `for`/`while` in both directions, `break`/
# `continue`, `try`/`catch`, if/elseif chains, multi-exit loops) reliably produces the "skip" arm as
# `GotoIfNot`'s `dest`, never as the fallthrough — Julia's own `if`/`while`/`for` lowering always
# encodes "skip past code" as the explicit jump. The fallthrough-ambiguous fixture here is therefore
# a hand-built `IRCode`, in the same style `test_cfg_ir.jl` uses for its own round-trip tests, just
# assembled from `CFGBlock`/`ID` primitives rather than compiled from source.

using Test
using Differ
using Differ: gradient
using Differ: ID, CFGBlock, IDGotoIfNot, IDGotoNode, IDPhiNode, new_inst,
              _ircode_to_cfg_blocks, lower_cfg_blocks_to_ir, phi_nodes, terminator,
              _split_ambiguous_block_pushes, _is_expected_block_push, Stack
using Differ: CoDual, NoRData, rrule!!, build_ctx, zero_fcodual, Ctx

include(joinpath(@__DIR__, "testutils.jl"))

const CC = Core.Compiler

loopsum(x, N) = (s = 0.0; for i in 1:N; s = s + x; end; s)

# Every block `b` (not the last) whose terminator is a `GotoIfNot` with exactly one ambiguous
# successor (>1 real predecessor) — i.e. exactly `_split_ambiguous_block_pushes`'s Stage 0
# classification, reimplemented here (over a plain `IRCode`, not `pir`-specific) so this file can
# assert on a fixture's *shape* without reaching into `Differ`'s own classification helper.
function _raw_mixed_candidates(ir::CC.IRCode)
    nblocks = length(ir.cfg.blocks)
    out = Tuple{Int,Symbol,Int}[]
    for b in 1:(nblocks - 1)
        term = ir.stmts[ir.cfg.blocks[b].stmts.stop][:stmt]
        isa(term, Core.GotoIfNot) || continue
        dest, fall = Int(term.dest), b + 1
        npd = length(filter(!=(0), ir.cfg.blocks[dest].preds))
        npf = length(filter(!=(0), ir.cfg.blocks[fall].preds))
        if npd > 1 && npf > 1
            continue
        elseif npd > 1
            push!(out, (b, :dest, dest))
        elseif npf > 1
            push!(out, (b, :fallthrough, fall))
        end
    end
    return out
end

@testset "loopsum: primal IR has the dest-ambiguous loop-exit-diamond shape" begin
    pir = first(only(Base.code_ircode(loopsum, (Float64, Int))))
    cs = _raw_mixed_candidates(pir)
    @test !isempty(cs)
    @test any(c -> c[2] === :dest, cs)
    b, side, target = only(filter(c -> c[2] === :dest, cs))
    # The ambiguous target must carry a `PhiNode` for this to exercise the edge-fixup path:
    # redirecting `b`'s edge through a relay changes the target's real predecessor, and a `PhiNode`
    # there is exactly what goes stale if that's missed.
    target_stmts = pir.cfg.blocks[target].stmts
    @test isa(pir.stmts[target_stmts.start][:stmt], Core.PhiNode)
end

@testset "loopsum: gradient correctness across the disambiguation boundary (N=0,1,2,3,5,50)" begin
    x = 3.0
    for N in (0, 1, 2, 3, 5, 50)
        _, dx = gradient(loopsum, x, N)
        @test dx ≈ N atol = 1e-9
        h = 1e-6
        @test dx ≈ central_diff(z -> loopsum(z, N), x) rtol = 1e-5
    end
end

@testset "loopsum: IR verification and stack balance" begin
    checkverify_rev(loopsum, (Float64, Int))
    for N in (0, 1, 2, 3, 5, 50)
        check_stack_balance(loopsum, 3.0, N)
    end
end

# Has the block-stack push shape `_is_expected_block_push` looks for.
_has_push(b::CFGBlock) =
    any(i -> i.stmt isa Expr && i.stmt.head === :call && i.stmt.args[1] === Base.push!, b.insts)

# The relay `_split_ambiguous_block_pushes` inserts: exactly the push plus an unconditional goto.
_is_relay(b::CFGBlock) = length(b) == 2 && terminator(b) isa IDGotoNode && _has_push(b)

@testset "_split_ambiguous_block_pushes: direct unit test, dest-ambiguous case (loopsum-shaped)" begin
    # Hand-built mirror of loopsum's own block-13-style shape (see the file header): `pre` stands in
    # for the "did we skip the whole loop" check (both its own arms are already ambiguous, exactly
    # like loopsum's block 9, so it is *not* itself a candidate). `chk` mirrors block 13: it
    # branches to `exitb` (dest, ambiguous, also reached directly from `pre`) or `body`
    # (fallthrough, unambiguous), and `exitb` carries a `PhiNode`, the edge-fixup path.
    template_ir = first(only(Base.code_ircode(x -> x, (Bool,))))

    pre, chk, body, exitb = ID(), ID(), ID(), ID()
    stack_iid, push_iid, term_chk_iid = ID(), ID(), ID()
    add_iid, term_body_iid = ID(), ID()
    phi_iid, ret_iid = ID(), ID()

    block_pre = CFGBlock(pre, [ID()], CC.NewInstruction[new_inst(IDGotoIfNot(Core.Argument(2), exitb), Any)])
    block_chk = CFGBlock(chk, [stack_iid, push_iid, term_chk_iid],
        CC.NewInstruction[
            new_inst(Expr(:call, Stack{Int32}), Stack{Int32}),
            new_inst(Expr(:call, Base.push!, stack_iid, Int32(2)), Any),
            new_inst(IDGotoIfNot(Core.Argument(2), exitb), Any),
        ])
    block_body = CFGBlock(body, [add_iid, term_body_iid],
        CC.NewInstruction[
            new_inst(Expr(:call, GlobalRef(Core.Intrinsics, :add_float), 1.0, 1.0), Float64),
            new_inst(IDGotoNode(chk), Any),
        ])
    block_exit = CFGBlock(exitb, [phi_iid, ret_iid],
        CC.NewInstruction[
            new_inst(IDPhiNode(ID[pre, chk], Any[0.0, 1.0]), Float64),
            new_inst(Core.ReturnNode(phi_iid), Any),
        ])

    blks = [block_pre, block_chk, block_body, block_exit]   # block numbers 1,2,3,4
    ir_built = lower_cfg_blocks_to_ir(blks, template_ir; argtypes=Any[Tuple{}, Bool],
                                      def=template_ir.debuginfo.def)
    CC.verify_ir(ir_built)   # the fixture itself must be legal before testing the split

    cs = _raw_mixed_candidates(ir_built)
    @test cs == [(2, :dest, 4)]   # only `chk` (block 2) -- `pre` is both-ambiguous, already optimal

    is_unique_pred = falses(4)
    result = _split_ambiguous_block_pushes(ir_built, ir_built, is_unique_pred)
    CC.verify_ir(result)
    @test length(result.cfg.blocks) == 5   # one relay appended

    # `ID`s minted by `lower_cfg_blocks_to_ir`'s own round trip don't survive being converted back to
    # a real `IRCode` and re-read via `_ircode_to_cfg_blocks` (fresh `ID`s are assigned from block
    # *position*, same as `_split_ambiguous_block_pushes` does internally), so blocks in `result`
    # are identified structurally below, not by reusing the `pre`/`chk`/... variables above.
    rblks = _ircode_to_cfg_blocks(result)
    relay = only(filter(_is_relay, rblks))
    gotoifnot_blks = filter(b -> terminator(b) isa IDGotoIfNot, rblks)
    @test length(gotoifnot_blks) == 2
    chk_blk = only(filter(b -> terminator(b).dest == relay.id, gotoifnot_blks))
    pre_blk = only(filter(b -> b !== chk_blk, gotoifnot_blks))
    merge_blk = only(filter(b -> !isempty(phi_nodes(b)[2]), rblks))

    @test !_has_push(chk_blk)                             # push moved out of `chk`
    @test terminator(relay) == IDGotoNode(merge_blk.id)    # relay forwards on to the real target
    @test terminator(pre_blk).dest == merge_blk.id         # `pre`'s own edge into the merge is untouched

    phi = phi_nodes(merge_blk)[2][1].stmt
    @test Set(phi.edges) == Set([pre_blk.id, relay.id])    # `chk`'s edge renamed to the relay
end

@testset "_split_ambiguous_block_pushes: direct unit test, fallthrough-ambiguous case" begin
    # The mirror shape: block `chk`'s *fallthrough* (not `dest`) is the ambiguous arm. `merge`
    # carries a `PhiNode` and is reached both as `chk`'s fallthrough and via a back-edge from `body`.
    # `chk`'s `dest` goes straight to a single-predecessor `ret` block. No source-level Julia function
    # was found that produces this polarity (see the file header), so it's built directly via `cfg_ir.jl`.
    template_ir = first(only(Base.code_ircode(x -> x, (Bool,))))

    chk, merge, ret, body = ID(), ID(), ID(), ID()
    stack_iid, push_iid, term_chk_iid = ID(), ID(), ID()
    phi_iid, term_merge_iid = ID(), ID()
    ret_iid = ID()
    add_iid, term_body_iid = ID(), ID()

    block_chk = CFGBlock(chk, [stack_iid, push_iid, term_chk_iid],
        CC.NewInstruction[
            new_inst(Expr(:call, Stack{Int32}), Stack{Int32}),
            new_inst(Expr(:call, Base.push!, stack_iid, Int32(1)), Any),
            new_inst(IDGotoIfNot(Core.Argument(2), ret), Any),
        ])
    block_merge = CFGBlock(merge, [phi_iid, term_merge_iid],
        CC.NewInstruction[
            new_inst(IDPhiNode(ID[chk, body], Any[0.0, add_iid]), Float64),
            new_inst(IDGotoNode(body), Any),
        ])
    block_ret = CFGBlock(ret, [ret_iid], CC.NewInstruction[new_inst(Core.ReturnNode(1.0), Any)])
    block_body = CFGBlock(body, [add_iid, term_body_iid],
        CC.NewInstruction[
            new_inst(Expr(:call, GlobalRef(Core.Intrinsics, :add_float), phi_iid, 1.0), Float64),
            new_inst(IDGotoNode(merge), Any),
        ])

    blks = [block_chk, block_merge, block_ret, block_body]   # block numbers 1,2,3,4
    ir_built = lower_cfg_blocks_to_ir(blks, template_ir; argtypes=Any[Tuple{}, Bool],
                                      def=template_ir.debuginfo.def)
    CC.verify_ir(ir_built)

    cs = _raw_mixed_candidates(ir_built)
    @test cs == [(1, :fallthrough, 2)]

    is_unique_pred = falses(4)
    result = _split_ambiguous_block_pushes(ir_built, ir_built, is_unique_pred)
    CC.verify_ir(result)
    @test length(result.cfg.blocks) == 5   # one relay inserted right after `chk`

    # See the comment in the dest-ambiguous testset above: `ID`s don't survive the round trip, so
    # blocks in `result` are identified structurally, not via the `chk`/`merge`/... variables above.
    rblks = _ircode_to_cfg_blocks(result)
    relay = only(filter(_is_relay, rblks))
    chk_blk = only(filter(b -> terminator(b) isa IDGotoIfNot, rblks))   # the only GotoIfNot here
    ret_blk = only(filter(b -> terminator(b) isa Core.ReturnNode, rblks))
    merge_blk = only(filter(b -> !isempty(phi_nodes(b)[2]), rblks))
    body_blk = only(filter(b -> b !== relay && b !== chk_blk && b !== ret_blk && b !== merge_blk, rblks))

    @test !_has_push(chk_blk)
    @test terminator(chk_blk).dest == ret_blk.id           # `dest` is untouched
    @test terminator(relay) == IDGotoNode(merge_blk.id)    # relay forwards on to the real target

    phi = phi_nodes(merge_blk)[2][1].stmt
    @test Set(phi.edges) == Set([relay.id, body_blk.id])   # `chk`'s edge renamed to the relay
end

@testset "memloop!: block-stack traffic scales 2N+3, not 3N+2 (ISSUES #52)" begin
    # Same shape as `bench/workloads.jl`'s `memloop!` benchmark, redefined locally rather than
    # `include`d (the bench project pulls in `BenchmarkTools`, not a `test/Project.toml` dependency).
    # `Memory` has no `zero_tangent` method (ISSUES #50), so its `CoDual` is built by hand, as the
    # benchmark does.
    memloop!(o::Memory{Float64}, x::Float64, N::Int) = (for i in 1:N; @inbounds o[i] = x; end; nothing)

    function run_memloop(N)
        o = Memory{Float64}(undef, N); fill!(o, 0.0)
        d = Memory{Float64}(undef, N); fill!(d, 0.0)
        ocd = CoDual(o, d)
        fcd = zero_fcodual(memloop!)
        xcd = zero_fcodual(3.0)
        ctx = build_ctx(memloop!, (Memory{Float64}, Float64, Int); prealloc=false)
        y, pb = rrule!!(fcd, ctx, ocd, xcd, zero_fcodual(N))
        pb(NoRData())
        @test pb.block_stack.position == 0
        @test all(s -> !(s isa Differ.Stack) || s.position == 0, pb.comms)
        return length(pb.block_stack.memory)
    end

    # The fix removes the wasteful loop-exit-diamond per-block push (ISSUES #49): traffic drops from
    # 3N+2 to 2N+3. The remaining 2/iteration are irreducible (the loop header's two real
    # predecessors, and the loop-body→merge edge's two real predecessors, both need runtime
    # disambiguation — see the ISSUES #52 writeup), so this asserts the *reduction*, not flatness.
    # Measured: N=2→7, N=3→9, N=5→13, N=100→203, N=10_000→20_003.
    @test run_memloop(2)  == 7
    @test run_memloop(5)  == 13
    large = run_memloop(10_000)
    @test large ≤ 3 * 10_000 + 4    # comfortably below the old 3N+2 floor
    @test large > 2 * 10_000        # and not better than the irreducible 2N floor
    @test large == 2 * 10_000 + 3   # exact: 2N+3
end
