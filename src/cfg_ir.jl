# ===========================================================================
# The reverse-mode working IR: an immutable, basic-block representation of `Core.Compiler.IRCode`
# keyed by `ID` rather than by position, so blocks can be inserted/removed/reordered while building
# the pullback pass (Mooncake's terminology and design — ported from
# `/project/Mooncake.jl/src/interpreter/{ir_utils,reverse_mode}.jl`, lines covering the generic
# CFGBlock/ID layer only, none of the AD-specific `rrule!!`/`CoDual` logic).
#
# `dualize_to_ircode` (`forward_interp.jl`) never needed this: forward mode preserves the primal's
# block topology 1:1 (only within-block instruction counts change), so plain block numbers suffice.
# The reverse-mode pullback pass does not preserve topology — it inserts extra phi-routing blocks and
# lowers `Switch` dispatches into `GotoIfNot` chains — so block *identity* must survive insertion/
# reordering, which position-based numbering cannot do. Hence the `ID` indirection here.
#
# Adapted for Differ (Julia 1.13 only, no multi-version `@static if` branching) and for this
# project's existing conventions (`Core.PhiNode`/`Core.GotoNode`/... qualified, `CC` alias for
# `Core.Compiler` from `contextual.jl`, bare `copy` rather than `CC.copy`).
# ===========================================================================

# ---------------------------------------------------------------------------
# ID: a unique name for a block or statement in the working IR, independent of position.
# ---------------------------------------------------------------------------

const _id_count = Dict{Int,Int32}()

"""
    ID()

An `ID` (read: unique name) is just a wrapper around an `Int32`. Uniqueness is ensured via a
global (per-thread) counter, incremented each time an `ID` is created.
"""
struct ID
    id::Int32
    function ID()
        tid = Threads.threadid()
        n = get(_id_count, tid, Int32(0))
        _id_count[tid] = n + Int32(1)
        return new(n)
    end
end

Base.copy(id::ID) = id

"""
    seed_id!()

Reset the global `ID`-uniqueness counter to `0`. Useful for deterministic `ID`s across runs (e.g.
in tests).
"""
seed_id!() = (global _id_count[Threads.threadid()] = 0)

# ---------------------------------------------------------------------------
# Instructions: reuse `Core.Compiler.NewInstruction` as the per-statement representation
# throughout the working IR (not just for genuinely new instructions, as `Core.Compiler` itself
# uses it).
# ---------------------------------------------------------------------------

const InstVector = Vector{CC.NewInstruction}

"""
    new_inst(stmt, type=Any, flag=CC.IR_FLAG_REFINED)::CC.NewInstruction

Convenience constructor: a `CC.NewInstruction` with a fresh `CC.NoCallInfo()` and a placeholder
source line (`Int32(1)`, converted by `NewInstruction`'s own inner constructor into the
`(Int32,Int32,Int32)` codeloc triple `lower_cfg_blocks_to_ir` later flattens back out) — `nothing`
would mean "copy the line from whatever precedes this at the insertion point", which doesn't apply
here since every instruction in this working-IR layer is freshly built, not spliced into existing IR.
"""
new_inst(@nospecialize(stmt), @nospecialize(type)=Any, flag=CC.IR_FLAG_REFINED) =
    CC.NewInstruction(stmt, type, CC.NoCallInfo(), Int32(1), flag)

# ---------------------------------------------------------------------------
# ID-addressed analogues of `Core.PhiNode`/`GotoNode`/`GotoIfNot`, plus `Switch`, a pseudo-node for
# a multi-way branch (not a real Julia IR node — lowered to a `GotoIfNot` chain by
# `lower_cfg_blocks_to_ir` before the result is handed back as a real `IRCode`).
# ---------------------------------------------------------------------------

"""
    IDPhiNode(edges::Vector{ID}, values::Vector{Any})

Like `Core.PhiNode`, but `edges` are `ID`s rather than `Int32` block numbers.
"""
struct IDPhiNode
    edges::Vector{ID}
    values::Vector{Any}
end

Base.:(==)(x::IDPhiNode, y::IDPhiNode) = x.edges == y.edges && x.values == y.values
Base.copy(node::IDPhiNode) = IDPhiNode(copy(node.edges), copy(node.values))

"""
    IDGotoNode(label::ID)

Like `Core.GotoNode`, but `label` is an `ID` rather than an `Int64`.
"""
struct IDGotoNode
    label::ID
end
Base.copy(node::IDGotoNode) = IDGotoNode(copy(node.label))

"""
    IDGotoIfNot(cond::Any, dest::ID)

Like `Core.GotoIfNot`, but `dest` is an `ID` rather than an `Int64`.
"""
struct IDGotoIfNot
    cond::Any
    dest::ID
end
Base.copy(node::IDGotoIfNot) = IDGotoIfNot(copy(node.cond), copy(node.dest))

"""
    Switch(conds::Vector{Any}, dests::Vector{ID}, fallthrough_dest::ID)

A switch-statement pseudo-node, insertable into a `CFGBlock`, with semantics
```julia
goto dests[1] if not conds[1]
goto dests[2] if not conds[2]
...
goto dests[N] if not conds[N]
goto fallthrough_dest
```
Each `conds[i]` is a `Bool`-typed value/ID. Not a real Julia IR node: `lower_cfg_blocks_to_ir` lowers
it into the `GotoIfNot`/`GotoNode` chain above before producing a real `IRCode`.
"""
struct Switch
    conds::Vector{Any}
    dests::Vector{ID}
    fallthrough_dest::ID
    function Switch(conds::Vector{Any}, dests::Vector{ID}, fallthrough_dest::ID)
        @assert length(conds) == length(dests)
        return new(conds, dests, fallthrough_dest)
    end
end

"""
    Terminator = Union{Switch, IDGotoIfNot, IDGotoNode, Core.ReturnNode}
"""
const Terminator = Union{Switch,IDGotoIfNot,IDGotoNode,Core.ReturnNode}

# ---------------------------------------------------------------------------
# CFGBlock: an immutable, `ID`-named basic block.
# ---------------------------------------------------------------------------

"""
    CFGBlock(id::ID, inst_ids::Vector{ID}, insts::InstVector)

An immutable basic block: the working-IR unit reverse mode builds and rearranges. `id` is a unique,
position-independent name for the block. The `n`th line of code is associated to `ID` `inst_ids[n]`
and instruction `insts[n]`.

`Core.PhiNode`/`Core.GotoIfNot`/`Core.GotoNode` must never appear inside a `CFGBlock` — use
`IDPhiNode`/`IDGotoIfNot`/`IDGotoNode` instead. A `CFGBlock` is immutable: build a new one rather
than editing in place (see `insert_before_terminator`).
"""
struct CFGBlock
    id::ID
    inst_ids::Vector{ID}
    insts::InstVector
    function CFGBlock(id::ID, inst_ids::Vector{ID}, insts::InstVector)
        @assert length(inst_ids) == length(insts)
        return new(id, inst_ids, insts)
    end
end

const IDInstPair = Tuple{ID,CC.NewInstruction}

"""
    CFGBlock(id::ID, inst_pairs::Vector{IDInstPair})

Convenience constructor: splits `inst_pairs` into parallel `ID`/instruction vectors.
"""
CFGBlock(id::ID, inst_pairs::Vector{IDInstPair}) = CFGBlock(id, first.(inst_pairs), last.(inst_pairs))

Base.length(bb::CFGBlock) = length(bb.inst_ids)
Base.copy(bb::CFGBlock) = CFGBlock(bb.id, copy(bb.inst_ids), copy(bb.insts))

"""
    phi_nodes(bb::CFGBlock)::Tuple{Vector{ID}, Vector{IDPhiNode}}

All `IDPhiNode`s at the start of `bb` (Julia IR guarantees phis are contiguous at block start),
along with their `ID`s. Empty vectors if `bb` has none.
"""
function phi_nodes(bb::CFGBlock)
    n = findlast(x -> x.stmt isa IDPhiNode, bb.insts)
    n = n === nothing ? 0 : n
    return bb.inst_ids[1:n], bb.insts[1:n]
end

"""
    terminator(bb::CFGBlock)

`bb`'s terminator (a `Terminator`) if its last instruction is one, else `nothing`.
"""
terminator(bb::CFGBlock) = isa(bb.insts[end].stmt, Terminator) ? bb.insts[end].stmt : nothing

"""
    insert_before_terminator(insts::Vector{IDInstPair}, extra::Vector{IDInstPair})::Vector{IDInstPair}

Pure/functional splice: `extra` inserted immediately before `insts`'s terminator (or appended at the
end if there is none). `insts` is not mutated.
"""
function insert_before_terminator(insts::Vector{IDInstPair}, extra::Vector{IDInstPair})
    isempty(extra) && return insts
    has_terminator = !isempty(insts) && last(insts)[2].stmt isa Terminator
    pos = length(insts) + (has_terminator ? 0 : 1)
    return vcat(insts[1:(pos - 1)], extra, insts[pos:end])
end

collect_stmts(bb::CFGBlock)::Vector{IDInstPair} = collect(zip(bb.inst_ids, bb.insts))
collect_stmts(blks::Vector{CFGBlock})::Vector{IDInstPair} = reduce(vcat, map(collect_stmts, blks))
concatenate_ids(blks::Vector{CFGBlock}) = reduce(vcat, map(b -> b.inst_ids, blks))
concatenate_stmts(blks::Vector{CFGBlock}) = reduce(vcat, map(b -> b.insts, blks))

# ---------------------------------------------------------------------------
# CFG adjacency, over `ID`s.
# ---------------------------------------------------------------------------

"""
    _compute_cfg_successors(blks::Vector{CFGBlock})::Dict{ID, Vector{ID}}

Map from each block's `ID` to its possible successor `ID`s.
"""
@noinline function _compute_cfg_successors(blks::Vector{CFGBlock})::Dict{ID,Vector{ID}}
    succs = map(enumerate(blks)) do (n, blk)
        is_final = n == length(blks)
        t = terminator(blk)
        if t === nothing
            return is_final ? ID[] : ID[blks[n + 1].id]
        elseif t isa IDGotoNode
            return ID[t.label]
        elseif t isa IDGotoIfNot
            return is_final ? ID[t.dest] : ID[t.dest, blks[n + 1].id]
        elseif t isa Core.ReturnNode
            return ID[]
        elseif t isa Switch
            return vcat(t.dests, t.fallthrough_dest)
        else
            error("Unhandled terminator $t")
        end
    end
    return Dict{ID,Vector{ID}}((b.id, succ) for (b, succ) in zip(blks, succs))
end

"""
    _compute_cfg_predecessors(blks::Vector{CFGBlock})::Dict{ID, Vector{ID}}

Map from each block's `ID` to its possible predecessor `ID`s (the inverse of
`_compute_cfg_successors`).
"""
function _compute_cfg_predecessors(blks::Vector{CFGBlock})::Dict{ID,Vector{ID}}
    succs = _compute_cfg_successors(blks)
    ks = collect(keys(succs))
    preds = Dict{ID,Vector{ID}}(zip(ks, map(_ -> ID[], ks)))
    for (k, ss) in succs, s in ss
        push!(preds[s], k)
    end
    return preds
end

"""
    control_flow_graph(blks::Vector{CFGBlock})::CC.CFG

The `CC.CFG` (position-numbered) associated to `blks` (`ID`-named). Block numbers are assigned by
`blks`'s vector order.
"""
function control_flow_graph(blks::Vector{CFGBlock})::CC.CFG
    preds_ids = _compute_cfg_predecessors(blks)
    succs_ids = _compute_cfg_successors(blks)

    block_ids = map(b -> b.id, blks)
    id_to_num = Dict{ID,Int}(zip(block_ids, eachindex(block_ids)))

    preds = map(id -> sort(map(p -> id_to_num[p], preds_ids[id])), block_ids)
    succs = map(id -> sort(map(s -> id_to_num[s], succs_ids[id])), block_ids)
    push!(preds[1], 0)   # predecessor of the entry block is "0" (function entry)

    index = vcat(0, cumsum(map(length, blks))) .+ 1
    bbs = map(eachindex(blks)) do n
        CC.BasicBlock(CC.StmtRange(index[n], index[n + 1] - 1), preds[n], succs[n])
    end
    return CC.CFG(bbs, index[2:(end - 1)])
end

# ---------------------------------------------------------------------------
# IRCode -> Vector{CFGBlock}
# ---------------------------------------------------------------------------

"""
    new_inst_vec(x::CC.InstructionStream)::InstVector

Convert an `InstructionStream` into a flat vector of `NewInstruction`s, reconstructing each one's
3-tuple source-location field from the stream's flat `line` array (3 entries per instruction).
"""
function new_inst_vec(x::CC.InstructionStream)::InstVector
    n = length(x.stmt)
    return [
        CC.NewInstruction(x.stmt[i], x.type[i], x.info[i],
                          (x.line[3i - 2], x.line[3i - 1], x.line[3i]), x.flag[i])
        for i in 1:n
    ]
end

const SSAToIdDict = Dict{Core.SSAValue,ID}
const BlockNumToIdDict = Dict{Int,ID}

"""
    _ssas_to_ids(insts::InstVector)::Tuple{Vector{ID}, InstVector}

Assign a fresh `ID` to each line in `insts` and replace every `SSAValue` operand with the `ID`
assigned to the line it references.
"""
function _ssas_to_ids(insts::InstVector)::Tuple{Vector{ID},InstVector}
    ids = map(_ -> ID(), insts)
    d = SSAToIdDict(zip(Core.SSAValue.(eachindex(insts)), ids))
    return ids, map(x -> _ssa_to_ids(d, x), insts)
end

_ssa_to_ids(d::SSAToIdDict, inst::CC.NewInstruction) = CC.NewInstruction(inst; stmt=_ssa_to_ids(d, inst.stmt))
_ssa_to_ids(d::SSAToIdDict, x::Core.ReturnNode) =
    isdefined(x, :val) ? Core.ReturnNode(get(d, x.val, x.val)) : x
_ssa_to_ids(d::SSAToIdDict, x::Expr) = Expr(x.head, map(a -> get(d, a, a), x.args)...)
_ssa_to_ids(d::SSAToIdDict, x::Core.PiNode) = Core.PiNode(get(d, x.val, x.val), get(d, x.typ, x.typ))
_ssa_to_ids(d::SSAToIdDict, x::QuoteNode) = x
_ssa_to_ids(d::SSAToIdDict, x) = x
function _ssa_to_ids(d::SSAToIdDict, x::Core.PhiNode)
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        isassigned(x.values, n) && (new_values[n] = get(d, x.values[n], x.values[n]))
    end
    return Core.PhiNode(x.edges, new_values)
end
_ssa_to_ids(d::SSAToIdDict, x::Core.GotoNode) = x
_ssa_to_ids(d::SSAToIdDict, x::Core.GotoIfNot) = Core.GotoIfNot(get(d, x.cond, x.cond), x.dest)

"""
    _block_nums_to_ids(insts::InstVector, cfg::CC.CFG)::Tuple{Vector{ID}, InstVector}

Assign a fresh `ID` to each basic block in `cfg`, and replace every block-number operand
(`PhiNode.edges`/`GotoNode.label`/`GotoIfNot.dest`) with the corresponding `ID`, converting
`Core.PhiNode`/`GotoNode`/`GotoIfNot` to their `ID`-addressed analogues.
"""
function _block_nums_to_ids(insts::InstVector, cfg::CC.CFG)::Tuple{Vector{ID},InstVector}
    ids = map(_ -> ID(), cfg.blocks)
    d = BlockNumToIdDict(zip(eachindex(cfg.blocks), ids))
    return ids, map(x -> _block_num_to_ids(d, x), insts)
end

_block_num_to_ids(d::BlockNumToIdDict, x::CC.NewInstruction) = CC.NewInstruction(x; stmt=_block_num_to_ids(d, x.stmt))
_block_num_to_ids(d::BlockNumToIdDict, x::Core.PhiNode) = IDPhiNode(ID[d[e] for e in x.edges], x.values)
_block_num_to_ids(d::BlockNumToIdDict, x::Core.GotoNode) = IDGotoNode(d[x.label])
_block_num_to_ids(d::BlockNumToIdDict, x::Core.GotoIfNot) = IDGotoIfNot(x.cond, d[x.dest])
_block_num_to_ids(d::BlockNumToIdDict, x) = x

"""
    _ircode_to_cfg_blocks(ir::CC.IRCode)::Vector{CFGBlock}

Convert `ir` into a fresh, independent `Vector{CFGBlock}` (mutating the result never mutates `ir`).
Every `PhiNode`/`GotoIfNot`/`GotoNode` becomes an `IDPhiNode`/`IDGotoIfNot`/`IDGotoNode`.
`lower_cfg_blocks_to_ir(_ircode_to_cfg_blocks(ir), ir)` should be the identity.
"""
function _ircode_to_cfg_blocks(ir::CC.IRCode)::Vector{CFGBlock}
    insts = new_inst_vec(ir.stmts)
    ssa_ids, insts = _ssas_to_ids(insts)
    block_ids, insts = _block_nums_to_ids(insts, ir.cfg)
    return map(zip(ir.cfg.blocks, block_ids)) do (bb, id)
        CFGBlock(id, ssa_ids[bb.stmts], insts[bb.stmts])
    end
end

# ---------------------------------------------------------------------------
# Vector{CFGBlock} -> IRCode
# ---------------------------------------------------------------------------

"""
    lower_cfg_blocks_to_ir(blks::Vector{CFGBlock}, ir::CC.IRCode; argtypes=ir.argtypes, def=ir.debuginfo.def)::CC.IRCode

Produce an `IRCode` equivalent to `blks`. Non-statement metadata (`sptypes`, debug info, `meta`,
valid worlds) is taken from `ir`; `argtypes` and the debug info's `def` (the owning `MethodInstance`)
may be overridden — the forwards- and pullback-pass carriers have different argument types and are
different `MethodInstance`s from the primal `ir` was taken from. Shares no memory with `blks`/`ir`.

Every `IDPhiNode`/`IDGotoIfNot`/`IDGotoNode` becomes a `Core.PhiNode`/`GotoIfNot`/`GotoNode`; every
`Switch` is lowered into a semantically equivalent `GotoIfNot` chain.
"""
function lower_cfg_blocks_to_ir(blks::Vector{CFGBlock}, ir::CC.IRCode;
                                argtypes=ir.argtypes, def=ir.debuginfo.def)
    blks = _cfg_lower_switch_statements(blks)
    blks = _cfg_remove_double_edges(blks)
    insts = _ids_to_line_numbers(blks)
    cfg = control_flow_graph(blks)
    insts = _lines_to_blocks(insts, cfg)

    lines = Int32[v for inst in insts for v in inst.line]
    debuginfo = copy(ir.debuginfo)
    debuginfo.def = def
    debuginfo.codelocs = lines
    stream = CC.InstructionStream(
        Any[x.stmt for x in insts], Any[x.type for x in insts],
        CC.CallInfo[x.info for x in insts], lines, UInt32[x.flag for x in insts],
    )
    return CC.IRCode(stream, cfg, debuginfo, Any[argtypes...], copy(ir.meta), copy(ir.sptypes))
end

"""
    _cfg_lower_switch_statements(blks::Vector{CFGBlock})::Vector{CFGBlock}

Replace every `Switch` terminator with a semantically equivalent chain of singleton `IDGotoIfNot`
blocks followed by a singleton fallthrough `IDGotoNode` block (see `Switch`'s docstring).
"""
function _cfg_lower_switch_statements(blks::Vector{CFGBlock})
    new_blocks = CFGBlock[]
    for block in blks
        t = terminator(block)
        if t isa Switch
            push!(new_blocks, CFGBlock(block.id, block.inst_ids[1:(end - 1)], block.insts[1:(end - 1)]))
            foreach(t.conds, t.dests) do cond, dest
                push!(new_blocks, CFGBlock(ID(), [ID()], [new_inst(IDGotoIfNot(cond, dest), Any)]))
            end
            push!(new_blocks, CFGBlock(ID(), [ID()], [new_inst(IDGotoNode(t.fallthrough_dest), Any)]))
        else
            push!(new_blocks, block)
        end
    end
    return new_blocks
end

"""
    _cfg_remove_double_edges(blks::Vector{CFGBlock})::Vector{CFGBlock}

If block `n`'s `IDGotoIfNot` targets block `n+1` (a redundant conditional that falls through to the
same place either way), replace it with an unconditional `IDGotoNode` — Julia's IR forbids two edges
between the same pair of blocks.
"""
function _cfg_remove_double_edges(blks::Vector{CFGBlock})
    return map(enumerate(blks)) do (n, blk)
        t = terminator(blk)
        if t isa IDGotoIfNot && t.dest == blks[n + 1].id
            term = CC.NewInstruction(blk.insts[end]; stmt=IDGotoNode(t.dest))
            CFGBlock(blk.id, blk.inst_ids, vcat(blk.insts[1:(end - 1)], term))
        else
            blk
        end
    end
end

"""
    _ids_to_line_numbers(blks::Vector{CFGBlock})::InstVector

For every statement in `blks`, replace each `ID` operand with the `SSAValue` (or plain `Int`
referencing one) it names.
"""
function _ids_to_line_numbers(blks::Vector{CFGBlock})::InstVector
    block_ids = [b.id for b in blks]
    block_lengths = map(length, blks)
    block_start_ssas = Core.SSAValue.(vcat(1, cumsum(block_lengths)[1:(end - 1)] .+ 1))
    line_ids = concatenate_ids(blks)
    line_ssas = Core.SSAValue.(eachindex(line_ids))
    d = Dict(zip(vcat(block_ids, line_ids), vcat(block_start_ssas, line_ssas)))
    return [_to_ssas(d, s) for s in concatenate_stmts(blks)]
end

_to_ssas(d::Dict, inst::CC.NewInstruction) = CC.NewInstruction(inst; stmt=_to_ssas(d, inst.stmt))
_to_ssas(d::Dict, x::Core.ReturnNode) = isdefined(x, :val) ? Core.ReturnNode(get(d, x.val, x.val)) : x
_to_ssas(d::Dict, x::Expr) = Expr(x.head, map(a -> get(d, a, a), x.args)...)
_to_ssas(d::Dict, x::Core.PiNode) = Core.PiNode(get(d, x.val, x.val), get(d, x.typ, x.typ))
_to_ssas(d::Dict, x::QuoteNode) = x
_to_ssas(d::Dict, x) = x
function _to_ssas(d::Dict, x::IDPhiNode)
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        isassigned(x.values, n) && (new_values[n] = get(d, x.values[n], x.values[n]))
    end
    return Core.PhiNode(map(e -> Int32(getindex(d, e).id), x.edges), new_values)
end
_to_ssas(d::Dict, x::IDGotoNode) = Core.GotoNode(d[x.label].id)
_to_ssas(d::Dict, x::IDGotoIfNot) = Core.GotoIfNot(get(d, x.cond, x.cond), d[x.dest].id)

"""
    _lines_to_blocks(insts::InstVector, cfg::CC.CFG)::InstVector

Convert line/`SSAValue`-numbered `GotoNode`/`GotoIfNot`/`PhiNode` edges into real basic-block
numbers, via `Core.Compiler.block_for_inst` — the same routine Julia's own lowering uses.
"""
function _lines_to_blocks(insts::InstVector, cfg::CC.CFG)::InstVector
    stmts = __line_numbers_to_block_numbers!(Any[x.stmt for x in insts], cfg)
    return map((inst, s) -> CC.NewInstruction(inst; stmt=s), insts, stmts)
end

"""
    __line_numbers_to_block_numbers!(insts::Vector{Any}, cfg::CC.CFG)

Converts any edges in `GotoNode`s, `GotoIfNot`s, and `PhiNode`s which refer to line numbers into
references to block numbers (copied from the body of `Core.Compiler.inflate_ir!`).
"""
function __line_numbers_to_block_numbers!(insts::Vector{Any}, cfg::CC.CFG)
    for i in eachindex(insts)
        s = insts[i]
        if isa(s, Core.GotoNode)
            insts[i] = Core.GotoNode(CC.block_for_inst(cfg, s.label))
        elseif isa(s, Core.GotoIfNot)
            insts[i] = Core.GotoIfNot(s.cond, CC.block_for_inst(cfg, s.dest))
        elseif isa(s, Core.PhiNode)
            insts[i] = Core.PhiNode(Int32[CC.block_for_inst(cfg, Int(e)) for e in s.edges], s.values)
        end
    end
    return insts
end

# ---------------------------------------------------------------------------
# Block-ordering / reachability utilities.
# ---------------------------------------------------------------------------

"""
    _distance_to_entry(blks::Vector{CFGBlock})::Vector{Int}

BFS distance from the entry block (`blks[1]`) to every block, `typemax(Int)` if unreachable.
"""
function _distance_to_entry(blks::Vector{CFGBlock})::Vector{Int}
    succs = _compute_cfg_successors(blks)
    id_to_int = Dict{ID,Int}(blk.id => n for (n, blk) in enumerate(blks))
    dists = fill(typemax(Int), length(blks))
    dists[1] = 0
    queue = Int[1]
    while !isempty(queue)
        u = popfirst!(queue)
        for s in succs[blks[u].id]
            v = id_to_int[s]
            if dists[v] == typemax(Int)
                dists[v] = dists[u] + 1
                push!(queue, v)
            end
        end
    end
    return dists
end

"""
    _sort_cfg_blocks!(blks::Vector{CFGBlock})::Vector{CFGBlock}

Reorder `blks` in place by distance-from-entry. WARNING: only valid when arbitrary block reordering
preserves meaning — in particular, not valid in the presence of `IDGotoIfNot`'s implicit
fallthrough-to-next-block semantics until after `_cfg_remove_double_edges`/`_cfg_lower_switch_statements`.
"""
function _sort_cfg_blocks!(blks::Vector{CFGBlock})::Vector{CFGBlock}
    blks .= blks[sortperm(_distance_to_entry(blks))]
    return blks
end

"""
    is_reachable_return_node(x)

`true` iff `x` is a `Core.ReturnNode` with a defined `val` (i.e. an actually-reachable return,
as opposed to the unreachable terminator a throw-only block ends in).
"""
is_reachable_return_node(x::Core.ReturnNode) = isdefined(x, :val)
is_reachable_return_node(x) = false

"""
    _characterise_unique_predecessor_blocks(blks::Vector{CFGBlock}) -> (Dict{ID,Bool}, Dict{ID,Bool})

Block `b` is a *unique predecessor* if it is the only predecessor of every one of its successors —
i.e. reaching any successor of `b` proves control passed through `b`. Returns
`(is_unique_pred, pred_is_unique_pred)`:
- `is_unique_pred[b]`: is `b` itself a unique predecessor?
- `pred_is_unique_pred[b]`: does `b` have exactly one predecessor, and is *that* predecessor a
  unique predecessor?

This drives the control-flow-replay optimization: if `pred_is_unique_pred[b]`, there is no need to
push to (forwards pass) or pop from (pullback pass) the block stack when entering `b`, since its
predecessor is already known unambiguously.
"""
function _characterise_unique_predecessor_blocks(blks::Vector{CFGBlock})::Tuple{Dict{ID,Bool},Dict{ID,Bool}}
    blk_ids = ID[b.id for b in blks]
    preds = _compute_cfg_predecessors(blks)
    succs = _compute_cfg_successors(blks)

    is_unique_pred = Dict{ID,Bool}()
    for id in blk_ids
        ss = succs[id]
        is_unique_pred[id] = !isempty(ss) && all(s -> length(preds[s]) == 1, ss)
    end

    # A block ending in the sole reachable return is a de facto unique predecessor: control leaves
    # the function only through it.
    reachable_returns = filter(blk -> is_reachable_return_node(terminator(blk)), blks)
    if length(reachable_returns) == 1
        is_unique_pred[only(reachable_returns).id] = true
    end

    pred_is_unique_pred = Dict{ID,Bool}()
    for id in blk_ids
        pred_is_unique_pred[id] = length(preds[id]) == 1 && is_unique_pred[only(preds[id])]
    end

    # The entry block, if it has no predecessors, can only be entered one way (function entry).
    entry_id = blk_ids[1]
    pred_is_unique_pred[entry_id] = isempty(preds[entry_id])

    return is_unique_pred, pred_is_unique_pred
end

"""
    _is_reachable(blks::Vector{CFGBlock})::Vector{Bool}

`true` at index `n` iff control flow can reach `blks[n]`.
"""
_is_reachable(blks::Vector{CFGBlock})::Vector{Bool} = _distance_to_entry(blks) .< typemax(Int)

"""
    _remove_unreachable_cfg_blocks!(blks::Vector{CFGBlock})::Vector{CFGBlock}

Drop every block control flow can never reach, then strip any surviving `IDPhiNode`'s edges/values
that referenced a removed block (mutates the surviving blocks' `IDPhiNode`s in place).
"""
function _remove_unreachable_cfg_blocks!(blks::Vector{CFGBlock})
    reachable = _is_reachable(blks)
    remaining = blks[reachable]
    removed_ids = map(idx -> blks[idx].id, findall(!, reachable))
    for blk in remaining, inst in blk.insts
        s = inst.stmt
        s isa IDPhiNode || continue
        for n in reverse(eachindex(s.edges))
            if s.edges[n] in removed_ids
                deleteat!(s.edges, n)
                deleteat!(s.values, n)
            end
        end
    end
    return remaining
end
