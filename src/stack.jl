# A stack specialised for reverse-mode AD control flow: the block stack (which basic blocks were
# visited on the forwards pass, replayed in reverse on the pullback pass) and the per-block "comms"
# stacks (forward-computed values the pullback needs back, one push per execution of that block).
# Ported verbatim from Mooncake.jl's `src/stack.jl` — this layer is generic and AD-framework-agnostic.

"""
    Stack{T}()

A stack specialised for reverse-mode AD.

Semantically equivalent to a usual stack, but never de-allocates memory once allocated.
"""
mutable struct Stack{T}
    const memory::Vector{T}
    position::Int
    Stack{T}() where {T} = new{T}(Vector{T}(undef, 0), 0)
    # Pre-allocated capacity: keeps the first `n` pushes on the in-bounds branch so LLVM can fold
    # the boundscheck away when the stack is pre-sized (used by `_fresh_tape_expr`).
    Stack{T}(n::Int) where {T} = new{T}(Vector{T}(undef, n), 0)
end

_copy(::Stack{T}) where {T} = Stack{T}()

# NOTE on the fully-qualified `Base.push!` here (and elsewhere in the reverse-mode runtime
# helpers): these functions get inlined into synthetic IR built by `reverse_interp.jl`, whose
# `:invoke`s are emitted against resolved `CodeInstance`s. A bare `push!` resolves against this
# file's module and inlines as `GlobalRef(Differ, :push!)` — an implicit `using Base` binding that
# `Core.Compiler.verify_ir` rejects as an unbound/partitioned GlobalRef once re-embedded in the
# carrier's compiled unit (same hazard documented in `reverse_interp.jl` for hand rules). Naming
# `Base` explicitly makes it a genuine bound cross-module reference that survives inlining. Keeping
# the grow call inlineable also lets `ssa_inlining_pass!` fold `Stack.push!` into the carrier, so
# LLVM can eliminate the grow path when the stack is pre-sized (`_fresh_tape_expr` allocates
# capacity 1 per slot).
@inline function Base.push!(x::Stack{T}, val::T) where {T}
    position = x.position + 1
    memory = x.memory
    x.position = position
    if position <= length(memory)
        @inbounds memory[position] = val
    else
        Base.push!(memory, val)
    end
    return nothing
end

@inline function Base.pop!(x::Stack)
    position = x.position
    val = @inbounds x.memory[position]
    x.position = position - 1
    return val
end

# Reusable buffers for bulk primal save/restore (`_bulk_save_args`, `reverse_interp.jl`), held in a
# `Tape`'s `bufs` field. Indexed by slot, one per bulk-saved argument, assigned statically by the
# transform. `Any` element type because slots have unrelated types — costs one dynamic check per
# call, never per element.
#
# `Base.copyto!`/`Base.similar` are fully qualified for the same reason as `Base.push!` above:
# inlined into synthetic carrier IR, where a bare name resolves as `GlobalRef(Differ, :copyto!)` and
# trips `verify_ir`'s unbound-GlobalRef check.
const _NO_BULK_BUFS = Any[]

@noinline function _bulk_save!(bufs::Vector{Any}, slot::Int, src::M) where {M}
    length(bufs) < slot && Base.resize!(bufs, slot)
    b = Base.isassigned(bufs, slot) ? bufs[slot] : nothing
    # Reallocate only when the buffer can't be reused — a pre-allocated context calling repeatedly
    # with same-shaped arguments allocates here exactly once, ever.
    if !isa(b, M) || Base.length(b::M) != Base.length(src)
        b = Base.similar(src)
        bufs[slot] = b
    end
    Base.copyto!(b::M, src)
    return nothing
end

@noinline function _bulk_restore!(bufs::Vector{Any}, slot::Int, dst::M) where {M}
    Base.copyto!(dst, bufs[slot]::M)
    return nothing
end

# Stack for a singleton element type: nothing is actually stored (there's only one possible value),
# so push!/pop! are no-ops that materialise `T.instance`. Used for a block whose comms tuple is
# empty/singleton, so the tape carries no real storage for it.
struct SingletonStack{T} end

Base.push!(::SingletonStack, ::Any) = nothing
Base.pop!(::SingletonStack{T}) where {T} = T.instance

# ===========================================================================
# `CommsCell{T}` — single-slot holder for a non-loop block whose comms tuple is `isbits`.
#
# Fully-unrolled static stack: no `position`, no `push!`/`pop!`, no boundscheck. The transform emits
# the read/write of `val` directly into the carrier IR (`setfield!` on push, `getfield` on pop). When
# `T` is `isbits` the cell is `isbits` — inline in the tape's comms tuple, no heap object, no write
# barrier.
#
# Selected only when the block is not in any loop (`_loop_blocks`) and `isbitstype(CommsT)`; a loop
# block or non-`isbits` tuple keeps `Stack{T}`, an empty tuple uses `SingletonStack`.
# ===========================================================================
mutable struct CommsCell{T}
    val::T
    CommsCell{T}() where {T} = new{T}()
end

# ===========================================================================
# Nested-tape recycling (`reverse_interp.jl`'s Stage 1/2): a non-inlined callee's own tape is stored
# as a `(:subtape, SSAValue(i))` comms item, pushed onto block `b`'s comms `Stack` by
# `emit_epilogue!`. A `Stack` never deallocates its backing memory (above), so after the block's
# first execution, the slot the next push lands in already holds a structurally identical inner tape
# from a previous call. Handing the callee that object (via its own pre-allocated `Ctx{<:Tape}`
# prologue, which resets stack positions instead of allocating them) instead of a fresh `Ctx()` is
# what makes a nested/recursive inner call allocation-free in steady state — see `_inner_ctx`'s
# caller in `reverse_interp.jl` for why the peek position (`st.position + 1`) is always the slot the
# matching push will use.
#
# `Base.length`/`Base.isassigned` are fully qualified for the same reason as `Base.push!` above:
# gets inlined into synthetic carrier IR, where a bare name re-embeds as `GlobalRef(Differ, …)` and
# trips `verify_ir`'s unbound-GlobalRef check.
@inline function _inner_ctx(st::Stack{CommsT}, ::Val{k}, ::Type{TapeT}) where {CommsT,k,TapeT}
    p = st.position + 1
    mem = st.memory
    if p <= Base.length(mem) && Base.isassigned(mem, p)
        t = Core.getfield(@inbounds(mem[p]), k)
        # This callee's comms declaration always names `TapeT` concretely (a direct self-recursive
        # edge, which used to force an abstract `Tape` marker here, routes through `_inner_self_ctx`
        # below instead) — so this `isa` narrows to a no-op.
        t isa TapeT && return Ctx(t)
    end
    return Ctx(_alloc_tape(TapeT))
end

# Direct self-recursion's dedicated storage (`Tape.subtapes`, `reverse_interp.jl`). Mirrors
# `_inner_ctx` above but simpler: the stack's element type already is the recycled `Tape`. Every
# self-recursive call site in one primal shares this single field, so there's no per-site `k` to
# multiplex, and no tuple wrapper or abstractly-typed slot to read out of — reading a value stored
# under a concrete `Tape{ArgsTT,CS}` element type costs nothing, unlike the abstract bare-`Tape`-
# marker comms item this replaces.
@inline function _inner_self_ctx(st::Stack{TapeT}) where {TapeT}
    p = st.position + 1
    mem = st.memory
    if p <= Base.length(mem) && Base.isassigned(mem, p)
        return Ctx(@inbounds mem[p])
    end
    return Ctx(_alloc_tape(TapeT))
end
