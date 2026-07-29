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
end

_copy(::Stack{T}) where {T} = Stack{T}()

# NOTE on the fully-qualified `Base.push!` in the grow branch (and likewise anywhere else in the
# reverse-mode runtime helpers): these functions are *inlined into synthetic IR* built by
# `reverse_interp.jl`, whose `:invoke`s are emitted against resolved `CodeInstance`s. A bare `push!`
# here resolves against this file's enclosing module and inlines as `GlobalRef(Differ, :push!)` — an
# implicit `using Base` binding, which `Core.Compiler.verify_ir` rejects as an "unbound or
# partitioned GlobalRef ... in value position" once re-embedded in the carrier's own compiled unit
# (see the same hazard documented in `reverse_interp.jl`, for hand rules). Naming `Base` explicitly
# makes it a genuine bound cross-module reference that survives inlining.
@inline function Base.push!(x::Stack{T}, val::T) where {T}
    position = x.position + 1
    memory = x.memory
    x.position = position
    if position <= length(memory)
        @inbounds memory[position] = val
    else
        @noinline Base.push!(memory, val)
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
# `Tape`'s `bufs` field. Indexed by slot — one slot per bulk-saved argument, assigned statically by
# the transform. `Any` element type because the slots have unrelated types; that costs one dynamic
# check per *call*, never per element.
#
# `Base.copyto!`/`Base.similar` fully qualified for the same reason as `Base.push!` above: these are
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

# A stack for a singleton element type: nothing is ever actually stored (there's only one possible
# value), so push!/pop! are no-ops that just materialise `T.instance`. Used for a block whose comms
# tuple type is empty/singleton (nothing to communicate) so the tape carries no real storage for it.
struct SingletonStack{T} end

Base.push!(::SingletonStack, ::Any) = nothing
Base.pop!(::SingletonStack{T}) where {T} = T.instance
