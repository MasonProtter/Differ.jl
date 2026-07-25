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

# A stack for a singleton element type: nothing is ever actually stored (there's only one possible
# value), so push!/pop! are no-ops that just materialise `T.instance`. Used for a block whose comms
# tuple type is empty/singleton (nothing to communicate) so the tape carries no real storage for it.
struct SingletonStack{T} end

Base.push!(::SingletonStack, ::Any) = nothing
Base.pop!(::SingletonStack{T}) where {T} = T.instance
