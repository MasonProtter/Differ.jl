# Portions of this file are derived from Mooncake.jl (https://github.com/chalk-lab/Mooncake.jl),
# Copyright (c) 2024 Will Tebbutt and Hong Ge, licensed under the MIT License.

# Block stack (which blocks were visited on the forwards pass, replayed in reverse on the pullback
# pass) and per-block "comms" stacks (forward-computed values the pullback needs, one push per
# block execution). Ported from Mooncake.jl's `src/stack.jl`.

"""
    Stack{T}()

A stack specialised for reverse-mode AD.

Semantically equivalent to a usual stack, but never de-allocates memory once allocated.
"""
mutable struct Stack{T}
    const memory::Vector{T}
    position::Int
    Stack{T}() where {T} = new{T}(Vector{T}(undef, 0), 0)
    # Pre-sized so the first `n` pushes stay on the in-bounds branch (used by `_fresh_tape_expr`).
    Stack{T}(n::Int) where {T} = new{T}(Vector{T}(undef, n), 0)
end

_copy(::Stack{T}) where {T} = Stack{T}()

# `Base.push!` is qualified (not bare `push!`) because this body gets inlined into synthetic
# carrier IR; a bare name would re-embed as `GlobalRef(Differ, :push!)`, an unbound GlobalRef
# `verify_ir` rejects. Same hazard as elsewhere in this file and in `reverse_interp.jl`.
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
# `Tape`'s `bufs` field, one slot per bulk-saved argument, assigned statically by the transform.
# `Any` element type since slots have unrelated types.
#
# `Base.copyto!`/`Base.similar` are qualified for the same GlobalRef-inlining reason as `Base.push!`
# above.
const _NO_BULK_BUFS = Any[]

@noinline function _bulk_save!(bufs::Vector{Any}, slot::Int, src::M) where {M}
    length(bufs) < slot && Base.resize!(bufs, slot)
    b = Base.isassigned(bufs, slot) ? bufs[slot] : nothing
    # Reallocate only when the buffer can't be reused, so a pre-allocated context calling
    # repeatedly with same-shaped arguments allocates here exactly once, ever.
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

# Stack for a singleton element type: nothing is actually stored, push!/pop! are no-ops that
# materialise `T.instance`. Used for a block whose comms tuple is empty/singleton.
struct SingletonStack{T} end

Base.push!(::SingletonStack, ::Any) = nothing
Base.pop!(::SingletonStack{T}) where {T} = T.instance

# Single-slot holder for a non-loop block whose comms tuple is `isbits`: no `position`, no
# `push!`/`pop!`, no boundscheck — the transform emits `setfield!`/`getfield` on `val` directly.
# Selected only when the block isn't in any loop (`_loop_blocks`) and `isbitstype(CommsT)`; a loop
# block or non-`isbits` tuple keeps `Stack{T}`, an empty tuple uses `SingletonStack`.
mutable struct CommsCell{T}
    val::T
    CommsCell{T}() where {T} = new{T}()
end

# Nested-tape recycling: a non-inlined callee's tape is a `(:subtape, SSAValue(i))` comms item
# pushed by `emit_epilogue!` (`reverse_interp.jl`). Since `Stack` never deallocates, after a block's
# first execution the next push's slot already holds a structurally identical inner tape from the
# previous call — handing the callee that recycled tape (via `Ctx{<:Tape}`) instead of a fresh
# `Ctx()` is what makes steady-state nested/recursive calls allocation-free.
#
# `Base.length`/`Base.isassigned` are qualified for the same GlobalRef-inlining reason as
# `Base.push!` above.
@inline function _inner_ctx(st::Stack{CommsT}, ::Val{k}, ::Type{TapeT}) where {CommsT,k,TapeT}
    p = st.position + 1
    mem = st.memory
    if p <= Base.length(mem) && Base.isassigned(mem, p)
        t = Core.getfield(@inbounds(mem[p]), k)
        # Direct self-recursion routes through `_inner_self_ctx` below instead, so this callee's
        # comms declaration always names `TapeT` concretely — the `isa` narrows to a no-op.
        t isa TapeT && return Ctx(t)
    end
    return Ctx(_alloc_tape(TapeT))
end

# Direct self-recursion's dedicated storage (`Tape.subtapes`). Mirrors `_inner_ctx` but simpler:
# every self-recursive call site in one primal shares this one field, so there's no per-site `k`.
@inline function _inner_self_ctx(st::Stack{TapeT}) where {TapeT}
    p = st.position + 1
    mem = st.memory
    if p <= Base.length(mem) && Base.isassigned(mem, p)
        return Ctx(@inbounds mem[p])
    end
    return Ctx(_alloc_tape(TapeT))
end

# Tangent types for the reverse-mode runtime's own bookkeeping types (`Tape` itself is in
# `reverse_interp.jl`, next to its definition).
#
# Each shadow is self-similar: `tangent_type(Stack{T}) == Stack{tangent_type(T)}`, not a
# `MutableTangent` — so primal and shadow push/pop in lockstep with the same methods, and a fresh
# shadow tape builds via the same `_alloc_tape` code as a fresh primal one (a `MutableTangent`
# wouldn't type-check there — its constructor doesn't take zero arguments).
#
# Self-typed collapse when `tangent_type(T) === NoTangent`: the shadow is `Stack{T}` itself, not
# `Stack{NoTangent}`. Needed so `Tape.block_stack::Stack{Int32}` — hardcoded regardless of `Tape`'s
# parameters — matches its own shadow's field type exactly (see `_bi_selfsim_shadow_field`,
# `builtins.jl`). `SingletonStack`/`CommsCell` get the same treatment for consistency.

@foldable tangent_type(::Type{Stack{T}}) where {T} =
    tangent_type(T) === NoTangent ? Stack{T} : Stack{tangent_type(T)}
@foldable tangent_type(::Type{SingletonStack{T}}) where {T} =
    tangent_type(T) === NoTangent ? SingletonStack{T} : SingletonStack{tangent_type(T)}
@foldable tangent_type(::Type{CommsCell{T}}) where {T} =
    tangent_type(T) === NoTangent ? CommsCell{T} : CommsCell{tangent_type(T)}

# `zero_tangent`/`set_to_zero!!` for these three build/mutate the wrapper directly rather than
# delegating to the generic per-field derivation, which assumes a `.fields::NamedTuple` shell these
# self-typed shadows don't have. No `increment!!`: a `Stack`/`CommsCell` is append-only bookkeeping,
# not an additive quantity.

function zero_tangent_internal(x::Stack{T}, d::MaybeCache) where {T}
    ST = tangent_type(Stack{T})
    haskey(d, x) && return d[x]::ST
    s = ST()
    d[x] = s
    return s
end
zero_tangent_internal(::SingletonStack{T}, ::MaybeCache) where {T} = tangent_type(SingletonStack{T})()
function zero_tangent_internal(x::CommsCell{T}, d::MaybeCache) where {T}
    CT = tangent_type(CommsCell{T})
    haskey(d, x) && return d[x]::CT
    c = CT()
    d[x] = c
    return c
end

# Resets a shadow `Stack`/`CommsCell` to its fresh-construction state (position 0 / undef `val`)
# without reallocating, mirroring the primal-tape reset `reverse_fwds_to_ircode` emits for reuse.
set_to_zero_internal!!(::SetToZeroCache, x::Stack) = (x.position = 0; x)
set_to_zero_internal!!(::SetToZeroCache, x::SingletonStack) = x
function set_to_zero_internal!!(c::SetToZeroCache, x::CommsCell)
    isdefined(x, :val) && (x.val = set_to_zero_internal!!(c, x.val))
    return x
end
