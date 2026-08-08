# Portions of this file are derived from Mooncake.jl (https://github.com/chalk-lab/Mooncake.jl),
# Copyright (c) 2024 Will Tebbutt and Hong Ge, licensed under the MIT License.

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

# ===========================================================================
# Tangent types for the reverse-mode runtime's own bookkeeping types (`Tape` itself is in
# `reverse_interp.jl`, next to its definition).
#
# Each shadow is self-similar: the tangent of a `Stack{T}` is a `Stack{tangent_type(T)}`, not a
# `MutableTangent` — so primal and shadow can be pushed/popped independently, in lockstep, using the
# very same `push!`/`pop!` methods (hand `frule!!`s in `rules_ad_runtime.jl`), and a fresh shadow tape
# can be built by the same `_fresh_tape_expr`/`_alloc_tape` code that builds a fresh primal one. Falling
# through to the generic per-field derivation would produce a `MutableTangent` here instead, which
# breaks that symmetry: its constructor doesn't take zero arguments, so `_alloc_tape`'s `$S()`/`$S(1)`
# construction wouldn't even type-check against it.
#
# Self-typed collapse when `T` itself has no tangent: the shadow is `Stack{T}` (same `T`, not
# `Stack{NoTangent}`). This is what makes `Tape`'s per-field invariant `tangent_type(fieldtype(Tape,f))
# === fieldtype(tangent_type(Tape),f)` hold for `block_stack::Stack{Int32}` (`reverse_interp.jl`):
# that field is hardcoded `Stack{Int32}` regardless of `Tape`'s type parameters, so a shadow `Tape`'s
# `block_stack` is always `Stack{Int32}`, never `Stack{NoTangent}` — the collapse matches that exactly,
# so every field of `Tape` mirrors the same way with no carve-out (see `_bi_selfsim_shadow_field`,
# `builtins.jl`, which encodes this invariant directly).
#
# `SingletonStack`/`CommsCell` get the identical treatment for consistency, and because it's the
# common case for `SingletonStack`: it's only chosen for a genuinely singleton comms type, and every
# singleton type has `NoTangent` tangent here, so its shadow is self-typed almost always.
# ===========================================================================

@foldable tangent_type(::Type{Stack{T}}) where {T} =
    tangent_type(T) === NoTangent ? Stack{T} : Stack{tangent_type(T)}
@foldable tangent_type(::Type{SingletonStack{T}}) where {T} =
    tangent_type(T) === NoTangent ? SingletonStack{T} : SingletonStack{tangent_type(T)}
@foldable tangent_type(::Type{CommsCell{T}}) where {T} =
    tangent_type(T) === NoTangent ? CommsCell{T} : CommsCell{tangent_type(T)}

# `zero_tangent`/`set_to_zero!!` for these three: never delegate to the generic per-field
# derivation (which assumes the `MutableTangent`/`Tangent` shell with a `.fields::NamedTuple` these
# self-typed shadows don't have — see above), build/mutate the wrapper directly instead.
#
# No `increment!!`: a `Stack`/`CommsCell` is append-only bookkeeping, not an additive quantity, and
# nothing in the engine asks to "sum" two of them — add a real method if a genuine call site turns up.
#
# Each constructs `tangent_type(...)` directly rather than re-deriving `Stack{tangent_type(T)}`/etc
# itself, so the self-typed-collapse rule above lives in exactly one place — duplicating it here
# would silently drift the moment that rule changes.

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

# Resetting a shadow `Stack`/`CommsCell` means putting it back in the state a fresh one starts in
# (position 0 / undef `val`), not reallocating — mirrors the raw `setfield!(stack, :position, 0)`
# reset `reverse_fwds_to_ircode` emits when a pre-allocated primal tape is reused across calls.
set_to_zero_internal!!(::SetToZeroCache, x::Stack) = (x.position = 0; x)
set_to_zero_internal!!(::SetToZeroCache, x::SingletonStack) = x
function set_to_zero_internal!!(c::SetToZeroCache, x::CommsCell)
    isdefined(x, :val) && (x.val = set_to_zero_internal!!(c, x.val))
    return x
end
