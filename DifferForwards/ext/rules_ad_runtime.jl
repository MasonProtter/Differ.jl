# Hand-written frule!! for the reverse-mode runtime's own primitives (`Stack`/`SingletonStack`
# push!/pop!, the tape-recycling/bulk-save helpers) — what makes forward-over-reverse
# (`D(x -> rev_gradient(f, x), v)`) possible without forward mode supporting growable-array mutation.
#
# Mooncake differentiates `push!`/`pop!` structurally (its `Stack.memory` is an ordinary `Vector`,
# grown by the same dualized `push!` running on both primal and shadow), which needs growable-array
# support in forward mode. Differ doesn't have that yet, so this file takes the cheaper route the
# self-similar shadows (`stack.jl`, `reverse_interp.jl`) set up for: since the shadow of a `Stack` is
# itself a `Stack`, `push!`/`pop!` on the shadow is just the same `push!`/`pop!` called again, on the
# shadow object instead of the primal one — no dualization of `Stack`'s own body needed. Once forward
# mode supports growable-array mutation, these rules become an optimization rather than a requirement.
#
# `Stack`, `SingletonStack`, `_bulk_save!`/`_bulk_restore!`'s buffers, and `_alloc_tape` are all marked
# `@inline`/`@noinline` for the ordinary reverse-mode carrier; forward mode's own inlining-policy
# override is what keeps a call to any of these surviving, un-inlined, in the IR this file's rules
# actually get to see. This file only needs the rules to exist so `has_hand_frule`
# (`forward_interp.jl`) has something to find.
#
# Activity: stacks, buffers and tapes are activity roots (`%new` of a mutable, `Core.memorynew`,
# `_alloc_tape`), so only the *recorded value* slots below can arrive with an `Inactive()` shadow —
# a constant argument being pushed onto the tape. These rules are plumbing, not derivative
# arithmetic: every push must pair with a pop and every save with a restore whatever the value's
# activity, so an inactive value gets a real zero here rather than being skipped. A structural slot
# that somehow did arrive inactive raises a `MethodError` rather than silently desynchronizing the
# stacks.
_rt_tangent(d::Dual) = isactive(tangent(d)) ? tangent(d) : zero_tangent(primal(d))

# ===========================================================================
# Stack push!/pop!
#
# `tangent_type(T) === NoTangent` (e.g. `Stack{Int32}`, `Tape.block_stack` — block ids, pure
# control-flow bookkeeping with no derivative content) collapses `tangent_type(Stack{T})` to the
# self-typed `Stack{T}` rather than `Stack{NoTangent}` (`stack.jl`) — a real object of the *same*
# type as the primal's, needed so it can sit in a field (`Tape.block_stack`) whose declared type
# never varies with the enclosing `Tape`'s own type parameters. But nothing pushed onto it would
# ever be read back (nothing downstream reads a shadow block-stack), so there's nothing worth
# pushing: skip the shadow push/pop entirely rather than push a value of the wrong element type
# (`tangent(vd)` is `NoTangent()`, not a `T`) onto a `Stack{T}`. `T` is resolved at compile time
# (`tangent_type` is `@foldable`), so this branch is free — it specializes away entirely.
#
# Skipping is safe *because it's symmetric*: every pop against this shadow is skipped too (below),
# so primal and shadow positions never need to agree, and nothing ever observes the shadow stack's
# contents. This is the same shape as `SingletonStack`'s existing no-op push!/pop! just below.
# ===========================================================================

function frule!!(::Dual{typeof(push!)}, sd::Dual{Stack{T}}, vd::Dual{T}) where {T}
    push!(primal(sd), primal(vd))
    tangent_type(T) !== NoTangent && push!(tangent(sd), _rt_tangent(vd))
    return Dual(nothing, NoTangent())
end

function frule!!(::Dual{typeof(pop!)}, sd::Dual{Stack{T}}) where {T}
    p = pop!(primal(sd))
    t = tangent_type(T) === NoTangent ? NoTangent() : pop!(tangent(sd))
    return Dual(p, t)
end

# ===========================================================================
# SingletonStack push!/pop! — both primal and shadow are no-ops/singleton reads, but the rule still
# has to exist: `push!`/`pop!` on a `SingletonStack` are separate methods from `Stack`'s (dispatched
# on the primal argument type), so `Stack`'s rules above don't cover this shape.
# ===========================================================================

function frule!!(::Dual{typeof(push!)}, sd::Dual{SingletonStack{T}}, vd::Dual) where {T}
    push!(primal(sd), primal(vd))
    push!(tangent(sd), _rt_tangent(vd))
    return Dual(nothing, NoTangent())
end

function frule!!(::Dual{typeof(pop!)}, sd::Dual{SingletonStack{T}}) where {T}
    return Dual(pop!(primal(sd)), pop!(tangent(sd)))
end

# ===========================================================================
# __pop_blk_stack! (`reverse_interp.jl`) — `Base.pop!(block_stack)::Int32` plus a type assert. It's
# `@inline`, so under *ordinary* (order-1) compilation of the pullback carrier it normally inlines
# away into a plain `pop!` call before this rule would ever matter — the `Stack` rule above already
# covers that revealed call. This rule is a direct safety net for the case it doesn't (e.g. some
# future change makes it `@noinline`), so `has_hand_frule` has something to find regardless.
#
# Always the `Stack{Int32}` push!/pop! rule's own skip case (`tangent_type(Int32) === NoTangent`
# unconditionally) — never touch `tangent(sd)` at all, both because there's nothing to pop (the
# matching push was skipped, symmetric with the `Stack` rule above) and because `tangent(sd)` is
# itself a `Stack{Int32}` (the self-typed collapse, `stack.jl`) with nothing pushed onto it, so
# popping it would underflow. The result's own tangent is simply `tangent_type(Int32) === NoTangent`.
# ===========================================================================

function frule!!(::Dual{typeof(__pop_blk_stack!)}, sd::Dual{Stack{Int32}})
    return Dual(__pop_blk_stack!(primal(sd)), NoTangent())
end

# ===========================================================================
# _bulk_save!/_bulk_restore! (`stack.jl`) — bulk primal save/restore of a loop-mutated argument's
# contents, keyed by a static `slot`. The shadow needs the exact same treatment for the *shadow*
# array, into the *shadow* tape's own `bufs::Vector{Any}` (self-typed — `tangent_type(Vector{Any}) ===
# Vector{Any}`, since `Any` is its own tangent type — so the shadow's `bufs` field is a plain
# `Vector{Any}` too, and the very same `_bulk_save!`/`_bulk_restore!` apply to it unchanged).
# `slot` is a static index, not a differentiable value (`tangent_type(Int) === NoTangent`), so only
# its primal is used on both calls.
# ===========================================================================

function frule!!(
    ::Dual{typeof(_bulk_save!)}, bufsd::Dual{Vector{Any}}, slotd::Dual{Int}, srcd::Dual{M},
) where {M}
    _bulk_save!(primal(bufsd), primal(slotd), primal(srcd))
    _bulk_save!(tangent(bufsd), primal(slotd), _rt_tangent(srcd))
    return Dual(nothing, NoTangent())
end

function frule!!(
    ::Dual{typeof(_bulk_restore!)}, bufsd::Dual{Vector{Any}}, slotd::Dual{Int}, dstd::Dual{M},
) where {M}
    _bulk_restore!(primal(bufsd), primal(slotd), primal(dstd))
    _bulk_restore!(tangent(bufsd), primal(slotd), _rt_tangent(dstd))
    return Dual(nothing, NoTangent())
end

# ===========================================================================
# _inner_ctx/_inner_self_ctx (`stack.jl`) — nested/self-recursive-call tape recycling. Both are
# already generic over the tape type they recycle, so the shadow call is literally the *same*
# function called again: `_inner_ctx`/`_inner_self_ctx` on the shadow stack, with the shadow tape
# type (`tangent_type(TapeT)`) in place of the primal one. Correct because primal and shadow stacks
# agree on *position* at every call (the `Stack` rule above keeps pushes/pops in lockstep) — not on
# *occupancy*: a pre-allocated primal tape can recycle an already-populated slot while a per-call
# `zero_tangent` shadow tape allocates fresh, so `_inner_ctx` routinely takes different
# recycle-vs-allocate-fresh branches on the two sides. Sound anyway, because the pairing this rule
# relies on is established freshly within the call, not carried over from occupancy agreeing across
# calls.
#
# `Ctx{P}` gets its tangent from the ordinary generic struct derivation (a 1-field immutable wrapper,
# not one of the self-typed shadows) — `Tangent{@NamedTuple{tape::tangent_type(P)}}` — so the result
# is built with `build_tangent`, not a bespoke `Ctx` constructor call.
# ===========================================================================

function frule!!(
    ::Dual{typeof(_inner_ctx)}, sd::Dual{Stack{CommsT}}, ::Dual{Val{k}}, ::Dual{Type{TapeT}},
) where {CommsT,k,TapeT}
    pctx = _inner_ctx(primal(sd), Val(k), TapeT)
    sctx = _inner_ctx(tangent(sd), Val(k), tangent_type(TapeT))
    return Dual(pctx, build_tangent(typeof(pctx), sctx.tape))
end

function frule!!(::Dual{typeof(_inner_self_ctx)}, sd::Dual{Stack{TapeT}}) where {TapeT}
    pctx = _inner_self_ctx(primal(sd))
    sctx = _inner_self_ctx(tangent(sd))
    return Dual(pctx, build_tangent(typeof(pctx), sctx.tape))
end

# ===========================================================================
# _alloc_tape (`reverse_interp.jl`) — allocates a fresh tape of exactly the shape its `Type`
# argument names, from the type alone (no primal state read). The shadow is the same allocation
# applied to the shadow tape type — exactly what `zero_tangent_internal(x::Tape, d)` does too.
# ===========================================================================

function frule!!(::Dual{typeof(_alloc_tape)}, ::Dual{Type{TapeT}}) where {TapeT<:Tape}
    return Dual(_alloc_tape(TapeT), _alloc_tape(tangent_type(TapeT)))
end
