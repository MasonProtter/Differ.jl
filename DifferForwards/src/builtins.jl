# ===========================================================================
# Builtin rules — dispatch-based, direct-IR-emission handling of `Core.Builtin`s
# (`getfield`, `setfield!`, `Core.tuple`, `Core.ifelse`, the `memorynew`/`memoryref*` array-allocation
# builtins, `===`).
#
# Mirrors `intrinsics.jl`: `Core.Builtin` is a single type too (dispatch on `typeof(f)` can't tell
# `getfield` from `setfield!` apart), so each specific builtin is named via `Val{f}` and picked out by
# ordinary multiple dispatch. `apply_builtin_frule!(Val(f), actual, Ti, ctx)` is called from the main
# statement loop in `dualize_to_ircode` (`forward_interp.jl`) for every `Core.Builtin` call in the
# primal IR, and emits the primal + shadow IR directly into the caller's instruction stream, returning
# `(primal_ssa, shadow_ssa)` (or `nothing` if unregistered).
#
# `ctx` is a `NamedTuple` of the closures `dualize_to_ircode` builds once per call:
#   * `ctx.emit!(ex, ty)`          — emit a typed IR statement
#   * `ctx.presolve(x)`/`ctx.tresolve(x)` — resolve an operand AST node to its primal/shadow SSA
#   * `ctx.optype(x)`              — the primal IR's own declared type of operand `x`
#   * `ctx.tt(T)`                  — tangent type of primal type `T`
#   * `ctx.zero_shadow(Ti, primal_ssa)` — the zero tangent of a computed non-differentiable result
#
# The fallback below returns `nothing`, so a builtin with no registered rule (e.g.
# `Core.memoryrefoffset`, used by `push!`/`resize!`) bails in `dualize_to_ircode` with a clear,
# located reason instead of silently miscompiling.
# ===========================================================================

apply_builtin_frule!(::Val{F}, actual, Ti, ctx) where {F} = nothing

# `_bi_literal_index`/`_bi_homog_tangent_type`/`_tangent_field_slot`/`_widen` (compile-time-index
# detection, homogeneous-field-type check, direct-emission field-slot resolution, lattice-element
# widening) now live in `DifferCore/src/shared_ir_helpers.jl`, shared with reverse mode's
# `builtins_reverse.jl`.

# 1-based field index for a compile-time literal `getfield` name/index operand, or `nothing` when the
# operand is dynamic or names no field of `Pobj`.
function _bi_field_index(@nospecialize(Pobj), @nospecialize(name))
    _bi_literal_index(name) || return nothing
    sym = name isa QuoteNode ? name.value : name
    i = sym isa Int ? sym : findfirst(==(sym), fieldnames(Pobj))
    return (i isa Int && 1 <= i <= fieldcount(Pobj)) ? i : nothing
end

# `_foreign_selfsim_shadow_field` (defined in `forward_interp.jl`) is the general form of this: for
# a `Stack`/`CommsCell`/`Tape` field index `fi`, the shadow's own field type there when it's sound to
# mirror `getfield`/`setfield!` directly, or `nothing` when the field carries no tangent.
# `Tape.block_stack` mirrors despite looking like it shouldn't: it's always `Stack{Int32}`
# regardless of `Tape`'s type parameters, so shadow and primal have the same shape. The real
# non-mirroring cases are `Stack.position` (handled via `_foreign_selfsim_mirror_field`),
# `CommsCell.val` with no tangent, and a degenerate all-`NoTangent` tuple. Independent of `Ti`: this
# keeps `getfield` and `setfield!` agreeing on the same field.

# `MemoryRef.ptr_or_offset` / `Memory.ptr` hold a real address only when the buffer's element layout
# is inline and non-empty. Otherwise the field is an offset — for a bits-union element (`arrayelem == 2`)
# or a zero-size one (`layoutsize == 0`), where a `MemoryRef` stores its 0-based index there. Base's
# own `unsafe_convert(::Type{Ptr{Cvoid}}, ::GenericMemoryRef)` branches on exactly this pair of
# conditions.
#
# The catch: a shadow buffer's element type is `tangent_type(P)`, which can sit in a different regime
# than `P`. Classify the mirrored read rather than allowing/refusing it outright:
#
#   `:null`    — the shadow's elements are `NoTangent` (`Vector{Int}`, `Vector{Bool}`): zero-size, so
#                there is no tangent storage to address at all. The caller hands back
#                `NULL_SHADOW_PTR` (`src/intrinsics.jl`) — not a mirrored read, which would yield the
#                shadow ref's index dressed up as an address — and every downstream pointer rule
#                either skips the shadow operation (nothing to transfer) or declines.
#   `:address` — both buffers store their elements inline, so mirroring the read gives a genuine
#                address into tangent storage.
#   `nothing`  — declined, `ctx.reason` set. The case that matters is a bits-union shadow, whose
#                offset is scaled by an element size the primal doesn't share.
function _bi_mem_ptr_field_regime(@nospecialize(Pobj), @nospecialize(Tobj), ctx)
    eltype(Tobj) === NoTangent && return :null
    for (obj, side) in ((Pobj, "primal"), (Tobj, "shadow"))
        M = obj <: Memory ? obj : fieldtype(obj, :mem)
        if !(M isa DataType && Base.datatype_arrayelem(M) == 0 && Base.datatype_layoutsize(M) != 0)
            ctx.reason[] = "reading the data pointer of `$Pobj`: the $side buffer `$M` does not store " *
                           "its elements inline (a bits-union or zero-size element type), so that " *
                           "field is an offset rather than an address"
            return nothing
        end
    end
    return :address
end

# `_getfieldg`/`_setfieldg`/`_ctupleg` now come from `DifferCore` (shared with reverse mode's
# `builtins_reverse.jl`); the rest are forward-mode-only.
const _memnewg    = GlobalRef(Core, :memorynew)
const _memrefnewg = GlobalRef(Core, :memoryrefnew)
const _memrefgetg = GlobalRef(Core, :memoryrefget)
const _memrefsetg = GlobalRef(Core, :memoryrefset!)
const _eqeqg      = GlobalRef(Core, :(===))
const _isdefinedg = GlobalRef(Core, :isdefined)
const _ifelseg    = GlobalRef(Core, :ifelse)

# getfield(obj, name[, ordering][, boundscheck]) — trailing operands beyond (obj, name) are forwarded
# verbatim (faithful primal reconstruction; harmless on the same-shape shadow branches too, since no
# bounds are touched by `getfield` itself). The general-struct branch never receives them: a
# `Tangent`/`MutableTangent` field read has no atomicity/boundscheck concept.
#
# `actual[2]` (the field name/index) is usually a literal `Symbol`/`Int`/`QuoteNode`, for which
# `presolve` is a no-op — but a dynamic index (`t[i]` inside a loop, lowered to `getfield(t, i)` with
# `i` a genuine `SSAValue`/`Argument`) must be resolved to this pass's own numbering like any other
# operand, not embedded as a dangling reference into the primal's numbering (embedding it raw crashed
# with a `TypeError`, since the two numberings diverge once shadow instructions are interleaved).
function apply_builtin_frule!(::Val{Core.getfield}, actual, Ti, ctx)
    # `_widen`: `ctx.optype` can return a `PartialStruct`/`Const` lattice element (const-prop
    # narrowing), not a bare `Type`. Every `<:`/`isa DataType` check below needs a real `Type`;
    # skipping this widening `TypeError`s at the first `<:` (gotcha #9 in the
    # `differ-forward-dualization` skill).
    Pobj = _widen(ctx.optype(actual[1]))
    idx = ctx.presolve(actual[2])
    TT = ctx.tt(Ti)
    if !_bi_literal_index(actual[2]) && TT !== NoTangent
        # Dynamic index into a differentiable field: only safe when every field of the object shares
        # one tangent type (a homogeneous Tuple/NamedTuple/Array/Dual, or a homogeneous mutable struct
        # — the common tuple-iteration pattern plus its mutable-struct analogue), so the runtime index
        # always selects a validly-typed shadow value regardless of which field it lands on. A
        # heterogeneous struct has no such guarantee (different fields could need different tangent
        # types); bail rather than guess.
        if !(Pobj <: Dual || Pobj <: Tuple || Pobj <: NamedTuple || Pobj <: Array ||
             (Pobj isa DataType && ismutabletype(Pobj))) ||
           _bi_homog_tangent_type(ctx.tt, Pobj) !== TT
            return nothing
        end
    end
    # Foreign self-similar shadows owned by a different AD-mode package (forward-over-reverse:
    # DifferReverse's `Stack`/`CommsCell`/`Tape`) — needs its own per-field check via the
    # `_foreign_selfsim_shadow_field` hook, independent of `Ti`/`TT`.
    if Pobj isa DataType && ctx.fsel_shadow_type(Pobj)
        fi = _bi_field_index(Pobj, actual[2])
        if fi === nothing
            ctx.reason[] = "dynamic field index into `$Pobj` — its fields do not share one tangent type"
            return nothing
        end
        p = ctx.emit!(Expr(:call, _getfieldg, ctx.presolve(actual[1]), idx,
                           (ctx.presolve(a) for a in actual[3:end])...), Ti)
        Fsh = ctx.fsel_shadow_field(Pobj, fi)
        Fsh === nothing && return p, NoTangent()
        t = ctx.emit!(Expr(:call, _getfieldg, ctx.tresolve(actual[1]), idx,
                           (ctx.presolve(a) for a in actual[3:end])...), Fsh)
        return p, t
    end
    # `MemoryRef`/`Memory` are same-shape too (`tangent_type(MemoryRef{P}) === MemoryRef{tangent_type(P)}`),
    # but their shadow field types don't all match the primal's, so they need their own branch —
    # settled here, before anything is emitted. Deliberately not `GenericMemoryRef`/`GenericMemory`:
    # only the `:not_atomic`/`Core.CPU` aliases have same-shape `tangent_type` methods
    # (`src/tangents.jl`), and an `AtomicMemoryRef`'s tangent is an ordinary `Tangent`, which a
    # mirrored `getfield` would not find.
    memfield = nothing
    nullshadow = false
    if TT !== NoTangent && Pobj isa DataType && (Pobj <: MemoryRef || Pobj <: Memory)
        Tobj = ctx.tt(Pobj)
        fi = _bi_field_index(Pobj, actual[2])
        if fi === nothing
            ctx.reason[] = "dynamic field index into `$Pobj` — its fields do not share one tangent type"
            return nothing
        end
        # The *shadow object's* own field type, which is what the mirrored read actually produces.
        Fsh = fieldtype(Tobj, fi)
        if Fsh <: Ptr
            regime = _bi_mem_ptr_field_regime(Pobj, Tobj, ctx)
            regime === nothing && return nothing
            # A `Ptr` field is the one place the shadow's field type disagrees with the shadow
            # statement's required type: `:ptr_or_offset` is a `Ptr{Nothing}` on both sides, while the
            # statement must be declared `tangent_type(Ptr{Nothing}) === Ptr{NoTangent}` to keep the
            # "shadow is typed `tangent_type(primal)`" invariant — a value that violates it can reach a
            # `%new(Dual{P,tangent_type(P)}, …)`, which type-checks its fields at run time. Reconcile
            # honestly with a no-op `bitcast` (what `Base.convert(::Type{Ptr{T}}, ::Ptr)` compiles to)
            # instead of mis-declaring the read.
            if !(TT isa DataType && TT <: Ptr)
                ctx.reason[] = "reading `$Pobj`'s `Ptr` field, whose tangent type `$TT` is not a `Ptr`"
                return nothing
            end
            if regime === :null
                # The sentinel is typed `Ptr{NoTangent}`; anything else would violate that same
                # invariant, so decline rather than mis-declare it.
                if TT !== typeof(NULL_SHADOW_PTR)
                    ctx.reason[] = "reading the data pointer of `$Pobj`, whose shadow has no tangent " *
                                   "storage, but the required tangent type is `$TT` rather than " *
                                   "`$(typeof(NULL_SHADOW_PTR))`"
                    return nothing
                end
                nullshadow = true
            end
        elseif Fsh !== TT
            ctx.reason[] = "reading field $fi of `$Pobj`: the shadow's field type `$Fsh` is neither " *
                           "the required tangent type `$TT` nor a `Ptr` that can be reinterpreted"
            return nothing
        end
        nullshadow || (memfield = Fsh)     # memfield !== nothing => mirror the read
    end
    p = ctx.emit!(Expr(:call, _getfieldg, ctx.presolve(actual[1]), idx,
                       (ctx.presolve(a) for a in actual[3:end])...), Ti)
    t = if TT === NoTangent
        NoTangent()
    elseif nullshadow
        # No tangent storage behind the primal's address (`:null` regime above) — synthesize the null
        # sentinel instead of mirroring the read.
        NULL_SHADOW_PTR
    elseif memfield !== nothing
        m = ctx.emit!(Expr(:call, _getfieldg, ctx.tresolve(actual[1]), idx,
                           (ctx.presolve(a) for a in actual[3:end])...), memfield)
        memfield === TT ? m : ctx.opf(:bitcast, TT, TT, m)
    elseif Pobj <: Dual || Pobj <: Tuple || Pobj <: NamedTuple || Pobj <: Array
        # Same-shape tangent (Dual/Tuple/NamedTuple/Array): index/name the shadow aggregate directly.
        ctx.emit!(Expr(:call, _getfieldg, ctx.tresolve(actual[1]), idx,
                       (ctx.presolve(a) for a in actual[3:end])...), TT)
    else
        # General struct (including a homogeneous mutable struct admitted by the dynamic-index gate
        # above): read the field's tangent out of the Tangent/MutableTangent's `fields` NamedTuple.
        # With a literal field and an always-initialized slot, emit it as two builtin `getfield`s
        # (`getfield(shadow, :fields)` then `getfield(_, i)`) — no `get_tangent_field` call to
        # dynamic-dispatch, and SROA removes the intermediate. Otherwise (a dynamic homogeneous-mutable
        # index, or a `PossiblyUninitTangent` slot needing a `val`-unwrap) fall back to the generic
        # helper, type-stable over both cases exactly when the gate above allowed us here.
        slot = _bi_literal_index(actual[2]) ? _tangent_field_slot(ctx.tt, Pobj, actual[2]) : nothing
        if slot === nothing
            # Fallback (dynamic homogeneous-mutable index, or a PossiblyUninitTangent slot): emit the
            # generic helper as a static `:invoke` so it runs compiled rather than dynamic-dispatched.
            ctx.emit_invoke!(get_tangent_field, TT, (ctx.tt(Pobj), ctx.optype(actual[2])),
                             ctx.tresolve(actual[1]), idx)
        else
            NT, fi = slot
            fnt = ctx.emit!(Expr(:call, _getfieldg, ctx.tresolve(actual[1]), QuoteNode(:fields)), NT)
            ctx.emit!(Expr(:call, _getfieldg, fnt, fi), TT)
        end
    end
    p, t
end

# setfield!(obj, name, value[, ordering]) mutates obj in place and returns value. Only legal on a
# genuinely mutable primal, whose shadow is therefore always a MutableTangent. A 4th (atomic-ordering)
# arg is forwarded to the primal call only; `set_tangent_field!` has no atomics concept.
#
# `actual[2]` gets the same always-resolve treatment as `getfield` above. Unlike `getfield`, a dynamic
# write index gets no same-shape support (Phase B): `set_tangent_field!` needs a statically-known
# field to place the new value into the right `NamedTuple` slot type, and a same-shape aggregate is
# never itself mutable, so there's no tractable common case to support. Bail unless the object is
# entirely non-differentiable (every field's tangent type is `NoTangent`), in which case no field a
# dynamic index could hit carries a tangent anyway.
function apply_builtin_frule!(::Val{Core.setfield!}, actual, Ti, ctx)
    idx = ctx.presolve(actual[2])
    # `_widen` — same reason as `getfield`'s arm above. Every direct use of `Pobj` below already sits
    # behind an `isa DataType` guard, but widen at the source rather than rely on future edits
    # preserving the guard ordering.
    Pobj = _widen(ctx.optype(actual[1]))
    if !_bi_literal_index(actual[2]) && !(Pobj isa DataType && _bi_homog_tangent_type(ctx.tt, Pobj) === NoTangent)
        return nothing
    end
    p = ctx.emit!(Expr(:call, _setfieldg, ctx.presolve(actual[1]), idx, ctx.presolve(actual[3]),
                       (ctx.presolve(a) for a in actual[4:end])...), Ti)

    # Foreign self-similar shadows owned by a different AD-mode package (see `getfield`'s branch
    # above). `Ti` is useless here too: DifferReverse's carrier's own `setfield!`s on these
    # declare statement type `Any` regardless of field (hand-built IR, not inferred), so decide
    # purely from `Pobj`'s own field type.
    if Pobj isa DataType && ctx.fsel_shadow_type(Pobj)
        fi = _bi_field_index(Pobj, actual[2])
        if fi === nothing
            ctx.reason[] = "dynamic field index into `$Pobj` — its fields do not share one tangent type"
            return nothing
        end
        Fsh = ctx.fsel_shadow_field(Pobj, fi)
        if Fsh === nothing
            # A field with no tangent (e.g. `Stack.position`, an `Int`) can still be lockstep
            # bookkeeping that must stay identical between primal and shadow tape — not
            # differentiable content, just an index. `_foreign_selfsim_mirror_field` names such a
            # field; when it does, mirror the write with the **primal** value so a recycled
            # shadow (forward-over-reverse's `Ctx{<:Tape}` case) doesn't retain a stale value from
            # a previous call.
            if ctx.fsel_mirror_field(Pobj, fi)
                ctx.emit!(Expr(:call, _setfieldg, ctx.tresolve(actual[1]), idx, ctx.presolve(actual[3])),
                          fieldtype(Pobj, fi))
            end
            return p, NoTangent()
        end
        newtan = ctx.tresolve(actual[3])
        ctx.emit!(Expr(:call, _setfieldg, ctx.tresolve(actual[1]), idx, newtan), Fsh)
        return p, newtan
    end

    TT = ctx.tt(Ti)
    TT === NoTangent && return p, NoTangent()
    newtan = ctx.tresolve(actual[3])
    slot = _bi_literal_index(actual[2]) ? _tangent_field_slot(ctx.tt, Pobj, actual[2]) : nothing
    if slot === nothing
        # Fallback (a PossiblyUninitTangent target slot): emit the generic helper as a static
        # `:invoke` so it runs compiled rather than dynamic-dispatched.
        t = ctx.emit_invoke!(set_tangent_field!, TT, (ctx.tt(Pobj), ctx.optype(actual[2]), TT),
                             ctx.tresolve(actual[1]), idx, newtan)
        return p, t
    end
    # Direct emission — what `set_tangent_field!` compiles to, minus the dynamic-dispatched call: read
    # the shadow MutableTangent's `fields` NamedTuple, rebuild it with slot `fi` replaced by the new
    # tangent (other slots read back verbatim), and `setfield!` it back. The per-write NamedTuple
    # allocation stays (a MutableTangent is mutable identity), but boxing the call's result is gone.
    NT, fi = slot
    shadow = ctx.tresolve(actual[1])
    old = ctx.emit!(Expr(:call, _getfieldg, shadow, QuoteNode(:fields)), NT)
    nf = fieldcount(NT)
    slots = Any[j == fi ? newtan : ctx.emit!(Expr(:call, _getfieldg, old, j), fieldtype(NT, j))
                for j in 1:nf]
    nt = ctx.emit!(Expr(:new, NT, slots...), NT)
    ctx.emit!(Expr(:call, _setfieldg, shadow, QuoteNode(:fields), nt), NT)
    # `set_tangent_field!` returns the assigned value `x`, not the NamedTuple — mirror that.
    return p, newtan
end

# Array allocation, step 1: `Core.memorynew(Memory{P}, n)` allocates a fresh, uninitialized
# `Memory{P}`. Mirrors Mooncake's reference rule: allocate a same-length shadow `Memory{tangent_type(P)}`,
# also uninitialized (safe — every element the primal ever reads was necessarily written first). The
# length is structural, so it's `presolve`d, never `tresolve`d, in both primal and shadow calls.
function apply_builtin_frule!(::Val{Core.memorynew}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, _memnewg, ctx.presolve(actual[1]),
                       (ctx.presolve(a) for a in actual[2:end])...), Ti)
    TT = ctx.tt(Ti)
    t = ctx.emit!(Expr(:call, _memnewg, TT, (ctx.presolve(a) for a in actual[2:end])...), TT)
    p, t
end

# tangent_type(MemoryRef{P}) === MemoryRef{tangent_type(P)}, so mirroring the operation on the shadow
# ref yields the correctly-typed shadow handle directly.
#
# SAFETY: the trailing boundscheck flag (3-arg ref-offsetting form) is NOT mirrored from the primal —
# always forced `true` on the shadow ref, so a mismatched-length tangent raises a catchable
# `BoundsError` instead of corrupting memory via an unchecked out-of-bounds `MemoryRef`. `Dual`'s
# constructor only checks `tangent_type(P) == T`, never that a user tangent array matches its primal's
# length, so this check is the only thing catching that mismatch.
function apply_builtin_frule!(::Val{Core.memoryrefnew}, actual, Ti, ctx)
    nargs = length(actual)
    p = ctx.emit!(Expr(:call, _memrefnewg, ctx.presolve(actual[1]),
                       (ctx.presolve(a) for a in actual[2:end])...), Ti)
    TT = ctx.tt(Ti)
    t = if nargs >= 3
        ctx.emit!(Expr(:call, _memrefnewg, ctx.tresolve(actual[1]),
                       (ctx.presolve(a) for a in actual[2:end-1])..., true), TT)
    else
        ctx.emit!(Expr(:call, _memrefnewg, ctx.tresolve(actual[1]),
                       (ctx.presolve(a) for a in actual[2:end])...), TT)
    end
    p, t
end

# Reads one array element; the shadow array is a real same-shape `Array{tangent_type(P),N}`, so
# applying the same builtin to the shadow ref reads exactly the tangent at that position. The ref was
# already forced bounds-checked by `memoryrefnew` above, so this call's own boundscheck flag mirrors
# the primal unchanged.
function apply_builtin_frule!(::Val{Core.memoryrefget}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, _memrefgetg, ctx.presolve(actual[1]),
                       (ctx.presolve(a) for a in actual[2:end])...), Ti)
    TT = ctx.tt(Ti)
    t = TT === NoTangent ? NoTangent() :
        ctx.emit!(Expr(:call, _memrefgetg, ctx.tresolve(actual[1]),
                       (ctx.presolve(a) for a in actual[2:end])...), TT)
    p, t
end

# Writes one array element, returns the written value. `val` (actual[2]) is the only differentiable
# operand; ref/ordering/boundscheck mirror the primal call unchanged.
function apply_builtin_frule!(::Val{Core.memoryrefset!}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, _memrefsetg, ctx.presolve(actual[1]), ctx.presolve(actual[2]),
                       (ctx.presolve(a) for a in actual[3:end])...), Ti)
    TT = ctx.tt(Ti)
    t = TT === NoTangent ? NoTangent() :
        ctx.emit!(Expr(:call, _memrefsetg, ctx.tresolve(actual[1]), ctx.tresolve(actual[2]),
                       (ctx.presolve(a) for a in actual[3:end])...), TT)
    p, t
end

# Identity/egal — always Bool, never differentiable.
function apply_builtin_frule!(::Val{Core.:(===)}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, _eqeqg, ctx.presolve(actual[1]), ctx.presolve(actual[2])), Ti)
    p, NoTangent()
end

# Field-definedness check — always Bool, never differentiable. Shows up in boxed-capture IR (the
# `throw_undef_if_not` guard around a `Core.Box`'s `.contents` field) and any `isdefined(x, :f)` call.
function apply_builtin_frule!(::Val{Core.isdefined}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, _isdefinedg, (ctx.presolve(a) for a in actual)...), Ti)
    p, NoTangent()
end

# Same-shape tangent tuple: a non-differentiable slot holds NoTangent(), a differentiable slot holds
# its resolved tangent.
#
# `Ti` needs widening before `fieldtype`: a tuple whose elements inference only partly pinned down
# (e.g. the `Broadcasted` argument tuple in `x .+ 1.0`) carries a `Core.PartialStruct` lattice
# element, not a bare `Type`, and `fieldtype` throws a `TypeError` on one. Same trap `tt` already
# guards against.
function apply_builtin_frule!(::Val{Core.tuple}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, _ctupleg, (ctx.presolve(a) for a in actual)...), Ti)
    TT = ctx.tt(Ti)
    Tw = Ti isa Type ? Ti : CC.widenconst(Ti)
    t = TT === NoTangent ? NoTangent() :
        ctx.emit!(Expr(:call, _ctupleg,
                  (ctx.tt(fieldtype(Tw, j)) === NoTangent ? NoTangent() : ctx.tresolve(actual[j])
                   for j in eachindex(actual))...), TT)
    p, t
end

# A branchless select: same-shape shadow, indexed by the (non-differentiable) primal condition.
function apply_builtin_frule!(::Val{Core.ifelse}, actual, Ti, ctx)
    cond, a, b = actual[1], actual[2], actual[3]
    p = ctx.emit!(Expr(:call, _ifelseg, ctx.presolve(cond), ctx.presolve(a), ctx.presolve(b)), Ti)
    TT = ctx.tt(Ti)
    t = TT === NoTangent ? NoTangent() :
        ctx.emit!(Expr(:call, _ifelseg, ctx.presolve(cond), ctx.tresolve(a), ctx.tresolve(b)), TT)
    p, t
end
