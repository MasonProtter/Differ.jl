# ===========================================================================
# Reverse-mode builtin rules — dispatch-based, direct-IR-emission handling of `Core.Builtin`s that
# need more than "replay primally, nothing to route" treatment: `getfield`/`setfield!` and the
# `memoryref*` array-element builtins. Mirrors `intrinsics_reverse.jl`'s `Val(f)`-dispatch trick
# (itself mirroring forward mode's `builtins.jl`), but three-sided instead of one-sided, since
# reverse mode has three separate passes that must agree on a statement's shape: the static comms
# scan (`_scan_block_comms`), the forwards emission (`reverse_fwds_to_ircode`), and the pullback
# emission (`reverse_pullback_to_ircode`), all in `reverse_interp.jl`.
#
# Three generic functions, each with a `Val{F}`-keyed fallback that returns the literal singleton
# `nothing` — meaning "no rule registered for this builtin", which every call site distinguishes from
# a rule that ran and produced a real (non-`nothing`) result, however trivial that result is (an empty
# comms vector, a tuple of all-`nothing` contributions, ...). A *registered* rule that decides this
# particular call is out of scope (wrong types, untracked provenance, ...) signals that by setting
# `ctx.reason[]` and returning `false` from `builtin_rrule_comms` — distinct from both `nothing`
# ("try the next thing") and a real result ("here it is"). Only the comms-scan function needs this:
# it is the single place scope is decided (types + static provenance, nothing runtime), so by the time
# the fwds/pullback emission functions run for the same statement, the rule is guaranteed applicable.
#
#   (a) builtin_rrule_comms(::Val{F}, actual, Ti, ctx) -> Vector{Tuple{item,type}} | false | nothing
#   (b) apply_builtin_rrule_fwds!(::Val{F}, actual, Ti, ctx) -> (primal_ssa, shadow_ssa|nothing, saved) | nothing
#   (c) apply_builtin_rrule!(::Val{F}, actual, Ti, ctx) -> Tuple (one entry per operand, `nothing` for
#       an operand with no contribution) | nothing
#
# `ctx` is a `NamedTuple`, one shape per side:
#   (a) (optype, ssa, tracked, arg_tracked, reason)
#   (b) (emit!, icall!, presolve, sresolve, optype, tracked, ssa)
#   (c) (emit!, icall, comms, pb_presolve, deref_and_zero!, optype, ssa, ref_for)
#
# `ctx.tracked`/`ctx.arg_tracked` are `_fdata_tracked`/`_arg_fdata_tracked` (below/`reverse_interp.jl`):
# which SSA values/arguments have a statically-known fdata (shadow) value, needed by any rule whose
# pullback must reach into an object's `MutableTangent`/shadow `MemoryRef`. `ctx.deref_and_zero!(Pi)`
# derefs-and-zeros *this statement's own* rdata accumulator (always `ssa_ref_id[i]` — no rule ever
# needs another statement's). `ctx.ref_for(node)` is the rdata-accumulator lookup available for *any*
# SSA/Argument node (immutable-struct field accumulation, unlike fdata accumulation, needs no
# provenance tracking at all — see `reverse_interp.jl`'s header). `ctx.comms` is the current block's
# `comms_val_id` dict, keyed by the same tagged items `builtin_rrule_comms` declared.
#
# New comms item kinds beyond the pre-existing `:primal`/`:subtape`/`:shadow_ref`:
#   * `(:fshadow, obj_node)` — `obj_node`'s fdata handle (`MutableTangent`/shadow `Array`), resolved
#     by `sresolve` exactly like `:shadow_ref` (needed because `reverse_pullback_impl` has no access
#     to the argument coduals — every fdata handle must arrive via the tape).
#   * `(:old_primal, SSAValue(i))` / `(:old_tangent, SSAValue(i))` — the field/element value a
#     mutating statement overwrote, keyed by the mutating statement itself rather than an operand.
#     Unlike every other comms kind, these are computed by (b)'s own emitted statements rather than
#     resolved from an existing node, so (b) must return them in its `saved` dict for
#     `emit_epilogue!` to find (`reverse_interp.jl`) — the "single source of truth" invariant: (a)
#     declares an item, (b) must resolve or save a value for it, or `emit_epilogue!` raises an
#     internal error rather than silently mis-typing the comms tuple.
# ===========================================================================

builtin_rrule_comms(::Val{F}, actual, Ti, ctx) where {F} = nothing
apply_builtin_rrule_fwds!(::Val{F}, actual, Ti, ctx) where {F} = nothing
apply_builtin_rrule!(::Val{F}, actual, Ti, ctx) where {F} = nothing

_bi_fieldname(@nospecialize(node)) = isa(node, QuoteNode) ? node.value : node

# `@noinline` wrappers around the small `Tangent`-system accessors this file threads through `icall`/
# `icall!` into hand-built carrier IR. Without this, `CC.ssa_inlining_pass!` happily inlines the real
# (tiny, `@inline`-marked) functions' bodies straight into the carrier — and those bodies contain a
# bare `getfield`/`setfield!` call, which compiles to `GlobalRef(Differ, :getfield)` rather than
# `GlobalRef(Core, :getfield)` (an *implicit* `using Core` binding, not a directly-named one).
# `Core.Compiler.verify_ir` rejects that GlobalRef once embedded in value position in a
# freshly-built `IRCode` — exactly the `__pop_blk_stack!`/`__switch_case` hazard already documented
# above `reverse_pullback_to_ircode` (`reverse_interp.jl`), here triggered by data accessors instead of
# control-flow helpers. `@noinline` keeps each of these a genuine `:invoke`, so its internals are never
# re-embedded in the carrier's own IR at all.
@noinline _rr_get_tangent_field(t, i) = get_tangent_field(t, i)
@noinline _rr_set_tangent_field!(t, i, x) = set_tangent_field!(t, i, x)
@noinline _rr_get_fdata_field(f, name) = _get_fdata_field(f, name)
@noinline _rr_increment_field_rdata!(dx, dy, v) = increment_field_rdata!(dx, dy, v)
@noinline _rr_rdata(t) = rdata(t)
@noinline _rr_zero_tangent2(p, f) = zero_tangent(p, f)
@noinline _rr_build_tangent(::Type{P}, fields...) where {P} = build_tangent(P, fields...)

# ---------------------------------------------------------------------------
# `Core.getfield` — immutable structs accumulate via the object's own rdata `Ref` (`ref_for` +
# `increment_field!!`), unchanged from before this file existed. A *mutable* struct has no rdata of
# its own (its whole tangent lives in fdata) — its field's rdata contribution instead increments the
# `MutableTangent` in place via `increment_field_rdata!` (`tangents.jl`), which is what removes the
# mutable-struct bail. Either way `getfield`'s own *shadow* (its result's fdata, if any — a nested
# array or mutable substruct field) is handled independently in the fwds pass, per `_fdata_tracked`.
# ---------------------------------------------------------------------------

function builtin_rrule_comms(::Val{Core.getfield}, actual, Ti, ctx)
    length(actual) < 2 && return Tuple{Any,Any}[]
    obj = actual[1]
    P = ctx.optype(obj)
    dyn = !_bi_literal_index(actual[2])
    if dyn && tangent_type(_widen(Ti)) !== NoTangent
        # Dynamic (non-literal) field index into a differentiable field: only supported when every
        # field of the object shares one pure-rdata tangent type (`_bi_homog_tangent_type`) — a
        # homogeneous immutable Tuple/NamedTuple (the `for i in eachindex(t); s += t[i]; end`
        # pattern; `increment_field!!` has a runtime-`Int` method for it, tangents.jl) or a
        # homogeneous MUTABLE struct (whose contribution instead routes into the object's own
        # `MutableTangent` via the runtime-`Int` `increment_field_rdata!`, fwds_rvs_data.jl). A
        # HETEROGENEOUS object (mutable or not) has no such guarantee — bail rather than guess (a
        # raw, unresolved index here previously degenerated `Val(fieldidx)` into a bogus type and
        # silently dropped the gradient — see the bug this guards against).
        #
        # This is a deliberate scope boundary, not an unfinished TODO: it mirrors Mooncake's own
        # restriction of dynamic `getfield` to homogeneous immutable structures (see
        # `is_homogeneous_and_immutable`, `Mooncake.jl/src/rules/builtins.jl:1069`) — a
        # generated-unrolling path for the heterogeneous, per-field-typed case would be fragile and
        # type-unstable, and is out of scope even in the mature reference implementation.
        ok = P isa DataType && isconcretetype(P) && (P <: Tuple || P <: NamedTuple || ismutabletype(P)) &&
             fdtype(Ti) === NoFData && _bi_homog_tangent_type(P) === tangent_type(Ti)
        if !ok
            ctx.reason[] = "reverse mode `getfield` with a dynamic (non-literal) field index is only " *
                           "supported for a homogeneous Tuple/NamedTuple/mutable struct whose fields " *
                           "all share one pure-rdata tangent type — a deliberate limitation matching " *
                           "Mooncake's `is_homogeneous_and_immutable` restriction on dynamic getfield " *
                           "(Mooncake.jl/src/rules/builtins.jl), not an unfinished TODO; got object " *
                           "type $(P), field type $(Ti) at %$(ctx.ssa.id)"
            return false
        end
    end
    if rdtype(Ti) !== NoRData
        if P isa DataType && ismutabletype(P)
            tracked_here = isa(obj, Core.SSAValue) ? (obj.id <= length(ctx.tracked) && ctx.tracked[obj.id]) :
                           isa(obj, Core.Argument) ? (obj.n <= length(ctx.arg_tracked) && ctx.arg_tracked[obj.n]) :
                           false
            if !tracked_here
                ctx.reason[] = "mutable-struct `getfield` has no differentiable provenance traceable " *
                               "to a function argument at %$(ctx.ssa.id) (object type $(P))"
                return false
            end
            # A dynamic index additionally needs its own runtime value routed to the pullback
            # (`pb_presolve` looks up this `(:primal, ...)` item), exactly like the immutable case
            # below — `apply_builtin_rrule!` passes it to `increment_field_rdata!`'s runtime-`Int`
            # method as a plain `Int`, since `Val{fieldidx}` can't be built from a value not known
            # until the pullback runs.
            items = Tuple{Any,Any}[((:fshadow, obj), fdtype(P))]
            dyn && push!(items, ((:primal, actual[2]), ctx.optype(actual[2])))
            return items
        elseif !(P isa DataType)
            ctx.reason[] = "reverse mode does not support `getfield` on a non-concrete-struct type " *
                           "($(P)) at %$(ctx.ssa.id)"
            return false
        end
        # Immutable concrete struct: `ref_for` covers every SSA/Argument node unconditionally (no
        # provenance tracking needed for rdata accumulation) — a dynamic index additionally needs its
        # own runtime value routed to the pullback (`pb_presolve` looks up exactly this item).
        dyn && return Tuple{Any,Any}[((:primal, actual[2]), ctx.optype(actual[2]))]
    end
    return Tuple{Any,Any}[]
end

function apply_builtin_rrule_fwds!(::Val{Core.getfield}, actual, Ti, ctx)
    obj = actual[1]
    # `actual[2]` (the field name/index) is usually a literal `QuoteNode`/`Int`, for which
    # `presolve` is a no-op — but a homogeneous-tuple `getfield` with a *dynamic* index (e.g. the
    # `X[i]` inside `Base.vect`'s fill loop, `X::Tuple` a captured vararg) is a genuine `SSAValue`/
    # `Argument` operand that must be resolved to this pass's own numbering like any other operand,
    # not embedded as a dangling reference into the primal's numbering.
    p = ctx.emit!(Expr(:call, _getfieldg, ctx.presolve(obj), ctx.presolve(actual[2])), Ti)
    shadow = nothing
    if ctx.ssa.id <= length(ctx.tracked) && ctx.tracked[ctx.ssa.id]
        P = ctx.optype(obj)
        if P isa DataType && P <: Array
            # Same-shape shadow array: mirror the primal `.ref` access on the shadow directly.
            shadow = ctx.emit!(Expr(:call, _getfieldg, ctx.sresolve(obj), ctx.presolve(actual[2])), Ti)
        else
            # General struct: pull the field's fdata out of the object's fdata handle. Covers an
            # `FData`-wrapped immutable struct and a raw `MutableTangent` uniformly (`_get_fdata_field`).
            # `actual[2]` is always a literal here: `builtin_rrule_comms` bails on a dynamic index
            # whenever `fdtype(Ti) !== NoFData` (i.e. whenever this branch would otherwise run), so a
            # tracked fdata-carrying result never reaches this point with a runtime-computed index.
            fname = _bi_fieldname(actual[2])
            shadow = ctx.icall!(_rr_get_fdata_field, fdtype(Ti), (fdtype(P), typeof(fname)),
                                ctx.sresolve(obj), actual[2])
        end
    end
    return p, shadow, Dict{Any,Any}()
end

function apply_builtin_rrule!(::Val{Core.getfield}, actual, Ti, ctx)
    nores = ntuple(_ -> nothing, length(actual))
    rdtype(Ti) === NoRData && return nores
    acc = ctx.deref_and_zero!(Ti)
    obj = actual[1]
    P = ctx.optype(obj)
    # A literal index picks one field at compile time (`Val{fieldidx}`, dispatched statically). A
    # dynamic index is only reachable here when `builtin_rrule_comms` proved the object homogeneous
    # (a Tuple/NamedTuple or a mutable struct — never heterogeneous, see there). It must be resolved
    # to its own runtime value (via `pb_presolve`, looking up the `(:primal, ...)` comms item declared
    # above) and passed as a plain `Int` — `Val{fieldidx}` can't be built from a value not known until
    # the pullback runs; embedding the raw, unresolved operand instead (the original bug) made
    # `fieldidx` a bogus `SSAValue`, so `increment_field!!`'s generated `Val` dispatch compared it
    # against real field indices, always failed, and silently dropped the gradient.
    # `increment_field!!(x::Tuple/NamedTuple, y, i::Int)` (tangents.jl) and
    # `increment_field_rdata!(dx::MutableTangent, y, i::Int)` (fwds_rvs_data.jl) are the runtime-`Int`
    # methods built for the immutable and mutable cases respectively.
    if _bi_literal_index(actual[2])
        fname = _bi_fieldname(actual[2])
        fieldidx = fname isa Symbol ? findfirst(==(fname), fieldnames(P)) : fname
        idxty, idxval = Val{fieldidx}, Val(fieldidx)
    else
        idxty, idxval = Int, ctx.pb_presolve(actual[2])
    end
    # `acc`'s actual type is `zero_like_rdata_type(Ti)`, not `rdtype(Ti)` (see `deref_and_zero!` in
    # `reverse_interp.jl`) — may include `ZeroRData` when `Ti` isn't concrete enough (e.g. reading an
    # abstractly-typed field). Likewise `target`'s (`ctx.ref_for(obj)`) actual declared element type
    # is `zero_like_rdata_type(P)`, not `rdtype(P)`, whenever `obj`'s own primal type isn't concrete.
    if ismutabletype(P)
        mt = ctx.comms[(:fshadow, obj)]
        ctx.emit!(ctx.icall(_rr_increment_field_rdata!, (fdtype(P), zero_like_rdata_type(_widen(Ti)), idxty),
                            mt, acc, idxval), fdtype(P))
    else
        target = ctx.ref_for(obj)
        if target !== nothing
            RT = zero_like_rdata_type(_widen(P))
            cur = ctx.emit!(Expr(:call, _getfieldg, target, 1), RT)
            new = ctx.emit!(ctx.icall(increment_field!!, (RT, zero_like_rdata_type(_widen(Ti)), idxty), cur, acc, idxval), RT)
            ctx.emit!(Expr(:call, _setfieldg, target, 1, new), Any)
        end
    end
    return nores
end

# ---------------------------------------------------------------------------
# `Core.memorynew` — array allocation step 1: `Core.memorynew(Memory{P}, n)` allocates a fresh,
# uninitialized `Memory{P}`. Its own rdata is always `NoRData` (a `Memory` handle, not a
# differentiable value — mirrors `Base.memoryrefnew` below); the shadow allocates a same-length,
# uninitialized `Memory{tangent_type(P)}`, safe because every element the primal ever *reads* was
# necessarily *written* first, by an already-handled `memoryrefset!`. The length is structural, so
# it's `presolve`d, never `sresolve`d, in both primal and shadow calls (mirrors forward mode's
# identical rule, `builtins.jl`).
# ---------------------------------------------------------------------------
builtin_rrule_comms(::Val{Core.memorynew}, actual, Ti, ctx) = Tuple{Any,Any}[]
function apply_builtin_rrule_fwds!(::Val{Core.memorynew}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, Core.memorynew, ctx.presolve(actual[1]),
                       (ctx.presolve(a) for a in actual[2:end])...), Ti)
    shadow = nothing
    if ctx.ssa.id <= length(ctx.tracked) && ctx.tracked[ctx.ssa.id]
        TT = tangent_type(_widen(Ti))
        shadow = ctx.emit!(Expr(:call, Core.memorynew, TT,
                           (ctx.presolve(a) for a in actual[2:end])...), TT)
    end
    return p, shadow, Dict{Any,Any}()
end
apply_builtin_rrule!(::Val{Core.memorynew}, actual, Ti, ctx) = ntuple(_ -> nothing, length(actual))

# ---------------------------------------------------------------------------
# `Base.memoryrefnew` — its own rdata is always `NoRData` (a `MemoryRef` handle, not a differentiable
# value); the shadow chain it participates in is handled by `_fdata_tracked`/the fwds pass directly
# (rebuilt deterministically by both passes independently, needing no comms — mirrors `getfield`'s
# `.ref` case). Registered here only so the dispatch layer has an explicit (trivial) entry for it
# rather than silently relying on the generic no-tangent fallback.
#
# SAFETY: mirrors forward mode's identical rule (`builtins.jl`) — the 3-arg offsetting form's
# trailing boundscheck flag is NOT mirrored from the primal, always forced to `true` on the shadow
# ref, so a mismatched-length tangent array raises a catchable `BoundsError` instead of corrupting
# memory via an unchecked out-of-bounds `MemoryRef`.
# ---------------------------------------------------------------------------
builtin_rrule_comms(::Val{Base.memoryrefnew}, actual, Ti, ctx) = Tuple{Any,Any}[]
function apply_builtin_rrule_fwds!(::Val{Base.memoryrefnew}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, Base.memoryrefnew, (ctx.presolve(a) for a in actual)...), Ti)
    shadow = nothing
    if ctx.ssa.id <= length(ctx.tracked) && ctx.tracked[ctx.ssa.id]
        nargs = length(actual)
        shadow_args = Any[ctx.sresolve(actual[1])]
        if nargs >= 3
            for a in actual[2:end-1]
                push!(shadow_args, ctx.presolve(a))
            end
            push!(shadow_args, true)
        else
            for a in actual[2:end]
                push!(shadow_args, ctx.presolve(a))
            end
        end
        shadow = ctx.emit!(Expr(:call, Base.memoryrefnew, shadow_args...), Ti)
    end
    return p, shadow, Dict{Any,Any}()
end
apply_builtin_rrule!(::Val{Base.memoryrefnew}, actual, Ti, ctx) = ntuple(_ -> nothing, length(actual))

# ---------------------------------------------------------------------------
# `Base.memoryrefget` — two cases, by the result's own type:
#   * a bits (rdata-carrying) result needs its shadow `MemoryRef` handle communicated forward
#     (`:shadow_ref`), but only if traceable to a tracked argument; the pullback increments the
#     shadow element in place through that handle, no `Ref` accumulator needed (a `MemoryRef` is
#     mutable/pointer-like, exactly like an `Array`'s own tangent).
#   * an fdata-carrying result (a nested array) needs no comms at all: its shadow is resolved
#     directly in the fwds pass (mirroring this same call onto the shadow ref, below) and consumed
#     by `sresolve` wherever it's used, exactly like `Core.getfield`'s general-struct case — the
#     aliasing itself is the backward flow, nothing to route through the tape.
# ---------------------------------------------------------------------------
function builtin_rrule_comms(::Val{Base.memoryrefget}, actual, Ti, ctx)
    rdtype(Ti) === NoRData && return Tuple{Any,Any}[]
    ref_node = actual[1]
    if !(isa(ref_node, Core.SSAValue) && ref_node.id <= length(ctx.tracked) && ctx.tracked[ref_node.id])
        ctx.reason[] = "array read has no differentiable provenance traceable to a function " *
                       "argument at %$(ctx.ssa.id)"
        return false
    end
    return Tuple{Any,Any}[((:shadow_ref, ref_node), ctx.optype(ref_node))]
end

function apply_builtin_rrule_fwds!(::Val{Base.memoryrefget}, actual, Ti, ctx)
    # Primal replay: for a bits result, the shadow *value* is never read on the fwds pass, only the
    # shadow `MemoryRef` *handle* is communicated forward (registered above) for the pullback to
    # mutate later. For an fdata-carrying (nested-array) result, tracked by `_fdata_tracked`, mirror
    # the identical `memoryrefget` onto the shadow ref: the shadow of a nested element is the
    # corresponding element of the shadow array.
    p = ctx.emit!(Expr(:call, Base.memoryrefget, (ctx.presolve(a) for a in actual)...), Ti)
    shadow = nothing
    if ctx.ssa.id <= length(ctx.tracked) && ctx.tracked[ctx.ssa.id]
        shadow = ctx.emit!(Expr(:call, Base.memoryrefget, ctx.sresolve(actual[1]),
                                (ctx.presolve(a) for a in actual[2:end])...), Ti)
    end
    return p, shadow, Dict{Any,Any}()
end

function apply_builtin_rrule!(::Val{Base.memoryrefget}, actual, Ti, ctx)
    nores = ntuple(_ -> nothing, length(actual))
    rdtype(Ti) === NoRData && return nores
    acc = ctx.deref_and_zero!(Ti)
    shadow_ref = ctx.comms[(:shadow_ref, actual[1])]
    # This rule only ever reaches a "bits" (rdata-carrying-directly) element, so ordinarily
    # `rdtype(Ti) == Ti` -- but an array with an abstract eltype (e.g. `Vector{Real}`) makes `Ti`
    # non-concrete, in which case `acc`'s actual type is `zero_like_rdata_type(Ti)`, not bare `Ti`.
    RT = zero_like_rdata_type(_widen(Ti))
    cur = ctx.emit!(Expr(:call, Base.memoryrefget, shadow_ref, QuoteNode(:not_atomic), false), RT)
    new = ctx.emit!(ctx.icall(increment!!, (RT, RT), cur, acc), RT)
    ctx.emit!(Expr(:call, Base.memoryrefset!, shadow_ref, new, QuoteNode(:not_atomic), false), RT)
    return nores
end

# ---------------------------------------------------------------------------
# `Core.setfield!` — Mooncake's `lsetfield_rrule` (`Mooncake.jl/src/rules/misc.jl`), adapted to
# Differ's IR-emission (rather than closure-based) pullback: everything the pullback needs (the old
# field value, the old tangent, the object's fdata handle, the object's primal handle) must be either
# resolved from an existing comms item or explicitly saved by the fwds emission, since the pullback
# carrier shares no state with the fwds carrier except the tape.
#
# Two field shapes, both handled by the same comms/pullback shape (only the fwds emission of the new
# field tangent differs — see `apply_builtin_rrule_fwds!` below):
#   * pure rdata (a scalar, or an immutable-scalar struct) — the field's new tangent is a fresh zero,
#     the assigned value's gradient flows back purely through the pullback's rdata return.
#   * fdata-carrying (an array- or mutable-struct-valued field) — the field's new tangent *aliases*
#     the assigned value's own shadow, so any later in-place accumulation into that shared shadow (a
#     downstream array write/mutable-struct field write reached through this field) already lands in
#     the assigned value's own gradient; the pullback still returns rdata for the value, but it only
#     ever carries whatever rdata the field separately accumulated (`NoRData` when the field is purely
#     an array, since an array has no rdata component).
# This requires the assigned value's own fdata to be resolvable at the point of the call — an
# `Argument` (always resolvable) or an `SSAValue` with statically-known provenance (`ctx.tracked`),
# exactly like the object's own provenance guard below.
# ---------------------------------------------------------------------------
function builtin_rrule_comms(::Val{Core.setfield!}, actual, Ti, ctx)
    length(actual) < 3 && return false
    obj = actual[1]
    P = ctx.optype(obj)
    if !(P isa DataType && ismutabletype(P))
        ctx.reason[] = "reverse mode `setfield!` requires a concrete mutable struct, got $(P) at " *
                       "%$(ctx.ssa.id)"
        return false
    end
    if !_bi_literal_index(actual[2])
        # Dynamic (non-literal) write index — Phase A only, no same-shape support: unlike a read,
        # `set_tangent_field!` needs a statically-known field to place the new value into the right
        # slot type, and a mutable object is never itself a same-shape aggregate. This file's
        # pullback/fwds emission for `setfield!` also embeds the field index as a genuine compile-time
        # constant (unlike `getfield`'s, which was fixed to resolve a dynamic index through the
        # tape) — so bail unconditionally here rather than partially fix a path with no test coverage.
        ctx.reason[] = "reverse mode `setfield!` does not support a dynamic (non-literal) field " *
                       "index ($(P)) at %$(ctx.ssa.id)"
        return false
    end
    fname = _bi_fieldname(actual[2])
    if fdtype(tangent_type(fieldtype(P, fname))) !== NoFData
        val_node = actual[3]
        val_tracked = isa(val_node, Core.SSAValue) ? (val_node.id <= length(ctx.tracked) && ctx.tracked[val_node.id]) :
                      isa(val_node, Core.Argument) ? (val_node.n <= length(ctx.arg_tracked) && ctx.arg_tracked[val_node.n]) :
                      false
        if !val_tracked
            ctx.reason[] = "reverse mode `setfield!` of a field whose tangent carries fdata " *
                           "($(fieldtype(P, fname))) requires the assigned value's own fdata to be " *
                           "traceable to a function argument at %$(ctx.ssa.id)"
            return false
        end
    end
    tracked_here = isa(obj, Core.SSAValue) ? (obj.id <= length(ctx.tracked) && ctx.tracked[obj.id]) :
                   isa(obj, Core.Argument) ? (obj.n <= length(ctx.arg_tracked) && ctx.arg_tracked[obj.n]) :
                   false
    if !tracked_here
        ctx.reason[] = "mutable-struct `setfield!` has no differentiable provenance traceable to a " *
                       "function argument at %$(ctx.ssa.id) (object type $(P))"
        return false
    end
    return Tuple{Any,Any}[((:fshadow, obj), fdtype(P)), ((:primal, obj), P),
                          ((:old_primal, ctx.ssa), Ti), ((:old_tangent, ctx.ssa), tangent_type(_widen(Ti)))]
end

function apply_builtin_rrule_fwds!(::Val{Core.setfield!}, actual, Ti, ctx)
    obj, name_node, val_node = actual[1], actual[2], actual[3]
    P = ctx.optype(obj)
    mt = ctx.sresolve(obj)
    TF = tangent_type(_widen(Ti))
    old_primal = ctx.emit!(Expr(:call, _getfieldg, ctx.presolve(obj), name_node), Ti)
    fname = _bi_fieldname(name_node)
    fieldidx = fname isa Symbol ? findfirst(==(fname), fieldnames(P)) : fname
    old_tangent = ctx.icall!(_rr_get_tangent_field, TF, (fdtype(P), Int), mt, fieldidx)
    # The field's new tangent: `zero_tangent(p, f)` embeds `f` (the assigned value's own fdata)
    # directly rather than fabricating a fresh zero when the field carries fdata — that embedding
    # *is* the alias that makes later in-place accumulation into this field flow straight into the
    # assigned value's own shadow. `f = NoFData()` for a pure-rdata field collapses this back to the
    # original fresh-zero behavior (`zero_tangent(p, ::NoFData) = zero_tangent(p)`).
    FTi = fdtype(Ti)
    fdata_val = FTi === NoFData ? NoFData() : ctx.sresolve(val_node)
    zt = ctx.icall!(_rr_zero_tangent2, TF, (Ti, FTi), ctx.presolve(val_node), fdata_val)
    ctx.icall!(_rr_set_tangent_field!, TF, (fdtype(P), Int, TF), mt, fieldidx, zt)
    p = ctx.emit!(Expr(:call, _setfieldg, ctx.presolve(obj), name_node, ctx.presolve(val_node)), Ti)
    saved = Dict{Any,Any}((:old_primal, ctx.ssa) => old_primal, (:old_tangent, ctx.ssa) => old_tangent)
    return p, nothing, saved
end

# Unchanged by the fdata-aliasing branch above: `rdata(cur_tangent)` already reads whatever rdata the
# field separately accumulated (`NoRData` for a pure array field, since the fdata contribution went
# straight into the aliased shadow in place, not through this rdata path at all), and restoring the
# field slot to `old_primal`/`old_tangent` is exactly the same operation either way.
function apply_builtin_rrule!(::Val{Core.setfield!}, actual, Ti, ctx)
    obj, name_node = actual[1], actual[2]
    P = ctx.optype(obj)
    mt = ctx.comms[(:fshadow, obj)]
    primal_obj = ctx.comms[(:primal, obj)]
    old_primal = ctx.comms[(:old_primal, ctx.ssa)]
    old_tangent = ctx.comms[(:old_tangent, ctx.ssa)]
    fname = _bi_fieldname(name_node)
    fieldidx = fname isa Symbol ? findfirst(==(fname), fieldnames(P)) : fname
    TF = tangent_type(_widen(Ti))
    # `zero_like_rdata_type`, not `rdtype`: `acc` (below) may be `ZeroRData` when `Ti` (the field's
    # own type) isn't concrete enough (e.g. an abstractly-typed field). `cur_rdata` is always a real
    # value regardless (`_rr_rdata` on a genuine tangent), so declaring it at the same (possibly
    # wider) `RT` is harmless.
    RT = zero_like_rdata_type(_widen(Ti))
    acc = ctx.deref_and_zero!(Ti)
    cur_tangent = ctx.emit!(ctx.icall(_rr_get_tangent_field, (fdtype(P), Int), mt, fieldidx), TF)
    cur_rdata = ctx.emit!(ctx.icall(_rr_rdata, (TF,), cur_tangent), RT)
    new_dx = ctx.emit!(ctx.icall(increment!!, (RT, RT), acc, cur_rdata), RT)
    ctx.emit!(Expr(:call, _setfieldg, primal_obj, name_node, old_primal), Any)
    ctx.emit!(ctx.icall(_rr_set_tangent_field!, (fdtype(P), Int, TF), mt, fieldidx, old_tangent), TF)
    return ntuple(j -> j == 3 ? new_dx : nothing, length(actual))
end

# ---------------------------------------------------------------------------
# `Base.memoryrefset!` — the array analogue of `Core.setfield!` above: the "object" is a `MemoryRef`,
# its "field" the pointed-to element, saved/restored the same way. Two element shapes, exactly
# mirroring `setfield!`'s two field shapes:
#   * pure rdata (a bits scalar like `Float64`) — the element's new tangent is a fresh zero, the
#     assigned value's gradient flows back purely through the pullback's rdata return.
#   * fdata-carrying (a nested array) — the element's new tangent *aliases* the assigned value's own
#     shadow, so a later in-place accumulation into that shared shadow (reached through this element,
#     e.g. `x[1]`) already lands in the assigned value's own gradient. This requires the assigned
#     value's own fdata to be resolvable at the point of the call — an `Argument` or a tracked
#     `SSAValue` — exactly like the object's own provenance guard below.
# ---------------------------------------------------------------------------
function builtin_rrule_comms(::Val{Base.memoryrefset!}, actual, Ti, ctx)
    length(actual) < 2 && return false
    elt = Ti
    ref_node, val_node = actual[1], actual[2]
    if fdtype(elt) !== NoFData
        val_tracked = isa(val_node, Core.SSAValue) ? (val_node.id <= length(ctx.tracked) && ctx.tracked[val_node.id]) :
                      isa(val_node, Core.Argument) ? (val_node.n <= length(ctx.arg_tracked) && ctx.arg_tracked[val_node.n]) :
                      false
        if !val_tracked
            ctx.reason[] = "reverse mode `memoryrefset!` of a non-bits element ($(elt)) requires the " *
                           "assigned value's own fdata to be traceable to a function argument at " *
                           "%$(ctx.ssa.id)"
            return false
        end
    elseif rdtype(elt) !== tangent_type(_widen(elt))
        ctx.reason[] = "reverse mode does not support array mutation of element type ($(elt)) at " *
                       "%$(ctx.ssa.id)"
        return false
    end
    if !(isa(ref_node, Core.SSAValue) && ref_node.id <= length(ctx.tracked) && ctx.tracked[ref_node.id])
        ctx.reason[] = "array write has no differentiable provenance traceable to a function " *
                       "argument at %$(ctx.ssa.id)"
        return false
    end
    return Tuple{Any,Any}[((:shadow_ref, ref_node), ctx.optype(ref_node)), ((:primal, ref_node), ctx.optype(ref_node)),
                          ((:old_primal, ctx.ssa), elt), ((:old_tangent, ctx.ssa), tangent_type(_widen(elt)))]
end

function apply_builtin_rrule_fwds!(::Val{Base.memoryrefset!}, actual, Ti, ctx)
    ref_node, val_node = actual[1], actual[2]
    rest = @view actual[3:end]
    pref, sref = ctx.presolve(ref_node), ctx.sresolve(ref_node)
    TT = tangent_type(_widen(Ti))
    old_primal = ctx.emit!(Expr(:call, Base.memoryrefget, pref, (ctx.presolve(a) for a in rest)...), Ti)
    old_tangent = ctx.emit!(Expr(:call, Base.memoryrefget, sref, (ctx.presolve(a) for a in rest)...), TT)
    # `zero_tangent(p, f)` embeds `f` (the assigned value's own fdata) directly rather than
    # fabricating a fresh zero when the element carries fdata — that embedding *is* the alias that
    # makes later in-place accumulation into this element flow straight into the assigned value's
    # own shadow. `f = NoFData()` for a pure-rdata element collapses this back to the original
    # fresh-zero behavior (`zero_tangent(p, ::NoFData) = zero_tangent(p)`).
    FTi = fdtype(Ti)
    fdata_val = FTi === NoFData ? NoFData() : ctx.sresolve(val_node)
    zt = ctx.icall!(_rr_zero_tangent2, TT, (Ti, FTi), ctx.presolve(val_node), fdata_val)
    ctx.emit!(Expr(:call, Base.memoryrefset!, sref, zt, (ctx.presolve(a) for a in rest)...), TT)
    p = ctx.emit!(Expr(:call, Base.memoryrefset!, pref, ctx.presolve(val_node), (ctx.presolve(a) for a in rest)...), Ti)
    saved = Dict{Any,Any}((:old_primal, ctx.ssa) => old_primal, (:old_tangent, ctx.ssa) => old_tangent)
    return p, nothing, saved
end

# Unchanged by the fdata-aliasing branch above: `rdata(cur_tangent)` already reads whatever rdata the
# element separately accumulated (`NoRData` for a pure array element, since the fdata contribution
# went straight into the aliased shadow in place, not through this rdata path at all), and restoring
# the element slot to `old_primal`/`old_tangent` is exactly the same operation either way.
function apply_builtin_rrule!(::Val{Base.memoryrefset!}, actual, Ti, ctx)
    ref_node = actual[1]
    sref = ctx.comms[(:shadow_ref, ref_node)]
    pref = ctx.comms[(:primal, ref_node)]
    old_primal = ctx.comms[(:old_primal, ctx.ssa)]
    old_tangent = ctx.comms[(:old_tangent, ctx.ssa)]
    TT = tangent_type(_widen(Ti))
    # `zero_like_rdata_type`, not `rdtype`: `acc` (below) may be `ZeroRData` when `Ti` (the element's
    # own type) isn't concrete enough (e.g. an array with an abstract eltype). `cur_rdata` is always
    # a real value regardless, so declaring it at the same (possibly wider) `RT` is harmless.
    RT = zero_like_rdata_type(_widen(Ti))
    acc = ctx.deref_and_zero!(Ti)
    cur_tangent = ctx.emit!(Expr(:call, Base.memoryrefget, sref, QuoteNode(:not_atomic), false), TT)
    cur_rdata = ctx.emit!(ctx.icall(_rr_rdata, (TT,), cur_tangent), RT)
    new_dx = ctx.emit!(ctx.icall(increment!!, (RT, RT), acc, cur_rdata), RT)
    ctx.emit!(Expr(:call, Base.memoryrefset!, sref, old_tangent, QuoteNode(:not_atomic), false), TT)
    ctx.emit!(Expr(:call, Base.memoryrefset!, pref, old_primal, QuoteNode(:not_atomic), false), Ti)
    return ntuple(j -> j == 2 ? new_dx : nothing, length(actual))
end
