# Reverse-mode builtin rules — dispatch-based, direct-IR-emission handling of `Core.Builtin`s that
# need more than "replay primally, nothing to route": `getfield`/`setfield!` and the `memoryref*`
# array-element builtins. Mirrors `intrinsics_reverse.jl`'s `Val(f)`-dispatch trick, but three-sided:
# three passes must agree on a statement's shape — the static comms scan (`_scan_block_comms`),
# forwards emission (`reverse_fwds_to_ircode`), and pullback emission (`reverse_pullback_to_ircode`),
# all in `reverse_interp.jl`.
#
#   (a) builtin_rrule_comms(::Val{F}, actual, Ti, ctx) -> Vector{Tuple{item,type}} | false | nothing
#   (b) apply_builtin_rrule_fwds!(::Val{F}, actual, Ti, ctx) -> (primal_ssa, shadow_ssa|nothing, saved) | nothing
#   (c) apply_builtin_rrule!(::Val{F}, actual, Ti, ctx) -> Tuple (one entry per operand, `nothing` for
#       an operand with no contribution) | nothing
#
# `nothing` means "no rule registered — try something else". `false` (only `(a)` returns it) means "a
# rule is registered but this call is out of scope" (wrong types, untracked provenance, ...), signaled
# via `ctx.reason[]`. Only the comms-scan function needs `false`: it's the one place scope is decided
# from types + static provenance, so by the time `(b)`/`(c)` run the rule is already known to apply.
#
# `ctx` is a `NamedTuple`, one shape per side:
#   (a) (optype, ssa, tracked, arg_tracked, reason)
#   (b) (emit!, icall!, presolve, sresolve, optype, tracked, ssa)
#   (c) (emit!, icall, fetch_shadow, fetch_primal, fetch_saved, pb_presolve, deref_and_zero!,
#        optype, ssa, ref_for)
#
# `Ti` and `ctx.optype(x)` are always bare `Type`s — all three passes widen before dispatching here.
#
# `ctx.tracked`/`ctx.arg_tracked` (`_fdata_tracked`/`_arg_fdata_tracked`) say which SSA values/
# arguments have a statically-known fdata (shadow) value — needed by any rule whose pullback reaches
# into an object's `MutableTangent`/shadow `MemoryRef`. `ctx.deref_and_zero!(Pi)` derefs-and-zeros
# this statement's own rdata accumulator. `ctx.ref_for(node)` is the rdata-accumulator lookup for any
# SSA/Argument node.
#
# A pullback reads forwards-recorded values only through three resolvers, never by indexing the
# comms dict directly: `ctx.fetch_shadow(node)` (fdata handle), `ctx.fetch_primal(node)` (primal
# value, also resolves literals), `ctx.fetch_saved(item)` (a value `(b)` stashed under a tagged item,
# e.g. `(:old_primal, ssa)`). Each raises a located internal error rather than silently returning a
# value the forwards pass never recorded.
#
# New comms item kinds beyond `:primal`/`:subtape`/`:shadow_ref`:
#   * `(:fshadow, obj_node)` — `obj_node`'s fdata handle, resolved like `:shadow_ref`. Needed because
#     `reverse_pullback_impl` has no access to the original argument coduals — every fdata handle
#     must arrive via the tape.
#   * `(:old_primal, SSAValue(i))` / `(:old_tangent, SSAValue(i))` — the field/element value a
#     mutating statement overwrote, keyed by the mutating statement itself. Unlike other comms kinds
#     these are computed by `(b)`'s own emitted statements, so `(b)` must return them in its `saved`
#     dict for `emit_epilogue!` to find, or it raises an internal error rather than silently
#     mis-typing the comms tuple.

builtin_rrule_comms(::Val{F}, actual, Ti, ctx) where {F} = nothing
apply_builtin_rrule_fwds!(::Val{F}, actual, Ti, ctx) where {F} = nothing
apply_builtin_rrule!(::Val{F}, actual, Ti, ctx) where {F} = nothing

_bi_fieldname(@nospecialize(node)) = isa(node, QuoteNode) ? node.value : node

# Whether `node`'s fdata (shadow) is statically known: an `SSAValue` traceable to a tracked
# statement, an `Argument` (always tracked), or `false` for anything else (a literal/`GlobalRef`).
_bi_tracked(@nospecialize(node), ctx) =
    isa(node, Core.SSAValue) ? (node.id <= length(ctx.tracked) && ctx.tracked[node.id]) :
    isa(node, Core.Argument) ? (node.n <= length(ctx.arg_tracked) && ctx.arg_tracked[node.n]) :
    false

# `@noinline` wrappers around the small `Tangent`-system accessors threaded through `icall`/`icall!`
# into hand-built carrier IR. `@noinline` keeps each a genuine `:invoke` rather than expanding the
# tiny body at every emission site.
@noinline _rr_get_tangent_field(t, i) = get_tangent_field(t, i)
@noinline _rr_set_tangent_field!(t, i, x) = set_tangent_field!(t, i, x)
@noinline _rr_get_fdata_field(f, name) = _get_fdata_field(f, name)
@noinline _rr_increment_field_rdata!(dx, dy, v) = increment_field_rdata!(dx, dy, v)
@noinline _rr_rdata(t) = rdata(t)
@noinline _rr_fdata(t) = fdata(t)
@noinline _rr_increment_rdata!!(t, r) = increment_rdata!!(t, r)

@noinline _rr_zero_tangent2(p, f) = zero_tangent(p, f)
# An inactive value's shadow slot: a fresh zero, since there is no real shadow to alias.
@noinline _rr_zero_fdata(p) = fdata(zero_tangent(p))
@noinline _rr_build_tangent(::Type{P}, fields...) where {P} = build_tangent(P, fields...)

# Runtime aliasing guard (`_collect_alias_guard_globals` in `reverse_interp.jl`): an active argument
# that is identically a module global reverse mode replays as a constant would otherwise get a
# silently wrong gradient. `@noinline` for the usual carrier-embedding reason above.
@noinline function _rr_check_global_alias(argval, gv, msg::String)
    argval === gv && error(msg)
    return nothing
end

# Save/restore of a `Memory` slot that may not be assigned yet: a fresh array's `Core.memorynew`
# leaves every slot undefined, so `memoryrefset!`'s read of the value it overwrites throws
# `UndefRefError` for a non-`isbits` element. Restoring `nothing` leaves the slot undefined, which is
# correct — in reverse order, anything reading it afterwards ran before the store in the primal.
# `isbits` elements keep the plain path, so their comms tuple stays `isbits`.
@noinline _rr_memref_get_or_nothing(ref) =
    Base.memoryref_isassigned(ref, :not_atomic, false) ?
        Base.memoryrefget(ref, :not_atomic, false) : nothing
@noinline function _rr_memref_restore!(ref, v)
    v === nothing || Base.memoryrefset!(ref, v, :not_atomic, false)
    return nothing
end
_rr_saved_slot_type(@nospecialize T) = isbitstype(T) ? T : Union{Nothing,T}

# Direct-emission fast paths for `get_tangent_field`/`set_tangent_field!` (reverse-mode analogue of
# forward mode's `_tangent_field_slot`). For the common case — a concrete struct whose
# `MutableTangent` field is not a `PossiblyUninitTangent` — the equivalent instructions are emitted
# directly with qualified `Core` GlobalRefs (`_getfieldg`/`_setfieldg`), dropping the per-iteration
# `:invoke`. `_tangent_field_slot(P, name)` returns `(NT, fi)` for that case and `nothing` otherwise,
# routing the latter back to the `_rr_*` barrier above.

# `getfield(getfield(mt, :fields), fi)`; `PossiblyUninitTangent` slots go through the barrier.
function _emit_gtf!(ctx, mt, slot)
    NT, fi = slot
    fnt = ctx.emit!(Expr(:call, _getfieldg, mt, QuoteNode(:fields)), NT)
    return ctx.emit!(Expr(:call, _getfieldg, fnt, fi), fieldtype(NT, fi))
end

# Rebuild `mt.fields` with slot `fi` set to `val`, then `setfield!` it back — what
# `set_tangent_field!` compiles to.
function _emit_stf!(ctx, mt, slot, val)
    NT, fi = slot
    old = ctx.emit!(Expr(:call, _getfieldg, mt, QuoteNode(:fields)), NT)
    slots = Any[j == fi ? val : ctx.emit!(Expr(:call, _getfieldg, old, j), fieldtype(NT, j))
                for j in 1:fieldcount(NT)]
    nt = ctx.emit!(Expr(:new, NT, slots...), NT)
    ctx.emit!(Expr(:call, _setfieldg, mt, QuoteNode(:fields), nt), NT)
    return nothing
end

# `rdata` of a tangent, emitted only when it actually does something: identity when the tangent
# type is its own rdata type (bits scalars), `NoRData()` when there's none — both common in a
# mutation pullback's hot loop, where the `@noinline` `_rr_rdata` call would otherwise cost a real
# call per reverse iteration. Falls back to the call for a genuine `Tangent` fdata/rdata split.
function _emit_rdata!(ctx, @nospecialize(TT), @nospecialize(RT), cur_tangent)
    ctx.rdtype(TT) === NoRData && return ctx.emit!(QuoteNode(NoRData()), RT)
    (RT === TT && ctx.rdtype(TT) === TT) && return cur_tangent
    return ctx.emit!(ctx.icall(_rr_rdata, (TT,), cur_tangent), RT)
end

# Fold an rdata contribution `acc::RT` into a tangent `cur::TT`. `ZeroRData` means nothing was
# accumulated, and `increment_rdata!!` has no method for it anyway.
function _emit_increment_rdata!(ctx, @nospecialize(TT), @nospecialize(RT), cur, acc)
    ctx.rdtype(TT) === NoRData && return cur
    RT === ZeroRData && return cur
    ctx.rdtype(TT) === TT && return ctx.emit!(ctx.icall(increment!!, (TT, RT), cur, acc), TT)
    return ctx.emit!(ctx.icall(_rr_increment_rdata!!, (TT, RT), cur, acc), TT)
end

# `fdata` of a tangent loaded at its own type `TT`, at the declared fdata type `FT`. Mirror of
# `_emit_rdata!` above, same fast paths.
function _emit_fdata!(ctx, @nospecialize(TT), @nospecialize(FT), cur_tangent)
    ctx.fdtype(TT) === NoFData && return ctx.emit!(QuoteNode(NoFData()), FT)
    (FT === TT && ctx.fdtype(TT) === TT) && return cur_tangent
    return ctx.icall!(_rr_fdata, FT, (TT,), cur_tangent)
end

# ---------------------------------------------------------------------------
# `Core.getfield` — an immutable struct accumulates via the object's own rdata `Ref` (`ref_for` +
# `increment_field!!`). A mutable struct has no rdata of its own (its tangent lives entirely in
# fdata) — its field's rdata contribution instead increments the `MutableTangent` in place via
# `increment_field_rdata!`. Either way `getfield`'s own shadow (a nested array or mutable substruct
# field) is handled independently in the fwds pass, per `_fdata_tracked`.
# ---------------------------------------------------------------------------

function builtin_rrule_comms(::Val{Core.getfield}, actual, Ti, ctx)
    length(actual) < 2 && return Tuple{Any,Any}[]
    obj = actual[1]
    P = ctx.optype(obj)
    dyn = !_bi_literal_index(actual[2])
    if dyn
        # A dynamic index into a mixed-activity tuple (a packed vararg tail with some trailing
        # arguments held constant) cannot be resolved: which slot is read — and so whether it is
        # constant — is not statically decidable. All-constant never reaches here (the read itself
        # is inactive and replayed primally), so this is a genuine mix.
        Fobj = ctx.sty(obj)
        if Fobj isa DataType && Fobj <: Tuple && any(T -> T === Inactive, Fobj.parameters)
            ctx.reason[] = "reverse mode `getfield` with a dynamic (non-literal) index into a " *
                           "tuple of mixed activity (shadow type $(Fobj)) is not supported: hold " *
                           "the trailing arguments uniformly active, or all constant, at " *
                           "%$(ctx.ssa.id)"
            return false
        end
    end
    if dyn && ctx.tt(_widen(Ti)) !== NoTangent
        # Dynamic (non-literal) field index into a differentiable field: two accepted shapes, both
        # requiring a homogeneous Tuple/NamedTuple (or mutable struct, pure-rdata case only) —
        # pure-rdata (`_bi_homog_tangent_type`, mirrors Mooncake's `is_homogeneous_and_immutable`),
        # or fdata-carrying with no rdata (every array element type). Heterogeneous bails rather
        # than guessing — deliberate scope boundary, not a TODO.
        homog_pure_rdata = P isa DataType && isconcretetype(P) &&
            (P <: Tuple || P <: NamedTuple || ismutabletype(P)) &&
            ctx.fdtype(Ti) === NoFData && _bi_homog_tangent_type(ctx.tt, P) === ctx.tt(Ti)
        homog_fdata_tuple = P isa DataType && isconcretetype(P) && (P <: Tuple || P <: NamedTuple) &&
            ctx.rdtype(Ti) === NoRData && ctx.fdtype(Ti) !== NoFData &&
            _bi_homog_tangent_type(ctx.fdtype, P) === ctx.fdtype(Ti)
        if !(homog_pure_rdata || homog_fdata_tuple)
            ctx.reason[] = "reverse mode `getfield` with a dynamic (non-literal) field index is only " *
                           "supported for a homogeneous Tuple/NamedTuple/mutable struct whose fields " *
                           "all share one pure-rdata tangent type, or a homogeneous Tuple/NamedTuple " *
                           "whose fields all share one fdata type and carry no rdata — a deliberate " *
                           "limitation matching Mooncake's `is_homogeneous_and_immutable` restriction " *
                           "on dynamic getfield (Mooncake.jl/src/rules/builtins.jl), not an unfinished " *
                           "TODO; got object type $(P), field type $(Ti) at %$(ctx.ssa.id)"
            return false
        end
    end
    if ctx.rdtype(Ti) !== NoRData
        if P isa DataType && ismutabletype(P)
            if !_bi_tracked(obj, ctx)
                ctx.reason[] = "mutable-struct `getfield` has no differentiable provenance traceable " *
                               "to a function argument at %$(ctx.ssa.id) (object type $(P))"
                return false
            end
            # A dynamic index additionally needs its own runtime value routed to the pullback
            # (`pb_presolve` looks up this `(:primal, ...)` item), like the immutable case below —
            # `apply_builtin_rrule!` passes it to `increment_field_rdata!`'s runtime-`Int` method as a
            # plain `Int`, since `Val{fieldidx}` can't be built from a value not known until the
            # pullback runs.
            items = Tuple{Any,Any}[((:fshadow, obj), ctx.fdtype(P))]
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
    # `actual[2]` is usually a literal, for which `presolve` is a no-op — but a homogeneous-tuple
    # `getfield` with a dynamic index is a genuine SSAValue/Argument operand that must be resolved
    # to this pass's own numbering like any other operand.
    p = ctx.emit!(Expr(:call, _getfieldg, ctx.presolve(obj), ctx.presolve(actual[2])), Ti)
    shadow = nothing
    if _bi_tracked(ctx.ssa, ctx)
        P = ctx.optype(obj)
        if (P isa DataType && P <: Array) || !_bi_literal_index(actual[2])
            # Array shadow, or a homogeneous fdata-tuple's own dynamic-index shadow (both a plain
            # `getfield` at `ctx.fdtype(Ti)` — homogeneity makes the result type static).
            shadow = ctx.emit!(Expr(:call, _getfieldg, ctx.sresolve(obj), ctx.presolve(actual[2])), ctx.sty(ctx.ssa))
        else
            # General struct, literal index: pull the field's fdata out of the object's fdata
            # handle, covering an `FData`-wrapped immutable struct and a raw `MutableTangent`
            # uniformly.
            fname = _bi_fieldname(actual[2])
            shadow = ctx.icall!(_rr_get_fdata_field, ctx.sty(ctx.ssa), (ctx.sty(obj), typeof(fname)),
                                ctx.sresolve(obj), actual[2])
        end
    end
    return p, shadow, Dict{Any,Any}()
end

function apply_builtin_rrule!(::Val{Core.getfield}, actual, Ti, ctx)
    nores = ntuple(_ -> nothing, length(actual))
    ctx.rdtype(Ti) === NoRData && return nores
    acc = ctx.deref_and_zero!(Ti)
    obj = actual[1]
    P = ctx.optype(obj)
    # A literal index picks one field at compile time (`Val{fieldidx}`). A dynamic index is only
    # reachable here when `builtin_rrule_comms` proved the object homogeneous; it's resolved to its
    # runtime value via `pb_presolve` and passed as a plain `Int` — `Val{fieldidx}` can't be built
    # from a value not known until the pullback runs. Embedding the raw unresolved operand instead
    # (a past bug) made `fieldidx` a bogus `SSAValue` and silently dropped the gradient.
    if _bi_literal_index(actual[2])
        fname = _bi_fieldname(actual[2])
        fieldidx = fname isa Symbol ? findfirst(==(fname), fieldnames(P)) : fname
        idxty, idxval = Val{fieldidx}, Val(fieldidx)
    else
        idxty, idxval = Int, ctx.pb_presolve(actual[2])
    end
    # `acc`'s actual type is `zero_like_rdata_type(Ti)`, not `rdtype(Ti)` — may be `ZeroRData` when
    # `Ti` isn't concrete (e.g. an abstractly-typed field). Same for `target`'s declared element type.
    if ismutabletype(P)
        mt = ctx.fetch_shadow(obj)
        slot = _bi_literal_index(actual[2]) ? _tangent_field_slot(ctx.tt, P, actual[2]) : nothing
        if slot !== nothing
            # `increment_field_rdata!` is `set_tangent_field!(dx, f, increment_rdata!!(get_tangent_field(dx, f), acc))`;
            # emit the read/write directly — safe since a scalar field touches no `MutableTangent.fields` rebuild.
            TFslot = fieldtype(slot[1], slot[2])
            RTcur = zero_like_rdata_type(_widen(Ti))
            cur = _emit_gtf!(ctx, mt, slot)
            cur_rdata = _emit_rdata!(ctx, TFslot, RTcur, cur)
            new = ctx.emit!(ctx.icall(increment_rdata!!, (TFslot, RTcur), cur_rdata, acc), TFslot)
            _emit_stf!(ctx, mt, slot, new)
        else
            ctx.emit!(ctx.icall(_rr_increment_field_rdata!, (ctx.fdtype(P), zero_like_rdata_type(_widen(Ti)), idxty),
                                mt, acc, idxval), ctx.fdtype(P))
        end
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
# `Core.tuple` — a tuple's rdata is a bare `Tuple`, not an `RData{NamedTuple}` wrapper, so the
# pullback is the same shape as the immutable `%new` arm minus the unwrap step. A tuple is
# immutable, so no comms items are needed: the rdata accumulator `Ref` is allocated generically by
# the pullback prologue, and the shadow tuple (when tracked) is only consumed on the forwards side
# via `sresolve` by a later `getfield`.
#
# All three methods return `nothing` when `tangent_type` collapses the tuple to `NoTangent` (e.g.
# `Tuple{Int,Symbol}`), so the pre-existing primal-replay fallbacks in `reverse_interp.jl` keep
# handling that case.
# ---------------------------------------------------------------------------

function builtin_rrule_comms(::Val{Core.tuple}, actual, Ti, ctx)
    ctx.tt(_widen(Ti)) === NoTangent && return nothing
    T = _widen(Ti)
    ok = T isa DataType && T <: Tuple && isconcretetype(T) &&
         !(!isempty(T.parameters) && isa(last(T.parameters), Core.TypeofVararg)) &&
         fieldcount(T) == length(actual)
    if !ok
        ctx.reason[] = "reverse mode `tuple` requires a concrete, non-vararg Tuple type with one " *
                       "field per operand, got $(T) for $(length(actual)) operand(s) at " *
                       "%$(ctx.ssa.id)"
        return false
    end
    FT = ctx.fdtype(T)
    if FT !== NoFData
        if !(FT isa DataType && FT <: Tuple)
            ctx.reason[] = "reverse mode `tuple` requires a concrete fdata type, got $(FT) at " *
                           "%$(ctx.ssa.id)"
            return false
        end
        for j in eachindex(actual)
            ctx.fdtype(fieldtype(T, j)) === NoFData && continue
            _bi_tracked(actual[j], ctx) && continue
            # Inactive: the fwds pass synthesises a fresh zero for this slot instead of aliasing.
            ctx.inactive(actual[j]) && continue
            ctx.reason[] = "reverse mode `tuple` operand $(j) (type $(fieldtype(T, j))) carries " *
                           "fdata but has no provenance traceable to a function argument at " *
                           "%$(ctx.ssa.id)"
            return false
        end
    end
    return Tuple{Any,Any}[]
end

function apply_builtin_rrule_fwds!(::Val{Core.tuple}, actual, Ti, ctx)
    ctx.tt(_widen(Ti)) === NoTangent && return nothing
    p = ctx.emit!(Expr(:call, _ctupleg, (ctx.presolve(a) for a in actual)...), Ti)
    shadow = nothing
    if _bi_tracked(ctx.ssa, ctx)
        T = _widen(Ti)
        # `ctx.sty`, not `ctx.fdtype`: an inactive operand's slot is declared `Inactive`, so the
        # shadow tuple's type is not the primal-derived one.
        FT = ctx.sty(ctx.ssa)
        shadow = ctx.emit!(Expr(:call, _ctupleg,
                     (begin
                          FTj = ctx.fdtype(fieldtype(T, j))
                          if FTj === NoFData
                              NoFData()
                          elseif ctx.inactive(actual[j])
                              # Nothing to synthesise: `Inactive` is zero-size and absorbs whatever
                              # is accumulated into it, so the slot costs neither an allocation nor
                              # a word of storage.
                              Inactive()
                          else
                              ctx.sresolve(actual[j])
                          end
                      end
                      for j in eachindex(actual))...), FT)
    end
    return p, shadow, Dict{Any,Any}()
end

function apply_builtin_rrule!(::Val{Core.tuple}, actual, Ti, ctx)
    ctx.tt(_widen(Ti)) === NoTangent && return nothing
    nores = ntuple(_ -> nothing, length(actual))
    ctx.rdtype(Ti) === NoRData && return nores
    T = _widen(Ti)
    acc = ctx.deref_and_zero!(Ti)
    RDataT = ctx.rdtype(T)
    real_acc = ctx.emit!(
        ctx.icall(_rr_realize_rdata, (zero_like_rdata_type(_widen(Ti)), Type{RDataT}), acc, RDataT),
        RDataT)
    contribs = Vector{Any}(undef, length(actual))
    for j in eachindex(actual)
        Fty = ctx.rdtype(fieldtype(T, j))
        contribs[j] = Fty === NoRData ? nothing : ctx.emit!(Expr(:call, _getfieldg, real_acc, j), Fty)
    end
    return Tuple(contribs)
end

# ---------------------------------------------------------------------------
# `Core.ifelse` — branchless select. In scope only when both branches share `Ti`'s own concrete,
# fdata-free type; an fdata-carrying `ifelse` (array/mutable-struct select) is a deliberate, located
# bail — soundly expressible via shadow aliasing, but not implemented here. Routes the accumulated
# rdata to whichever branch primally ran, zero to the other, with no block splitting.
# ---------------------------------------------------------------------------

function builtin_rrule_comms(::Val{Core.ifelse}, actual, Ti, ctx)
    ctx.tt(_widen(Ti)) === NoTangent && return nothing
    P = _widen(Ti)
    if !(isconcretetype(P) && ctx.optype(actual[2]) === P && ctx.optype(actual[3]) === P)
        ctx.reason[] = "reverse mode `ifelse` requires both branches to share the concrete result " *
                       "type $(P) at %$(ctx.ssa.id)"
        return false
    end
    if !can_produce_zero_rdata_from_type(P)
        ctx.reason[] = "reverse mode `ifelse` requires a type whose zero rdata is derivable from " *
                       "the type alone, got $(P) at %$(ctx.ssa.id)"
        return false
    end
    if ctx.fdtype(P) !== NoFData
        ctx.reason[] = "reverse mode `ifelse` does not support an fdata-carrying result ($(P)) at " *
                       "%$(ctx.ssa.id)"
        return false
    end
    ctx.rdtype(Ti) === NoRData && return Tuple{Any,Any}[]
    cond = actual[1]
    (isa(cond, Core.SSAValue) || isa(cond, Core.Argument)) || return Tuple{Any,Any}[]
    return Tuple{Any,Any}[((:primal, cond), ctx.optype(cond))]
end

function apply_builtin_rrule_fwds!(::Val{Core.ifelse}, actual, Ti, ctx)
    ctx.tt(_widen(Ti)) === NoTangent && return nothing
    p = ctx.emit!(Expr(:call, _ifelseg, ctx.presolve(actual[1]),
                       ctx.presolve(actual[2]), ctx.presolve(actual[3])), Ti)
    return p, nothing, Dict{Any,Any}()
end

function apply_builtin_rrule!(::Val{Core.ifelse}, actual, Ti, ctx)
    ctx.tt(_widen(Ti)) === NoTangent && return nothing
    ctx.rdtype(Ti) === NoRData && return (nothing, nothing, nothing)
    P = _widen(Ti)
    RT = zero_like_rdata_type(P)
    cnd = ctx.fetch_primal(actual[1])
    acc = ctx.deref_and_zero!(Ti)
    z = ctx.emit!(zero_like_rdata_from_type(P), RT)
    ca = ctx.emit!(Expr(:call, _ifelseg, cnd, acc, z), RT)
    cb = ctx.emit!(Expr(:call, _ifelseg, cnd, z, acc), RT)
    return (nothing, ca, cb)
end

# ---------------------------------------------------------------------------
# `Core.memorynew` — array allocation step 1: allocates a fresh, uninitialized `Memory{P}`. Its own
# rdata is always `NoRData` (a handle, not a differentiable value); the shadow allocates a
# same-length, uninitialized `Memory{tangent_type(P)}`, safe because every element the primal ever
# reads was necessarily written first by an already-handled `memoryrefset!`. The length is
# structural, so it's `presolve`d, never `sresolve`d, in both primal and shadow calls.
# ---------------------------------------------------------------------------
builtin_rrule_comms(::Val{Core.memorynew}, actual, Ti, ctx) = Tuple{Any,Any}[]
function apply_builtin_rrule_fwds!(::Val{Core.memorynew}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, Core.memorynew, ctx.presolve(actual[1]),
                       (ctx.presolve(a) for a in actual[2:end])...), Ti)
    shadow = nothing
    if _bi_tracked(ctx.ssa, ctx)
        TT = ctx.tt(_widen(Ti))
        shadow = ctx.emit!(Expr(:call, Core.memorynew, TT,
                           (ctx.presolve(a) for a in actual[2:end])...), TT)
    end
    return p, shadow, Dict{Any,Any}()
end
apply_builtin_rrule!(::Val{Core.memorynew}, actual, Ti, ctx) = ntuple(_ -> nothing, length(actual))

# ---------------------------------------------------------------------------
# `Base.memoryrefnew` — its own rdata is always `NoRData` (a handle, not differentiable); the shadow
# chain it participates in is handled by `_fdata_tracked`/the fwds pass directly, needing no comms.
# Registered here only so the dispatch layer has an explicit entry rather than silently relying on
# the generic no-tangent fallback.
#
# SAFETY: the 3-arg offsetting form's trailing boundscheck flag is NOT mirrored from the primal,
# always forced `true` on the shadow ref, so a mismatched-length tangent array raises a catchable
# `BoundsError` instead of corrupting memory via an unchecked out-of-bounds `MemoryRef`.
# ---------------------------------------------------------------------------
builtin_rrule_comms(::Val{Base.memoryrefnew}, actual, Ti, ctx) = Tuple{Any,Any}[]
function apply_builtin_rrule_fwds!(::Val{Base.memoryrefnew}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, Base.memoryrefnew, (ctx.presolve(a) for a in actual)...), Ti)
    shadow = nothing
    if _bi_tracked(ctx.ssa, ctx)
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
        shadow = ctx.emit!(Expr(:call, Base.memoryrefnew, shadow_args...), ctx.fdtype(Ti))
    end
    return p, shadow, Dict{Any,Any}()
end
apply_builtin_rrule!(::Val{Base.memoryrefnew}, actual, Ti, ctx) = ntuple(_ -> nothing, length(actual))

# ---------------------------------------------------------------------------
# `Base.memoryrefget` — two cases, by the result's own type:
#   * a bits (rdata-carrying) result needs its shadow `MemoryRef` handle communicated forward
#     (`:shadow_ref`), but only if traceable to a tracked argument; the pullback increments the
#     shadow element in place through that handle, no `Ref` accumulator needed.
#   * an fdata-carrying result (a nested array) needs no comms: its shadow is resolved directly in
#     the fwds pass and consumed by `sresolve` wherever it's used — the aliasing itself is the
#     backward flow, nothing to route through the tape.
# ---------------------------------------------------------------------------
function builtin_rrule_comms(::Val{Base.memoryrefget}, actual, Ti, ctx)
    ctx.rdtype(Ti) === NoRData && return Tuple{Any,Any}[]
    ref_node = actual[1]
    if !(isa(ref_node, Core.SSAValue) && ref_node.id <= length(ctx.tracked) && ctx.tracked[ref_node.id])
        ctx.reason[] = "array read has no differentiable provenance traceable to a function " *
                       "argument at %$(ctx.ssa.id)"
        return false
    end
    # A statically-derivable `MemoryRef` is re-derived in the pullback, so skip pushing its shadow
    # handle, keeping the comms tuple empty for static-index reads. A dynamic (loop) index still
    # needs its own value on the tape, as a plain `Int` rather than the 16-byte `MemoryRef`.
    d = ctx.static_ref(ref_node)
    d === nothing && return Tuple{Any,Any}[((:shadow_ref, ref_node), ctx.fdtype(ctx.optype(ref_node)))]
    idx = d[2]
    return isa(idx, Core.SSAValue) ? Tuple{Any,Any}[((:primal, idx), Int)] : Tuple{Any,Any}[]
end

function apply_builtin_rrule_fwds!(::Val{Base.memoryrefget}, actual, Ti, ctx)
    # For a bits result, only the shadow `MemoryRef` handle is communicated forward (registered
    # above), never the shadow value itself. For an fdata-carrying (nested-array) result, mirror the
    # identical `memoryrefget` onto the shadow ref: the shadow of a nested element is the
    # corresponding element of the shadow array.
    p = ctx.emit!(Expr(:call, Base.memoryrefget, (ctx.presolve(a) for a in actual)...), Ti)
    shadow = nothing
    if _bi_tracked(ctx.ssa, ctx)
        # The shadow array stores elements at their own tangent type, not at `fdtype(Ti)`.
        TT = ctx.tt(_widen(Ti))
        cur = ctx.emit!(Expr(:call, Base.memoryrefget, ctx.sresolve(actual[1]),
                             (ctx.presolve(a) for a in actual[2:end])...), TT)
        shadow = _emit_fdata!(ctx, TT, ctx.fdtype(Ti), cur)
    end
    return p, shadow, Dict{Any,Any}()
end

function apply_builtin_rrule!(::Val{Base.memoryrefget}, actual, Ti, ctx)
    nores = ntuple(_ -> nothing, length(actual))
    ctx.rdtype(Ti) === NoRData && return nores
    acc = ctx.deref_and_zero!(Ti)
    shadow_ref = ctx.fetch_shadow(actual[1])
    # Shadow elements are stored at their tangent type, which equals `Ti`'s rdata type only for a
    # bits scalar. `acc` may be `ZeroRData` when `Ti` isn't concrete.
    TT = ctx.tt(_widen(Ti))
    RT = zero_like_rdata_type(_widen(Ti))
    cur = ctx.emit!(Expr(:call, Base.memoryrefget, shadow_ref, QuoteNode(:not_atomic), false), TT)
    new = _emit_increment_rdata!(ctx, TT, RT, cur, acc)
    ctx.emit!(Expr(:call, Base.memoryrefset!, shadow_ref, new, QuoteNode(:not_atomic), false), TT)
    return nores
end

# ---------------------------------------------------------------------------
# `Core.setfield!` — adapted from Mooncake's `lsetfield_rrule` to Differ's IR-emission pullback:
# everything the pullback needs (old field value, old tangent, object's fdata/primal handles) must
# be either resolved from an existing comms item or explicitly saved by the fwds emission, since the
# pullback carrier shares no state with the fwds carrier except the tape.
#
# Two field shapes, same comms/pullback shape (only the fwds emission of the new field tangent
# differs, in `apply_builtin_rrule_fwds!` below):
#   * pure rdata (scalar or immutable-scalar struct) — new tangent is a fresh zero; the assigned
#     value's gradient flows back purely through the pullback's rdata return.
#   * fdata-carrying (array- or mutable-struct-valued field) — new tangent *aliases* the assigned
#     value's own shadow, so a later in-place accumulation reached through this field already lands
#     in the assigned value's own gradient; the pullback's rdata return then carries only whatever
#     rdata the field separately accumulated (`NoRData` for a pure array field).
# Requires the assigned value's own fdata to be resolvable at the call site — an `Argument` or a
# tracked `SSAValue` — same provenance guard as the object's own, below.
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
        # Dynamic (non-literal) write index unsupported: unlike a read, `set_tangent_field!` needs a
        # statically-known field to place the new value into the right slot type, and this file's
        # fwds/pullback emission for `setfield!` embeds the field index as a compile-time constant.
        ctx.reason[] = "reverse mode `setfield!` does not support a dynamic (non-literal) field " *
                       "index ($(P)) at %$(ctx.ssa.id)"
        return false
    end
    fname = _bi_fieldname(actual[2])
    if ctx.fdtype(ctx.tt(fieldtype(P, fname))) !== NoFData
        val_node = actual[3]
        # Inactive: the fwds pass zeroes the field's shadow instead, severing the gradient chain.
        if !_bi_tracked(val_node, ctx) && !ctx.inactive(val_node)
            ctx.reason[] = "reverse mode `setfield!` of a field whose tangent carries fdata " *
                           "($(fieldtype(P, fname))) requires the assigned value's own fdata to be " *
                           "traceable to a function argument at %$(ctx.ssa.id)"
            return false
        end
    end
    if !_bi_tracked(obj, ctx)
        ctx.reason[] = "mutable-struct `setfield!` has no differentiable provenance traceable to a " *
                       "function argument at %$(ctx.ssa.id) (object type $(P))"
        return false
    end
    return Tuple{Any,Any}[((:fshadow, obj), ctx.fdtype(P)), ((:primal, obj), P),
                          ((:old_primal, ctx.ssa), Ti), ((:old_tangent, ctx.ssa), ctx.tt(_widen(Ti)))]
end

function apply_builtin_rrule_fwds!(::Val{Core.setfield!}, actual, Ti, ctx)
    obj, name_node, val_node = actual[1], actual[2], actual[3]
    P = ctx.optype(obj)
    mt = ctx.sresolve(obj)
    TF = ctx.tt(_widen(Ti))
    old_primal = ctx.emit!(Expr(:call, _getfieldg, ctx.presolve(obj), name_node), Ti)
    fname = _bi_fieldname(name_node)
    fieldidx = fname isa Symbol ? findfirst(==(fname), fieldnames(P)) : fname
    FTi = ctx.fdtype(Ti)
    slot = _tangent_field_slot(ctx.tt, P, name_node)
    if slot !== nothing
        # Emit the field read/write directly instead of the `_rr_*` `:invoke` barriers.
        # New field tangent is `zero_tangent(p, f)`; for a pure-rdata IEEEFloat field this is `zero(TF)`.
        old_tangent = _emit_gtf!(ctx, mt, slot)
        if FTi === NoFData && TF <: IEEEFloat
            zt = ctx.emit!(zero(TF), TF)
        else
            # Inactive forces the fresh-zero branch; `FTarg` must then be `NoFData`, not `FTi` —
            # it declares `fdata_val`'s actual runtime type. `normalize_shadow!`, not a bare
            # `sresolve`: the assigned value's own shadow can be a mixed-activity tuple (narrower
            # than `FTi`), which must be widened to `FTi` before it reaches this `:invoke`.
            inact = ctx.inactive(val_node)
            fdata_val = (FTi === NoFData || inact) ? NoFData() : ctx.normalize_shadow!(val_node, FTi)
            FTarg = (FTi === NoFData || inact) ? NoFData : FTi
            zt = ctx.icall!(_rr_zero_tangent2, TF, (Ti, FTarg), ctx.presolve(val_node), fdata_val)
        end
        _emit_stf!(ctx, mt, slot, zt)
    else
        old_tangent = ctx.icall!(_rr_get_tangent_field, TF, (ctx.fdtype(P), Int), mt, fieldidx)
        # `zero_tangent(p, f)` embeds `f` (the assigned value's own fdata) directly rather than
        # fabricating a fresh zero when the field carries fdata — that embedding is the alias that
        # makes later in-place accumulation into this field flow into the assigned value's own
        # shadow. `f = NoFData()` for a pure-rdata field collapses to the original fresh-zero
        # behavior — also what an inactive `val_node` forces, since it has no real shadow to embed.
        # `normalize_shadow!`, not a bare `sresolve` — see the same note above.
        inact = ctx.inactive(val_node)
        fdata_val = (FTi === NoFData || inact) ? NoFData() : ctx.normalize_shadow!(val_node, FTi)
        FTarg = (FTi === NoFData || inact) ? NoFData : FTi
        zt = ctx.icall!(_rr_zero_tangent2, TF, (Ti, FTarg), ctx.presolve(val_node), fdata_val)
        ctx.icall!(_rr_set_tangent_field!, TF, (ctx.fdtype(P), Int, TF), mt, fieldidx, zt)
    end
    p = ctx.emit!(Expr(:call, _setfieldg, ctx.presolve(obj), name_node, ctx.presolve(val_node)), Ti)
    saved = Dict{Any,Any}((:old_primal, ctx.ssa) => old_primal, (:old_tangent, ctx.ssa) => old_tangent)
    return p, nothing, saved
end

# Unaffected by the fdata-aliasing branch above: `rdata(cur_tangent)` already reads whatever rdata
# the field separately accumulated (`NoRData` for a pure array field, since that contribution went
# straight into the aliased shadow), and restoring the slot to `old_primal`/`old_tangent` is the
# same operation either way.
function apply_builtin_rrule!(::Val{Core.setfield!}, actual, Ti, ctx)
    obj, name_node = actual[1], actual[2]
    P = ctx.optype(obj)
    mt = ctx.fetch_shadow(obj)
    primal_obj = ctx.fetch_primal(obj)
    old_primal = ctx.fetch_saved((:old_primal, ctx.ssa))
    old_tangent = ctx.fetch_saved((:old_tangent, ctx.ssa))
    fname = _bi_fieldname(name_node)
    fieldidx = fname isa Symbol ? findfirst(==(fname), fieldnames(P)) : fname
    TF = ctx.tt(_widen(Ti))
    # `zero_like_rdata_type`, not `rdtype`: `acc` may be `ZeroRData` when `Ti` isn't concrete (e.g.
    # an abstractly-typed field). `cur_rdata` is always a real value regardless, so declaring it at
    # the same (possibly wider) `RT` is harmless.
    RT = zero_like_rdata_type(_widen(Ti))
    acc = ctx.deref_and_zero!(Ti)
    slot = _tangent_field_slot(ctx.tt, P, name_node)
    if slot !== nothing
        # Emit the read, rdata extract, increment, and restore directly, avoiding the `_rr_*` barriers.
        cur_tangent = _emit_gtf!(ctx, mt, slot)
    else
        cur_tangent = ctx.emit!(ctx.icall(_rr_get_tangent_field, (ctx.fdtype(P), Int), mt, fieldidx), TF)
    end
    cur_rdata = _emit_rdata!(ctx, TF, RT, cur_tangent)
    new_dx = ctx.emit!(ctx.icall(increment!!, (RT, RT), acc, cur_rdata), RT)
    ctx.emit!(Expr(:call, _setfieldg, primal_obj, name_node, old_primal), Any)
    if slot !== nothing
        _emit_stf!(ctx, mt, slot, old_tangent)
    else
        ctx.emit!(ctx.icall(_rr_set_tangent_field!, (ctx.fdtype(P), Int, TF), mt, fieldidx, old_tangent), TF)
    end
    return ntuple(j -> j == 3 ? new_dx : nothing, length(actual))
end

# ---------------------------------------------------------------------------
# `Base.memoryrefset!` — the array analogue of `Core.setfield!` above: the "object" is a `MemoryRef`,
# its "field" the pointed-to element, saved/restored the same way. Same two element shapes as
# `setfield!`'s field shapes (pure rdata: fresh-zero new tangent; fdata-carrying: new tangent
# aliases the assigned value's own shadow), same provenance requirement on the assigned value.
# ---------------------------------------------------------------------------
function builtin_rrule_comms(::Val{Base.memoryrefset!}, actual, Ti, ctx)
    length(actual) < 2 && return false
    elt = Ti
    ref_node, val_node = actual[1], actual[2]
    # No tangent at all (e.g. an `Int` element): no shadow traffic, no provenance requirement.
    notangent = ctx.tt(_widen(elt)) === NoTangent
    if !notangent
        if ctx.fdtype(elt) !== NoFData
            # Inactive: the fwds pass zeroes the element's shadow instead.
            if !_bi_tracked(val_node, ctx) && !ctx.inactive(val_node)
                ctx.reason[] = "reverse mode `memoryrefset!` of a non-bits element ($(elt)) requires the " *
                               "assigned value's own fdata to be traceable to a function argument at " *
                               "%$(ctx.ssa.id)"
                return false
            end
        elseif ctx.rdtype(elt) !== ctx.tt(_widen(elt))
            ctx.reason[] = "reverse mode does not support array mutation of element type ($(elt)) at " *
                           "%$(ctx.ssa.id)"
            return false
        end
        if !(isa(ref_node, Core.SSAValue) && ref_node.id <= length(ctx.tracked) && ctx.tracked[ref_node.id])
            ctx.reason[] = "array write has no differentiable provenance traceable to a function " *
                           "argument at %$(ctx.ssa.id)"
            return false
        end
    end
    # A statically-derivable `MemoryRef` is re-derived in the pullback, so neither its shadow nor
    # primal handle is pushed — only the overwritten values remain. A dynamic (loop) index still
    # needs its own value on the tape; it belongs with the shadow side (never optional), so it's
    # declared unconditionally here rather than inside the `bulk_saved` branch below.
    d = ctx.static_ref(ref_node)
    derivable = d !== nothing
    items = Tuple{Any,Any}[]
    if !derivable
        if !notangent
            push!(items, ((:shadow_ref, ref_node), ctx.fdtype(ctx.optype(ref_node))))
        end
    elseif isa(d[2], Core.SSAValue)
        push!(items, ((:primal, d[2]), Int))
    end
    # The primal side — the element this store overwrote, and the `MemoryRef` to write it back through
    # — is needed only to restore the primal, and only when that restore is done one element at a
    # time. When this array is bulk-saved (`_bulk_save_args`) the whole thing is copied back at the
    # end of the pullback instead, so neither is recorded. The shadow side is not optional either way:
    # the pullback reads and rewrites the shadow slot as it goes, so its old value is genuinely live
    # during the reverse sweep (see the `old_tangent` note in `apply_builtin_rrule!`).
    if !ctx.bulk_saved(ref_node)
        if !derivable
            push!(items, ((:primal, ref_node), ctx.optype(ref_node)))
        end
        push!(items, ((:old_primal, ctx.ssa), _rr_saved_slot_type(elt)))
    end
    if !notangent
        push!(items, ((:old_tangent, ctx.ssa), _rr_saved_slot_type(ctx.tt(_widen(elt)))))
    end
    return items
end

function apply_builtin_rrule_fwds!(::Val{Base.memoryrefset!}, actual, Ti, ctx)
    ref_node, val_node = actual[1], actual[2]
    rest = @view actual[3:end]
    pref = ctx.presolve(ref_node)
    TT = ctx.tt(_widen(Ti))
    bulk = ctx.bulk_saved(ref_node)
    # The old primal is read only to put it back one element at a time; a bulk-saved array already has
    # its pre-call contents copied aside in the prologue, so this load is pure waste there.
    PRT = ctx.optype(ref_node)
    old_primal = bulk ? nothing :
        (isbitstype(Ti) ?
            ctx.emit!(Expr(:call, Base.memoryrefget, pref, (ctx.presolve(a) for a in rest)...), Ti) :
            ctx.icall!(_rr_memref_get_or_nothing, Union{Nothing,Ti}, (PRT,), pref))
    if TT === NoTangent
        p = ctx.emit!(Expr(:call, Base.memoryrefset!, pref, ctx.presolve(val_node), (ctx.presolve(a) for a in rest)...), Ti)
        saved = Dict{Any,Any}()
        bulk || (saved[(:old_primal, ctx.ssa)] = old_primal)
        return p, nothing, saved
    end
    sref = ctx.sresolve(ref_node)
    SRT = ctx.fdtype(PRT)
    old_tangent = isbitstype(TT) ?
        ctx.emit!(Expr(:call, Base.memoryrefget, sref, (ctx.presolve(a) for a in rest)...), TT) :
        ctx.icall!(_rr_memref_get_or_nothing, Union{Nothing,TT}, (SRT,), sref)
    # `zero_tangent(p, f)` embeds `f` (the assigned value's own fdata) directly rather than
    # fabricating a fresh zero when the element carries fdata — the alias that makes later in-place
    # accumulation into this element flow into the assigned value's own shadow. `f = NoFData()` for
    # a pure-rdata element collapses to the original fresh-zero behavior — also what an inactive
    # `val_node` forces, since it has no real shadow to embed. `normalize_shadow!`, not a bare
    # `sresolve`: the assigned value's own shadow can be a mixed-activity tuple (narrower than
    # `FTi`), which must be widened to `FTi` before it reaches this `:invoke`.
    FTi = ctx.fdtype(Ti)
    inact = ctx.inactive(val_node)
    fdata_val = (FTi === NoFData || inact) ? NoFData() : ctx.normalize_shadow!(val_node, FTi)
    FTarg = (FTi === NoFData || inact) ? NoFData : FTi
    zt = ctx.icall!(_rr_zero_tangent2, TT, (Ti, FTarg), ctx.presolve(val_node), fdata_val)
    ctx.emit!(Expr(:call, Base.memoryrefset!, sref, zt, (ctx.presolve(a) for a in rest)...), TT)
    p = ctx.emit!(Expr(:call, Base.memoryrefset!, pref, ctx.presolve(val_node), (ctx.presolve(a) for a in rest)...), Ti)
    saved = Dict{Any,Any}((:old_tangent, ctx.ssa) => old_tangent)
    bulk || (saved[(:old_primal, ctx.ssa)] = old_primal)
    return p, nothing, saved
end

# Restore a slot, matching whichever read shape `apply_builtin_rrule_fwds!` used. `RT` is `ref`'s
# declared type, primal or shadow.
function _rr_emit_slot_restore!(ctx, @nospecialize(RT), ref, val, @nospecialize(T))
    if isbitstype(T)
        ctx.emit!(Expr(:call, Base.memoryrefset!, ref, val, QuoteNode(:not_atomic), false), T)
    else
        ctx.emit!(ctx.icall(_rr_memref_restore!, (RT, Union{Nothing,T}), ref, val), Nothing)
    end
    return nothing
end

# Unaffected by the fdata-aliasing branch above: `rdata(cur_tangent)` already reads whatever rdata
# the element separately accumulated (`NoRData` for a pure array element), and restoring the slot
# to `old_primal`/`old_tangent` is the same operation either way.
function apply_builtin_rrule!(::Val{Base.memoryrefset!}, actual, Ti, ctx)
    ref_node = actual[1]
    bulk = ctx.bulk_saved(ref_node)
    nores = ntuple(_ -> nothing, length(actual))
    TT = ctx.tt(_widen(Ti))
    PRT = ctx.optype(ref_node)
    if TT === NoTangent
        if !bulk
            pref = ctx.fetch_primal(ref_node)
            old_primal = ctx.fetch_saved((:old_primal, ctx.ssa))
            _rr_emit_slot_restore!(ctx, PRT, pref, old_primal, Ti)
        end
        return nores
    end
    sref = ctx.fetch_shadow(ref_node)
    old_tangent = ctx.fetch_saved((:old_tangent, ctx.ssa))
    # `zero_like_rdata_type`, not `rdtype`: `acc` may be `ZeroRData` when `Ti` isn't concrete (e.g.
    # an array with an abstract eltype). `cur_rdata` is always a real value regardless, so declaring
    # it at the same (possibly wider) `RT` is harmless.
    RT = zero_like_rdata_type(_widen(Ti))
    acc = ctx.deref_and_zero!(Ti)
    cur_tangent = ctx.emit!(Expr(:call, Base.memoryrefget, sref, QuoteNode(:not_atomic), false), TT)
    cur_rdata = _emit_rdata!(ctx, TT, RT, cur_tangent)
    new_dx = ctx.emit!(ctx.icall(increment!!, (RT, RT), acc, cur_rdata), RT)
    _rr_emit_slot_restore!(ctx, ctx.fdtype(PRT), sref, old_tangent, TT)
    # Primal restore, unless the whole array is copied back at the end of the pullback instead.
    if !bulk
        pref = ctx.fetch_primal(ref_node)
        old_primal = ctx.fetch_saved((:old_primal, ctx.ssa))
        _rr_emit_slot_restore!(ctx, PRT, pref, old_primal, Ti)
    end
    return ntuple(j -> j == 2 ? new_dx : nothing, length(actual))
end
