# Small, mode-agnostic IR-inspection helpers used by both AD engines (`DifferForwards`/
# `DifferReverse`). Homed here rather than in either package since they carry no AD-mode opinion
# of their own.

# Resolve a `GlobalRef` to its bound value, distinguishing "resolved" from "undefined/unresolvable".
# `world` is mandatory, never the ambient task world — this runs inside code compiled at a
# specific inference world that can predate a user's own definitions.
function _globalref_val(gr::GlobalRef, world::UInt)
    try
        return (true, Base.getglobalref(gr, world))
    catch
        return (false, nothing)
    end
end

# Whether `gr`'s binding is a defined constant — i.e. whether its value may be frozen into the IR —
# asked at `world`, never the ambient task world. Same check `verify_ir` performs on a
# value-position `GlobalRef`.
function _globalref_isconst(gr::GlobalRef, world::UInt)
    try
        bpart = Base.lookup_binding_partition(world, gr)
        return Base.is_defined_const_binding(Base.binding_kind(bpart))
    catch
        return false
    end
end

# Resolve an IR operand to its statically-known value: a `GlobalRef`'s bound value (at `world`,
# not the ambient task world), a `QuoteNode`'s literal, `nothing` for an `SSAValue`/`Argument`
# (not a name to look up — resolve those via `presolve`/`_optype` instead), or the node itself
# for a bare literal.
_calleeval(@nospecialize(x), world::UInt) =
    isa(x, GlobalRef) ? (try Base.getglobalref(x, world) catch; nothing end) :
    isa(x, QuoteNode) ? x.value :
    isa(x, Core.SSAValue) || isa(x, Core.Argument) ? nothing :
    x

# Re-embed a value resolved by `_calleeval`/`_globalref_val` as an IR operand. Codegen reads a bare
# `Symbol` as a global load in the *emitting* module (so a field-name literal recovered from a
# `QuoteNode` becomes an `UndefVarError`), and the other node types below as references rather than
# data; a `QuoteNode` wrapper makes them literals again. Only for values, never for operand nodes —
# an `SSAValue` produced by `presolve` must stay a reference.
_ir_literal(@nospecialize x) =
    (isa(x, Symbol) || isa(x, Expr) || isa(x, QuoteNode) || isa(x, GlobalRef) ||
     isa(x, Core.SSAValue) || isa(x, Core.Argument) || isa(x, Core.SlotNumber) ||
     isa(x, Core.NewvarNode) || isa(x, Core.GotoNode) || isa(x, Core.GotoIfNot) ||
     isa(x, Core.ReturnNode) || isa(x, Core.EnterNode) || isa(x, Core.PhiNode) ||
     isa(x, Core.PhiCNode) || isa(x, Core.PiNode) || isa(x, Core.UpsilonNode) ||
     isa(x, LineNumberNode)) ? QuoteNode(x) : x

# The primal IR's own declared type for operand `x` — taken directly from `pir`, so exact rather
# than guessed; the fallback (`typeof(x)`) only covers genuine literal constants.
_optype(pir, @nospecialize x) = isa(x, Core.SSAValue) ? pir.stmts[x.id][:type] :
                                isa(x, Core.Argument) ? pir.argtypes[x.n] : typeof(x)

# Statement `i`'s type as a bare `Type`. `stmts[i][:type]` is a lattice element, illegal in
# type-parameter position and a false fact once stamped on a shadow/tangent instruction.
_stype(stmts, i::Int) = _widen(stmts[i][:type])

# `_stype`, but recovers an `:invoke`'s real result type when inference widened it to `Any` because
# the caller discards it (`call_result_unused`). The callee's own `rettype` is not widened.
function _stype_invoke(stmts, i::Int)
    T = _stype(stmts, i)
    T === Any || return T
    s = stmts[i][:stmt]
    (isa(s, Expr) && s.head === :invoke) || return T
    ci = s.args[1]
    isa(ci, Core.CodeInstance) || return T
    return ci.rettype
end

# The type of the *value* an operand denotes, as a bare (widened) `Type`: `_optype` alone reports
# `GlobalRef`/`QuoteNode` as the node's own type rather than the value it denotes, and can return a
# lattice element (`Core.Const`/`PartialStruct`) for an `SSAValue`/`Argument` rather than a bare
# `Type`. Resolves a `GlobalRef` the same const-gated, world-parameterized way `_globalref_isconst`/
# `_globalref_val` do; a non-`const` binding is only knowable at run time, hence `Any`.
function _optype_w(pir, world::UInt, @nospecialize x)
    if isa(x, GlobalRef)
        _globalref_isconst(x, world) || return Any
        ok, v = _globalref_val(x, world)
        return ok ? Core.Typeof(v) : Any
    end
    isa(x, QuoteNode) && return Core.Typeof(x.value)
    return _widen(_optype(pir, x))
end

# Compact, single-line rendering of a primal IR statement for a bail `reason` message, so the error
# tells the user *what* IR construct was unsupported rather than just its kind. Kept defensive:
# `show` on a stray node must never mask the real bail.
function _stmt_str(@nospecialize s)
    str = try sprint(show, s) catch; string(s) end
    str = replace(str, r"\s+" => " ")
    length(str) > 200 ? string(first(str, 197), "...") : str
end

# True for a `getfield`/`setfield!` name/index operand that's already a compile-time literal
# (`QuoteNode(sym)`, a bare `Symbol`, or a bare `Int`) — needs no resolution. Anything else
# (`SSAValue`/`Argument`) is a dynamic, runtime-computed index.
_bi_literal_index(@nospecialize(x)) = isa(x, QuoteNode) || isa(x, Symbol) || isa(x, Int)

# The common tangent type shared by every field of concrete type `P`, or `nothing` if `P` isn't
# concrete or its fields don't all share one tangent type. Lets a dynamic (runtime-computed)
# field index be handled uniformly for a homogeneous aggregate.
#
# Takes the caller's `tangent_type` funnel rather than calling `tangent_type` directly: both
# callers run inside a `@generated` generator, where plain dispatch is pinned to the generator's
# world and would miss a method owned by a later-loaded package (Contextual's `at_world`
# contract). DifferCore stays independent of Contextual by taking the funnel as an argument.
function _bi_homog_tangent_type(tt, P)
    (P isa DataType && isconcretetype(P)) || return nothing
    nf = fieldcount(P)
    nf == 0 && return nothing
    tt1 = tt(fieldtype(P, 1))
    for j in 2:nf
        tt(fieldtype(P, j)) === tt1 || return nothing
    end
    return tt1
end

# For a general-struct field access whose object primal type is `Pobj` and whose field is named by
# the compile-time literal `name`: return `(NT, i)` — the object's tangent-backing `NamedTuple`
# type and the 1-based field index into it — or `nothing` when the direct-emission preconditions
# don't hold (non-concrete primal, non-`Tangent`/`MutableTangent` tangent, unknown/out-of-range
# field, or a `PossiblyUninitTangent` slot needing the generic unwrap/wrap helper instead).
# Takes the caller's `tangent_type` funnel, for the reason on `_bi_homog_tangent_type` above.
function _tangent_field_slot(tt, @nospecialize(Pobj), @nospecialize(name))
    (Pobj isa DataType && isconcretetype(Pobj)) || return nothing
    Tobj = tt(Pobj)
    (Tobj isa DataType && Tobj <: Union{Tangent,MutableTangent}) || return nothing
    NT = fields_type(Tobj)
    (NT isa DataType && isconcretetype(NT)) || return nothing
    sym = name isa QuoteNode ? name.value : name
    i = sym isa Int ? sym : findfirst(==(sym), fieldnames(Pobj))
    (i isa Int && 1 <= i <= fieldcount(NT)) || return nothing
    fieldtype(NT, i) <: PossiblyUninitTangent && return nothing
    return (NT, i)
end

# Like `_tangent_field_slot`, but serves a `PossiblyUninitTangent` slot too: the caller gets the
# slot as stored (the wrapper itself), for save/restore paths that must not unwrap an
# uninitialised one.
function _raw_tangent_slot(tt, @nospecialize(Pobj), @nospecialize(name))
    (Pobj isa DataType && isconcretetype(Pobj)) || return nothing
    Tobj = tt(Pobj)
    (Tobj isa DataType && Tobj <: Union{Tangent,MutableTangent}) || return nothing
    NT = fields_type(Tobj)
    (NT isa DataType && isconcretetype(NT)) || return nothing
    sym = name isa QuoteNode ? name.value : name
    i = sym isa Int ? sym : findfirst(==(sym), fieldnames(Pobj))
    (i isa Int && 1 <= i <= fieldcount(NT)) || return nothing
    return (NT, i)
end

# Widen a lattice element (`Core.Const`/`Core.PartialStruct`/…) to a real `Type`, or pass a real
# `Type` through unchanged.
_widen(@nospecialize T) = T isa Type ? T : CC.widenconst(T)

# `GlobalRef` constants for the builtins both engines special-case as raw `:call`/`:invoke` targets
# when reconstructing IR. Only the ones both packages actually need are here — `DifferForwards`'s
# own `builtins.jl` has a larger set including ones only forward mode uses.
const _getfieldg = GlobalRef(Core, :getfield)
const _setfieldg = GlobalRef(Core, :setfield!)
const _ctupleg   = GlobalRef(Core, :tuple)
const _ifelseg   = GlobalRef(Core, :ifelse)

# `:foreigncall` IR-inspection helpers, shared by DifferForwards/DifferReverse via a `ctx` NamedTuple
# (pstmt/calleeval/optype/tt/reason).
#
# Layout (Julia 1.13, `Compiler/src/tfuncs.jl` FOREIGNCALL_ARG_START):
#   Expr(:foreigncall, name, RT, ATs::SimpleVector, nreq, cconv, args..., roots...)
#   #                   1     2    3                 4     5      6:5+length(ATs)   rest
# name: `Expr(:tuple, QuoteNode(sym))` (+ optional lib), or SSAValue/Argument for a runtime pointer
# (rejected by `_fc_parse` — not dispatchable).

# Parts of a foreigncall stmt, or nothing if target isn't a literal symbol. args/roots split at
# `5 + length(ATs)` — ATs' length is the value-operand count vs. trailing GC roots.
function _fc_parse(s::Expr)
    length(s.args) >= 5 || return nothing
    nm = s.args[1]
    name = nm; lib = nothing
    if isa(nm, Expr) && nm.head === :tuple
        length(nm.args) in (1, 2) || return nothing
        name = nm.args[1]
        length(nm.args) == 2 && (lib = nm.args[2])
    end
    isa(name, QuoteNode) && (name = name.value)
    isa(name, String) && (name = Symbol(name))
    isa(name, Symbol) || return nothing
    ATs = s.args[3]
    isa(ATs, Core.SimpleVector) || return nothing
    nval = 5 + length(ATs)
    length(s.args) >= nval || return nothing
    return (name = name, lib = lib, name_node = nm, RT = s.args[2], ATs = ATs,
            nreq = s.args[4], cconv = s.args[5],
            args = s.args[6:nval], roots = s.args[(nval + 1):end])
end

# Rebuilds a foreigncall stmt with substituted operands. name_node is copied, not shared: it's a
# mutable Expr owned by the cached primal IRCode, reused across dualizations.
_fc_stmt(fc, args, roots) =
    Expr(:foreigncall,
         isa(fc.name_node, Expr) ? Expr(fc.name_node.head, fc.name_node.args...) : fc.name_node,
         fc.RT, fc.ATs, fc.nreq, fc.cconv, args..., roots...)

# Walks primal IR from a pointer back to its Memory/MemoryRef origin; `(P, ref_operand)` or nothing.
# Only PiNode/Ptr->Ptr bitcast are followed — add_ptr/sub_ptr end the walk since an offset breaks
# the extent check's relation to the ref. `ctx.optype` uses primal numbering, valid post-interleave.
function _fc_ptr_origin(@nospecialize(x), ctx, depth::Int = 0)
    depth > 8 && return nothing
    isa(x, Core.SSAValue) || return nothing
    s = ctx.pstmt(x)
    if isa(s, Core.PiNode)
        return _fc_ptr_origin(s.val, ctx, depth + 1)
    elseif isa(s, Expr) && s.head === :call && length(s.args) >= 2
        f = ctx.calleeval(s.args[1])
        if f === Core.Intrinsics.bitcast && length(s.args) == 3
            Pin = ctx.optype(s.args[3])
            (Pin isa DataType && Pin <: Ptr) || return nothing
            return _fc_ptr_origin(s.args[3], ctx, depth + 1)
        elseif f === Core.getfield && length(s.args) >= 3
            Pobj = ctx.optype(s.args[2])
            # isa(DataType) first: optype may return Const/PartialStruct, where <: throws TypeError.
            (Pobj isa DataType && (Pobj <: MemoryRef || Pobj <: Memory)) || return nothing
            nm = s.args[3]
            isa(nm, QuoteNode) && (nm = nm.value)
            (nm === :ptr_or_offset || nm === :ptr) || return nothing
            P = eltype(Pobj)
            return P isa Type ? (P, s.args[2]) : nothing
        end
    end
    return nothing
end

# Whether a byte count means the same for P's tangent storage as for P — keyed on the provenance
# element type, not the pointer's own Ptr{…} (opposite of add_ptr's gate; needed for order>=2).
# Takes `tt` (ctx.tt), not tangent_type directly, for the generator-world dispatch pin.
function _fc_same_stride(tt, @nospecialize(P))
    T = tt(P)
    return isbitstype(P) && isbitstype(T) && Base.aligned_sizeof(P) == Base.aligned_sizeof(T) > 0
end

# Catchable BoundsError guard: Dual's ctor never checks tangent length, and memmove has no bounds
# check of its own.
# The literal `true` boundscheck forces the check (elided elsewhere under --check-bounds=auto).
# @noinline: inlining would split a block, breaking dualize_to_ircode's 1:1 block-topology invariant.
@noinline function _fc_check_extent(ref::MemoryRef, nelem::Int)
    if nelem > 0
        Core.memoryrefnew(ref, nelem, true)
    end
    return nothing
end
@noinline function _fc_check_extent(mem::Memory, nelem::Int)
    if nelem > 0
        Core.memoryrefnew(Core.memoryrefnew(mem), nelem, true)
    end
    return nothing
end

# memmove/memcpy signature check only; the mode-specific rule bodies stay in each package
# (`apply_foreigncall_frule!` / the reverse dispatch layer).

const _FC_COPY_ATS = Core.svec(Ptr{Cvoid}, Ptr{Cvoid}, Csize_t)

# Checks the whole signature, not just the name: a same-named foreigncall with a different
# ATs/nreq/cconv would otherwise get operands 6/7/8 mis-assigned as (dst, src, nbytes).
function _fc_copy_sig_ok(fc, ctx, what::String)
    ok = fc.lib === nothing && fc.cconv === QuoteNode(:ccall) && fc.nreq === 0 &&
         fc.RT === Ptr{Cvoid} && fc.ATs == _FC_COPY_ATS && length(fc.args) == 3
    ok || (ctx.reason[] = "`$what` with an unrecognised signature — expected the 3-argument " *
                          "`(dst, src, nbytes)` form returning `Ptr{Cvoid}` via `:ccall`, with no " *
                          "library and no varargs, but got return type `$(fc.RT)` and argument " *
                          "types `$(fc.ATs)`")
    return ok
end

# --- Shared pieces of both engines' `_activity` scans (`DifferForwards`/`DifferReverse`'s own
# `forward_interp.jl`/`reverse_interp.jl`). The two walkers stay separate — enough of their arms
# are genuinely mode-specific (Dual/tangent- vs CoDual/fdata-oriented type queries, reverse's
# mixed-activity vararg-tail and rule-less-builtin exemptions) that a single merged walker would
# need as many hooks as it saves lines. These are the arms that are byte-identical, or identical
# up to a mode-supplied hook. ---

# `fpos`/`actual` out of a `:call`/`:invoke` `Expr`.
_call_parts(s::Expr) = s.head === :invoke ? (s.args[2], @view s.args[3:end]) : (s.args[1], @view s.args[2:end])

# A pointer-typed statement, or a load/store through one, is unconditionally active: what the
# address addresses is outside static analysis, so how the address was computed does not bound
# what reading through it depends on. Without this, e.g. `unsafe_load(Ptr{Float64}(u))` for a
# `u::UInt` (no tangent space, hence inactive) comes out inactive and is silently mishandled.
function _act_ptr_deref(@nospecialize(s), iworld)
    (isa(s, Expr) && (s.head === :call || s.head === :invoke)) || return false
    fpos, _ = _call_parts(s)
    f = _calleeval(fpos, iworld)
    return f === Core.Intrinsics.pointerref || f === Core.Intrinsics.pointerset ||
           f === Core.Intrinsics.atomic_pointerref || f === Core.Intrinsics.atomic_pointerset ||
           f === Core.Intrinsics.atomic_pointerswap || f === Core.Intrinsics.atomic_pointermodify ||
           f === Core.Intrinsics.atomic_pointerreplace
end

# Whether a (widened) type can carry caller-visible mutable shadow state: the same test
# `fdata_shadow_type`/`CoDual` use for "this needs real shadow storage, not just a value copied
# through" — `fdata_type(tangent_type(T)) !== NoFData`. A plain differentiable scalar (`Float64`)
# answers `false` (its fdata is `NoFData`, all information rides in the rdata half); a mutable
# struct/`Array`/`Memory`, or an immutable type with such a thing nested inside it, answers `true`.
#
# Used to make a composite call/invoke an activity root regardless of operand activity when its
# result could be a fresh mutable container — e.g. a `@noinline` allocation helper behind a
# function barrier, called with only constant arguments. A store never propagates activity
# backward into where a container came from, so without this the container is misclassified
# constant and an active value later written into it is lost.
#
# `tt`/`ft` are the caller's tangent/fdata-type queries. `is_composed_carrier(W)` excludes a mode's
# own higher-order self-tangent carrier (forward's `Dual{T,T}` composing over itself) from being
# treated as a fresh user-level container — that type has no `fdata_type` method, and a statement
# typed with it is composed machinery, not something a store could allocate.
function _act_container_result(@nospecialize(T), tt, ft, is_composed_carrier)
    T isa Type || return false
    T === Union{} && return false
    W = tt(T)
    (W isa Type && !is_composed_carrier(W)) || return false
    return ft(W) !== NoFData
end
