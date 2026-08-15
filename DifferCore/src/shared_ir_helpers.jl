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

# The primal IR's own declared type for operand `x` — taken directly from `pir`, so exact rather
# than guessed; the fallback (`typeof(x)`) only covers genuine literal constants.
_optype(pir, @nospecialize x) = isa(x, Core.SSAValue) ? pir.stmts[x.id][:type] :
                                isa(x, Core.Argument) ? pir.argtypes[x.n] : typeof(x)

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

# Widen a lattice element (`Core.Const`/`Core.PartialStruct`/…) to a real `Type`, or pass a real
# `Type` through unchanged.
_widen(@nospecialize T) = T isa Type ? T : CC.widenconst(T)

# `GlobalRef` constants for the builtins both engines special-case as raw `:call`/`:invoke` targets
# when reconstructing IR. Only the ones both packages actually need are here — `DifferForwards`'s
# own `builtins.jl` has a larger set including ones only forward mode uses.
const _getfieldg = GlobalRef(Core, :getfield)
const _setfieldg = GlobalRef(Core, :setfield!)
const _ctupleg   = GlobalRef(Core, :tuple)

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
