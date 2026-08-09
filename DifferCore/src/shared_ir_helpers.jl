# Small, mode-agnostic IR-inspection helpers used by both AD engines (`DifferForwards`/
# `DifferReverse`). Lived in DifferForwards.jl's `forward_interp.jl`/`builtins.jl` in the
# pre-split monolith, reached by reverse-mode code only because everything shared one module
# namespace. Homed here rather than in either AD package: self-contained, no-state utilities with
# no AD-mode opinion of their own, needed by both sides.

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
# concrete or its fields don't all share one tangent type. Used to allow a dynamic (runtime-
# computed) field index into a homogeneous same-shape aggregate, and to recognize an object that's
# entirely non-differentiable regardless of which field a dynamic index happens to hit.
function _bi_homog_tangent_type(P)
    (P isa DataType && isconcretetype(P)) || return nothing
    nf = fieldcount(P)
    nf == 0 && return nothing
    tt1 = tangent_type(fieldtype(P, 1))
    for j in 2:nf
        tangent_type(fieldtype(P, j)) === tt1 || return nothing
    end
    return tt1
end

# For a general-struct field access whose object primal type is `Pobj` and whose field is named by
# the compile-time literal `name`: return `(NT, i)` — the object's tangent-backing `NamedTuple`
# type and the 1-based field index into it — or `nothing` when the direct-emission preconditions
# don't hold (non-concrete primal, non-`Tangent`/`MutableTangent` tangent, unknown/out-of-range
# field, or a `PossiblyUninitTangent` slot needing the generic unwrap/wrap helper instead).
function _tangent_field_slot(@nospecialize(Pobj), @nospecialize(name))
    (Pobj isa DataType && isconcretetype(Pobj)) || return nothing
    Tobj = tangent_type(Pobj)
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
