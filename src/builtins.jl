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
# The fallback method below returns `nothing`, so a builtin with no registered rule (e.g.
# `Core.memoryrefoffset`, used by `push!`/`resize!`) bails in `dualize_to_ircode` with a clear,
# located reason instead of silently miscompiling.
# ===========================================================================

apply_builtin_frule!(::Val{F}, actual, Ti, ctx) where {F} = nothing

const _getfieldg  = GlobalRef(Core, :getfield)
const _setfieldg  = GlobalRef(Core, :setfield!)
const _memnewg    = GlobalRef(Core, :memorynew)
const _memrefnewg = GlobalRef(Core, :memoryrefnew)
const _memrefgetg = GlobalRef(Core, :memoryrefget)
const _memrefsetg = GlobalRef(Core, :memoryrefset!)
const _eqeqg      = GlobalRef(Core, :(===))
const _ctupleg    = GlobalRef(Core, :tuple)
const _ifelseg    = GlobalRef(Core, :ifelse)

# getfield(obj, name[, ordering][, boundscheck]) — trailing operands beyond (obj, name) are forwarded
# verbatim (faithful primal reconstruction; harmless on the same-shape shadow branches too, since no
# bounds are touched by `getfield` itself). The general-struct branch never receives them: a
# `Tangent`/`MutableTangent` field read has no atomicity/boundscheck concept.
function apply_builtin_frule!(::Val{Core.getfield}, actual, Ti, ctx)
    Pobj = ctx.optype(actual[1])
    p = ctx.emit!(Expr(:call, _getfieldg, ctx.presolve(actual[1]), actual[2],
                       (ctx.presolve(a) for a in actual[3:end])...), Ti)
    TT = ctx.tt(Ti)
    t = if TT === NoTangent
        NoTangent()
    elseif Pobj <: Dual || Pobj <: Tuple || Pobj <: NamedTuple || Pobj <: Array
        # Same-shape tangent (Dual/Tuple/NamedTuple/Array): index/name the shadow aggregate directly.
        ctx.emit!(Expr(:call, _getfieldg, ctx.tresolve(actual[1]), actual[2],
                       (ctx.presolve(a) for a in actual[3:end])...), TT)
    else
        # General struct: read the field's tangent out of the Tangent/MutableTangent.
        ctx.emit!(Expr(:call, get_tangent_field, ctx.tresolve(actual[1]), actual[2]), TT)
    end
    p, t
end

# setfield!(obj, name, value[, ordering]) mutates obj in place and returns value. Only legal on a
# genuinely mutable primal, whose shadow is therefore always a MutableTangent. A 4th (atomic-ordering)
# arg is forwarded to the primal call only; `set_tangent_field!` has no atomics concept.
function apply_builtin_frule!(::Val{Core.setfield!}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, _setfieldg, ctx.presolve(actual[1]), actual[2], ctx.presolve(actual[3]),
                       (ctx.presolve(a) for a in actual[4:end])...), Ti)
    TT = ctx.tt(Ti)
    t = TT === NoTangent ? NoTangent() :
        ctx.emit!(Expr(:call, set_tangent_field!, ctx.tresolve(actual[1]), actual[2],
                       ctx.tresolve(actual[3])), TT)
    p, t
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
# always forced to `true` on the shadow ref, so a mismatched-length tangent raises a catchable
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

# Same-shape tangent tuple: a non-differentiable slot holds NoTangent(), a differentiable slot holds
# its resolved tangent.
function apply_builtin_frule!(::Val{Core.tuple}, actual, Ti, ctx)
    p = ctx.emit!(Expr(:call, _ctupleg, (ctx.presolve(a) for a in actual)...), Ti)
    TT = ctx.tt(Ti)
    t = TT === NoTangent ? NoTangent() :
        ctx.emit!(Expr(:call, _ctupleg,
                  (ctx.tt(fieldtype(Ti, j)) === NoTangent ? NoTangent() : ctx.tresolve(actual[j])
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
