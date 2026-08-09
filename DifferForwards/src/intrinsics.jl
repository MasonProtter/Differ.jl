# ===========================================================================
# Intrinsic rules — dispatch-based, direct-IR-emission handling of `Core.Intrinsics`.
#
# Every `Core.Intrinsics` function (`add_float`, `mul_float`, …) is an instance of the single type
# `Core.IntrinsicFunction`, so ordinary dispatch on `typeof(f)` can't tell them apart — but an
# intrinsic value is itself a valid type parameter, so `Val{Core.Intrinsics.add_float}` names one
# specific intrinsic and ordinary multiple dispatch on `Val` works.
#
# `apply_intrinsic_frule!(Val(f), actual, Ti, ctx)` is called from the main statement loop in
# `dualize_to_ircode` (`forward_interp.jl`) for every intrinsic call in the primal IR. It emits the
# primal + shadow IR directly into the caller's instruction stream and returns
# `(primal_ssa, shadow_ssa)` (or `nothing` if unregistered) — no `Dual` boxing, no `frule!!` dispatch,
# no `CodeInstance` resolution/compile the way a surviving high-level call (`frule_split!`, e.g.
# `sin`/`cos`) needs. That machinery is fine for the handful of calls that survive a function's body,
# but every arithmetic op in a differentiated function is an intrinsic call — routing each one through
# a full `frule!!`/`CodeInstance` round trip (as an earlier version of this file did: wrap each
# intrinsic in a thin wrapper function with its own singleton type, rewrite the call to it, and
# dispatch `frule!!` on that) bloated both compile time and the generated code. Direct emission keeps
# intrinsics exactly as cheap as the primal computation itself, while still reaching each rule via
# ordinary dispatch instead of a hand-rolled if-else chain.
#
# `ctx` is a `NamedTuple` of the closures `dualize_to_ircode` builds once per call:
#   * `ctx.opf(name, ty, args...)` — emit `Expr(:call, GlobalRef(Core.Intrinsics, name), args...)`
#   * `ctx.emit!(ex, ty)`          — emit any other typed IR statement (e.g. a `Core.ifelse` select)
#   * `ctx.presolve(x)`/`ctx.tresolve(x)` — resolve an operand AST node to its primal/shadow SSA
#   * `ctx.zero_shadow(Ti, primal_ssa)` — the zero tangent of a computed non-differentiable result
#
# The fallback below returns `nothing`, so an intrinsic with no registered rule bails (in
# `dualize_to_ircode`) with a clear, located reason instead of silently miscompiling — e.g. a missing
# derivative silently returning a wrong zero tangent. Register a differentiable intrinsic by hand (see
# below); register a non-differentiable one (comparisons, integer/bit ops, …) with
# `@inactive_intrinsic`, which emits the primal and its zero tangent.
# ===========================================================================

apply_intrinsic_frule!(::Val{F}, actual, Ti, ctx) where {F} = nothing

# ---------------------------------------------------------------------------
# Differentiable float intrinsics — hand-written rules, emitted directly.
# ---------------------------------------------------------------------------

# Linear binary ops (`add_float`, `sub_float`): d(a ∘ b) = da ∘ db
for op in (:add_float, :add_float_fast, :sub_float, :sub_float_fast)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a, b = actual[1], actual[2]
        ctx.opf($(QuoteNode(op)), Ti, ctx.presolve(a), ctx.presolve(b)),
        ctx.opf($(QuoteNode(op)), Ti, ctx.tresolve(a), ctx.tresolve(b))
    end
end

# Linear unary op (`neg_float`): d(-a) = -da
for op in (:neg_float, :neg_float_fast)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a = actual[1]
        ctx.opf($(QuoteNode(op)), Ti, ctx.presolve(a)), ctx.opf($(QuoteNode(op)), Ti, ctx.tresolve(a))
    end
end

# Product rule (`mul_float`): d(a·b) = da·b + a·db
for (mul, add) in ((:mul_float, :add_float), (:mul_float_fast, :add_float_fast))
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$mul}, actual, Ti, ctx)
        a, b = actual[1], actual[2]

        pa, pb = ctx.presolve(a), ctx.presolve(b)
        ta, tb = ctx.tresolve(a), ctx.tresolve(b)
        ctx.opf($(QuoteNode(mul)), Ti, pa, pb),
        ctx.opf($(QuoteNode(add)), Ti, ctx.opf($(QuoteNode(mul)), Ti, ta, pb), ctx.opf($(QuoteNode(mul)), Ti, pa, tb))
    end
end

# Quotient rule (`div_float`): d(a/b) = (da·b − a·db) / b²
for (div, sub, mul) in ((:div_float, :sub_float, :mul_float),
                        (:div_float_fast, :sub_float_fast, :mul_float_fast))
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$div}, actual, Ti, ctx)
        a, b = actual[1], actual[2]
        pa, pb = ctx.presolve(a), ctx.presolve(b)
        ta, tb = ctx.tresolve(a), ctx.tresolve(b)
        num = ctx.opf($(QuoteNode(sub)), Ti, ctx.opf($(QuoteNode(mul)), Ti, ta, pb), ctx.opf($(QuoteNode(mul)), Ti, pa, tb))
        ctx.opf($(QuoteNode(div)), Ti, pa, pb),
        ctx.opf($(QuoteNode(div)), Ti, num, ctx.opf($(QuoteNode(mul)), Ti, pb, pb))
    end
end

# `sqrt_llvm(a)`: d(√a) = da / (2√a)
for op in (:sqrt_llvm, :sqrt_llvm_fast)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a = actual[1]
        s = ctx.opf($(QuoteNode(op)), Ti, ctx.presolve(a))
        s, ctx.opf(:div_float, Ti, ctx.tresolve(a), ctx.opf(:add_float, Ti, s, s))
    end
end

# `abs_float(a)`: d|a| = sign(a)·da. `copysign_float(da, da·a)` = |da|·sign(da·a) = da·sign(a).
function apply_intrinsic_frule!(::Val{Core.Intrinsics.abs_float}, actual, Ti, ctx)
    a = actual[1]
    pa, da = ctx.presolve(a), ctx.tresolve(a)
    ctx.opf(:abs_float, Ti, pa), ctx.opf(:copysign_float, Ti, da, ctx.opf(:mul_float, Ti, da, pa))
end

# `max_float`/`min_float`: the tangent follows whichever operand is selected. A branchless
# `Core.ifelse` select picks it out — not a Julia `?:`, which would require splitting the block and so
# break the 1:1 block-topology invariant `dualize_to_ircode` relies on. `Core.ifelse` is itself
# dualizable (see the builtin case next to `getfield` in `forward_interp.jl`), so this stays correct
# if re-dualized at a higher order.
for (op, le) in ((:max_float, :le_float), (:max_float_fast, :le_float_fast))
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a, b = actual[1], actual[2]
        pa, pb = ctx.presolve(a), ctx.presolve(b)
        cond = ctx.opf($(QuoteNode(le)), Bool, pb, pa)      # pb <= pa  <=>  a is the max
        ctx.opf($(QuoteNode(op)), Ti, pa, pb),
        ctx.emit!(Expr(:call, GlobalRef(Core, :ifelse), cond, ctx.tresolve(a), ctx.tresolve(b)), Ti)
    end
end
for (op, le) in ((:min_float, :le_float), (:min_float_fast, :le_float_fast))
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a, b = actual[1], actual[2]
        pa, pb = ctx.presolve(a), ctx.presolve(b)
        cond = ctx.opf($(QuoteNode(le)), Bool, pa, pb)      # pa <= pb  <=>  a is the min
        ctx.opf($(QuoteNode(op)), Ti, pa, pb),
        ctx.emit!(Expr(:call, GlobalRef(Core, :ifelse), cond, ctx.tresolve(a), ctx.tresolve(b)), Ti)
    end
end

# `fma_float`/`muladd_float`, both `a·b + c`: d = da·b + a·db + dc.
for op in (:fma_float, :muladd_float)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        a, b, c = actual[1], actual[2], actual[3]
        pa, pb, pc = ctx.presolve(a), ctx.presolve(b), ctx.presolve(c)
        ta, tb, tc = ctx.tresolve(a), ctx.tresolve(b), ctx.tresolve(c)
        ctx.opf($(QuoteNode(op)), Ti, pa, pb, pc),
        ctx.opf(:add_float, Ti, ctx.opf(:add_float, Ti, ctx.opf(:mul_float, Ti, ta, pb), ctx.opf(:mul_float, Ti, pa, tb)), tc)
    end
end

# `copysign_float(a, b)` = |a|·sign(b): d/da = sign(a)·sign(b), d/db = 0.
# `copysign_float(da, a·b·da)` = |da|·sign(a·b·da) = da·sign(a)·sign(b).
function apply_intrinsic_frule!(::Val{Core.Intrinsics.copysign_float}, actual, Ti, ctx)
    a, b = actual[1], actual[2]
    pa, pb = ctx.presolve(a), ctx.presolve(b)
    da = ctx.tresolve(a)
    ctx.opf(:copysign_float, Ti, pa, pb),
    ctx.opf(:copysign_float, Ti, da, ctx.opf(:mul_float, Ti, pa, ctx.opf(:mul_float, Ti, pb, da)))
end

# Floating-point width conversions (`Float32(::Float64)` → `fptrunc`, `Float64(::Float32)` →
# `fpext`) carry a type as their first argument (arg 1 has no tangent — only `ctx.presolve` is ever
# called on it, never `ctx.tresolve`). They're linear in the value: d(convert(T, a)) = convert(T, da).
for op in (:fpext, :fptrunc)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        T, a = actual[1], actual[2]
        pT = ctx.presolve(T)
        ctx.opf($(QuoteNode(op)), Ti, pT, ctx.presolve(a)), ctx.opf($(QuoteNode(op)), Ti, pT, ctx.tresolve(a))
    end
end

# ---------------------------------------------------------------------------
# Raw pointers: `bitcast` in `Ptr` space, `pointerref`/`pointerset` (`unsafe_load`/`unsafe_store!`),
# and `add_ptr`/`sub_ptr` (`p ± k`).
#
# These all rest on `tangent_type(Ptr{P}) === Ptr{tangent_type(P)}` (`src/tangents.jl`): a shadow
# pointer is a handle to tangent storage at the position its primal addresses, so every rule here is a
# mirror. Two things make the mirror conditional rather than automatic:
#
#  * A shadow buffer's element type is `tangent_type(P)`, whose stride and alignment need not match
#    `P`'s. An element index survives that (`pointerref` scales by the shadow pointer's own element
#    type) but a byte offset does not — hence the stride check in `add_ptr`/`sub_ptr`.
#  * A pointer with no tangent storage behind it (one built from an integer address) has no shadow at
#    all: `Ptr` has no zero tangent (`zero_tangent(::Ptr)` throws by design), so there's nothing to
#    fall back on and the rule declines via `ctx.reason` rather than inventing one.
#
# NOTE (limitation, see ISSUES): a `Ptr` field of a struct gets the primal's own address as its
# tangent (`zero_tangent_internal(x::Ptr)`/`uninit_tangent(x::Ptr)` — a type-correct placeholder that
# must not be dereferenced). `pointerset` cannot tell that apart from a genuine shadow pointer, so
# storing through such a field writes tangent values over primal data. Pointer provenance isn't
# tracked; nothing here can catch it.
# ---------------------------------------------------------------------------

# The null shadow pointer: what a rule hands back when the primal pointer addresses a buffer whose
# tangent elements are `NoTangent`, i.e. there is no shadow storage to point at (`Vector{Int}`,
# `Vector{Bool}`, `collect(1:n)`). Not read off the shadow buffer — a zero-size-element `MemoryRef`
# stores its 0-based index in `ptr_or_offset`, not an address (measured: element 3 of a
# `Memory{NoTangent}` reads back `Ptr{Nothing}(0x2)`), so mirroring the read would produce a small
# bogus address. Synthesized instead, and recognized downstream by `===`: it's a compile-time literal,
# and `Ptr{NoTangent}` only ever arises as a tangent type, never from user code.
#
# Every rule that would dereference a shadow pointer must therefore either skip the shadow operation
# (when the element's tangent type is `NoTangent`, so there's nothing to transfer) or decline — see
# `pointerref`/`pointerset`/`bitcast`/`add_ptr` below and the `memmove` rule in
# `src/foreigncalls.jl`. `tangent_type(Ptr{NoTangent}) === Ptr{NoTangent}`, so the sentinel is
# type-correct wherever `tangent_type(Ptr{Nothing})` is required, and is its own tangent (which is
# what `const_tangent` relies on to keep IR containing it re-dualizable at order ≥ 2).
const NULL_SHADOW_PTR = Ptr{NoTangent}(0)

# Element type of a concrete `Ptr{P}`; `nothing` for anything else. Also the guard that keeps every
# type test in this section away from a non-`Type`: a lattice element (`Core.PartialStruct`) isn't a
# `DataType`, so it lands here as `nothing` instead of a `TypeError` from `<:`.
function _ptr_eltype(@nospecialize(T))
    (T isa DataType && T <: Ptr && isconcretetype(T)) || return nothing
    P = T.parameters[1]
    return P isa Type ? P : nothing
end

# Shared gate for the dereferencing intrinsics: the primal pointer must be a concrete `Ptr{P}` (an
# unparameterized `Ptr` has `tangent_type === NoTangent`, i.e. no shadow pointer to mirror onto), and
# a non-default alignment must mean the same thing for the tangent's element type as for the primal's.
# Base always passes `1` (`base/pointer.jl`), claiming nothing about either type's alignment.
function _ptr_deref_ok(@nospecialize(Pptr), @nospecialize(align), ctx, what::String)
    P = _ptr_eltype(Pptr)
    if P === nothing
        ctx.reason[] = "`$what` through `$Pptr` — only a concrete `Ptr{P}` has a shadow pointer " *
                       "(`tangent_type` of an abstract `Ptr` is `NoTangent`)"
        return false
    end
    T = tangent_type(P)
    if align !== 1 && !(isbitstype(P) && isbitstype(T) &&
                        Base.datatype_alignment(P) == Base.datatype_alignment(T))
        ctx.reason[] = "`$what` on `$Pptr` with alignment `$align` — the tangent element type `$T` " *
                       "is not known to share `$P`'s alignment, so the primal's alignment claim " *
                       "does not carry over to the shadow buffer"
        return false
    end
    return true
end

# `bitcast(T, x)`: raw bit reinterpretation. Non-differentiable in general, except between pointer
# types, which is where `pointer(v)` ends up (`getfield(ref, :ptr_or_offset)::Ptr{Nothing}` then
# `bitcast(Ptr{Float64}, _)`). `bitcast` preserves the address whatever the element-type label says,
# so reinterpreting the shadow pointer the same way keeps it pointing at tangent storage — this is
# also what lets a `Ptr{Cvoid}` round trip survive.
function apply_intrinsic_frule!(::Val{Core.Intrinsics.bitcast}, actual, Ti, ctx)
    T, a = actual[1], actual[2]
    TT = ctx.tt(Ti)
    inptrspace = TT !== NoTangent && _ptr_eltype(Ti) !== nothing
    if inptrspace
        Pa = ctx.optype(a)
        if !(Pa isa DataType && Pa <: Ptr)
            ctx.reason[] = "`bitcast` to `$Ti` from `$Pa` — a pointer built from a non-pointer has " *
                           "no tangent storage to point at, and a `Ptr` has no zero tangent"
            return nothing
        end
        # Relabeling a pointer with no tangent storage behind it (`NULL_SHADOW_PTR`): the address is
        # preserved on the primal side, but there's still nothing on the shadow side, so the sentinel
        # is carried through unchanged. It can only stay the sentinel if that's still the required
        # tangent type — otherwise this cast is asking for a differentiable view of a buffer with no
        # shadow (`unsafe_load(Ptr{Float64}(pointer(v_int)))`), and mirroring it would launder a null
        # into a genuine dereference. Decline that.
        if ctx.tresolve(a) === NULL_SHADOW_PTR
            if TT !== typeof(NULL_SHADOW_PTR)
                ctx.reason[] = "`bitcast` to `$Ti` from a pointer with no tangent storage behind it " *
                               "(its buffer's elements have no tangent): the result would need a " *
                               "`$TT` shadow pointer, but there is no shadow buffer to point at"
                return nothing
            end
            return ctx.opf(:bitcast, Ti, ctx.presolve(T), ctx.presolve(a)), NULL_SHADOW_PTR
        end
    end
    p = ctx.opf(:bitcast, Ti, ctx.presolve(T), ctx.presolve(a))
    inptrspace && return p, ctx.opf(:bitcast, TT, TT, ctx.tresolve(a))
    # Non-pointer reinterpretation. `NoTangent` results (the `UInt` cast every bounds check does) are
    # the common case. A differentiable non-pointer result — `bitcast(Float64, ::UInt64)`, i.e.
    # `reinterpret` — gets a zero tangent, which is deliberate but not universally right: such a cast
    # is genuinely zero-derivative in some kernels (`exp`'s `reinterpret(Float64, (k+1023) << 52)`
    # scale factor) and value-carrying in others (`atanh`'s `|x|` bit trick). Differ's answer is a
    # zero here plus a hand-written rule for each affected function (`src/rules_math.jl`), which is
    # why hand-ruled kernels are never dualized in the first place — see `test/test_math_rules.jl`.
    p, ctx.zero_shadow(Ti, p)
end

# `pointerref(p, i, align)` — `unsafe_load(p, i)`. Mirror the load on the shadow pointer. Mirroring
# the element index (rather than computing a byte offset) is what makes a differing tangent stride
# correct: `pointerref` scales by the shadow pointer's own element type, so element `i` is element `i`.
# Matches Mooncake's rule for this intrinsic.
function apply_intrinsic_frule!(::Val{Core.Intrinsics.pointerref}, actual, Ti, ctx)
    ptr, idx, align = actual[1], actual[2], actual[3]
    TT = ctx.tt(Ti)
    TT === NoTangent ||
        _ptr_deref_ok(ctx.optype(ptr), align, ctx, "pointerref") || return nothing
    # Defensive: the `bitcast` rule already refuses to relabel the null sentinel into a differentiable
    # pointer, so this should be unreachable — but a shadow load through it would be a null
    # dereference, so never take it on trust.
    if TT !== NoTangent && ctx.tresolve(ptr) === NULL_SHADOW_PTR
        ctx.reason[] = "`pointerref` of a `$Ti` through a pointer with no tangent storage behind it"
        return nothing
    end
    pidx, palign = ctx.presolve(idx), ctx.presolve(align)
    p = ctx.opf(:pointerref, Ti, ctx.presolve(ptr), pidx, palign)
    TT === NoTangent && return p, NoTangent()
    p, ctx.opf(:pointerref, TT, ctx.tresolve(ptr), pidx, palign)
end

# `pointerset(p, v, i, align)` — `unsafe_store!(p, v, i)`, returns `p`. Mirror the store: the tangent
# of the stored value goes to the same element of the shadow buffer.
function apply_intrinsic_frule!(::Val{Core.Intrinsics.pointerset}, actual, Ti, ctx)
    ptr, val, idx, align = actual[1], actual[2], actual[3], actual[4]
    Pptr = ctx.optype(ptr)
    _ptr_deref_ok(Pptr, align, ctx, "pointerset") || return nothing
    # `Ti` is `Ptr{P}` (the pointer is returned), so `ctx.tt(Ti)` is a `Ptr` and never `NoTangent` —
    # gate on the stored element's tangent type instead. With nothing to store, the shadow "result" is
    # just the shadow pointer (already typed `Ptr{NoTangent} === ctx.tt(Ti)`, and the null sentinel
    # when the buffer has no tangent storage), and the shadow store is skipped entirely.
    nostore = tangent_type(_ptr_eltype(Pptr)) === NoTangent
    # Defensive, as in `pointerref`: a genuine tangent store through the null sentinel would be a null
    # dereference. The `bitcast` rule should have declined before this is reachable.
    if !nostore && ctx.tresolve(ptr) === NULL_SHADOW_PTR
        ctx.reason[] = "`pointerset` through a pointer with no tangent storage behind it, storing a " *
                       "`$(_ptr_eltype(Pptr))` whose tangent is not `NoTangent`"
        return nothing
    end
    pidx, palign = ctx.presolve(idx), ctx.presolve(align)
    p = ctx.opf(:pointerset, Ti, ctx.presolve(ptr), ctx.presolve(val), pidx, palign)
    nostore && return p, ctx.tresolve(ptr)
    p, ctx.opf(:pointerset, ctx.tt(Ti), ctx.tresolve(ptr), ctx.tresolve(val), pidx, palign)
end

# `p ± k` — byte arithmetic. In Julia 1.13 this stays in `Ptr` space
# (`add_ptr(::Ptr{P}, ::UInt)::Ptr{P}`), so the shadow address survives and the same byte offset can
# be applied to it — but only when a byte offset means the same thing on both sides, i.e. the tangent
# element has the primal's stride. `> 0` is load-bearing, not pedantic: `Base.aligned_sizeof` is `0`
# for both `Nothing` and `NoTangent`, so accepting an equal-but-zero stride would wave through
# `Ptr{Nothing}` (void) and every `tangent_type(P) === NoTangent` element — pointers whose shadow is a
# placeholder or a bare offset, which the `bitcast` rule above would then happily launder into a real
# dereference at the wrong stride. Mooncake has no `frule` for these at all, so there's no reference
# implementation to mirror here.
for op in (:add_ptr, :sub_ptr)
    @eval function apply_intrinsic_frule!(::Val{Core.Intrinsics.$op}, actual, Ti, ctx)
        ptr, off = actual[1], actual[2]
        P = _ptr_eltype(Ti)
        T = P === nothing ? nothing : tangent_type(P)
        # No tangent storage behind this address: there's no shadow buffer to offset into, so carry the
        # null sentinel through unchanged rather than computing an offset from it. Must be tested
        # before the stride gate below, which would otherwise reject this case outright (`NoTangent`'s
        # stride is 0, and `> 0` is exactly what that gate demands).
        if ctx.tresolve(ptr) === NULL_SHADOW_PTR
            if ctx.tt(Ti) !== typeof(NULL_SHADOW_PTR)
                ctx.reason[] = "`$($(QuoteNode(op)))` on `$Ti` from a pointer with no tangent storage " *
                               "behind it: the result would need a `$(ctx.tt(Ti))` shadow pointer, " *
                               "but there is no shadow buffer to point at"
                return nothing
            end
            return ctx.opf($(QuoteNode(op)), Ti, ctx.presolve(ptr), ctx.presolve(off)), NULL_SHADOW_PTR
        end
        if P === nothing || !(isbitstype(P) && isbitstype(T) &&
                              Base.aligned_sizeof(P) == Base.aligned_sizeof(T) > 0)
            ctx.reason[] = "`$($(QuoteNode(op)))` on `$Ti`: a byte offset only carries over to the " *
                           "shadow pointer when the tangent element type has the same stride, and " *
                           (P === nothing ? "`$Ti` is not a concrete `Ptr{P}`" :
                            "`$P` (stride $(isbitstype(P) ? Base.aligned_sizeof(P) : "?")) and its " *
                            "tangent `$T` (stride $(isbitstype(T) ? Base.aligned_sizeof(T) : "?")) " *
                            "do not")
            return nothing
        end
        poff = ctx.presolve(off)
        ctx.opf($(QuoteNode(op)), Ti, ctx.presolve(ptr), poff),
        ctx.opf($(QuoteNode(op)), ctx.tt(Ti), ctx.tresolve(ptr), poff)
    end
end

# ---------------------------------------------------------------------------
# Non-differentiable intrinsics — comparisons, integer arithmetic, bit/boolean ops, rounding to an
# integer value, and int↔float / bit conversions. Each gets an auto-generated rule via
# `@inactive_intrinsic`: compute the primal from the argument primals, give the result a zero tangent.
#
# The conversion intrinsics (`sitofp`, `fptosi`, `bitcast`, `trunc_int`, …) carry a type as their
# first argument too. `ctx.presolve` is called uniformly over every argument (never `ctx.tresolve`),
# so the type argument just passes through unchanged — no special-casing needed, unlike the
# `Dual`-boxing approach this replaced (which had to resolve a `GlobalRef` type argument to its actual
# `DataType` value before it could be wrapped in a `Dual{DataType,NoTangent}`).
# ---------------------------------------------------------------------------
macro inactive_intrinsic(name)
    intr = :(Core.Intrinsics.$name)
    nmq = QuoteNode(name)
    esc(quote
        function apply_intrinsic_frule!(::Val{$intr}, actual, Ti, ctx)
            p = ctx.opf($nmq, Ti, (ctx.presolve(a) for a in actual)...)
            p, ctx.zero_shadow(Ti, p)
        end
    end)
end

for name in (
    # integer & float comparisons (result is a `Bool`)
    :eq_int, :ne_int, :slt_int, :sle_int, :ult_int, :ule_int,
    :eq_float, :ne_float, :lt_float, :le_float, :fpiseq,
    :eq_float_fast, :ne_float_fast, :lt_float_fast, :le_float_fast,
    # integer arithmetic
    :add_int, :sub_int, :mul_int, :neg_int, :sdiv_int, :udiv_int, :srem_int, :urem_int,
    :checked_sadd_int, :checked_ssub_int, :checked_smul_int, :checked_sdiv_int, :checked_srem_int,
    :checked_uadd_int, :checked_usub_int, :checked_umul_int, :checked_udiv_int, :checked_urem_int,
    # bit / boolean ops
    :and_int, :or_int, :xor_int, :not_int, :shl_int, :lshr_int, :ashr_int,
    :bswap_int, :ctlz_int, :ctpop_int, :cttz_int, :flipsign_int,
    # misc queries (result is a `Bool`) — e.g. `have_fma(T)`, emitted by `fma`
    :have_fma,
    # rounding to an integral floating-point value (piecewise-constant → zero derivative)
    :floor_llvm, :ceil_llvm, :trunc_llvm, :rint_llvm,
    # int↔float and integer-width conversions (first argument is a type). `bitcast` is *not* here:
    # it needs the hand-written rule above, which keeps this same zero-tangent behaviour for
    # non-pointer results (e.g. the `UInt` cast in a bounds-check comparison) but mirrors the cast
    # onto the shadow pointer in `Ptr` space.
    :sitofp, :uitofp, :fptosi, :fptoui, :trunc_int, :sext_int, :zext_int,
)
    @eval @inactive_intrinsic $name
end
