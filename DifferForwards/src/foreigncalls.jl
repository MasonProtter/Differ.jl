# ===========================================================================
# Foreigncall rules — dispatch-based handling of `Expr(:foreigncall)` (i.e. `ccall`).
#
# `apply_foreigncall_frule!(Val(name), fc, Ti, ctx)` is called from the main statement loop in
# `dualize_to_ircode` (`forward_interp.jl`) for every foreigncall in the primal IR, dispatching on the
# target symbol the same way `apply_intrinsic_frule!` dispatches on `Val(intrinsic)`. A rule emits the
# primal + shadow IR directly into the caller's instruction stream and returns `(primal_ssa,
# shadow_ssa)`; the fallback returns `nothing`, so an unregistered target bails with a located reason
# instead of silently miscompiling.
#
# Bailing is the only safe default here, more so than for intrinsics: native code can write through
# any pointer it's handed, so "compute the primal, hand back a zero tangent" isn't sound — a
# `memmove` given that treatment would leave the destination's tangent stale rather than zero.
#
# `name` is `Expr(:tuple, QuoteNode(:memmove))` (optionally with a library operand), or an
# `SSAValue`/`Argument` for a runtime function pointer — the latter names nothing dispatchable, and
# `_fc_parse` rejects it.
#
# `ctx` is the `foreigncall_ctx` `NamedTuple` `dualize_to_ircode` builds once per call. It carries the
# same closures the builtin rules get (`emit!`/`opf`/`presolve`/`tresolve`/`optype`/`tt`/
# `zero_shadow`/`emit_invoke!`/`reason`) plus two the provenance walk below needs: `pstmt(x)`, the
# primal statement node behind an old `SSAValue`, and `calleeval(x)`, the resolved callee value.
#
# Parsing/provenance helpers (`_fc_parse`, `_fc_stmt`, `_fc_ptr_origin`, `_fc_same_stride`,
# `_fc_check_extent`, `_fc_copy_sig_ok`, `_FC_COPY_ATS`) live in `DifferCore/src/shared_ir_helpers.jl`,
# shared with `DifferReverse`'s foreigncall layer.
# ===========================================================================

apply_foreigncall_frule!(::Val{F}, fc, Ti, ctx) where {F} = nothing

# `@threads :static` emits a bare `ccall(:jl_in_threaded_region, Cint, ())` in the *enclosing*
# function, where no `frule!!` on a Julia-level function can reach it. It reads a scheduler flag,
# takes no operands and writes through no pointer, so the primal alone plus a zero shadow is exact —
# the reason the blanket bail exists (native code writing through a handed-over pointer) doesn't
# apply to a nullary query.
function apply_foreigncall_frule!(::Val{:jl_in_threaded_region}, fc, Ti, ctx)
    p = ctx.emit!(_fc_stmt(fc, (), map(ctx.presolve, fc.roots)), Ti)
    return p, ctx.zero_shadow(Ti, p)
end

# `jl_set_task_tid` is `@spawnat`'s pinning call — StableTasks inlines its retry loop into the
# enclosing function. Task and tid are non-differentiable and it writes only scheduler state, so
# the primal call with renumbered operands plus a zero shadow is exact.
function apply_foreigncall_frule!(::Val{:jl_set_task_tid}, fc, Ti, ctx)
    p = ctx.emit!(_fc_stmt(fc, map(ctx.presolve, fc.args), map(ctx.presolve, fc.roots)), Ti)
    return p, ctx.zero_shadow(Ti, p)
end

# `jl_object_id` — object-identity hash, what `IdDict`/`ScopedValues` lookups bottom out in. Reads
# the object's identity, never differentiable memory; a `UInt` result has no tangent.
function apply_foreigncall_frule!(::Val{:jl_object_id}, fc, Ti, ctx)
    p = ctx.emit!(_fc_stmt(fc, map(ctx.presolve, fc.args), map(ctx.presolve, fc.roots)), Ti)
    return p, ctx.zero_shadow(Ti, p)
end

for op in (:memmove, :memcpy)
    @eval function apply_foreigncall_frule!(::Val{$(QuoteNode(op))}, fc, Ti, ctx)
        what = $(string(op))
        _fc_copy_sig_ok(fc, ctx, what) || return nothing
        dstx, srcx, nx = fc.args[1], fc.args[2], fc.args[3]

        # The byte count is consumed unchanged by the shadow copy, so it must already be in the
        # declared `Csize_t` representation (`ccall` lowering converts before the foreigncall). Widen
        # first: a const-propagated count carries a `Core.Const`, not a bare `Type`.
        Pn = ctx.optype(nx)
        if (Pn isa Type ? Pn : CC.widenconst(Pn)) !== Csize_t
            ctx.reason[] = "`$what` whose byte count is declared `$Pn`, not `$(Csize_t)`"
            return nothing
        end

        # Both addresses must trace back to a `Memory`/`MemoryRef` this pass gave a real shadow
        # buffer, and the tangent element must have the primal's stride — a byte count only transfers
        # when a byte offset means the same thing on both sides.
        #
        # Exception, settled first: when a buffer's elements have no tangent (`copy(::Vector{Int})`, a
        # `Bool` mask, `collect(1:n)`) there's no shadow storage to copy into or out of, and the pass
        # hands out `NULL_SHADOW_PTR` for its address. Copying nothing is exactly the tangent of
        # copying non-differentiable data, so the primal call is emitted alone — no shadow copy, no
        # extent guards either, since a `Memory{NoTangent}` holds nothing that could be overrun. Both
        # sides must agree: a byte copy between a shadowed and an unshadowed buffer is a reinterpreting
        # copy whose tangent this rule can't express.
        sides = ((dstx, "destination"), (srcx, "source"))
        walked = Any[_fc_ptr_origin(x, ctx) for (x, _) in sides]
        nulls = Bool[]
        for ((x, side), o) in zip(sides, walked)
            nt = o !== nothing && ctx.tt(o[1]) === NoTangent
            # The walk's verdict and the shadow operand must agree, or something other than this
            # rule's assumptions produced that operand — decline rather than guess which one is right.
            if nt != (ctx.tresolve(x) === NULL_SHADOW_PTR)
                ctx.reason[] = "`$what` whose $side pointer disagrees with its shadow: the buffer's " *
                               "elements $(nt ? "have no tangent, but the shadow pointer is not the " *
                               "null sentinel" : "have a tangent, but the shadow pointer is the null " *
                               "sentinel")"
                return nothing
            end
            push!(nulls, nt)
        end
        if all(nulls)
            p = ctx.emit!(_fc_stmt(fc, (ctx.presolve(dstx), ctx.presolve(srcx), ctx.presolve(nx)),
                                   map(ctx.presolve, fc.roots)), Ti)
            TT = ctx.tt(Ti)
            if TT !== NoTangent && TT !== typeof(NULL_SHADOW_PTR)
                ctx.reason[] = "`$what` over elements with no tangent, but its own result needs a " *
                               "`$TT` shadow rather than `$(typeof(NULL_SHADOW_PTR))`"
                return nothing
            end
            return p, TT === NoTangent ? NoTangent() : NULL_SHADOW_PTR
        elseif any(nulls)
            ctx.reason[] = "`$what` between buffers in different tangent regimes — one side's " *
                           "elements have a tangent and the other's do not, so the byte copy has no " *
                           "shadow counterpart"
            return nothing
        end

        origins = Any[]
        for ((x, side), o) in zip(sides, walked)
            if o === nothing
                ctx.reason[] = "`$what` whose $side pointer is not traceable to a `Memory`/" *
                               "`MemoryRef` this pass shadows (only a `pointer`-style " *
                               "`getfield(::MemoryRef, :ptr_or_offset)` chain is)"
                return nothing
            end
            P = o[1]
            if !_fc_same_stride(ctx.tt, P)
                T = ctx.tt(P)
                ctx.reason[] = "`$what` over `$P` elements: a byte count only carries over to the " *
                               "shadow buffer when the tangent element type has the same stride, " *
                               "and `$P` (stride $(isbitstype(P) ? Base.aligned_sizeof(P) : "?")) " *
                               "and its tangent `$T` (stride " *
                               "$(isbitstype(T) ? Base.aligned_sizeof(T) : "?")) do not"
                return nothing
            end
            Tref = ctx.tt(ctx.optype(o[2]))
            if !(Tref isa DataType && (Tref <: MemoryRef || Tref <: Memory))
                ctx.reason[] = "`$what` whose $side buffer has tangent type `$Tref`, which is not " *
                               "a same-shape `Memory`/`MemoryRef`"
                return nothing
            end
            push!(origins, (P, o[2], Tref))
        end

        pdst, psrc, pn = ctx.presolve(dstx), ctx.presolve(srcx), ctx.presolve(nx)

        # Guard each shadow buffer's extent, so a caller-supplied short tangent raises a
        # `BoundsError` instead of corrupting the heap. See `_fc_check_extent`. Emitted before both
        # copies so a failure leaves the primal untouched too.
        for (P, refx, Tref) in origins
            stride = Base.aligned_sizeof(P)
            nelem = ctx.opf(:bitcast, Int, Int,
                            ctx.opf(:udiv_int, Csize_t, pn, Csize_t(stride)))
            ctx.emit_invoke!(_fc_check_extent, Nothing, (Tref, Int), ctx.tresolve(refx), nelem)
        end

        p = ctx.emit!(_fc_stmt(fc, (pdst, psrc, pn), map(ctx.presolve, fc.roots)), Ti)

        # The shadow pointers are `Ptr{NoTangent}`-typed. `Ptr{Cvoid}` happens to accept any cpointer
        # in `ccall` codegen, but don't lean on that escape — cast to the declared argument type.
        tdst = ctx.opf(:bitcast, Ptr{Cvoid}, Ptr{Cvoid}, ctx.tresolve(dstx))
        tsrc = ctx.opf(:bitcast, Ptr{Cvoid}, Ptr{Cvoid}, ctx.tresolve(srcx))
        troots = Any[a === dstx ? tdst : a === srcx ? tsrc : ctx.presolve(a) for a in fc.roots]
        sh = ctx.emit!(_fc_stmt(fc, (tdst, tsrc, pn), troots), fc.RT)

        # The shadow statement must be *declared* `tangent_type(Ti)` even though the mirrored call
        # produces the same `Ptr{Cvoid}` the primal does (verify gotcha #6) — reconcile with a no-op
        # `bitcast`, as the `MemoryRef` `getfield` branch does. The value is a genuine tangent pointer
        # (it addresses the shadow destination); every observed IR discards it in practice.
        TT = ctx.tt(Ti)
        t = TT === NoTangent ? NoTangent() :
            TT === fc.RT ? sh :
            (TT isa DataType && TT <: Ptr) ? ctx.opf(:bitcast, TT, TT, sh) : ctx.zero_shadow(Ti, p)
        return p, t
    end
end
