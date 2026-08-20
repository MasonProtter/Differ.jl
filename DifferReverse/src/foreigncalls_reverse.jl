# Reverse-mode foreigncall rules — dispatch-based handling of `Expr(:foreigncall)` (`ccall`). Mirrors
# `builtins_reverse.jl`'s three-sided dispatch contract: comms scan, forwards emission, and pullback
# emission (all in `reverse_interp.jl`) must agree on a foreigncall statement's shape:
#
#   foreigncall_rrule_comms(::Val{name}, fc, Ti, ctx)       -> Vector{Tuple{item,type}} | false | nothing
#   apply_foreigncall_rrule_fwds!(::Val{name}, fc, Ti, ctx) -> (primal_ssa, shadow|nothing, saved::Dict) | nothing
#   apply_foreigncall_rrule!(::Val{name}, fc, Ti, ctx)      -> Tuple (one entry per `fc.args`, `nothing`
#       for an operand with no contribution) | nothing
#
# `nothing` = no rule registered — always a bail here, never primal-only replay: native code can write
# through any pointer it's handed. `false` (comms only) = a rule exists but this call is out of scope
# (wrong signature, untracked provenance, ...), signaled via `ctx.reason[]`.
#
# `fc` is `_fc_parse(s)`'s result (`nothing` for a runtime-function-pointer target). `ctx` is a
# `NamedTuple` built fresh at each of the three call sites in `reverse_interp.jl`.
#
# Parsing/provenance helpers (`_fc_parse`, `_fc_stmt`, `_fc_ptr_origin`, `_fc_same_stride`,
# `_fc_check_extent`, `_fc_copy_sig_ok`, `_FC_COPY_ATS`) live in `DifferCore/src/shared_ir_helpers.jl`,
# shared with `DifferForwards/src/foreigncalls.jl`. Not mirror images: forward mode mirrors the copy
# onto the shadow buffer (`dst_shadow = src_shadow`), reverse mode accumulates the destination's
# cotangent into the source and zeroes the destination (`src_shadow += dst_shadow; dst_shadow = 0`).

foreigncall_rrule_comms(::Val{F}, fc, Ti, ctx) where {F} = nothing
apply_foreigncall_rrule_fwds!(::Val{F}, fc, Ti, ctx) where {F} = nothing
apply_foreigncall_rrule!(::Val{F}, fc, Ti, ctx) where {F} = nothing

# ---------------------------------------------------------------------------
# `memmove`/`memcpy`: `dst = copy(src)` in reverse is `src̄ += d̄st`, not a mirrored copy. Destination
# tangent starts at zero after the call; the pullback walks the accumulated destination cotangent
# into the source, then restores what the destination held before (same old-tangent-restore
# discipline as `Base.memoryrefset!`'s rule and `mapbang_pullback`, `rules_broadcast.jl`).
#
# Three modes:
#   - both buffers provenance-tracked (`_bi_tracked`, `builtins_reverse.jl`) — the ordinary case
#     above, same requirement `memoryrefget`/`memoryrefset!`/`setfield!` already impose.
#   - both-sides-`NoTangent` (`copy(::Vector{Int})`, a `Bool` mask) — emits the primal call alone.
#   - destination tracked, source declared inactive (`ctx.inactive`) — no source shadow and no
#     accumulation; the destination's cotangent has nowhere to go and is discarded. Gated on
#     activity, never on `_bi_tracked`: an active-but-untraceable source still bails below, since
#     dropping *its* gradient would be wrong.
#
# A mixed tracked/`NoTangent` pair bails, as does a pair of tracked buffers whose tangent element
# types differ — the accumulation step needs `increment!!(::Td, ::Td)`.
# ---------------------------------------------------------------------------

# Normalizes `Memory`/`MemoryRef` to `MemoryRef` so loops below only walk `MemoryRef`s. Plain runtime
# code (not threaded through `icall!`/`icall`), so no `@noinline` barrier needed here (unlike
# `_fc_save_zero!`/`_fc_accum_restore!` below).
_fc_as_ref(x::MemoryRef) = x
_fc_as_ref(x::Memory) = Core.memoryrefnew(x)

# `nelem = nbytes ÷ aligned_sizeof(P)`, same `udiv_int`+`bitcast` pair `DifferForwards/src/
# foreigncalls.jl`'s rule uses. `stride` is a transform-time constant (`P` static), so only `pn`
# (byte count) is a runtime operand.
_fc_emit_nelem!(emit!, pn, stride::Int) =
    emit!(Expr(:call, GlobalRef(Core.Intrinsics, :bitcast), Int,
               emit!(Expr(:call, GlobalRef(Core.Intrinsics, :udiv_int), pn, Csize_t(stride)), Csize_t)),
          Int)

# Fwds-side: saves the destination shadow's `nelem` elements into a fresh `Memory{Td}`, zeros that
# range. Reads primal dest ref `pref` (post-copy, see `_fcr_fwds!`) only for `zero_tangent`'s sake;
# result doesn't depend on the value since `P` is `isbits` (guaranteed by `_fc_same_stride`).
@noinline function _fc_save_zero!(pref::Union{Memory{P},MemoryRef{P}},
                                  sref::Union{Memory{Td},MemoryRef{Td}}, nelem::Int) where {P,Td}
    pr, sr = _fc_as_ref(pref), _fc_as_ref(sref)
    saved = Memory{Td}(undef, nelem)
    for i in 1:nelem
        pi = Core.memoryrefnew(pr, i, false)
        si = Core.memoryrefnew(sr, i, false)
        saved[i] = si[]
        si[] = zero_tangent(pi[])
    end
    return saved
end

# Pullback-side: `src[i] += dst[i]`, then restore `dst[i]` to what `_fc_save_zero!` saved.
@noinline function _fc_accum_restore!(dref::Union{Memory{Td},MemoryRef{Td}},
                                      sref::Union{Memory{Td},MemoryRef{Td}},
                                      saved::Memory{Td}, nelem::Int) where {Td}
    dr, sr = _fc_as_ref(dref), _fc_as_ref(sref)
    for i in 1:nelem
        di = Core.memoryrefnew(dr, i, false)
        si = Core.memoryrefnew(sr, i, false)
        dval = di[]
        si[] = increment!!(si[], dval)
        di[] = saved[i]
    end
    return nothing
end

# Restore-only sibling of `_fc_accum_restore!`, for the "destination tracked, source inactive" mode:
# there's no source to accumulate into, just restore `dst[i]` to what `_fc_save_zero!` saved.
@noinline function _fc_restore!(dref::Union{Memory{Td},MemoryRef{Td}}, saved::Memory{Td},
                                nelem::Int) where {Td}
    dr = _fc_as_ref(dref)
    for i in 1:nelem
        di = Core.memoryrefnew(dr, i, false)
        di[] = saved[i]
    end
    return nothing
end

function _fcr_comms(fc, Ti, ctx, what::String)
    _fc_copy_sig_ok(fc, ctx, what) || return false
    dstx, srcx, nx = fc.args[1], fc.args[2], fc.args[3]
    Pn = ctx.optype(nx)
    if (Pn isa Type ? Pn : CC.widenconst(Pn)) !== Csize_t
        ctx.reason[] = "`$what` whose byte count is declared `$Pn`, not `$(Csize_t)`"
        return false
    end
    sides = ((dstx, "destination"), (srcx, "source"))
    walked = Any[_fc_ptr_origin(x, ctx) for (x, _) in sides]
    for ((x, side), o) in zip(sides, walked)
        o === nothing || continue
        ctx.reason[] = "`$what` whose $side pointer is not traceable to a `Memory`/`MemoryRef` " *
                       "this pass shadows (only a `pointer`-style `getfield(::MemoryRef, " *
                       ":ptr_or_offset)` chain is)"
        return false
    end
    nulls = Bool[ctx.tt(o[1]) === NoTangent for o in walked]
    if all(nulls)
        return Tuple{Any,Any}[]
    elseif any(nulls)
        ctx.reason[] = "`$what` between buffers in different tangent regimes — one side's elements " *
                       "have a tangent and the other's do not, so the byte copy has no shadow " *
                       "counterpart"
        return false
    end
    # Third mode: destination tracked, source declared inactive (constant). The source needs no
    # shadow at all, so it skips the stride/shape/provenance checks below — an active-but-untraceable
    # source (not `ctx.inactive`, still `!_bi_tracked`) still falls through to the ordinary bail.
    src_inactive = ctx.inactive(walked[2][2])
    for ((x, side), o) in zip(sides, walked)
        side == "source" && src_inactive && continue
        P = o[1]
        if !_fc_same_stride(ctx.tt, P)
            T = ctx.tt(P)
            ctx.reason[] = "`$what` over `$P` elements: a byte count only carries over to the shadow " *
                           "buffer when the tangent element type has the same stride, and `$P` and " *
                           "its tangent `$T` do not"
            return false
        end
        Tref = ctx.tt(ctx.optype(o[2]))
        if !(Tref isa DataType && (Tref <: MemoryRef || Tref <: Memory))
            ctx.reason[] = "`$what` whose $side buffer has tangent type `$Tref`, which is not a " *
                           "same-shape `Memory`/`MemoryRef`"
            return false
        end
        if !_bi_tracked(o[2], ctx)
            ctx.reason[] = "`$what`'s $side buffer has no differentiable provenance traceable to a " *
                           "function argument at %$(ctx.ssa.id)"
            return false
        end
    end
    dst_o, src_o = walked[1], walked[2]
    Td_dst, Td_src = ctx.tt(dst_o[1]), ctx.tt(src_o[1])
    if !src_inactive && Td_dst !== Td_src
        ctx.reason[] = "`$what` between buffers whose tangent element types differ ($(Td_dst) vs " *
                       "$(Td_src)) — the pullback's accumulation step requires them to match"
        return false
    end
    dst_ref, src_ref = dst_o[2], src_o[2]
    items = Tuple{Any,Any}[
        ((:fshadow, dst_ref), ctx.tt(ctx.optype(dst_ref))),
        ((:primal, nx), Csize_t),
        ((:old_tangent, ctx.ssa), Memory{Td_dst}),
    ]
    src_inactive || insert!(items, 2, ((:fshadow, src_ref), ctx.tt(ctx.optype(src_ref))))
    return items
end

function _fcr_fwds!(fc, Ti, ctx, what::String)
    dstx, srcx, nx = fc.args[1], fc.args[2], fc.args[3]
    pdst, psrc, pn = ctx.presolve(dstx), ctx.presolve(srcx), ctx.presolve(nx)
    od = _fc_ptr_origin(dstx, ctx)
    P = od[1]
    if ctx.tt(P) === NoTangent
        p = ctx.emit!(_fc_stmt(fc, (pdst, psrc, pn), map(ctx.presolve, fc.roots)), Ti)
        return p, nothing, Dict{Any,Any}()
    end
    os = _fc_ptr_origin(srcx, ctx)
    dst_ref, src_ref = od[2], os[2]
    src_inactive = ctx.inactive(src_ref)
    Td = ctx.tt(P)
    Tref_dst = ctx.tt(ctx.optype(dst_ref))
    s_dst = ctx.sresolve(dst_ref)
    stride = Base.aligned_sizeof(P)
    nelem = _fc_emit_nelem!(ctx.emit!, pn, stride)
    # Extent-guards both shadows (`_fc_check_extent`) before the primal call, so a short tangent
    # array fails before the primal runs — same as the forward rule. Inactive source has no shadow
    # to guard.
    ctx.icall!(_fc_check_extent, Nothing, (Tref_dst, Int), s_dst, nelem)
    if !src_inactive
        Tref_src = ctx.tt(ctx.optype(src_ref))
        s_src = ctx.sresolve(src_ref)
        ctx.icall!(_fc_check_extent, Nothing, (Tref_src, Int), s_src, nelem)
    end
    p = ctx.emit!(_fc_stmt(fc, (pdst, psrc, pn), map(ctx.presolve, fc.roots)), Ti)
    # Reads the dest primal ref after the primal call, so `zero_tangent` sees the post-copy value
    # (immaterial since `P` is `isbits`, but consistent with `memoryrefset!`'s `zero_tangent(p, f)`
    # discipline).
    p_dst = ctx.presolve(dst_ref)
    Pref_dst = ctx.optype(dst_ref)
    saved = ctx.icall!(_fc_save_zero!, Memory{Td}, (Pref_dst, Tref_dst, Int), p_dst, s_dst, nelem)
    return p, nothing, Dict{Any,Any}((:old_tangent, ctx.ssa) => saved)
end

function _fcr_pullback!(fc, Ti, ctx, what::String)
    dstx, srcx, nx = fc.args[1], fc.args[2], fc.args[3]
    nores = (nothing, nothing, nothing)
    od = _fc_ptr_origin(dstx, ctx)
    P = od[1]
    ctx.tt(P) === NoTangent && return nores
    os = _fc_ptr_origin(srcx, ctx)
    dst_ref, src_ref = od[2], os[2]
    src_inactive = ctx.inactive(src_ref)
    Td = ctx.tt(P)
    Tref_dst = ctx.tt(ctx.optype(dst_ref))
    stride = Base.aligned_sizeof(P)
    pn = ctx.fetch_primal(nx)
    nelem = _fc_emit_nelem!(ctx.emit!, pn, stride)
    s_dst = ctx.fetch_shadow(dst_ref)
    saved = ctx.fetch_saved((:old_tangent, ctx.ssa))
    if src_inactive
        ctx.emit!(ctx.icall(_fc_restore!, (Tref_dst, Memory{Td}, Int), s_dst, saved, nelem), Nothing)
    else
        Tref_src = ctx.tt(ctx.optype(src_ref))
        s_src = ctx.fetch_shadow(src_ref)
        ctx.emit!(ctx.icall(_fc_accum_restore!, (Tref_dst, Tref_src, Memory{Td}, Int),
                            s_dst, s_src, saved, nelem), Nothing)
    end
    return nores
end

for op in (:memmove, :memcpy)
    @eval foreigncall_rrule_comms(::Val{$(QuoteNode(op))}, fc, Ti, ctx) =
        _fcr_comms(fc, Ti, ctx, $(string(op)))
    @eval apply_foreigncall_rrule_fwds!(::Val{$(QuoteNode(op))}, fc, Ti, ctx) =
        _fcr_fwds!(fc, Ti, ctx, $(string(op)))
    @eval apply_foreigncall_rrule!(::Val{$(QuoteNode(op))}, fc, Ti, ctx) =
        _fcr_pullback!(fc, Ti, ctx, $(string(op)))
end
