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
# Bailing is the only safe default here, more so than for intrinsics. Native code can write through
# any pointer it's handed, so "compute the primal, hand back a zero tangent" — the treatment a
# non-differentiable intrinsic gets — isn't sound: a `memmove` given that treatment would leave the
# destination's tangent stale rather than zero. Each target has to be understood individually before
# it can be registered.
#
# Statement layout (Julia 1.13, `Compiler/src/tfuncs.jl`'s `FOREIGNCALL_ARG_START`):
#
#   Expr(:foreigncall, name, RT, ATs::SimpleVector, nreq, cconv, args..., roots...)
#   #                   1     2    3                 4     5      6:5+length(ATs)   rest
#
# `name` is `Expr(:tuple, QuoteNode(:memmove))` (optionally with a library operand), or an
# `SSAValue`/`Argument` for a runtime function pointer — the latter names nothing dispatchable, and
# `_fc_parse` rejects it.
#
# `ctx` is the `foreigncall_ctx` `NamedTuple` `dualize_to_ircode` builds once per call. It carries the
# same closures the builtin rules get (`emit!`/`opf`/`presolve`/`tresolve`/`optype`/`tt`/
# `zero_shadow`/`emit_invoke!`/`reason`) plus two the provenance walk below needs: `pstmt(x)`, the
# primal statement node behind an old `SSAValue`, and `calleeval(x)`, the resolved callee value.
# ===========================================================================

apply_foreigncall_frule!(::Val{F}, fc, Ti, ctx) where {F} = nothing

# Split a foreigncall statement into its parts, or `nothing` when the target isn't a literal symbol
# (a runtime function pointer). `args`/`roots` split at `5 + length(ATs)` — the argument-type `svec`
# is what says how many of the trailing operands are values rather than GC roots.
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

# Rebuild a foreigncall statement with substituted value operands. The name node is *copied* rather
# than shared: it is a mutable `Expr` owned by the cached primal `IRCode`, which is reused across
# dualizations (including higher orders), and every other arm of the dispatch chain builds fresh
# `Expr`s for the same reason.
_fc_stmt(fc, args, roots) =
    Expr(:foreigncall,
         isa(fc.name_node, Expr) ? Expr(fc.name_node.head, fc.name_node.args...) : fc.name_node,
         fc.RT, fc.ATs, fc.nreq, fc.cconv, args..., roots...)

# ---------------------------------------------------------------------------
# Pointer provenance
# ---------------------------------------------------------------------------

# Walk the *primal* IR back from a pointer operand to the `Memory`/`MemoryRef` its address was read
# out of, returning `(P, ref_operand)` — the buffer's element type and the (old-numbered) operand
# naming the buffer. `nothing` if the chain is anything else.
#
# Only `PiNode` and a `Ptr`→`Ptr` `bitcast` are followed. `add_ptr`/`sub_ptr` end the walk
# deliberately: the address they produce is still fine, but the extent check below is relative to the
# originating ref, and an offset pointer breaks that relationship. (Real `copyto!`/broadcast IR never
# has one — the offset is baked into the `memoryrefnew` upstream.)
#
# `ctx.optype` reads the *primal* `IRCode` in its own numbering (`_optype(pir, x)`), so this is safe
# to call from a rule after the pass has already emitted interleaved shadow statements.
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
            # `isa(…, DataType)` first: `optype` can hand back a `Core.Const`/`Core.PartialStruct`
            # lattice element, on which `<:` is a hard `TypeError` rather than a bail.
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

# Whether a *byte* count means the same thing for `P`'s tangent storage as for `P` itself.
#
# Keyed on the element type recovered by the provenance walk, **not** the pointer's own declared
# `Ptr{…}` parameter — deliberately the opposite of `add_ptr`'s gate (`src/intrinsics.jl`), and what
# makes order ≥2 work: there the shadow pointer is a `Ptr{NoTangent}` (stride 0) reached by `bitcast`
# from a `MemoryRef{Float64}`, and the `Float64` is what governs. Don't unify the two gates.
function _fc_same_stride(@nospecialize(P))
    T = tangent_type(P)
    return isbitstype(P) && isbitstype(T) && Base.aligned_sizeof(P) == Base.aligned_sizeof(T) > 0
end

# Raises a catchable `BoundsError` when a shadow buffer is shorter than the bulk copy about to run on
# it. `Dual`'s constructor only checks `tangent_type(P) == T`, never that a caller's tangent array has
# its primal's *length*, and a raw `memmove` has no bounds check of its own — a short destination
# tangent segfaults, and a short source tangent silently reads uninitialised heap.
#
# The literal `true` boundscheck argument is what forces the check: Julia's own `@boundscheck` guards
# in `unsafe_copyto!` sit inside an `@inbounds` block and get elided under the default
# `--check-bounds=auto`, so they can't be relied on.
#
# `@noinline` for the usual reason (`__pop_blk_stack!`, `_rr_get_tangent_field`, …): the branch must
# live inside a helper, since emitting it inline would split a basic block and break the 1:1
# block-topology invariant `dualize_to_ircode` relies on.
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

# ---------------------------------------------------------------------------
# Bulk memory copies: `memmove`/`memcpy`.
#
# This is the target that blocks real code: `copyto!`, `copy`, `Base.unsafe_copyto!` — hence
# broadcast — all bottom out in one. A copy is linear and structure-preserving, so the identical copy
# between the two shadow buffers is exactly the tangent of copying the primals; the rule is that
# mirror plus the preconditions that make it meaningful.
# ---------------------------------------------------------------------------

const _FC_COPY_ATS = Core.svec(Ptr{Cvoid}, Ptr{Cvoid}, Csize_t)

# A bare target symbol says nothing about arity or types, so check the *whole* signature before
# treating operands 6/7/8 as (dst, src, nbytes). Without this, a same-named foreigncall with a
# different `ATs`/`nreq`/`cconv` would have its operands silently mis-assigned.
function _fc_copy_sig_ok(fc, ctx, what::String)
    ok = fc.lib === nothing && fc.cconv === QuoteNode(:ccall) && fc.nreq === 0 &&
         fc.RT === Ptr{Cvoid} && fc.ATs == _FC_COPY_ATS && length(fc.args) == 3
    ok || (ctx.reason[] = "`$what` with an unrecognised signature — expected the 3-argument " *
                          "`(dst, src, nbytes)` form returning `Ptr{Cvoid}` via `:ccall`, with no " *
                          "library and no varargs, but got return type `$(fc.RT)` and argument " *
                          "types `$(fc.ATs)`")
    return ok
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
            nt = o !== nothing && tangent_type(o[1]) === NoTangent
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
            if !_fc_same_stride(P)
                T = tangent_type(P)
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
