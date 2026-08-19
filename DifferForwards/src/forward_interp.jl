# Forward-mode AD: the `frule!!` entry point, the `build_contextual_ir` override `ContextualInterpreter`
# calls into, and the split-shadow dualization engine itself.
#
# `frule!!(dualargs::Dual...)` is a `@generated` fallback: for a composite primal `g` with no
# hand-written rule, it compiles a dualized version of `g`'s body under `ContextualInterpreter{Forward}`
# and invokes the resulting `CodeInstance`. Dualization is a split-shadow transform on `g`'s
# post-optimization `IRCode`, spliced into the typeinf pipeline via `build_contextual_ir` below
# (called by `finishinfer!` for the return type, by `optimize` to install the result).

# Carrier stub: gives a MethodInstance whose specTypes is the *dual* signature.
# `ContextualInterpreter{Forward}` replaces its source with the dualized primal body, so this
# body must never actually run.
dualized_impl(dualargs::Dual...) =
    error("Differ.dualized_impl ran directly: ContextualInterpreter could not dualize the primal ",
          "(likely unsupported IR — e.g. a growable-array mutation like `push!`/`resize!`, a ",
          "builtin with no rule, or a splat of something whose length isn't statically known).")


# Runtime dispatcher for a dynamic (`apply_generic`) call that survived into the primal IR (callee or
# an argument too abstractly typed to wrap statically). `map(Dual, ...)` rebuilds concrete `Dual`s
# from the runtime values, so `frule!!`'s `@generated` dispatch can resolve a concrete primal method —
# a statically-built `Dual{Any,Any}` couldn't. Carries the callee's real tangent `tf`, not a forced
# zero, so a dynamically-dispatched closure's capture derivatives still propagate.
function dynamic_frule(f, tf, primals::Tuple, tangents::Tuple)
    duals = map(Dual, primals, tangents)
    return frule!!(Dual(f, tf), duals...)
end


# ---------------------------------------------------------------------------
# Forward-mode transform hook: compiles the `dualized_impl` MethodInstance the generated `frule!!`
# fallback asks for by transforming the primal method's post-optimization `IRCode` (`build_dual_ir`).
# Non-`dualized_impl` MethodInstances return `nothing` and flow through the ordinary pipeline.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Coupling-point hooks: forward-over-reverse composition (DifferReverse.jl is not a dependency
# of DifferForwards.jl). Default-inert here; overridden in `DifferForwardsOverReverseExt`
# (loaded only when both `DifferForwards` and `DifferReverse` are present).
# ---------------------------------------------------------------------------

# Coupling point 1: does `mi` refer to a carrier method owned by a different AD-mode package
# (e.g. DifferReverse's `reverse_fwds_impl`/`reverse_pullback_impl`)? Default `false` — no other
# mode is loaded.
#
# Argument deliberately left untyped (not `::MethodInstance`): an override in a package extension
# must be strictly more specific than the default or Julia treats it as an illegal same-signature
# overwrite across modules ("Method overwriting is not permitted during Module precompilation").
_is_foreign_mode_carrier(@nospecialize(mi)) = false

# Build the optimized IR for a foreign-mode carrier `primal_mi`, sharing this dualization's
# world age. Only called when `_is_foreign_mode_carrier` returned `true`; `nothing` here means a
# genuine build failure (propagated), not "not applicable".
_foreign_mode_primal_ir(interp, @nospecialize(primal_mi), reason::Ref{String}, edges::Vector{Any}) =
    error("Contextual coupling hook `_foreign_mode_primal_ir` has no handler installed for $primal_mi")

# Coupling points 2-3: is `T` a self-similar-shadow bookkeeping type owned by a different AD-mode
# package (e.g. DifferReverse's `Stack`/`SingletonStack`/`CommsCell`/`Tape`, reached under
# forward-over-reverse when that package's own constructor/field-access calls get inlined into raw
# `%new`/`getfield`/`setfield!` before this dualizer ever sees them)? Default `false`.
_foreign_selfsim_shadow_type(@nospecialize(T)) = false

# The shadow's field type at index `fi` of a foreign self-similar-shadow type `T`, when it's sound
# to mirror `getfield`/`setfield!` directly, or `nothing` when the field carries no tangent (the
# primal value is carried through instead — see `builtins.jl`). Only called when
# `_foreign_selfsim_shadow_type(T)` is `true`; generic over any type whose `tangent_type` maps
# fields through `fieldtype` directly, so this default never needs overriding itself.
function _foreign_selfsim_shadow_field(@nospecialize(T::Type), fi::Int)
    FT = tangent_type(fieldtype(T, fi))
    Fsh = fieldtype(tangent_type(T), fi)
    return (Fsh === FT && FT !== NoTangent) ? Fsh : nothing
end

# Some foreign self-similar-shadow types have a non-differentiable field that must still be kept
# in lockstep on a `setfield!` write (bookkeeping, e.g. DifferReverse's `Stack.position`) rather
# than left untouched. Only consulted when `_foreign_selfsim_shadow_field` returned `nothing`.
_foreign_selfsim_mirror_field(@nospecialize(T), fi::Int) = false

function build_contextual_ir(interp::ContextualInterpreter{Forward}, mi::MethodInstance)
    is_dualized_impl(mi) || return nothing
    reason = Ref("Differ could not dualize the primal (no specific reason recorded).")
    edges = Any[]
    ir = build_dual_ir(interp, mi, reason, edges)
    # Stash discovered backedges even on a bail: if the missing piece (a primal method, a
    # not-yet-written `frule!!`) later appears, the mt-backedges recorded below invalidate this
    # carrier so it gets a real chance to dualize instead of staying pinned to the error stub.
    interp.transformed_edges[mi] = edges
    ir === nothing && return error_ircode(interp, mi, reason[])
    return ir
end

# `isa(specTypes, DataType)`, not just `isa(mi.def, Method)`: a MethodInstance's `specTypes` can be a
# `UnionAll` (a signature with free typevars), and `.parameters` on a `UnionAll` throws an uncaught
# `FieldError` from inside `finishinfer!`, killing the whole compile rather than bailing. A carrier
# signature is always a concrete `DataType`. Same guard on every predicate below that reads an
# arbitrary callee's `specTypes`.
is_dualized_impl(mi) = isa(mi.def, Method) && isa(mi.specTypes, DataType) &&
                       !isempty(mi.specTypes.parameters) &&
                       mi.specTypes.parameters[1] === typeof(dualized_impl)

# Minimal IRCode that just `error(msg)`s when invoked, installed via the same path as a real
# dualized body. Used when `build_dual_ir` bails, so calling the carrier reports why instead of
# `dualized_impl`'s generic stub message.
function error_ircode(interp, impl_mi::MethodInstance, msg::String)
    stream = CC.InstructionStream(2)
    stream.stmt[1] = Expr(:call, error, msg); stream.type[1] = Union{}; stream.flag[1] = CC.IR_FLAG_NULL
    stream.stmt[2] = Core.ReturnNode();       stream.type[2] = Union{}; stream.flag[2] = CC.IR_FLAG_NULL
    cfg = CC.CFG(CC.BasicBlock[CC.BasicBlock(CC.StmtRange(1,2), Int[], Int[])], Int[3])
    di = CC.DebugInfoStream(stream.line)
    di.def = impl_mi
    dualparams = impl_mi.specTypes.parameters[2:end]
    argtypes = Any[impl_mi.specTypes.parameters[1], Tuple{dualparams...}]
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[], carrier_world_range(interp))
    CC.verify_ir(ir)
    return ir
end

# The `frule_tt` a hypothetical differentiation of `callee_mi` would resolve against — same shape
# `frule_codeinstance`/`primal_of_impl` build from a surviving call, but derived from
# `callee_mi.specTypes` instead. Returns `nothing` for anything the shape doesn't apply to
# (`Type`-valued parameters, a parameter with no `tangent_type`, …) rather than throwing: `callee_mi`
# may be an arbitrary callee Julia's compiler discovered, not something Differ validated.
function implicit_frule_tt(interp::ContextualInterpreter, callee_mi::MethodInstance)
    isa(callee_mi.def, Method) || return nothing
    isa(callee_mi.specTypes, DataType) || return nothing   # `UnionAll` sig — see `is_dualized_impl`
    params = callee_mi.specTypes.parameters
    isempty(params) && return nothing
    ftype = params[1]
    (ftype isa Type) || return nothing
    try
        # Julia's compilation-signature heuristic collapses a vararg callee's trailing arguments into
        # a single `Vararg{T}`, so `specTypes` doesn't record the call's arity and
        # `Dual{Vararg{Float64},…}` would throw. Mirror the collapse into the `frule!!` signature as
        # an open-ended `Vararg` tail instead: `findsup` resolves such a query fine, and a hand rule
        # for a vararg function needs exactly this. An imprecise match only ever restricts (sound).
        rest = Any[params[2:end]...]     # `params[2:end]` is a SimpleVector; need a Vector to `pop!`
        va = !isempty(rest) && isa(last(rest), Core.TypeofVararg) ? pop!(rest) : nothing
        dualargs = Any[Dual{P,at_world(interp, tangent_type, P)} for P in rest]
        if va !== nothing
            D = Dual{va.T,at_world(interp, tangent_type, va.T)}
            push!(dualargs, isdefined(va, :N) ? Vararg{D,va.N} : Vararg{D})
        end
        return Tuple{typeof(frule!!), Dual{ftype,NoTangent}, dualargs...}
    catch
        return nothing
    end
end

# An mt-backedge on the `frule!!` resolution a hypothetical differentiation of `callee_mi` would use,
# registered even though `callee_mi`'s call was (or may have been) inlined away and never went
# through `frule_split!`/`frule_codeinstance`. So a user later hand-writing `frule!!` for that
# function still invalidates a derivative built before the rule existed. Best-effort: a `frule_tt`
# that can't be built is silently skipped rather than aborting the dualization.
function register_implicit_frule_backedge!(interp::ContextualInterpreter, edges::Vector{Any},
                                           callee_mi::MethodInstance)
    frule_tt = implicit_frule_tt(interp, callee_mi)
    frule_tt === nothing || mt_edge!(edges, frule_tt)
    return nothing
end

# The generated composite fallback (`frule_body`, installed by `refresh_frule`) is the only method of
# `frule!!` with this exact vararg signature — every hand-written rule (`frule!!(::Dual{typeof(f)},
# dualargs::Dual...)` for a concrete `f`) has a strictly narrower signature. So a `Method` matches the
# fallback, rather than some hand rule, iff its signature is exactly this one.
is_generated_frule_fallback(m::Method) = m.sig === Tuple{typeof(frule!!), Vararg{Dual}}

# Does a hand-written `frule!!` (vs. the always-matching generated fallback) apply to a hypothetical
# differentiation of `callee_mi`? Used by `src_inlining_policy` below to keep such a call from being
# inlined away before `dualize_to_ircode` can route it through that rule.
function has_hand_frule(interp::ContextualInterpreter, callee_mi::MethodInstance)
    frule_tt = implicit_frule_tt(interp, callee_mi)
    frule_tt === nothing && return false
    m, _ = CC.findsup(frule_tt, CC.method_table(interp))
    m === nothing && return false
    return !is_generated_frule_fallback(m.method)
end

# Inlining policy: never inline a call whose callee has a hand-written `frule!!` — otherwise the call
# vanishes into the caller's optimized IR before `dualize_to_ircode` ever sees it, regardless of what
# Julia's cost-based heuristic (which has no concept of `frule!!`) thinks. Falls back to the ordinary
# policy otherwise, so this only ever restricts inlining relative to normal Julia behavior.
function CC.src_inlining_policy(interp::ContextualInterpreter{Forward}, mi::MethodInstance,
                                @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::UInt32)
    has_hand_frule(interp, mi) && return false
    return @invoke CC.src_inlining_policy(interp::CC.AbstractInterpreter, mi::MethodInstance,
                                          src::Any, info::CC.CallInfo, stmt_flag::UInt32)
end

# Resolve the primal MethodInstance and dual arity for a `dualized_impl` specialization. `reason`
# records why this bails, for the caller to surface instead of a generic message. `edges` collects:
# an unconditional mt-backedge on `primal_tt` (any new/more-specific method must invalidate this
# derivative), plus, on a match, a direct edge to `primal_mi` (this dual IR inlines its optimized IR
# directly, so redefining it must invalidate us).
function primal_of_impl(interp::ContextualInterpreter, impl_mi::MethodInstance, reason::Ref{String}=Ref(""),
                        edges::Vector{Any}=Any[])
    dualparams = impl_mi.specTypes.parameters[2:end]
    # Defensive, not an assert: `frule!!`/`dualized_impl` are `@generated` and never handed a
    # `Vararg`-collapsed signature (verified for 1..10 arguments), but guard in case a carrier
    # `MethodInstance` is ever reached some other way.
    if !all(P -> P isa Type && P <: Dual, dualparams)
        reason[] = "not every dual argument type is a concrete `Dual`"
        return nothing
    end
    primal_tt = Base.to_tuple_type(Any[_dual_primal_type(P) for P in dualparams])
    push!(edges, primal_tt, Core.methodtable)   # mt-backedge: a new applicable method must invalidate
    pmatch, _ = CC.findsup(primal_tt, CC.method_table(interp))
    if pmatch === nothing
        reason[] = "no unique primal method resolves for argument types " *
                   "$(Tuple(_dual_primal_type(P) for P in dualparams)) — likely an ambiguous " *
                   "dynamic-dispatch call (abstractly-typed arguments) or a nonexistent method"
        return nothing
    end
    if !isa(pmatch.method, Method)
        reason[] = "the resolved primal match is not a concrete Method"
        return nothing
    end
    primal_mi = specialize_method(pmatch.method, pmatch.spec_types, pmatch.sparams)::MethodInstance
    CC.add_inlining_edge!(edges, primal_mi)
    return (primal_mi, length(dualparams))
end

# Recursion cycle guard for `build_dual_ir`, task-local rather than per-instance (unlike reverse
# mode's `interp.in_progress` field): a self-/mutually-recursive primal's nested resolution crosses
# the `frule!!` `@generated`-function boundary, so each recursive level uses a different `interp`
# object even though it targets the same `impl_mi` — a guard scoped to one interpreter can't observe
# the cycle.
#
# NOT the primary recursion mechanism — `frule_split!`'s `dual_recursive_impl_mi` resolves a
# self-/mutually-recursive call at the call site, before ever reaching a `frule_codeinstance` call
# that would recurse back into `build_dual_ir`. This survives as a backstop for anything that reaches
# `build_dual_ir` some other way. Confirmed empirically (before that resolver existed): a
# `@noinline` self-recursive function run through `frule!!` stack-overflows without this.
#
# Task-local because the recursion is synchronous and never leaves the compiling task: shared across
# nested interpreter instances (so the cycle stays observable) but isolated between concurrently
# compiling tasks. A plain global `IdDict` would be corruptible by concurrent compilation.
function dualized_impl_in_progress()
    return get!(() -> IdDict{MethodInstance,Nothing}(), task_local_storage(),
                :differ_dualized_impl_in_progress)::IdDict{MethodInstance,Nothing}
end

# Build the dualized `IRCode` for a `dualized_impl` specialization from the primal's optimized
# `IRCode`. Returns the dual `IRCode` or `nothing` (unsupported IR → bail).
#
# Two cases, distinguished by whether any value argument's primal is itself a `Dual`:
#
#  * First order (base case): the primal is an ordinary user method (or a hand-written `frule!!`),
#    found via `primal_of_impl`, dualized directly. For a vararg primal (`f(a, xs...)`), varargs
#    arrive as a single already-packed tuple slot; `primal_nfixed` tells the prologue to re-pack the
#    trailing flat dual args into it.
#  * Higher order (compose the transform): nested Duals (e.g. `Dual{Dual{F,F},Dual{F,F}}`) means
#    differentiating the order-(k-1) dualized function. Its optimized dual IR is obtained by peeling
#    one `Dual` level off each value arg, resolving the inner `dualized_impl` carrier via
#    `optimized_dual_ir`, then re-dualizing. That inner IR is vararg-shaped
#    (`dualized_impl(dualargs...)`), so `pir_is_vararg=true` selects the tuple-reconstruction
#    prologue in `dualize_to_ircode`.
function build_dual_ir(interp::ContextualInterpreter, impl_mi::MethodInstance, reason::Ref{String}=Ref(""),
                       edges::Vector{Any}=Any[])
    in_progress = dualized_impl_in_progress()
    if haskey(in_progress, impl_mi)
        reason[] = "recursive dualization of $(impl_mi) detected (a self- or mutually-recursive " *
                   "primal) — not yet supported; bailing instead of recursing forever"
        return nothing
    end
    in_progress[impl_mi] = nothing
    try
        return _build_dual_ir(interp, impl_mi, reason, edges)
    finally
        delete!(in_progress, impl_mi)
    end
end

function _build_dual_ir(interp::ContextualInterpreter, impl_mi::MethodInstance, reason::Ref{String}=Ref(""),
                        edges::Vector{Any}=Any[])
    dualparams = impl_mi.specTypes.parameters[2:end]
    if !all(P -> P isa Type && P <: Dual, dualparams)
        reason[] = "not every dual argument type is a `Dual` (a vararg call?)"
        return nothing
    end
    n = length(dualparams)

    # Compose the transform: peel one `Dual` level off `dualparams[1+offset:end]` to form the
    # order-(k-1) carrier signature, obtain that carrier's optimized dual IR, and re-dualize it.
    #  * `offset=0` — uniform seeds where the function nests like every value arg, so the whole list
    #    peels: `Dual{Dual{f,NoTangent},…}` → `Dual{f,NoTangent}`, `Dual{Dual{F,F},…}` → `Dual{F,F}`.
    #  * `offset=1` — a re-dualized carrier invoke whose `dualparams[1]` is a non-nested function slot
    #    (`dualized_impl`/`frule!!`) naming the function being re-differentiated: drop it, peel only
    #    the trailing (nested) value args.
    function compose(offset::Int)
        inner_dualparams = Any[_dual_primal_type(P) for P in dualparams[1+offset:end]]
        inner_tt = Tuple{typeof(dualized_impl), inner_dualparams...}
        m, _ = CC.findsup(inner_tt, CC.method_table(interp))
        if m === nothing
            reason[] = "could not find an inner carrier method to compose (higher-order re-dualization)"
            return nothing
        end
        inner_mi = specialize_method(m.method, m.spec_types, m.sparams)::MethodInstance
        # This dual IR is built on top of the inner carrier's own optimized dual IR (copied/
        # re-dualized wholesale below), so invalidating `inner_mi` must invalidate this one too.
        # `inner_mi`'s own dependencies are tracked on its own edges list, not duplicated here.
        CC.add_inlining_edge!(edges, inner_mi)
        pir = optimized_dual_ir(interp, inner_mi, reason)
        pir === nothing && return nothing
        return dualize_to_ircode(interp, impl_mi, pir, n; pir_is_vararg=true, pir_arg_offset=offset, reason, edges)
    end

    f1 = n >= 1 ? _dual_primal_type(dualparams[1]) : Union{}

    # Higher-order requests re-dualize an order-(k-1) carrier. Two shapes reach here:
    #
    #  * Uniform seeds (order≥2 / a `frule!!(fseed_k, seed_k)` call): every dual arg, function
    #    included, is nested one level, so the whole list peels (`offset=0`). A `frule!!` slot is
    #    excluded: a Dual-valued arg under `frule!!` means "differentiate a hand rule once" (base
    #    case), not "differentiate the derivative".
    #
    #  * A re-dualized surviving carrier invoke ("D-of-D"): when the outer pass dualizes a function
    #    whose body called `frule!!`, that inner call survives as a `dualized_impl`/`frule!!`
    #    `:invoke` whose callee `frule_split!` re-wrapped as the non-nested function slot
    #    `Dual{typeof(dualized_impl),NoTangent}`/`Dual{typeof(frule!!),NoTangent}` at position 1 — so
    #    it's dropped (`offset=1`) and the trailing nested args peel.
    if f1 === typeof(dualized_impl)
        return compose(1)
    elseif f1 !== typeof(frule!!) && any(i -> _dual_primal_type(dualparams[i]) <: Dual, 1:n)
        return compose(0)
    end

    # Base case: an ordinary user method or a hand-written `frule!!`, dualized directly. The `frule!!`
    # slot lands here — `primal_of_impl` peels its args to the concrete rule signature. If that
    # resolves to the generated (vararg) `frule!!` fallback rather than a hand rule, this is actually
    # a composed derivative (3rd+-order nested `D`); fall back to `compose(1)`.
    info = primal_of_impl(interp, impl_mi, reason, edges)
    if info === nothing
        f1 === typeof(frule!!) && return compose(1)
        return nothing
    end
    primal_mi, _ = info
    # A vararg primal (`f(a, xs...)`) declares one extra argument slot holding its varargs already
    # packed into a tuple, while the dual call always supplies `n` flat `Dual`s. `nargs` counts
    # `#self#` and the vararg slot, so `nargs - 1` is the number of slots before it; `dualize_to_ircode`'s
    # prologue re-packs dual args `nargs..n` into that slot.
    pmethod = primal_mi.def::Method
    primal_nfixed = pmethod.isva ? Int(pmethod.nargs) - 1 : nothing

    # Forward-over-reverse: `primal_mi` can itself be a carrier owned by a different AD-mode package
    # (e.g. DifferReverse's `reverse_fwds_impl`/`reverse_pullback_impl`) rather than an ordinary user
    # method — reached when forward-differentiating a `rev_gradient`/`value_and_gradient!`/`Tape`
    # pullback call. Only that other package's interpreter recognizes such mi's, so
    # `CC.typeinf_frame` here would just recompile the untransformed stub; `_foreign_mode_primal_ir`
    # fetches the real optimized IR instead, sharing this build's world age. Default-inert
    # (DifferForwards has no dependency on any other AD-mode package); overridden in
    # `DifferForwardsOverReverseExt` once `DifferReverse` is also loaded.
    # `at_world`/`mt_edge!`: both hooks are overridden in a package extension defined at a strictly
    # later world than this generator's pin, so a direct call would silently take the inert default.
    mt_edge!(edges, Tuple{typeof(_is_foreign_mode_carrier),MethodInstance})
    if at_world(interp, _is_foreign_mode_carrier, primal_mi)
        # The forwards and pullback passes must share one shadow tape (`===`), for both context
        # shapes `reverse_fwds_impl` can be called with: `Ctx{Nothing}` (fresh tape, allocated inline
        # via `%new` — the shadow tape falls out of ordinary split-shadow SSA tracking) and
        # `Ctx{<:Tape}` (pre-allocated — the shadow tape comes out of `ctx`'s own tangent via the
        # ordinary general-struct `getfield` path). Confirmed against the forward-over-forward oracle
        # (`code_dual_ircode(...; order=2)`). A genuinely mismatched pre-allocated tape is still
        # caught downstream by `reverse_fwds_to_ircode`'s own `PreTapeT !== TapeT` bail.
        #
        # A self-recursive primal also falls out cleanly: `reverse_fwds_recursive_ci`'s self-edge
        # targets `Ctx{own_TapeT}` at every recursion depth, and `frule_split!`'s own recursion
        # resolver routes that self-call to a static self-`:invoke` rather than recursing into
        # `build_dual_ir` again — one bounded nested compile total.
        mt_edge!(edges, Tuple{typeof(_foreign_mode_primal_ir),typeof(interp),MethodInstance,
                              Ref{String},Vector{Any}})
        pir = at_world(interp, _foreign_mode_primal_ir, interp, primal_mi, reason, edges)
        pir === nothing && return nothing
    else
        # Optimized primal IR, computed by hand (mirroring `typeinf_ircode`) rather than calling that
        # function directly, so we can also read off `frame.edges` below. Compiled with `interp`
        # itself, not a bare `NativeInterpreter`, so our `src_inlining_policy` override applies — a
        # callee with a hand-written `frule!!` isn't inlined away before `dualize_to_ircode` can
        # route it through that rule.
        frame = CC.typeinf_frame(interp, primal_mi, false)
        if frame === nothing
            reason[] = "inference failed to produce optimized IR for the primal method $(primal_mi)"
            return nothing
        end
        opt = CC.OptimizationState(frame, interp)
        pir = CC.run_passes_ipo_safe(opt.src, opt, nothing)
        # `primal_of_impl`'s `add_inlining_edge!(edges, primal_mi)` alone isn't enough: if the primal
        # inlines a callee (common), that callee's identity is gone from `pir` by the time we see it.
        # `frame.edges`, populated by ordinary inference before the optimizer ran and appended to by
        # inlining, is the primal's real dependency set including everything inlined away.
        append!(edges, frame.edges)
        # For every concrete callee discovered above, register the mt-backedge a hand-written
        # `frule!!` for it would need. `ForwardToBackedgeIterator` decodes the same variable-width
        # edge encoding `store_backedges` understands, regardless of entry shape.
        for (_, item) in CC.ForwardToBackedgeIterator(Core.svec(frame.edges...))
            isa(item, MethodInstance) && register_implicit_frule_backedge!(interp, edges, item)
        end
    end
    return dualize_to_ircode(interp, impl_mi, pir, n; pir_is_vararg=false, primal_nfixed, reason, edges)
end

# The optimized dual `IRCode` for a `dualized_impl` carrier: exactly what `CC.optimize` installs —
# `build_dual_ir` followed by the IPO-safe passes. Used by the higher-order recursion above (to
# obtain the order-(k-1) dual IR as a primal) and by the reflection entry point.
function optimized_dual_ir(interp::ContextualInterpreter, impl_mi::MethodInstance, reason::Ref{String}=Ref(""),
                           edges::Vector{Any}=Any[])
    ir = build_dual_ir(interp, impl_mi, reason, edges)
    ir === nothing && return nothing
    world = CC.get_inference_world(interp)
    opt = CC.OptimizationState(impl_mi, CC.retrieve_code_info(impl_mi, world), interp)
    return begin
        run_ipo_passes!(ir, opt)
    end
end

# Resolve and compile the `frule!!(Dual{typeof(f),ftangty}, dualargs...)` rule for a surviving
# high-level call to an invoke-able `CodeInstance`, so the dualized IR can emit a static `:invoke`.
# `:invoke` targets must be `CodeInstance`s: `collectinvokes!` only JITs those, so a bare
# `MethodInstance` would fall back to a boxed dynamic call. Returns `nothing` if unresolved.
#
# `ftangty` is the callee's own tangent type at the call site — `NoTangent` for a plain function, a
# real `Tangent{…}`/`Dual`/… for a closure with captures. Must exactly match the tangent type
# `frule_split!`'s `dual!(ftype, tt(ftype), fcallee, ftang)` builds the callee `Dual` with: `Dual` is
# invariant, so a mismatched tangent type resolves a different `@generated` specialization than the
# one actually built and `:invoke`d — `verify_ir` doesn't catch it, but codegen crashes with
# "Unreachable reached" at run time.
#
# `edges` collects: an unconditional mt-backedge on `frule_tt` (a new/more-specific user `frule!!`
# invalidates this dual IR), plus, on a match, a direct invoke edge to the resolved `CodeInstance`.
function frule_codeinstance(interp::ContextualInterpreter, @nospecialize(ftype), @nospecialize(ftangty),
                            dual_argtypes, edges::Vector{Any}=Any[])
    frule_tt = Tuple{typeof(frule!!), Dual{ftype,ftangty}, dual_argtypes...}
    push!(edges, frule_tt, Core.methodtable)   # mt-backedge: a new/more-specific frule!! must invalidate
    fm, _ = CC.findsup(frule_tt, CC.method_table(interp))
    fm === nothing && return nothing
    isa(fm.method, Method) || return nothing
    frule_mi = specialize_method(fm.method, fm.spec_types, fm.sparams)::MethodInstance
    world = CC.get_inference_world(interp)
    ci = CC.typeinf_ext_toplevel(CC.NativeInterpreter(world), frule_mi, CC.SOURCE_MODE_ABI)::CodeInstance
    CC.add_invoke_edge!(edges, frule_tt, ci)
    return ci
end

# The `dualized_impl` carrier a *derived* (non-hand-ruled) call to `Dual{ftype,ftangty}(dual_argtypes...)`
# would compile to, or `nothing` when a hand-written `frule!!` wins instead (a leaf, no derived
# recursion possible) or nothing resolves. Used by `frule_split!` to detect a recursive edge (self or
# mutual) *before* resolving/compiling anything, routing around `frule_codeinstance`'s
# `typeinf_ext_toplevel` call, which would recurse forever on a self-/mutually-recursive primal.
#
# Mirrors `frule_body` exactly (`findsup` then the 3-argument `specialize_method`) so that
# `callee_impl_mi === impl_mi` in `frule_split!` below means anything — object identity with what
# `dualized_impl`'s own generator would itself produce is the whole point.
function dual_recursive_impl_mi(interp, @nospecialize(ftype), @nospecialize(ftangty), dual_argtypes)
    frule_tt = Tuple{typeof(frule!!), Dual{ftype,ftangty}, dual_argtypes...}
    fm, _ = CC.findsup(frule_tt, CC.method_table(interp))
    (fm === nothing || !isa(fm.method, Method) || !is_generated_frule_fallback(fm.method)) && return nothing
    impl_tt = Tuple{typeof(dualized_impl), Dual{ftype,ftangty}, dual_argtypes...}
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    match === nothing && return nothing
    isa(match.method, Method) || return nothing
    return specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
end

# Resolve an arbitrary call whose dispatch tuple is `tt` to a `CodeInstance` for a static `:invoke`,
# or `nothing` if unresolved. Generic sibling of `frule_codeinstance`: same pipeline and invalidation
# edges. Used for the tangent-helper calls (`get_tangent_field`/`set_tangent_field!` fallbacks,
# `zero_tangent`) the direct-emission path can't take: a synthesized bare `Expr(:call, helper, …)`
# carries no `CallInfo`, so it can't be inlined and runs as a dynamic dispatch — an `:invoke` to a CI
# runs the compiled method directly instead.
function static_codeinstance(interp::ContextualInterpreter, @nospecialize(tt), edges::Vector{Any}=Any[])
    push!(edges, tt, Core.methodtable)   # mt-backedge: a new/more-specific method must invalidate
    m, _ = CC.findsup(tt, CC.method_table(interp))
    m === nothing && return nothing
    isa(m.method, Method) || return nothing
    mi = specialize_method(m.method, m.spec_types, m.sparams)::MethodInstance
    world = CC.get_inference_world(interp)
    ci = CC.typeinf_ext_toplevel(CC.NativeInterpreter(world), mi, CC.SOURCE_MODE_ABI)::CodeInstance
    CC.add_invoke_edge!(edges, tt, ci)
    return ci
end


# ===========================================================================
# The split-shadow dualization engine: post-optimization forward-AD on typed `IRCode`.
#
# `dualize_to_ircode` (called by `build_dual_ir` above) transforms a primal method's fully optimized
# `IRCode` into a dualized `IRCode`: the primal computation is reconstructed and a parallel tangent
# computation is emitted beside it, then packed into a `Dual`. Intrinsic/builtin/`%new` statements
# get direct per-construct rules; surviving `:invoke`/`:call`s go through `frule!!` dispatch.
#
# Types are derived directly from the primal IR, not guessed: every shadow statement shares its
# primal statement's type `Ti` under `tangent_type`. The result is a fully typed `IRCode` whose
# return type `finishinfer!` reads off directly — no re-inference. Block topology is preserved 1:1
# from the primal, so branches/loops need no CFG remapping. An unreachable `ReturnNode` block (a
# throw target) is reconstructed primal-only (`unreachable_block` below), so the derivative
# reproduces the same throw. try/catch is supported: `UpsilonNode`/`PhiCNode` duplicate into
# primal+shadow, `EnterNode`/`:leave`/`:pop_exception` carry over as control markers. Returns
# `nothing` to bail on unsupported constructs (an unrecognized `Core.Builtin`, a non-bits/
# undef-checked element access, `Core._apply_iterate` over an unknown-length splat).
# ===========================================================================

const _Intr = Core.Intrinsics

# ===========================================================================
# Activity analysis: which values may carry a derivative.
#
# `Inactive()` in a `Dual`'s tangent slot is the caller declaring that argument constant. Everything
# reached only through constants is replayed primally below — no shadow statement, no rule dispatch.
# The payoff is mostly *coverage*: an inactive call never reaches `frule_split!`, so undifferentiated
# bookkeeping (logging, `Dict` lookups, string handling) no longer bails the whole build.
#
# Ported from `DifferReverse`'s `_activity`; the reverse side's provenance/rdata machinery has no
# forward counterpart and is deliberately not ported.
# ===========================================================================

# Values whose primal or tangent this transform reads. Erring wide is safe: over-reporting costs a
# materialised zero nothing reads, under-reporting leaves an `Inactive()` where a rule wants a
# tangent. `:loopinfo`'s arguments aren't operands — `userefs` never visits them either.
function _each_operand(f, @nospecialize s)
    if isa(s, Core.PiNode) || isa(s, Core.UpsilonNode) || isa(s, Core.ReturnNode)
        isdefined(s, :val) && f(s.val)
    elseif isa(s, Core.PhiNode) || isa(s, Core.PhiCNode)
        vals = s.values
        for j in 1:length(vals)
            isassigned(vals, j) && f(vals[j])
        end
    elseif isa(s, Core.GotoIfNot)
        f(s.cond)
    elseif isa(s, Expr)
        if s.head === :invoke                     # args[1] is the CodeInstance/MethodInstance
            for j in 2:length(s.args); f(s.args[j]); end
        elseif s.head === :foreigncall            # args[2:5] are literal ABI descriptors
            f(s.args[1])
            for j in 6:length(s.args); f(s.args[j]); end
        elseif s.head === :new                    # args[1] is the constructed type
            for j in 2:length(s.args); f(s.args[j]); end
        elseif s.head === :throw_undef_if_not     # args[1] is a bare Symbol/GlobalRef name
            length(s.args) >= 2 && f(s.args[2])
        elseif s.head !== :loopinfo
            for j in 1:length(s.args); f(s.args[j]); end
        end
    elseif isa(s, Core.SSAValue) || isa(s, Core.Argument)
        f(s)
    end
    return nothing
end

# Shadow built out of its operands' shadows rather than computed fresh, so a `zero_shadow` can't
# stand in when materialised (a `PhiNode` must lead its block): materialise the operands instead.
_act_phi_like(@nospecialize s) =
    isa(s, Core.PiNode) || isa(s, Core.PhiNode) || isa(s, Core.PhiCNode) ||
    isa(s, Core.UpsilonNode)

# Primal replayable on its own, with no shadow beside it. Control-flow nodes and marker `Expr` heads
# carry no value and keep their existing arms; `:foreigncall` is never inactive to begin with.
_act_replayable(@nospecialize s) =
    _act_phi_like(s) ||
    (isa(s, Expr) ? (s.head === :call || s.head === :invoke || s.head === :new) :
     !(isa(s, Core.GotoNode) || isa(s, Core.GotoIfNot) || isa(s, Core.ReturnNode) ||
       isa(s, Core.EnterNode)))

# Which argument slots carry a derivative, in the *packed* space the primal IR's `Core.Argument`
# numbering lives in. A non-concrete `dualparams[k]` yields `Any`, which reads as active — sound.
#
# A vararg primal binds every trailing argument to one packed slot, so that slot is active if *any*
# trailing element is. Per-element constancy isn't modelled: the prologue materialises a zero for
# each constant element, keeping the packed tangent tuple at its primal-derived type.
function _arg_active(dualparams, nfixed::Int, nslots::Int, tt)
    aa = falses(nslots)
    live(D) = _dual_tangent_type(D) !== Inactive && tt(_dual_primal_type(D)) !== NoTangent
    for k in 1:min(nfixed, nslots)
        aa[k] = live(dualparams[k])
    end
    if nslots > nfixed
        # `dualparams` may be a `Core.SimpleVector` (straight off `specTypes`), so index it directly.
        aa[nslots] = any(j -> live(dualparams[j]), (nfixed + 1):length(dualparams))
    end
    return aa
end

# Which SSA values may carry a derivative. Monotone least fixpoint — a loop-carried `PhiNode` reads
# a back-edge value not yet computed — and the conservatism grows "may be active", so an
# unrecognised value-producing statement must default to *active*.
function _activity(pir, iworld::UInt, tt, nslots::Int, arg_active::BitVector)
    stmts = pir.stmts
    N = length(stmts)
    active = falses(N)
    operand_active(@nospecialize node) =
        isa(node, Core.SSAValue) ? active[node.id] :
        isa(node, Core.Argument) ? (node.n <= nslots && arg_active[node.n]) : false
    changed = true
    while changed
        changed = false
        for i in 1:N
            active[i] && continue
            s = stmts[i][:stmt]
            (isa(s, Core.GotoNode) || isa(s, Core.GotoIfNot) || isa(s, Core.ReturnNode) ||
             isa(s, Core.EnterNode)) && continue
            isa(s, Expr) && s.head in
                (:boundscheck, :loopinfo, :gc_preserve_begin, :gc_preserve_end) && continue
            # "Result has no tangent space ⇒ inactive" holds only for a *pure value producer*, so it
            # gates the alias/merge/`%new` arms and never a call: a generic call routinely returns
            # `Nothing` while writing through an argument (`Base._growend_internal!`, `copyto!`).
            # Reverse mode also exempts a rule-less `Core.Builtin`/intrinsic; forward mode must not —
            # such a bail is how `push!` (`Core.memoryrefoffset`) is refused, and replaying it
            # primally walks on into code the transform cannot handle.
            notan = tt(stmts[i][:type]) === NoTangent
            # `Union{} <: Ptr` is true, so excluding it keeps `throw`-typed statements from becoming
            # activity roots and dragging their operands into materialisation.
            Tw = _widen(stmts[i][:type])
            act = if (Tw isa Type && Tw !== Union{} && Tw <: Ptr) || _act_ptr_deref(s, iworld)
                true                      # raw pointer, or a load/store through one — see below
            elseif isa(s, Core.PiNode)
                !notan && operand_active(s.val)
            elseif isa(s, Core.PhiNode) || isa(s, Core.PhiCNode)
                vals = s.values
                !notan && any(j -> isassigned(vals, j) && operand_active(vals[j]), 1:length(vals))
            elseif isa(s, Core.UpsilonNode)
                !notan && isdefined(s, :val) && operand_active(s.val)
            elseif isa(s, Expr) && s.head === :new
                # An activity root, not a function of its initialiser operands: an active value may
                # be written into it further down.
                T = resolve_new_type_at(s.args[1], iworld)
                if T isa DataType && ismutabletype(T) && tt(T) !== NoTangent
                    true
                else
                    !notan && any(operand_active, @view s.args[2:end])
                end
            elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
                fpos = s.head === :invoke ? s.args[2] : s.args[1]
                actual = s.head === :invoke ? (@view s.args[3:end]) : (@view s.args[2:end])
                f = _calleeval(fpos, iworld)
                if f === Core.memorynew
                    true                      # the array-allocation half of the same root case
                else
                    operand_active(fpos) || any(operand_active, actual)
                end
            elseif isa(s, Expr) && s.head === :foreigncall
                # Always active: native code can write through any pointer it is handed, so operand
                # activity does not bound its effects.
                true
            elseif isa(s, Expr)
                true
            elseif isa(s, Core.SSAValue) || isa(s, Core.Argument)
                !notan && operand_active(s)
            else
                false
            end
            if act
                active[i] = true
                changed = true
            end
        end
    end
    return active
end

# Unconditionally active, for the same reason `:foreigncall` is: what a pointer addresses is outside
# this analysis, so how the *address* was computed does not bound what reading through it depends on.
# Without this `unsafe_load(Ptr{Float64}(u))` for a `u::UInt` comes out inactive and is silently
# zeroed, where forward mode deliberately bails.
function _act_ptr_deref(@nospecialize(s), iworld::UInt)
    (isa(s, Expr) && (s.head === :call || s.head === :invoke)) || return false
    f = _calleeval(s.head === :invoke ? s.args[2] : s.args[1], iworld)
    return f === _Intr.pointerref || f === _Intr.pointerset ||
           f === _Intr.atomic_pointerref || f === _Intr.atomic_pointerset
end

# `resolve_new_type` (inside `dualize_to_ircode`) without the closure — `%new`'s type argument is a
# `GlobalRef` whenever the struct is named by a module-level binding.
function resolve_new_type_at(@nospecialize(T), iworld::UInt)
    if isa(T, GlobalRef)
        ok, gv = _globalref_val(T, iworld)
        (ok && isa(gv, Type)) && return gv
    end
    return T
end

# Which inactive values still need a real zero shadow, because something active reads them. Emitting
# it at the point of *definition* — which dominates every use, including a phi edge — is what lets
# every hand, intrinsic and builtin rule stay untouched: `Inactive()` never reaches one.
#
# A fixpoint rather than a single scan: a phi-like statement's shadow is built from its operands, so
# materialising one materialises those too, and a loop-carried merge closes back on itself.
function _materialized(pir, active::BitVector, nslots::Int, arg_active::BitVector)
    stmts = pir.stmts
    N = length(stmts)
    mat = falses(N)
    arg_mat = falses(nslots)
    function note!(@nospecialize node)
        if isa(node, Core.SSAValue)
            if !active[node.id] && !mat[node.id]
                mat[node.id] = true
                return true
            end
        elseif isa(node, Core.Argument)
            node.n <= nslots && !arg_active[node.n] && (arg_mat[node.n] = true)
        end
        return false
    end
    for i in 1:N
        s = stmts[i][:stmt]
        # `ReturnNode` is never *active* (it produces no value of its own), but its operand's tangent
        # goes straight into the returned `Dual`, so seed from it explicitly.
        (active[i] || isa(s, Core.ReturnNode)) && _each_operand(note!, s)
    end
    changed = true
    while changed
        changed = false
        for i in 1:N
            s = stmts[i][:stmt]
            (mat[i] && _act_phi_like(s)) || continue
            _each_operand(n -> (note!(n) && (changed = true)), s)
        end
    end
    return mat, arg_mat
end


# `_calleeval`/`_globalref_val`/`_globalref_isconst` (world-parameterized `GlobalRef` resolution —
# load-bearing for the "world-age inside the generated `frule!!` body" reason documented on them)
# now live in `DifferCore/src/shared_ir_helpers.jl`, shared with `DifferReverse`.

# Build the dualized IRCode for `impl_mi` (a `dualized_impl` specialization) from the primal's
# optimized IRCode `pir`. `n` = number of flat dual arguments.
# `pir_is_vararg` selects the argument-unpacking prologue: `false` for an ordinary primal
# (first-order/base case), `true` when `pir` is itself an order-(k-1) dual IR (higher-order case).
# `primal_nfixed` is `nothing` for a non-vararg primal, or — when the primal method is itself vararg
# (`f(a, b, xs...)`) — the number of slots before the vararg one (`method.nargs - 1`). The prologue
# re-packs the trailing flat dual args into the single tuple slot `pir` expects, after which the rest
# of this transform needs no vararg awareness. Only meaningful with `pir_is_vararg=false`: an
# order-(k-1) dual carrier has already absorbed its own primal's vararg-ness into flat `argtypes`.
# Returns `ir::IRCode` or `nothing`.
function dualize_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int;
                           pir_is_vararg::Bool=false, pir_arg_offset::Int=0,
                           primal_nfixed::Union{Int,Nothing}=nothing,
                           reason::Ref{String}=Ref(""), edges::Vector{Any}=Any[])
    @assert !(pir_is_vararg && primal_nfixed !== nothing)   # internal invariant, not user input
    pstmts = pir.stmts
    N = length(pstmts)
    # Resolve `GlobalRef` callees/operands at the interpreter's *inference* world (see `_calleeval`):
    # the ambient generation world we run under can predate a user function's definition.
    iworld = CC.get_inference_world(interp)
    # Defensive: a leading PhiNode in block 1 would collide with the arg-extraction prologue, which
    # must land before any real phi. Not observed in practice (slot2ssa! keeps block 1 phi-free), but
    # bail rather than emit invalid IR if it ever occurs.
    if isa(pstmts[1][:stmt], Core.PhiNode)
        reason[] = "primal IR has a leading PhiNode in block 1 (unsupported shape)"
        return nothing
    end

    # Embed the actual (stable, singleton) function objects as literals rather than `GlobalRef`s to a
    # non-Core/Base module: `verify_ir` rejects a bare `GlobalRef` in value position unless its
    # binding is proven constant, which these Differ-module bindings aren't considered to be even
    # though they never change identity.
    zerotang_g   = zero_tangent       # runtime zero-tangent fallback for non-diff results
    buildtang_g  = build_tangent      # construct a struct's `Tangent`/`MutableTangent` shadow
    fruleg = frule!!
    Dualg  = Dual                # the `Dual` constructor, for a runtime (dynamic) pack of a non-concrete result
    dynfrule_g = dynamic_frule   # runtime dispatcher for a surviving dynamic (`apply_generic`) call
    dualimplg  = dualized_impl   # self-recursive edge target (`frule_split!`'s static self-`:invoke`)
    getf   = GlobalRef(Core, :getfield)
    ctuple = GlobalRef(Core, :tuple)
    intrg(name) = GlobalRef(Core.Intrinsics, name)

    # Tangent type of a primal type — drives every shadow SSA's declared type (scalars are a no-op,
    # structs get `Tangent`/`MutableTangent`, tuples a per-field tangent tuple, a `Dual` carrier
    # itself). `T` is usually a plain `Type`, but a const-prop-narrowed statement (e.g.
    # `Core.memorynew` with a literal length) can carry a `PartialStruct`/`Const` lattice element
    # instead — widen it first, since only the backing type matters for the tangent's type.
    #
    # `at_world`, not a direct call: `tangent_type` is extended by other packages (DifferReverse's
    # `Stack`/`Tape` methods) at a world strictly later than this generator's pin. A direct call
    # would silently take DifferCore's generic per-field fallback, which never terminates for a
    # self-referential type like `Tape`. `mt_edge!` so a later `tangent_type` method invalidates this
    # carrier. Shared with `builtins.jl`/`intrinsics.jl`/`foreigncalls.jl` via `builtin_ctx`/
    # `intrinsic_ctx` below.
    function tt(@nospecialize T)
        W = _widen(T)
        mt_edge!(edges, Tuple{typeof(tangent_type),Type{W}})
        return at_world(interp, tangent_type, W)
    end

    # Same treatment for the other two entry points this transform calls on primal types: `dual_type`
    # calls `tangent_type` internally, and `zero_tangent` dispatches to `zero_tangent_internal`
    # (extended by DifferReverse) — the *nested* dispatch has to land at `interp.world` too.
    # `dualt` gets its argument straight from `pstmts[i][:type]` (via `frule_split!`'s `R`), so it
    # sees the same lattice elements `tt` does — widen for the same reason.
    function dualt(@nospecialize T)
        W = _widen(T)
        mt_edge!(edges, Tuple{typeof(tangent_type),Type{W}})
        return at_world(interp, dual_type, W)
    end
    function zt(@nospecialize v)
        mt_edge!(edges, Tuple{typeof(zero_tangent),Core.Typeof(v)})
        return at_world(interp, zero_tangent, v)
    end

    # Forward-over-reverse coupling hooks, funnelled through `at_world` for the same world-age
    # reason (a direct call would resolve to the inert default). Shared by the `%new` arm and
    # `builtin_ctx`/`intrinsic_ctx`.
    function fsel_shadow_type(@nospecialize T)
        mt_edge!(edges, Tuple{typeof(_foreign_selfsim_shadow_type),Type{T}})
        return at_world(interp, _foreign_selfsim_shadow_type, T)
    end
    function fsel_shadow_field(@nospecialize(T), fi::Int)
        mt_edge!(edges, Tuple{typeof(_foreign_selfsim_shadow_field),Type{T},Int})
        return at_world(interp, _foreign_selfsim_shadow_field, T, fi)
    end
    function fsel_mirror_field(@nospecialize(T), fi::Int)
        mt_edge!(edges, Tuple{typeof(_foreign_selfsim_mirror_field),Type{T},Int})
        return at_world(interp, _foreign_selfsim_mirror_field, T, fi)
    end

    code = Any[]; types = Any[]; flags = UInt32[]
    # `flag` defaults to `CC.IR_FLAG_NULL`. The one caller that passes something else is the
    # self-recursive `:invoke` in `frule_split!` (`IR_FLAG_NOINLINE`, so the inliner doesn't try to
    # inline a self-call into itself).
    emit!(ex, @nospecialize(ty), flag::UInt32=CC.IR_FLAG_NULL) =
        (push!(code, ex); push!(types, ty); push!(flags, flag); Core.SSAValue(length(code)))
    opf(name, ty, args...) = emit!(Expr(:call, intrg(name), args...), ty)
    # Emit `f(args...)` as a static `:invoke` to a resolved `CodeInstance` when possible (runs the
    # compiled method directly instead of dynamic-dispatching a `CallInfo`-less synthesized `:call`),
    # falling back to a bare `:call` otherwise. `argtypes` builds the dispatch tuple.
    #
    # Only invoke when every `argtype` is concrete: a non-concrete argtype means dispatch is
    # genuinely runtime — `findsup` would resolve an over-general method, and freezing an `:invoke`
    # to that would both defeat dynamic dispatch and hit its unbound-static-param path.
    emit_invoke!(@nospecialize(f), @nospecialize(R), argtypes::Tuple, args...) = begin
        ci = all(_conc, argtypes) ?
                static_codeinstance(interp, Tuple{Core.Typeof(f), argtypes...}, edges) : nothing
        ci === nothing ? emit!(Expr(:call, f, args...), R) :
                         emit!(Expr(:invoke, ci, f, args...), R)
    end
    # Construct a `Dual{P,T}` directly with `%new` (no dispatch/allocation) rather than a dynamic
    # `Dual(...)` call the inliner couldn't reach in synthetic IR.
    dual!(@nospecialize(P), @nospecialize(T), @nospecialize(p), @nospecialize(t)) =
        emit!(Expr(:new, Dual{P,T}, p, t), Dual{P,T})
    # Whether a declared type is a specific leaf we can `%new` and freeze into an SSA's type.
    # `isconcretetype(Type{P})` is always `false` (documented Julia quirk), so a `Type{P}`-typed
    # operand takes the dynamic path below instead of a static `:invoke` — correct, just slower.
    _conc(@nospecialize T) = T isa DataType && isconcretetype(T)
    # Pack a primal/tangent into a `Dual` for a non-concrete primal type `R` (e.g. `Any`, from a
    # dynamic dispatch). A `%new(Dual{R,tt(R)}, …)` would freeze the over-wide declared type into the
    # value (a boxed `Dual{Any,Any}` that can't flow back into `frule!!`); call the `Dual` constructor
    # dynamically instead so it infers the concrete leaf type from the actual values
    # (exactly as `dynamic_frule`'s `map(Dual, …)` does). Declared type is the abstract `dual_type(R)`
    # — the UnionAll `Dual` for `R === Any`, or a `Union` of leaf `Dual`s for a `Union` `R` — of which
    # every runtime leaf is a subtype (whereas a `%new`-built `Dual{Union{…},…}` is not). Allocation
    # is moot: a non-concrete result already means the call dynamic-dispatched.
    dyn_dual!(@nospecialize(R), @nospecialize(p), @nospecialize(t)) =
        emit!(Expr(:call, Dualg, p, t), dualt(R))
    # `(true, value)` when `gr` names a defined constant binding at the inference world — the test
    # that licenses embedding its value and using its exact type. Separate flag rather than a
    # `nothing` sentinel: a binding whose value is `nothing` must still resolve. Asks constness at
    # `iworld`, never the ambient world.
    function gref_constval(gr::GlobalRef)
        ok, gv = _globalref_val(gr, iworld)
        return (ok && _globalref_isconst(gr, iworld)) ? (true, gv) : (false, nothing)
    end
    # A `GlobalRef` in value position (a `%new` field, an `:invoke` operand) is rejected by
    # `verify_ir` unless its module is `Core`/`Base` or its binding is proven constant (gotcha #2 in
    # the `differ-forward-dualization` skill). A defined `const` binding is embedded as its value (no
    # instruction); anything else becomes a real global-load instruction whose `SSAValue` is used
    # instead. Never cached across uses — a reused `SSAValue` from a non-dominating block would fail
    # `verify_ir`'s dominance check.
    function gref_operand!(gr::GlobalRef)
        ok, gv = gref_constval(gr)
        return ok ? gv : emit!(gr, Any)
    end
    # Declared type of what `gref_operand!` produces for `gr` (`Core.Typeof` of a `const` binding's
    # value; `Any` for a load).
    function gref_optype(gr::GlobalRef)
        ok, gv = gref_constval(gr)
        return ok ? Core.Typeof(gv) : Any
    end
    # A `%new` type argument named by a module-level binding lowers to a `GlobalRef`
    # (`%new(Main.S, …)`), not the type itself. Resolve it before anything downstream tests it with
    # `<:`/`fieldtype` (which throw a raw `TypeError` on a `GlobalRef`). Returns the input unchanged
    # when unresolvable, leaving the caller's `isa(T, Type)` check to catch it.
    function resolve_new_type(@nospecialize T)
        if isa(T, GlobalRef)
            ok, gv = _globalref_val(T, iworld)
            (ok && isa(gv, Type)) && return gv
        end
        return T
    end
    # The tangent of a compile-time-constant primal is its zero tangent, computed now and embedded as
    # a literal so no call survives into the IR.
    #
    # A bare `GlobalRef` operand is not itself a constant value — it names one. Splicing
    # `zero_tangent(globalref)` would compute the tangent of the `GlobalRef` struct (always
    # `NoTangent`), not of the bound value. Resolve the binding to learn the tangent type, then emit
    # a genuine runtime `zero_tangent` call on the re-embedded `GlobalRef` operand — never splice a
    # constructed tangent object as a compile-time literal: a mutable tangent (e.g.
    # `MutableTangent`) would be one frozen object shared across every invocation of this compiled
    # carrier, corrupted by the first call that mutates it.
    function const_tangent(@nospecialize x)
        isa(x, QuoteNode) && return zt(x.value)
        # The null shadow-pointer sentinel (`src/intrinsics.jl`): a `Ptr` operand with no tangent
        # storage behind it. `zero_tangent(::Ptr)` throws by design, but this one has a well-defined
        # tangent — a null shadow's own shadow is again null — which keeps IR containing it
        # re-dualizable at order ≥ 2.
        x === NULL_SHADOW_PTR && return NULL_SHADOW_PTR
        if isa(x, GlobalRef)
            # `_globalref_val`, not `_calleeval`: a binding holding `nothing` (e.g. `return nothing`)
            # must still resolve here, not fall through to `zero_tangent(gr)` (the GlobalRef struct's
            # own tangent, not the bound value's).
            ok, gv = _globalref_val(x, iworld)
            if ok
                T = tt(Core.Typeof(gv))
                # The `zero_tangent` argument goes through `gref_operand!`, never the raw node: an
                # `:invoke` operand is a value position too.
                return T === NoTangent ? NoTangent() :
                       emit_invoke!(zerotang_g, T, (Core.Typeof(gv),), gref_operand!(x))
            end
        end
        return zt(x)
    end
    # Zero tangent for a computed primal value of type `Ti`. `NoTangent()` when the tangent type is
    # trivial, a literal `zero(Ti)` for a concrete `Number`, otherwise a runtime `zero_tangent` call.
    # `Ti` may be a lattice element rather than a bare `Type` — widen it first, same as `tt`.
    function zero_shadow(@nospecialize(Ti), @nospecialize(primal_ssa))
        T = tt(Ti)
        T === NoTangent && return NoTangent()
        Tw = _widen(Ti)
        (Tw isa DataType && isconcretetype(Tw) && Tw <: Number) && return zero(Tw)::Tw
        return emit_invoke!(zerotang_g, T, (Tw,), primal_ssa)
    end

    dualparams = impl_mi.specTypes.parameters[2:end]     # the Dual{…} argument types
    vararg_tt = Tuple{dualparams...}                     # dualargs is one vararg tuple (Argument 2)

    primal = Vector{Any}(undef, N); shadow = Vector{Any}(undef, N)
    if !pir_is_vararg
        # Base case: `pir` has positional args — `Argument(k)` is its k-th declared slot. Unpack each
        # incoming `Dual` into its primal/tangent, indexed by slot so `presolve`/`tresolve` can treat
        # `Argument`s uniformly with `SSAValue`s.
        #
        # A vararg primal (`primal_nfixed !== nothing`) declares one extra slot holding its varargs
        # already packed into a tuple. The dual call is always flat, so the trailing dual args are
        # re-packed here into that tuple — after which the rest of this transform needs no vararg
        # awareness at all.
        nfixed = primal_nfixed === nothing ? n : primal_nfixed
        if nfixed > n
            reason[] = "the primal vararg method declares $nfixed argument slots before its vararg " *
                       "slot, but only $n dual arguments were supplied"
            return nothing
        end
        # Settle the packed tuple's primal/tangent types *before* emitting anything, so a shape
        # disagreement bails cleanly instead of leaving half-emitted statements behind.
        vptys = Any[_dual_primal_type(dualparams[i]) for i in (nfixed + 1):n]
        # A constant trailing element is materialised in the prologue below, so its slot in the
        # packed tangent tuple holds an ordinary zero of its primal-derived tangent type — the shape
        # check just below, and everything downstream, then sees no activity at all.
        vttys = Any[_dual_tangent_type(dualparams[i]) === Inactive ?
                        tt(_dual_primal_type(dualparams[i])) : _dual_tangent_type(dualparams[i])
                    for i in (nfixed + 1):n]
        Pva = Tuple{vptys...}
        Tva = tt(Pva)
        if primal_nfixed !== nothing
            # `pir`'s own vararg slot type is a lattice element (widen before comparing). `<:` rather
            # than `===`: the reconstruction can be legitimately sharper (`Core.Const((Float64,))`
            # widens to `Tuple{DataType}` where this rebuilds `Tuple{Type{Float64}}`), which is sound.
            # A genuine arity mismatch still fails. Bails rather than asserts, since this shape is
            # user-driven — an `AssertionError` out of `@generated frule!!` aborts compilation instead
            # of producing an `error_ircode` carrier that can report why.
            Pvaslot = CC.widenconst(pir.argtypes[nfixed + 1])
            if !(Pva <: Pvaslot)
                reason[] = "the reconstructed vararg tuple type $Pva does not match the primal " *
                           "method's own vararg slot type $Pvaslot"
                return nothing
            end
            # The tangent side is NOT simply the mirror of the primal tuple: `tangent_type` collapses
            # an all-`NoTangent` tuple (and `Tuple{}`) to plain `NoTangent`. `Dual{P,T}` requires
            # `T == tangent_type(P)`, so a collapsed slot must hold the literal `NoTangent()` — an
            # emitted `Core.tuple(NoTangent(), NoTangent())` would `TypeError` at the
            # `%new(Dual{P,NoTangent}, …)` `frule_split!` builds. Same rule as `Core.tuple`'s own
            # shadow. Nothing ever reads a tangent out of a collapsed slot either way.
            if !(Tva === NoTangent || Tva === Tuple{vttys...})
                reason[] = "the vararg tuple's tangent type $Tva is neither `NoTangent` nor the " *
                           "tuple $(Tuple{vttys...}) of its elements' tangent types"
                return nothing
            end
        end

        nslots = primal_nfixed === nothing ? n : nfixed + 1
        arg_active = _arg_active(dualparams, nfixed, nslots, tt)
        active = _activity(pir, iworld, tt, nslots, arg_active)
        mat, arg_mat = _materialized(pir, active, nslots, arg_active)
        parg = Vector{Any}(undef, nslots); targ = Vector{Any}(undef, nslots)
        argty = Vector{Any}(undef, nslots)
        pelts = Any[]; telts = Any[]
        for i in 1:n
            Di = dualparams[i]
            Pi = _dual_primal_type(Di)
            inact = _dual_tangent_type(Di) === Inactive
            di = emit!(Expr(:call, getf, Core.Argument(2), i), Di)
            p = emit!(Expr(:call, getf, di, 1), Pi)
            if i <= nfixed
                parg[i]  = p
                argty[i] = Pi
                # A constant argument reads no tangent at all: the `getfield(di, 2)` would be dead
                # code typed `Inactive`. One zero per call — not per use, not per iteration — when
                # something active still reads it.
                targ[i] = !inact ? emit!(Expr(:call, getf, di, 2), _dual_tangent_type(Di)) :
                          arg_mat[i] ? zero_shadow(Pi, p) : Inactive()
            else
                push!(pelts, p)
                # No tangent read at all for a collapsed vararg tangent — it would be dead code.
                # A constant trailing element gets a materialised zero rather than an `Inactive()`
                # slot, which keeps the packed tangent tuple at its primal-derived type `Tva` and
                # so needs no vararg awareness anywhere downstream.
                Tva === NoTangent ||
                    push!(telts, inact ? zero_shadow(Pi, p) :
                                 emit!(Expr(:call, getf, di, 2), _dual_tangent_type(Di)))
            end
        end
        if primal_nfixed !== nothing
            va = nfixed + 1
            parg[va]  = emit!(Expr(:call, ctuple, pelts...), Pva)
            targ[va]  = Tva === NoTangent ? NoTangent() : emit!(Expr(:call, ctuple, telts...), Tva)
            argty[va] = Pva
        end
    else
        # Higher-order case: `pir` is a vararg `dualized_impl` whose only real argument is
        # `Argument(2)`, the tuple of order-(k-1) dual args. The `pir_arg_offset` leading dual args
        # are non-value carrier function slots to skip (0 for a uniform seed; 1 for a re-dualized
        # `dualized_impl` invoke — see `build_dual_ir`). Each remaining `Di` is nested; peel one level
        # off each to reconstruct a primal tuple and a tangent tuple. The reconstructed tuple must
        # match `pir`'s own vararg-tuple arg type, so index `i` is offset-shifted while tuple position
        # `j` is offset-free. `pir` always has exactly two arguments, so `parg`/`targ` are length 2.
        nreal = n - pir_arg_offset
        pelts = Vector{Any}(undef, nreal); telts = Vector{Any}(undef, nreal)
        ptys  = Vector{Any}(undef, nreal); ttys  = Vector{Any}(undef, nreal)
        for j in 1:nreal
            i = j + pir_arg_offset
            Di = dualparams[i]
            if !(_dual_primal_type(Di) <: Dual)  # non-uniformly-nested seed at order ≥2 → bail
                reason[] = "argument #$i is not nested as a Dual at order ≥2 (a non-uniformly-nested higher-order seed)"
                return nothing
            end
            di = emit!(Expr(:call, getf, Core.Argument(2), i), Di)
            Pj = _dual_primal_type(Di)
            pelts[j] = emit!(Expr(:call, getf, di, 1), Pj);  ptys[j] = Pj
            # At order >= 2 an inactive seed degrades to a materialised zero: the nested tuple keeps
            # its primal-derived tangent type, so the re-dualized carrier needs no activity
            # awareness of its own.
            if _dual_tangent_type(Di) === Inactive
                telts[j] = zero_shadow(Pj, pelts[j]);                    ttys[j] = tt(Pj)
            else
                telts[j] = emit!(Expr(:call, getf, di, 2), _dual_tangent_type(Di))
                ttys[j] = _dual_tangent_type(Di)
            end
        end
        ptuple = emit!(Expr(:call, ctuple, pelts...), Tuple{ptys...})
        ttuple = emit!(Expr(:call, ctuple, telts...), Tuple{ttys...})
        @assert Tuple{ptys...} === pir.argtypes[2]     # reconstructed tuple == pir's own arg type
        parg = Any[dualized_impl, ptuple]; targ = Any[NoTangent(), ttuple]
        # `pir` here is a dual carrier this pass built itself (see `argtypes` at the bottom of this
        # function), so both slots are already plain `Type`s; spelled out for `optype` below.
        argty = Any[typeof(dualized_impl), Tuple{ptys...}]
        # Both slots are materialised above, so the analysis inside this carrier sees an ordinary
        # active argument list; the elision happened one order down.
        arg_active = _arg_active(Any[Dual{typeof(dualized_impl),NoTangent},
                                     Dual{Tuple{ptys...},Tuple{ttys...}}], 2, 2, tt)
        active = _activity(pir, iworld, tt, 2, arg_active)
        mat, arg_mat = _materialized(pir, active, 2, arg_active)
    end

    presolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? primal[x.id] : isa(x, Core.Argument) ? parg[x.n] : x
    # `presolve` for an operand landing in value position (a `%new` field): plain `presolve` passes a
    # `GlobalRef` through unchanged, which `verify_ir` rejects — resolve via `gref_operand!`. Call/
    # invoke argument positions don't need this: `frule_split!` already embeds a resolved value.
    vpresolve(@nospecialize x) = isa(x, GlobalRef) ? gref_operand!(x) : presolve(x)
    tresolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? shadow[x.id] :
        isa(x, Core.Argument) ? targ[x.n] :
        const_tangent(x)                                 # constant tangent (literal)
    # Declared primal type of an operand. `Argument`s go through `argty` rather than `pir.argtypes`
    # directly: (1) `pir.argtypes` holds lattice elements, not bare `Type`s, and results here are
    # consumed as genuine type parameters that throw on a non-`Type`; (2) for a vararg primal the
    # vararg slot's type is the tuple this pass reconstructed above; (3) a `GlobalRef` operand:
    # `_optype`'s literal fallback would report `GlobalRef`, the node's type, not the bound value's —
    # a silent miscompile (`getfield(Main.CONST_VEC, :ref)` would wrongly take the general-struct
    # branch). Take the type from the binding instead. `_optype` still handles `SSAValue`s/literals,
    # but an `SSAValue`'s statement type is itself a lattice element, not necessarily a bare `Type`;
    # widen it, since the result here is used as a genuine type parameter (`Tuple{…}`, `Dual{P,…}`).
    optype(@nospecialize x) = isa(x, Core.Argument) ? argty[x.n] :
                              isa(x, GlobalRef) ? gref_optype(x) : _widen(_optype(pir, x))

    function frule_split!(fpos, actual, R)
        # Dual(callee, NoTangent()) and each Dual(arg_primal, arg_tangent), constructed via %new.
        # Embed the resolved callee value rather than the raw AST node when it's statically known: a
        # bare `GlobalRef` outside Core/Base in value position is rejected by `verify_ir` unless its
        # binding is proven constant. A genuinely dynamic callee (an `SSAValue`/`Argument`) resolves
        # like any other operand via `presolve`/`_optype`.
        fval = _calleeval(fpos, iworld)
        fcallee = fval === nothing ? presolve(fpos) : fval
        ftype   = fval === nothing ? optype(fpos) : _typeof(fval)
        # A statically-known function is a code constant (zero tangent); a dynamic callee (read out
        # of a container) carries whatever tangent the shadow pass computed for it.
        ftang   = fval === nothing ? tresolve(fpos) : zt(fval)
        # A call is a genuine dynamic dispatch (`apply_generic`) when its callee or any argument has a
        # non-concrete declared type — the method that runs depends on runtime types unknowable here.
        # Can't wrap such a call statically: a `%new` of a `Dual` would freeze the abstract type into
        # the object, and `frule!!`'s `@generated` dispatch couldn't resolve a concrete primal from
        # `Dual{Any,…}`. Defer to the runtime `dynamic_frule` dispatcher: pass the callee's primal +
        # tangent and tuples of the argument primals/tangents, and let it rebuild concrete `Dual`s and
        # dispatch `frule!!` at run time. Result typed `Any`; extract primal (`R`) and tangent
        # (`tt(R)`). No branch on `R` needed: a call with concrete callee+args but an
        # inference-widened abstract result takes the static path below instead, annotated with the
        # abstract `dual_type(R)` — sound because `Dual` is invariant, so the concrete `Dual{Rc,Tc}`
        # the rule returns is `<: Dual` but not `<: Dual{Any,Any}`.
        if !_conc(ftype) || !all(a -> _conc(optype(a)), actual)
            # A statically-known operand (a `GlobalRef`/`QuoteNode`) must be embedded as its resolved
            # value, not the raw node — same `verify_ir` reason as the callee above. Resolve each
            # arg's primal, declared type, and compile-time-zero tangent together so the tuple's
            # element types match.
            pvals = Any[]; ptys = Any[]; tvals = Any[]; ttys = Any[]
            for a in actual
                v = _calleeval(a, iworld)
                if v === nothing                       # genuinely dynamic operand (SSAValue/Argument)
                    P = optype(a)
                    push!(pvals, presolve(a)); push!(ptys, P)
                    push!(tvals, tresolve(a)); push!(ttys, tt(P))
                else                                   # statically-known: embed the value + its zero
                    P = _typeof(v)
                    push!(pvals, v); push!(ptys, P)
                    push!(tvals, zt(v)); push!(ttys, tt(P))
                end
            end
            ptup = emit!(Expr(:call, ctuple, pvals...), Tuple{ptys...})
            ttup = emit!(Expr(:call, ctuple, tvals...), Tuple{ttys...})
            dd = emit!(Expr(:call, dynfrule_g, fcallee, ftang, ptup, ttup), Any)
            return emit!(Expr(:call, getf, dd, 1), R),
                   emit!(Expr(:call, getf, dd, 2), tt(R))
        end
        # `ftang` carries the callee's real tangent on both paths — a dynamic callee's capture
        # derivatives must not be silently zeroed just because the call is concrete enough to take
        # this static path.
        fd = dual!(ftype, tt(ftype), fcallee, ftang)
        # Each argument dual is `Dual{P, tangent_type(P)}`. A statically-known operand (a `GlobalRef`/
        # `QuoteNode`) must be embedded as its resolved value with `P = _typeof(value)`: e.g. an
        # intrinsic's leading type argument (`fpext(Base.Float64, …)`) is a `GlobalRef` whose value is
        # a `DataType` instance, and wrapping the raw node as `Dual{GlobalRef,…}` would TypeError at
        # the `%new`. `_typeof`, not plain `typeof`, is required: `_typeof(Float64) === Type{Float64}`
        # sharpens the type, needed for `Dual{Type{Float64},…}`-keyed `frule!!` dispatch to resolve.
        # Only genuinely dynamic operands fall back to `presolve`/`_optype`/`tresolve`.
        dualtys = Any[]; duals = Any[]
        for a in actual
            v = _calleeval(a, iworld)
            if v === nothing                            # genuinely dynamic operand
                P = optype(a)
                push!(dualtys, Dual{P,tt(P)})
                push!(duals, dual!(P, tt(P), presolve(a), tresolve(a)))
            else                                        # statically-known: embed value + its zero tangent
                P = _typeof(v)
                push!(dualtys, Dual{P,tt(P)})
                push!(duals, dual!(P, tt(P), v, zt(v)))
            end
        end
        # Emit the surviving high-level rule as a static `:invoke` to a compiled `CodeInstance` when
        # we can resolve one; otherwise a plain non-inlined `:call`. `dual_type(R)` is the declared
        # result type: exact for concrete `R`, the abstract `Dual` when `R` is non-concrete — the
        # concrete `Dual` the rule actually returns is a subtype of the latter (the invariant
        # `Dual{Any,Any}` would be unsound and miscompile).
        DR = dualt(R)
        # Recursion: does this call's *derived* carrier resolve to `impl_mi` itself (direct
        # self-recursion) or to some other carrier mid-compile on this task (mutual recursion)?
        # Checked before ever calling `frule_codeinstance`, which would otherwise
        # `typeinf_ext_toplevel` straight into the cycle `dualized_impl_in_progress` exists to catch.
        # Order matters: `impl_mi` is already a member of `dualized_impl_in_progress()`, so the
        # self-edge test must run first — checking `haskey` before `===` would misroute every
        # self-edge onto the mutual-recursion path below.
        callee_impl_mi = dual_recursive_impl_mi(interp, ftype, tt(ftype), dualtys)
        dd = if callee_impl_mi !== nothing && callee_impl_mi === impl_mi
            # Self-recursion: `impl_mi` is the bare, uncompiled `MethodInstance` currently being
            # dualized. `Expr(:invoke, mi, f, args...)` against a bare `MethodInstance` (not a
            # `CodeInstance`) is legal and fast ONLY for this exact case: codegen's `mi ==
            # ctx.linfo` self-recursion fast path emits a direct specsig call, no `CodeInstance`
            # needed. Never for a non-self target: a bare non-self MI degrades to a boxed
            # `jl_invoke` against the native method cache, silently running `dualized_impl`'s
            # throwing stub instead of the derivative (a wrong answer, not an error).
            # `dualized_impl(dualargs::Dual...)` takes the callee's own dual `fd` FIRST, matching
            # `frule!!`'s own convention. Omitting it is an arity mismatch against `impl_mi`'s
            # `specTypes` — caught the hard way once (a `verify_ir`-clean but run-time "Unreachable
            # reached" codegen crash).
            frule_tt = Tuple{typeof(frule!!), Dual{ftype,tt(ftype)}, dualtys...}
            push!(edges, frule_tt, Core.methodtable)   # a later hand frule!! must invalidate this derived choice
            CC.add_inlining_edge!(edges, callee_impl_mi)
            # `IR_FLAG_NOINLINE` so the inliner doesn't try to inline this self-call into itself —
            # surgical vs. widening `src_inlining_policy`, which would also block wanted inlining of
            # non-recursive dual carriers.
            emit!(Expr(:invoke, impl_mi, dualimplg, fd, duals...), DR, CC.IR_FLAG_NOINLINE)
        elseif callee_impl_mi !== nothing && haskey(dualized_impl_in_progress(), callee_impl_mi)
            # Mutual recursion: the callee's own carrier is mid-compile on this task but isn't this
            # build's own `impl_mi`, so no `CodeInstance` exists for it yet and the bare-MI self-invoke
            # trick doesn't apply. Fall back to a plain (uninlined) `:call` to `frule!!`, resolved at
            # run time against whatever `CodeInstance` the in-progress build eventually installs. This
            # is what breaks the compile-time cycle: any SCC stops recursing at the first back-edge
            # into an already-in-progress carrier.
            frule_tt = Tuple{typeof(frule!!), Dual{ftype,tt(ftype)}, dualtys...}
            push!(edges, frule_tt, Core.methodtable)
            emit!(Expr(:call, fruleg, fd, duals...), DR)
        else
            ci = frule_codeinstance(interp, ftype, tt(ftype), dualtys, edges)
            ci === nothing ? emit!(Expr(:call, fruleg, fd, duals...), DR) :
                             emit!(Expr(:invoke, ci, fruleg, fd, duals...), DR)
        end
        return emit!(Expr(:call, getf, dd, 1), R), emit!(Expr(:call, getf, dd, 2), tt(R))
    end

    # Bundle of closures `apply_intrinsic_frule!` (`src/intrinsics.jl`) needs to emit an intrinsic's
    # primal + shadow IR directly, without going through `frule_split!`'s `Dual`-boxing/`CodeInstance`
    # machinery. `reason` lets a rule that declines say why.
    intrinsic_ctx = (opf=opf, emit! =emit!, presolve=presolve, tresolve=tresolve, zero_shadow=zero_shadow,
                     optype=optype, tt=tt, reason=reason)
    # Same idea for `apply_builtin_frule!` (`src/builtins.jl`), which handles `Core.Builtin`s
    # (`getfield`, `setfield!`, `Core.tuple`, `Core.ifelse`, array allocation, `===`). `fsel_*` are the
    # forward-over-reverse coupling hooks funnelled through `at_world` — `getfield`/`setfield!` on a
    # foreign self-similar-shadow type need the same answers the `%new` arm gets.
    builtin_ctx = (emit! =emit!, presolve=presolve, tresolve=tresolve, vpresolve=vpresolve,
                   zero_shadow=zero_shadow, optype=optype, tt=tt, emit_invoke! =emit_invoke!,
                   opf=opf, reason=reason, fsel_shadow_type=fsel_shadow_type,
                   fsel_shadow_field=fsel_shadow_field, fsel_mirror_field=fsel_mirror_field)
    # And for `apply_foreigncall_frule!` (`src/foreigncalls.jl`). Two extra members for the
    # pointer-provenance walk: `pstmt` reads a primal statement back out of `pir` (old numbering), and
    # `calleeval` resolves a callee node so the walk can recognise `bitcast`/`getfield` in the chain.
    pstmt(@nospecialize x) = isa(x, Core.SSAValue) ? pstmts[x.id][:stmt] : nothing
    foreigncall_ctx = (emit! =emit!, presolve=presolve, tresolve=tresolve, zero_shadow=zero_shadow,
                       optype=optype, tt=tt, emit_invoke! =emit_invoke!, opf=opf, reason=reason,
                       pstmt=pstmt, calleeval=(@nospecialize(x) -> _calleeval(x, iworld)))

    # Block topology (count, order, preds/succs) is preserved 1:1 from the primal: this transform
    # only expands each statement into more instructions, never splits/merges/reorders blocks. So
    # `GotoNode.label`/`GotoIfNot.dest`/`PhiNode.edges` carry over unchanged; only each block's
    # `StmtRange` needs recomputing, tracked live below as blocks are crossed.
    nblocks = length(pir.cfg.blocks)
    block_start_new = Vector{Int}(undef, nblocks)
    block_start_new[1] = 1                          # includes the arg-extraction prologue above
    bidx = 1

    # A block terminating in an unreachable `ReturnNode` (no `val`) is an error/throw block: every
    # statement in it only ever leads to a `throw`, never a returned value or a live `PhiNode`
    # (dominance guarantees this). Reconstructed primal-only below: the derivative raises the same
    # error on the same inputs, with no shadow computed.
    unreachable_block = falses(nblocks)
    for b in 1:nblocks
        term = pstmts[pir.cfg.blocks[b].stmts.stop][:stmt]
        unreachable_block[b] = isa(term, Core.ReturnNode) && !isdefined(term, :val)
    end

    # Forward-reference patches for `PhiNode` operands not yet resolved (loop back-edges, defined
    # later in linear order). Keyed by the referenced original SSA index; each entry is
    # (target values-vector, slot, want_primal). `PhiNode.values` is mutable in place.
    pending = Dict{Int,Vector{Tuple{Vector{Any},Int,Bool}}}()

    # Resolve a `PhiNode`/`PhiCNode`'s `values` into parallel primal/shadow operand vectors. An
    # operand not yet processed in the linear walk registers a `pending` forward-reference instead.
    function resolve_phi_values(vals)
        k = length(vals)
        pvals = Vector{Any}(undef, k); tvals = Vector{Any}(undef, k)
        for j in 1:k
            isassigned(vals, j) || continue    # mirror the primal's own unassigned slot
            v = vals[j]
            if isa(v, Core.SSAValue) && !isassigned(primal, v.id)
                push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (pvals, j, true))
                push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (tvals, j, false))
            else
                pvals[j] = presolve(v); tvals[j] = tresolve(v)
            end
        end
        return pvals, tvals
    end

    for i in 1:N
        while bidx < nblocks && i > pir.cfg.blocks[bidx].stmts.stop
            # A block whose every statement is a pure alias emits no instructions, which would leave
            # an empty `StmtRange`. Backfill a placeholder so every block keeps at least one
            # statement, matching the primal's own convention.
            if length(code) < block_start_new[bidx]
                emit!(nothing, Nothing)
            end
            bidx += 1
            block_start_new[bidx] = length(code) + 1
        end
        s = pstmts[i][:stmt]; Ti = pstmts[i][:type]
        if unreachable_block[bidx]
            # Primal-only reconstruction. `shadow[i] = primal[i]` is a never-consumed placeholder so
            # `presolve`/`tresolve` on later error-path operands still resolve.
            if isa(s, Core.ReturnNode)
                emit!(Core.ReturnNode(), Union{})   # unreachable terminator
            elseif isa(s, Expr) && s.head === :invoke
                # Resolve the display-callee to its value: a bare non-Core/Base `GlobalRef` there is
                # a value position `verify_ir` rejects (see frule_split!). A dynamic callee resolves
                # like any other operand via `presolve`.
                fv = _calleeval(s.args[2], iworld)
                ex = Expr(:invoke, s.args[1], fv === nothing ? presolve(s.args[2]) : fv,
                          (presolve(a) for a in s.args[3:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :call
                # Resolve the callee too: a bare non-Core/Base `GlobalRef` (e.g. `throw`) fails the
                # const-binding check when re-embedded in this synthetic IR's world range.
                fv = _calleeval(s.args[1], iworld)
                ex = Expr(:call, fv === nothing ? presolve(s.args[1]) : fv,
                          (presolve(a) for a in s.args[2:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :new
                # Type argument and field operands get the same `GlobalRef` resolution as the
                # live-path `:new` arm below — a throw block builds exception objects, and
                # `%new(Main.MyError, …)` is exactly the shape that trips it.
                T = resolve_new_type(s.args[1])
                ex = Expr(:new, T, (vpresolve(a) for a in s.args[2:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :boundscheck
                primal[i] = emit!(Expr(:boundscheck, (presolve(a) for a in s.args)...), Ti)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :throw_undef_if_not
                # `args[1]` is a bare Symbol/GlobalRef name, copied through verbatim; `args[2]` is the
                # non-differentiable Bool condition, resolved uniformly with the live-path arm below.
                # No shadow value: never consumed.
                primal[i] = emit!(Expr(:throw_undef_if_not, s.args[1], presolve(s.args[2])), Ti)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :gc_preserve_begin
                # Primal-only: nothing in a throw-only block carries a tangent, so no shadow to root.
                primal[i] = emit!(Expr(:gc_preserve_begin, (presolve(a) for a in s.args)...), Any)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :gc_preserve_end
                primal[i] = emit!(Expr(:gc_preserve_end, presolve(s.args[1])), Ti)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :foreigncall
                # Primal-only reconstruction. `args[2:5]` are literals copied verbatim; `args[1]`
                # usually is too, but the runtime-function-pointer form puts an `SSAValue` (or a bare
                # `GlobalRef`) there, which needs resolving like any other operand.
                nm = s.args[1]
                pnm = isa(nm, Expr) ? Expr(nm.head, nm.args...) : presolve(nm)
                primal[i] = emit!(Expr(:foreigncall, pnm, s.args[2], s.args[3], s.args[4], s.args[5],
                                       (presolve(a) for a in s.args[6:end])...), Ti)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :loopinfo
                primal[i] = emit!(Expr(:loopinfo, s.args...), Ti); shadow[i] = primal[i]
            elseif isa(s, Core.PiNode)
                primal[i] = presolve(s.val); shadow[i] = primal[i]
            elseif isa(s, GlobalRef)
                # A bare GlobalRef statement is a global-variable load, not a pure alias — must be
                # emitted as a real instruction (see the main-loop GlobalRef case below).
                primal[i] = emit!(s, Ti); shadow[i] = primal[i]
            elseif !isa(s, Expr)
                primal[i] = presolve(s); shadow[i] = primal[i]
            else
                reason[] = "unexpected statement kind $(typeof(s)) in an unreachable (throw-only) " *
                           "block at %$i: `$(_stmt_str(s))`"
                return nothing
            end
        elseif !active[i] && !_act_phi_like(s) && _act_replayable(s)
            # Nothing downstream of this statement contributes to any derivative — every value it
            # depends on was declared constant. Reconstruct the primal faithfully (kept for its own
            # sake: effects, or simply not inlined) and emit no shadow computation at all. This is
            # where the coverage payoff lives: the statement never reaches `frule_split!`, so a
            # callee the transform could not have differentiated does not bail the whole build.
            if isa(s, Expr) && (s.head === :call || s.head === :invoke)
                fpos = s.head === :invoke ? s.args[2] : s.args[1]
                actual = s.head === :invoke ? s.args[3:end] : s.args[2:end]
                fv = _calleeval(fpos, iworld)
                fcallee = fv === nothing ? presolve(fpos) : fv
                ex = s.head === :invoke ?
                        Expr(:invoke, s.args[1], fcallee, (presolve(a) for a in actual)...) :
                        Expr(:call, fcallee, (presolve(a) for a in actual)...)
                primal[i] = emit!(ex, Ti)
            elseif isa(s, Expr) && s.head === :new
                primal[i] = emit!(Expr(:new, resolve_new_type(s.args[1]),
                                       (vpresolve(a) for a in s.args[2:end])...), Ti)
            elseif isa(s, GlobalRef)
                primal[i] = emit!(s, Ti)          # a global load, not a pure alias
            else
                primal[i] = presolve(s)           # literal / SSAValue / Argument alias
            end
            # A zero only where something active reads it — `_materialized` decides, and emitting it
            # here (at the definition, which dominates every use) is what keeps `Inactive()` out of
            # every rule.
            shadow[i] = mat[i] ? zero_shadow(Ti, primal[i]) : Inactive()
        elseif !active[i] && !mat[i] && _act_phi_like(s)
            # Same idea for a merge, but a merge's shadow is built from its operands rather than
            # computed fresh, so there is nothing to replay — just emit the primal half and leave the
            # shadow as `Inactive()`. A loop-carried inactive value therefore costs no shadow phi at
            # all. (Materialised phi-likes fall through to the ordinary arms below: `_materialized`
            # has already materialised every operand they read, so `tresolve` finds real tangents.)
            if isa(s, Core.PiNode)
                primal[i] = presolve(s.val)                  # pure alias, no instruction
            elseif isa(s, Core.PhiNode)
                pvals, _ = resolve_phi_values(s.values)      # tangent half resolved and discarded
                primal[i] = emit!(Core.PhiNode(s.edges, pvals), Ti)
            elseif isa(s, Core.PhiCNode)
                pvals, _ = resolve_phi_values(s.values)
                primal[i] = emit!(Core.PhiCNode(pvals), Ti)
            else
                primal[i] = isdefined(s, :val) ?
                                emit!(Core.UpsilonNode(presolve(s.val)), Ti) :
                                emit!(Core.UpsilonNode(), Ti)
            end
            shadow[i] = Inactive()
        elseif isa(Ti, Core.Const) && isa(s, Expr) && (s.head === :call || s.head === :invoke)
            # Const-prop proved this call's result is always exactly this literal, so its derivative
            # is definitionally zero regardless of the callee. (Only `Core.Const` licenses this — a
            # `PartialStruct` narrows some fields but doesn't pin the whole value.) Still reconstruct
            # the call faithfully (kept for its own sake — effects, or simply not inlined), but skip
            # the intrinsic/builtin/`frule!!` dispatch for the shadow, straight to `zero_shadow`.
            fpos = s.head === :invoke ? s.args[2] : s.args[1]
            actual = s.head === :invoke ? s.args[3:end] : s.args[2:end]
            fv = _calleeval(fpos, iworld)
            fcallee = fv === nothing ? presolve(fpos) : fv
            ex = s.head === :invoke ?
                    Expr(:invoke, s.args[1], fcallee, (presolve(a) for a in actual)...) :
                    Expr(:call, fcallee, (presolve(a) for a in actual)...)
            primal[i] = emit!(ex, Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif isa(s, Core.ReturnNode)
            if !isdefined(s, :val)
                emit!(Core.ReturnNode(), Union{})   # unreachable terminator
            else
                if isa(s.val, GlobalRef)
                    # `return <global>` — e.g. `return nothing`, which survives optimization as the
                    # bare GlobalRef `Main.nothing`, not a literal. `presolve` would pass the raw node
                    # into the returned `Dual`'s primal field, which `verify_ir` rejects; `_optype`
                    # would report `GlobalRef`, not the bound value's type. `gref_operand!`/
                    # `gref_optype` resolve both. A defined `const` binding keeps the allocation-free
                    # `%new` path (`Nothing` for `return nothing`); otherwise falls to `Any`/dynamic pack.
                    gr = s.val::GlobalRef
                    p = gref_operand!(gr)
                    R = gref_optype(gr)
                else
                    p = presolve(s.val)
                    R = optype(s.val)
                end
                t = tresolve(s.val)
                # Concrete `R`: pack with an allocation-free `%new` of the exact `Dual{R,tt(R)}`.
                # Non-concrete `R` (a dynamic-dispatch result, e.g. `Any`): build the `Dual` dynamically
                # so the runtime type is the concrete leaf, not a frozen `Dual{Any,Any}`.
                res = _conc(R) ? dual!(R, tt(R), p, t) : dyn_dual!(R, p, t)
                emit!(Core.ReturnNode(res), Any)
            end
        elseif isa(s, Core.PiNode)
            primal[i] = presolve(s.val); shadow[i] = tresolve(s.val)
        elseif isa(s, Expr) && s.head === :new
            # The constructed type is a `GlobalRef` whenever the struct is named by a module-level
            # binding (`%new(Main.S, …)`). Must be resolved before anything else in this arm: every
            # branch below tests it with `<:`/`fieldtype`, which throw a raw `TypeError` on a
            # `GlobalRef`, and it's a `verify_ir`-checked value position too. `isa(gv, Type)` is the
            # actual guard, not the `const` test field operands get: a struct name is always constant.
            T = resolve_new_type(s.args[1])
            if !isa(T, Type)
                reason[] = "`%new` whose type argument `$(s.args[1])` is not a statically-known " *
                           "type at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            args = @view s.args[2:end]
            # `vpresolve`, not `presolve`: a field operand is a value position, so a `GlobalRef` there
            # must be resolved — same issue as the type argument above.
            pf = Any[vpresolve(a) for a in args]
            primal[i] = emit!(Expr(:new, T, pf...), Ti)
            TT = tt(Ti)
            if T <: Dual
                # `Dual` is its own tangent type: shadow is a same-typed `Dual` built via `%new`.
                # A non-differentiable field (tangent `NoTangent`, can't fill e.g. a `typeof(sin)` or
                # `Int` slot) carries the primal value through unchanged instead — what lets a
                # `Dual{typeof(sin),NoTangent}` be re-dualized at higher order.
                tf = Any[_nondiff_field(interp, edges, fieldtype(T, j)) ? vpresolve(args[j]) : tresolve(args[j])
                         for j in eachindex(args)]
                # `_widen(Ti)`, not `Ti`: `Ti` is the *primal*'s statement type. If it's a
                # `PartialStruct` pinning a field, that fact is false for the shadow, whose fields are
                # tangents, not primal values.
                shadow[i] = emit!(Expr(:new, T, tf...), _widen(Ti))
            elseif fsel_shadow_type(T)
                # Self-similar-shadow bookkeeping types owned by a different AD-mode package (e.g.
                # DifferReverse's `Stack`/`Tape`) — reached under forward-over-reverse, whose own
                # constructor calls get inlined into raw `%new`s before this dualizer sees them. Same
                # shape as the `Dual` arm above, generalized per field via `_foreign_selfsim_shadow_field`
                # (the same predicate `getfield`/`setfield!` use, so shadows never disagree on which
                # fields mirror). A declined field carries the primal value through instead.
                tf = Any[fsel_shadow_field(T, j) === nothing ? vpresolve(args[j]) : tresolve(args[j])
                         for j in eachindex(args)]
                shadow[i] = emit!(Expr(:new, TT, tf...), TT)
            elseif TT === NoTangent
                # Whole aggregate is non-differentiable (e.g. `Tuple{Int,Int}`, a fieldless struct).
                shadow[i] = NoTangent()
            elseif T <: Tuple || T <: NamedTuple
                # Same-shape but tangent-typed: a non-differentiable slot holds `NoTangent()`,
                # differentiable slots hold their tangent.
                tf = Any[tt(fieldtype(T, j)) === NoTangent ? NoTangent() : tresolve(args[j])
                         for j in eachindex(args)]
                shadow[i] = emit!(Expr(:new, TT, tf...), TT)
            elseif T <: Array
                # Array construction, step 4/4 of the allocation sequence (see `Core.memorynew` below
                # for the other end): `Expr(:new, Vector{P}, ref, size)`.
                # `tangent_type(Array{P,N}) === Array{tangent_type(P),N}`, so `TT` is directly the
                # concrete tangent type to `%new`. Exactly 2 fields:
                #  * `:ref` is differentiable (a `MemoryRef{P}` into the shadow memory allocated by
                #    `Core.memorynew`) -> `tresolve` to the shadow `MemoryRef{tangent_type(P)}`.
                #  * `:size` is structural, non-differentiable -> use the primal's own size tuple
                #    verbatim (`presolve`), matching Mooncake's reference `_new_` rule for `Array{P,N}`.
                shadow[i] = emit!(Expr(:new, TT, tresolve(args[1]), vpresolve(args[2])), TT)
            else
                # General (im)mutable user struct → `Tangent`/`MutableTangent`. `tresolve`d field
                # tangents already carry `NoTangent()` for non-diff fields. When every slot is
                # always-initialised and `%new` supplies a value for all of them, build the backing
                # `NamedTuple` and wrap it with `%new` directly — no `build_tangent` call to
                # dynamic-dispatch. Otherwise fall back to `build_tangent` for the
                # `PossiblyUninitTangent` wrapping and empty-slot construction the direct `%new` can't.
                tf = Any[tresolve(args[j]) for j in eachindex(args)]
                NT = (TT isa DataType && TT <: Union{Tangent,MutableTangent}) ? fields_type(TT) : nothing
                if NT isa DataType && isconcretetype(NT) && length(args) == fieldcount(NT) &&
                   !any(j -> fieldtype(NT, j) <: PossiblyUninitTangent, 1:fieldcount(NT))
                    nt = emit!(Expr(:new, NT, tf...), NT)
                    shadow[i] = emit!(Expr(:new, TT, nt), TT)
                else
                    shadow[i] = emit!(Expr(:call, buildtang_g, T, tf...), TT)
                end
            end
        elseif isa(s, Expr) && s.head === :boundscheck
            # Marks a compile-time-decided bounds-check mode, consulted downstream by a `GotoIfNot`/
            # boundscheck argument. Emitted unchanged; result is a non-differentiable `Bool`.
            primal[i] = emit!(Expr(:boundscheck, (presolve(a) for a in s.args)...), Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif isa(s, Expr) && s.head === :throw_undef_if_not
            # Raises `UndefVarError`/`UndefRefError` for an unassigned slot or boxed-capture field.
            # `args[1]` is a bare Symbol/GlobalRef name, copied through verbatim; `args[2]` is the
            # Bool condition and needs resolving. No shadow-bearing value, so the zero tangent.
            primal[i] = emit!(Expr(:throw_undef_if_not, s.args[1], presolve(s.args[2])), Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos = s.head === :invoke ? s.args[2] : s.args[1]
            actual = s.head === :invoke ? s.args[3:end] : s.args[2:end]
            f = _calleeval(fpos, iworld)
            if isa(f, Core.IntrinsicFunction)
                # Dispatch straight to a per-intrinsic rule (`apply_intrinsic_frule!` in
                # `src/intrinsics.jl`), which emits the primal + shadow IR directly — no `Dual`
                # boxing, `frule!!` dispatch, or `CodeInstance` resolution. The fallback method
                # returns `nothing`, so an unregistered intrinsic bails gracefully with a located reason.
                why = reason[]
                res = apply_intrinsic_frule!(Val(f), actual, Ti, intrinsic_ctx)
                if res === nothing
                    # A registered rule can also decline (the pointer rules do, on a stride mismatch),
                    # recording its own reason in `ctx.reason`. Only claim "no rule registered" when
                    # the rule left the reason untouched, or the message would be a lie.
                    reason[] = reason[] === why ?
                        "unsupported intrinsic `$(nameof(f))` at %$i: `$(_stmt_str(s))` " *
                        "(no rule registered; add one in src/intrinsics.jl via " *
                        "`apply_intrinsic_frule!`)" :
                        "$(reason[]) at %$i: `$(_stmt_str(s))`"
                    return nothing
                end
                primal[i], shadow[i] = res
            elseif isa(f, Core.Builtin)
                # Dispatch straight to a per-builtin rule (`apply_builtin_frule!` in
                # `src/builtins.jl`), mirroring the intrinsic dispatch above. Unregistered builtin
                # (e.g. `Core.memoryrefoffset`, or a non-bits/undef-checked element access) bails
                # gracefully with a located reason.
                why = reason[]
                res = apply_builtin_frule!(Val(f), actual, Ti, builtin_ctx)
                if res === nothing
                    # As with intrinsics above, a registered rule that declines records its own reason.
                    reason[] = reason[] === why ?
                        "no dualization rule for builtin `$f` (e.g. `Core.memoryrefoffset` used " *
                        "by `push!`/`resize!`, or a non-bits/undef-checked array element " *
                        "access) at %$i: `$(_stmt_str(s))`" :
                        "$(reason[]) at %$i: `$(_stmt_str(s))`"
                    return nothing
                end
                primal[i], shadow[i] = res
            else
                res = frule_split!(fpos, actual, Ti)
                res === nothing && return nothing            # dynamic dispatch: bail (see frule_split!)
                primal[i], shadow[i] = res
            end
        elseif isa(s, Core.GotoNode)
            emit!(Core.GotoNode(s.label), Any)
        elseif isa(s, Core.GotoIfNot)
            emit!(Core.GotoIfNot(presolve(s.cond), s.dest), Any)
        elseif isa(s, Core.PhiNode)
            pvals, tvals = resolve_phi_values(s.values)
            primal[i] = emit!(Core.PhiNode(s.edges, pvals), Ti)
            shadow[i] = emit!(Core.PhiNode(s.edges, tvals), tt(Ti))
        elseif isa(s, Core.UpsilonNode)
            # try/catch: a value live into a handler is captured by an `UpsilonNode` and collected by
            # a `PhiCNode` at the handler top. Duplicate into a primal + shadow upsilon, like a
            # `PhiNode`. An unassigned `ϒ ()` slot mirrors as an empty upsilon in both copies.
            if isdefined(s, :val)
                primal[i] = emit!(Core.UpsilonNode(presolve(s.val)), Ti)
                shadow[i] = emit!(Core.UpsilonNode(tresolve(s.val)), tt(Ti))
            else
                primal[i] = emit!(Core.UpsilonNode(), Ti)
                shadow[i] = emit!(Core.UpsilonNode(), tt(Ti))
            end
        elseif isa(s, Core.PhiCNode)
            # Collects `UpsilonNode`s at a handler entry. Operands are `SSAValue`s referencing
            # upsilons, so the primal-phic references primal upsilons and the shadow-phic shadow
            # ones. Same `resolve_phi_values`/`pending` mechanism as `PhiNode`.
            pvals, tvals = resolve_phi_values(s.values)
            primal[i] = emit!(Core.PhiCNode(pvals), Ti)
            shadow[i] = emit!(Core.PhiCNode(tvals), tt(Ti))
        elseif isa(s, Core.EnterNode)
            # Begins a protected region. `catch_dest` is a basic-block number, carried over unchanged;
            # its type must be `Any` (it terminates its block). `scope`, if present, is an ordinary
            # value operand.
            ent = isdefined(s, :scope) ?
                    Core.EnterNode(s.catch_dest, presolve(s.scope)) : Core.EnterNode(s.catch_dest)
            primal[i] = emit!(ent, Any); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :leave
            # Pops one or more `:enter` scopes, referencing the `EnterNode`(s) by `SSAValue` (or
            # `nothing`); `presolve` remaps each to the enter's primal SSA.
            emit!(Expr(:leave, Any[presolve(a) for a in s.args]...), Any)
        elseif isa(s, Expr) && s.head === :pop_exception
            primal[i] = emit!(Expr(:pop_exception, presolve(s.args[1])), Ti); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :the_exception
            # The caught exception object has no meaningful tangent: keep the primal, zero shadow.
            primal[i] = emit!(Expr(:the_exception), Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif isa(s, Expr) && s.head === :foreigncall
            # `ccall`. Dispatched per target symbol to `apply_foreigncall_frule!`
            # (`src/foreigncalls.jl`). Native code can write through any pointer it's handed, so
            # unlike an unregistered intrinsic there's no safe "primal + zero tangent" fallback: an
            # unregistered target bails outright.
            fc = _fc_parse(s)
            if fc === nothing
                reason[] = "`foreigncall` with a non-literal target (a runtime function pointer) " *
                           "at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            why = reason[]
            res = apply_foreigncall_frule!(Val(fc.name), fc, Ti, foreigncall_ctx)
            if res === nothing
                # As with intrinsics/builtins, a declining rule records its own reason.
                reason[] = reason[] === why ?
                    "no dualization rule for `foreigncall` target `$(fc.name)` at %$i: " *
                    "`$(_stmt_str(s))` (add one in src/foreigncalls.jl via " *
                    "`apply_foreigncall_frule!`)" :
                    "$(reason[]) at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            primal[i], shadow[i] = res
        elseif isa(s, Expr) && s.head === :loopinfo
            # `@simd`'s loop marker. Pure metadata: `:loopinfo` isn't in the compiler's
            # `is_relevant_expr`, so its operands (`Symbol`s/`nothing`) must never be resolved.
            # Copied through verbatim, no shadow of its own.
            #
            # Two invariants: codegen consumes the marker positionally from the block's terminator
            # backwards, so nothing may be emitted between it and the terminator (this pass emits
            # shadow statements before it — fine); and a `julia.ivdep` marker now also asserts
            # non-aliasing for the shadow's accesses, true whenever it was true of the primal since
            # the shadow mirrors the primal's access pattern one-for-one.
            primal[i] = emit!(Expr(:loopinfo, s.args...), Ti); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :gc_preserve_begin
            # `GC.@preserve`: roots its operands until the matching `:gc_preserve_end`. The dualized
            # code holds interior pointers into both the primal object and its shadow, so root both —
            # preserving only the primal would let the shadow array be collected while a shadow `Ptr`
            # into it is still live. A tangent that isn't an SSA/argument is skipped: no heap object
            # of ours to root.
            pargs = Any[]
            for a in s.args
                push!(pargs, presolve(a))
                t = tresolve(a)
                (isa(t, Core.SSAValue) || isa(t, Core.Argument)) && push!(pargs, t)
            end
            primal[i] = emit!(Expr(:gc_preserve_begin, pargs...), Any); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :gc_preserve_end
            # Ends the region, referencing the `:gc_preserve_begin` token by `SSAValue`. `verify_ir`
            # skips its usual dominance check for this head (a token may span try/catch blocks).
            primal[i] = emit!(Expr(:gc_preserve_end, presolve(s.args[1])), Ti); shadow[i] = primal[i]
        elseif isa(s, GlobalRef)
            # A bare GlobalRef statement is a global-variable load, not a pure alias like a `PiNode`:
            # must be emitted as a real instruction, or the raw `GlobalRef` leaks into a later operand
            # position, which `verify_ir` rejects unless the binding is proven constant. Non-diff
            # external state, so zero tangent.
            primal[i] = emit!(s, Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif !isa(s, Expr)
            primal[i] = presolve(s); shadow[i] = tresolve(s)
        else
            reason[] = "unsupported statement kind $(typeof(s)) at %$i: `$(_stmt_str(s))`"
            return nothing
        end
        if haskey(pending, i)
            for (arr, slot, wantp) in pending[i]
                arr[slot] = wantp ? primal[i] : shadow[i]
            end
            delete!(pending, i)
        end
    end
    if length(code) < block_start_new[nblocks]      # the final block emitted nothing; see above
        emit!(nothing, Nothing)
    end
    if !isempty(pending)  # unreachable on well-formed IR; bail, don't emit invalid IR
        reason[] = "internal error — an unresolved forward-reference remained (a bug in this transform, not unsupported input)"
        return nothing
    end

    len = length(code)
    stream = CC.InstructionStream(len)
    for i in 1:len
        stream.stmt[i] = code[i]
        stream.type[i] = types[i]
        stream.flag[i] = flags[i]
    end
    new_blocks = Vector{CC.BasicBlock}(undef, nblocks)
    for b in 1:nblocks
        lo = block_start_new[b]
        hi = b == nblocks ? len : block_start_new[b+1] - 1
        ob = pir.cfg.blocks[b]
        new_blocks[b] = CC.BasicBlock(CC.StmtRange(lo, hi), copy(ob.preds), copy(ob.succs))
    end
    cfg = CC.CFG(new_blocks, Int[bb.stmts.stop + 1 for bb in new_blocks])   # index[b] is one past block b's end
    di  = CC.DebugInfoStream(stream.line)
    di.def = impl_mi                                # required: Core.DebugInfo(di, n) does something(di.def)
    argtypes = Any[impl_mi.specTypes.parameters[1], vararg_tt]
    # See `carrier_world_range` (Contextual.jl) for why this world range.
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[], carrier_world_range(interp, pir))
    CC.verify_ir(ir)                                # a failure here is a bug in this transform, not
                                                     # unsupported input IR — let it throw plainly
    return ir
end

# `_optype`/`_optype_w`/`_stmt_str` now live in `DifferCore/src/shared_ir_helpers.jl`, shared with
# `DifferReverse`.

# A field's shadow carries the primal value through, rather than its tangent, exactly when the
# field's tangent is `NoTangent` but its slot type would reject a `NoTangent` value.
# `tangent_type(T) === NoTangent` is the principled test (every non-`NoTangent` singleton, `Int`,
# `Bool`, `Symbol`, `Tuple{Int,Int}`, … would otherwise get a spurious `NoTangent()`). Two guards:
#  * `isconcretetype(T)` — only concrete types have a total, non-throwing `tangent_type`; an abstract
#    slot (`::Integer`, `::Any`) conservatively stays on the tangent path.
#  * `T <: Type` handled separately — `Type{Float64}`/`DataType` are `NoTangent` but not concrete
#    (`isconcretetype(Type{Float64}) === false`), and not singleton either (a documented Julia quirk).
#
# Takes `interp`/`edges` so its `tangent_type` query lands at the interpreter's inference world.
function _nondiff_field(interp::ContextualInterpreter, edges::Vector{Any}, @nospecialize(T))
    (T isa DataType && T !== NoTangent) || return false
    T <: Type && return true
    isconcretetype(T) || return false
    mt_edge!(edges, Tuple{typeof(tangent_type),Type{T}})
    return at_world(interp, tangent_type, T) === NoTangent
end


function frule_body(world::UInt, source, self, dual_argtypes)
    argnames = Any[Symbol("#self#"), :dualargs]

    # Resolve the `dualized_impl` specialization for these dual argument types.
    impl_tt = Tuple{typeof(dualized_impl), dual_argtypes...}
    interp = ContextualInterpreter(Forward(), nothing; world)
    match, _ = Core.Compiler.findsup(impl_tt, Core.Compiler.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                                :(error("Differ: no dualized_impl match")), true)
    end
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance

    # Compile the dualized body under ContextualInterpreter -> an invoke-able CodeInstance.
    # `typeinf_ext_toplevel` directly, not a wrapper that would recreate the interpreter at the
    # (stale, inside a generator) ambient world — must use `interp`'s own generation world.
    cinst = Compiler.typeinf_ext_toplevel(interp, impl_mi, Compiler.SOURCE_MODE_ABI)

    # Trivial generated body: return invoke(dualized_impl, cinst, dualargs...)
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(dualized_impl, $cinst, dualargs...)), true)

    # `impl_mi` is a real backedge: if its own CodeInstance is invalidated (where the primal/
    # `frule!!` dependencies actually get registered), this wrapper must regenerate too.
    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

function refresh_frule()
    @eval function frule!!(dualargs::Dual...)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, frule_body))
    end
end
refresh_frule()

"""
    frule!!(fdual::Dual, argduals::Dual...) -> Dual

Forward-mode rule for `primal(fdual)(primal.(argduals)...)`, returning the result and its
directional derivative together as a single `Dual`.

Hand-written primitives (see `src/rules_math.jl` for the shape to follow) are methods with a
specific `fdual`/`argduals` shape; a composite function is handled by an `@generated` fallback
that derives the rule from `f`'s IR, so `frule!!` works on anything.
"""
frule!!
