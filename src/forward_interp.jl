# Forward-mode AD: the `frule!!` entry point, the mode-specific `build_contextual_ir` override that
# `ADInterpreter` calls into (`contextual.jl`), and the split-shadow dualization engine itself.
#
# `frule!!(dualargs::Dual...)` is a `@generated` fallback: for a composite primal `g` with no
# hand-written rule, it compiles a *dualized* version of `g`'s body under
# `ADInterpreter{Forward}` and invokes the resulting `CodeInstance`. The dualization is a
# split-shadow transform on `g`'s post-optimization `IRCode`, spliced into the typeinf pipeline via
# the `build_contextual_ir` hook below, which `finishinfer!` calls for the return type and
# `optimize` calls to install the result.

# Carrier stub: gives a MethodInstance whose specTypes is the *dual* signature.
# `ADInterpreter{Forward}` replaces its source with the dualized primal body, so this
# body must never actually run.
dualized_impl(dualargs::Dual...) =
    error("Differ.dualized_impl ran directly: ADInterpreter could not dualize the primal ",
          "(likely unsupported IR — e.g. a growable-array mutation like `push!`/`resize!`, a ",
          "builtin with no rule, or a splat of something whose length isn't statically known).")


# Runtime dispatcher for a *dynamic* (`apply_generic`) call that survived into the primal IR (its
# callee or an argument was too abstractly typed to wrap statically). Rebuild concrete `Dual`s from
# the runtime primal/tangent values — the `map` is the whole point: `Dual(1.0, 1.0)` infers the
# concrete `Dual{Float64,Float64}`, so `frule!!`'s `@generated` dispatch (keyed on the `Dual` type
# parameters) can name a concrete primal method. A statically-built `Dual{Any,Any}` could not. The
# callee carries its *real* tangent `tf` (not a forced zero), so a dynamically-dispatched closure's
# capture derivatives still propagate.
function dynamic_frule(f, tf, primals::Tuple, tangents::Tuple)
    duals = map(Dual, primals, tangents)
    return frule!!(Dual(f, tf), duals...)
end


# ---------------------------------------------------------------------------
# Forward-mode transform hook.
#
# The generated `frule!!` fallback asks the interpreter to compile a `dualized_impl` MethodInstance
# whose `specTypes` is the *dual* signature. We compile it by transforming the corresponding primal
# method's post-optimization `IRCode` into a dualized `IRCode` (`build_dual_ir`). Non-`dualized_impl`
# MethodInstances return `nothing` here and flow through the ordinary pipeline (see
# `contextual.jl`).
# ---------------------------------------------------------------------------

function build_contextual_ir(interp::ADInterpreter{Forward}, mi::MethodInstance)
    is_dualized_impl(mi) || return nothing
    reason = Ref("Differ could not dualize the primal (no specific reason recorded).")
    edges = Any[]
    ir = build_dual_ir(interp, mi, reason, edges)
    # Stash whatever backedges were discovered even on a bail: if the missing piece (a primal
    # method, a not-yet-written `frule!!`) later appears, the mt-backedges recorded below (see
    # `primal_of_impl`/`frule_codeinstance`) invalidate this carrier so it gets a real chance to
    # dualize instead of staying pinned to the error stub forever. `frule_body` reads this back to
    # set the generated wrapper's `ci.edges`.
    interp.transformed_edges[mi] = edges
    ir === nothing && return error_ircode(mi, reason[])
    return ir
end

# `isa(specTypes, DataType)`, not just `isa(mi.def, Method)`: a MethodInstance's `specTypes` can be a
# `UnionAll` (a signature with free typevars, e.g. `Tuple{typeof(convert),Type{T},Type} where T`,
# which ordinary inference produces), and `.parameters` on a `UnionAll` throws a `FieldError` —
# uncaught, from inside `finishinfer!`, killing the whole compile rather than bailing. A carrier
# signature is always a concrete `DataType`, so a `UnionAll` is simply "not a carrier". Same guard on
# every other predicate that reads an arbitrary callee's `specTypes` (see `implicit_frule_tt` below,
# and the reverse-mode set in `reverse_interp.jl`).
is_dualized_impl(mi) = isa(mi.def, Method) && isa(mi.specTypes, DataType) &&
                       !isempty(mi.specTypes.parameters) &&
                       mi.specTypes.parameters[1] === typeof(dualized_impl)

# Build a minimal IRCode whose only effect is to `error(msg)` when invoked, installed via the same
# `finishinfer!`/`optimize` path as a real dualized body (see the hard project constraint: transform
# IRCode only, never patch a rettype after the fact). Used when `build_dual_ir` bails, so calling the
# carrier reports *why* (the specific unsupported construct) instead of `dualized_impl`'s generic
# stub message.
function error_ircode(impl_mi::MethodInstance, msg::String)
    stream = CC.InstructionStream(2)
    stream.stmt[1] = Expr(:call, error, msg); stream.type[1] = Union{}; stream.flag[1] = CC.IR_FLAG_NULL
    stream.stmt[2] = Core.ReturnNode();       stream.type[2] = Union{}; stream.flag[2] = CC.IR_FLAG_NULL
    cfg = CC.CFG(CC.BasicBlock[CC.BasicBlock(CC.StmtRange(1,2), Int[], Int[])], Int[3])
    di = CC.DebugInfoStream(stream.line)
    di.def = impl_mi
    dualparams = impl_mi.specTypes.parameters[2:end]
    argtypes = Any[impl_mi.specTypes.parameters[1], Tuple{dualparams...}]
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[])
    CC.verify_ir(ir)
    return ir
end

# The `frule_tt` a *hypothetical* differentiation of `callee_mi` would resolve against — the same
# shape `frule_codeinstance`/`primal_of_impl` build from a genuine surviving call, but derived
# instead from `callee_mi.specTypes` (the exact concrete call signature Julia's own call-graph
# discovery already found — see `register_implicit_frule_backedge!`/`src_inlining_policy` below).
# Returns `nothing` for anything the shape doesn't apply to (`Type`-valued parameters, a parameter
# with no `tangent_type`, …) rather than throwing: `callee_mi` may be an arbitrary callee Julia's
# compiler discovered, not something Differ validated.
function implicit_frule_tt(callee_mi::MethodInstance)
    isa(callee_mi.def, Method) || return nothing
    isa(callee_mi.specTypes, DataType) || return nothing   # `UnionAll` sig — see `is_dualized_impl`
    params = callee_mi.specTypes.parameters
    isempty(params) && return nothing
    ftype = params[1]
    (ftype isa Type) || return nothing
    try
        # Julia's compilation-signature heuristic collapses a vararg callee's trailing arguments into
        # a single `Vararg{T}` (`invoke vg(_2::Float64, _2::Float64, %1::Vararg{Float64})`), so
        # `specTypes` doesn't record the call's arity and `Dual{Vararg{Float64},…}` would throw.
        # Mirror the collapse into the `frule!!` signature as an open-ended `Vararg` tail instead of
        # giving up: `findsup` resolves such a query fine (returning a hand rule when one exists and
        # the generated fallback otherwise), and a hand-written rule for a vararg function is exactly
        # the case that needs it. An imprecise match only ever *restricts*: `has_hand_frule` suppresses
        # an inline, `register_implicit_frule_backedge!` over-invalidates. Both are sound.
        rest = Any[params[2:end]...]     # `params[2:end]` is a SimpleVector; need a Vector to `pop!`
        va = !isempty(rest) && isa(last(rest), Core.TypeofVararg) ? pop!(rest) : nothing
        dualargs = Any[Dual{P,tangent_type(P)} for P in rest]
        if va !== nothing
            D = Dual{va.T,tangent_type(va.T)}
            push!(dualargs, isdefined(va, :N) ? Vararg{D,va.N} : Vararg{D})
        end
        return Tuple{typeof(frule!!), Dual{ftype,NoTangent}, dualargs...}
    catch
        return nothing
    end
end

# An mt-backedge on the `frule!!` resolution a *hypothetical* differentiation of `callee_mi` would
# use, registered even though `callee_mi`'s call was (or may have been) inlined away and never
# actually went through `frule_split!`/`frule_codeinstance` — see the call site in `build_dual_ir`'s
# base case. So a user later hand-writing `frule!!(::Dual{typeof(callee_mi's function)}, ...)`
# invalidates a derivative built before that rule existed, exactly like a surviving high-level call
# would get via `frule_codeinstance`. Best-effort, same caveats as `implicit_frule_tt`: this is a
# nice-to-have extra invalidation trigger, not core, so a `frule_tt` that can't be built is silently
# skipped rather than aborting the whole dualization.
function register_implicit_frule_backedge!(edges::Vector{Any}, callee_mi::MethodInstance)
    frule_tt = implicit_frule_tt(callee_mi)
    frule_tt === nothing || push!(edges, frule_tt, Core.methodtable)
    return nothing
end

# The generated composite fallback (`frule_body`, installed by `refresh_frule`) is the *only* method
# of `frule!!` with this exact vararg signature — every hand-written rule (`frule!!(::Dual{typeof(f)},
# dualargs::Dual...)` for a concrete `f`) has a strictly narrower signature. So a `Method` matches the
# fallback, rather than some hand rule, iff its signature is exactly this one.
is_generated_frule_fallback(m::Method) = m.sig === Tuple{typeof(frule!!), Vararg{Dual}}

# Does a *hand-written* `frule!!` (as opposed to the generated composite fallback, which always
# matches) apply to a hypothetical differentiation of `callee_mi`? Used by `src_inlining_policy`
# below to keep such a call from being inlined away before `dualize_to_ircode` ever gets a chance to
# route it through that rule (see the "inlined `foo`, hand rule added, still gives the wrong
# derivative" class of bug this defends against).
function has_hand_frule(interp::ADInterpreter, callee_mi::MethodInstance)
    frule_tt = implicit_frule_tt(callee_mi)
    frule_tt === nothing && return false
    m, _ = CC.findsup(frule_tt, CC.method_table(interp))
    m === nothing && return false
    return !is_generated_frule_fallback(m.method)
end

# Inlining policy: never inline a call whose callee has a hand-written `frule!!` — otherwise the call
# vanishes into the caller's optimized IR (as plain arithmetic/whatever primitives it lowers to)
# before `dualize_to_ircode` ever sees it, so the hand rule can never be consulted, regardless of how
# small/inlinable the callee looks to Julia's ordinary cost-based heuristic (which has no concept of
# `frule!!` at all). Falls back to the ordinary policy otherwise, so this only ever *restricts*
# inlining relative to normal Julia behavior, never expands it.
function CC.src_inlining_policy(interp::ADInterpreter{Forward}, mi::MethodInstance,
                                @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::UInt32)
    has_hand_frule(interp, mi) && return false
    return @invoke CC.src_inlining_policy(interp::CC.AbstractInterpreter, mi::MethodInstance,
                                          src::Any, info::CC.CallInfo, stmt_flag::UInt32)
end

# Resolve the primal MethodInstance and dual arity for a `dualized_impl` specialization. `reason`
# records why (a human-readable message) when this bails, for `build_contextual_ir`/`code_dual_ircode`
# to surface to the user instead of a generic message. `edges` collects the backedges this
# resolution depends on (see `build_contextual_ir`/`frule_body`): an mt-backedge on `primal_tt` —
# unconditional, since *any* new method that could apply to these argument types (more specific,
# or previously nonexistent) must invalidate a derivative built against the old resolution — plus,
# when a match is found, a direct edge to `primal_mi` itself (this dual IR is built directly from
# its optimized IR, i.e. effectively inlines it, so redefining/invalidating it must invalidate us).
function primal_of_impl(interp::ADInterpreter, impl_mi::MethodInstance, reason::Ref{String}=Ref(""),
                        edges::Vector{Any}=Any[])
    dualparams = impl_mi.specTypes.parameters[2:end]
    # Defensive: `frule!!`/`dualized_impl` are `@generated`, and a generator is never handed a
    # `Vararg`-collapsed signature (verified for 1..10 arguments), so `dualparams` is always the
    # flat, exact list of `Dual`s the call supplied — no arity ceiling. Kept as a guard rather than
    # an assert in case a carrier `MethodInstance` is ever reached by some other route.
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

# Recursion cycle guard for `build_dual_ir`, analogous to reverse mode's `interp.in_progress` field
# (`contextual.jl`/`reverse_interp.jl`) but **task-local** rather than per-instance: a self- or
# mutually-recursive primal's nested resolution crosses the `frule!!` `@generated`-function boundary
# (`frule_codeinstance` below compiles the generic `frule!!` fallback under a *fresh*
# `CC.NativeInterpreter`, whose generator body `frule_body` — `refresh_frule`, bottom of this file —
# then spins up a *brand-new* `ADInterpreter{Forward}` instance to resolve the inner `dualized_impl`),
# so a guard scoped to one interpreter instance can never observe the cycle: each recursive level
# uses a different `interp` object even though it targets the exact same `impl_mi`. Confirmed
# empirically: a `@noinline` self-recursive function run through `frule!!` stack-overflows without
# this (a hard crash before this fix, not a catchable error) — the identical failure mode reverse
# mode's `in_progress` field exists to prevent, just reached by a different code path.
#
# That whole recursion is synchronous and never leaves the compiling task, so task-local storage is
# exactly the right scope: it is shared across those nested interpreter instances (so the cycle stays
# observable) yet isolated between concurrently-compiling tasks/threads. A plain `const` global
# `IdDict` was both — a shared bag that concurrent compilation on another thread could corrupt; keying
# it to the task fixes that without narrowing the scope the guard actually needs.
function dualized_impl_in_progress()
    return get!(() -> IdDict{MethodInstance,Nothing}(), task_local_storage(),
                :differ_dualized_impl_in_progress)::IdDict{MethodInstance,Nothing}
end

# Build the dualized `IRCode` for a `dualized_impl` specialization from the primal's optimized
# `IRCode`. Returns the dual `IRCode` or `nothing` (unsupported IR → bail).
#
# Two cases, distinguished by whether any *value* argument's primal is itself a `Dual`:
#
#  * First order (base case): the primal is an ordinary user method (or a hand-written `frule!!`),
#    found via `primal_of_impl`, and dualized directly. `pir` has positional arguments — one slot per
#    declared parameter, which for a *vararg* primal (`f(a, xs...)`) means its varargs arrive as a
#    single already-packed tuple slot; `primal_nfixed` below tells the prologue to re-pack the
#    trailing (flat) dual args into it.
#  * Higher order (Option A — compose the transform): a request whose value args are nested Duals
#    (e.g. `Dual{Dual{F,F},Dual{F,F}}`) is differentiating the *order-(k-1) dualized function*. That
#    function's optimized dual IR is obtained by peeling one `Dual` level off each value arg,
#    resolving the inner `dualized_impl` carrier, and recursing through `optimized_dual_ir`; the
#    result is then re-dualized. That inner dual IR is vararg-shaped (`dualized_impl(dualargs...)`),
#    so `pir_is_vararg=true` selects the tuple-reconstruction prologue in `dualize_to_ircode`.
function build_dual_ir(interp::ADInterpreter, impl_mi::MethodInstance, reason::Ref{String}=Ref(""),
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

function _build_dual_ir(interp::ADInterpreter, impl_mi::MethodInstance, reason::Ref{String}=Ref(""),
                        edges::Vector{Any}=Any[])
    dualparams = impl_mi.specTypes.parameters[2:end]
    if !all(P -> P isa Type && P <: Dual, dualparams)
        reason[] = "not every dual argument type is a `Dual` (a vararg call?)"
        return nothing
    end
    n = length(dualparams)

    # Compose the transform (Option A): peel one `Dual` level off `dualparams[1+offset:end]` to form
    # the order-(k-1) carrier signature, obtain that carrier's optimized dual IR, and re-dualize it.
    #  * `offset=0` — uniform seeds where the function nests like every value arg, so the *whole* list
    #    peels: `Dual{Dual{f,NoTangent},…}` → `Dual{f,NoTangent}`, `Dual{Dual{F,F},…}` → `Dual{F,F}`.
    #  * `offset=1` — a re-dualized carrier invoke whose `dualparams[1]` is a non-nested function slot
    #    (`dualized_impl`/`frule!!`) naming the function being re-differentiated: drop it, peel only the
    #    trailing (nested) value args.
    function compose(offset::Int)
        inner_dualparams = Any[_dual_primal_type(P) for P in dualparams[1+offset:end]]
        inner_tt = Tuple{typeof(dualized_impl), inner_dualparams...}
        m, _ = CC.findsup(inner_tt, CC.method_table(interp))
        if m === nothing
            reason[] = "could not find an inner carrier method to compose (higher-order re-dualization)"
            return nothing
        end
        inner_mi = specialize_method(m.method, m.spec_types, m.sparams)::MethodInstance
        # This dual IR is built directly on top of the inner carrier's own optimized dual IR (its
        # instructions are copied/re-dualized wholesale below), so redefining/invalidating
        # `inner_mi` must invalidate this one too. `inner_mi`'s *own* dependencies (its primal
        # method, any `frule!!`s it resolved) are tracked on its own edges list whenever it is
        # itself compiled via `build_contextual_ir` — not duplicated into `edges` here.
        CC.add_inlining_edge!(edges, inner_mi)
        pir = optimized_dual_ir(interp, inner_mi, reason)
        pir === nothing && return nothing
        return dualize_to_ircode(interp, impl_mi, pir, n; pir_is_vararg=true, pir_arg_offset=offset, reason, edges)
    end

    f1 = n >= 1 ? _dual_primal_type(dualparams[1]) : Union{}

    # Higher-order requests re-dualize an order-(k-1) carrier. Two shapes reach here:
    #
    #  * Uniform seeds (`code_dual_ircode` order≥2 / a `frule!!(fseed_k, seed_k)` call): every dual arg
    #    — the function included — is nested one level, so an order-≥2 request is one where *any*
    #    argument's primal is itself a `Dual`. The function nests as arg 1, so the whole list peels
    #    (`offset=0`). A `frule!!` slot is excluded: `frule!!` is the only Dual-consuming function, so a
    #    Dual-valued arg *under `frule!!`* means "differentiate a hand rule once" (the base case), not
    #    "differentiate the derivative" — that's handled below.
    #
    #  * A re-dualized *surviving carrier invoke* (nested `frule!!`/`D` — "D-of-D"): when the outer pass
    #    dualizes a function whose body called `frule!!`, that inner call survives in the primal IR as a
    #    `dualized_impl` `:invoke` (inlined generated `frule!!`) or a `frule!!` `:invoke`. `frule_split!`
    #    re-wrapped its callee as the non-nested function slot `Dual{typeof(dualized_impl),NoTangent}`
    #    / `Dual{typeof(frule!!),NoTangent}` at position 1, naming the function being re-differentiated
    #    rather than a value arg — so it is dropped (`offset=1`) and the trailing nested args peel.
    if f1 === typeof(dualized_impl)
        return compose(1)
    elseif f1 !== typeof(frule!!) && any(i -> _dual_primal_type(dualparams[i]) <: Dual, 1:n)
        return compose(0)
    end

    # Base case: an ordinary user method or a hand-written `frule!!`, dualized directly. The `frule!!`
    # slot lands here — `primal_of_impl` peels its args to the concrete rule signature. If that
    # resolves to the *generated* (vararg) `frule!!` rather than a hand rule, this is not a base case
    # but a composed derivative (`D` applied to a surviving `frule!!` invoke, e.g. a 3rd+-order nested
    # `D`); fall back to `compose(1)`, which drops the `frule!!` slot and re-dualizes the inner carrier.
    info = primal_of_impl(interp, impl_mi, reason, edges)
    if info === nothing
        f1 === typeof(frule!!) && return compose(1)
        return nothing
    end
    primal_mi, _ = info
    # A vararg primal (`f(a, xs...)`) declares one extra argument slot holding its varargs already
    # packed into a tuple, so its optimized IR has `nargs` slots while the dual call always supplies
    # `n` *flat* `Dual`s. `nargs` counts `#self#` and the vararg slot, so `nargs - 1` is the number of
    # slots before it, and `dualize_to_ircode`'s prologue re-packs dual args `nargs..n` into that
    # slot. Read off `primal_mi.def` rather than threading it out of `primal_of_impl`, whose `(mi, n)`
    # return shape is referenced elsewhere.
    pmethod = primal_mi.def::Method
    primal_nfixed = pmethod.isva ? Int(pmethod.nargs) - 1 : nothing
    # Optimized primal IR, computed by hand (mirroring `Core.Compiler.typeinf_ircode`'s own body)
    # rather than calling that function directly, so we can also read off `frame.edges` — see below.
    # Compiled with `interp` itself (not a bare `NativeInterpreter`): this is what makes our
    # `src_inlining_policy` override actually apply, so a callee with a hand-written `frule!!` isn't
    # inlined away before `dualize_to_ircode` gets a chance to route it through that rule — the same
    # reason `sin`/`cos` and other hand-ruled functions already survived as `:invoke`s even before
    # that override existed (their bodies simply weren't cheap enough to inline by ordinary cost
    # heuristics; the override now makes that hold *regardless* of cost for anything hand-ruled).
    frame = CC.typeinf_frame(interp, primal_mi, false)
    if frame === nothing
        reason[] = "inference failed to produce optimized IR for the primal method $(primal_mi)"
        return nothing
    end
    opt = CC.OptimizationState(frame, interp)
    pir = CC.run_passes_ipo_safe(opt.src, opt, nothing)
    # Pre-inlining call-graph edges: a plain `add_inlining_edge!(edges, primal_mi)` (added by
    # `primal_of_impl` above) is *not* enough on its own — if the primal inlines a callee (the
    # common case, e.g. `bar(x) = foo(x)` inlining `foo`), that callee's identity is gone from `pir`
    # by the time we ever see it, so we could never discover it by walking `pir` ourselves. But
    # `frame.edges` was already populated by ordinary inference (`compute_edges!`, called from
    # `finishinfer!` as the frame completes) *before* the optimizer ran — from each call site's
    # `CallInfo`, independent of what the later inlining pass does to the IR — and the inlining pass
    # (`InliningState(frame, interp)` aliases `frame.edges` as `opt.inlining.edges`) appends to that
    # very same array as it inlines. So `frame.edges` after `run_passes_ipo_safe` is exactly the
    # primal's real dependency set, including everything it inlined away.
    append!(edges, frame.edges)
    # For every concrete callee discovered above (regardless of whether its call survived or was
    # inlined away), also register the mt-backedge a hand-written `frule!!` for it would need — see
    # `register_implicit_frule_backedge!`. `ForwardToBackedgeIterator` decodes the same variable-width
    # edge encoding `store_backedges` itself understands, so this walks `frame.edges` correctly
    # regardless of which entry shape (plain MI/CI, invoke pair, mt-backedge pair, multi-match
    # record) each dependency happened to take.
    for (_, item) in CC.ForwardToBackedgeIterator(Core.svec(frame.edges...))
        isa(item, MethodInstance) && register_implicit_frule_backedge!(edges, item)
    end
    return dualize_to_ircode(interp, impl_mi, pir, n; pir_is_vararg=false, primal_nfixed, reason, edges)
end

# The optimized dual `IRCode` for a `dualized_impl` carrier: exactly what `CC.optimize`
# installs (and what `code_dual_ircode` returns) — `build_dual_ir` followed by the IPO-safe passes.
# Used both by the higher-order recursion above (to obtain the order-(k-1) dual IR as a primal) and
# by the reflection entry point.
function optimized_dual_ir(interp::ADInterpreter, impl_mi::MethodInstance, reason::Ref{String}=Ref(""),
                           edges::Vector{Any}=Any[])
    ir = build_dual_ir(interp, impl_mi, reason, edges)
    ir === nothing && return nothing
    world = CC.get_inference_world(interp)
    opt = CC.OptimizationState(impl_mi, CC.retrieve_code_info(impl_mi, world), interp)
    return begin
        run_ipo_passes!(ir, opt)
    end
end

# Resolve and compile the `frule!!(Dual{typeof(f),NoTangent}, dualargs...)` rule for a surviving
# high-level call to an *invoke-able `CodeInstance`*, so the dualized IR can emit a static
# `:invoke` (mirroring how the primal IR keeps `sin`/`cos` as `:invoke`s to a `CodeInstance`).
# `:invoke` targets *must* be `CodeInstance`s: `collectinvokes!` only JITs those, so a bare
# `MethodInstance` would fall back to a boxed dynamic call. Returns `nothing` if unresolved.
#
# `edges` collects the backedges this resolution depends on: an mt-backedge on `frule_tt` —
# unconditional, so a *new* user `frule!!` method (one that didn't exist, or wasn't as specific, when
# this was built — e.g. someone hand-writing a rule for a function that previously fell through to
# the generated composite fallback) invalidates this dual IR — plus, when a rule is found, a direct
# invoke edge to the resolved `CodeInstance` (this call is emitted as a static `:invoke` to it).
function frule_codeinstance(interp::ADInterpreter, @nospecialize(ftype), dual_argtypes,
                            edges::Vector{Any}=Any[])
    frule_tt = Tuple{typeof(frule!!), Dual{ftype,NoTangent}, dual_argtypes...}
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

# Resolve an arbitrary call whose dispatch tuple is `tt` to a `CodeInstance` for a static `:invoke`,
# or `nothing` if unresolved. The generic sibling of `frule_codeinstance`: identical pipeline
# (`findsup` against the interpreter's method table, `specialize_method`,
# `typeinf_ext_toplevel(NativeInterpreter, …, SOURCE_MODE_ABI)`) and identical invalidation edges —
# an mt-backedge so a new/more-specific method invalidates this carrier, plus a direct invoke edge to
# the resolved CI. Used for the tangent-helper calls (`get_tangent_field`/`set_tangent_field!`
# fallbacks, `zero_tangent`) that the direct-emission path can't take: a synthesized bare
# `Expr(:call, helper, …)` carries no `CallInfo`, so `ssa_inlining_pass!` can't inline it and it runs
# as a dynamic dispatch — an `:invoke` to a CI runs the compiled method directly instead.
function static_codeinstance(interp::ADInterpreter, @nospecialize(tt), edges::Vector{Any}=Any[])
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
# `dualize_to_ircode` (called by `build_dual_ir` above) transforms a primal method's fully
# optimized `IRCode` into a *dualized* `IRCode` using the "split-shadow" scheme: the primal
# computation is reconstructed and a parallel tangent computation is emitted beside it, then packed
# into a `Dual`. Low-level rules describe how each intrinsic (`add_float`, `mul_float`, …), builtin
# (`getfield`), and `%new` propagates tangents; surviving `:invoke`/`:call`s (e.g. `sin`/`cos`) go
# through `frule!!` dispatch, which picks up hand-written rules.
#
# Types are derived directly and exactly from the primal IR, not guessed: every shadow (tangent)
# statement shares its primal statement's type `Ti`; a `Dual{R,R}` wrapper uses `R = Ti`; a
# surviving `frule!!` result is `Dual{R,R}` with `R` the primal call's result type. The result is
# therefore a fully typed `IRCode` that installs as the optimization result and whose return type
# `finishinfer!` reads off via `compute_ir_rettype` — no re-inference. Branches and loops
# (`GotoNode`/`GotoIfNot`/`PhiNode`) are supported: block topology is preserved 1:1 from the primal,
# so only within-block instruction counts change. Throwing error paths are supported too: a block
# terminating in an unreachable `ReturnNode` (a domain/bounds check's throw target) is reconstructed
# primal-only (the derivative reproduces the same throw) — see `unreachable_block` below. Exception
# *handling* (`try`/`catch`) is supported as well: `UpsilonNode`/`PhiCNode` are duplicated into
# primal + shadow copies and `EnterNode`/`:leave`/`:pop_exception` carry over as control markers.
# Returns `nothing` to bail on still-unsupported constructs — e.g. a `Core.Builtin` with no rule
# (`Core.memoryrefoffset`, used by `push!`/`resize!`; a non-bits/undef-checked element access; or the
# `Core._apply_iterate` left behind by splatting something whose length isn't statically known) — and
# the caller then bails gracefully. Vararg primal *methods* are supported (see `primal_nfixed`
# below); only a splat *call site* over a non-tuple is not. Array element read/write, array
# *allocation* (`zeros`/`similar`/`Vector{T}(undef,n)`/comprehensions, via `memorynew`), and
# mutable-struct field mutation (`setfield!`) *are* supported — see the `memorynew`/`memoryrefnew`/
# `memoryrefget`/`memoryrefset!`/`setfield!` arms below.
# ===========================================================================

const _Intr = Core.Intrinsics

# Resolve a callee-position AST node to its actual value when it's *statically* known (a `GlobalRef`
# to a defined binding, or a `QuoteNode` literal). An `SSAValue`/`Argument` is a genuinely dynamic
# value, not a name to look up: returning it as-is would let the raw *old*-numbered node leak
# directly into the freshly built IR as an operand (referencing the wrong slot there). `nothing`
# signals "not statically resolvable — resolve via `presolve`/`_optype` instead" to callers.
#
# A `GlobalRef` is resolved *at `world`* — the interpreter's inference world — via `Base.getglobalref`
# (`jl_eval_globalref`), NOT at the ambient task world. This is load-bearing: `dualize_to_ircode` runs
# transitively inside the `@generated frule!!` body (`frule_body` -> `typeinf_ext_toplevel`), whose
# generation world can *predate* a user's (`Main`) function definition. A bare `getglobal`/`isdefined`
# (or `invoke_in_world`, which reparameterizes dispatch but *not* global-binding lookup) would there
# see a genuinely-defined user function as undefined, returning `nothing` — the callee then degrades to
# a raw `GlobalRef` and miscompiles at higher order (a `Dual{GlobalRef,…}` -> `%new` `TypeError`).
# `Base` functions escape this only because they are defined at an early world. The `world` argument is
# therefore *mandatory* (no default): every caller must thread the inference world in consciously — see
# the "world-age inside the generated `frule!!` body" note in the `adnext-ircode-dualization` skill.
# `try/catch -> nothing` keeps the "genuinely unresolvable ⇒ dynamic" contract for undefined bindings.
_calleeval(@nospecialize(x), world::UInt) =
    isa(x, GlobalRef) ? (try Base.getglobalref(x, world) catch; nothing end) :
    isa(x, QuoteNode) ? x.value :
    isa(x, Core.SSAValue) || isa(x, Core.Argument) ? nothing :
    x

# Resolve a `GlobalRef` to its bound value, distinguishing "resolved" from "undefined/unresolvable" —
# which `_calleeval` cannot, since its `nothing` sentinel collides with a binding whose value *is*
# `nothing` (the exact case of `return nothing`, which survives as `Main.nothing`). Returns
# `(true, val)` for a defined binding, `(false, nothing)` otherwise. Same world-parameterized
# `Base.getglobalref` lookup as `_calleeval` (see its note): the `world` argument is mandatory, never
# the ambient task world.
function _globalref_val(gr::GlobalRef, world::UInt)
    try
        return (true, Base.getglobalref(gr, world))
    catch
        return (false, nothing)
    end
end

# Whether `gr`'s binding is a defined *constant* — i.e. whether its value may be frozen into the IR —
# asked **at `world`**, the interpreter's inference world, never the ambient task world. The world
# argument is as load-bearing here as in `_calleeval`: `Base.isconst(mod, name)` answers for the
# *current* task world, and inside the generated `frule!!` body that world predates a user's `const`
# declaration, so it reports `false` for a genuinely constant binding (measured, not theoretical —
# a `const` global array operand degraded to `Any` and took the wrong branch in the `getfield` rule).
# This is the same check `verify_ir` performs on a value-position `GlobalRef`
# (`is_defined_const_binding(binding_kind(bpart))`, `Compiler/src/ssair/verify.jl`).
function _globalref_isconst(gr::GlobalRef, world::UInt)
    try
        bpart = Base.lookup_binding_partition(world, gr)
        return Base.is_defined_const_binding(Base.binding_kind(bpart))
    catch
        return false
    end
end

# Build the dualized IRCode for `impl_mi` (a `dualized_impl` specialization) from the primal's
# optimized IRCode `pir`. `n` = number of (flat) dual arguments.
# `pir_is_vararg` selects the argument-unpacking prologue: `false` for an ordinary primal (positional
# args, first-order/base case), `true` when `pir` is itself an order-(k-1) dual IR (a vararg
# `dualized_impl` taking one tuple of Duals — the higher-order case; see the prologue below).
# `primal_nfixed` is `nothing` for a non-vararg primal, or — when the primal *method* is itself
# vararg (`f(a, b, xs...)`) — the number of slots it declares before the vararg one (`#self#`
# included, i.e. `method.nargs - 1`). The prologue then re-packs the trailing flat dual args into the
# single tuple slot `pir` expects at `Argument(primal_nfixed+1)`, after which the rest of this
# transform sees exactly the argument shape `pir` was compiled with and needs no vararg awareness.
# Only meaningful with `pir_is_vararg=false`: an order-(k-1) dual carrier has already absorbed any
# vararg-ness of its own primal into its flat `argtypes`.
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
    # Defensive: a leading PhiNode in block 1 (predecessor-free) would collide with the
    # unconditional arg-extraction prologue below, which must land before any real phi. Not
    # observed in practice (slot2ssa! always keeps block 1 phi-free), but bail rather than emit
    # invalid IR if it ever occurs.
    if isa(pstmts[1][:stmt], Core.PhiNode)
        reason[] = "primal IR has a leading PhiNode in block 1 (unsupported shape)"
        return nothing
    end

    # Embed the actual (stable, singleton) function objects as literals rather than `GlobalRef`s
    # to a non-Core/Base module: `Core.Compiler.verify_ir` rejects a bare `GlobalRef` used directly
    # as a value unless its binding is proven constant across the IR's valid worlds, which these
    # Differ-module bindings aren't considered to be even though they never change identity.
    # Embed the actual functions as literals (see comment above re: value-position GlobalRefs).
    zerotang_g   = zero_tangent       # runtime zero-tangent fallback for non-diff results
    buildtang_g  = build_tangent      # construct a struct's `Tangent`/`MutableTangent` shadow
    fruleg = frule!!
    Dualg  = Dual                # the `Dual` constructor, for a runtime (dynamic) pack of a non-concrete result
    dynfrule_g = dynamic_frule   # runtime dispatcher for a surviving dynamic (`apply_generic`) call
    getf   = GlobalRef(Core, :getfield)
    ctuple = GlobalRef(Core, :tuple)
    intrg(name) = GlobalRef(Core.Intrinsics, name)

    # Tangent type of a primal type — drives every shadow SSA's declared type. For scalars
    # (`Float64`, `Float32`, `Complex`, …) `tt(T) == T`, so scalar shadows are unchanged from the
    # old same-typed scheme; for general structs it is a `Tangent`/`MutableTangent`, for tuples a
    # per-field tangent tuple, and for a `Dual` carrier the `Dual` itself (see `dual.jl`).
    # `T` is usually a plain `Type` (`pstmts[i][:type]`), but a statement whose result the primal's
    # own const-prop narrowed (e.g. `Core.memorynew` with a literal length) can carry a
    # `Core.PartialStruct`/`Const` lattice element instead — widen it first, since `tangent_type` is
    # only ever defined on `Type`s and only the backing type (not the narrowed const value) matters
    # for the tangent's own type.
    tt(@nospecialize T) = tangent_type(T isa Type ? T : CC.widenconst(T))

    code = Any[]; types = Any[]
    emit!(ex, @nospecialize(ty)) = (push!(code, ex); push!(types, ty); Core.SSAValue(length(code)))
    opf(name, ty, args...) = emit!(Expr(:call, intrg(name), args...), ty)
    # Emit a call `f(args...)` (declared result type `R`) as a static `:invoke` to a resolved
    # `CodeInstance` when its signature resolves, falling back to a bare `:call` otherwise. The
    # fallback keeps behavior identical to before; the win is that a resolved call runs the compiled
    # method directly instead of dynamic-dispatching a `CallInfo`-less synthesized `:call`. `argtypes`
    # are the argument types (excluding `f`), used only to build the dispatch tuple — pass the
    # operands' own declared types so the resolved `CodeInstance` matches what runs.
    #
    # Only invoke when every `argtype` is *concrete*: a non-concrete argtype (e.g. `Any`, the type of a
    # dynamic-dispatch result) means dispatch is genuinely runtime — `findsup` would resolve it to an
    # over-general method (`zero_tangent(::Any)`) and freezing an `:invoke` to that would both defeat
    # dynamic dispatch and hit that method's unbound-static-param path. A bare `:call` there still
    # dispatches correctly on the concrete runtime value, exactly as before Part 3.
    emit_invoke!(@nospecialize(f), @nospecialize(R), argtypes::Tuple, args...) = begin
        ci = all(_conc, argtypes) ?
                static_codeinstance(interp, Tuple{Core.Typeof(f), argtypes...}, edges) : nothing
        ci === nothing ? emit!(Expr(:call, f, args...), R) :
                         emit!(Expr(:invoke, ci, f, args...), R)
    end
    # Construct a `Dual{P,T}` directly with `%new` (an immutable-struct construction, no dispatch /
    # allocation) rather than a dynamic `Dual(...)` call the inliner couldn't reach in synthetic IR.
    dual!(@nospecialize(P), @nospecialize(T), @nospecialize(p), @nospecialize(t)) =
        emit!(Expr(:new, Dual{P,T}, p, t), Dual{P,T})
    # Whether a declared type is a specific leaf we can `%new` and freeze into an SSA's type.
    # Note: `isconcretetype(Type{P})` is always `false` (a documented Julia quirk), so a
    # `Type{P}`-typed callee/operand always takes the dynamic (`dynamic_frule`) path below rather
    # than a static `:invoke` — correct, just not the fast path.
    _conc(@nospecialize T) = T isa DataType && isconcretetype(T)
    # Pack a primal/tangent into a `Dual` for a *non-concrete* primal type `R` (e.g. `Any`, produced by
    # a dynamic dispatch). A `%new(Dual{R,tt(R)}, …)` would freeze the over-wide declared type into the
    # value — `dynbox(3.0)` would return a boxed `Dual{Any,Any}(9.0, 6.0)` instead of a genuine
    # `Dual{Float64,Float64}`, and that couldn't flow back into `frule!!`. Instead call the `Dual`
    # constructor *dynamically*: at run time it infers the concrete leaf type from the actual values
    # (exactly as `dynamic_frule`'s `map(Dual, …)` does). The declared type is the abstract
    # `dual_type(R)` — the UnionAll `Dual` for `R === Any`, or a `Union` of leaf `Dual`s for a `Union`
    # `R` — of which every possible runtime leaf is a subtype (whereas the `%new`-built
    # `Dual{Union{…},…}` is *not* a subtype of that union, so `%new` cannot be widened this way).
    # Allocation-freedom is moot: a call whose result is non-concrete already dynamic-dispatched, so
    # the `%new` fast path (kept for concrete `R`, asserted allocation-free by the test suite) is
    # unaffected.
    dyn_dual!(@nospecialize(R), @nospecialize(p), @nospecialize(t)) =
        emit!(Expr(:call, Dualg, p, t), dual_type(R))
    # `(true, value)` when `gr` names a defined *constant* binding at the inference world — the test
    # that licenses both embedding its value in the IR and using that value's exact type. Returns a
    # separate flag rather than a `nothing` sentinel (`_globalref_val`'s reason: a binding whose
    # value *is* `nothing` must still resolve), and asks constness at `iworld`, never the ambient
    # world (`_globalref_isconst`'s reason).
    function gref_constval(gr::GlobalRef)
        ok, gv = _globalref_val(gr, iworld)
        return (ok && _globalref_isconst(gr, iworld)) ? (true, gv) : (false, nothing)
    end
    # A `GlobalRef` in *value* position (a `%new` field, an `:invoke` operand) is rejected by
    # `verify_ir` unless its module is `Core`/`Base` or its binding is proven constant across the
    # IR's valid worlds (`check_op` in `Compiler/src/ssair/verify.jl`) — see gotcha #2 in the
    # `differ-ircode-dualization` skill. Turn one into something legal: a defined `const` binding is
    # embedded as its *value* (no instruction, and what keeps `return nothing`/`%new(…, nothing)`
    # allocation-free), anything else becomes a real global-load instruction whose `SSAValue` is used
    # instead. Never cached across uses — an `SSAValue` reused from a non-dominating block would fail
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
    # The tangent of a compile-time-constant primal is its zero tangent, computed now and embedded
    # as a literal so no call survives into the IR. `zero_tangent` handles singletons (-> the zero
    # tangent, e.g. `NoTangent()` for a function/`Int`), scalars (`0.0`), `Dual` carriers (a
    # same-typed zero), and composite constants (a `Tangent`/tuple of zeros).
    #
    # A bare `GlobalRef` operand is not itself a constant value — it *names* one. Splicing
    # `zero_tangent(globalref)` here would compute the tangent of the `GlobalRef` struct (a
    # `Module`+`Symbol` pair, always `NoTangent`), not of the value the binding holds. Resolve the
    # binding (same lookup `_calleeval` uses for callees) to learn the tangent type, then emit a
    # genuine runtime `zero_tangent` call on the (re-embedded, raw) `GlobalRef` operand — never
    # splice a *constructed* tangent object as a compile-time literal: a mutable tangent (e.g. a
    # `MutableTangent`) would then be one frozen object shared/aliased across every invocation of
    # this compiled carrier, corrupted by the first call that mutates it.
    function const_tangent(@nospecialize x)
        isa(x, QuoteNode) && return zero_tangent(x.value)
        # The null shadow-pointer sentinel (`src/intrinsics.jl`): a `Ptr` operand with no tangent
        # storage behind it. `zero_tangent(::Ptr)` throws by design, but this one *does* have a
        # well-defined tangent — a null shadow's own shadow is again null
        # (`tangent_type(Ptr{NoTangent}) === Ptr{NoTangent}`) — which is what keeps IR containing it
        # re-dualizable at order ≥ 2.
        x === NULL_SHADOW_PTR && return NULL_SHADOW_PTR
        if isa(x, GlobalRef)
            # `_globalref_val`, not `_calleeval`: a binding holding `nothing` (e.g. `return nothing`)
            # must still resolve here, not be mistaken for undefined and fall through to
            # `zero_tangent(gr)` (the tangent of the GlobalRef *struct*, not the bound value).
            ok, gv = _globalref_val(x, iworld)
            if ok
                T = tangent_type(Core.Typeof(gv))
                # The `zero_tangent` argument goes through `gref_operand!`, never the raw node: an
                # `:invoke` operand is a value position too.
                return T === NoTangent ? NoTangent() :
                       emit_invoke!(zerotang_g, T, (Core.Typeof(gv),), gref_operand!(x))
            end
        end
        return zero_tangent(x)
    end
    # Zero tangent for a *computed* primal value of type `Ti` (the tangent of a non-differentiable
    # operation's result). `NoTangent()` when the tangent type is trivial (`Int`, `Bool`, …), a
    # literal `zero(Ti)` for a concrete `Number`, otherwise a runtime `zero_tangent` on the primal.
    # `Ti` may be a `Core.PartialStruct`/`Core.Const` lattice element rather than a bare `Type` (see
    # `tt`'s own note above) — widen it first, same as `tt`, so the `Number` literal-zero fast path
    # and the `emit_invoke!` argtype (which needs a real `Type` for its dispatch tuple) both still
    # apply instead of silently falling back to a plain dynamic `:call`.
    function zero_shadow(@nospecialize(Ti), @nospecialize(primal_ssa))
        T = tt(Ti)
        T === NoTangent && return NoTangent()
        Tw = Ti isa Type ? Ti : CC.widenconst(Ti)
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
        # A *vararg* primal (`primal_nfixed !== nothing`) declares one extra slot holding its varargs
        # already packed into a tuple, read in its body as `getfield(_va, j)`. The dual call is always
        # flat (`frule!!(Dual(f), Dual(a), Dual(x1), Dual(x2))`), so the trailing dual args are
        # re-packed here into exactly that tuple — after which the rest of this transform sees the
        # argument shape `pir` expects and needs no vararg awareness at all.
        nfixed = primal_nfixed === nothing ? n : primal_nfixed
        if nfixed > n
            reason[] = "the primal vararg method declares $nfixed argument slots before its vararg " *
                       "slot, but only $n dual arguments were supplied"
            return nothing
        end
        # Settle the packed tuple's primal/tangent types *before* emitting anything, so a shape
        # disagreement bails cleanly instead of leaving half-emitted statements behind.
        vptys = Any[_dual_primal_type(dualparams[i]) for i in (nfixed + 1):n]
        vttys = Any[_dual_tangent_type(dualparams[i]) for i in (nfixed + 1):n]
        Pva = Tuple{vptys...}
        Tva = tangent_type(Pva)
        if primal_nfixed !== nothing
            # `pir`'s own vararg slot type is a lattice element, not necessarily a bare `Type`
            # (`Core.Const(())` when the vararg is empty, `Core.Const((Float64,))` for a `Type`-valued
            # element), so widen before comparing. `<:` rather than `===`: the reconstruction can be
            # legitimately *sharper* than inference's summary of the same slot (that `Core.Const((Float64,))`
            # widens to `Tuple{DataType}` where this rebuilds `Tuple{Type{Float64}}`), which is sound —
            # every use of the slot then carries a subtype of what the primal statement was typed
            # against. A genuine disagreement (a different arity) still fails. And this *bails* rather
            # than asserting, unlike the `pir_is_vararg` branch below where both sides are this pass's
            # own construction: here the shape is driven by user argument types, and an `AssertionError`
            # thrown out of the `@generated frule!!` body aborts compilation instead of producing an
            # `error_ircode` carrier that can report why.
            Pvaslot = CC.widenconst(pir.argtypes[nfixed + 1])
            if !(Pva <: Pvaslot)
                reason[] = "the reconstructed vararg tuple type $Pva does not match the primal " *
                           "method's own vararg slot type $Pvaslot"
                return nothing
            end
            # The tangent side is NOT simply the mirror of the primal tuple: `tangent_type` *collapses*
            # an all-`NoTangent` tuple to plain `NoTangent` (`tangent_type(Tuple{Int,Int}) === NoTangent`,
            # and likewise `Tuple{}` — the empty-vararg case). `Dual{P,T}` requires `T == tangent_type(P)`,
            # so a collapsed slot must hold the *literal* `NoTangent()`: an emitted
            # `Core.tuple(NoTangent(), NoTangent())` is a `Tuple{NoTangent,NoTangent}` and would
            # `TypeError` at the `%new(Dual{P,NoTangent}, …)` `frule_split!` builds when the whole tuple
            # is passed on to a surviving call. Same rule, same reason, as `Core.tuple`'s own shadow —
            # see `apply_builtin_frule!(::Val{Core.tuple}, …)` in `builtins.jl`. Nothing ever *reads* a
            # tangent out of a collapsed slot: collapse means every element's own tangent type is
            # `NoTangent`, so every `getfield` on it takes the `NoTangent()` branch too.
            if !(Tva === NoTangent || Tva === Tuple{vttys...})
                reason[] = "the vararg tuple's tangent type $Tva is neither `NoTangent` nor the " *
                           "tuple $(Tuple{vttys...}) of its elements' tangent types"
                return nothing
            end
        end

        nslots = primal_nfixed === nothing ? n : nfixed + 1
        parg = Vector{Any}(undef, nslots); targ = Vector{Any}(undef, nslots)
        argty = Vector{Any}(undef, nslots)
        pelts = Any[]; telts = Any[]
        for i in 1:n
            Di = dualparams[i]
            di = emit!(Expr(:call, getf, Core.Argument(2), i), Di)
            p = emit!(Expr(:call, getf, di, 1), _dual_primal_type(Di))
            if i <= nfixed
                parg[i]  = p
                targ[i]  = emit!(Expr(:call, getf, di, 2), _dual_tangent_type(Di))
                argty[i] = _dual_primal_type(Di)
            else
                push!(pelts, p)
                # No tangent read at all for a collapsed vararg tangent — it would be dead code.
                Tva === NoTangent ||
                    push!(telts, emit!(Expr(:call, getf, di, 2), _dual_tangent_type(Di)))
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
        # `Argument(2)`, the tuple of order-(k-1) dual args (accessed as `getfield(Argument(2), i)`).
        # The `pir_arg_offset` leading dual args are non-value carrier function slots to skip (0 for a
        # uniform seed where the function nests like every other value; 1 for a re-dualized
        # `dualized_impl` invoke, whose `dualparams[1]` is the `Dual{typeof(dualized_impl),NoTangent}`
        # function slot — see `build_dual_ir`). Each remaining `Di` is nested (`primal_type(Di) <:
        # Dual`); peel one level off each: reconstruct a primal tuple (each element the order-(k-1) dual
        # value) and a tangent tuple (its new, k-th tangent direction). The reconstructed tuple must
        # match `pir`'s own vararg-tuple arg type, so the tuple index `i` is the *original* dualparam
        # position (offset-shifted) while the tuple element position `j` is offset-free. `pir` has
        # exactly two arguments (#self# and the tuple), so `parg`/`targ` are length 2 regardless of `n`.
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
            pelts[j] = emit!(Expr(:call, getf, di, 1), _dual_primal_type(Di));  ptys[j] = _dual_primal_type(Di)
            telts[j] = emit!(Expr(:call, getf, di, 2), _dual_tangent_type(Di)); ttys[j] = _dual_tangent_type(Di)
        end
        ptuple = emit!(Expr(:call, ctuple, pelts...), Tuple{ptys...})
        ttuple = emit!(Expr(:call, ctuple, telts...), Tuple{ttys...})
        @assert Tuple{ptys...} === pir.argtypes[2]     # reconstructed tuple == pir's own arg type
        parg = Any[dualized_impl, ptuple]; targ = Any[NoTangent(), ttuple]
        # `pir` here is a dual carrier this pass built itself (see the `argtypes` at the bottom of
        # this function), so both slots are already plain `Type`s; spelled out for `optype` below.
        argty = Any[typeof(dualized_impl), Tuple{ptys...}]
    end

    presolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? primal[x.id] : isa(x, Core.Argument) ? parg[x.n] : x
    # `presolve` for an operand that lands in *value* position (a `%new` field). It passes a
    # `GlobalRef` through unchanged, which `verify_ir` rejects — resolve it (`gref_operand!`).
    # Call/invoke argument positions don't need this: `frule_split!` and the `Core.Const` arm already
    # embed a statically-known operand's resolved value via `_calleeval`.
    vpresolve(@nospecialize x) = isa(x, GlobalRef) ? gref_operand!(x) : presolve(x)
    tresolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? shadow[x.id] :
        isa(x, Core.Argument) ? targ[x.n] :
        const_tangent(x)                                 # constant tangent (literal)
    # Declared primal type of an operand. `Argument`s go through `argty` rather than `pir.argtypes`
    # directly, for two reasons. (1) `pir.argtypes` holds *lattice elements*, not necessarily bare
    # `Type`s — `Core.Const(f)` for a singleton function slot, `Core.Const(())` for an empty vararg
    # slot — and the results below are consumed as genuine *type parameters* (`Dual{P,tt(P)}`,
    # `Tuple{ptys...}`, `dual_type(R)`, `Pobj <: Tuple` in the builtin rules), each of which throws on
    # a non-`Type`; `_conc` is the only tolerant consumer and it just routes into one of the throwing
    # ones. (2) For a vararg primal the vararg slot's type is the tuple this pass reconstructed above,
    # which is what actually flows there. (3) A `GlobalRef` operand — `_optype`'s literal fallback
    # would report `GlobalRef`, the type of the *node*, not of the value the binding names. That is a
    # silent miscompile, not a bail: `getfield(Main.CONST_VEC, :ref)` would take the builtin rule's
    # general-struct branch (`GlobalRef` is not `<: Array`) and emit `get_tangent_field` against an
    # array shadow, which `MethodError`s at run time. Take the type from the binding instead, exactly
    # as the `ReturnNode` arm does. `_optype` still handles `SSAValue`s and literals, and stays as-is
    # for reverse mode's own call sites.
    optype(@nospecialize x) = isa(x, Core.Argument) ? argty[x.n] :
                              isa(x, GlobalRef) ? gref_optype(x) : _optype(pir, x)

    function frule_split!(fpos, actual, R)
        # Dual(callee, NoTangent()) and each Dual(arg_primal, arg_tangent), constructed via %new.
        # Embed the *resolved* callee value (e.g. the `sin` function object) rather than the raw
        # callee-position AST node when it's statically known: a bare `GlobalRef` outside Core/Base
        # used directly as a value is rejected by `Core.Compiler.verify_ir` unless its binding is
        # proven constant. When the callee is a genuinely dynamic value (an `SSAValue`/`Argument` —
        # e.g. a `Function` read out of a container), resolve it like any other operand via
        # `presolve`/`_optype` rather than embedding the raw (old-numbered) AST node.
        fval = _calleeval(fpos, iworld)
        fcallee = fval === nothing ? presolve(fpos) : fval
        ftype   = fval === nothing ? optype(fpos) : _typeof(fval)
        # The callee's own tangent: a statically-known function is a code constant (zero tangent —
        # `NoTangent()` for a plain function); a genuinely dynamic callee (read out of a container)
        # carries whatever tangent the shadow pass computed for it.
        ftang   = fval === nothing ? tresolve(fpos) : zero_tangent(fval)
        # A call is a genuine dynamic dispatch (`apply_generic`) when its callee or any argument has a
        # non-concrete declared type — the method that runs depends on runtime types unknowable here
        # (e.g. a value read out of an `Any`-typed global/field/container). We can't wrap such a call
        # statically: a `%new` of a `Dual` would freeze the abstract type into the object, and
        # `frule!!`'s `@generated` dispatch — keyed on exactly those `Dual` type parameters — could not
        # resolve a concrete primal from `Dual{Any,…}`. Defer to the runtime `dynamic_frule` dispatcher
        # (see its definition above): pass the callee's primal + tangent and a tuple of the argument
        # *primals* and a tuple of their *tangents*, and let it rebuild concrete `Dual`s and dispatch
        # `frule!!` at run time. Its result is typed `Any`; extract the primal (typed `R`) and tangent
        # (typed `tt(R)`, which widens to `Any` when `R` is abstract). Note we do *not* branch on `R`:
        # a call with concrete callee+args but an inference-widened abstract result (e.g. `_2 * _2 ::
        # Any`, left after a `Ref{Any}` was SROA'd away) takes the static path below, which annotates
        # its result with the *abstract* `dual_type(R)` — sound because `Dual` is invariant so the
        # concrete `Dual{Rc,Tc}` the rule returns is `<: Dual` but not `<: Dual{Any,Any}`.
        if !_conc(ftype) || !all(a -> _conc(optype(a)), actual)
            # A statically-known operand (a `GlobalRef` to a defined binding, or a `QuoteNode` — e.g.
            # the `^` passed as an argument to `Base.literal_pow`) must be embedded as its *resolved
            # value*, not the raw node: a bare non-Core/Base `GlobalRef` in value position is rejected
            # by `verify_ir` (and `tresolve` would take `zero_tangent` of the `GlobalRef` object
            # itself rather than the value it names). Resolve each arg's primal, its declared type,
            # and its (compile-time-zero) tangent together so the tuple's element types match.
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
                    push!(tvals, zero_tangent(v)); push!(ttys, tt(P))
                end
            end
            ptup = emit!(Expr(:call, ctuple, pvals...), Tuple{ptys...})
            ttup = emit!(Expr(:call, ctuple, tvals...), Tuple{ttys...})
            dd = emit!(Expr(:call, dynfrule_g, fcallee, ftang, ptup, ttup), Any)
            return emit!(Expr(:call, getf, dd, 1), R),
                   emit!(Expr(:call, getf, dd, 2), tt(R))
        end
        fd = dual!(ftype, NoTangent, fcallee, NoTangent())
        # Each argument dual is `Dual{P, tangent_type(P)}` (the Mooncake invariant). For scalar
        # primals `tangent_type(P) == P`, so this matches the old `Dual{P,P}`. A statically-known
        # operand (a `GlobalRef` to a defined binding, or a `QuoteNode`) must be embedded as its
        # *resolved value* with `P = _typeof(value)`, not as the raw node: e.g. an intrinsic's leading
        # *type* argument (`fpext(Base.Float64, …)`) is a `GlobalRef` whose value is a `DataType`
        # instance (`Float64`), and wrapping the raw node as `Dual{GlobalRef,…}` would both mis-declare
        # the field and TypeError at the `%new` when the ref loads as the type. `_typeof`, not plain
        # `typeof`, is required here: `typeof(Float64) === DataType` loses the value entirely, whereas
        # `_typeof(Float64) === Type{Float64}` sharpens it — needed for `Dual{Type{Float64},…}`-keyed
        # `frule!!` dispatch to resolve. Only genuinely dynamic operands (SSAValue/Argument) fall back to
        # `presolve`/`_optype`/`tresolve`.
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
                push!(duals, dual!(P, tt(P), v, zero_tangent(v)))
            end
        end
        # Emit the surviving high-level rule as a static `:invoke` to a compiled `CodeInstance` when we
        # can resolve one (direct, unboxed call); otherwise a plain non-inlined `:call`. `dual_type(R)`
        # is the declared result type: `Dual{R,tangent_type(R)}` for a concrete `R`, and the *abstract*
        # `Dual` when `R` is non-concrete — the concrete `Dual` the rule actually returns is a subtype
        # of the latter (whereas the invariant `Dual{Any,Any}` would be unsound and miscompile).
        ci = frule_codeinstance(interp, ftype, dualtys, edges)
        DR = dual_type(R)
        dd = ci === nothing ? emit!(Expr(:call, fruleg, fd, duals...), DR) :
                              emit!(Expr(:invoke, ci, fruleg, fd, duals...), DR)
        return emit!(Expr(:call, getf, dd, 1), R), emit!(Expr(:call, getf, dd, 2), tt(R))
    end

    # Bundle of closures `apply_intrinsic_frule!` (`src/intrinsics.jl`) needs to emit an intrinsic's
    # primal + shadow IR directly, without going through `frule_split!`'s `Dual`-boxing/`CodeInstance`
    # machinery — see that file for why intrinsics get this cheaper path.
    # `optype`/`tt` give the pointer rules (`bitcast`/`pointerref`/`pointerset`/`add_ptr`) the same
    # primal-type introspection the builtin rules get; `reason` lets a rule that *declines* say why
    # (see the dispatch sites below).
    intrinsic_ctx = (opf=opf, emit! =emit!, presolve=presolve, tresolve=tresolve, zero_shadow=zero_shadow,
                     optype=optype, tt=tt, reason=reason)
    # Same idea for `apply_builtin_frule!` (`src/builtins.jl`), which handles `Core.Builtin`s
    # (`getfield`, `setfield!`, `Core.tuple`, `Core.ifelse`, the array-allocation builtins, `===`).
    # `optype`/`tt` give those rules the primal-type introspection they need for same-shape-vs-general
    # struct branching that a plain intrinsic never has to do.
    builtin_ctx = (emit! =emit!, presolve=presolve, tresolve=tresolve, zero_shadow=zero_shadow,
                   optype=optype, tt=tt, emit_invoke! =emit_invoke!, opf=opf, reason=reason)
    # And for `apply_foreigncall_frule!` (`src/foreigncalls.jl`). Two extra members beyond the builtin
    # bundle, both for the pointer-provenance walk: `pstmt` reads a *primal* statement node back out of
    # `pir` (old numbering — unaffected by the shadow statements already interleaved into `code`), and
    # `calleeval` resolves a callee node so the walk can recognise `bitcast`/`getfield` in the chain.
    pstmt(@nospecialize x) = isa(x, Core.SSAValue) ? pstmts[x.id][:stmt] : nothing
    foreigncall_ctx = (emit! =emit!, presolve=presolve, tresolve=tresolve, zero_shadow=zero_shadow,
                       optype=optype, tt=tt, emit_invoke! =emit_invoke!, opf=opf, reason=reason,
                       pstmt=pstmt, calleeval=(@nospecialize(x) -> _calleeval(x, iworld)))

    # Block topology (block count, order, preds/succs) is preserved 1:1 from the primal: this
    # transform only expands each original statement into more instructions, never splits, merges,
    # or reorders blocks. So `GotoNode.label`/`GotoIfNot.dest`/`PhiNode.edges` (already basic-block
    # numbers in post-optimization IRCode) carry over unchanged; only each block's `StmtRange` needs
    # recomputing, tracked live below as blocks are crossed during the single linear pass.
    nblocks = length(pir.cfg.blocks)
    block_start_new = Vector{Int}(undef, nblocks)
    block_start_new[1] = 1                          # includes the arg-extraction prologue above
    bidx = 1

    # A block that terminates in an *unreachable* `ReturnNode` (one with no `val`) is an error/throw
    # block: `GotoIfNot -> error block -> unreachable`. Every statement in it only ever leads to a
    # `throw` and never contributes to a returned value (nor reaches a live `PhiNode` — dominance
    # guarantees this), so such statements are reconstructed *primal-only* below (see the main loop):
    # the primal computation is rebuilt faithfully so the derivative raises the same error on the
    # same inputs, but no shadow/tangent is computed for it.
    unreachable_block = falses(nblocks)
    for b in 1:nblocks
        term = pstmts[pir.cfg.blocks[b].stmts.stop][:stmt]
        unreachable_block[b] = isa(term, Core.ReturnNode) && !isdefined(term, :val)
    end

    # Forward-reference patches for `PhiNode` operands not yet resolved when the phi is processed
    # (loop back-edges: the operand is defined later in the linear statement order). Keyed by the
    # referenced *original* SSA index; each entry is (target values-vector, slot, want_primal).
    # `PhiNode.values` is a plain `Vector`, mutable in place even though `PhiNode` is immutable.
    pending = Dict{Int,Vector{Tuple{Vector{Any},Int,Bool}}}()

    for i in 1:N
        while bidx < nblocks && i > pir.cfg.blocks[bidx].stmts.stop
            # A block whose every original statement is a pure alias (e.g. a bare `nothing`
            # fallthrough placeholder block, common between adjacent loops) emits no instructions
            # at all, which would leave it an empty `StmtRange`. Backfill a placeholder so every
            # block keeps at least one statement, matching the primal's own convention.
            if length(code) < block_start_new[bidx]
                emit!(nothing, Nothing)
            end
            bidx += 1
            block_start_new[bidx] = length(code) + 1
        end
        s = pstmts[i][:stmt]; Ti = pstmts[i][:type]
        if unreachable_block[bidx]
            # Primal-only reconstruction (see `unreachable_block` above). `shadow[i] = primal[i]` is
            # a never-consumed placeholder so `presolve`/`tresolve` on later error-path operands
            # still resolve, without emitting a bogus tangent computation.
            if isa(s, Core.ReturnNode)
                emit!(Core.ReturnNode(), Union{})   # unreachable terminator
            elseif isa(s, Expr) && s.head === :invoke
                # Resolve the display-callee to its value: a bare non-Core/Base `GlobalRef` in the
                # invoke's callee position is a value position `verify_ir` rejects (see frule_split!).
                # A genuinely dynamic callee (SSAValue/Argument) isn't a name to look up — resolve it
                # like any other operand via `presolve`.
                fv = _calleeval(s.args[2], iworld)
                ex = Expr(:invoke, s.args[1], fv === nothing ? presolve(s.args[2]) : fv,
                          (presolve(a) for a in s.args[3:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :call
                # Resolve the callee to its value too: `userefs` checks the callee operand, and a
                # bare non-Core/Base `GlobalRef` (e.g. `throw`) fails the const-binding check when
                # re-embedded in this synthetic IR's world range.
                fv = _calleeval(s.args[1], iworld)
                ex = Expr(:call, fv === nothing ? presolve(s.args[1]) : fv,
                          (presolve(a) for a in s.args[2:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :new
                # Type argument and field operands both get the same `GlobalRef` resolution as the
                # live-path `:new` arm below (ISSUES #60) — a throw block builds exception objects,
                # and `%new(Main.MyError, …)` is exactly the shape that trips it.
                T = s.args[1]
                if isa(T, GlobalRef)
                    ok, gv = _globalref_val(T, iworld)
                    (ok && isa(gv, Type)) && (T = gv)
                end
                ex = Expr(:new, T, (vpresolve(a) for a in s.args[2:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :boundscheck
                primal[i] = emit!(Expr(:boundscheck, (presolve(a) for a in s.args)...), Ti)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :throw_undef_if_not
                # Pure control marker (undef-var/boxed-capture guard): `args[1]` is a bare
                # Symbol/GlobalRef name, copied through verbatim (never a value to dualize/resolve);
                # `args[2]` is the non-differentiable Bool condition, which does need this pass's own
                # resolution (it's a literal in a throw-only block, but resolve it uniformly with the
                # live-path arm below regardless). No shadow value: it never feeds a `PhiNode` or is
                # otherwise consumed.
                primal[i] = emit!(Expr(:throw_undef_if_not, s.args[1], presolve(s.args[2])), Ti)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :gc_preserve_begin
                # Primal-only reconstruction: nothing in a throw-only block carries a tangent, so
                # there is no shadow object to root (unlike the live-path arm below).
                primal[i] = emit!(Expr(:gc_preserve_begin, (presolve(a) for a in s.args)...), Any)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :gc_preserve_end
                primal[i] = emit!(Expr(:gc_preserve_end, presolve(s.args[1])), Ti)
                shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :foreigncall
                # Primal-only reconstruction. `args[2:5]` (return type, argument-type `svec`, `nreq`,
                # calling convention) are literals copied verbatim; `args[1]` usually is too, but the
                # runtime-function-pointer form puts an `SSAValue` there, which must be resolved into
                # this pass's numbering like any other operand — and `verify_ir`'s `check_op`
                # whitelists only an `Expr(:tuple)` at that position, so a bare non-Core/Base
                # `GlobalRef` needs resolving as well.
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
                # A bare GlobalRef statement is a global-variable load, not a pure alias — it must be
                # emitted as a real instruction (see the main-loop GlobalRef case below for why).
                primal[i] = emit!(s, Ti); shadow[i] = primal[i]
            elseif !isa(s, Expr)
                primal[i] = presolve(s); shadow[i] = primal[i]
            else
                reason[] = "unexpected statement kind $(typeof(s)) in an unreachable (throw-only) " *
                           "block at %$i: `$(_stmt_str(s))`"
                return nothing
            end
        elseif isa(Ti, Core.Const) && isa(s, Expr) && (s.head === :call || s.head === :invoke)
            # Julia's own constant-propagation proved this call's result is always exactly this
            # literal value, regardless of its arguments — the derivative of a provably-constant
            # quantity is definitionally zero, independent of whatever the callee actually computes.
            # (Only `Core.Const` licenses this — a `Core.PartialStruct` narrows *some* fields but
            # doesn't pin the whole value, e.g. `Core.memorynew`'s result, which does carry a real
            # shadow.) The call is kept in the primal IR rather than folded away only for its own
            # sake (effects, or simply not inlined), so still reconstruct it faithfully — but skip
            # the intrinsic/builtin/`frule!!` dispatch below entirely for the shadow, straight to
            # `zero_shadow`. Resolve the display-callee to its value exactly like the unreachable-block
            # arm above (a bare non-Core/Base `GlobalRef` in value position is rejected by `verify_ir`).
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
                    # bare GlobalRef `Main.nothing`, not a literal. Two things the generic path gets
                    # wrong here: (1) `presolve` would pass the raw node through and it would land in
                    # the returned `Dual`'s primal field, which `verify_ir` rejects (a non-Core/Base
                    # GlobalRef in value position — gotcha #2/#4); `gref_operand!` resolves it.
                    # (2) `_optype` would report `GlobalRef`, not the bound value's type; take the type
                    # from the binding. A defined `const` binding gives a concrete `Core.Typeof(gv)` —
                    # for `return nothing` that is `Nothing`, keeping the allocation-free `%new` path;
                    # a non-`const` (or unresolvable) binding falls to `Any` and the dynamic pack.
                    gr = s.val::GlobalRef
                    p = gref_operand!(gr)
                    R = gref_optype(gr)
                else
                    p = presolve(s.val)
                    R = optype(s.val)
                end
                t = tresolve(s.val)
                # Concrete `R`: pack with an allocation-free `%new` of the exact `Dual{R,tt(R)}`.
                # Non-concrete `R` (a dynamic-dispatch result, e.g. `Any`): build the `Dual`
                # dynamically so the runtime type is the concrete leaf, not a frozen `Dual{Any,Any}`.
                res = _conc(R) ? dual!(R, tt(R), p, t) : dyn_dual!(R, p, t)
                emit!(Core.ReturnNode(res), Any)
            end
        elseif isa(s, Core.PiNode)
            primal[i] = presolve(s.val); shadow[i] = tresolve(s.val)
        elseif isa(s, Expr) && s.head === :new
            # The constructed type is a `GlobalRef` whenever the struct is named by a module-level
            # binding — a top-level `struct S` lowers to `%new(Main.S, …)`. It has to be resolved
            # *before* anything else in this arm: every branch below tests it with `<:`/`fieldtype`,
            # which throw a raw `TypeError` on a `GlobalRef`, and it is an operand position
            # `verify_ir` checks too (ISSUES #60). Resolution alone here, without the `const` test
            # `gref_constval` applies to *field* operands: a struct name is always a constant
            # binding, and refusing one that a world-lookup happened to report otherwise would turn a
            # working case into a bail. `isa(gv, Type)` is the actual guard.
            T = s.args[1]
            if isa(T, GlobalRef)
                ok, gv = _globalref_val(T, iworld)
                (ok && isa(gv, Type)) && (T = gv)
            end
            if !isa(T, Type)
                reason[] = "`%new` whose type argument `$(s.args[1])` is not a statically-known " *
                           "type at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            args = @view s.args[2:end]
            # `vpresolve`, not `presolve`: a field operand is a value position, so a `GlobalRef` there
            # (`%new(Broadcasted{…}, …, Base.Broadcast.nothing)`) must be resolved — same issue.
            pf = Any[vpresolve(a) for a in args]
            primal[i] = emit!(Expr(:new, T, pf...), Ti)
            TT = tt(Ti)
            if T <: Dual
                # `Dual` is its own tangent type, so its shadow is a same-typed `Dual` built via
                # `%new`. Each field is its `tresolve`d tangent, except a *non-differentiable* field
                # (whose tangent is `NoTangent` and can't fill e.g. a `typeof(sin)` or `Int` slot)
                # which carries the primal value through unchanged. This is what lets a
                # `Dual{typeof(sin),NoTangent}` be re-dualized at higher order.
                tf = Any[_nondiff_field(fieldtype(T, j)) ? vpresolve(args[j]) : tresolve(args[j])
                         for j in eachindex(args)]
                shadow[i] = emit!(Expr(:new, T, tf...), Ti)
            elseif TT === NoTangent
                # Whole aggregate is non-differentiable (e.g. `Tuple{Int,Int}`, a fieldless struct).
                shadow[i] = NoTangent()
            elseif T <: Tuple || T <: NamedTuple
                # Tuple/NamedTuple tangents are same-shape but tangent-typed: a non-differentiable
                # slot holds `NoTangent()`, differentiable slots hold their tangent. Built via `%new`
                # of the (tangent-typed) aggregate.
                tf = Any[tt(fieldtype(T, j)) === NoTangent ? NoTangent() : tresolve(args[j])
                         for j in eachindex(args)]
                shadow[i] = emit!(Expr(:new, TT, tf...), TT)
            elseif T <: Array
                # Array construction — step 4 of 4 of the allocation sequence (see the
                # `Core.memorynew` arm below for the other end): `Expr(:new, Vector{P}, ref, size)`.
                # `tangent_type(Array{P,N}) === Array{tangent_type(P),N}` (tangents.jl), so `TT` is
                # *directly* the concrete tangent type to `%new` here — no manual `eltype`/`ndims`
                # unwrapping, unlike the generic-struct path below. Exactly 2 fields (`args[1]` =
                # `:ref`, `args[2]` = `:size`):
                #  * `:ref` is differentiable data (a `MemoryRef{P}` into the shadow memory allocated
                #    by `Core.memorynew` above) -> `tresolve` it to the shadow `MemoryRef{tangent_type(P)}`
                #    (mirrors the `Core.memoryrefnew` arm, whose own shadow is exactly this ref).
                #  * `:size` is the array's *shape* — structural, non-differentiable (`Core.tuple(n)`'s
                #    own shadow is deliberately `NoTangent()`) -> use the *primal*'s own size tuple
                #    verbatim (`presolve`, not `tresolve`) rather than fabricate a bogus
                #    all-`NoTangent()` shadow tuple. Matches Mooncake's reference `_new_` rule for
                #    `Array{P,N}`: shadow ref, primal's own size.
                shadow[i] = emit!(Expr(:new, TT, tresolve(args[1]), vpresolve(args[2])), TT)
            else
                # General (im)mutable user struct → `Tangent`/`MutableTangent`. `tresolve`d field
                # tangents already carry `NoTangent()` for non-diff fields (dropped into the matching
                # `NoTangent` slot). When every slot is always-initialised (no `PossiblyUninitTangent`)
                # and `%new` supplies a value for all of them, build the backing `NamedTuple` and wrap
                # it with `%new` directly — no `build_tangent` call to dynamic-dispatch. Otherwise fall
                # back to `build_tangent`, which does the `PossiblyUninitTangent` wrapping and
                # empty-slot construction the direct `%new` can't.
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
            # a 3-arg `getfield`/`memoryref*` boundscheck argument. Emitted through unchanged; its
            # own result is a non-differentiable `Bool`.
            primal[i] = emit!(Expr(:boundscheck, (presolve(a) for a in s.args)...), Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif isa(s, Expr) && s.head === :throw_undef_if_not
            # Pure control marker: raises `UndefVarError`/`UndefRefError` for an unassigned slot or
            # boxed-capture field (the guard Julia inserts around a captured variable's read once
            # reassignment has forced it into a `Core.Box`). `args[1]` is a bare Symbol/GlobalRef name
            # — copied through verbatim, never presolved/dualized. `args[2]` is the Bool condition
            # (a literal in a throw-only block, or a genuine SSA operand on a live path) and does need
            # resolving. Its own result is never consumed (no shadow-bearing value), so give it the
            # zero tangent of its (non-differentiable) type like `:boundscheck` above.
            primal[i] = emit!(Expr(:throw_undef_if_not, s.args[1], presolve(s.args[2])), Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos = s.head === :invoke ? s.args[2] : s.args[1]
            actual = s.head === :invoke ? s.args[3:end] : s.args[2:end]
            f = _calleeval(fpos, iworld)
            if isa(f, Core.IntrinsicFunction)
                # Dispatch straight to a per-intrinsic rule (`apply_intrinsic_frule!` in
                # `src/intrinsics.jl`), which emits the primal + shadow IR directly using the same
                # `opf`/`presolve`/`tresolve`/`zero_shadow` primitives as every other case in this
                # loop — no `Dual` boxing, `frule!!` dispatch, or `CodeInstance` resolution. Explicit,
                # not implicit: the fallback method returns `nothing`, so an *unregistered* intrinsic
                # bails gracefully with a located reason rather than crashing or silently miscompiling.
                why = reason[]
                res = apply_intrinsic_frule!(Val(f), actual, Ti, intrinsic_ctx)
                if res === nothing
                    # A *registered* rule can also decline — the pointer rules do, when the shadow's
                    # element stride wouldn't match the primal's — recording its own explanation in
                    # `ctx.reason` (unlocated; the location is added here). Only claim "no rule
                    # registered" when the rule left the reason untouched, or the message is a lie.
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
                # `src/builtins.jl`), which emits the primal + shadow IR directly using the
                # `presolve`/`tresolve`/`optype`/`tt` primitives — mirrors the intrinsic dispatch
                # above. The fallback method returns `nothing`, so an unregistered builtin (e.g.
                # `Core.memoryrefoffset`, used by `push!`/`resize!`, or a non-bits/undef-checked
                # array element access) bails gracefully with a located reason.
                why = reason[]
                res = apply_builtin_frule!(Val(f), actual, Ti, builtin_ctx)
                if res === nothing
                    # As with intrinsics above: a registered rule that declines records its own reason.
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
            k = length(s.values)
            pvals = Vector{Any}(undef, k); tvals = Vector{Any}(undef, k)
            for j in 1:k
                isassigned(s.values, j) || continue    # mirror the primal's own unassigned slot
                v = s.values[j]
                if isa(v, Core.SSAValue) && !isassigned(primal, v.id)
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (pvals, j, true))
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (tvals, j, false))
                else
                    pvals[j] = presolve(v); tvals[j] = tresolve(v)
                end
            end
            primal[i] = emit!(Core.PhiNode(s.edges, pvals), Ti)
            shadow[i] = emit!(Core.PhiNode(s.edges, tvals), tt(Ti))
        elseif isa(s, Core.UpsilonNode)
            # try/catch: a value live into a handler is captured by an `UpsilonNode` (at a def or
            # right before the `:enter`) and collected by a `PhiCNode` at the handler top. Duplicate
            # into a primal + shadow upsilon, exactly like a `PhiNode`. An unassigned `ϒ ()` slot
            # (a live-but-undefined capture) is mirrored as an empty upsilon in both copies.
            if isdefined(s, :val)
                primal[i] = emit!(Core.UpsilonNode(presolve(s.val)), Ti)
                shadow[i] = emit!(Core.UpsilonNode(tresolve(s.val)), tt(Ti))
            else
                primal[i] = emit!(Core.UpsilonNode(), Ti)
                shadow[i] = emit!(Core.UpsilonNode(), tt(Ti))
            end
        elseif isa(s, Core.PhiCNode)
            # Collects `UpsilonNode`s at a handler entry. Its operands are `SSAValue`s that must
            # reference upsilons (`verify_ir`), so the primal-phic references the primal upsilons and
            # the shadow-phic the shadow ones — reusing `primal[v.id]`/`shadow[v.id]`. The same
            # `pending` forward-ref mechanism as `PhiNode` covers a capture defined later in linear
            # order (a try/catch inside a loop).
            k = length(s.values)
            pvals = Vector{Any}(undef, k); tvals = Vector{Any}(undef, k)
            for j in 1:k
                isassigned(s.values, j) || continue
                v = s.values[j]
                if isa(v, Core.SSAValue) && !isassigned(primal, v.id)
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (pvals, j, true))
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (tvals, j, false))
                else
                    pvals[j] = presolve(v); tvals[j] = tresolve(v)
                end
            end
            primal[i] = emit!(Core.PhiCNode(pvals), Ti)
            shadow[i] = emit!(Core.PhiCNode(tvals), tt(Ti))
        elseif isa(s, Core.EnterNode)
            # Control marker beginning a protected region. `catch_dest` is a basic-block number
            # (topology preserved 1:1) so it carries over unchanged; the token it defines is
            # referenced by `:leave`/`:pop_exception` via its primal SSA. It terminates its block, so
            # its type must be `Any` (verify_ir). `scope`, if present, is an ordinary value operand.
            ent = isdefined(s, :scope) ?
                    Core.EnterNode(s.catch_dest, presolve(s.scope)) : Core.EnterNode(s.catch_dest)
            primal[i] = emit!(ent, Any); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :leave
            # Pops one or more `:enter` scopes, referencing the `EnterNode`(s) by `SSAValue` (or
            # `nothing`); `presolve` remaps each to the enter's primal SSA. Pure control, no value.
            emit!(Expr(:leave, Any[presolve(a) for a in s.args]...), Any)
        elseif isa(s, Expr) && s.head === :pop_exception
            primal[i] = emit!(Expr(:pop_exception, presolve(s.args[1])), Ti); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :the_exception
            # The caught exception object has no meaningful tangent (non-differentiable): keep the
            # primal, give it a zero tangent like other non-diff results.
            primal[i] = emit!(Expr(:the_exception), Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif isa(s, Expr) && s.head === :foreigncall
            # `ccall`. Dispatched per *target symbol* to `apply_foreigncall_frule!`
            # (`src/foreigncalls.jl`), mirroring the intrinsic/builtin dispatch above. Native code can
            # write through any pointer it is handed, so unlike an unregistered intrinsic there is no
            # "primal plus a zero tangent" fallback that would be safe here: an unregistered target
            # bails, full stop.
            fc = _fc_parse(s)
            if fc === nothing
                reason[] = "`foreigncall` with a non-literal target (a runtime function pointer) " *
                           "at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            why = reason[]
            res = apply_foreigncall_frule!(Val(fc.name), fc, Ti, foreigncall_ctx)
            if res === nothing
                # As with intrinsics/builtins: a registered rule that declines records its own reason.
                reason[] = reason[] === why ?
                    "no dualization rule for `foreigncall` target `$(fc.name)` at %$i: " *
                    "`$(_stmt_str(s))` (add one in src/foreigncalls.jl via " *
                    "`apply_foreigncall_frule!`)" :
                    "$(reason[]) at %$i: `$(_stmt_str(s))`"
                return nothing
            end
            primal[i], shadow[i] = res
        elseif isa(s, Expr) && s.head === :loopinfo
            # `@simd`'s loop marker (`base/simdloop.jl` is Base's only construction site). Pure
            # metadata: its operands are `Symbol`s/`nothing` that codegen turns into LLVM loop
            # metadata, and `:loopinfo` isn't in the compiler's `is_relevant_expr`, so `userefs` never
            # traverses them at all — presolving them would be actively wrong, not just unnecessary.
            # Copied through verbatim, with no shadow value of its own.
            #
            # Two invariants this relies on: codegen consumes the marker positionally from the
            # block's terminator backwards, so nothing may be emitted between it and the terminator
            # (this pass emits shadow statements *before* it, which is fine); and a `julia.ivdep`
            # marker carries over to the dualized loop, where it now also asserts non-aliasing for the
            # shadow buffer's accesses — true whenever it was true of the primal, since the shadow
            # mirrors the primal's access pattern one-for-one.
            primal[i] = emit!(Expr(:loopinfo, s.args...), Ti); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :gc_preserve_begin
            # `GC.@preserve`: roots its operands until the matching `:gc_preserve_end`, so an interior
            # pointer into them (`pointer(v)`) stays valid. The dualized code holds interior pointers
            # into *both* the primal object and its shadow, so root both — preserving only the primal
            # would let the shadow array be collected while a shadow `Ptr` into it is still live.
            # A tangent that isn't an SSA/argument (a literal `NoTangent()`, a spliced constant
            # tangent) is skipped: there is no heap object of ours to root. Type is `Any`, matching
            # what inference assigns the token.
            pargs = Any[]
            for a in s.args
                push!(pargs, presolve(a))
                t = tresolve(a)
                (isa(t, Core.SSAValue) || isa(t, Core.Argument)) && push!(pargs, t)
            end
            primal[i] = emit!(Expr(:gc_preserve_begin, pargs...), Any); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :gc_preserve_end
            # Ends the region, referencing the `:gc_preserve_begin` token by `SSAValue` — `presolve`
            # remaps it to the begin's primal SSA. `verify_ir` skips its usual dominance check for
            # this head (a token may span try/catch blocks), so nothing else is needed.
            primal[i] = emit!(Expr(:gc_preserve_end, presolve(s.args[1])), Ti); shadow[i] = primal[i]
        elseif isa(s, GlobalRef)
            # A bare GlobalRef *statement* is a global-variable load (not a pure alias like a
            # `PiNode`): it must be *emitted* as a real instruction. Aliasing it away would let the
            # raw `GlobalRef` leak into a later operand position, which `Core.Compiler.verify_ir`
            # rejects unless the binding is proven constant (a non-`const` global like a plain
            # `Ref{Any}` isn't). The loaded value is external, non-differentiable state, so its
            # shadow is the zero tangent (matching how other non-diff sources are treated).
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
        stream.flag[i] = CC.IR_FLAG_NULL
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
    # `pir.valid_worlds` (not the constructor's unbounded default): this IR mirrors `pir` statement
    # for statement, so it's valid for exactly as long as `pir` is — see `cfg_ir.jl`'s
    # `lower_cfg_blocks_to_ir`, which threads the same value through for the same reason. Leaving the
    # default sentinel here makes `Base`-namespaced builtin/intrinsic `GlobalRef`s (pulled in by
    # inlined primal library code) print as "dynamic" in the IR pretty-printer, even though they're
    # ordinary static calls.
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[], pir.valid_worlds)
    CC.verify_ir(ir)                                # a failure here is a bug in this transform, not
                                                     # unsupported input IR — let it throw plainly
    return ir
end

# Type helper. Types are taken directly from the primal IR, so they are exact rather than guessed;
# the fallback (`typeof(x)`) only covers genuine literal constants.
_optype(pir, @nospecialize x) = isa(x, Core.SSAValue) ? pir.stmts[x.id][:type] :
                                isa(x, Core.Argument) ? pir.argtypes[x.n] : typeof(x)

# The type of the *value* an operand denotes, as a bare `Type` — what a caller that is about to type
# a `CoDual`/`Dual` slot for that operand needs, and what `_optype` alone cannot give:
#
#  * a `GlobalRef` *names* a binding rather than being a value, so `_optype`'s literal fallback
#    answers `GlobalRef` — the type of the node. Resolve it the way `gref_optype` does (const-gated
#    at `world`, never the ambient task world — see `_globalref_isconst`); a non-`const` binding is
#    only knowable at run time, hence `Any`. (In practice a non-`const` binding survives
#    optimization as its own `%k = Main.g` load statement, so it arrives here as an `SSAValue`.)
#  * a `QuoteNode` is the same defect one node kind over (`typeof(node) === QuoteNode`).
#  * an `SSAValue`/`Argument` type may be a lattice element (`Core.Const`/`PartialStruct`) the
#    primal's own const-prop produced; every consumer here wants the widened `Type` (`_widen`, defined
#    with the rest of the reverse-mode type helpers in `reverse_interp.jl`).
#
# Used by reverse mode's `_static_recursible_call`; forward mode resolves the same two node kinds
# through its own `gref_operand!`/`gref_optype` closures, which also *emit* for a non-`const` load.
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
# tells the user *what* IR construct was unsupported (e.g. the actual `Base.arrayref(...)` call)
# rather than just its kind. Kept defensive: `show` on a stray node must never mask the real bail.
function _stmt_str(@nospecialize s)
    str = try sprint(show, s) catch; string(s) end
    str = replace(str, r"\s+" => " ")
    length(str) > 200 ? string(first(str, 197), "...") : str
end
# A field's shadow carries the *primal* value through (rather than its tangent) exactly when the
# field's tangent is `NoTangent` but its slot type would reject a `NoTangent` value — i.e. the field
# is non-differentiable yet its slot is not itself a `NoTangent` slot. `tangent_type(T) === NoTangent`
# is the principled statement of that (it subsumes the old singleton-only test: every non-`NoTangent`
# singleton, like a `typeof(sin)` slot, has `NoTangent` tangent, and so do `Int`, `Bool`, `Symbol`,
# `Tuple{Int,Int}`, … — all of which would otherwise get a spurious `NoTangent()` dropped into their
# slot). Two guards:
#  * `isconcretetype(T)` — only concrete types have a total, non-throwing `tangent_type`; an abstract
#    slot (`::Integer`, `::Any`) can't be classified statically (its runtime value may or may not be
#    differentiable), so we conservatively leave it on the tangent path, unchanged from before.
#  * `T <: Type` handled separately — `Type{Float64}`/`DataType` are `Type`-valued (tangent `NoTangent`)
#    but *not* concrete (`isconcretetype(Type{Float64}) === false`) and `Base.issingletontype` reports
#    them non-singleton (a documented Julia quirk), so the concrete clause misses them; every
#    `T <: Type` has `NoTangent` tangent, so carrying the primal is always right there.
# Differentiable structs/scalars have a non-`NoTangent` tangent, so they still take their tangent.
_nondiff_field(@nospecialize T) = T isa DataType && T !== NoTangent &&
    (T <: Type || (isconcretetype(T) && tangent_type(T) === NoTangent))


function frule_body(world::UInt, source, self, dual_argtypes)
    argnames = Any[Symbol("#self#"), :dualargs]

    # Resolve the `dualized_impl` specialization for these dual argument types.
    impl_tt = Tuple{typeof(dualized_impl), dual_argtypes...}
    interp = ADInterpreter{Forward}(; world)
    match, _ = Core.Compiler.findsup(impl_tt, Core.Compiler.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                                :(error("Differ: no dualized_impl match")), true)
    end
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance

    # Compile the dualized body under ADInterpreter -> an invoke-able CodeInstance.
    # Call typeinf_ext_toplevel directly (not CompilerPlugins.typeinf, which would recreate the
    # interpreter at tls_world_age() — stale inside a generator) so it uses `interp`'s generation
    # world; `finishinfer!`/`optimize` then build and install the dual IR (see contextual.jl).
    cinst = Compiler.typeinf_ext_toplevel(interp, impl_mi, Compiler.SOURCE_MODE_ABI)

    # Trivial generated body: return invoke(dualized_impl, cinst, dualargs...)
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(dualized_impl, $cinst, dualargs...)), true)

    # `impl_mi` is a real backedge: if its own CodeInstance is invalidated (see below — that's where
    # the primal/`frule!!` dependencies actually get registered, via `finishinfer!` in `contextual.jl`),
    # this wrapper must be invalidated and regenerated too, so it re-embeds a fresh `cinst`.
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
