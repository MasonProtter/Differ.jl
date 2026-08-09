module Contextual

using Core: MethodInstance, CodeInstance, CodeInfo, Compiler
using Core.Compiler:
    AbstractInterpreter, InferenceParams, OptimizationParams, InferenceResult, IRCode
using Base: specialize_method

const CC = Core.Compiler

# ---------------------------------------------------------------------------
# ContextualInterpreter: a generic AbstractInterpreter that compiles a *contextually
# transformed* version of a primal method. The transform is a single source-to-source pass on
# the primal's post-optimization `IRCode`, installed into the normal typeinf pipeline so it
# produces an ordinary `invoke`-able `CodeInstance`. The concrete transform is supplied by
# a plugin (e.g. Differ's forward/reverse-mode AD engines) via the `build_contextual_ir` hook.
# Modeled on the plugin shape in `julia/Compiler/extras/CompilerDevTools`.
# ---------------------------------------------------------------------------

# `owner::T` is immutable, portable plugin identity only (e.g. "which mode", plus small
# config). It IS the `cache_owner` partition key directly, with no indirection — that's
# exactly why it must stay immutable/portable: two `owner` values that are `===`/egal-equal
# must always be safe to share a CodeInstance cache partition, which only holds if `owner`
# never carries per-session mutable scratch.
#
# `custom_state::S` is whatever additional bookkeeping the owner needs beyond what this
# framework itself manages, in whatever shape the owner chooses (`NamedTuple`, a dedicated
# struct, or `nothing`). Mutable, per-session scratch (in-progress sets, bail-reason logs, …)
# goes here — deliberately kept out of `owner` and out of `cache_owner`'s reach.
struct ContextualInterpreter{T,S} <: AbstractInterpreter
    owner::T
    custom_state::S
    inf_cache::Vector{InferenceResult}
    world::UInt
    inf_params::InferenceParams
    opt_params::OptimizationParams
    codegen_cache::IdDict{CodeInstance, CodeInfo}
    # Per-compile scratch: the transformed IRCode built in `finishinfer!` (which also supplies
    # the return type) and installed as the optimization result in `optimize`. Keyed by the
    # carrier MethodInstance being compiled. Safe because one interpreter instance serves both
    # hooks of a given frame. Framework-owned, not the owner's concern: `finishinfer!` writes
    # `interp.transformed_ir[me.linfo] = ir` and `optimize` reads it back for the same
    # MethodInstance, passing the built IR between the two pipeline hooks.
    transformed_ir::IdDict{MethodInstance, IRCode}
    # Backedges discovered while building `transformed_ir[mi]`, folded into `me.src.edges` in
    # `finishinfer!` so the ordinary `compute_edges!`/`store_backedges` path registers real
    # Julia backedges, without a plugin calling any invalidation ccall itself.
    transformed_edges::IdDict{MethodInstance, Vector{Any}}
end

function ContextualInterpreter(owner::T, custom_state::S;
                                world::UInt=Base.get_world_counter(),
                                inf_params::InferenceParams=InferenceParams(),
                                opt_params::OptimizationParams=OptimizationParams()) where {T,S}
    # `jl_get_world_counter` returns the `typemax` sentinel while `in_pure_callback` is set — i.e.
    # inside any `@generated` generator (`julia/src/gf.c`). So the obvious
    # `@assert world <= Base.get_world_counter()` is *vacuously true* in exactly the context this
    # interpreter is normally built from, and would never catch the real mistake: defaulting `world`
    # from `Base.get_world_counter()` at generator time and silently getting `typemax`. Reject that
    # sentinel outright, and only compare against the counter when we're not in a pure callback.
    @assert world != typemax(UInt) """
        ContextualInterpreter needs a concrete inference world, got the `typemax` sentinel that \
        `Base.get_world_counter()` returns inside a `@generated` generator. Pass the generator's own \
        `world` argument explicitly."""
    let current = Base.get_world_counter()
        @assert current == typemax(UInt) || world <= current
    end
    return ContextualInterpreter{T,S}(
        owner, custom_state,
        InferenceResult[],
        world, inf_params, opt_params,
        IdDict{CodeInstance, CodeInfo}(),
        IdDict{MethodInstance, IRCode}(),
        IdDict{MethodInstance, Vector{Any}}(),
    )
end

Core.Compiler.InferenceParams(interp::ContextualInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::ContextualInterpreter) = interp.opt_params
Core.Compiler.get_inference_world(interp::ContextualInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::ContextualInterpreter) = interp.inf_cache
Core.Compiler.cache_owner(interp::ContextualInterpreter) = interp.owner
Core.Compiler.codegen_cache(interp::ContextualInterpreter) = interp.codegen_cache

# ---------------------------------------------------------------------------
# World-age hygiene for pass code that runs at generator time.
#
# A plugin's entry point is a `@generated` function, and `jl_call_staged` pins the generator body's
# task world age to the *generated method's* `Method.primary_world` — fixed when that method was
# defined, and unmovable afterwards (regenerating the body does not move it; neither does
# invalidation). Every ordinary function the transform calls therefore dispatches at that pin, so
# methods added by any *later-loaded* package — a sibling package, and always a package extension —
# are invisible to it. That is what broke forward-over-reverse across the package split: DifferReverse's
# `tangent_type` overrides and every one of the extension's coupling hooks silently resolved to
# DifferCore's generic fallback / to their own inert defaults.
#
# THE CONTRACT for any pass built on this framework:
#
#   Pass code running at generator time may reach another package only through `at_world`, or
#   through a call emitted into the IR (which is compiled later, at the real world). Never by
#   direct dispatch, and never through mutable global state — a generator reading a registry
#   populated in `__init__` is impure: nothing invalidates its cached result when that registry
#   changes, and a result baked into a precompile image while the registry was empty is simply
#   wrong. Every such lookup must record an `mt_edge!` so a later method definition invalidates
#   the carrier.
#
# `Core._call_in_world_total` is the primitive that works here. Unlike `Base.invoke_in_world` — which
# is a no-op while `in_pure_callback` is set, so it silently leaves the pin in place — it is not
# guarded, and the world it switches to covers *nested* dispatch inside the callee too. That nesting
# is load-bearing: `tangent_type(Stack{T})` recurses into `tangent_type(T)`, and a switch that only
# covered the outermost frame would fix nothing. It is the same primitive inference itself uses for
# concrete evaluation, which is why it is legitimate from a pure context; its one restriction is that
# the callee must not `eval`/`include`.
# ---------------------------------------------------------------------------

# Call `f(args...)` with dispatch resolved at the interpreter's inference world instead of at the
# generator's pin. Use for every call from transform code into a generic function another package can
# extend.
at_world(world::UInt, @nospecialize(f), @nospecialize(args...)) =
    Core._call_in_world_total(world, f, args...)
# Convenience for the common case; transform code that already carries a bare `iworld`
# (`CC.get_inference_world(interp)`) rather than the interpreter itself uses the method above.
at_world(interp::ContextualInterpreter, @nospecialize(f), @nospecialize(args...)) =
    at_world(interp.world, f, args...)

# Record a method-table backedge on `sig`, so that a method *appearing later* for that signature
# invalidates the carrier being built. This is the encoding the ordinary
# `compute_edges!`/`store_backedges` path expects (a `sig`/`Core.methodtable` pair inline in the edge
# vector) and is what `finishinfer!` folds into `me.src.edges` below.
#
# A resolved-`MethodInstance` edge is NOT a substitute: it covers "this method changed", whereas the
# case that matters for `at_world` is "a more specific method now exists". Deduplicated because the
# transform queries the same handful of signatures once per statement, and an edge vector with a
# thousand copies of `Tuple{typeof(tangent_type), Type{Float64}}` would bloat the invalidation graph
# for no benefit.
function mt_edge!(edges::Vector{Any}, @nospecialize(sig))
    for i in 1:(length(edges) - 1)
        edges[i] === sig && edges[i + 1] === Core.methodtable && return edges
    end
    push!(edges, sig, Core.methodtable)
    return edges
end

# Build a lowered `CodeInfo` from an expression. Used by a plugin's `@generated` fallback to
# produce the trivial generated body that `invoke`s the compiled transformed `CodeInstance`.
function expr_to_codeinfo(m::Module, argnames, spnames, sp, e::Expr, isva::Bool=false)
    lam = Expr(:lambda, argnames,
               Expr(Symbol("scope-block"),
                    Expr(:block,
                         Expr(:return,
                              Expr(:block,
                                   e,
                                   )))))
    ex = if spnames === nothing || isempty(spnames)
        lam
    else
        Expr(Symbol("with-static-parameters"), lam, spnames...)
    end
    ci = Base.generated_body_to_codeinfo(ex, m, isva)
    @assert ci isa Core.CodeInfo "Failed to create a CodeInfo from the given expression. This might mean it contains a closure or comprehension?\n Offending expression: $e"
    ci
end

# ---------------------------------------------------------------------------
# Installing the transformed IR via `finishinfer!` and `optimize`.
#
# A plugin's entry point (e.g. a `@generated` fallback) asks this interpreter to compile a
# *carrier* MethodInstance whose `specTypes` encodes the transformed signature. We compile it
# by transforming the corresponding primal method's post-optimization `IRCode` into a new
# `IRCode` and splicing that transform into the pipeline at two points:
#
#   * `finishinfer!` — the pipeline function that *supplies* the `CodeInstance` return type. We
#     build the transformed IR here and set `me.bestguess` to its return type, so the ordinary
#     `jl_fill_codeinst` writes the correct type once (no post-hoc patching).
#   * `optimize` — installs the already-built IR as the optimization result via the ordinary
#     `ipo_dataflow_analysis!` + `finishopt!`.
#
# Plugins hook in by overriding `build_contextual_ir`; the default builds nothing, so any MI a
# plugin doesn't handle flows through the ordinary pipeline unchanged. When a plugin's
# transform hits a construct it can't handle, it returns `nothing`: we leave `me.bestguess`/the
# cache untouched, the carrier stub's throwing body flows through the pipeline normally, and
# the compiled CI raises when invoked — a graceful bail.
# ---------------------------------------------------------------------------

# Plugin hook: build the transformed `IRCode` for a carrier MethodInstance, or `nothing` to
# leave `mi` to the ordinary pipeline. Overridden per plugin (e.g. Differ's forward/reverse
# engines).
build_contextual_ir(::ContextualInterpreter, ::MethodInstance) = nothing

# Builds the transformed IR and sets `me.bestguess` to its return type so the generic
# `finishinfer!` freezes the correct type into the CodeInstance.
function CC.finishinfer!(me::CC.InferenceState, interp::ContextualInterpreter, cycleid::Int,
                         opt_cache::IdDict{MethodInstance, CodeInstance})
    ir = build_contextual_ir(interp, me.linfo)
    if ir !== nothing
        # `build_contextual_ir` builds `ir` via the 6-arg `CC.IRCode(...)` constructor, which
        # defaults `valid_worlds` to the unbounded sentinel `WorldRange(0, typemax(UInt))`. That
        # sentinel fails `abstract_eval_globalref_type`'s partition-coverage check
        # (`Compiler/src/abstractinterpretation.jl`) for any `GlobalRef` whose binding partition
        # doesn't itself span the full sentinel range — e.g. `Base.add_float` (an *imported*
        # binding, pulled in when inlining primal library code), whose partition starts at a
        # finite world — so `argextype`/the IR pretty-printer mislabels those calls "dynamic"
        # even though they're ordinary builtin/intrinsic calls. Fix by giving `ir` a real world
        # range: `interp.world` onward, since every binding this IR references already resolved
        # successfully at that world.
        ir = CC.IRCode(ir.stmts, ir.cfg, ir.debuginfo, ir.argtypes, ir.meta, ir.sptypes,
                       CC.WorldRange(interp.world, typemax(UInt)))
        interp.transformed_ir[me.linfo] = ir
        me.bestguess = CC.compute_ir_rettype(ir)
        # Fold in whatever backedges the plugin's transform discovered. These must land on
        # *this* CodeInstance (`me.linfo`'s own compile), not merely on some caller of it:
        # `finish!` on THIS `InferenceState` is what calls `store_backedges` against
        # `me.linfo`'s CodeInstance below, and a `@generated` caller's own cached CodeInstance
        # for a call to `me.linfo` is only re-examined for staleness by looking `me.linfo`'s
        # cache up again — so if `me.linfo`'s CodeInstance were never itself invalidated, a
        # caller regenerating around it would just keep reusing the same stale result. Setting
        # `me.src.edges` here, before delegating to the generic `finishinfer!` (which internally
        # calls `compute_edges!(me)`, reading this field), is the standard, documented way a
        # generator-produced `CodeInfo` declares extra edges (`compute_edges!` in
        # `typeinfer.jl`).
        edges = get(interp.transformed_edges, me.linfo, nothing)
        if edges !== nothing && !isempty(edges)
            existing = me.src.edges
            me.src.edges = (existing === nothing || existing === CC.empty_edges) ? edges :
                Any[existing..., edges...]
        end
    end
    @invoke CC.finishinfer!(me::CC.InferenceState, interp::CC.AbstractInterpreter,
                            cycleid::Int, opt_cache::IdDict{MethodInstance, CodeInstance})
end

# Replaces the optimization result with the transformed IR built in `finishinfer!`, then runs
# the ordinary IPO-safe optimization passes on it (inlining, SROA, ADCE, …) so the synthetic
# construction/rule calls are inlined and immutable results scalar-replaced.
function CC.optimize(interp::ContextualInterpreter, opt::CC.OptimizationState,
                     caller::CC.InferenceResult)
    ir = get(interp.transformed_ir, caller.linfo, nothing)
    if ir !== nothing
        ir = run_ipo_passes!(ir, opt)
        CC.ipo_dataflow_analysis!(interp, opt, ir, caller)
        CC.finishopt!(interp, opt, ir)
        return nothing
    end
    @invoke CC.optimize(interp::CC.AbstractInterpreter, opt::CC.OptimizationState,
                        caller::CC.InferenceResult)
end

# The IRCode half of `run_passes_ipo_safe` (our transformed IR is already SSA IRCode, so
# CONVERT/SLOT2REG are skipped).
function run_ipo_passes!(ir::IRCode, opt::CC.OptimizationState)
    ir = CC.compact!(ir)
    ir = CC.ssa_inlining_pass!(ir, opt.inlining, opt.src.propagate_inbounds)
    ir = CC.compact!(ir)
    ir = CC.sroa_pass!(ir, opt.inlining)
    ir, _ = CC.adce_pass!(ir, opt.inlining)
    ir = CC.compact!(ir, true)
    return ir
end

export ContextualInterpreter, build_contextual_ir, expr_to_codeinfo, run_ipo_passes!
export at_world, mt_edge!

end # module Contextual
