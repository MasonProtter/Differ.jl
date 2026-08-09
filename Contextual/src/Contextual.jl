module Contextual

using Core: MethodInstance, CodeInstance, CodeInfo, Compiler
using Core.Compiler:
    AbstractInterpreter, InferenceParams, OptimizationParams, InferenceResult, IRCode
using Base: specialize_method

const CC = Core.Compiler

# A generic AbstractInterpreter that compiles a *contextually transformed* version of a
# primal method: a source-to-source pass on the primal's post-optimization `IRCode`, installed
# into the normal typeinf pipeline so it produces an ordinary `invoke`-able `CodeInstance`. The
# concrete transform is supplied by a plugin (Differ's forward/reverse-mode engines) via the
# `build_contextual_ir` hook. Modeled on `julia/Compiler/extras/CompilerDevTools`.

# `owner::T` must be immutable/portable: it's used directly as the `cache_owner` partition key,
# so two egal-equal `owner`s must always be safe to share a CodeInstance cache partition.
# `custom_state::S` is whatever mutable per-session bookkeeping the owner needs beyond that
# (in-progress sets, bail-reason logs, …) — kept separate so it never leaks into `cache_owner`.
struct ContextualInterpreter{T,S} <: AbstractInterpreter
    owner::T
    custom_state::S
    inf_cache::Vector{InferenceResult}
    world::UInt
    inf_params::InferenceParams
    opt_params::OptimizationParams
    codegen_cache::IdDict{CodeInstance, CodeInfo}
    # Transformed IRCode built in `finishinfer!`, installed as the optimization result in
    # `optimize`; keyed by the carrier MethodInstance being compiled.
    transformed_ir::IdDict{MethodInstance, IRCode}
    # Backedges discovered while building `transformed_ir[mi]`, folded into `me.src.edges` in
    # `finishinfer!` so the ordinary `compute_edges!`/`store_backedges` path registers them.
    transformed_edges::IdDict{MethodInstance, Vector{Any}}
end

function ContextualInterpreter(owner::T, custom_state::S;
                                world::UInt=Base.get_world_counter(),
                                inf_params::InferenceParams=InferenceParams(),
                                opt_params::OptimizationParams=OptimizationParams()) where {T,S}
    # `Base.get_world_counter()` returns the `typemax` sentinel inside any `@generated` generator
    # (`in_pure_callback`), so the obvious `world <= get_world_counter()` assert would be
    # vacuously true right where the real mistake — defaulting `world` from the counter at
    # generator time — happens. Reject the sentinel outright instead.
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

# World-age hygiene for pass code that runs at generator time.
#
# `jl_call_staged` pins a generator body's task world age to the generated method's
# `Method.primary_world`, fixed at definition and never moved afterwards. Ordinary dispatch from
# inside the generator therefore can't see methods added by any later-loaded package (a sibling
# package, or any package extension) — the cause of the forward-over-reverse regression across the
# package split, where DifferReverse's `tangent_type` overrides were invisible to it.
#
# Contract: pass code running at generator time may reach another package only through
# `at_world`, or through a call emitted into the IR (compiled later, at the real world) — never by
# direct dispatch, and never through mutable global state (a generator reading a registry mutated
# in `__init__` bakes in whatever the registry held at that world, with nothing to invalidate it
# later). Every `at_world` lookup must record an `mt_edge!` so a later method definition
# invalidates the carrier.
#
# `Core._call_in_world_total` is what `at_world` wraps. Unlike `Base.invoke_in_world` (a no-op
# while `in_pure_callback` is set), it actually switches worlds, and covers nested dispatch inside
# the callee too — load-bearing, since e.g. `tangent_type(Stack{T})` recurses into
# `tangent_type(T)`. Its one restriction is that the callee must not `eval`/`include`.

# Call `f(args...)` with dispatch resolved at the interpreter's inference world instead of at the
# generator's pin. Use for every call from transform code into a generic function another package
# can extend.
at_world(world::UInt, @nospecialize(f), @nospecialize(args...)) =
    Core._call_in_world_total(world, f, args...)
at_world(interp::ContextualInterpreter, @nospecialize(f), @nospecialize(args...)) =
    at_world(interp.world, f, args...)

# Record a method-table backedge on `sig` so a method appearing later for that signature
# invalidates the carrier being built — the `sig`/`Core.methodtable` pair inline in the edge
# vector that `compute_edges!`/`store_backedges` expects. A resolved-`MethodInstance` edge is not
# a substitute: it covers "this method changed", not "a more specific method now exists".
# Deduplicated since the same signature gets queried repeatedly per statement.
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

# Installing the transformed IR happens at two pipeline hooks: `finishinfer!` builds it and sets
# `me.bestguess` to its return type (so `jl_fill_codeinst` writes the correct type once), and
# `optimize` installs it as the optimization result via the ordinary `ipo_dataflow_analysis!` +
# `finishopt!`. When a plugin's transform can't handle a construct, `build_contextual_ir` returns
# `nothing`: the carrier stub's throwing body flows through the pipeline unchanged, and the
# compiled CodeInstance raises when invoked instead of hard-erroring at compile time.

# Plugin hook: build the transformed `IRCode` for a carrier MethodInstance, or `nothing` to leave
# `mi` to the ordinary pipeline. Overridden per plugin.
build_contextual_ir(::ContextualInterpreter, ::MethodInstance) = nothing

# Builds the transformed IR and sets `me.bestguess` to its return type so the generic
# `finishinfer!` freezes the correct type into the CodeInstance.
function CC.finishinfer!(me::CC.InferenceState, interp::ContextualInterpreter, cycleid::Int,
                         opt_cache::IdDict{MethodInstance, CodeInstance})
    ir = build_contextual_ir(interp, me.linfo)
    if ir !== nothing
        # `build_contextual_ir`'s 6-arg `CC.IRCode(...)` constructor defaults `valid_worlds` to
        # the unbounded sentinel `WorldRange(0, typemax(UInt))`, which fails
        # `abstract_eval_globalref_type`'s partition-coverage check for any `GlobalRef` whose
        # binding partition starts at a finite world (e.g. `Base.add_float`, pulled in via
        # inlining) — such calls get mislabeled "dynamic" instead of ordinary builtin/intrinsic
        # calls. Fix by giving `ir` a real world range from `interp.world` onward.
        ir = CC.IRCode(ir.stmts, ir.cfg, ir.debuginfo, ir.argtypes, ir.meta, ir.sptypes,
                       CC.WorldRange(interp.world, typemax(UInt)))
        interp.transformed_ir[me.linfo] = ir
        me.bestguess = CC.compute_ir_rettype(ir)
        # Fold in backedges the plugin's transform discovered, before delegating to the generic
        # `finishinfer!` (which calls `compute_edges!(me)`, reading `me.src.edges`) — the
        # documented way a generator-produced `CodeInfo` declares extra edges. Must land on this
        # CodeInstance itself so a later method definition invalidates it, not just its callers.
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
