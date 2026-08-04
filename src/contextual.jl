using Core: MethodInstance, CodeInstance, CodeInfo, Compiler
using Core.Compiler:
    AbstractInterpreter, InferenceParams, OptimizationParams, InferenceResult, IRCode
using Base: specialize_method

const CC = Core.Compiler

# ---------------------------------------------------------------------------
# AD modes.
#
# `ADInterpreter` is parameterized by the AD mode it runs in, so one interpreter and one set
# of `finishinfer!`/`optimize` overrides serve every mode; only the mode-specific IR transform
# differs. Forward mode is implemented in `forward_interp.jl`, reverse mode in `reverse_interp.jl`.
# ---------------------------------------------------------------------------
abstract type ADMode end
struct Forward <: ADMode end
struct Reverse <: ADMode end

# ---------------------------------------------------------------------------
# ADInterpreter: a custom AbstractInterpreter that compiles a *contextually transformed*
# version of a primal method. The transform is a single source-to-source pass on the primal's
# post-optimization `IRCode`, installed into the normal typeinf pipeline so it produces an ordinary
# `invoke`-able `CodeInstance`. The concrete transform is chosen by the mode parameter `M` (e.g.
# forward-mode dualization in `forward_interp.jl`). Modeled on the plugin shape in
# `julia/Compiler/extras/CompilerDevTools`.
# ---------------------------------------------------------------------------

# `Forward`/`Reverse` double as the cache owner / plugin context: a compiled `CodeInstance`
# recompiles under an interpreter of the same mode (see the `CompilerPlugins.typeinf`/
# `typeinf_edge` entry points below).

struct ADInterpreter{M<:ADMode} <: AbstractInterpreter
    inf_cache::Vector{InferenceResult}
    world::UInt
    inf_params::InferenceParams
    opt_params::OptimizationParams
    codegen_cache::IdDict{CodeInstance, CodeInfo}
    # Per-compile scratch: the transformed IRCode built in `finishinfer!` (which also supplies the
    # return type) and installed as the optimization result in `optimize`. Keyed by the carrier
    # MethodInstance being compiled. Safe because one interpreter instance serves both hooks of a
    # given frame.
    transformed_ir::IdDict{MethodInstance, IRCode}
    # Backedges discovered while building `transformed_ir[mi]` (e.g. the primal method it was
    # dualized from, and any `frule!!` method resolved for a surviving high-level call) — see the
    # `build_contextual_ir`/`build_dual_ir` machinery in `forward_interp.jl`. `finishinfer!` folds
    # these into `me.src.edges` so the ordinary `compute_edges!`/`store_backedges` path registers
    # real Julia backedges, without Differ calling any invalidation ccall itself.
    transformed_edges::IdDict{MethodInstance, Vector{Any}}
    # Reverse-mode recursion cycle guard: carrier MethodInstances currently being built by
    # `build_reverse_fwds_ir`/`build_reverse_pullback_ir` (`reverse_interp.jl`), so a mutually-
    # recursive primal (A -> B -> A) bails cleanly instead of recursing into the transform forever.
    # Keyed by the *carrier* mi, not the primal mi — the fwds carrier alone has two independent
    # specializations per primal (`Ctx{Nothing}` fresh-tape vs `Ctx{<:Tape}` pre-allocated), and
    # building the pre-allocated one for a self-recursive primal legitimately requires *also*
    # compiling the `Ctx{Nothing}` sibling (a bounded, one-off nested compile, not a cycle) — a
    # primal-keyed guard would conflate that with genuine in-progress-ness and bail incorrectly. See
    # the comment on `build_reverse_fwds_ir` for the full reasoning.
    #
    # Direct self-recursion (a callee whose primal mi equals the current build's own) is a *separate*
    # question from this guard, answered locally and ctx-independently by
    # `reverse_fwds_recursive_ci`/`reverse_pullback_recursive_ci` from an explicitly-passed
    # `primal_mi` — not by consulting this field. It only reaches this guard at all when the
    # recursive edge's target carrier differs from the one currently being built (the `Ctx{Nothing}`-
    # sibling case above); a literal self-edge resolves to a static self-`:invoke` without recursing
    # into the builder at all. Mode-agnostic field (harmless, always empty, for `Forward`).
    in_progress::IdDict{MethodInstance, Nothing}
    # Why a carrier's transform bailed, keyed by that carrier MethodInstance — recorded by
    # `build_contextual_ir` right where it installs the error-raising IRCode, so a *caller* whose own
    # build recursed into it (`reverse_fwds_recursive_ci`, `reverse_interp.jl`) can report the inner
    # reason instead of just "the callee bailed". Best-effort: `typeinf_ext_toplevel` can hand back a
    # cached CodeInstance without re-running the transform, so a reader must tolerate a missing entry.
    #
    # For `Reverse`, this field *is* the module-level `REVERSE_BAIL_REASONS` below, not a fresh dict
    # per instance (see the outer `ADInterpreter{Reverse}` constructor). The `CodeInstance` cache is
    # keyed on the mode-level `cache_owner` (`Reverse()`, shared across every `ADInterpreter{Reverse}`
    # instance), so a later interpreter can hit an already-cached bailed carrier without ever
    # re-running `build_contextual_ir` itself — a fresh per-instance `bail_reasons` would then have
    # nothing in it, defeating the whole point of this field. Entries can go stale after invalidation;
    # that's fine, since they're only ever read when a build has in fact bailed, and a rebuild
    # overwrites the entry before anything reads it.
    bail_reasons::IdDict{MethodInstance, String}
    function ADInterpreter{M}(world::UInt,
                              ip::InferenceParams,
                              op::OptimizationParams,
                              bail_reasons::IdDict{MethodInstance, String}=IdDict{MethodInstance, String}()) where {M<:ADMode}
        @assert world <= Base.get_world_counter()
        return new{M}(
            InferenceResult[],
            world,
            ip,
            op,
            IdDict{CodeInstance, CodeInfo}(),
            IdDict{MethodInstance, IRCode}(),
            IdDict{MethodInstance, Vector{Any}}(),
            IdDict{MethodInstance, Nothing}(),
            bail_reasons,
        )
    end
end
function ADInterpreter{M}(;world=Base.get_world_counter(),
                          inf_params=InferenceParams(),
                          opt_params=OptimizationParams()) where {M<:ADMode}
    ADInterpreter{M}(world, inf_params, opt_params)
end

# Shared across every `ADInterpreter{Reverse}` instance — see the `bail_reasons` field comment above.
const REVERSE_BAIL_REASONS = IdDict{MethodInstance, String}()
function ADInterpreter{Reverse}(;world=Base.get_world_counter(),
                                inf_params=InferenceParams(),
                                opt_params=OptimizationParams())
    ADInterpreter{Reverse}(world, inf_params, opt_params, REVERSE_BAIL_REASONS)
end

Core.Compiler.InferenceParams(interp::ADInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::ADInterpreter) = interp.opt_params
Core.Compiler.get_inference_world(interp::ADInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::ADInterpreter) = interp.inf_cache
Core.Compiler.cache_owner(interp::ADInterpreter{M}) where {M} = M()
Core.Compiler.codegen_cache(interp::ADInterpreter) = interp.codegen_cache

# Build a lowered `CodeInfo` from an expression. Used by `frule_body` (`forward_interp.jl`) to
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
    ci = Base.generated_body_to_codeinfo(ex, @__MODULE__(), isva)
    @assert ci isa Core.CodeInfo "Failed to create a CodeInfo from the given expression. This might mean it contains a closure or comprehension?\n Offending expression: $e"
    ci
end


# ---------------------------------------------------------------------------
# Installing the transformed IR via `finishinfer!` and `optimize`.
#
# A mode's entry point (e.g. the generated `frule!!` fallback in `forward_interp.jl`) asks this
# interpreter to compile a *carrier* MethodInstance whose `specTypes` encodes the transformed
# signature. We compile it by transforming the corresponding primal method's post-optimization
# `IRCode` into a new `IRCode` and splicing that transform into the pipeline at two points:
#
#   * `finishinfer!` — the pipeline function that *supplies* the `CodeInstance` return type. We
#     build the transformed IR here and set `me.bestguess` to its return type, so the ordinary
#     `jl_fill_codeinst` writes the correct type once (no post-hoc patching).
#   * `optimize` — installs the already-built IR as the optimization result via the ordinary
#     `ipo_dataflow_analysis!` + `finishopt!`.
#
# Modes plug in by overriding `build_contextual_ir`; the default builds nothing, so any MI a mode
# doesn't handle (and any not-yet-implemented mode) flows through the ordinary pipeline unchanged.
# When a mode's transform hits a construct it can't handle (control flow, unsupported builtins, …)
# it returns `nothing`: we leave `me.bestguess`/the cache untouched, the carrier stub's throwing
# body flows through the pipeline normally, and the compiled CI raises when invoked — a graceful
# bail.
# ---------------------------------------------------------------------------

# Mode-specific hook: build the transformed `IRCode` for a carrier MethodInstance, or `nothing` to
# leave `mi` to the ordinary pipeline. Overridden per mode (see `forward_interp.jl`).
build_contextual_ir(::ADInterpreter, ::MethodInstance) = nothing

# Builds the transformed IR and sets `me.bestguess` to its return type so the generic
# `finishinfer!` freezes the correct type into the CodeInstance.
function CC.finishinfer!(me::CC.InferenceState, interp::ADInterpreter, cycleid::Int,
                         opt_cache::IdDict{MethodInstance, CodeInstance})
    ir = build_contextual_ir(interp, me.linfo)
    if ir !== nothing
        # `build_contextual_ir` builds `ir` via the 6-arg `CC.IRCode(...)` constructor, which
        # defaults `valid_worlds` to the unbounded sentinel `WorldRange(0, typemax(UInt))`. That
        # sentinel fails `abstract_eval_globalref_type`'s partition-coverage check
        # (`Compiler/src/abstractinterpretation.jl`) for any `GlobalRef` whose binding partition
        # doesn't itself span the full sentinel range — e.g. `Base.add_float` (an *imported*
        # binding, pulled in when inlining primal library code like `increment!!`), whose partition
        # starts at a finite world — so `argextype`/the IR pretty-printer mislabels those calls
        # "dynamic" even though they're ordinary builtin/intrinsic calls. Fix by giving `ir` a real
        # world range: `interp.world` onward, since every binding this IR references was already
        # resolved successfully at that world.
        ir = CC.IRCode(ir.stmts, ir.cfg, ir.debuginfo, ir.argtypes, ir.meta, ir.sptypes,
                       CC.WorldRange(interp.world, typemax(UInt)))
        interp.transformed_ir[me.linfo] = ir
        me.bestguess = CC.compute_ir_rettype(ir)
        # Fold in whatever backedges the mode's transform discovered (e.g. the primal method(s) a
        # dualized body was built from, and any `frule!!` methods resolved for surviving high-level
        # calls — see `build_contextual_ir`/`build_dual_ir` in `forward_interp.jl`). These must land
        # on *this* CodeInstance (`me.linfo`'s own compile), not merely on some caller of it: a
        # `finish!` on THIS `InferenceState` is what actually calls `store_backedges` against
        # `me.linfo`'s CodeInstance below, and a `@generated` caller's own cached CodeInstance for a
        # call to `me.linfo` is only re-examined for staleness by looking `me.linfo`'s cache up
        # again — so if `me.linfo`'s CodeInstance were never itself invalidated, a caller
        # regenerating around it would just find (and keep reusing) the same stale result. Setting
        # `me.src.edges` here — *before* delegating to the generic `finishinfer!`, which internally
        # calls `compute_edges!(me)` reading exactly this field — is the standard, documented way a
        # generator-produced `CodeInfo` declares extra edges (`compute_edges!` in `typeinfer.jl`).
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
# construction / rule calls are inlined and immutable results scalar-replaced.
function CC.optimize(interp::ADInterpreter, opt::CC.OptimizationState,
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
