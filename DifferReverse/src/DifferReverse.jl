module DifferReverse

import ADTypes
using LinearAlgebra
using Random: AbstractRNG

using Core: MethodInstance, CodeInstance, CodeInfo, Compiler
const CC = Core.Compiler
using Base: specialize_method

using Contextual: Contextual, ContextualInterpreter, expr_to_codeinfo, run_ipo_passes!
import Contextual: build_contextual_ir

# Every name DifferReverse adds new methods to (as opposed to merely calling) must come in via
# `import`, not a bare `using DifferCore`/`using DifferCore: name` — see the stage-3/4
# "module-boundary extension gotcha" note in the migration progress notes: a name brought in any
# other way either errors loudly ("must be explicitly imported to be extended") if it's also
# reachable some other way, or — worse — silently creates a brand-new, disconnected same-named
# local function with no error at all. `tangent`, `_copy`, `zero_tangent_internal`,
# `set_to_zero_internal!!`, `tangent_type` are all extended (with new methods for `Stack`/
# `SingletonStack`/`CommsCell`/`Tape`/`CoDual`/`NoPullback`) — found by cross-referencing every
# top-level function definition in this package's source against DifferCore's own, not by
# inspection alone.
import DifferCore: DifferCore, NoTangent, NoFData, NoRData, FData, RData, Tangent, MutableTangent,
    PossiblyUninitTangent, tangent_type, fdata_type, rdata_type,
    zero_tangent, zero_rdata, randn_tangent, increment!!, set_to_zero!!,
    build_tangent, get_tangent_field, set_tangent_field!,
    as_tangent, unit_tangent, LazyZeroRData, fdata, rdata, tangent, primal,
    _typeof, _findall, IEEEFloat, always_initialised, _new_, _copy,
    require_tangent_cache, MaybeCache, NoCache, fields_type, zero_tangent_internal,
    uninit_tangent, uninit_fdata, lazy_zero_rdata, instantiate, set_to_zero_internal!!,
    SetToZeroCache, tuple_map, @foldable, zero_like_rdata_type, zero_like_rdata_from_type,
    zero_rdata_from_type, _get_fdata_field, increment_field!!, increment_field_rdata!,
    increment_rdata!!, is_always_fully_initialised, split_union_tuple_type, ZeroRData,
    CannotProduceZeroRDataFromType,
    _globalref_val, _globalref_isconst, _calleeval, _optype, _optype_w, _stmt_str,
    _bi_literal_index, _bi_homog_tangent_type, _tangent_field_slot, _widen,
    _getfieldg, _setfieldg, _ctupleg

# `Reverse` is the plugin owner type identifying DifferReverse to `Contextual.jl`'s
# `ContextualInterpreter{T,S}`. Unlike `Forward`, it carries one bit of immutable config
# (`nested_forward`) — see the `Contextual.jl` API design: `owner` IS the `cache_owner`
# partition key directly, so a build compiling on behalf of an outer forward-mode dualization
# (forward-over-reverse) must get a genuinely distinct partition from an ordinary build, which a
# plain fieldless `Reverse()` singleton could not provide.
struct Reverse
    nested_forward::Bool
end
Reverse(; nested_forward::Bool=false) = Reverse(nested_forward)

# Mutable, per-session bookkeeping the framework doesn't manage itself — deliberately kept out of
# `owner`/`cache_owner`'s reach (see `Contextual.jl`'s `custom_state` field). Shared across every
# ordinary (non-nested) `Reverse` build so a later interpreter can hit an already-cached bailed
# carrier's reason without re-running the transform (see `custom_state.bail_reasons`'s docstring
# at its use site in `reverse_interp.jl`, ported from `contextual.jl`'s old `bail_reasons` field
# comment).
const REVERSE_BAIL_REASONS = IdDict{MethodInstance,String}()

function build_reverse_interp(; world::UInt=Base.get_world_counter(),
                              inf_params::CC.InferenceParams=CC.InferenceParams(),
                              opt_params::CC.OptimizationParams=CC.OptimizationParams(),
                              nested_forward::Bool=false)
    custom_state = (; in_progress=IdDict{MethodInstance,Nothing}(), bail_reasons=REVERSE_BAIL_REASONS)
    return ContextualInterpreter(Reverse(nested_forward), custom_state; world, inf_params, opt_params)
end

include("codual.jl")
include("stack.jl")
include("cfg_ir.jl")
include("intrinsics_reverse.jl")
include("builtins_reverse.jl")
include("reverse_interp.jl")
include("rrules.jl")
include("rules_math.jl")
include("rules_reductions.jl")
include("rules_broadcast.jl")
include("rules_indexing.jl")
include("rules_linalg.jl")
include("reflection.jl")

struct AutoDifferReverse <: ADTypes.AbstractADType end
ADTypes.mode(::AutoDifferReverse) = ADTypes.ReverseMode()

export CoDual, primal, tangent, NoTangent, rrule!!
export AbstractCtx, Ctx, build_ctx
export value_and_gradient!, zero_fcodual
export code_reverse_fwds_ircode, @code_reverse_fwds_ircode
export code_reverse_pullback_ircode, @code_reverse_pullback_ircode
export tape_type, comms_element_types
export tangent_type, fdata_type, rdata_type
export Tangent, MutableTangent, PossiblyUninitTangent
export NoFData, NoRData, FData, RData
export fdata, rdata, zero_tangent
export as_tangent, unit_tangent
export AutoDifferReverse

public rev_gradient, rev_gradient!

end # module DifferReverse
