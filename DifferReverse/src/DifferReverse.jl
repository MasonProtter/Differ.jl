module DifferReverse

import ADTypes
using LinearAlgebra
using Random: AbstractRNG

using Core: MethodInstance, CodeInstance, CodeInfo, Compiler
const CC = Core.Compiler
using Base: specialize_method

using Contextual: Contextual, ContextualInterpreter, expr_to_codeinfo, run_ipo_passes!,
    at_world, mt_edge!
import Contextual: build_contextual_ir

# Any name DifferReverse adds new methods to (not just calls) must come in via `import`, not bare
# `using DifferCore`/`using DifferCore: name` — otherwise it either errors ("must be explicitly
# imported to be extended") or silently creates a disconnected same-named local function.
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
    _getfieldg, _setfieldg, _ctupleg,
    _fc_parse, _fc_stmt, _fc_ptr_origin, _fc_same_stride, _fc_check_extent,
    _fc_copy_sig_ok, _FC_COPY_ATS

# `Reverse` is the plugin owner type identifying DifferReverse to `Contextual`'s
# `ContextualInterpreter{T,S}`. `owner` doubles as the `cache_owner` partition key, so
# `nested_forward` must be a real field (not a fieldless singleton) — a build compiling on behalf
# of an outer forward-over-reverse dualization needs a distinct partition from an ordinary build.
struct Reverse
    nested_forward::Bool
end
Reverse(; nested_forward::Bool=false) = Reverse(nested_forward)

# Per-session bookkeeping kept out of `owner`/`cache_owner`'s reach (`Contextual`'s `custom_state`
# field). Shared across every ordinary (non-nested) `Reverse` build so a later interpreter can look
# up why a cached carrier bailed without re-running the transform.
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
include("foreigncalls_reverse.jl")
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
