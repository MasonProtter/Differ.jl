module DifferForwards

import ADTypes
using LinearAlgebra
using Random: AbstractRNG

using Contextual: Contextual, ContextualInterpreter, expr_to_codeinfo, run_ipo_passes!,
    at_world, mt_edge!
import Contextual: build_contextual_ir

import DifferCore: DifferCore, NoTangent, NoFData, NoRData, FData, RData, Tangent, MutableTangent,
    PossiblyUninitTangent, tangent_type, fdata_type, rdata_type,
    zero_tangent, zero_rdata, randn_tangent, increment!!, set_to_zero!!,
    build_tangent, get_tangent_field, set_tangent_field!,
    as_tangent, unit_tangent, LazyZeroRData, fdata, rdata, tangent, primal,
    _typeof, _findall, IEEEFloat, always_initialised, _new_, _copy,
    require_tangent_cache, MaybeCache, NoCache, fields_type, zero_tangent_internal,
    uninit_tangent,
    _globalref_val, _globalref_isconst, _calleeval, _optype, _optype_w, _stmt_str,
    _bi_literal_index, _bi_homog_tangent_type, _tangent_field_slot, _widen,
    _getfieldg, _setfieldg, _ctupleg

using Core: MethodInstance, CodeInstance, CodeInfo, Compiler
const CC = Core.Compiler
using Base: specialize_method

# `Forward` is the plugin owner type identifying DifferForwards to `Contextual.jl`'s
# `ContextualInterpreter{T,S}` — a plain immutable singleton, no per-session state needed
# (custom_state is always `nothing`).
struct Forward end

include("dual.jl")
include("intrinsics.jl")
include("builtins.jl")
include("foreigncalls.jl")
include("frules.jl")
include("forward_interp.jl")
include("rules_math.jl")
include("rules_reductions.jl")
include("rules_broadcast.jl")
include("rules_indexing.jl")
include("rules_linalg.jl")
include("reflection.jl")
include("adtypes.jl")

export Dual, primal, tangent, NoTangent, frule!!
export code_dual_ircode, @code_dual_ircode
export tangent_type, fdata_type, rdata_type
export Tangent, MutableTangent, PossiblyUninitTangent
export NoFData, NoRData, FData, RData
export fdata, rdata, zero_tangent
export as_tangent, unit_tangent
export AutoDifferForwards

end # module DifferForwards
