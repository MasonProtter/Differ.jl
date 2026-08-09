# Exercises ContextualInterpreter end-to-end with a trivial toy plugin (no AD machinery
# involved) to prove the generic finishinfer!/optimize wiring works, and that the owner/
# custom_state split gives the cache_owner correctness the design is for.

using Test
using Contextual
using Contextual: ContextualInterpreter, build_contextual_ir, expr_to_codeinfo

const CC = Core.Compiler

# --- Toy plugin: "Double" transforms `doubled_impl(x)` into IR computing `x * 2.0`. ---

struct Double end

doubled_impl(x) = error("Contextual test: doubled_impl not compiled through the plugin")

# Second toy owner type for the cache_owner egal test below (must be top-level: `struct`
# cannot be defined inside a `@testset` block).
struct Config
    tag::Int
end

function build_double_ir(mi::Core.MethodInstance)
    stream = CC.InstructionStream(2)
    stream.stmt[1] = Expr(:call, GlobalRef(Base, :*), Core.Argument(2), 2.0)
    stream.type[1] = Float64; stream.flag[1] = CC.IR_FLAG_NULL
    stream.stmt[2] = Core.ReturnNode(Core.SSAValue(1))
    stream.type[2] = Float64; stream.flag[2] = CC.IR_FLAG_NULL
    cfg = CC.CFG(CC.BasicBlock[CC.BasicBlock(CC.StmtRange(1, 2), Int[], Int[])], Int[3])
    di = CC.DebugInfoStream(stream.line)
    di.def = mi
    argtypes = Any[typeof(doubled_impl), Float64]
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[])
    return ir
end

function Contextual.build_contextual_ir(interp::ContextualInterpreter{Double}, mi::Core.MethodInstance)
    isa(mi.def, Method) || return nothing
    isa(mi.specTypes, DataType) || return nothing
    params = mi.specTypes.parameters
    (!isempty(params) && params[1] === typeof(doubled_impl)) || return nothing
    return build_double_ir(mi)
end

# `@generated`-style entry point mirroring Differ's own `frule_body` pattern: resolve the
# carrier MethodInstance, compile it through the plugin interpreter via
# `typeinf_ext_toplevel`, and return a trivial body that `invoke`s the result.
function call_doubled_body(world::UInt, source, self, xtype)
    argnames = Any[Symbol("#self#"), :x]
    impl_tt = Tuple{typeof(doubled_impl), xtype}
    interp = ContextualInterpreter(Double(), nothing; world)
    match, _ = CC.findsup(impl_tt, CC.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (), :(error("no match")), false)
    end
    impl_mi = Base.specialize_method(match.method, match.spec_types, match.sparams)::Core.MethodInstance
    cinst = CC.typeinf_ext_toplevel(interp, impl_mi, CC.SOURCE_MODE_ABI)
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(doubled_impl, $cinst, x)), false)
    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

@eval function call_doubled(x)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, call_doubled_body))
end

@testset "Contextual.jl" begin
    @testset "end-to-end plugin transform" begin
        @test call_doubled(3.0) == 6.0
        @test call_doubled(-1.5) == -3.0
    end

    @testset "verify_ir on the transformed carrier" begin
        # Build the transform directly (not through typeinf_ext_toplevel, which may hit an
        # already-cached CodeInstance from the "end-to-end" testset above since cache_owner is
        # shared across every `Double()` interpreter) and verify its IR is well-formed.
        tt = Tuple{typeof(doubled_impl), Float64}
        interp = ContextualInterpreter(Double(), nothing)
        mi = Base.specialize_method(CC.findall(tt, CC.method_table(interp))[1])
        ir = build_double_ir(mi)
        @test CC.verify_ir(ir) === nothing
    end

    @testset "owner/custom_state split: cache_owner is keyed on owner only" begin
        # Two independently-constructed but egal-equal immutable owners, paired with distinct
        # (non-egal) mutable custom_state, must still share one cache_owner partition — the
        # entire point of splitting owner from custom_state.
        owner_a = Config(1)
        owner_b = Config(1)
        @test owner_a === owner_b   # isbits immutable structs are egal by value

        state_a = Dict{Symbol,Int}()
        state_b = Dict{Symbol,Int}()
        @test state_a !== state_b

        interp_a = ContextualInterpreter(owner_a, state_a)
        interp_b = ContextualInterpreter(owner_b, state_b)
        @test CC.cache_owner(interp_a) === CC.cache_owner(interp_b)
        @test interp_a.custom_state !== interp_b.custom_state

        # A different owner value gets a distinct partition.
        interp_c = ContextualInterpreter(Config(2), state_a)
        @test CC.cache_owner(interp_a) !== CC.cache_owner(interp_c)
    end
end
