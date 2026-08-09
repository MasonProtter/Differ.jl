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

# --- The world-age pin `at_world` exists to work around. ---
#
# `jl_call_staged` pins a generator body's task world age to the generated method's
# `Method.primary_world`, so a method defined *after* that entry point is invisible to plain
# dispatch from inside it — which is what silently broke cross-package pass composition
# (`tangent_type` overrides and coupling hooks owned by a later-loaded package resolving to their
# generic/inert fallbacks; ISSUES #85). Two escape hatches look like they should work and do not:
# `Base.invoke_in_world` is a no-op while `in_pure_callback` is set, and `invoke` with a
# `CodeInstance` inferred at a newer world throws. `Core._call_in_world_total` — what `at_world`
# wraps — is the one that works, and the world it switches to covers nested dispatch inside the
# callee too (load-bearing: `tangent_type(Stack{T})` recurses into `tangent_type(T)`).
#
# This whole design rests on that asymmetry, so assert it directly rather than only observing its
# downstream effects. If a future Julia changes any of it, this fails here instead of surfacing as
# a hang somewhere in an AD transform.

pinned_probe(::Int) = :before
nested_probe() = pinned_probe(1)

function pin_body(world::UInt, source, self, x)
    results = (plain          = pinned_probe(1),
               invoke_in_world = try Base.invoke_in_world(world, pinned_probe, 1) catch; :threw end,
               at_world        = at_world(world, pinned_probe, 1),
               at_world_nested = at_world(world, nested_probe),
               pinned_age      = Base.tls_world_age(),
               world_arg       = world)
    return expr_to_codeinfo(@__MODULE__, Any[Symbol("#self#"), :x], [], (),
                            :(return $results), false)
end

@eval function pin_entry(x)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, pin_body))
end

const PIN_ENTRY_WORLD = which(pin_entry, Tuple{Int}).primary_world

# Redefined strictly after `pin_entry`, so it is invisible to that generator's pinned world.
pinned_probe(::Int) = :after

@testset "generator world-age pin / at_world" begin
    r = pin_entry(1)

    # The pin itself: the generator ran long after the redefinition, and was still handed the
    # current world as its `world` argument, yet its task world age is the entry's primary_world.
    @test r.pinned_age == PIN_ENTRY_WORLD
    @test r.world_arg > PIN_ENTRY_WORLD

    # Plain dispatch and `invoke_in_world` both see the stale method...
    @test r.plain === :before
    @test r.invoke_in_world === :before

    # ...while `at_world` sees the current one, including through a nested dispatch inside the
    # callee (`nested_probe` calls `pinned_probe` dynamically).
    @test r.at_world === :after
    @test r.at_world_nested === :after
end

@testset "mt_edge! dedupes" begin
    edges = Any[]
    mt_edge!(edges, Tuple{typeof(pinned_probe),Int})
    mt_edge!(edges, Tuple{typeof(pinned_probe),Int})
    mt_edge!(edges, Tuple{typeof(nested_probe)})
    @test length(edges) == 4      # two (sig, methodtable) pairs, the duplicate dropped
    @test edges[2] === Core.methodtable
    @test edges[4] === Core.methodtable
end

@testset "ContextualInterpreter rejects the pure-callback world sentinel" begin
    # `Base.get_world_counter()` returns `typemax(UInt)` inside a generator, so defaulting `world`
    # from it there would silently build an interpreter at that sentinel — and the obvious
    # `world <= get_world_counter()` assert cannot catch it, being vacuously true in exactly that
    # context. Rejected explicitly instead.
    @test_throws AssertionError ContextualInterpreter(Double(), nothing; world=typemax(UInt))
end
