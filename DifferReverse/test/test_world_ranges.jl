# Synthetic carrier `IRCode` must carry a real world range, not the `CC.IRCode` constructor's
# unbounded default and not the primal's own `valid_worlds`.
#
# `Core.Compiler.verify_ir` exempts `Core`/`Base` `GlobalRef`s but requires every other module's
# binding partition to cover `ir.valid_worlds`. DifferReverse's own bindings only exist from the
# package's load world onward, so a carrier inheriting a Base primal's `min_world` rejects the
# `GlobalRef`s the transform emits itself. `carrier_world_range` is what prevents that.

using Test
using DifferReverse
using DifferReverse: rrule!!, CoDual, AbstractCtx, Ctx, NoFData, NoRData,
                     code_reverse_fwds_ircode, code_reverse_pullback_ircode, rev_gradient,
                     carrier_world_range

const CC = Core.Compiler

# Load world of DifferReverse's own bindings — the lower bound a carrier's `min_world` must clear.
_binding_min_world(m::Module, name::Symbol) =
    Base.lookup_binding_partition(Base.get_world_counter(),
                                  convert(Core.Binding, GlobalRef(m, name))).min_world

@testset "carrier_world_range" begin
    interp = DifferReverse.build_reverse_interp()
    @test CC.min_world(carrier_world_range(interp)) != typemin(UInt)
    @test CC.max_world(carrier_world_range(interp)) == typemax(UInt)

    # Intersecting with a Base primal's range lifts `min_world` without widening `max_world`.
    pir, _ = only(Base.code_ircode(sum, (Vector{Float64},)))
    wr = carrier_world_range(interp, pir)
    @test CC.min_world(wr) == interp.world
    @test CC.max_world(wr) == CC.max_world(pir.valid_worlds)
end

@testset "Base-rooted primal: carrier clears the package load world" begin
    lo = _binding_min_world(DifferReverse, :_NO_BULK_BUFS)
    basey(x) = sum(abs2, [x, 2x, 3x])
    for ir in (code_reverse_fwds_ircode(basey, (Float64,))[1],
               code_reverse_pullback_ircode(basey, (Float64,))[1])
        @test CC.min_world(ir.valid_worlds) >= lo
        CC.verify_ir(ir)                      # throws on failure
    end
    @test rev_gradient(basey, 1.5)[2] ≈ 28 * 1.5      # basey is 14x²
end

# The folklore this replaced claimed a bare name in a rule body re-embeds as an unbound
# `GlobalRef(Differ, :sin)`. It does re-embed; it is not unbound, and it verifies.
@testset "DifferReverse-module GlobalRefs survive in carrier IR" begin
    loopsum(x) = (s = 0.0; for i in 1:5; s += sin(x*i); end; s)
    ir = code_reverse_pullback_ircode(loopsum, (Float64,))[1]
    mods = Set{Module}()
    for s in ir.stmts.stmt, op in (isa(s, Expr) ? s.args : ())
        isa(op, GlobalRef) && push!(mods, op.mod)
    end
    @test DifferReverse in mods
    CC.verify_ir(ir)
    @test rev_gradient(loopsum, 0.5)[2] ≈ sum(i*cos(0.5*i) for i in 1:5)
end

# A hand rule written with bare names differentiates correctly — the qualification discipline the
# rule files used to follow is not required.
bare_primal(x) = 2x
function DifferReverse.rrule!!(::CoDual{typeof(bare_primal),NoFData}, ::AbstractCtx,
                               (; x)::CoDual{Float64,NoFData})
    bare_pullback(dy) = (NoRData(), cos(x)*dy)
    CoDual(sin(x), NoFData()), bare_pullback
end

@testset "hand rule with bare names" begin
    f(x) = bare_primal(x) + bare_primal(2x)
    @test rev_gradient(f, 1.0)[2] ≈ cos(1.0) + 2cos(2.0)
    CC.verify_ir(code_reverse_fwds_ircode(f, (Float64,))[1])
end

# Hand rules inline into an ordinary carrier; under forward-over-reverse they must stay real calls.
_rrule_invokes(ir) = count(s -> isa(s, Expr) && s.head === :invoke &&
                                length(s.args) >= 2 && s.args[2] === rrule!!, ir.stmts.stmt)

@testset "hand-rule inlining is gated on nested_forward" begin
    g(x) = sin(x) + cos(2x)
    @test _rrule_invokes(code_reverse_fwds_ircode(g, (Float64,))[1]) == 0

    interp = DifferReverse.build_reverse_interp(; nested_forward=true)
    tt = Tuple{typeof(DifferReverse.reverse_fwds_impl),
               DifferReverse.fcodual_type(typeof(g)), Ctx{Nothing},
               DifferReverse.fcodual_type(Float64)}
    match, _ = CC.findsup(tt, CC.method_table(interp))
    mi = Base.specialize_method(match.method, match.spec_types, match.sparams)
    nested = DifferReverse.optimized_reverse_fwds_ir(interp, mi, Ref(""))
    @test _rrule_invokes(nested) >= 1
end

# A primal that itself calls `rrule!!` must stay detectable: that call never carries the transform's
# `IR_FLAG_INLINE` opt-in, so it survives for the composition scan to reject.
@testset "reverse-over-reverse still bails" begin
    @test_throws ErrorException rev_gradient(x -> rev_gradient(sin, x)[2], 1.0)
end
