# Phase A checkpoint (see the "reverse-mode control flow, Mooncake style" plan): round-trip a
# handful of real multi-block `IRCode`s through the `ID`/`CFGBlock` working-IR layer (`src/cfg_ir.jl`)
# with no AD involved at all. This isolates "did the IR-plumbing port correctly" from any
# reverse-mode AD concern before that layer gets its first real user (the pullback-pass builder).
#
# `lower_cfg_blocks_to_ir(_ircode_to_cfg_blocks(ir), ir)` should be the identity (mod object
# identity) — verified two ways per statement kind exercised (branches, loops, nested branches):
# `Core.Compiler.verify_ir` doesn't throw, and the round-tripped IR actually computes the same
# result as the original when run (via `Core.OpaqueClosure`, a convenient way to execute a bare
# `IRCode` directly — used here purely as test-harness plumbing, unrelated to the "no OpaqueClosure
# in the AD engine itself" design constraint the plan follows for reverse mode).

using Test
using Differ: _ircode_to_cfg_blocks, lower_cfg_blocks_to_ir

cfg_branch(x) = x > 0.0 ? x * 2.0 : -x
function cfg_loop(x, k)
    s = 0.0
    i = 0
    while i < k
        s = s + x
        i = i + 1
    end
    return s
end
function cfg_nested(x, y)
    if x > 0.0
        if y > 0.0
            return x + y
        else
            return x - y
        end
    else
        return -x
    end
end

@testset "CFGBlock/ID working-IR round-trip (Phase A, no AD)" begin
    roundtrip(ir) = lower_cfg_blocks_to_ir(_ircode_to_cfg_blocks(ir), ir)

    cases = ((cfg_branch, (Float64,), (2.0,)), (cfg_branch, (Float64,), (-3.0,)),
             (cfg_loop, (Float64, Int), (2.0, 5)), (cfg_loop, (Float64, Int), (3.0, 0)),
             (cfg_nested, (Float64, Float64), (1.0, 2.0)), (cfg_nested, (Float64, Float64), (1.0, -2.0)),
             (cfg_nested, (Float64, Float64), (-1.0, 5.0)))

    for (f, types, args) in cases
        ir = first(only(Base.code_ircode(f, types)))
        ir2 = roundtrip(ir)
        @test ir2 !== ir
        Core.Compiler.verify_ir(ir2)      # throws on any IR-legality violation

        ir2.argtypes[1] = Tuple{}         # `OpaqueClosure`'s captures-tuple slot, not `typeof(f)`
        oc = Core.OpaqueClosure(ir2)
        @test oc(args...) === f(args...)
    end
end
