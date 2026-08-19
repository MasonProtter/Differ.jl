using Test
using DifferReverse
using DifferReverse: NoTangent, rev_gradient, zero_tangent
# `Dual`/`frule!!`/`primal`/`tangent` here are DifferForwards' forward-mode carrier, used purely
# as an independent numerical oracle (not testing forward/reverse composition).
using DifferForwards: Dual, frule!!, primal, tangent

@testset "boxed captured variable (reassigned closure variable)" begin
    # Reassigning a captured variable inside a closure forces Julia to box it (`Core.Box`, with
    # an `Any`-typed `.contents` field), which lowers each read behind a `Core.isdefined` guard
    # and a `throw_undef_if_not` marker. This used to crash reverse mode with a `MethodError` from
    # deep inside `set_to_zero_internal!!` (no method for `FData`/`RData`); it now bails cleanly
    # with a located `ErrorException` instead (reverse mode's own, separate limitation on
    # dynamic dispatch through an `Any`-typed box).
    #
    # `y += x` lowers to a dynamic `%box_contents + x` call (both operands typed `Any`/`Float64`,
    # not concrete enough to resolve statically), which is what actually bails.
    err = try
        let y = 1.0
            rev_gradient(1.0) do x
                y += x
                x * y
            end
        end
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test !(err isa MethodError)
    @test occursin("non-concrete argument type", err.msg)
    @test occursin("at %", err.msg)

    # Forward mode fully supports this case (Phase 1): `Core.isdefined` is a registered builtin
    # frule, and `throw_undef_if_not` is a pure control marker dualized on both the live-path and
    # throw-only-block code paths. y0=1.0, x=1.0: primal = x*(y0+x) = 1*2 = 2;
    # tangent = d/dx (x*(y0+x)) = y0 + 2x = 1 + 2 = 3.
    let y = 1.0
        f = x -> (y += x; x * y)
        d = frule!!(Dual(f, zero_tangent(f)), Dual(1.0, 1.0))
        @test primal(d) ≈ 2.0
        @test tangent(d) ≈ 3.0
    end

    # A second, independent call must not observe state left over from the first (each call
    # starts from `y = 1.0` again via a fresh `let`). Not a load-bearing aliasing check, just
    # confirming the boxed capture itself is per-closure-instance.
    let y = 10.0
        f = x -> (y += x; x * y)
        d = frule!!(Dual(f, zero_tangent(f)), Dual(2.0, 1.0))
        @test primal(d) ≈ 24.0          # (10+2)*2
        @test tangent(d) ≈ 14.0         # y0 + 2x = 10 + 4
    end
end

# Reverse mode replays a global read as a constant (Mooncake-consistent). Sound for a genuinely
# separate object; silently wrong if the caller differentiates w.r.t. the very object the global
# aliases — these guard against that specific misuse.
const _alias_global = [2.0]
_alias_f(x) = _alias_global[1] * x[1] + x[1]

@testset "runtime aliasing guard: mutable global vs. differentiated argument" begin
    err = try
        rev_gradient(_alias_f, _alias_global)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("aliases the module global", err.msg)
    @test occursin("_alias_global", err.msg)

    # A fresh, non-aliasing array still differentiates correctly — the global genuinely is
    # constant here. d/dx (G[1]*x[1] + x[1]) = G[1] + 1 = 3.
    grads = rev_gradient(_alias_f, [2.0])
    @test grads[2] ≈ [3.0]

    # Prealloc path hits the same guard: it lives in the fwds carrier, run every call, not in the
    # allocating entry point.
    ctx = build_ctx(_alias_f, (Vector{Float64},))
    fcd = zero_fcodual(_alias_f)
    err2 = try
        value_and_gradient!(ctx, fcd, CoDual(_alias_global, zero(_alias_global)))
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("aliases the module global", err2.msg)

    y, grads2 = value_and_gradient!(ctx, fcd, CoDual([2.0], [0.0]))
    @test y ≈ 6.0
    @test grads2[2] ≈ [3.0]
end

const _alias_bits_global = 3.0
_alias_h(x) = _alias_bits_global * x[1]

@testset "runtime aliasing guard: bits-typed global needs none" begin
    ir, _ = code_reverse_fwds_ircode(_alias_h, (Vector{Float64},))
    @test !any(ir.stmts.stmt) do s
        isa(s, Expr) && s.head === :invoke && length(s.args) >= 2 &&
            s.args[2] === DifferReverse._rr_check_global_alias
    end
    @test rev_gradient(_alias_h, [5.0])[2] ≈ [3.0]
end
