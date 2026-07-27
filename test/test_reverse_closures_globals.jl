using Test
using Differ
using Differ: Dual, NoTangent, frule!!, gradient, primal, tangent, zero_tangent

@testset "boxed captured variable (reassigned closure variable)" begin
    # Reassigning a captured variable inside a closure forces Julia to box it (`Core.Box`, with
    # an `Any`-typed `.contents` field), which lowers each read behind a `Core.isdefined` guard
    # and a `throw_undef_if_not` marker. This used to crash reverse mode with a `MethodError`
    # from deep inside `set_to_zero_internal!!` (no method for `FData`/`RData`); it now bails
    # cleanly with a located `ErrorException` instead (reverse mode's own, separate limitation on
    # `setfield!` of a field whose tangent carries fdata — Phase 2 territory, out of scope here).
    err = try
        let y = 1.0
            gradient(1.0) do x
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
    @test occursin("setfield!", err.msg)
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
    # starts from `y = 1.0` again via a fresh `let`) — not a load-bearing aliasing check, just
    # confirming the boxed capture itself is per-closure-instance.
    let y = 10.0
        f = x -> (y += x; x * y)
        d = frule!!(Dual(f, zero_tangent(f)), Dual(2.0, 1.0))
        @test primal(d) ≈ 24.0          # (10+2)*2
        @test tangent(d) ≈ 14.0         # y0 + 2x = 10 + 4
    end
end
