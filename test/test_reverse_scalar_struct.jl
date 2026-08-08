using Test
using Differ
using Differ: Dual, NoTangent, frule!!, rev_gradient

include(joinpath(@__DIR__, "testutils.jl"))

@testset "reverse mode: scalar intrinsics" begin
    # Scalar float intrinsics only (add_float/mul_float/div_float), checked two ways: finite
    # differences, and a cross-check against the already-trusted forward-mode `frule!!` (one seed
    # direction per argument), independent verification of the new reverse engine.
    rprod(x, y) = x*y + x                             # ∂/∂x = y+1, ∂/∂y = x
    rquot(x, y) = (x*y + x) / y                        # mul/add/div composed

    for f in (rprod, rquot)
        x, y = 2.0, 3.0
        _, dx, dy = rev_gradient(f, x, y)
        @test dx ≈ central_diff(f, x, y, 1) rtol = 1e-5
        @test dy ≈ central_diff(f, x, y, 2) rtol = 1e-5
        @test dx ≈ frule!!(Dual(f, NoTangent()), Dual(x, 1.0), Dual(y, 0.0)).dx
        @test dy ≈ frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 1.0)).dx
    end

    checkverify_rev(rprod, (Float64, Float64))
    checkverify_rev(rquot, (Float64, Float64))
    check_stack_balance(rprod, 2.0, 3.0)   # straight-line: no block-stack push at all
end

@testset "reverse mode: immutable struct (%new/getfield, RData)" begin
    # Immutable struct via `%new`/`getfield`, exercising `RData`/`increment_field!!`.
    # rstruct(a,b) = a*b + a  =>  ∂/∂a = b+1, ∂/∂b = a
    struct V2; a::Float64; b::Float64; end
    function rstruct(a, b)
        v = V2(a, b)
        v.a*v.b + v.a
    end

    _, da, db = rev_gradient(rstruct, 2.0, 3.0)
    @test da ≈ 3.0 + 1.0
    @test db ≈ 2.0

    checkverify_rev(rstruct, (Float64, Float64))
end

@testset "reverse mode: conversion intrinsics (sitofp/fpext/fptrunc)" begin
    # `sitofp` (Int->Float promotion) is the INACTIVE bucket: its result carries a real tangent but
    # its operands don't, so its pullback consumes the seed and contributes `NoRData`, d/dx
    # (x·(1+2+3)) = 6. `fpext`/`fptrunc` (Float32<->Float64) is the LINEAR bucket: genuinely
    # differentiable, d/dx (Float64(Float32(x)·2) + x) = 3. Before these rules existed, either
    # bailed with "no reverse rule for intrinsic `sitofp`/`fptrunc`". Cross-checked against forward
    # mode and finite differences.
    sitofp_ctl(x::Float64) = (s = 0.0; for i in 1:3; s += x*i; end; s)
    mix32_ctl(x::Float64) = Float64(Float32(x) * Float32(2.0)) + x

    _, dx_si = rev_gradient(sitofp_ctl, 2.0)
    @test dx_si == 6.0
    @test dx_si == frule!!(Dual(sitofp_ctl, NoTangent()), Dual(2.0, 1.0)).dx
    _, dx_mx = rev_gradient(mix32_ctl, 1.0)
    @test dx_mx == 3.0
    @test dx_mx == frule!!(Dual(mix32_ctl, NoTangent()), Dual(1.0, 1.0)).dx
    @test dx_si ≈ central_diff(sitofp_ctl, 2.0; h=1e-5) rtol = 1e-5
    @test dx_mx ≈ central_diff(mix32_ctl, 1.0; h=1e-5) rtol = 1e-2  # Float32 FD is noisy

    checkverify_rev(sitofp_ctl, (Float64,))
    checkverify_rev(mix32_ctl, (Float64,))
end
