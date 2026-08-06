using Test
using Differ
using Differ: Dual, NoTangent, frule!!, zero_tangent, code_dual_ircode

include(joinpath(@__DIR__, "testutils.jl"))

# Forward mode over `Expr(:foreigncall)` (`src/foreigncalls.jl` and the `:foreigncall` arms of
# `dualize_to_ircode`), plus the `Expr(:loopinfo)` marker that arrives with it.
#
# Only bulk memory copies (`memmove`/`memcpy`) have a rule. That is not a placeholder: a copy is
# linear and structure-preserving, so performing the identical copy between the two *shadow* buffers
# is exactly the tangent of performing it between the primals. Native code in general can write
# through any pointer it is handed, so an unregistered target has no safe default: "primal plus a
# zero tangent", the treatment a non-differentiable intrinsic gets, would leave a destination's
# tangent stale rather than zero. Unregistered targets bail, and the last testset pins one located
# reason per gate.
#
# The single most important property here is the *length* check. `Dual`'s constructor only checks
# `tangent_type(P) == T`, never that a caller's tangent array is as long as its primal, and a raw
# `memmove` has no bounds check of its own: an unguarded mirror segfaults on a short destination
# tangent and silently reads uninitialised heap on a short source. Julia's own `@boundscheck` guards
# in `unsafe_copyto!` do not cover this: they sit inside an `@inbounds` block and are elided under
# the default `--check-bounds=auto`. So the "short shadow" testset must be run under the default,
# which is why the suite is invoked as `julia --project=test test/runtests.jl` rather than through
# `Pkg.test()` (which passes `--check-bounds=yes` and would mask exactly this).

D(f, x, dx) = frule!!(Dual(f, zero_tangent(f)), Dual(x, dx)).dx
D2(f, a, da, b, db) = frule!!(Dual(f, zero_tangent(f)), Dual(a, da), Dual(b, db)).dx

# Must be top level: a struct whose tangent has a different stride than the primal (`NoTangent` for
# the `Int` field collapses 16 bytes to 8), so a byte count does not carry over to the shadow buffer.
struct FCStridePair
    a::Float64
    b::Int
end

@testset "copyto! / copy (the memmove that blocks broadcast)" begin
    f(x) = (y = similar(x); copyto!(y, x); y[1] * y[2])
    x, dx = [0.3, 0.7, 1.1, 2.0], [1.0, 0.0, 0.0, 0.0]
    @test D(f, x, dx) ≈ x[2]
    @test D(f, x, [0.0, 1.0, 0.0, 0.0]) ≈ x[1]
    checkverify(f, (Vector{Float64},))

    g(x) = sum(abs2, copy(x))
    @test D(g, x, dx) ≈ 2x[1]

    # Destination supplied by the caller, so both shadow buffers are the caller's, not freshly
    # allocated ones. This is the shape the extent checks below actually protect.
    cp!(y, x) = (copyto!(y, x); y[1] + 2y[3])
    y, dy = zeros(4), zeros(4)
    @test D2(cp!, y, dy, x, dx) ≈ 1.0
    @test y == x                                     # the primal copy landed
    @test dy == dx                                   # ...and its tangent landed in the shadow buffer
end

@testset "elements with no tangent: the copy is primal-only" begin
    # `copy(::Vector{Int})` and friends. The shadow of a `Vector{Int}` is a `Memory{NoTangent}` with
    # no addressable storage, so the pass hands out `NULL_SHADOW_PTR` for its data pointer and this
    # rule emits the primal `memmove` alone: copying nothing is exactly the tangent of copying
    # non-differentiable data. (Before this, the whole path bailed one statement earlier, in the
    # `MemoryRef` data-pointer layout gate.)
    ints(v::Vector{Int}) = copy(v)
    v = [1, 2, 3]
    r = frule!!(Dual(ints, NoTangent()), Dual(v, zero_tangent(v)))
    @test r.x == v && r.x !== v                       # a real copy
    @test r.dx isa Vector{NoTangent} && length(r.dx) == 3
    checkverify(ints, (Vector{Int},))

    # The copied values still drive a differentiable result, and the `Int` array contributes no
    # tangent of its own.
    scale(v::Vector{Int}, x::Float64) = copy(v)[2] * x
    @test D2(scale, v, zero_tangent(v), 2.0, 1.0) ≈ 2.0

    @test D(w -> copy(w)[1] * 1.0, [true, false], zero_tangent([true, false])) ≈ 0.0
    checkverify(w -> copy(w), (Vector{Bool},))
    checkverify(n -> collect(1:n), (Int,))

    # No shadow copy and, unlike the mirrored case above, no extent guards either: a
    # `Memory{NoTangent}` holds nothing that could be overrun.
    ir, _ = code_dual_ircode(ints, (Vector{Int},))
    stmts = ir.stmts.stmt
    @test count(s -> s isa Expr && s.head === :foreigncall, stmts) == 1
    @test !any(s -> s isa Expr && s.head === :invoke &&
                    occursin("_fc_check_extent", string(s.args[2])), stmts)

    # Re-dualizable: the null sentinel is a `Ptr` literal in the emitted IR, and `zero_tangent(::Ptr)`
    # throws by design: `const_tangent` has to recognise it (a null shadow's shadow is again null).
    checkverify2(scale, (Vector{Int}, Float64); order=2)
end

@testset "emitted IR: the copy is mirrored, the guards are static invokes" begin
    f(x) = (y = similar(x); copyto!(y, x); y[1])
    ir, _ = code_dual_ircode(f, (Vector{Float64},))
    Core.Compiler.verify_ir(ir)

    stmts = ir.stmts.stmt
    # Exactly two: the primal memmove and its shadow mirror.
    @test count(s -> s isa Expr && s.head === :foreigncall, stmts) == 2

    # One extent check per shadow buffer, and each is a static `:invoke` to a resolved
    # `CodeInstance`. A `CallInfo`-less `Expr(:call)` would survive as a runtime dynamic dispatch
    # (see the perf gotcha in the dualization skill), which on a bulk-copy path is exactly what we
    # do not want to pay.
    checks = filter(s -> s isa Expr && s.head === :invoke &&
                         occursin("_fc_check_extent", string(s.args[2])), stmts)
    @test length(checks) == 2
end

@testset "sin.(x) — broadcast end to end" begin
    # The motivating case: `memmove` plus the `:loopinfo` marker in the same function.
    #
    # This validates *dualization* of the memmove, not its run-time behaviour: blocks 6-13 of the
    # primal are broadcast's `mightalias` check against a freshly allocated destination, so the
    # memmove block is never entered at run time. The copyto! testset above is what exercises the
    # shadow copy for real.
    x, dx = [0.3, 0.7, 1.1, 2.0], [1.0, 0.5, 0.0, 0.0]

    f(x) = sin.(x)
    @test D(f, x, dx) ≈ cos.(x) .* dx
    checkverify(f, (Vector{Float64},))

    g(x) = exp.(x)
    @test D(g, x, dx) ≈ exp.(x) .* dx

    # Array-with-scalar forms. These need the `Core.tuple` rule to widen a `Core.PartialStruct`
    # before `fieldtype` (`src/builtins.jl`): inference partly pins down `Broadcasted`'s argument
    # tuple here, which it does not for the single-array case above.
    @test D(x -> x .+ 1.0, x, dx) ≈ dx
    @test D(x -> x .* 2.0, x, dx) ≈ 2 .* dx
    @test D(x -> 2.0 .* x, x, dx) ≈ 2 .* dx

    # Two-array broadcast, which needed the ISSUES #60 fix on top of this file's memmove support:
    # `Base.broadcasted` builds `%new(Broadcasted{…}, …, Base.Broadcast.nothing)`, whose raw
    # `GlobalRef` operand `verify_ir` rejected until the `:new` arm learned to resolve it.
    y, dy = [2.0, 3.0, 5.0, 7.0], [0.0, 1.0, 0.0, 0.0]
    @test D2((a, b) -> a .* b, x, dx, y, dy) ≈ dx .* y .+ x .* dy
    @test D2((a, b) -> a .+ b, x, dx, y, dy) ≈ dx .+ dy
    @test D2((a, b) -> a .* b .+ 2.0 .* a, x, dx, y, dy) ≈ dx .* y .+ x .* dy .+ 2 .* dx
    checkverify((a, b) -> a .* b, (Vector{Float64}, Vector{Float64}))
end

@testset "a short shadow buffer raises BoundsError, not memory corruption" begin
    # Both directions. Without the emitted extent checks the destination case segfaults and the
    # source case returns garbage read out of uninitialised heap, neither of which a test can catch,
    # which is why these are asserted rather than left to the pointer rules' own gates.
    cp!(y, x) = (copyto!(y, x); y[1])
    x, dx = [1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 0.0]

    @test_throws BoundsError D2(cp!, zeros(4), zeros(1), x, dx)      # short destination tangent
    @test_throws BoundsError D2(cp!, zeros(4), zeros(4), x, [1.0])   # short source tangent

    # A correctly sized pair still works, i.e. the guard is not just rejecting everything.
    @test D2(cp!, zeros(4), zeros(4), x, dx) ≈ 1.0
end

@testset ":loopinfo (@simd) is carried through" begin
    # A hand-written `@simd` loop rather than a Base reduction: `sum`/`mapreduce` have hand-written
    # `frule!!`s that intercept before the generic path, so they would not exercise this at all. Run
    # past 1024 elements, the size at which the generic path used to bail on `:loopinfo`.
    function simdsum(x)
        s = 0.0
        @simd for i in eachindex(x)
            s += x[i]
        end
        return s
    end
    n = 2000
    x = collect(range(0.1, 2.0; length=n))
    dx = zeros(n); dx[1] = 1.0; dx[7] = 3.0
    @test D(simdsum, x, dx) ≈ sum(dx)
    checkverify(simdsum, (Vector{Float64},))
end

@testset "a ccall on a throw-only path does not block dualization" begin
    # An unreachable (throw-terminated) block is reconstructed primal-only, so a foreigncall there
    # never needs a rule: the derivative just has to raise the same error on the same inputs.
    function f(x)
        if x < 0.0
            ccall(:abort, Cvoid, ())
            error("negative")
        end
        return x * x
    end
    @test frule!!(Dual(f, NoTangent()), Dual(3.0, 1.0)) === Dual(9.0, 6.0)
    checkverify(f, (Float64,))
end

@testset "second order (the dualized memmove is re-dualizable)" begin
    f(x) = (y = similar(x); copyto!(y, x); y[1] * y[1])
    x = [2.0, 3.0]
    fseed2(g) = Dual(Dual(g, NoTangent()), Dual(g, NoTangent()))   # uniformly nested constant fn
    seed2 = Dual(Dual(x, [1.0, 0.0]), Dual([1.0, 0.0], [0.0, 0.0]))
    r = frule!!(fseed2(f), seed2)
    @test Differ.primal(Differ.primal(r)) ≈ 4.0     # f(x)     = x[1]^2
    @test Differ.tangent(Differ.primal(r)) ≈ 4.0    # f'(x)    = 2x[1]
    @test Differ.tangent(Differ.tangent(r)) ≈ 2.0   # f''(x)   = 2
    checkverify2(f, (Vector{Float64},); order=2)
end

# Top level so the pointers are genuine function arguments (an `Argument`, which the provenance walk
# cannot trace back to a buffer this pass shadows).
mvptr!(dst::Ptr{Float64}, src::Ptr{Float64}, n::Int) =
    ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, src, n * sizeof(Float64))

# A bulk copy whose two buffers are in different tangent regimes: the destination has shadow storage,
# the source (a `Vector{Int}`) has none. Not reachable through `copyto!`, which converts elementwise.
mixedcopy!(dst::Vector{Float64}, src::Vector{Int}) = begin
    GC.@preserve dst src ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                               pointer(dst), pointer(src), length(src) * sizeof(Int))
    dst[1]
end

@testset "graceful bails (located reason, no miscompile)" begin
    # No rule for the target. The message must name the target and point at where to add one.
    unreg(x) = (ccall(:getpid, Cint, ()); x * x)
    r = bail_reason(unreg, (Float64,))
    @test r !== nothing
    @test occursin("no dualization rule for `foreigncall` target `getpid`", r)
    @test occursin("src/foreigncalls.jl", r)

    # A runtime function pointer names nothing that can be dispatched on.
    fnptr(p::Ptr{Cvoid}, x) = (ccall(p, Cvoid, ()); x * x)
    r = bail_reason(fnptr, (Ptr{Cvoid}, Float64))
    @test r !== nothing
    @test occursin("non-literal target", r)

    # A `memmove` whose operands are not traceable to a `Memory`/`MemoryRef` this pass shadows: the
    # rule cannot establish either the stride or the extent, so it declines rather than guessing.
    r = bail_reason(mvptr!, (Ptr{Float64}, Ptr{Float64}, Int))
    @test r !== nothing
    @test occursin("not traceable", r)

    # A byte count does not carry over when the tangent element has a different stride.
    strided(v::Vector{FCStridePair}) = copy(v)[1].a
    r = bail_reason(strided, (Vector{FCStridePair},))
    @test r !== nothing
    @test occursin("same stride", r)

    # A same-named foreigncall with a different signature would have its operands mis-assigned, so
    # the whole signature is checked, not just the target symbol.
    badsig(v::Vector{Float64}, x) =
        (GC.@preserve v ccall(:memmove, Cvoid, (Ptr{Cvoid},), pointer(v)); x * x)
    r = bail_reason(badsig, (Vector{Float64}, Float64))
    @test r !== nothing
    @test occursin("unrecognised signature", r)

    # A copy between buffers in different tangent regimes, one side has shadow storage and the
    # other does not, is a reinterpreting copy whose tangent this rule cannot express.
    r = bail_reason(mixedcopy!, (Vector{Float64}, Vector{Int}))
    @test r !== nothing
    @test occursin("different tangent regimes", r)

    # Every one of these is a *reason*, not the misleading "no rule registered" fallback.
    for (f, at) in ((fnptr, (Ptr{Cvoid}, Float64)), (mvptr!, (Ptr{Float64}, Ptr{Float64}, Int)),
                    (strided, (Vector{FCStridePair},)), (mixedcopy!, (Vector{Float64}, Vector{Int})))
        @test !occursin("no rule registered", bail_reason(f, at))
    end
end
