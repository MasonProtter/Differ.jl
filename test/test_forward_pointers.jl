using Test
using Differ
using Differ: Dual, NoTangent, frule!!, zero_tangent, code_dual_ircode

include(joinpath(@__DIR__, "testutils.jl"))

# Forward mode over `GC.@preserve` and raw pointer traffic (`src/intrinsics.jl`'s pointer rules, the
# `MemoryRef`/`Memory` `getfield` branch in `src/builtins.jl`, and the `:gc_preserve_begin`/`_end` arms
# of `dualize_to_ircode`).
#
# The whole thing rests on `tangent_type(Ptr{P}) === Ptr{tangent_type(P)}`: a shadow pointer addresses
# *tangent* storage at the position its primal addresses, so each rule mirrors the operation onto it.
# Two dualization-specific properties are worth stating, because they're what these tests protect:
#
#  * `GC.@preserve` must root the **shadow** object as well as the primal — the dualized code holds an
#    interior pointer into both, so rooting only the primal would let the shadow array be collected
#    while a live shadow `Ptr` still points into it.
#  * A mirror is only valid when a position means the same thing on both sides. An element *index*
#    always does (`pointerref` scales by the shadow pointer's own element type); a *byte offset* only
#    does when the tangent element has the primal's stride, and a `MemoryRef`'s data-pointer field only
#    holds an address at all when the buffer stores its elements inline. Both are checked, and both
#    bail with a located reason rather than miscompiling — see the last testset.

D(f, x) = frule!!(Dual(f, zero_tangent(f)), Dual(x, one(x))).dx

@testset "GC.@preserve with no pointer traffic" begin
    # The construct on its own: carried through, and the region's contents dualize normally.
    f(x) = (v = [x, 2x]; GC.@preserve v sum(v))
    @test D(f, 1.0) == 3.0
    checkverify(f, (Float64,))
end

@testset "unsafe_store! through a pointer into a local array" begin
    # The original motivating case. `v == [3x, 2x]` after the store, so `sum(v) == 5x`.
    f(x) = begin
        v = [x, 2x]
        ptr = pointer(v)
        GC.@preserve v begin
            Base.unsafe_store!(ptr, 3x)
        end
        sum(v)
    end
    @test f(1.0) == 5.0
    @test D(f, 1.0) == 5.0
    @test D(f, 2.5) == 5.0
    checkverify(f, (Float64,))

    # The shadow array must be rooted by the *same* `gc_preserve_begin` as the primal, and the store
    # must be mirrored. (The IR does still contain one `frule!!` invoke — for `sum`, a hand-ruled
    # callee. That the *pointer* ops need no such round trip is asserted in the pointer-only testset
    # below, where nothing else survives to muddy the count.)
    ir, _ = code_dual_ircode(f, (Float64,))
    stmts = string.(ir.stmts.stmt)
    preserves = filter(s -> occursin("gc_preserve_begin", s), stmts)
    @test length(preserves) == 1
    @test length(findall("%", only(preserves))) == 2          # two rooted objects: primal and shadow
    @test count(s -> occursin("gc_preserve_end", s), stmts) == 1
    @test count(s -> occursin("pointerset", s), stmts) == 2
end

@testset "unsafe_load" begin
    # Element *index* mirroring: `unsafe_load(p, 2)` reads the second tangent, not a byte offset.
    f(x) = (v = [x, 2x]; Base.unsafe_load(pointer(v), 2) * 3)
    @test f(1.0) == 6.0
    @test D(f, 1.0) == 6.0
    checkverify(f, (Float64,))

    # Load-after-store round trip through the same pointer.
    g(x) = (v = [x, 2x]; p = pointer(v); GC.@preserve v (Base.unsafe_store!(p, 4x); Base.unsafe_load(p)))
    @test g(1.0) == 4.0
    @test D(g, 1.0) == 4.0
end

@testset "pointer arithmetic (`p + k`)" begin
    # `p + k` stays in `Ptr` space (`add_ptr(::Ptr{P}, ::UInt)::Ptr{P}`), so the same byte offset
    # applies to the shadow pointer — `Float64`'s tangent has `Float64`'s stride.
    f(x) = begin
        v = [x, 2x]
        p = pointer(v)
        GC.@preserve v Base.unsafe_store!(p + sizeof(Float64), 5x)
        sum(v)
    end
    @test f(1.0) == 6.0        # [x, 5x] -> 6x
    @test D(f, 1.0) == 6.0
    checkverify(f, (Float64,))

    # `pointer(v, i)` is the same machinery (an `add_ptr` with a primal-stride byte offset).
    g(x) = (v = [x, 2x]; GC.@preserve v Base.unsafe_store!(pointer(v, 2), 5x); sum(v))
    @test D(g, 1.0) == 6.0
end

@testset "caller-supplied shadow pointer, allocation-free" begin
    # Pointers in, pointers out: no array allocation of its own, so this isolates the pointer rules.
    # The tangent pointer is the caller's shadow buffer — exactly the contract `Dual{Ptr{P},Ptr{P}}`
    # describes.
    roundtrip(p::Ptr{Float64}, x) = (Base.unsafe_store!(p, 3x); Base.unsafe_load(p) * 2)
    control(::Ptr{Float64}, x) = x * 2                  # same signature, no pointer traffic

    v, dv = [1.0], [0.0]
    GC.@preserve v dv begin
        p, dp = pointer(v), pointer(dv)
        r = frule!!(Dual(roundtrip, NoTangent()), Dual(p, dp), Dual(2.0, 1.0))
        @test r === Dual(12.0, 6.0)
        @test v == [6.0]                                # the primal store landed
        @test dv == [3.0]                               # ...and its tangent landed in the shadow buffer

        # The rules emit bare intrinsics, so they add no allocation over an otherwise-identical
        # pointer-free call. (Both are nonzero: a `Dual{Ptr,…}` argument tuple allocates regardless of
        # what the body does, which is why this compares against a control rather than against 0.)
        dr, dc = Dual(roundtrip, NoTangent()), Dual(control, NoTangent())
        frule!!(dr, Dual(p, dp), Dual(2.0, 1.0)); frule!!(dc, Dual(p, dp), Dual(2.0, 1.0))
        @test (@allocated frule!!(dr, Dual(p, dp), Dual(2.0, 1.0))) ==
              (@allocated frule!!(dc, Dual(p, dp), Dual(2.0, 1.0)))
    end

    ir, rt = code_dual_ircode(roundtrip, (Ptr{Float64}, Float64))
    @test rt === Dual{Float64,Float64}
    stmts = string.(ir.stmts.stmt)
    @test count(s -> occursin("pointerset", s), stmts) == 2
    @test count(s -> occursin("pointerref", s), stmts) == 2
    @test !any(s -> occursin("frule!!", s), stmts)
end

@testset "second-order (dualized pointer IR is re-dualizable)" begin
    # Kept minimal on purpose: no `sum` hand rule, no reduction, so a failure here points at the new
    # constructs rather than at something they happen to sit next to.
    f(x) = (v = [x, 2x]; GC.@preserve v Base.unsafe_store!(pointer(v), 3x); v[1])
    @test D(f, 1.0) == 3.0
    checkverify2(f, (Float64,); order=2)
end

# Must be top level: a struct whose tangent has a different stride than the primal (`NoTangent` for
# the `Int` field collapses 16 bytes to 8).
struct StridePair
    a::Float64
    b::Int
end

@testset "graceful bails (located reason, no miscompile)" begin
    # A `Vector{Int}`'s shadow is a `Memory{NoTangent}` — zero-size elements, so its `ptr_or_offset`
    # is `Ptr(0x0)` rather than an address. Mirroring that read would hand back a null for the
    # Ptr->Ptr `bitcast` rule to launder into a real dereference.
    ints(x) = Base.unsafe_load(Ptr{Float64}(pointer(Int[1, 2]))) * x
    r = bail_reason(ints, (Float64,))
    @test r !== nothing
    @test occursin("does not store its elements inline", r)

    # A pointer built from an integer address has no tangent storage behind it, and `Ptr` has no zero
    # tangent to fall back on (`zero_tangent(::Ptr)` throws by design).
    fromint(u::UInt, x) = Base.unsafe_load(Ptr{Float64}(u)) * x
    r = bail_reason(fromint, (UInt, Float64))
    @test r !== nothing
    @test occursin("no tangent storage", r)

    # A byte offset does not carry over when the tangent element has a different stride.
    stride(v::Vector{StridePair}, x) = Base.unsafe_load(pointer(v) + 16).a * x
    r = bail_reason(stride, (Vector{StridePair}, Float64))
    @test r !== nothing
    @test occursin("same stride", r)
    @test occursin("add_ptr", r)

    # An unparameterized `Ptr` has no shadow pointer (`tangent_type` of an abstract `Ptr` is
    # `NoTangent`), so there is nothing to mirror onto.
    anyptr(p::Ptr, x) = Base.unsafe_load(p) * x
    r = bail_reason(anyptr, (Ptr, Float64))
    @test r !== nothing
    @test occursin("only a concrete `Ptr{P}`", r)

    # Every one of these is a *reason*, not the misleading "no rule registered" fallback the caller
    # uses for a genuinely unregistered intrinsic.
    for f_at in ((ints, (Float64,)), (fromint, (UInt, Float64)),
                 (stride, (Vector{StridePair}, Float64)), (anyptr, (Ptr, Float64)))
        @test !occursin("no rule registered", bail_reason(f_at[1], f_at[2]))
    end
end
