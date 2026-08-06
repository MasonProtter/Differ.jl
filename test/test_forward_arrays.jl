using Test
using Differ
using Differ: Dual, NoTangent, frule!!, build_tangent, get_tangent_field
using Differ: MutableTangent, code_dual_ircode

include(joinpath(@__DIR__, "testutils.jl"))

@testset "array indexing (forward mode)" begin
    # `v[i]`/`v[i]=x` lower to `Expr(:boundscheck)` + `memoryrefnew`/`memoryrefget`/
    # `memoryrefset!`. An element read differentiates like a per-element `getfield`; a write
    # mirrors the same builtin onto the shadow array (itself a real same-shape
    # `Array{tangent_type(P),N}`, no wrapper needed).
    getidx(v, i) = v[i]
    setidx!(v::Vector{Float64}, i::Int, x::Float64) = (v[i] = x; v)   # element write (memoryrefset!)
    function mysum(v::Vector{Float64})   # eachindex reduction: 3-arg getfield + `===` + memoryref chain
        s = 0.0
        for i in eachindex(v)
            s += v[i]
        end
        return s
    end
    tupfirst(t::Tuple{Float64,Float64}) = t[1]   # boundschecked tuple getfield (3-arg getfield forwarding)
    # Dynamic (non-literal) `getfield` index: `for i in 1:2` does not unroll, so this lowers to
    # `getfield(t, i)` with `i` a genuine SSAValue, not a literal. The homogeneous-tuple/NamedTuple
    # same-shape case (Phase B). d/dt_1 = d/dt_2 = 1.
    function tupsum_dyn(t::Tuple{Float64,Float64})
        s = 0.0
        for i in 1:2
            s += t[i]
        end
        return s
    end
    function ntupsum_dyn(t::@NamedTuple{a::Float64,b::Float64})
        s = 0.0
        for i in 1:2
            s += t[i]
        end
        return s
    end
    # Heterogeneous struct, dynamic `getfield` index: the genuinely hard case (per-field tangent
    # types differ) that must always bail, never miscompile.
    struct Het2; a::Float64; b::Int; end
    hetdyn(h::Het2, i::Int) = Core.getfield(h, i)
    # Dynamic `setfield!` index, Phase A only (no same-shape support for writes).
    mutable struct MP2; x::Float64; y::Float64; end
    function setdyn!(m::MP2, i::Int, v::Float64)
        Core.setfield!(m, i, v)
        return m.x + m.y
    end
    # Homogeneous MUTABLE struct, dynamic `getfield` READ index (Part 2b, the tractable gap, reusing
    # the homogeneous mutable `MP2` above). Every field shares one tangent type, so a runtime index
    # selects a validly-typed field whichever one it lands on; the contribution routes through the
    # object's own `MutableTangent` via the runtime-`Int` `increment_field_rdata!`. The index is a
    # genuine `Argument`/`SSAValue` (confirmed via `Base.code_ircode`), NOT const-folded to a literal.
    mp2get(m::MP2, i::Int) = Core.getfield(m, i)     # index is a genuine `Argument` (`_3`)
    function mp2sum_dyn(m::MP2)                       # loop index is a genuine `SSAValue` (a phi)
        s = 0.0
        for i in 1:2
            s += Core.getfield(m, i)
        end
        return s
    end

    # element read: one-hot seed picks out exactly the seeded component's directional derivative
    v = [1.0, 2.0, 3.0]
    @test frule!!(Dual(getidx, NoTangent()), Dual(v, [0.0,1.0,0.0]), Dual(2, NoTangent())) ==
          Dual(2.0, 1.0)
    @test frule!!(Dual(getidx, NoTangent()), Dual(v, [0.0,1.0,0.0]), Dual(1, NoTangent())).dx == 0.0
    checkverify(getidx, (Vector{Float64}, Int))

    # boundschecked tuple getfield (exercises the 3-arg getfield extra-args forwarding)
    rt = frule!!(Dual(tupfirst, NoTangent()),
                 Dual((1.0, 2.0), build_tangent(Tuple{Float64,Float64}, 5.0, 6.0)))
    @test rt.x == 1.0 && rt.dx == 5.0
    checkverify(tupfirst, (Tuple{Float64,Float64},))

    # dynamic (non-literal) getfield index (Phase B, forward): `for i in 1:2` does not unroll,
    # so `t[i]` reaches `dualize_to_ircode` as a genuine dynamic index. Raw (unresolved) before
    # the fix, this crashed with a TypeError: the primal index referenced the primal IR's own
    # (stale) SSA numbering once shadow instructions were interleaved.
    for seed in ((1.0, 0.0), (0.0, 1.0))
        d = frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), seed))
        @test d.x == 7.0 && d.dx == 1.0   # d(t1+t2)/dt_i = 1 for either seed direction
    end
    checkverify(tupsum_dyn, (Tuple{Float64,Float64},))

    # same, over a homogeneous NamedTuple.
    dn = frule!!(Dual(ntupsum_dyn, NoTangent()),
                 Dual((a=3.0, b=4.0), build_tangent(@NamedTuple{a::Float64,b::Float64}, 1.0, 0.0)))
    @test dn.x == 7.0 && dn.dx == 1.0
    checkverify(ntupsum_dyn, (@NamedTuple{a::Float64,b::Float64},))

    # regression: a dynamic getfield index into a HETEROGENEOUS struct (fields with different
    # tangent types) is the genuinely hard case. Must bail with a located error, never crash or
    # silently return a wrong/zero derivative.
    @test_throws "no dualization rule for builtin `getfield`" frule!!(
        Dual(hetdyn, NoTangent()),
        Dual(Het2(1.0, 2), build_tangent(Het2, 1.0, NoTangent())), Dual(1, NoTangent()))

    # regression: a dynamic setfield! index, Phase A only, always bails (no same-shape support
    # for writes).
    @test_throws ErrorException frule!!(
        Dual(setdyn!, NoTangent()),
        Dual(MP2(1.0, 2.0), MutableTangent((x=1.0, y=1.0))),
        Dual(1, NoTangent()), Dual(5.0, 1.0))

    # dynamic getfield index into a homogeneous MUTABLE struct (Part 2b): the runtime index is a
    # genuine `Argument` (`_3`), not a const-folded literal. Assert that, then differentiate. The
    # one-hot tangent seed (dx=1, dy=0) picks out exactly the selected field's derivative.
    @test !(Base.code_ircode(mp2get, (MP2, Int))[1][1].stmts.stmt[1].args[3] isa Union{Int,Symbol,QuoteNode})
    dmp1 = frule!!(Dual(mp2get, NoTangent()), Dual(MP2(3.0, 4.0), build_tangent(MP2, 1.0, 0.0)), Dual(1, NoTangent()))
    @test dmp1.x == 3.0 && dmp1.dx == 1.0
    dmp2 = frule!!(Dual(mp2get, NoTangent()), Dual(MP2(3.0, 4.0), build_tangent(MP2, 1.0, 0.0)), Dual(2, NoTangent()))
    @test dmp2.x == 4.0 && dmp2.dx == 0.0
    # same via a genuinely-dynamic loop index (a phi `SSAValue`): d/dm_i = 1 for each field.
    dms = frule!!(Dual(mp2sum_dyn, NoTangent()), Dual(MP2(3.0, 4.0), build_tangent(MP2, 1.0, 0.0)))
    @test dms.x == 7.0 && dms.dx == 1.0
    checkverify(mp2get, (MP2, Int))
    checkverify(mp2sum_dyn, (MP2,))

    # element write: mutates the caller's own primal and tangent arrays in place at the written
    # index only (each aliased to the caller's array, not to each other).
    v2, dv2 = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    r = frule!!(Dual(setidx!, NoTangent()), Dual(v2, dv2), Dual(2, NoTangent()), Dual(5.0, 7.0))
    @test r.x == [1.0, 5.0, 3.0] && v2 == [1.0, 5.0, 3.0]
    @test r.dx == [10.0, 7.0, 30.0] && dv2 == [10.0, 7.0, 30.0]
    checkverify(setidx!, (Vector{Float64}, Int, Float64))

    # reduction loop: linear in v, so directional derivative == sum of the seed components
    v3, dv3 = [1.0, 2.0, 3.0, 4.0], [1.0, -1.0, 0.5, 2.0]
    r3 = frule!!(Dual(mysum, NoTangent()), Dual(v3, dv3))
    @test r3.x ≈ sum(v3) && r3.dx ≈ sum(dv3)
    checkverify(mysum, (Vector{Float64},))

    # safety regression: `Dual`'s constructor never checks a tangent array's *length* matches
    # its primal's, so a too-short tangent must raise a catchable BoundsError (from the shadow
    # `memoryrefnew`'s always-on boundscheck), not corrupt memory or segfault.
    @test_throws BoundsError frule!!(Dual(getidx, NoTangent()), Dual([1.0,2.0,3.0], [1.0]),
                                      Dual(2, NoTangent()))
end

@testset "nothing-returning primal (forward mode, ISSUES #53)" begin
    # A mutator that ends in `return nothing`, the ordinary way to write an in-place function.
    # `return nothing` survives optimization as a bare `GlobalRef` (`Main.nothing`), which used to
    # (1) leak into value position (verify_ir rejects a non-Core/Base GlobalRef there) and (2) get
    # typed `GlobalRef` instead of `Nothing`. Fixed: the result is a `Dual{Nothing,NoTangent}` and
    # the shadow array still receives the tangent write.
    noret!(v, x) = (v[1] = x; nothing)
    v, dv = [1.0, 2.0], [0.0, 0.0]
    r = frule!!(Dual(noret!, NoTangent()), Dual(v, dv), Dual(5.0, 7.0))
    @test r.x === nothing && r.dx === NoTangent()
    @test v == [5.0, 2.0] && dv == [7.0, 0.0]      # primal + shadow both written
    checkverify(noret!, (Vector{Float64}, Float64))

    # The `bench/workloads.jl` `vecloop!` shape: a `nothing`-returning write loop.
    vecloop!(v::Vector{Float64}, x::Float64) =
        (for i in 1:length(v); @inbounds v[i] = x * i; end; nothing)
    v2, dv2 = zeros(3), zeros(3)
    r2 = frule!!(Dual(vecloop!, NoTangent()), Dual(v2, dv2), Dual(3.0, 1.0))
    @test r2.x === nothing && r2.dx === NoTangent()
    @test v2 == [3.0, 6.0, 9.0] && dv2 == [1.0, 2.0, 3.0]   # d(x*i)/dx = i
    checkverify(vecloop!, (Vector{Float64}, Float64))
end

# Must be top level: the point of these is that the struct *name* and the global are module-level
# bindings, so the primal IR really does carry `%new(<Module>.NewHolder, …)` and a raw `GlobalRef`
# field operand. A testset-local `struct` would not reproduce either.
struct NewHolder; a::Float64; b::Any; end
struct NewPair;   a::Float64; b::Int; end
const NEWCONST = [1.5, 2.5]                      # mutable, so inference keeps it a `GlobalRef` operand
                                                 # rather than folding it into a literal
newnothing(x) = NewHolder(x*x, nothing)
newpair(x)    = [NewPair(x, 1)][1].a
newconst(x)   = NewHolder(x*x, NEWCONST)
newconstuse(x) = NEWCONST[1] * x

@testset "`%new` with GlobalRef operands (forward mode, ISSUES #60)" begin
    # `%new`'s type argument and its field operands are both value positions `verify_ir` checks, and
    # the arm used to test `T <: Dual` on the raw node. A struct defined at module level lowers to
    # `%new(<Module>.NewHolder, %1, <Module>.nothing)`, so both defects fired at once: a `TypeError`
    # from `<:` on a `GlobalRef`, and (once past that) "Unbound or partitioned GlobalRef not allowed
    # in value position". Both operands are now resolved through the binding.
    r = frule!!(Dual(newnothing, NoTangent()), Dual(3.0, 1.0))
    @test r.x == NewHolder(9.0, nothing)
    @test get_tangent_field(r.dx, :a) ≈ 6.0            # d(x²)/dx
    @test get_tangent_field(r.dx, :b) === NoTangent()  # `nothing` is non-differentiable
    checkverify(newnothing, (Float64,))

    # Same defect via the *type* argument alone, with every field operand an ordinary SSA/literal.
    @test frule!!(Dual(newpair, NoTangent()), Dual(4.0, 1.0)) === Dual(4.0, 1.0)
    checkverify(newpair, (Float64,))

    # A `const` global whose tangent is *not* `NoTangent`: the field's shadow is a genuine runtime
    # `zero_tangent` of the bound value, and that call's own operand is a value position too.
    r2 = frule!!(Dual(newconst, NoTangent()), Dual(3.0, 1.0))
    @test r2.x.b === NEWCONST
    @test get_tangent_field(r2.dx, :b) == [0.0, 0.0]   # a const global contributes no tangent
    checkverify(newconst, (Float64,))

    # Reading through the same binding. Resolving its *type* has to happen at the interpreter's
    # inference world, not the ambient one: inside the generated `frule!!` body the ambient world
    # predates the `const` declaration, and answering "not constant" there degrades the operand to
    # `Any` and sends the `getfield` rule down its general-struct branch, a `MethodError` at run
    # time, which `code_dual_ircode` (running at the ambient world) would not reproduce.
    @test frule!!(Dual(newconstuse, NoTangent()), Dual(3.0, 1.0)) === Dual(4.5, 1.5)
    checkverify(newconstuse, (Float64,))
end

@testset "mutable-struct field mutation (setfield!, forward mode)" begin
    # setfield! mutates the primal in place and its `MutableTangent` shadow via
    # `set_tangent_field!`, the mutation-side counterpart of the existing getfield/
    # get_tangent_field read path.
    mutable struct MPoint; x::Float64; y::Float64; end
    mpoint_read(p::MPoint) = p.x + p.y                    # read-only control for the setfield! test below
    mpoint_setx!(p::MPoint, v) = (p.x = v; p.x + p.y)     # mutable-struct field mutation (setfield!)

    p0, dp0 = MPoint(1.0, 2.0), build_tangent(MPoint, 1.0, 0.0)
    rr = frule!!(Dual(mpoint_read, NoTangent()), Dual(p0, dp0))   # read-only control
    @test rr.x ≈ 3.0 && rr.dx ≈ 1.0

    p, dp = MPoint(1.0, 2.0), build_tangent(MPoint, 1.0, 0.0)
    r = frule!!(Dual(mpoint_setx!, NoTangent()), Dual(p, dp), Dual(10.0, 3.0))
    @test p.x == 10.0 && p.y == 2.0                                       # primal mutated in place
    @test get_tangent_field(dp, :x) == 3.0 && get_tangent_field(dp, :y) == 0.0  # shadow mutated too
    @test r.x ≈ 12.0 && r.dx ≈ 3.0
    Core.Compiler.verify_ir(code_dual_ircode(mpoint_setx!, (MPoint, Float64))[1])
end

@testset "array allocation (forward mode)" begin
    # `zeros`/`similar`/`Vector{T}(undef,n)`/comprehensions all lower to the identical
    # `Core.memorynew -> Core.memoryrefnew -> Core.tuple -> %new` sequence. The shadow allocates
    # a same-length `Memory{tangent_type(P)}` and the shadow `%new` uses the shadow ref but the
    # primal's own (structural, non-differentiable) size tuple.
    allocarr(n) = (v = zeros(n); v[1] = 1.0; v[1])
    allocwrite(x) = (v = zeros(2); v[1] = x; v[2] = x + 1.0; v[1] + v[2])
    allocsim(v) = (w = similar(v); w[1] = 2.0*v[1]; w[2] = v[1]+v[2]; w[1]+w[2])
    alloc2d(m, n) = (A = zeros(m, n); A[1,2] = 5.0; A[2,1] = 7.0; A[1,2] + A[2,1])
    alloccomp(x, n) = (v = [x*i for i in 1:n]; v[1] + v[2])

    r = frule!!(Dual(allocarr, NoTangent()), Dual(3, NoTangent()))
    @test r.x == 1.0 && r.dx == 0.0
    checkverify(allocarr, (Int,))

    r = frule!!(Dual(allocwrite, NoTangent()), Dual(3.0, 1.0))
    @test r.x ≈ 7.0 && r.dx ≈ 2.0
    checkverify(allocwrite, (Float64,))

    r = frule!!(Dual(allocsim, NoTangent()), Dual([1.0, 2.0], [1.0, 0.0]))
    @test r.x ≈ 5.0 && r.dx ≈ 3.0
    checkverify(allocsim, (Vector{Float64},))

    r = frule!!(Dual(alloc2d, NoTangent()), Dual(2, NoTangent()), Dual(3, NoTangent()))
    @test r.x ≈ 12.0 && r.dx == 0.0
    checkverify(alloc2d, (Int, Int))

    r = frule!!(Dual(alloccomp, NoTangent()), Dual(3.0, 1.0), Dual(3, NoTangent()))
    @test r.x ≈ 9.0 && r.dx ≈ 3.0
    checkverify(alloccomp, (Float64, Int))

    # Still out of scope: growing an existing array (`push!`/`resize!`), which calls
    # `Core.memoryrefoffset` directly, a distinct, still-unhandled builtin (unrelated to
    # allocation). Should bail gracefully with an `ErrorException`.
    growvec!(v, x) = push!(v, x)
    err = try
        frule!!(Dual(growvec!, NoTangent()), Dual([1.0,2.0], [0.0,0.0]), Dual(3.0, 1.0))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("memoryrefoffset", err.msg)
    @test occursin("at %", err.msg)
end
