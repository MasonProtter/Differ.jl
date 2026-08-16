using Test
using DifferReverse
using DifferReverse: NoTangent, NoRData, NoFData, rev_gradient, rev_gradient!
using DifferReverse: MutableTangent, get_tangent_field, zero_tangent, zero_fcodual, rrule!!, Ctx
# `Dual`/`frule!!` here are DifferForwards' forward-mode carrier, used purely as an independent
# numerical oracle.
using DifferForwards: Dual, frule!!

include(joinpath(@__DIR__, "testutils.jl"))

mutable struct MPoint; x::Float64; y::Float64; end   # used by several testsets below

# Must be top-level, not testset-local: Julia's closure-conversion boxes a named function nested in
# local scope more conservatively than a closure literal, which would give `r` a real (boxed)
# tangent type instead of the plain `Ref{Float64}` capture this test wants to exercise.
function make_refmul_closure()
    r = Ref(1.0)
    g(y) = (r[] *= y; nothing)
    return g, r
end

@testset "reverse mode: mutable-struct getfield/setfield!" begin
    mpoint_read(p::MPoint) = p.x + p.y                    # read-only control
    mpoint_setx!(p::MPoint, v) = (p.x = v; p.x + p.y)     # mutable-struct field mutation (setfield!)

    p0 = MPoint(2.0, 3.0)
    _, dp_read = rev_gradient(mpoint_read, p0)
    @test dp_read == MutableTangent{@NamedTuple{x::Float64,y::Float64}}((x=1.0, y=1.0))

    p1 = MPoint(2.0, 3.0)
    _, dp_setx, dv_setx = rev_gradient(mpoint_setx!, p1, 10.0)
    # p.x is overwritten before use, so its own gradient contribution is 0; p.y and v flow straight
    # through to the `+`.
    @test dp_setx == MutableTangent{@NamedTuple{x::Float64,y::Float64}}((x=0.0, y=1.0))
    @test dv_setx == 1.0
    # Cross-check against the already-trusted forward-mode `frule!!` result for the same primal.
    p2 = MPoint(2.0, 3.0)
    fwd_setx = frule!!(Dual(mpoint_setx!, NoTangent()), Dual(p2, zero_tangent(p2)), Dual(10.0, 1.0))
    @test dv_setx ≈ fwd_setx.dx
    h = 1e-6
    fd_v = (mpoint_setx!(MPoint(2.0, 3.0), 10.0 + h) - mpoint_setx!(MPoint(2.0, 3.0), 10.0 - h)) / 2h
    @test dv_setx ≈ fd_v rtol = 1e-5

    checkverify_rev(mpoint_read, (MPoint,))
    checkverify_rev(mpoint_setx!, (MPoint, Float64))
    check_stack_balance(mpoint_setx!, MPoint(2.0, 3.0), 10.0)
end

@testset "reverse mode: array mutation combined with a full-array read" begin
    # d(sum)/dx[1] = 2 (mutated), d(sum)/dx[2] = 1 (untouched).
    function arr_mutate_sum!(x::Vector{Float64})
        x[1] *= 2.0
        return sum(x)
    end

    x8 = [3.0, 4.0]
    _, dx_mutsum = rev_gradient(arr_mutate_sum!, x8)
    @test dx_mutsum == [2.0, 1.0]
    for k in eachindex(x8)
        xp = copy(x8); xp[k] += 1e-6
        xm = copy(x8); xm[k] -= 1e-6
        @test dx_mutsum[k] ≈ (arr_mutate_sum!(xp) - arr_mutate_sum!(xm)) / 2e-6 rtol = 1e-5
    end

    checkverify_rev(arr_mutate_sum!, (Vector{Float64},))
    check_stack_balance(arr_mutate_sum!, [3.0, 4.0])
end

@testset "reverse mode: repeated mutation in a loop (save/restore)" begin
    # The case the save/restore machinery (`:old_primal`/`:old_tangent`) exists for: a version
    # without restore passes the straight-line cases above and fails only here. Also proves the
    # *same* `MutableTangent` is shared across every iteration's separate `getfield` access — a
    # broken-aliasing bug would give a numerically wrong answer, not an error.
    # d(result)/dr = ys[1]*ys[2] = 12, d(result)/dys[1] = r*ys[2] = 8, d(result)/dys[2] = r*ys[1] = 6.
    function refprod_loop!(r::Base.RefValue{Float64}, ys::Vector{Float64})
        for y in ys
            r[] *= y
        end
        return r[]
    end

    r0 = Ref(2.0)
    ys0 = [3.0, 4.0]
    _, dr_loop, dys_loop = rev_gradient(refprod_loop!, r0, ys0)
    @test get_tangent_field(dr_loop, 1) ≈ 12.0
    @test dys_loop == [8.0, 6.0]
    @test r0[] == 2.0   # forward-replay mutates r0; the pullback's restore leaves it as found

    checkverify_rev(refprod_loop!, (Base.RefValue{Float64}, Vector{Float64}))
    check_stack_balance(refprod_loop!, Ref(2.0), [3.0, 4.0])
end

@testset "reverse mode: closure over a Ref" begin
    # A closure over a `Ref`, read via two separate `getfield` calls (once before the mutation,
    # once after) that must resolve to the *same* underlying `MutableTangent`. An aliasing bug here
    # (two independently-zeroed copies instead of one shared object) would silently give the wrong
    # gradient rather than erroring, so this checks the value, not just that it doesn't throw.
    #
    # Must run without erroring, leave `r` restored after the round trip, and (since the primal
    # returns `nothing`) contribute no gradient to `y` through the return value.
    g, r = make_refmul_closure()
    gcd, pb = rrule!!(zero_fcodual(g), Ctx(), DifferReverse.CoDual(3.0, NoFData()))
    @test DifferReverse.primal(gcd) === nothing
    _, dy_closure = pb(NoRData())
    @test dy_closure == 0.0
    @test r[] == 1.0
end

@testset "reverse mode: setfield! of an array-valued field (fdata aliasing)" begin
    # `setfield!` of an array-valued field: the field's shadow is aliased to the argument's real
    # shadow (not a fresh `zero_tangent`), so in-place accumulation into that shared shadow after
    # the assignment flows back to `w`. `setbox!` returns `nothing` with no downstream read of
    # `b.v`, so it doesn't really exercise the aliasing; `setbox_sum!` reads `b.v` back through the
    # return value and is the real test.
    mutable struct MBox; v::Vector{Float64}; end
    setbox!(b::MBox, w::Vector{Float64}) = (b.v = w; nothing)
    setbox_sum!(b::MBox, w::Vector{Float64}) = (b.v = w; sum(b.v))

    b1 = MBox([1.0, 2.0])
    w1 = [3.0, 4.0]
    _, db_setbox, dw_setbox = rev_gradient(setbox_sum!, b1, w1)
    @test db_setbox == MutableTangent{@NamedTuple{v::Vector{Float64}}}((v=[0.0, 0.0],))
    @test dw_setbox == [1.0, 1.0]
    h2 = 1e-6
    for k in eachindex(w1)
        wp = copy(w1); wp[k] += h2
        wm = copy(w1); wm[k] -= h2
        fd = (setbox_sum!(MBox([1.0, 2.0]), wp) - setbox_sum!(MBox([1.0, 2.0]), wm)) / 2h2
        @test dw_setbox[k] ≈ fd rtol = 1e-5
        # Cross-check against forward mode: the reverse-mode gradient's k-th entry is exactly the
        # forward-mode directional derivative along the k-th basis vector.
        ek = zeros(length(w1)); ek[k] = 1.0
        fwd_setbox = frule!!(Dual(setbox_sum!, NoTangent()),
                             Dual(MBox([1.0, 2.0]), zero_tangent(MBox([1.0, 2.0]))),
                             Dual(w1, ek))
        @test dw_setbox[k] ≈ fwd_setbox.dx
    end
    # `setbox!` (returns `nothing`, `b.v` never read back downstream) runs to completion instead
    # of bailing: the aliasing mechanism handles it fine, there's just nothing downstream to carry
    # a gradient to `w`. `rev_gradient`/`rev_gradient!` seed the pullback with `one(y)`, which has
    # no method for `y === nothing`, so exercise `rrule!!` directly with an explicit `NoRData()`
    # seed instead.
    b0 = MBox([1.0, 2.0])
    bshadow0 = MutableTangent{@NamedTuple{v::Vector{Float64}}}((v=zeros(2),))
    w0 = [3.0, 4.0]
    wshadow0 = zeros(2)
    ycd0, pb0 = rrule!!(zero_fcodual(setbox!), Ctx(), DifferReverse.CoDual(b0, bshadow0), DifferReverse.CoDual(w0, wshadow0))
    @test DifferReverse.primal(ycd0) === nothing
    pb0(NoRData())
    @test bshadow0 == MutableTangent{@NamedTuple{v::Vector{Float64}}}((v=[0.0, 0.0],))
    @test wshadow0 == [0.0, 0.0]

    checkverify_rev(setbox_sum!, (MBox, Vector{Float64}))
    check_stack_balance(setbox_sum!, MBox([1.0, 2.0]), [3.0, 4.0])
end

@testset "reverse mode: array allocation" begin
    # `mutate_nested!` returns `nothing`, so `rev_gradient`'s `one(y)` seeding doesn't apply;
    # exercise `rrule!!` directly with an explicit `NoRData()` seed, as `setbox!` above. The
    # freshly-allocated `[9.0, 9.0]` has no dependency on `x`, so its aliased shadow is zero; the
    # pullback's restore leaves `x`/its shadow as found.
    function mutate_nested!(x::Vector{Vector{Float64}})
        x[1] = [9.0, 9.0]
        return nothing
    end

    xm0 = [[1.0, 2.0], [3.0, 4.0]]
    xmshadow0 = [[0.0, 0.0], [0.0, 0.0]]
    ycdm, pbm = rrule!!(zero_fcodual(mutate_nested!), Ctx(), DifferReverse.CoDual(xm0, xmshadow0))
    @test DifferReverse.primal(ycdm) === nothing
    @test xm0 == [[9.0, 9.0], [3.0, 4.0]]        # forward replay mutated x[1] in place
    @test xmshadow0 == [[0.0, 0.0], [0.0, 0.0]]  # aliased shadow of the fresh array is zero
    @test pbm(NoRData()) == (NoRData(), NoRData())
    @test xm0 == [[1.0, 2.0], [3.0, 4.0]]        # pullback restored the overwritten slot
    @test xmshadow0 == [[0.0, 0.0], [0.0, 0.0]]
    checkverify_rev(mutate_nested!, (Vector{Vector{Float64}},))

    # Scalar-returning allocation tests (so `rev_gradient` applies directly): `zeros`/explicit index
    # writes, not a `[a,b]` literal, to keep this testset about allocation specifically.
    #
    # `alloc_and_sum`: allocate, write both elements from `x`, read both back locally — d/dx = 3.
    alloc_and_sum(x::Float64) = (v = zeros(2); v[1] = x; v[2] = 2 * x; v[1] + v[2])
    _, dx_aas = rev_gradient(alloc_and_sum, 3.0)
    @test dx_aas ≈ 3.0
    h = 1e-6
    @test dx_aas ≈ (alloc_and_sum(3.0 + h) - alloc_and_sum(3.0 - h)) / 2h rtol = 1e-5
    checkverify_rev(alloc_and_sum, (Float64,))
    check_stack_balance(alloc_and_sum, 3.0)

    # `alloc_store_read!`: allocate, write from `a`, store into the argument array `x[1]`, then
    # read back through `x` — exercises allocation and argument-array aliasing together. `x`'s
    # original `x[1]` is overwritten before being read, so its own gradient is zero. d/da = 3.
    function alloc_store_read!(x::Vector{Vector{Float64}}, a::Float64)
        v = zeros(2)
        v[1] = a
        v[2] = 2 * a
        x[1] = v
        return x[1][1] + x[1][2]
    end
    _, dx_asr, da_asr = rev_gradient(alloc_store_read!, [[1.0, 2.0], [3.0, 4.0]], 5.0)
    @test dx_asr == [[0.0, 0.0], [0.0, 0.0]]
    @test da_asr ≈ 3.0
    @test da_asr ≈ (alloc_store_read!([[1.0, 2.0], [3.0, 4.0]], 5.0 + h) -
                    alloc_store_read!([[1.0, 2.0], [3.0, 4.0]], 5.0 - h)) / 2h rtol = 1e-5
    checkverify_rev(alloc_store_read!, (Vector{Vector{Float64}}, Float64))
    check_stack_balance(alloc_store_read!, [[1.0, 2.0], [3.0, 4.0]], 5.0)

    # Adversarial: reverse mode's shadow `memoryrefnew` forces its own boundscheck flag `true`
    # (`CoDual`'s constructor never checks a caller-supplied tangent array's length against its
    # primal's), so a too-short shadow raises a catchable `BoundsError` instead of corrupting
    # memory via an unchecked out-of-bounds `MemoryRef`.
    function arr_idx3(x::Vector{Float64})
        return x[3]
    end
    @test_throws BoundsError rrule!!(zero_fcodual(arr_idx3), Ctx(),
                                     DifferReverse.CoDual([1.0, 2.0, 3.0, 4.0], [1.0]))

    # Regression: growing an existing array (`push!`/`resize!`) is still out of scope, and must bail
    # cleanly (a located reason, not a crash) rather than miscompile.
    growvec!(v, x) = push!(v, x)
    @test_throws ErrorException rev_gradient(growvec!, [1.0, 2.0], 3.0)
end

@testset "reverse mode: nested-array aliasing" begin
    # Nested-array read and write-then-read, no allocation involved.
    #
    # `x[1]`'s own `memoryrefget` result is a tracked provenance root (its shadow is the
    # corresponding element of `x`'s own shadow array), so summing it differentiates directly.
    nested_read(x::Vector{Vector{Float64}}) = sum(x[1])

    # `x[1] = w` aliases the shadow slot to `w`'s own shadow (rather than a fresh zero), so reading
    # `x[1]` back out afterward resolves to `w`'s real shadow array, and later accumulation into it
    # lands in `w`'s gradient rather than a detached copy.
    function nested_write_existing(x::Vector{Vector{Float64}}, w::Vector{Float64})
        x[1] = w
        return sum(x[1])
    end

    # Adversarial: read the aliased array back through *both* names — the gradient must sum both
    # contributions, not just whichever one a broken (fresh-zero) shadow happened to see.
    function nested_write_read_both(x::Vector{Vector{Float64}}, w::Vector{Float64})
        x[1] = w
        return sum(x[1]) + sum(w)
    end

    # Adversarial: mutate a scalar element *through* the alias (`x[1][1] = ...`). Real Julia array
    # aliasing means this also mutates `w` itself in the primal; the gradient must track that
    # mutation back to `w`'s original value, not vanish or double-count it.
    function nested_write_mutate_through(x::Vector{Vector{Float64}}, w::Vector{Float64})
        x[1] = w
        x[1][1] = 3.0 * x[1][1]
        return w[1] + w[2]
    end

    _, dx_nr = rev_gradient(nested_read, [[1.0, 2.0], [3.0, 4.0]])
    @test dx_nr == [[1.0, 1.0], [0.0, 0.0]]
    h = 1e-6
    for k in 1:2
        xp = [[1.0, 2.0], [3.0, 4.0]]; xp[1][k] += h
        xm = [[1.0, 2.0], [3.0, 4.0]]; xm[1][k] -= h
        @test dx_nr[1][k] ≈ (nested_read(xp) - nested_read(xm)) / 2h rtol = 1e-5
    end

    _, dx_nwe, dw_nwe = rev_gradient(nested_write_existing, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    @test dx_nwe == [[0.0, 0.0], [0.0, 0.0]]   # x[1] overwritten before use — zero, not aliased
    @test dw_nwe == [1.0, 1.0]
    for k in 1:2
        wp = [5.0, 6.0]; wp[k] += h
        wm = [5.0, 6.0]; wm[k] -= h
        fd = (nested_write_existing([[1.0, 2.0], [3.0, 4.0]], wp) -
              nested_write_existing([[1.0, 2.0], [3.0, 4.0]], wm)) / 2h
        @test dw_nwe[k] ≈ fd rtol = 1e-5
    end

    _, dx_rb, dw_rb = rev_gradient(nested_write_read_both, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    @test dx_rb == [[0.0, 0.0], [0.0, 0.0]]
    @test dw_rb == [2.0, 2.0]

    _, dx_mt, dw_mt = rev_gradient(nested_write_mutate_through, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    @test dx_mt == [[0.0, 0.0], [0.0, 0.0]]
    @test dw_mt == [3.0, 1.0]

    # Adversarial: a pre-seeded non-zero incoming shadow on `w` must accumulate, not overwrite.
    # Exercise `rrule!!` directly with explicit shadows.
    x0 = [[1.0, 2.0], [3.0, 4.0]]
    xshadow0 = [[0.0, 0.0], [0.0, 0.0]]
    w0 = [5.0, 6.0]
    wshadow0 = [10.0, 20.0]
    ycd_nwe, pb_nwe = rrule!!(zero_fcodual(nested_write_existing), Ctx(),
                              DifferReverse.CoDual(x0, xshadow0), DifferReverse.CoDual(w0, wshadow0))
    pb_nwe(1.0)
    @test wshadow0 == [11.0, 21.0]                 # accumulated onto the pre-seeded [10.0, 20.0]
    @test xshadow0 == [[0.0, 0.0], [0.0, 0.0]]      # restore leaves x's own slot untouched

    checkverify_rev(nested_read, (Vector{Vector{Float64}},))
    checkverify_rev(nested_write_existing, (Vector{Vector{Float64}}, Vector{Float64}))
    checkverify_rev(nested_write_read_both, (Vector{Vector{Float64}}, Vector{Float64}))
    checkverify_rev(nested_write_mutate_through, (Vector{Vector{Float64}}, Vector{Float64}))

    check_stack_balance(nested_read, [[1.0, 2.0], [3.0, 4.0]])
    check_stack_balance(nested_write_existing, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    check_stack_balance(nested_write_read_both, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    check_stack_balance(nested_write_mutate_through, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])

    # `vs[1]`'s provenance is tracked, so aliasing it into a mutable struct's array field is safe
    # too: d/dvs[1] = [1.0, 1.0], d/dvs[2] = [0.0, 0.0] (untouched).
    mutable struct MArrBox; v::Vector{Float64}; end
    @noinline arrbox_inner(m::MArrBox) = m.v[1] + m.v[2]
    function arrbox_untraced(vs::Vector{Vector{Float64}})
        m = MArrBox(vs[1])
        return arrbox_inner(m)
    end
    _, dvs_abu = rev_gradient(arrbox_untraced, [[1.0, 2.0], [3.0, 4.0]])
    @test dvs_abu == [[1.0, 1.0], [0.0, 0.0]]
    checkverify_rev(arrbox_untraced, (Vector{Vector{Float64}},))
    check_stack_balance(arrbox_untraced, [[1.0, 2.0], [3.0, 4.0]])
end

@testset "reverse mode: %new of a mutable struct + recursion" begin
    # `%new` of a mutable struct, purely locally (no cross-call boundary), checked against finite
    # differences. Plain local create+mutate+read gets scalar-replaced away entirely by SROA before
    # reverse mode ever sees a `%new`; nesting the fresh `MPoint` inside a second, also-freshly-
    # created mutable wrapper forces the `MPoint`'s own `%new` to survive (the wrapper itself gets
    # scalarized away, so the IR reverse mode actually sees is just `%new(MPoint,...)` +
    # getfield/setfield!/getfield/getfield/add, no trace of the wrapper).
    mutable struct MPointBox; p::MPoint; end
    function newmut_local(x::Float64)
        p = MPoint(x, 1.0)
        p.x = p.x + 2.0
        box = MPointBox(p)
        return box.p.x + box.p.y
    end

    _, dx_nml = rev_gradient(newmut_local, 3.0)
    @test dx_nml ≈ 1.0
    h = 1e-6
    @test dx_nml ≈ (newmut_local(3.0 + h) - newmut_local(3.0 - h)) / 2h rtol = 1e-5
    checkverify_rev(newmut_local, (Float64,))
    check_stack_balance(newmut_local, 3.0)

    # `%new` of a mutable struct crossing a genuine `@noinline` recursive-call boundary. `p` is a
    # genuine argument, tracked, so it threads through into the recursive `:invoke`. `p` crosses a
    # call boundary Julia's SROA can't see through; otherwise `p` never escapes and the optimizer
    # elides the `%new` before reverse mode ever sees it.
    @noinline mpoint_xy(p::MPoint) = p.x + p.y
    newmut(x::Float64) = (p = MPoint(x, 1.0); mpoint_xy(p))

    _, dx_nm = rev_gradient(newmut, 5.0)
    @test dx_nm ≈ 1.0
    @test dx_nm ≈ (newmut(5.0 + h) - newmut(5.0 - h)) / 2h rtol = 1e-5
    checkverify_rev(newmut, (Float64,))
    check_stack_balance(newmut, 5.0)

    # Mutation inside the callee: `p` is created locally, then handed to a `@noinline` callee that
    # mutates it in place — the callee's `setfield!` accumulates into the very same shadow
    # `MutableTangent` the caller's local `%new` built.
    @noinline mpoint_mutate!(p::MPoint) = (p.x = p.x + 5.0; p.x + p.y)
    newmut_recursive_mutate(x::Float64) = mpoint_mutate!(MPoint(x, 1.0))

    _, dx_nmr = rev_gradient(newmut_recursive_mutate, 2.0)
    @test dx_nmr ≈ 1.0
    @test dx_nmr ≈ (newmut_recursive_mutate(2.0 + h) - newmut_recursive_mutate(2.0 - h)) / 2h rtol = 1e-5
    checkverify_rev(newmut_recursive_mutate, (Float64,))
    check_stack_balance(newmut_recursive_mutate, 2.0)
end

# Bulk primal save/restore (`_bulk_save_args`, `reverse_interp.jl`).
#
# A store's pullback restores the element it overwrote so the primal is left as the call found it.
# For an argument array written in a loop, that's done once for the whole array instead of per
# element — sound only because no pullback rule anywhere reads primal memory.
#
# The tests that matter here are the negative ones: bulk mode must not fire where it isn't sound,
# and must not disturb the shadow, which (unlike the primal) is live during the reverse sweep and
# is still restored one element at a time.
@testset "reverse mode: bulk primal save/restore" begin
    h = 1e-6

    # A loop writing an argument array, then reading it back. `d/da` = sum over i of i = 6.
    function bulk_write_read(v::Vector{Float64}, a::Float64)
        for i in 1:length(v)
            @inbounds v[i] = a * i
        end
        s = 0.0
        for i in 1:length(v)
            @inbounds s += v[i]
        end
        return s
    end
    v = [7.0, 8.0, 9.0]
    _, _, da = rev_gradient(bulk_write_read, v, 3.0)
    @test da ≈ 6.0
    @test da ≈ central_diff(a -> bulk_write_read(copy(v), a), 3.0) rtol = 1e-5
    @test v == [7.0, 8.0, 9.0]          # primal restored by the bulk copy-back
    checkverify_rev(bulk_write_read, (Vector{Float64}, Float64))
    check_stack_balance(bulk_write_read, v, 3.0)
    @test v == [7.0, 8.0, 9.0]

    # The `Memory` form of the same shape: the ref chain is rooted at a `Core.memorynew`-free
    # `Memory` argument via the 1-arg `memoryrefnew`, no `Array.ref` hop. `Memory` can't go through
    # `rev_gradient` (no `zero_tangent` method), so drive `rrule!!` directly.
    function bulk_mem!(out::Memory{Float64}, x::Float64, N::Int)
        for i in 1:N
            @inbounds out[i] = x
        end
        return nothing
    end
    om = Memory{Float64}(undef, 4); fill!(om, 2.0)
    dm = Memory{Float64}(undef, 4); fill!(dm, 0.0)
    _, pbm = rrule!!(zero_fcodual(bulk_mem!), Ctx(), DifferReverse.CoDual(om, dm),
                     zero_fcodual(5.0), zero_fcodual(4))
    pbm(NoRData())
    @test om == Memory{Float64}([2.0, 2.0, 2.0, 2.0])   # restored
    # One `MemoryRef` (shadow handle) + one `Float64` (old tangent) per iteration — the old primal
    # and the primal `MemoryRef` are exactly what bulk mode removed.
    check_tape_size(bulk_mem!, (Memory{Float64}, Float64, Int); bytes=24)

    # Aliasing within the loop: each iteration reads the element the previous one wrote. The
    # per-element *shadow* restore is what makes this come out right, and bulk mode must not touch
    # it. d/da = 2^(n-1) = 8.
    function bulk_shift!(x::Vector{Float64}, a::Float64)
        x[1] = a
        for i in 2:length(x)
            @inbounds x[i] = x[i-1] * 2
        end
        return x[length(x)]
    end
    xs = [1.0, 2.0, 3.0, 4.0]
    _, _, das = rev_gradient(bulk_shift!, xs, 5.0)
    @test das ≈ 8.0
    @test das ≈ central_diff(a -> bulk_shift!(copy(xs), a), 5.0) rtol = 1e-5
    @test xs == [1.0, 2.0, 3.0, 4.0]
    check_stack_balance(bulk_shift!, xs, 5.0)

    # Straight-line stores stay on the per-element scheme (not bulk), but their `MemoryRef`
    # handles are re-derived in the pullback rather than pushed — so no comms tuple for
    # `straightline!` contains a GC-tracked `MemoryRef`.
    straightline!(v::Vector{Float64}, a::Float64) = (v[1] = a; v[2] = 2a; v[1] + v[2])
    vsl = [1.0, 2.0]
    _, _, dasl = rev_gradient(straightline!, vsl, 4.0)
    @test dasl ≈ 3.0
    @test vsl == [1.0, 2.0]
    @test all(T -> !(T <: Tuple) || all(F -> !(F <: MemoryRef), fieldtypes(T)),
              check_tape_size(straightline!, (Vector{Float64}, Float64)))

    # NEGATIVE, and the test that discriminates a correct implementation from a plausible-looking
    # wrong one: an argument's *incoming* shadow must come back exactly as supplied. Every other
    # test in this suite starts from zero shadows (`value_and_gradient!` zeroes them), so nothing
    # else would catch a bulk scheme that wrongly swallowed the shadow too.
    function bulk_fill!(v::Vector{Float64}, a::Float64)
        for i in 1:length(v)
            @inbounds v[i] = a
        end
        return nothing
    end
    u = [1.0, 2.0, 3.0]
    du = [10.0, 20.0, 30.0]          # deliberately nonzero on entry
    _, pbu = rrule!!(zero_fcodual(bulk_fill!), Ctx(), DifferReverse.CoDual(u, du), zero_fcodual(4.0))
    pbu(NoRData())
    @test du == [10.0, 20.0, 30.0]
    @test u == [1.0, 2.0, 3.0]

    # A bulk-saving callee reached recursively, called twice on the same array: the inner tape's own
    # buffer restores the array at each inner pullback, and the outer's at the end. d/da = 1 + 2 = 3.
    @noinline bulk_inner!(v::Vector{Float64}, a::Float64) =
        (for i in 1:length(v); @inbounds v[i] = a * i; end; v[1])
    bulk_outer(v::Vector{Float64}, a::Float64) = bulk_inner!(v, a) + bulk_inner!(v, 2a)
    vr = [3.0, 4.0]
    _, _, dar = rev_gradient(bulk_outer, vr, 2.0)
    @test dar ≈ 3.0
    @test dar ≈ central_diff(a -> bulk_outer(copy(vr), a), 2.0) rtol = 1e-5
    @test vr == [3.0, 4.0]
    checkverify_rev(bulk_outer, (Vector{Float64}, Float64))
    check_stack_balance(bulk_outer, vr, 2.0)
end

@testset "reverse mode: store into an undefined `Memory` slot" begin
    # A fresh array's `Core.memorynew` leaves every slot undefined, so `memoryrefset!`'s read of the
    # value it overwrites threw `UndefRefError` for a non-`isbits` element.
    nested_lit(x) = (v = [[x]]; v[1][1] * 2.0)
    _, dx_n = rev_gradient(nested_lit, 1.5)
    @test dx_n ≈ 2.0
    @test dx_n ≈ central_diff(nested_lit, 1.5) rtol = 1e-5
    checkverify_rev(nested_lit, (Float64,))
    check_stack_balance(nested_lit, 1.5)

    # Inner array named first, so the stored value has its own tracked shadow.
    nested_named(x) = (a = [x]; v = [a]; v[1][1] * 2.0)
    _, dx_nn = rev_gradient(nested_named, 1.5)
    @test dx_nn ≈ 2.0
    checkverify_rev(nested_named, (Float64,))
    check_stack_balance(nested_named, 1.5)

    # The undefined-slot path must not disturb the other slot.
    nested_two(x) = (v = Vector{Float64}[[x], [2x]]; v[2][1] * 2.0)
    _, dx_n2 = rev_gradient(nested_two, 1.5)
    @test dx_n2 ≈ 4.0
    @test dx_n2 ≈ central_diff(nested_two, 1.5) rtol = 1e-5
    checkverify_rev(nested_two, (Float64,))
    check_stack_balance(nested_two, 1.5)

    # Explicitly `undef`-allocated: both slots start undefined.
    function undef_fill(x)
        v = Vector{Vector{Float64}}(undef, 2)
        v[1] = [x]
        v[2] = [2x]
        return v[1][1] * v[2][1]
    end
    _, dx_uf = rev_gradient(undef_fill, 1.5)
    @test dx_uf ≈ 6.0
    @test dx_uf ≈ central_diff(undef_fill, 1.5) rtol = 1e-5
    checkverify_rev(undef_fill, (Float64,))
    check_stack_balance(undef_fill, 1.5)

    # `isbits` control: keeps the plain path.
    bits_two(x) = (v = [x, 2x]; v[1] * v[2])
    _, dx_b = rev_gradient(bits_two, 1.5)
    @test dx_b ≈ 6.0
    checkverify_rev(bits_two, (Float64,))
    check_stack_balance(bits_two, 1.5)
end
