using Test
using DifferReverse
using DifferReverse: rev_gradient, value_and_gradient!, build_ctx, CoDual, Ctx, zero_fcodual,
    rrule!!, NoRData, Inactive, primal, tangent

include(joinpath(@__DIR__, "testutils.jl"))

# Every growable `Vector` operation funnels through Base's six growth helpers, each of which has a
# hand `rrule!!` in `rules_growable.jl`. The pullbacks put both carriers back on the ref and length
# they had on entry, which is what keeps comms-saved `MemoryRef` handles addressing live memory
# across a reallocation.

# Gradient by central differences, coordinate by coordinate, on a fresh copy per evaluation so a
# mutating `f` is safe.
function fd_gradient(f, x; h=1e-6)
    return [let e = [j == k ? h : 0.0 for j in eachindex(x)]
                (f(x .+ e) - f(x .- e)) / 2h
            end for k in eachindex(x)]
end

capacity(v::Vector) = length(v.ref.mem)
offset(v::Vector) = Base.memoryrefoffset(v.ref)

@testset "growable vectors: the family against finite differences" begin
    ops = Dict(
        "push!"      => v -> (push!(v, 4.0); sum(v)),
        "pop!"       => v -> (last = pop!(v); last * last),
        "pushfirst!" => v -> (pushfirst!(v, 4.0); sum(v)),
        "popfirst!"  => v -> (first = popfirst!(v); first * first),
        "insert!"    => v -> (insert!(v, 2, 4.0); sum(v)),
        "deleteat!"  => v -> (deleteat!(v, 2); sum(v)),
        "resize! (grow)"   => v -> (n = length(v); resize!(v, n + 2); v[n+1] = 1.0; v[n+2] = 2.0; sum(v)),
        "resize! (shrink)" => v -> (resize!(v, 2); sum(v)),
        "append!"    => v -> (append!(v, [4.0, 5.0]); sum(v)),
        "prepend!"   => v -> (prepend!(v, [4.0, 5.0]); sum(v)),
        "keepat!"    => v -> (keepat!(v, 1:2); sum(v)),
        # Read before emptying, so the gradient has to survive the size restore rather than being
        # trivially zero.
        "empty!"     => v -> (s = sum(v); empty!(v); s),
    )
    x = [1.0, 2.0, 3.0]
    @testset "$name" for (name, f) in ops
        _, g = rev_gradient(f, copy(x))
        @test g ≈ fd_gradient(f, x) rtol = 1e-4 atol = 1e-7
        checkverify_rev(f, (Vector{Float64},))
        check_stack_balance(f, copy(x))
    end
end

@testset "growable vectors: the growth helpers directly" begin
    # The rules are on `Base._growend!`/`_deleteend!`/… themselves; the family above reaches them
    # through Base's own bodies. Both paths need covering.
    v, dv = [1.0, 2.0, 3.0], [1.0, 10.0, 100.0]
    ycd, pb = rrule!!(zero_fcodual(Base._growend!), Ctx(), CoDual(v, dv), zero_fcodual(2))
    @test primal(ycd) === nothing
    @test length(v) == 5 && length(dv) == 5
    dv[4] = 7.0                                  # gradient landing in a slot that did not exist
    @test pb(NoRData()) == (NoRData(), NoRData(), NoRData())
    @test length(v) == 3 && length(dv) == 3      # both carriers rewound
    @test dv == [1.0, 10.0, 100.0]               # surviving gradients kept, the new slot dropped

    ycd, pb = rrule!!(zero_fcodual(Base._deleteend!), Ctx(), CoDual(v, dv), zero_fcodual(1))
    @test length(v) == 2 && length(dv) == 2
    pb(NoRData())
    @test v == [1.0, 2.0, 3.0] && dv == [1.0, 10.0, 100.0]

    ycd, pb = rrule!!(zero_fcodual(Base._deleteat!), Ctx(), CoDual(v, dv),
                      zero_fcodual(2), zero_fcodual(1))
    @test v == [1.0, 3.0] && dv == [1.0, 100.0]
    pb(NoRData())
    @test v == [1.0, 2.0, 3.0] && dv == [1.0, 10.0, 100.0]   # the cut slice is restored

    ycd, pb = rrule!!(zero_fcodual(Base._growat!), Ctx(), CoDual(v, dv),
                      zero_fcodual(2), zero_fcodual(1))
    @test length(v) == 4 && length(dv) == 4
    pb(NoRData())
    @test v == [1.0, 2.0, 3.0] && dv == [1.0, 10.0, 100.0]

    ycd, pb = rrule!!(zero_fcodual(Base._growbeg!), Ctx(), CoDual(v, dv), zero_fcodual(2))
    @test length(v) == 5 && length(dv) == 5
    pb(NoRData())
    @test v == [1.0, 2.0, 3.0] && dv == [1.0, 10.0, 100.0]

    ycd, pb = rrule!!(zero_fcodual(Base._deletebeg!), Ctx(), CoDual(v, dv), zero_fcodual(1))
    @test v == [2.0, 3.0] && dv == [10.0, 100.0]
    pb(NoRData())
    @test v == [1.0, 2.0, 3.0] && dv == [1.0, 10.0, 100.0]
end

@testset "growable vectors: gradient survives a reallocation" begin
    # A read taken before the array outgrows its memory saves a shadow handle into the memory the
    # grow abandons. Without the pullback putting the array back on that memory, the gradient
    # accumulated for the pre-growth read would be written somewhere nothing reads back.
    function readgrowread(v)
        a = v[1]
        for i in 1:16
            push!(v, float(i))
        end
        return a + v[1] + v[2]
    end
    x = [1.0, 2.0, 3.0]
    _, g = rev_gradient(readgrowread, copy(x))
    @test g ≈ fd_gradient(readgrowread, x) rtol = 1e-4 atol = 1e-7
    @test g ≈ [2.0, 1.0, 0.0]
    checkverify_rev(readgrowread, (Vector{Float64},))
    check_stack_balance(readgrowread, copy(x))

    # The same read reached through a merge, so the ref's provenance root is a `PhiNode` rather than
    # the argument itself — the shape whose shadow handle is saved verbatim instead of re-derived.
    function phiread(v)
        w = length(v) > 2 ? v : v
        a = w[1]
        for i in 1:16
            push!(v, float(i))
        end
        return a + w[1]
    end
    _, g = rev_gradient(phiread, copy(x))
    @test g ≈ fd_gradient(phiread, x) rtol = 1e-4 atol = 1e-7
    check_stack_balance(phiread, copy(x))

    # A locally-created accumulator grown past capacity several times.
    build(xs) = (acc = Float64[]; for y in xs; push!(acc, y * y); end; sum(acc))
    xs = collect(1.0:20.0)
    _, g = rev_gradient(build, copy(xs))
    @test g ≈ 2 .* xs
    checkverify_rev(build, (Vector{Float64},))
    check_stack_balance(build, copy(xs))
end

@testset "growable vectors: primal and shadow layouts may disagree" begin
    # A primal carrying spare capacity and an advanced offset, paired with a shadow allocated at
    # exactly its length. Each side resizes through its own layout. Passed without `copy`, which
    # compacts a vector back to capacity == length, offset 1 and would erase the divergence.
    x = Float64[]
    for i in 1:12
        push!(x, float(i))
    end
    for _ in 1:9
        popfirst!(x)
    end
    x .= [1.0, 2.0, 3.0]
    dx = zeros(3)
    @test capacity(x) > length(x) && offset(x) > 1
    @test capacity(dx) == length(dx) && offset(dx) == 1

    bothends(v) = (push!(v, 9.0); pushfirst!(v, 8.0); sum(v))
    ycd, pb = rrule!!(zero_fcodual(bothends), Ctx(), CoDual(x, dx))
    pb(1.0)
    @test dx ≈ [1.0, 1.0, 1.0]
    @test length(x) == 3 && offset(x) > 1        # rewound onto its original layout
end

@testset "growable vectors: slots reused after a shrink" begin
    # A slot dropped and rewritten must not leak the old value's gradient into the new one.
    poppush(v) = (pop!(v); push!(v, 7.0); sum(v))
    popfpushf(v) = (popfirst!(v); pushfirst!(v, 7.0); sum(v))
    x = [1.0, 2.0, 3.0]
    for f in (poppush, popfpushf)
        _, g = rev_gradient(f, copy(x))
        @test g ≈ fd_gradient(f, x) rtol = 1e-4 atol = 1e-7
        check_stack_balance(f, copy(x))
    end
    _, g = rev_gradient(poppush, copy(x))
    @test g ≈ [1.0, 1.0, 0.0]
    _, g = rev_gradient(popfpushf, copy(x))
    @test g ≈ [0.0, 1.0, 1.0]
end

@testset "growable vectors: a reused context restores the array" begin
    # The pullback puts the array back on its entry ref and length, so a second call through one
    # pre-allocated context sees the same state as the first.
    f(v) = (push!(v, 4.0); sum(v))
    ctx = build_ctx(f, (Vector{Float64},))
    v1, dv1 = [1.0, 2.0, 3.0], zeros(3)
    y1, = value_and_gradient!(ctx, zero_fcodual(f), CoDual(v1, dv1))
    @test length(v1) == 3
    v2, dv2 = [1.0, 2.0, 3.0], zeros(3)
    y2, = value_and_gradient!(ctx, zero_fcodual(f), CoDual(v2, dv2))
    @test y1 == y2 && dv1 == dv2 == [1.0, 1.0, 1.0]
end

@testset "growable vectors: sizehint! leaves the length alone" begin
    # Same divergence as forward mode: Base undoes `_growend!`'s length with a raw `setfield!` that
    # the shadow never sees. `sizehint!` can also reallocate the primal, so the pullback rewinds it.
    hint(v) = (sizehint!(v, 16); sum(v))
    hintpush(v) = (sizehint!(v, 64); for i in 1:20; push!(v, v[1] * i); end; sum(v))
    x = [1.0, 2.0, 3.0]
    for f in (hint, hintpush)
        _, g = rev_gradient(f, copy(x))
        @test g ≈ fd_gradient(f, x) rtol = 1e-4 atol = 1e-7
        check_stack_balance(f, copy(x))
    end

    ycd, pb = rrule!!(zero_fcodual(Base.sizehint!), Ctx(), const_codual([1.0, 2.0]), zero_fcodual(16))
    @test tangent(ycd) == [0.0, 0.0]      # a constant array's zero tangent, materialised
    @test pb(NoRData()) == (NoRData(), NoRData(), NoRData())
end

@testset "growable vectors: constant arrays" begin
    # A constant array's shadow is `Inactive()`; the helpers carry no value, so they leave it alone.
    grow(v) = (push!(v, 4.0); length(v) * 1.0)
    ycd, pb = rrule!!(zero_fcodual(grow), Ctx(), const_codual([1.0, 2.0]))
    @test primal(ycd) ≈ 3.0
    @test pb(1.0) == (NoRData(), NoRData())
end

@testset "growable vectors: splice! is still out of scope" begin
    # `splice!` reaches a `Vector{Any}` default argument whose fdata has no traceable provenance.
    # It must bail with a located reason rather than miscompile.
    cut!(v) = (splice!(v, 2); sum(v))
    err = try
        rev_gradient(cut!, [1.0, 2.0, 3.0])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("provenance", err.msg)
end
