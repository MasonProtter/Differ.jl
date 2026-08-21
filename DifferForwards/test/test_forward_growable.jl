using Test
using DifferForwards
using DifferForwards: Dual, NoTangent, Inactive, frule!!, primal, tangent, zero_tangent,
    code_dual_ircode

include(joinpath(@__DIR__, "testutils.jl"))

# Every growable `Vector` operation funnels through Base's six growth helpers, each of which has a
# hand `frule!!` in `rules_growable.jl`. Primal and shadow each grow through their own layout, so
# these tests deliberately include pairs whose capacity and offset disagree.

fdual(f) = Dual(f, NoTangent())

# Tangent of `f` at `v` along `dv`, on a fresh copy so the caller's arrays survive.
dirderiv(f, v, dv) = tangent(frule!!(fdual(f), Dual(copy(v), copy(dv))))

# The same without copying. `copy` compacts a vector — capacity back to the length, offset back to
# 1 — which erases exactly the states the layout and branch testsets below construct, so those must
# differentiate the arrays they built.
dirderiv!(f, v, dv) = tangent(frule!!(fdual(f), Dual(v, dv)))

capacity(v::Vector) = length(v.ref.mem)
offset(v::Vector) = Base.memoryrefoffset(v.ref)

# A vector carrying spare capacity and an advanced offset, grown then dropped from the front.
function slack_vector()
    v = Float64[]
    for i in 1:12
        push!(v, float(i))
    end
    for _ in 1:9
        popfirst!(v)
    end
    v .= [1.0, 2.0, 3.0]
    return v
end

# Longer, with front room: long enough for `_growat!` to reach both of its in-place branches
# (`slack_vector` is too short — an insert position can never satisfy `i <= div(len, 2)` there).
function wide_slack_vector()
    v = Float64[]
    for i in 1:20
        push!(v, float(i))
    end
    for _ in 1:10
        popfirst!(v)
    end
    return v
end

# A vector whose next `push!` exhausts its `Memory` while the offset is advanced far enough that
# `_growend_internal!` shuffles within that same `Memory` rather than allocating a new one.
function queue_vector()
    v = Float64[]
    for i in 1:22
        push!(v, float(i))
    end
    for _ in 1:20
        popfirst!(v)
    end
    for i in 1:12
        push!(v, float(i))
    end
    return v
end

@testset "growable vectors: the family against finite differences" begin
    ops = Dict(
        "push!"      => v -> (push!(v, 4.0); sum(v)),
        "pop!"       => v -> (x = pop!(v); x * x),
        "pushfirst!" => v -> (pushfirst!(v, 4.0); sum(v)),
        "popfirst!"  => v -> (x = popfirst!(v); x * x),
        "insert!"    => v -> (insert!(v, 2, 4.0); sum(v)),
        "deleteat!"  => v -> (deleteat!(v, 2); sum(v)),
        "resize! (grow)"   => v -> (n = length(v); resize!(v, n + 2); v[n+1] = 1.0; v[n+2] = 2.0; sum(v)),
        "resize! (shrink)" => v -> (resize!(v, 2); sum(v)),
        "empty!"     => v -> (empty!(v); length(v) * 1.0),
        "append!"    => v -> (append!(v, [4.0, 5.0]); sum(v)),
        "prepend!"   => v -> (prepend!(v, [4.0, 5.0]); sum(v)),
    )
    v = [1.0, 2.0, 3.0]
    dv = [1.0, 10.0, 100.0]
    @testset "$name" for (name, f) in ops
        @test dirderiv(f, v, dv) ≈ central_diff_dir(f, v, dv) rtol = 1e-5
        checkverify(f, (Vector{Float64},))
    end
end

@testset "growable vectors: the growth helpers directly" begin
    # The rules are on `Base._growend!`/`_deleteend!`/… themselves; the family above reaches them
    # through Base's own bodies. Both paths need covering.
    v, dv = [1.0, 2.0, 3.0], [1.0, 10.0, 100.0]
    r = frule!!(fdual(Base._growend!), Dual(v, dv), Dual(2, NoTangent()))
    @test primal(r) === nothing && tangent(r) === NoTangent()
    @test length(v) == 5 && length(dv) == 5
    @test dv[1:3] == [1.0, 10.0, 100.0]        # surviving tangents untouched

    frule!!(fdual(Base._deleteend!), Dual(v, dv), Dual(2, NoTangent()))
    @test length(v) == 3 && length(dv) == 3

    frule!!(fdual(Base._growbeg!), Dual(v, dv), Dual(2, NoTangent()))
    @test length(v) == 5 && length(dv) == 5
    @test dv[3:5] == [1.0, 10.0, 100.0]

    frule!!(fdual(Base._deletebeg!), Dual(v, dv), Dual(2, NoTangent()))
    @test dv == [1.0, 10.0, 100.0]

    frule!!(fdual(Base._growat!), Dual(v, dv), Dual(2, NoTangent()), Dual(1, NoTangent()))
    @test length(v) == 4 && length(dv) == 4
    @test dv[1] == 1.0 && dv[3:4] == [10.0, 100.0]

    frule!!(fdual(Base._deleteat!), Dual(v, dv), Dual(2, NoTangent()), Dual(1, NoTangent()))
    @test dv == [1.0, 10.0, 100.0]
end

@testset "growable vectors: shadow length tracks the primal" begin
    # Before these rules existed the shadow array's own `:size` was never written, so a `pop!` left
    # `length(tangent) != length(primal)`.
    twopop(v) = (pop!(v); pop!(v); sum(v))
    v, dv = [1.0, 2.0, 3.0, 4.0], [1.0, 10.0, 100.0, 1000.0]
    r = frule!!(fdual(twopop), Dual(v, dv))
    @test length(v) == 2 && length(dv) == 2
    @test tangent(r) ≈ 11.0
end

@testset "growable vectors: primal and shadow layouts may disagree" begin
    # A primal carrying spare capacity and an advanced offset, paired with a shadow fresh from
    # `zero_tangent` (capacity == length, offset 1). Each side resizes through its own layout, so
    # the mismatch is not observable. Differentiated without `copy`, which would compact both back
    # into agreement and make this pass for the wrong reason.
    v = slack_vector()
    dv = zero_tangent(v)
    dv .= [1.0, 10.0, 100.0]
    @test capacity(v) > length(v) && offset(v) > 1
    @test capacity(dv) == length(dv) && offset(dv) == 1

    bothends(x) = (push!(x, 9.0); pushfirst!(x, 8.0); sum(x))
    @test dirderiv!(bothends, v, dv) ≈
          central_diff_dir(bothends, slack_vector(), [1.0, 10.0, 100.0]) rtol = 1e-5
end

@testset "growable vectors: growth-helper internal branches" begin
    # Base decides between reallocating and moving within the existing `Memory` from the primal's
    # own capacity and offset. Each state is built explicitly, the branch it takes is pinned by
    # watching a twin run, and the same state is then differentiated — without `copy`, which would
    # compact the state away before the rule ever saw it.
    #
    # `check(reused_memory, offset_delta)` names the branch: reallocation gives `reused == false`,
    # an in-place move keeps the `Memory` and usually shifts the offset.
    function check_branch(f, build, check)
        twin = build()
        mem, off0 = twin.ref.mem, offset(twin)
        f(twin)
        @test check(twin.ref.mem === mem, offset(twin) - off0)

        v = build()
        dv = collect(1.0:length(v))
        @test dirderiv!(f, v, copy(dv)) ≈ central_diff_dir(f, build(), dv) rtol = 1e-5
    end

    tight() = [1.0, 2.0, 3.0]
    # No spare capacity: the grow reallocates.
    check_branch(v -> (push!(v, 9.0); sum(v)), tight, (reused, _) -> !reused)
    # Offset far enough advanced that `_growend_internal!` shuffles in place instead.
    check_branch(v -> (push!(v, 9.0); sum(v)), queue_vector, (reused, doff) -> reused && doff < 0)
    # `_growbeg!`'s fast path: enough front room to just move the offset back.
    check_branch(v -> (pushfirst!(v, 9.0); sum(v)), slack_vector, (reused, doff) -> reused && doff < 0)
    # `_growat!` shifts whichever side is shorter — front for a low index, tail for a high one — and
    # reallocates when the `Memory` is full.
    check_branch(v -> (insert!(v, 3, 0.5); sum(v)), wide_slack_vector,
                 (reused, doff) -> reused && doff < 0)
    check_branch(v -> (insert!(v, 9, 0.5); sum(v)), wide_slack_vector,
                 (reused, doff) -> reused && doff == 0)
    check_branch(v -> (insert!(v, 2, 0.5); sum(v)), tight, (reused, _) -> !reused)
end

@testset "growable vectors: growth across reallocations" begin
    # A locally-created accumulator grown past capacity several times: every element's tangent must
    # survive each reallocation.
    build(xs) = (acc = Float64[]; for x in xs; push!(acc, x * x); end; sum(acc))
    xs = collect(1.0:20.0)
    dxs = ones(20)
    @test dirderiv(build, xs, dxs) ≈ sum(2 .* xs)
    @test dirderiv(build, xs, dxs) ≈ central_diff_dir(build, xs, dxs) rtol = 1e-5
    checkverify(build, (Vector{Float64},))

    # Read an element, grow past capacity, read it again: the pre-growth read and the post-growth
    # read must agree about which tangent they saw.
    function readgrowread(v)
        a = v[1]
        for i in 1:16
            push!(v, float(i))
        end
        return a + v[1] + v[2]
    end
    v, dv = [1.0, 2.0, 3.0], [1.0, 10.0, 100.0]
    @test dirderiv(readgrowread, v, dv) ≈ central_diff_dir(readgrowread, v, dv) rtol = 1e-5
end

@testset "growable vectors: slots reused after a shrink" begin
    # A slot written, dropped and written again must not carry its old tangent into the new value.
    poppush(v) = (pop!(v); push!(v, 7.0); sum(v))
    popfpushf(v) = (popfirst!(v); pushfirst!(v, 7.0); sum(v))
    v, dv = [1.0, 2.0, 3.0], [1.0, 10.0, 100.0]
    for f in (poppush, popfpushf)
        @test dirderiv(f, v, dv) ≈ central_diff_dir(f, v, dv) rtol = 1e-5
    end
    # Both drop one element and replace it with a constant, so only the two survivors contribute.
    @test dirderiv(poppush, v, dv) ≈ 11.0
    @test dirderiv(popfpushf, v, dv) ≈ 110.0
end

@testset "growable vectors: sizehint! leaves the length alone" begin
    # Base implements the grow case as `_growend!` plus a raw `setfield!` undoing the length, and
    # that undo is not mirrored onto the shadow — so without a rule the shadow is left at the
    # hinted length while the primal keeps its own.
    hint(v) = (sizehint!(v, 16); sum(v))
    v, dv = [1.0, 2.0, 3.0], [1.0, 10.0, 100.0]
    r = frule!!(fdual(hint), Dual(v, dv))
    @test length(v) == 3 && length(dv) == 3
    @test tangent(r) ≈ 111.0

    hintpush(v) = (sizehint!(v, 16); push!(v, 4.0); sum(v))
    @test dirderiv(hintpush, v, dv) ≈ central_diff_dir(hintpush, v, dv) rtol = 1e-5

    # A constant array's tangent is genuinely zero; nothing flows in, so it is materialised rather
    # than refused.
    r = frule!!(fdual(hint), Dual([1.0, 2.0, 3.0], Inactive()))
    @test tangent(r) ≈ 0.0
end

@testset "growable vectors: constant arrays" begin
    # A constant array's shadow is `Inactive()`. The helpers carry no value, so they leave it alone.
    grow(v, y) = (push!(v, y); length(v) * 1.0)
    r = frule!!(fdual(grow), Dual([1.0, 2.0], Inactive()), Dual(3.0, Inactive()))
    @test primal(r) ≈ 3.0 && tangent(r) ≈ 0.0

    # An *active* value written into a constant array is still refused — by the element write, where
    # the derivative would actually be dropped, not by the grow.
    growsum(v, y) = (push!(v, y); sum(v))
    err = try
        frule!!(fdual(growsum), Dual([1.0, 2.0], Inactive()), Dual(3.0, 1.0))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("declared constant", err.msg)
end
