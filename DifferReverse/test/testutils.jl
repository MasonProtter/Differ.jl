# Shared test infrastructure: IR-verification wrappers and tape-hygiene checks reused across
# many test files. Not test fixtures — nothing here is itself differentiated. The bulk of
# forward-mode-only helpers (checkverify2, bail_reason) live in DifferForwards.jl/test/testutils.jl
# instead — `checkverify` is duplicated here (one line) because a couple of reverse-mode test
# files cross-check a reverse-mode bail against forward mode still working fine on the same
# primal, needing DifferForwards' `code_dual_ircode` as a test-only dependency (see test/Project.toml).

using Test
using DifferReverse
using DifferReverse: code_reverse_fwds_ircode, code_reverse_pullback_ircode
using DifferReverse: build_ctx, rrule!!, rev_gradient, rev_gradient!, zero_fcodual
using DifferReverse: tape_type, comms_element_types
using DifferForwards: code_dual_ircode

# Forward-mode dualized IR is legal (order-1). Duplicated from DifferForwards.jl/test/testutils.jl
# — see the file header.
checkverify(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at)[1])

# Central finite difference, one argument.
central_diff(f, x; h=1e-6) = (f(x + h) - f(x - h)) / 2h

# Central finite difference of a 2-argument function, w.r.t. argument `k` (1 or 2).
central_diff(f, x, y, k::Int; h=1e-6) =
    k == 1 ? (f(x+h, y) - f(x-h, y)) / 2h : (f(x, y+h) - f(x, y-h)) / 2h

# `code_reverse_fwds_ircode`/`code_reverse_pullback_ircode` inspect the tape-*allocating* carrier
# shape (`Ctx{Nothing}`). A `build_ctx(...; prealloc=true)` context compiles a *different*
# prologue — one that reads the caller's stacks out of the ctx and resets them instead of
# constructing them — so it needs its own check, or the pre-allocated path goes unchecked.
function checkverify_prealloc(f, at)
    ctx = build_ctx(f, at)
    interp = DifferReverse.build_reverse_interp()
    tt = Tuple{typeof(DifferReverse.reverse_fwds_impl),
               DifferReverse.fcodual_type(DifferReverse._typeof(f)), typeof(ctx),
               (DifferReverse.fcodual_type(T) for T in at)...}
    mi = Base.specialize_method(
        Core.Compiler.findall(tt, Core.Compiler.method_table(interp))[1])
    reason = Ref("no specific reason recorded")
    ir = DifferReverse.optimized_reverse_fwds_ir(interp, mi, reason)
    @test ir !== nothing || error("pre-allocated carrier bailed for $f: $(reason[])")
    Core.Compiler.verify_ir(ir)
end

function checkverify_rev(f, at)
    Core.Compiler.verify_ir(code_reverse_fwds_ircode(f, at)[1])
    Core.Compiler.verify_ir(code_reverse_pullback_ircode(f, at)[1])
    checkverify_prealloc(f, at)
end

# Recursively assert every `Tape` reachable from a tape's own comms is stack-balanced: its
# `block_stack` and every `Stack`-backed comms slot back at position 0. Walks into a `Stack`'s
# *whole* backing memory (not just up to `position`, which is already back at 0 by the time this
# runs) since a `Stack` never shrinks — a slot from an earlier, larger call still holds a real
# (possibly recycled) tape. Load-bearing for the nested-tape-recycling plan (`_inner_ctx`,
# `src/stack.jl`): a recycled inner tape left unbalanced by its own call would show up here as a
# stale nonzero position on *that* tape, not just on the outer one `check_stack_balance` used to
# check alone.
function _assert_tape_balanced(tape::DifferReverse.Tape, seen::Base.IdSet{Any}=Base.IdSet{Any}())
    tape in seen && return nothing
    push!(seen, tape)
    @test tape.block_stack.position == 0
    for s in tape.comms
        _assert_comms_balanced(s, seen)
    end
    # Direct self-recursion's dedicated storage (`Tape.subtapes`): same balance/recycling invariant
    # as an ordinary comms-embedded inner tape, just reached through its own field instead of a
    # `(:subtape, ssa)` comms item.
    @test tape.subtapes.position == 0
    for i in eachindex(tape.subtapes.memory)
        isassigned(tape.subtapes.memory, i) && _assert_tape_balanced(tape.subtapes.memory[i], seen)
    end
    return nothing
end
function _assert_comms_balanced(s::DifferReverse.Stack, seen)
    @test s.position == 0
    for i in eachindex(s.memory)
        isassigned(s.memory, i) && _assert_tuple_balanced(s.memory[i], seen)
    end
end
_assert_comms_balanced(::DifferReverse.SingletonStack, seen) = nothing
_assert_comms_balanced(s::DifferReverse.CommsCell, seen) = isdefined(s, :val) ? _assert_tuple_balanced(s.val, seen) : nothing
_assert_tuple_balanced(t::Tuple, seen) = foreach(v -> v isa DifferReverse.Tape && _assert_tape_balanced(v, seen), t)

# Phase D (unique-predecessor optimization): every push must still be matched by exactly one pop
# across a full rule+pullback round trip. The pullback *is* the tape, so this just calls it and
# confirms every `Stack`'s `position` (block stack, and every non-singleton per-block comms stack,
# recursively into any recycled inner tape) is back to 0.
#
# Doubly load-bearing since a `build_ctx(...; prealloc=true)` context *reuses* its tape across
# calls: balance is what makes reuse correct, so this also runs each case twice through one
# pre-allocated context and checks the answers agree.
#
# `seed` overrides the default `one(y)` seed, for a primal whose result has no `one` — a
# tuple-valued `f` (`test_reverse_tuples.jl`). That also skips the pre-allocated half below, which
# reaches the pullback through `rev_gradient!` and so seeds with `one(y)` itself.
function check_stack_balance(f, args...; seed=nothing)
    ctx = build_ctx(f, map(DifferReverse._typeof, args); prealloc=false)
    fcd, argcds = zero_fcodual(f), map(zero_fcodual, args)
    ycd, pb = rrule!!(fcd, ctx, argcds...)
    pb(seed === nothing ? one(DifferReverse.primal(ycd)) : seed)
    _assert_tape_balanced(pb)
    seed === nothing || return nothing

    # Same again through a pre-allocated (tape-reusing) context, twice.
    pctx = build_ctx(f, map(DifferReverse._typeof, args))
    g1 = rev_gradient!(pctx, zero_fcodual(f), map(zero_fcodual, args)...)
    g2 = rev_gradient!(pctx, zero_fcodual(f), map(zero_fcodual, args)...)
    @test g1 == g2
    @test g1 == rev_gradient(f, args...)
    _assert_tape_balanced(pctx.tape)
    return nothing
end

# Collapsible-region optimization (`_collapsible_regions`, `src/reverse_interp.jl`): asserts whether
# a fresh (non-preallocated) round trip ever actually grew the block stack's backing memory —
# `length(pb.block_stack.memory)`, not `.position` (already asserted back to 0 by
# `check_stack_balance`; a Stack never shrinks its backing `Vector`, so peak usage is what this
# checks). `expect_zero=true` is the "genuinely straight-line, modulo `@boundscheck`" claim: every
# array access in `f` collapsed to zero block-stack traffic. `expect_zero=false` documents the
# opposite for a case that must keep paying it (a real data-dependent branch or loop) — passing
# `false` here is itself a regression guard against the optimization over-firing.
function check_block_stack_traffic(f, args...; expect_zero::Bool)
    ctx = build_ctx(f, map(DifferReverse._typeof, args); prealloc=false)
    fcd, argcds = zero_fcodual(f), map(zero_fcodual, args)
    ycd, pb = rrule!!(fcd, ctx, argcds...)
    pb(one(DifferReverse.primal(ycd)))
    grew = length(pb.block_stack.memory) > 0
    @test grew == !expect_zero
    return nothing
end

# Tape size. Asserts on properties of the *whole* set of comms element types (their total size, and
# whether they are all `isbits` — i.e. whether the comms stacks are flat buffers or GC-tracked ones
# needing a write barrier per push), never on a particular block's index: block numbering shifts
# with any unrelated change to Julia's optimizer.
#
# `bytes` is the sum over comms stacks. Pass `isbits=true` to additionally require every stack to be
# pointer-free.
#
# `stacks` is the number of comms stacks — distinct from `bytes` because comms fusion
# (`_scan_block_comms`) can merge two stacks' values onto one without changing the byte total, only
# the push/pop count.
function check_tape_size(f, at; bytes::Union{Int,Nothing}=nothing, isbits::Union{Bool,Nothing}=nothing,
                         stacks::Union{Int,Nothing}=nothing)
    ts = comms_element_types(tape_type(f, at))
    isbits === nothing || @test all(isbitstype, ts) == isbits
    bytes === nothing || @test sum(sizeof, ts; init=0) == bytes
    stacks === nothing || @test length(ts) == stacks
    return ts
end
