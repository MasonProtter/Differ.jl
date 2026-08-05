# Shared test infrastructure: IR-verification wrappers and finite-difference/tape-hygiene checks
# reused across many test files. Not test fixtures — nothing here is itself differentiated.

using Test
using Differ
using Differ: code_dual_ircode, code_reverse_fwds_ircode, code_reverse_pullback_ircode
using Differ: build_ctx, rrule!!, gradient, gradient!, zero_fcodual
using Differ: tape_type, comms_element_types

# Central finite difference, one argument.
central_diff(f, x; h=1e-6) = (f(x + h) - f(x - h)) / 2h

# Central finite difference of a 2-argument function, w.r.t. argument `k` (1 or 2).
central_diff(f, x, y, k::Int; h=1e-6) =
    k == 1 ? (f(x+h, y) - f(x-h, y)) / 2h : (f(x, y+h) - f(x, y-h)) / 2h

# Forward-mode dualized IR is legal (order-1).
checkverify(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at)[1])

# The bail message for a function Differ declines to dualize, or `nothing` if it dualizes fine.
# Every graceful bail is supposed to name a *reason*, so tests assert on this rather than just on
# "it threw".
function bail_reason(f, at)
    try
        code_dual_ircode(f, at)
        return nothing
    catch e
        return sprint(showerror, e)
    end
end

# Forward-mode dualized IR is legal at a given nesting order.
checkverify2(f, at; order=2) = Core.Compiler.verify_ir(code_dual_ircode(f, at; order)[1])

# `code_reverse_fwds_ircode`/`code_reverse_pullback_ircode` inspect the tape-*allocating* carrier
# shape (`Ctx{Nothing}`). A `build_ctx(...; prealloc=true)` context compiles a *different*
# prologue — one that reads the caller's stacks out of the ctx and resets them instead of
# constructing them — so it needs its own check, or the pre-allocated path goes unchecked.
function checkverify_prealloc(f, at)
    ctx = build_ctx(f, at)
    interp = Differ.ADInterpreter{Differ.Reverse}()
    tt = Tuple{typeof(Differ.reverse_fwds_impl),
               Differ.fcodual_type(Differ._typeof(f)), typeof(ctx),
               (Differ.fcodual_type(T) for T in at)...}
    mi = Base.specialize_method(
        Core.Compiler.findall(tt, Core.Compiler.method_table(interp))[1])
    reason = Ref("no specific reason recorded")
    ir = Differ.optimized_reverse_fwds_ir(interp, mi, reason)
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
function _assert_tape_balanced(tape::Differ.Tape, seen::Base.IdSet{Any}=Base.IdSet{Any}())
    tape in seen && return nothing
    push!(seen, tape)
    @test tape.block_stack.position == 0
    for s in tape.comms
        _assert_comms_balanced(s, seen)
    end
    return nothing
end
function _assert_comms_balanced(s::Differ.Stack, seen)
    @test s.position == 0
    for i in eachindex(s.memory)
        isassigned(s.memory, i) && _assert_tuple_balanced(s.memory[i], seen)
    end
end
_assert_comms_balanced(::Differ.SingletonStack, seen) = nothing
_assert_comms_balanced(s::Differ.CommsCell, seen) = isdefined(s, :val) ? _assert_tuple_balanced(s.val, seen) : nothing
_assert_tuple_balanced(t::Tuple, seen) = foreach(v -> v isa Differ.Tape && _assert_tape_balanced(v, seen), t)

# Phase D (unique-predecessor optimization): every push must still be matched by exactly one pop
# across a full rule+pullback round trip. The pullback *is* the tape, so this just calls it and
# confirms every `Stack`'s `position` (block stack, and every non-singleton per-block comms stack,
# recursively into any recycled inner tape) is back to 0.
#
# Doubly load-bearing since a `build_ctx(...; prealloc=true)` context *reuses* its tape across
# calls: balance is what makes reuse correct, so this also runs each case twice through one
# pre-allocated context and checks the answers agree.
function check_stack_balance(f, args...)
    ctx = build_ctx(f, map(Differ._typeof, args); prealloc=false)
    fcd, argcds = zero_fcodual(f), map(zero_fcodual, args)
    ycd, pb = rrule!!(fcd, ctx, argcds...)
    pb(one(Differ.primal(ycd)))
    _assert_tape_balanced(pb)

    # Same again through a pre-allocated (tape-reusing) context, twice.
    pctx = build_ctx(f, map(Differ._typeof, args))
    g1 = gradient!(pctx, zero_fcodual(f), map(zero_fcodual, args)...)
    g2 = gradient!(pctx, zero_fcodual(f), map(zero_fcodual, args)...)
    @test g1 == g2
    @test g1 == gradient(f, args...)
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
    ctx = build_ctx(f, map(Differ._typeof, args); prealloc=false)
    fcd, argcds = zero_fcodual(f), map(zero_fcodual, args)
    ycd, pb = rrule!!(fcd, ctx, argcds...)
    pb(one(Differ.primal(ycd)))
    grew = length(pb.block_stack.memory) > 0
    @test grew == !expect_zero
    return nothing
end

# Tape size. Asserts on properties of the *whole* set of comms element types (their total size, and
# whether they are all `isbits` — i.e. whether the comms stacks are flat buffers or GC-tracked ones
# needing a write barrier per push), never on a particular block's index: block numbering shifts
# with any unrelated change to Julia's optimizer, but "how many bytes per loop iteration, and does
# pushing them touch the GC" is exactly what these optimizations are about.
#
# `bytes` is the sum over comms stacks. Pass `isbits=true` to additionally require every stack to be
# pointer-free.
function check_tape_size(f, at; bytes::Union{Int,Nothing}=nothing, isbits::Union{Bool,Nothing}=nothing)
    ts = comms_element_types(tape_type(f, at))
    isbits === nothing || @test all(isbitstype, ts) == isbits
    bytes === nothing || @test sum(sizeof, ts; init=0) == bytes
    return ts
end
