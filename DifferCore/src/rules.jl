# The two rule generic functions (`frule!!`, `rrule!!`) and the reverse-mode context supertype
# (`AbstractCtx`), owned by DifferCore so both AD modes and shared package extensions can extend
# them.

"""
    frule!!(fdual::Dual, argduals::Dual...) -> Dual

Forward-mode rule for `primal(fdual)(primal.(argduals)...)`, returning the result and its
directional derivative together as a single `Dual`.

Hand-written primitives (see `src/rules_math.jl` for the shape to follow) are methods with a
specific `fdual`/`argduals` shape; a composite function is handled by an `@generated` fallback
that derives the rule from `f`'s IR, so `frule!!` works on anything.
"""
function frule!! end

"""
    rrule!!(fcd::CoDual, ctx::AbstractCtx, argcds::CoDual...) -> (ycd, pullback)

Reverse-mode rule for `primal(fcd)(primal.(argcds)...)`, returning the result as a `CoDual` plus a
pullback. Call the pullback with an rdata seed for the result to get the tuple of rdatas for
`(f, args...)`; fdata-carried gradients (arrays, mutable structs) are accumulated in place into the
`CoDual`s' own shadows instead.

Hand-written primitives (see `src/rrules.jl` for the `sin`/`cos` rules and the shape to follow) are
methods with a specific `fcd`/`argcds` shape; a composite function is handled by an `@generated`
fallback that derives the rule from `f`'s IR. `ctx::`[`AbstractCtx`](@ref) carries the tape (build a
reusable one with [`build_ctx`](@ref DifferReverse.build_ctx)); a hand rule that needs no tape
ignores it. A hand rule
**must** declare its `ctx` slot as `::AbstractCtx` (never a concrete subtype) — that's what keeps
dispatch against the fallback unambiguous.

Contract for an fdata-carrying result (an array or mutable struct): the returned `CoDual`'s shadow
must be the exact object the rule's own pullback reads back, never a detached copy — a caller may
accumulate into it in place before the pullback runs. Every hand-written array rule already
follows this (`rules_broadcast.jl`, `rules_reductions.jl`, `rules_indexing.jl`, `rules_linalg.jl`).
"""
function rrule!! end

"""
    AbstractCtx

Supertype of reverse-mode differentiation contexts — the argument [`rrule!!`](@ref) threads its
per-call/per-task state through, chiefly the tape. [`Ctx`](@ref DifferReverse.Ctx) is the default.
Every `rrule!!`
method dispatches this slot as `::AbstractCtx` (never a concrete subtype), which keeps the
derived-fallback-vs-hand-rule dispatch ambiguity-free.
"""
abstract type AbstractCtx end
