# Hand-written reverse-mode primitive rules — methods of `rrule!!`. A hand-written rule for a callee
# overrides the derived (IR-transformed) path via ordinary dispatch: the recursion glue
# (`reverse_fwds_recursive_ci` in `reverse_interp.jl`) checks for an applicable `rrule!!` method
# first and only builds a derived rule when there isn't one.
#
# A rule returns `(ycd, pullback)`; the pullback closure IS the tape for the derived path, but a
# hand rule's pullback can be any opaque callable (never inspected, only called) — cheaper than the
# `Stack`-based `Tape{ArgsTT,CS}` the derived path needs to handle arbitrary control flow. Each rule
# gets its own pullback type so dispatch can't collide between rules remembering the same primal
# value type.
#
# `Base.sin`/`Base.cos` are qualified because these bodies get inlined into synthetic carrier IR,
# where a bare name would re-embed as `GlobalRef(Differ, :sin)` — an unbound GlobalRef `verify_ir`
# rejects.

# A hand rule's `::AbstractCtx` slot (never a concrete `Ctx{...}`) is what keeps it unambiguous
# against the `@generated` derived fallback under dispatch — see `reverse_interp.jl`.

function rrule!!(::CoDual{typeof(sin),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    sin_pullback(dy) = (NoRData(), Base.cos(x) * dy)
    CoDual(Base.sin(x), NoFData()), sin_pullback
end

function rrule!!(::CoDual{typeof(cos),NoFData}, ::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    cos_pullback(dy) = (NoRData(), -Base.sin(x) * dy)
    CoDual(Base.cos(x), NoFData()), cos_pullback
end


# `sum`/`sum(f, ·)` moved to `rules_perf_backstop.jl` (not included by `DifferReverse.jl`) — an
# efficient fallback only, not needed for correctness: the derived path composes through Base's
# `mapreduce` machinery correctly at any array size.
