# Hand-written reverse-mode *primitive* rules — methods of `rrule!!`. Mirrors `frules.jl` for forward
# mode: a hand-written rule for a callee overrides the derived (IR-transformed) path via ordinary
# multiple dispatch, since the recursion glue in `reverse_interp.jl` (`reverse_fwds_recursive_ci`)
# looks for an applicable `rrule!!` method first and only builds a derived rule when there isn't one.
# Same as `frule!!(::Dual{typeof(sin)}, ...)` overriding forward mode's generated fallback.
#
# A rule returns `(ycd, pullback)`, and **the pullback closure is the tape** — there's no separate
# tape value. It's opaque to the glue in `reverse_interp.jl` (never inspected, only threaded through
# `:invoke`s and called), so unlike the derived path — always a `Stack`-based `Tape{ArgsTT,CS}` sized
# for arbitrary control flow — a hand rule can remember whatever's cheapest. Each rule gets its own
# pullback type (not a bare primal type like `Float64`) so pullback dispatch, keyed on that type,
# can't collide between unrelated rules remembering the same primal value.
#
# `Base.sin`/`Base.cos` are fully qualified below because these bodies get inlined into synthetic
# carrier IR, where a bare name resolves against *this* module and re-embeds as
# `GlobalRef(Differ, :sin)` — an implicit `using Base` binding that `Core.Compiler.verify_ir` rejects
# in value position. The emitted pullback-recursion `:invoke` is also flagged `IR_FLAG_NOINLINE` for
# the same reason (see `reverse_pullback_recursive_ci`); qualifying here is a second, redundant
# safeguard against the same problem.

# The `::AbstractCtx` slot: a hand rule takes the differentiation context as its second argument
# (right after `fcd`), matching the uniform `rrule!!(fcd, ctx, argcds...)` convention. A primitive
# like `sin` needs no tape and ignores the context, but the slot **must** be declared `::AbstractCtx`
# (never a concrete `Ctx{…}`) — that dispatch-neutrality is what keeps a hand rule unambiguous against
# the `@generated` derived fallback (see the note above the type definitions in `reverse_interp.jl`).

struct SinPullback
    x::Float64
end
(pb::SinPullback)(seed::Float64) = (NoRData(), Base.cos(pb.x) * seed)

function rrule!!(::CoDual{typeof(sin),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.sin(x), NoFData()), SinPullback(x)
end

struct CosPullback
    x::Float64
end
(pb::CosPullback)(seed::Float64) = (NoRData(), -Base.sin(pb.x) * seed)

function rrule!!(::CoDual{typeof(cos),NoFData}, ::AbstractCtx, xcd::CoDual{Float64,NoFData})
    x = primal(xcd)
    return CoDual(Base.cos(x), NoFData()), CosPullback(x)
end


# `sum`/`sum(f, ·)` moved to `src/rules_perf_backstop.jl` (not included by `src/Differ.jl`) — see that
# file's header. They're a known-efficient fallback only, not needed for correctness: the derived path
# now composes through Base's `mapreduce`/`_mapreduce`/`mapreduce_impl` machinery correctly at any
# array size (ISSUES #65).
