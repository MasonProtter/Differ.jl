# Hand-written reverse-mode *primitive* rules — methods of `rrule!!`. Mirrors `frules.jl` for forward
# mode: a hand-written rule for a specific callee overrides the derived (IR-transformed) path through
# ordinary Julia multiple dispatch, because the recursion glue in `reverse_interp.jl`
# (`reverse_fwds_recursive_ci`) looks for an applicable `rrule!!` method *first* and only builds a
# derived rule when there isn't one. Exactly like `frule!!(::Dual{typeof(sin)}, ...)` overriding
# forward mode's generated composite fallback.
#
# A rule returns `(ycd, pullback)`, and **the pullback closure is the tape** — there is no separate
# tape value. It is entirely opaque to the glue in `reverse_interp.jl` (never inspected, only threaded
# through `:invoke`s and then called), so unlike the derived path — always a `Stack`-based
# `Tape{ArgsTT,CS}`, sized for arbitrary control flow — a hand rule is free to remember whatever is
# cheapest. Each rule gets its own dedicated pullback type (rather than reusing a bare primal type
# like `Float64`) so pullback dispatch, keyed on that type, can't collide between unrelated rules that
# happen to need the same primal value remembered.
#
# NOTE on fully-qualified `Base.sin`/`Base.cos` below: these bodies get inlined into synthetic carrier
# IR, where a bare name resolves against *this* module and re-embeds as `GlobalRef(Differ, :sin)` — an
# implicit `using Base` binding that `Core.Compiler.verify_ir` rejects in value position. The emitted
# pullback-recursion `:invoke` is additionally flagged `IR_FLAG_NOINLINE` for the same reason (see
# `reverse_pullback_recursive_ci`); qualifying here is a second, redundant safeguard against the
# same problem.

# NOTE on the `::AbstractCtx` slot: a hand rule takes the differentiation context as its second
# argument (right after `fcd`), matching the uniform `rrule!!(fcd, ctx, argcds...)` convention. A
# primitive like `sin` needs no tape, so it ignores the context — but the slot **must** be declared
# `::AbstractCtx` (never a concrete `Ctx{…}`): that dispatch-neutrality is what keeps a hand rule
# unambiguous against the `@generated` derived fallback (see the note above the type definitions in
# `reverse_interp.jl`).

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

# NOTE on `sum`/`sum(f, ·)`: these exist for the same reason Mooncake hand-writes them
# (`src/rules/performance_patches.jl`) rather than relying on generic recursion through Base's own
# `mapreduce`/`_mapreduce`/`mapreduce_impl` machinery — that internal implementation is a
# performance-tuned, *self-recursive* (divide-and-conquer, pairwise) algorithm for large arrays, and
# Differ's reverse-mode recursion engine has a permanent cycle guard that bails on genuine
# self-recursion (no finite `Tape` type exists for it here). Recursing generically into `Base`'s
# private `_mapreduce` internals therefore only works below `Base.pairwise_blocksize` elements (1024
# for `+`); these hand rules make `sum`/`sum(f, x)` correct and efficient at any array size, by never
# touching that internal implementation at all — `has_hand_reverse_rule`/`src_inlining_policy`
# (`reverse_interp.jl`) keep a hand-ruled call from being inlined away, so `sum(f, v)` survives in the
# primal IR as itself rather than as whatever Base's own optimizer would otherwise reduce it to.

struct SumPullback{Dx<:Array}
    dx::Dx
end
function (pb::SumPullback)(seed)
    dx = pb.dx
    for i in eachindex(dx)
        dx[i] = increment!!(dx[i], seed)
    end
    return (NoRData(), NoRData())
end

function rrule!!(
    ::CoDual{typeof(sum),NoFData}, ::AbstractCtx, xcd::CoDual{X,X}
) where {X<:Array{<:IEEEFloat}}
    x = primal(xcd)
    dx = tangent(xcd)
    s = zero(eltype(x))
    for xi in x
        s += xi
    end
    return zero_fcodual(s), SumPullback(dx)
end

# `sum(f, x)` — what `sum(v) do vi ... end` desugars to. `G`/`FG` are left unconstrained (rather than
# forced to `NoFData`, unlike the derived recursion path's own callee guard) since a hand rule is
# free to accept a closure with real differentiable captures directly.
struct SumMapPullback{G,PB,Dx<:Array}
    pbs::Vector{PB}
    dx::Dx
end
function (pb::SumMapPullback{G})(seed) where {G}
    pbs = pb.pbs
    dx = pb.dx
    # `zero_like_rdata_from_type`, not `zero_rdata_from_type`: `G` is normally concrete (an ordinary
    # dynamic dispatch to this hand rule always binds `G` to the closure's actual runtime type), but
    # the derived recursion glue (`reverse_fwds_recursive_ci` in `reverse_interp.jl`) can resolve a
    # hand rule via a *static* call-site type that isn't concrete (e.g. `g` reached through an
    # abstractly-typed field/container) — that binds `G` to that same non-concrete type here, and
    # `zero_rdata_from_type` would return the `CannotProduceZeroRDataFromType` sentinel instead of
    # throwing outright, which `increment!!` below would then choke on. `grdata` only ever flows into
    # `increment!!` (already `ZeroRData`-aware) or straight back out in the returned tuple (routed by
    # the caller, also `ZeroRData`-aware), so no further instantiation is needed here.
    grdata = zero_like_rdata_from_type(G)
    for i in length(pbs):-1:1
        gi_r, xi_r = pbs[i](seed)
        grdata = increment!!(grdata, gi_r)
        dx[i] = increment!!(dx[i], xi_r)
    end
    return (NoRData(), grdata, NoRData())
end

function rrule!!(
    ::CoDual{typeof(sum),NoFData}, ::AbstractCtx, gcd::CoDual{G,FG}, xcd::CoDual{X,X}
) where {G,FG,X<:Array{<:IEEEFloat}}
    x = primal(xcd)
    dx = tangent(xcd)
    n = length(x)
    n == 0 && error("Differ: sum(f, x) over an empty array is not supported by this rule")
    y1, pb1 = rrule!!(gcd, Ctx(), CoDual(x[1], NoFData()))
    s = primal(y1)
    pbs = Vector{typeof(pb1)}(undef, n)
    pbs[1] = pb1
    for i in 2:n
        yi, pbi = rrule!!(gcd, Ctx(), CoDual(x[i], NoFData()))
        s += primal(yi)
        pbs[i] = pbi
    end
    return zero_fcodual(s), SumMapPullback{G,typeof(pb1),typeof(dx)}(pbs, dx)
end
