using Test
using Differ
using Differ: Dual, CoDual, NoTangent, frule!!, code_dual_ircode
using Differ: tangent_type, fdata_type, rdata_type, fdata, rdata, tangent, zero_tangent
using Differ: Tangent, MutableTangent, PossiblyUninitTangent, NoFData, NoRData, FData, RData
using Differ: build_tangent, primal, increment!!, get_tangent_field
using Differ: _dual_primal_type, _dual_tangent_type

# Primal functions used by the forward-AD fallback. Defined at top level so the
# generated `frule!!` fallback can resolve them via the method table.
plus1(x)   = sin(x) + 1
nest(x)    = sin(cos(x))
prod2(x,y) = x*y + sin(x)
sqr(x)     = x*x
mul3(x)    = (x*x)*x   # explicit 2-arg grouping; `x*x*x` would be a 3-arg (vararg) `*`, unsupported
sincosp(x) = sin(x)*cos(x)

# reverse-mode PoC (straight-line): scalar intrinsics only, and an immutable-struct case reusing
# `V2` (defined below) to exercise `Expr(:new,...)`/`Core.getfield` via `RData`/`increment_field!!`.
rprod(x, y) = x*y + x                             # ∂/∂x = y+1, ∂/∂y = x
rquot(x, y) = (x*y + x) / y                        # mul/add/div composed
function rstruct(a, b)                            # a*b + a, routed through a %new + two getfields
    v = V2(a, b)
    v.a*v.b + v.a
end

# control flow + local assignments
relu(x)    = x > 0.0 ? x : -x                    # branch (== abs here)
function p4(x)                                    # local reassignment: x^4
    r = x*x
    r = r*x
    r = r*x
    r
end
function sumk(x, k)                               # while loop (backward goto): k*x
    s = x - x                                     # 0, without needing an frule!! for zero()
    i = 0
    while i < k
        s = s + x
        i = i + 1
    end
    s
end

function sumk2(x, k, m)                           # nested while loops: k*m*x
    s = x - x
    i = 0
    while i < k
        j = 0
        while j < m
            s = s + x
            j = j + 1
        end
        i = i + 1
    end
    s
end

function branch3(x)                               # if/elseif/else: 3-way merge into one phi
    if x > 2.0
        x*x
    elseif x > 0.0
        x + 1.0
    else
        -x
    end
end

function multiret(x)                              # multiple returns from nested branches
    if x > 10.0
        return x*x
    end
    if x > 0.0
        if x > 5.0
            return x + 100.0
        end
        return x + 1.0
    end
    return -x
end

function sinloop(x, k)                            # loop body calls a surviving frule!! (sin): k*sin(x)
    s = x - x
    i = 0
    while i < k
        s = s + sin(x)
        i = i + 1
    end
    s
end

function sumk_multi(x, y, k)                      # two live loop-carried phis in one block
    s = x - x
    t = y - y
    i = 0
    while i < k
        s = s + x
        t = t + y
        i = i + 1
    end
    s + t
end

trycatch(x) = try sin(x) catch; cos(x) end       # try/catch, non-throwing path: d/dx = cos(x)
# try/catch where the body genuinely throws on one branch (inline throw, M1) and the handler
# differentiates (M2): x>=0 runs the body (x*x), x<0 throws and the catch returns -x.
trythrow(x) = try (x < 0 ? throw(DomainError(x, "neg")) : x*x) catch; -x end

struct Point; x::Float64; y::Float64; end
mutable struct MPoint; x::Float64; y::Float64; end

function pointphi(x)                              # PhiNode merging a Point; one arm a compile-time
    p = x > 0.0 ? Point(1.0, 2.0) : Point(x, x)   # constant, exercising `const_tangent` on structs
    p.x + p.y
end

# Intrinsic-level targets: these differentiate through intrinsics / getfield / %new on the
# post-optimization IRCode, with NO hand-written frule!! methods for +, -, *, /.
struct V2; a::Float64; b::Float64; end
vprod(v::V2) = v.a * v.b
poly32(x::Float32) = x*x + x

# `sitofp` (Int→Float promotion): `x*i` promotes the loop counter `i::Int` via `sitofp` before the
# multiply. The RESULT (`Float64`) carries a real tangent, but `sitofp`'s own operands (the Int value
# and the leading type argument) are non-differentiable — the *inactive* bucket, not the linear one.
# d/dx = 1+2+3 = 6.
sitofp_ctl(x::Float64) = (s = 0.0; for i in 1:3; s += x*i; end; s)
# `fpext`/`fptrunc` (`Float64`<->`Float32` width conversion): genuinely LINEAR, unlike `sitofp` above
# — misclassifying either bucket either drops a real gradient or corrupts a zero one.
# d/dx (Float64(Float32(x)*Float32(2.0)) + x) = 2 + 1 = 3.
mix32_ctl(x::Float64) = Float64(Float32(x) * Float32(2.0)) + x

# Closures capturing differentiable data: the *function* carries a tangent (its captured field), so
# one can differentiate w.r.t. the capture as well as the argument. The capture is read via
# `getfield(#self#, :a)` in the primal IR, whose tangent flows from the function-dual's tangent.
mklin(a)  = x -> a*x         # a·x        : d/dx = a,   d/da = x
mkquad(a) = x -> a*(x*x)     # a·x²       : d/dx = 2a·x, d²/dx² = 2a

# Higher-order AD written the natural way — a differentiation operator composed with itself — rather
# than by hand-nesting `Dual` seeds. `Dop(f, x)` is `f'(x)` via one `frule!!`; nesting `Dop` gives
# `f''` etc. This works because the inner `frule!!` inlines into the differentiated closure's primal
# IR as a surviving `dualized_impl` `:invoke`, which the outer pass re-dualizes (see the
# `pir_arg_offset`/function-slot handling in `build_dual_ir`). `scplusx` is differentiated twice
# below (d²/dx²(sin x + x) = -sin x); `mkquad`'s capture is differentiated at second order too.
Dop(f, x)  = frule!!(Dual(f, zero_tangent(f)), Dual(x, unit_tangent(x))).dx
scplusx(x) = sin(x) + x
d1_scpx(x) = Dop(scplusx, x)   # closure-free first derivative: cos(x) + 1

# A *user* function (defined here, not in Base) with a hand-written `frule!!`, reached through a
# wrapper. Unlike `sin`/`cos`, a user callee is defined at a *late* world, so differentiating a
# caller of it must resolve the callee at the interpreter's *inference* world (`_calleeval`'s `world`
# argument, via `Base.getglobalref`) — resolving at the stale generation world the dualization runs
# under would see the binding as undefined, leak a raw `GlobalRef` into the dual IR, and miscompile at
# order ≥2 (a `Dual{GlobalRef,…}` → `%new` `TypeError`). `@noinline` keeps the callee a surviving
# `:invoke`. This is the exact shape of the original bug report.
@noinline hr_lin(x) = x + 1     # hand rule below; d/dx = 1,  higher derivatives = 0
@noinline hr_sqr(x) = x*x       # hand rule below; d/dx = 2x, d²/dx² = 2, d³/dx³ = 0
call_lin(x) = hr_lin(x)
call_sqr(x) = hr_sqr(x)
Differ.frule!!(::Dual{typeof(hr_lin)}, d::Dual) = Dual(d.x + 1, d.dx * one(d.x))
Differ.frule!!(::Dual{typeof(hr_sqr)}, d::Dual) = Dual(d.x * d.x, d.dx * (2 * d.x))

# Array indexing: `Expr(:boundscheck)` + memoryref builtins (`memoryrefnew`/`memoryrefget`/
# `memoryrefset!`) on the live path. An element read differentiates like a per-element `getfield`
# (the shadow array is a real same-shape `Array{tangent_type(P),N}`).
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
# `getfield(t, i)` with `i` a genuine SSAValue, not a literal — the homogeneous-tuple/NamedTuple
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
# Heterogeneous struct, dynamic `getfield` index: the genuinely hard case (per-field tangent types
# differ) that must always bail, never miscompile.
struct Het2; a::Float64; b::Int; end
hetdyn(h::Het2, i::Int) = Core.getfield(h, i)
# Dynamic `setfield!` index — Phase A only (no same-shape support for writes).
mutable struct MP2; x::Float64; y::Float64; end
function setdyn!(m::MP2, i::Int, v::Float64)
    Core.setfield!(m, i, v)
    return m.x + m.y
end
# Homogeneous MUTABLE struct, dynamic `getfield` READ index (Part 2b — the tractable gap, reusing the
# homogeneous mutable `MP2` above). Every field shares one tangent type, so a runtime index selects a
# validly-typed field whichever one it lands on; the contribution routes through the object's own
# `MutableTangent` via the runtime-`Int` `increment_field_rdata!`. The index is a genuine
# `Argument`/`SSAValue` (confirmed via `Base.code_ircode`), NOT const-folded to a literal.
mp2get(m::MP2, i::Int) = Core.getfield(m, i)     # index is a genuine `Argument` (`_3`)
function mp2sum_dyn(m::MP2)                       # loop index is a genuine `SSAValue` (a phi)
    s = 0.0
    for i in 1:2
        s += Core.getfield(m, i)
    end
    return s
end
# Array allocation inside the differentiated function (`Core.memorynew`, backing `zeros`/`similar`/
# `Vector{T}(undef,...)`/comprehensions) is supported — see "array allocation (forward mode)" below.
allocarr(n) = (v = zeros(n); v[1] = 1.0; v[1])
allocwrite(x) = (v = zeros(2); v[1] = x; v[2] = x + 1.0; v[1] + v[2])
allocsim(v) = (w = similar(v); w[1] = 2.0*v[1]; w[2] = v[1]+v[2]; w[1]+w[2])
alloc2d(m, n) = (A = zeros(m, n); A[1,2] = 5.0; A[2,1] = 7.0; A[1,2] + A[2,1])
alloccomp(x, n) = (v = [x*i for i in 1:n]; v[1] + v[2])
# Still out of scope: growing an existing array (`push!`/`resize!`), which calls
# `Core.memoryrefoffset` directly — a distinct, still-unhandled builtin (unrelated to allocation).
# Should bail gracefully with an `ErrorException`.
growvec!(v, x) = push!(v, x)
# Still out of scope: a genuinely vararg-defined primal method (unrelated to array support).
vfun(x, ys...) = x + sum(ys)

mpoint_read(p::MPoint) = p.x + p.y                    # read-only control for the setfield! test below
mpoint_setx!(p::MPoint, v) = (p.x = v; p.x + p.y)     # mutable-struct field mutation (setfield!)

# Throwing error paths: the happy path differentiates while the error path (a `throw` target whose
# block ends in an `unreachable` terminator) is reconstructed primal-only, so the derivative
# reproduces the same throw on the same inputs. `checkdom` uses the `Core.throw` builtin plus an
# inline `DomainError` construction; `checkdom_ni` routes through a `@noinline` `Union{}`-typed
# `:invoke` throw helper (the shape stdlib domain/bounds checks take); `guarded` guards a division.
checkdom(x) = x < 0 ? throw(DomainError(x, "neg")) : x*x
@noinline throwneg(x) = throw(DomainError(x, "neg"))
checkdom_ni(x) = x < 0 ? throwneg(x) : x*x
guarded(x) = x == 0.0 ? throw(ArgumentError("zero")) : 1/x

# Dynamic dispatch (`apply_generic`): reading a non-`const` global always infers as `Any` (regardless
# of the concrete type of the value it holds), so any call whose argument flows through it — here
# `getindex` on the `Ref`, then `+` — is a genuine `apply_generic`-style dynamic dispatch. These are
# handled by deferring the surviving call to the runtime `dynamic_frule` dispatcher, which rebuilds
# concrete `Dual`s from the runtime values and dispatches `frule!!` dynamically. `dyncall` holds the
# `Ref` constant, so d/dx (x + c) = 1. `dyn_g` is a dynamically-resolved *callee* read from a global.
dyn_ref = Ref{Any}(1.0)
dyncall(x) = x + dyn_ref[]
dyn_g = sin
dyncallee(x) = dyn_g(x)                       # callee itself is dynamic (read from a non-const global)
# A dynamic value that *carries a tangent*: box `x` in a `Ref{Any}`, read it back, and use it — the
# tangent must propagate, so d/dx (r[] * x) = 2x. SROA proves `r[] === x` and folds the read away,
# leaving `*(x, x)` with concrete args but an already-widened `::Any` result; that stays on the static
# `:invoke` path (result annotated `dual_type(R)` = abstract `Dual`), exercising the invariant-`Dual`
# typing rule rather than the `dynamic_frule` trampoline.
dynbox(x) = (r = Ref{Any}(x); r[] * x)
# A `Union`-typed return (a single `ReturnNode` whose value is a `PhiNode` typed `Union{Float64,Int}`):
# the packed `Dual` must be a concrete leaf (`Dual{Float64,Float64}` on this input), not the frozen
# `Dual{Union{Float64,Int},…}` a `%new` would build — which is *not* `<: dual_type(Union{…})`.
dynret(x) = (x > 0 ? x*x : 1)

# Reverse mode Part 1 (recursive `rrule` calls): a genuinely separate, non-inlined callee — `rec_sq`
# must actually survive as a surviving `:invoke` for this to exercise recursion, not just fold into
# ordinary arithmetic — so it's `@noinline`.
@noinline rec_sq(x) = x * x                         # x^2, d/dx = 2x
rec_call(x) = rec_sq(x) + x                          # d/dx = 2x + 1
function rec_branch(x)                               # recursion combined with a branch
    return x > 0.0 ? rec_sq(x) + 1.0 : rec_sq(x) - 1.0
end
function rec_loop(x, k)                              # recursion combined with a loop
    s = 0.0
    for i in 1:k
        s += rec_sq(x)
    end
    return s
end
# Regression: a self-recursive `@noinline` primal has no finite `Tape` type in this design (the
# `in_progress` cycle guard, `reverse_interp.jl`) — must bail with a clean `ErrorException`, not
# recurse forever / stack-overflow.
@noinline function rec_self(x::Float64, n::Int)
    n <= 0 && return x
    return rec_self(x, n - 1)
end

# Reverse mode Part 2 (read-only array indexing): fixed-index read, index chosen by a branch, and a
# minimal hand-written summation loop (avoiding `sum`/iterator-protocol machinery beyond the
# `Base.iterate`-state `===` check every `for` loop already needs — see the array shadow-chain notes
# in `reverse_interp.jl`).
function arr_idx3(x::Vector{Float64})                # a fixed-index read
    return x[3]
end
function arr_idx_branch(x::Vector{Float64}, pick::Bool)   # index chosen by a branch
    return pick ? x[1] : x[2]
end
function arr_sum(x::Vector{Float64})                  # hand-written summation loop
    s = 0.0
    for i in 1:length(x)
        s += x[i]
    end
    return s
end
function arr_mutate!(x::Vector{Float64})              # array mutation (memoryrefset!, Part 3 below)
    x[1] = 2.0 * x[1]
    return x[1]
end

# Reverse mode Part 3 (recursive calls with an array argument): the recursive-call guard now allows
# an array argument through when its identity is traceable back to a function argument, threading the
# real fdata array through the recursive `:invoke` instead of a detached `NoFData()`. `arr_inner` is a
# plain composite function (no hand-written rule) taking the array directly, so differentiating a
# caller of it exercises the *general* engine path, not `sum`'s own hand rule.
@noinline arr_inner(v::Vector{Float64}) = v[1]^2 + v[2]^2         # d/dv = [2v1, 2v2]
arr_outer(v::Vector{Float64}) = arr_inner(v)                       # one level of pass-through recursion
arr_nest_mid(v::Vector{Float64}) = arr_inner(v)
arr_nest(v::Vector{Float64}) = arr_nest_mid(v)                     # two levels of recursion
# Aliasing: the same array argument accumulated into by *two* separate recursive calls — the case
# most likely to expose an accumulation bug, since both inner pullbacks `increment!!` into the same
# shared fdata array.
arr_alias(v::Vector{Float64}) = arr_inner(v) + arr_inner(v)

# `sum(v) do vi ... end` desugars to `sum(f, v)`, which Julia's optimizer inlines down to
# `Base._mapreduce` — the exact shape from the original bug report this milestone fixes. Routed via
# the hand-written `sum`/`sum(f,·)` rules in `src/rrules.jl` (kept off Base's own internals, which are
# self-recursive above `Base.pairwise_blocksize` elements and would hit the unrelated self-recursion
# cycle guard).
f_sumdo(v::Vector{Float64}) = sum(v) do vi
    vi^2 + 2vi + 1
end

# `vs[1]` (an inner array read out of an array-of-arrays via ordinary indexing) used to be a
# genuine provenance dead end (`_fdata_tracked`'s `Core.getfield`/`memoryrefnew` branches only
# recognized the `.ref`-`MemoryRef` chain a *scalar* index produces, not a `memoryrefget` whose own
# result is itself an array). Now that a `memoryrefget` off a tracked ref is itself a tracked root
# when its result carries fdata (Piece 2, see `_fdata_tracked`, `reverse_interp.jl`), `w`'s identity
# threads straight through into the recursive call below and differentiates correctly.
@noinline arr_inner_box(v::Vector{Float64}) = v[1] + v[2]
function arr_via_box(vs::Vector{Vector{Float64}})
    w = vs[1]
    return arr_inner_box(w)
end

# Recursive call with a mutable-struct *argument* (reusing `MPoint`, defined above): `p` is a genuine
# function argument, tracked via `_arg_fdata_tracked`, so `_static_recursible_call`'s guard lets it
# through into the recursive `:invoke`'s `CoDual` — the inner call's rule accumulates straight into
# the caller's own shared `MutableTangent` in place, no rdata needed back from the call at all.
@noinline arr_inner_mut(p::MPoint) = p.x + p.y
arr_via_mut(p::MPoint) = arr_inner_mut(p)

# Reverse mode Part 3 (mutable-struct `getfield`/`setfield!` and array mutation): `mpoint_read`/
# `mpoint_setx!` (defined above, already exercised in forward mode) are reused here for reverse mode.

# Array mutation combined with a full-array read, so both the mutated element and an untouched one
# are exercised in the same gradient: d(sum)/dx[1] = 2, d(sum)/dx[2] = 1.
function arr_mutate_sum!(x::Vector{Float64})
    x[1] *= 2.0
    return sum(x)
end

# Repeated mutation in a loop — the case the save/restore machinery (`:old_primal`/`:old_tangent`)
# exists for; a version without restore passes the straight-line cases above and fails only here.
function refprod_loop!(r::Base.RefValue{Float64}, ys::Vector{Float64})
    for y in ys
        r[] *= y
    end
    return r[]
end

# The originally-reported bug: a closure over a `Ref`, read via two separate `getfield` calls (once
# before the mutation, once after) that must resolve to the *same* underlying `MutableTangent`. An
# aliasing bug here (two independently-zeroed copies instead of one shared object) would silently
# give the wrong gradient rather than erroring, so `refmul_closure`'s own correctness check below (not
# just "it doesn't throw") is what actually exercises this.
function make_refmul_closure()
    r = Ref(1.0)
    g(y) = (r[] *= y; nothing)
    return g, r
end

# `setfield!` of an array-valued field: the field's shadow is aliased to the argument's real shadow
# (rather than a fresh `zero_tangent`), so in-place accumulation into that shared shadow after the
# assignment flows back to `w`. `setbox!` itself returns `nothing` with no downstream read of `b.v`,
# so its own gradient is trivially zero (not a real exercise of the aliasing) — `setbox_sum!` reads
# `b.v` back out through the return value and is the real test.
mutable struct MBox; v::Vector{Float64}; end
setbox!(b::MBox, w::Vector{Float64}) = (b.v = w; nothing)
setbox_sum!(b::MBox, w::Vector{Float64}) = (b.v = w; sum(b.v))

# Array allocation (Case 2 final step, reverse mode): `x[1] = [9.0, 9.0]` allocates a fresh
# `Vector{Float64}` locally (`Core.memorynew` -> `Core.memoryrefnew` -> `%new(Vector,...)`) and
# writes it into the argument array — the allocation escapes, so it survives SROA. Returns `nothing`,
# so `gradient`'s `one(y)` seeding doesn't apply here (see the `rrule!!`-with-explicit-`NoRData()`
# pattern `setbox!` uses above) — differentiated directly via `rrule!!` below instead.
function mutate_nested!(x::Vector{Vector{Float64}})
    x[1] = [9.0, 9.0]
    return nothing
end

# Scalar-returning allocation tests (so `gradient` applies directly): `zeros`/explicit index writes,
# not a `[a,b]` literal — a literal array with 2+ elements lowers through `Base.vect`'s `X::Tuple`
# capture, and when an element is itself differentiable (e.g. `[x, 2x]`) that tuple is a genuine
# differentiable `Core.tuple` result, the one explicitly out-of-scope gap (see the skill doc); `zeros`
# + explicit writes avoids it entirely.
#
# `alloc_and_sum`: allocate, write both elements from `x`, read both back locally — d/dx = 3.
alloc_and_sum(x::Float64) = (v = zeros(2); v[1] = x; v[2] = 2 * x; v[1] + v[2])
# `alloc_store_read!`: allocate, write from `a`, store into the *argument* array `x[1]` (2b aliasing),
# then read back through `x` — exercises allocation and argument-array aliasing together. `x`'s
# original `x[1]` is overwritten before ever being read, so its own gradient is zero. d/da = 3.
function alloc_store_read!(x::Vector{Vector{Float64}}, a::Float64)
    v = zeros(2)
    v[1] = a
    v[2] = 2 * a
    x[1] = v
    return x[1][1] + x[1][2]
end

# Piece 2 (`memoryrefget` provenance for a nested-array result) / Piece 3 (`memoryrefset!` fdata
# aliasing) — no array allocation involved, so fully in scope for both (unlike `mutate_nested!` above).
#
# `x[1]`'s own `memoryrefget` result is a tracked provenance root (its shadow is the corresponding
# element of `x`'s own shadow array), so summing it differentiates directly.
nested_read(x::Vector{Vector{Float64}}) = sum(x[1])

# `x[1] = w` aliases the shadow slot to `w`'s own shadow (rather than a fresh zero), so reading
# `x[1]` back out afterward resolves to `w`'s real shadow array, and later accumulation into it lands
# in `w`'s gradient rather than a detached copy.
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
# aliasing means this also mutates `w` itself in the primal; the gradient must track that mutation
# back to `w`'s original value, not vanish or double-count it.
function nested_write_mutate_through(x::Vector{Vector{Float64}}, w::Vector{Float64})
    x[1] = w
    x[1][1] = 3.0 * x[1][1]
    return w[1] + w[2]
end

# `%new` of a mutable struct combined with recursion into a callee that takes it (3a+3b together): a
# freshly-created local `MPoint` is now a tracked provenance root (`_fdata_tracked`'s `Expr(:new,...)`
# case), and passing it into a recursive call is exactly `arr_via_mut`'s mutable-struct-argument case
# above, just with a locally-created object instead of a function argument. `mpoint_xy` is `@noinline`
# so `p` crosses a genuine call boundary Julia's SROA can't see through — otherwise `p` never escapes
# and the optimizer elides the `%new` entirely before reverse mode ever sees it.
@noinline mpoint_xy(p::MPoint) = p.x + p.y
newmut(x::Float64) = (p = MPoint(x, 1.0); mpoint_xy(p))

# Standalone 3a (no cross-call boundary at all): a mutable struct created and mutated purely locally.
# Plain local create+mutate+read gets scalar-replaced away entirely by SROA before reverse mode ever
# sees a `%new` (confirmed via `Base.code_ircode`) — nesting the fresh `MPoint` inside a second,
# also-freshly-created mutable wrapper forces the `MPoint`'s own `%new` to survive (the wrapper itself
# gets scalarized away by the optimizer, so the IR reverse mode actually sees is just
# `%new(MPoint,...)` + getfield/setfield!/getfield/getfield/add, no trace of the wrapper at all).
mutable struct MPointBox; p::MPoint; end
function newmut_local(x::Float64)
    p = MPoint(x, 1.0)
    p.x = p.x + 2.0
    box = MPointBox(p)
    return box.p.x + box.p.y
end

# 3a + 3b + mutation inside the callee: `p` is created locally, then handed to a `@noinline` callee
# that mutates it in place — the callee's `setfield!` accumulates into the very same shadow
# `MutableTangent` the caller's local `%new` built.
@noinline mpoint_mutate!(p::MPoint) = (p.x = p.x + 5.0; p.x + p.y)
newmut_recursive_mutate(x::Float64) = mpoint_mutate!(MPoint(x, 1.0))

# `vs[1]`'s provenance is now tracked (Piece 2, see `arr_via_box` above), so aliasing it into
# `MArrBox`'s `.v` field is safe and this differentiates correctly too — exercises the
# `%new`-field-provenance guard (`reverse_fwds_to_ircode`'s mutable-`%new` case) on a genuinely
# tracked nested array.
mutable struct MArrBox; v::Vector{Float64}; end
@noinline arrbox_inner(m::MArrBox) = m.v[1] + m.v[2]
function arrbox_untraced(vs::Vector{Vector{Float64}})
    m = MArrBox(vs[1])
    return arrbox_inner(m)
end

@testset "Differ" begin

    @testset "tangent_type / fdata_type / rdata_type" begin
        # Matches Mooncake's documented values (see rule_system / fdata_type docstrings).
        @test tangent_type(Int)     === NoTangent
        @test tangent_type(Float64) === Float64
        @test tangent_type(Bool)    === NoTangent
        @test tangent_type(Vector{Float64}) === Vector{Float64}
        @test tangent_type(Tuple{Float64,Vector{Float64},Int}) ===
              Tuple{Float64,Vector{Float64},NoTangent}
        @test tangent_type(Tuple{Int,Int}) === NoTangent          # all-non-diff collapses
        @test tangent_type(Point) === Tangent{@NamedTuple{x::Float64, y::Float64}}
        @test tangent_type(MPoint) === MutableTangent{@NamedTuple{x::Float64, y::Float64}}

        # fdata / rdata split
        @test (fdata_type(Float64), rdata_type(Float64)) === (NoFData, Float64)
        @test (fdata_type(Vector{Float64}), rdata_type(Vector{Float64})) ===
              (Vector{Float64}, NoRData)
        T = tangent_type(Tuple{Float64,Vector{Float64},Int})
        @test fdata_type(T) === Tuple{NoFData,Vector{Float64},NoFData}
        @test rdata_type(T) === Tuple{Float64,NoRData,NoRData}
        # mutable struct: fdata is the whole tangent, no rdata
        @test fdata_type(tangent_type(MPoint)) === tangent_type(MPoint)
        @test rdata_type(tangent_type(MPoint)) === NoRData

        # tangent(fdata(t), rdata(t)) === t round-trips
        for p in Any[5.0, (5.0, [1.0, 2.0], 3), Point(1.0, 2.0), (a=1.0, b=2)]
            t = zero_tangent(p)
            @test tangent(fdata(t), rdata(t)) == t
        end

        # `_nondiff_field` decides whether a shadow-`Dual` field carries the primal through: true iff
        # the field's tangent is `NoTangent` but its slot can't hold `NoTangent`. That is *any*
        # non-differentiable field, not only singletons — including concrete non-singletons (`Int`,
        # `Tuple{Int,Int}`) and `Type`-valued fields (which `Base.issingletontype` misreports as
        # non-singleton, a documented Julia quirk). Anything else would drop `NoTangent()` into a
        # slot that rejects it.
        @test Differ._nondiff_field(typeof(sin))     # singleton function
        @test Differ._nondiff_field(Type{Float64})   # regression: Type{P}, missed by issingletontype
        @test Differ._nondiff_field(DataType)
        @test Differ._nondiff_field(Int)             # concrete non-singleton, NoTangent tangent
        @test Differ._nondiff_field(Tuple{Int,Int})  # concrete aggregate that collapses to NoTangent
        @test !Differ._nondiff_field(Float64)        # differentiable, takes its tangent
        @test !Differ._nondiff_field(Point)          # differentiable struct, takes its tangent
        @test !Differ._nondiff_field(NoTangent)      # a NoTangent slot holds NoTangent fine
        @test !Differ._nondiff_field(Integer)        # abstract slot: conservatively left on tangent path
    end

    @testset "zero_tangent / increment!!" begin
        @test zero_tangent(2.0)  === 0.0
        @test zero_tangent(3)    === NoTangent()          # Int is non-differentiable
        @test zero_tangent(sin)  === NoTangent()          # singleton function
        # Complex{Float64} is a struct in Mooncake, so its tangent is a Tangent (not a Complex)
        @test zero_tangent(1.0 + 2.0im) == Tangent{@NamedTuple{re::Float64,im::Float64}}((re=0.0, im=0.0))
        @test zero_tangent(Point(1.0, 2.0)) == Tangent{@NamedTuple{x::Float64,y::Float64}}((x=0.0, y=0.0))
        @test zero_tangent([1.0, 2.0]) == [0.0, 0.0]
        # increment!! adds tangents; mutates array fdata in place
        @test increment!!(1.0, 2.0) === 3.0
        a = [1.0, 2.0]; @test increment!!(a, [3.0, 4.0]) === a && a == [4.0, 6.0]
        # `increment!!` decides its aliasing cache via `require_tangent_cache` (keyed on the tangent
        # type), the same authority `zero_tangent`/`set_to_zero!!` use — not a cruder `isbitstype`.
        # A `Vector{<:IEEEFloat}` tangent is provably tree-like, so no `IdDict` is built: after warmup
        # the only allocation is the result-copy path, never a cache. (Regression for the old
        # `isbitstype(T) ? … : IdDict` heuristic that disagreed with `zero_tangent`.)
        @test Differ.require_tangent_cache(Vector{Float64}) === Val{false}()
        let a = [1.0, 2.0, 3.0], b = [10.0, 20.0, 30.0]
            f(x, y) = increment!!(x, y)
            f(copy(a), b)                                  # warmup
            @test (@allocated f(copy(a), b)) == (@allocated copy(a))   # copy only, no IdDict
        end
    end

    @testset "CoDual basics" begin
        cd = CoDual([1.0, 2.0], [0.0, 0.0])
        @test primal(cd) == [1.0, 2.0]
        @test Differ.tangent(cd) == [0.0, 0.0]
        @test Differ.codual_type(Vector{Float64}) === CoDual{Vector{Float64},Vector{Float64}}
        @test Differ.fcodual_type(Float64) === CoDual{Float64,NoFData}
    end

    @testset "Dual basics" begin
        d = Dual(3.0, 4.0)
        @test d.x === 3.0
        @test d.dx === 4.0
        # getproperty aliases: x/y/z -> primal, dx/dy/dz -> tangent
        @test d.y === 3.0 && d.z === 3.0
        @test d.dy === 4.0 && d.dz === 4.0
        @test primal(d) === 3.0
        # type-level Dual field accessors
        @test _dual_primal_type(typeof(d)) === Float64
        @test _dual_tangent_type(typeof(d)) === Float64
        @test _dual_primal_type(typeof(Dual(sin, NoTangent()))) === typeof(sin)
        @test _dual_tangent_type(typeof(Dual(sin, NoTangent()))) === NoTangent
        # a Dual is its own tangent type (the key to higher-order nesting)
        @test tangent_type(Dual{Float64,Float64}) === Dual{Float64,Float64}
        @test tangent_type(typeof(Dual(sin, NoTangent()))) === typeof(Dual(sin, NoTangent()))
    end

    @testset "scalar rules" begin
        x, dx = 0.7, 2.0
        @test frule!!(Dual(sin, NoTangent()), Dual(x, dx)) === Dual(sin(x), cos(x)*dx)
        @test frule!!(Dual(cos, NoTangent()), Dual(x, dx)) === Dual(cos(x), -sin(x)*dx)
        # unary + is identity; unary - negates
        @test frule!!(Dual(+, NoTangent()), Dual(x, dx)) === Dual(x, dx)
        @test frule!!(Dual(-, NoTangent()), Dual(x, dx)) === Dual(-x, -dx)
        # binary +, -, *
        y, dy = 1.5, 3.0
        @test frule!!(Dual(+, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x+y, dx+dy)
        @test frule!!(Dual(-, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x-y, dx-dy)
        @test frule!!(Dual(*, NoTangent()), Dual(x,dx), Dual(y,dy)) === Dual(x*y, x*dy + dx*y)
    end

    @testset "composite fallback (forward AD)" begin
        # d/dx sin(x)+1 = cos(x)
        d = frule!!(Dual(plus1, NoTangent()), Dual(1.0, 2.0))
        @test d.x  ≈ sin(1.0) + 1
        @test d.dx ≈ cos(1.0) * 2.0

        # nested: d/dx sin(cos(x)) = cos(cos(x))*(-sin(x))
        dn = frule!!(Dual(nest, NoTangent()), Dual(0.5, 1.0))
        @test dn.x  ≈ sin(cos(0.5))
        @test dn.dx ≈ cos(cos(0.5)) * (-sin(0.5))

        # multi-arg: p(x,y)=x*y+sin(x); ∂/∂x and ∂/∂y via tangent seeding
        px = frule!!(Dual(prod2, NoTangent()), Dual(2.0,1.0), Dual(3.0,0.0))
        @test px.x  ≈ 2.0*3.0 + sin(2.0)
        @test px.dx ≈ 3.0 + cos(2.0)                 # ∂/∂x = y + cos(x)
        py = frule!!(Dual(prod2, NoTangent()), Dual(2.0,0.0), Dual(3.0,1.0))
        @test py.dx ≈ 2.0                            # ∂/∂y = x

        # products: d/dx x^2 = 2x, d/dx x^3 = 3x^2
        @test frule!!(Dual(sqr,  NoTangent()), Dual(3.0,1.0)).dx ≈ 2*3.0
        @test frule!!(Dual(mul3, NoTangent()), Dual(2.0,1.0)).dx ≈ 3*2.0^2

        # d/dx sin(x)cos(x) = cos(2x)
        dsc = frule!!(Dual(sincosp, NoTangent()), Dual(0.9, 1.0))
        @test dsc.dx ≈ cos(2*0.9)
    end

    @testset "intrinsic-level rules (no arithmetic frules)" begin
        # Complex arithmetic differentiated via add_float/mul_float/getfield/%new. Under the
        # Mooncake tangent system `Complex{Float64}` is a *struct*, so its tangent is a
        # `Tangent{@NamedTuple{re::Float64, im::Float64}}` (not another `Complex`). `ct` builds such
        # a tangent from a complex "direction"; the shadow reads `re`/`im` via `get_tangent_field`.
        ct(c) = build_tangent(ComplexF64, real(c), imag(c))
        z, w   = 1.0 + 2.0im, 3.0 + 4.0im
        dz, dw = 0.5 + 0.0im, 0.0 + 1.0im
        da = frule!!(Dual(+, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
        @test da.x  == z + w
        @test da.dx == ct(dz + dw)
        dm = frule!!(Dual(*, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
        @test dm.x  == z * w
        @test dm.dx == ct(z*dw + dz*w)                # complex product rule
        ds = frule!!(Dual(-, NoTangent()), Dual(z, ct(dz)), Dual(w, ct(dw)))
        @test ds.dx == ct(dz - dw)

        # Float32 straight-line composite: d/dx (x^2 + x) = 2x + 1
        d32 = frule!!(Dual(poly32, NoTangent()), Dual(2.0f0, 1.0f0))
        @test d32.x  === 2.0f0^2 + 2.0f0
        @test d32.dx === 2*2.0f0 + 1.0f0              # stays Float32

        # user struct via getfield: d/dv (v.a * v.b). The tangent of a `V2` is a `Tangent`, seeded
        # (da, db) = (1, 0). The shadow reads fields via `get_tangent_field`.
        dv = frule!!(Dual(vprod, NoTangent()), Dual(V2(2.0, 3.0), build_tangent(V2, 1.0, 0.0)))
        @test dv.x  == 6.0
        @test dv.dx == 1.0*3.0 + 2.0*0.0              # = b*da + a*db

        # `sitofp` (Int->Float promotion, `x*i` with `i::Int`): the INACTIVE bucket — the result
        # carries a real tangent but the operands (Int value + type) don't. d/dx (x·(1+2+3)) = 6.
        dsi = frule!!(Dual(sitofp_ctl, NoTangent()), Dual(2.0, 1.0))
        @test dsi.x == 12.0 && dsi.dx == 6.0
        # `fpext`/`fptrunc` (Float32<->Float64 width conversion): the LINEAR bucket — genuinely
        # differentiable. d/dx (Float64(Float32(x)·Float32(2)) + x) = 2 + 1 = 3.
        dmx = frule!!(Dual(mix32_ctl, NoTangent()), Dual(1.0, 1.0))
        @test dmx.x == 3.0 && dmx.dx == 3.0
        fd(f, x; h=1e-5) = (f(x+h) - f(x-h)) / 2h
        @test dsi.dx ≈ fd(sitofp_ctl, 2.0) rtol=1e-5
        @test dmx.dx ≈ fd(mix32_ctl, 1.0) rtol=1e-2   # loose: FD through Float32 quantization is noisy
    end

    @testset "local reassignment (straight-line after optimization)" begin
        # p4(x)=x^4 via reassignment; optimization lowers it to straight-line SSA (no phi),
        # so the IRCode engine handles it. derivative 4x^3
        d4 = frule!!(Dual(p4, NoTangent()), Dual(2.0, 1.0))
        @test d4.x  ≈ 2.0^4
        @test d4.dx ≈ 4 * 2.0^3
    end

    @testset "control flow: branches and loops" begin
        # Block topology is preserved 1:1 from the primal IR; GotoNode/GotoIfNot/PhiNode are
        # supported by duplicating each PhiNode into a primal phi + a shadow phi.
        function checkverify(f, argtypes)
            ir, _ = code_dual_ircode(f, argtypes)
            Core.Compiler.verify_ir(ir)
        end

        # branch (== abs here): d/dx = sign(x)
        r1 = frule!!(Dual(relu, NoTangent()), Dual(2.0, 1.0));  @test r1.x ≈ 2.0  && r1.dx ≈ 1.0
        r2 = frule!!(Dual(relu, NoTangent()), Dual(-2.0, 1.0)); @test r2.x ≈ 2.0  && r2.dx ≈ -1.0
        checkverify(relu, (Float64,))

        # while loop: k*x
        s1 = frule!!(Dual(sumk, NoTangent()), Dual(3.0, 1.0), Dual(4, 0))
        @test s1.x ≈ 12.0 && s1.dx ≈ 4.0
        checkverify(sumk, (Float64, Int))

        # nested while loops: k*m*x
        n1 = frule!!(Dual(sumk2, NoTangent()), Dual(2.0, 1.0), Dual(3, 0), Dual(5, 0))
        @test n1.x ≈ 2.0*3*5 && n1.dx ≈ 3.0*5.0
        checkverify(sumk2, (Float64, Int, Int))

        # if/elseif/else 3-way merge
        for (x, expected_x, expected_dx) in ((3.0, 9.0, 6.0), (1.0, 2.0, 1.0), (-1.0, 1.0, -1.0))
            b = frule!!(Dual(branch3, NoTangent()), Dual(x, 1.0))
            @test b.x ≈ expected_x && b.dx ≈ expected_dx
        end
        checkverify(branch3, (Float64,))

        # multiple returns from nested branches
        for (x, expected_x, expected_dx) in ((20.0, 400.0, 40.0), (7.0, 107.0, 1.0),
                                              (3.0, 4.0, 1.0), (-3.0, 3.0, -1.0))
            m = frule!!(Dual(multiret, NoTangent()), Dual(x, 1.0))
            @test m.x ≈ expected_x && m.dx ≈ expected_dx
        end
        checkverify(multiret, (Float64,))

        # loop body calling a surviving frule!! (sin): k*sin(x)
        sl = frule!!(Dual(sinloop, NoTangent()), Dual(0.6, 1.0), Dual(3, 0))
        @test sl.x ≈ 3*sin(0.6) && sl.dx ≈ 3*cos(0.6)
        checkverify(sinloop, (Float64, Int))

        # two live loop-carried phis in one block: seed x and y independently
        mx = frule!!(Dual(sumk_multi, NoTangent()), Dual(2.0, 1.0), Dual(3.0, 0.0), Dual(4, 0))
        @test mx.dx ≈ 4.0                                  # ds/dx = k
        my = frule!!(Dual(sumk_multi, NoTangent()), Dual(2.0, 0.0), Dual(3.0, 1.0), Dual(4, 0))
        @test my.dx ≈ 4.0                                  # dt/dy = k
        checkverify(sumk_multi, (Float64, Float64, Int))

        # PhiNode merging a Point struct, one arm a compile-time constant
        pt = frule!!(Dual(pointphi, NoTangent()), Dual(1.0, 1.0))
        @test pt.x ≈ 3.0 && pt.dx ≈ 0.0                    # constant arm: d/dx = 0
        pf = frule!!(Dual(pointphi, NoTangent()), Dual(-1.0, 1.0))
        @test pf.x ≈ -2.0 && pf.dx ≈ 2.0                   # Point(x,x): d/dx (x+x) = 2
        checkverify(pointphi, (Float64,))
    end

    @testset "error paths (throwing)" begin
        # A block ending in an unreachable `ReturnNode` (a throw target) is reconstructed
        # primal-only: the happy path differentiates, and the derivative still throws on inputs
        # that make the primal throw. (Distinct from exception *handling* / try-catch below.)
        checkverify(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at)[1])

        # happy-path derivatives: d/dx x*x = 2x; d/dx 1/x = -1/x^2
        for (f, x, d) in ((checkdom, 3.0, 6.0), (checkdom_ni, 3.0, 6.0), (guarded, 2.0, -0.25))
            @test frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx ≈ d
            checkverify(f, (Float64,))
        end

        # the derivative reproduces the primal's throw on the throwing input
        @test_throws DomainError   frule!!(Dual(checkdom,    NoTangent()), Dual(-2.0, 1.0))
        @test_throws DomainError   frule!!(Dual(checkdom_ni, NoTangent()), Dual(-2.0, 1.0))
        @test_throws ArgumentError frule!!(Dual(guarded,     NoTangent()), Dual(0.0, 1.0))
    end

    @testset "derivative matches finite differences" begin
        fd(f, x; h=1e-6) = (f(x+h) - f(x-h)) / 2h
        for (f, x) in ((plus1, 1.3), (nest, 0.4), (sqr, 2.1), (mul3, -0.7), (sincosp, 0.6))
            got = frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx
            @test got ≈ fd(f, x) rtol=1e-5
        end
    end

    @testset "exception handling (try/catch)" begin
        # try/catch dualizes: UpsilonNode/PhiCNode are duplicated into primal + shadow copies, and
        # EnterNode/:leave/:pop_exception carry over as control markers (block topology preserved
        # 1:1). The optimizer deletes provably-non-throwing try scopes, so `trycatch`/`trythrow`
        # retain a live EnterNode only because their bodies can actually throw.
        checkverify(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at)[1])

        # non-throwing path through a try body calling surviving frules: d/dx sin(x) = cos(x)
        t = frule!!(Dual(trycatch, NoTangent()), Dual(0.6, 1.0))
        @test t.x ≈ sin(0.6) && t.dx ≈ cos(0.6)
        checkverify(trycatch, (Float64,))

        # body throws on one branch, handler differentiates: x>=0 -> x*x (d=2x); x<0 caught -> -x (d=-1)
        rp = frule!!(Dual(trythrow, NoTangent()), Dual(3.0, 1.0))
        @test rp.x ≈ 9.0 && rp.dx ≈ 6.0                    # happy path (no throw)
        rc = frule!!(Dual(trythrow, NoTangent()), Dual(-2.0, 1.0))
        @test rc.x ≈ 2.0 && rc.dx ≈ -1.0                   # catch path (body threw)
        checkverify(trythrow, (Float64,))
    end

    @testset "array indexing (forward mode)" begin
        # `v[i]`/`v[i]=x` lower to `Expr(:boundscheck)` + `memoryrefnew`/`memoryrefget`/
        # `memoryrefset!`. An element read differentiates like a per-element `getfield`; a write
        # mirrors the same builtin onto the shadow array (itself a real same-shape
        # `Array{tangent_type(P),N}` — no wrapper needed).
        checkverify(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at)[1])

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
        # the fix, this crashed with a TypeError — the primal index referenced the primal IR's own
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
        # tangent types) is the genuinely hard case — must bail with a located error, never crash or
        # silently return a wrong/zero derivative.
        @test_throws "no dualization rule for builtin `getfield`" frule!!(
            Dual(hetdyn, NoTangent()),
            Dual(Het2(1.0, 2), build_tangent(Het2, 1.0, NoTangent())), Dual(1, NoTangent()))

        # regression: a dynamic setfield! index — Phase A only, always bails (no same-shape support
        # for writes).
        @test_throws ErrorException frule!!(
            Dual(setdyn!, NoTangent()),
            Dual(MP2(1.0, 2.0), MutableTangent((x=1.0, y=1.0))),
            Dual(1, NoTangent()), Dual(5.0, 1.0))

        # dynamic getfield index into a homogeneous MUTABLE struct (Part 2b): the runtime index is a
        # genuine `Argument` (`_3`), not a const-folded literal — assert that, then differentiate. The
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

    @testset "mutable-struct field mutation (setfield!, forward mode)" begin
        # setfield! mutates the primal in place and its `MutableTangent` shadow via
        # `set_tangent_field!` — the mutation-side counterpart of the existing getfield/
        # get_tangent_field read path.
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
        checkverify(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at)[1])

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
    end

    @testset "graceful bail on unsupported IR" begin
        # growing an existing array (`push!`/`resize!`) calls `Core.memoryrefoffset` directly — a
        # distinct, still-unhandled builtin (array allocation itself is now supported). Should
        # error, not miscompile. The message names the offending builtin.
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

    @testset "dynamic dispatch (apply_generic)" begin
        # A surviving dynamic call (callee/arg non-concrete — e.g. a value flowed through an
        # `Any`-typed global/field/`Ref`) is deferred to the runtime `dynamic_frule` dispatcher; a
        # concrete-args/abstract-result call (`dynbox`) stays static — see the primal defs above.
        # d/dx (x + const) = 1 (the `Ref` is held constant).
        @test frule!!(Dual(dyncall, NoTangent()), Dual(1.0, 1.0)).dx ≈ 1.0
        # dynamically-resolved callee read from a global: d/dx sin(x) = cos(x).
        @test frule!!(Dual(dyncallee, NoTangent()), Dual(0.5, 1.0)).dx ≈ cos(0.5)
        # tangent must propagate *through* the dynamic (Any-typed) value: d/dx (r[]*x) = 2x.
        @test frule!!(Dual(dynbox, NoTangent()), Dual(3.0, 1.0)).dx ≈ 6.0
        # matches finite differences on a nonlinear composition through a dynamic value.
        fdyn(x) = sin(x + dyn_ref[])
        h = 1e-6
        @test frule!!(Dual(fdyn, NoTangent()), Dual(0.7, 1.0)).dx ≈ (fdyn(0.7+h) - fdyn(0.7-h))/2h atol=1e-6

        # A non-concrete return type must pack the result as a *concrete leaf* `Dual` (built via a
        # runtime `Dual(p,t)` call), not a frozen `Dual{Any,Any}`/`Dual{Union,…}`, so the result stays
        # a well-typed dual (composable back into `frule!!`). `dynbox` returns `Any`; `dynret` a `Union`.
        @test frule!!(Dual(dynbox, NoTangent()), Dual(3.0, 1.0)) isa Dual{Float64,Float64}
        @test frule!!(Dual(dynret, NoTangent()), Dual(3.0, 1.0)) isa Dual{Float64,Float64}
    end

    @testset "function tangents (closures)" begin
        # The function isn't always constant: a closure carries a tangent in its captured field, so
        # the derivative w.r.t. the capture flows from the function-dual's tangent (read via
        # `getfield(#self#, :a)` in the body → `get_tangent_field` on the closure's `Tangent`).
        # Under the Mooncake tangent system a closure's tangent is a `Tangent{@NamedTuple{a::…}}`;
        # `zero_tangent(f)` is the "hold f constant" tangent (all-zero captures).
        f = mklin(2.0)                                     # a·x with a = 2
        capt(v) = build_tangent(typeof(f), v)              # closure tangent with da = v
        # d/dx (a constant): tangent = a·dx = 2
        r = frule!!(Dual(f, zero_tangent(f)), Dual(3.0, 1.0))
        @test r.x ≈ 6.0 && r.dx ≈ 2.0
        # d/da: seed the *function* tangent (da = 1), hold x fixed (dx = 0); tangent = da·x = 3
        ra = frule!!(Dual(f, capt(1.0)), Dual(3.0, 0.0))
        @test ra.x ≈ 6.0 && ra.dx ≈ 3.0
        # both directions at once (a=2, da=10, x=3, dx=1): a·dx + da·x = 2 + 30 = 32
        rb = frule!!(Dual(f, capt(10.0)), Dual(3.0, 1.0))
        @test rb.dx ≈ 32.0

        # a quadratic closure differentiated w.r.t. x, holding the capture constant: g(x)=a·x²,
        # g'(x)=2a·x. verify_ir on the dualized closure body (the capture getfield is dualized).
        g = mkquad(2.0)                                    # 2·x²
        rg = frule!!(Dual(g, zero_tangent(g)), Dual(1.5, 1.0))
        @test rg.x ≈ 2*1.5^2 && rg.dx ≈ 2*2*1.5
        Core.Compiler.verify_ir(code_dual_ircode(g, (Float64,))[1])
    end

    @testset "higher-order forward mode (uniform nesting)" begin
        # A second derivative differentiates the first-order dualized function itself (Option A —
        # compose the transform): the primal for a nested-dual request is the order-(k-1) dual IR,
        # re-dualized. ALL seeds — the function included — are nested uniformly to the order, with
        # NoTangent at the function's leaves. Then r.x.x = f(x), r.dx.x = r.x.dx = f'(x), r.dx.dx = f''(x).
        fseed2(f) = Dual(Dual(f, NoTangent()), Dual(f, NoTangent()))   # constant fn nested to order 2
        seed2(x)  = Dual(Dual(x, 1.0), Dual(1.0, 0.0))

        # analytic second derivatives: (x²)''=2, (x³)''=6x, (x⁴)''=12x²; and now sin(x)cos(x) too:
        # (sin·cos)'' = d/dx cos(2x) = -2 sin(2x)  (exercises the :new fix + sin/cos rewrite)
        for (f, x, fx, dfx, d2fx) in ((sqr,     2.0, 4.0,  4.0,  2.0),
                                      (mul3,    2.0, 8.0,  12.0, 12.0),
                                      (p4,      2.0, 16.0, 32.0, 48.0),
                                      (sincosp, 0.6, sin(0.6)*cos(0.6), cos(2*0.6), -2*sin(2*0.6)))
            r = frule!!(fseed2(f), seed2(x))
            @test r.x.x  ≈ fx
            @test r.x.dx ≈ dfx && r.dx.x ≈ dfx      # both first-derivative cross-terms agree
            @test r.dx.dx ≈ d2fx
        end

        # second derivative matches central finite differences of the (AD) first derivative, incl.
        # the transcendental cases now that sin/cos work to higher order
        d1(f, x) = frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx
        d2fd(f, x; h=1e-4) = (d1(f, x+h) - d1(f, x-h)) / 2h
        for (f, x) in ((sqr, 2.1), (mul3, -0.7), (p4, 1.3), (sincosp, 0.6), (nest, 0.4), (plus1, 1.1))
            @test frule!!(fseed2(f), seed2(x)).dx.dx ≈ d2fd(f, x) rtol=1e-4
        end

        # order-N general: 3rd derivative of x⁴ is 24x (= 48 at x=2) by nesting one level deeper
        # (the function is nested to order 3 as well)
        fz(f) = Dual(f, NoTangent())
        fseed3(f) = Dual(Dual(fz(f), fz(f)), Dual(fz(f), fz(f)))
        s3(x) = Dual(Dual(Dual(x,1.0),Dual(1.0,0.0)), Dual(Dual(1.0,0.0),Dual(0.0,0.0)))
        @test frule!!(fseed3(p4), s3(2.0)).dx.dx.dx ≈ 24*2.0

        # the 2nd-order transform produces valid IR, including sin/cos and through control flow (the
        # tuple-aware vararg prologue composes with phi/goto re-dualization)
        checkverify2(f, at) = Core.Compiler.verify_ir(code_dual_ircode(f, at; order=2)[1])
        for (f, at) in ((sqr,(Float64,)), (mul3,(Float64,)), (p4,(Float64,)), (sincosp,(Float64,)),
                        (relu,(Float64,)), (branch3,(Float64,)), (sumk,(Float64,Int)))
            checkverify2(f, at)
        end
        @test true   # reached here ⇒ every verify_ir above passed

        # a non-uniformly-nested seed (function NOT nested at order 2) is no longer valid: the uniform
        # peel can't form the inner carrier, so it bails rather than miscompiling.
        @test_throws Exception frule!!(Dual(sqr, NoTangent()), seed2(2.0))

        # graceful bail still holds at higher order: a vararg primal is unsupported → ErrorException
        # (unrelated to array support — kept as the "some construct is still unsupported" regression).
        err = try
            frule!!(fseed2(vfun), seed2(1.0), seed2(2.0))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("vararg", err.msg)
    end

    @testset "higher-order via composed differentiation (nested frule!! / D-of-D)" begin
        # Differentiate a function that itself calls `frule!!`. The inner `frule!!` inlines into the
        # outer closure's primal IR as a surviving `dualized_impl` `:invoke`; the outer dualization
        # pass re-dualizes it (the function slot `Dual{typeof(dualized_impl),NoTangent}` is dropped
        # and the remaining nested value args peel down to the inner order-1 carrier).

        # The exact form from the design goal: D(f, x) with `do` blocks, second derivative of sin+x.
        r = Dop(10.0) do x
            Dop(x) do x
                sin(x) + x
            end
        end
        @test r ≈ -sin(10.0)                               # d²/dx²(sin x + x) = -sin x

        # Named-function equivalents, checked against the analytic second derivative at several points.
        for x in (0.4, 1.3, -0.7, 2.1)
            @test Dop(scplusx, x) ≈ cos(x) + 1             # first derivative
            @test Dop(d1_scpx, x) ≈ -sin(x)                # second derivative via composed D
            @test Dop(x -> Dop(scplusx, x), x) ≈ -sin(x)   # same, as a closure literal
        end

        # Higher orders by composing D further: d³/dx³(sin x + x) = -cos x, d⁴/dx⁴ = sin x. The
        # `frule!!`-slot compose path (re-dualizing a surviving `frule!!` invoke) recurses cleanly.
        d2_scpx(x) = Dop(d1_scpx, x)
        d3_scpx(x) = Dop(d2_scpx, x)
        @test Dop(d2_scpx, 0.4) ≈ -cos(0.4)
        @test Dop(d3_scpx, 0.4) ≈  sin(0.4)

        # The re-dualized outer closure produces valid IR (verify_ir on the raw dualized IRCode).
        Core.Compiler.verify_ir(code_dual_ircode(d1_scpx, (Float64,))[1])
        @test true

        # Regression: forward-mode dualization of a self-recursive `@noinline` primal must bail
        # cleanly rather than stack-overflow. This exercises the `dualized_impl_in_progress` cycle
        # guard, whose forward-mode twist is that the recursion crosses *fresh* `ADInterpreter`
        # instances via the `frule!!` `@generated` boundary — so the guard is task-local (shared
        # across those instances) rather than a per-`interp` field like reverse mode's `in_progress`.
        @test_throws ErrorException frule!!(Dual(rec_self, NoTangent()), Dual(1.0, 1.0), Dual(3, NoTangent()))

        # Limitation: a closure/struct with *differentiable fields* cannot be differentiated at order
        # ≥2. The self-tangent `Dual` scheme (`tangent_type(Dual{P,T}) == Dual{P,T}`) requires each
        # carried type to be its own tangent type, which fails for such a struct (its tangent is a
        # `Tangent`, not itself). Surfaces as a clear Differ error, not a miscompile. `mkquad(3.0)` is
        # a closure with a `Float64` capture, so nesting D over it lands here (whereas nesting D over
        # the plain-function `scplusx` above is fine). First-order differentiation of the same closure
        # — including w.r.t. its capture — works and is covered by the closures testset above.
        @test_throws ErrorException Dop(z -> Dop(mkquad(3.0), z), 1.7)
    end

    @testset "user function with a hand-written frule!! (world-age callee resolution)" begin
        # Differentiating a caller of a *user* function with a hand rule: the callee must be resolved
        # at the interpreter's inference world, not the stale generation world (see `hr_lin`/`hr_sqr`).
        # First order works even on a clean tree; the crash was at order ≥2 (a `Dual{GlobalRef,…}`).
        @test Dop(call_lin, 1.0) == 1.0                          # d/dx (x+1) = 1

        # The exact original bug report: D-of-D over the wrapper. Second/third derivatives of x+1 = 0.
        @test (Dop(1.0) do x; Dop(call_lin, x) end) == 0.0
        @test (Dop(1.0) do x; Dop(y -> Dop(call_lin, y), x) end) == 0.0

        # Non-linear rule so the second derivative is non-trivial: d/dx(x²)=2x, d²/dx²(x²)=2.
        for x in (0.4, 1.3, -0.7, 2.1)
            @test Dop(call_sqr, x) ≈ 2x
            @test (Dop(z -> Dop(call_sqr, z), x)) ≈ 2.0
        end

        # The dualized caller is valid IR (verify_ir on the raw dualized IRCode).
        Core.Compiler.verify_ir(code_dual_ircode(call_sqr, (Float64,))[1])
        @test true
    end

    @testset "allocation-free (dualization is fully inlined)" begin
        # The dualized code is real post-optimization IRCode: Duals are built with %new and
        # surviving high-level rules are `:invoke`s to CodeInstances, so a straight-line dual is
        # allocation-free. `sincosp` exercises the surviving `frule!!(sin)`/`frule!!(cos)` :invoke path.
        allocs(f, x) = (d = Dual(x, 1.0); df = Dual(f, NoTangent());
                        frule!!(df, d); @allocated frule!!(df, d))     # measure warmed
        @test allocs(sqr, 2.0)     == 0        # pure intrinsics
        @test allocs(sincosp, 0.6) == 0        # surviving sin/cos rule :invokes
    end

    @testset "reverse mode (branches and loops, Mooncake-style tape — see the control-flow plan)" begin
        # Central finite differences, one argument at a time.
        fd1(f, x, y, k; h=1e-6) = k == 1 ? (f(x+h, y) - f(x-h, y)) / 2h : (f(x, y+h) - f(x, y-h)) / 2h

        # Tier 1: scalar float intrinsics only (add_float/mul_float/div_float), checked two ways:
        # finite differences, and a cross-check against the already-trusted forward-mode `frule!!`
        # (one seed direction per argument) — independent verification of the new reverse engine.
        for f in (rprod, rquot)
            x, y = 2.0, 3.0
            _, dx, dy = gradient(f, x, y)
            @test dx ≈ fd1(f, x, y, 1) rtol = 1e-5
            @test dy ≈ fd1(f, x, y, 2) rtol = 1e-5
            @test dx ≈ frule!!(Dual(f, NoTangent()), Dual(x, 1.0), Dual(y, 0.0)).dx
            @test dy ≈ frule!!(Dual(f, NoTangent()), Dual(x, 0.0), Dual(y, 1.0)).dx
        end

        # Tier 2: immutable struct via `%new`/`getfield`, exercising `RData`/`increment_field!!`.
        # rstruct(a,b) = a*b + a  =>  ∂/∂a = b+1, ∂/∂b = a
        _, da, db = gradient(rstruct, 2.0, 3.0)
        @test da ≈ 3.0 + 1.0
        @test db ≈ 2.0

        # branches: `relu` is the multiple-reachable-`return`s shape (the common shape
        # Julia's optimizer actually produces for an `if/else` with a value in each arm — see
        # `_exit_blocks`'s docstring); `branch3` merges all three arms into a single `return` via one
        # `PhiNode` with three predecessors (exercises the `Switch`-with-more-than-two-targets path).
        # Both are checked against finite differences AND the trusted forward-mode `frule!!` (already
        # supports control flow) as an independent cross-check of the new tape/block-stack machinery.
        for (f, x) in ((relu, 2.0), (relu, -2.0), (branch3, 3.0), (branch3, 1.0), (branch3, -1.0))
            _, dx = gradient(f, x)
            h = 1e-6
            @test dx ≈ (f(x + h) - f(x - h)) / 2h rtol = 1e-5
            @test dx ≈ frule!!(Dual(f, NoTangent()), Dual(x, 1.0)).dx
        end

        # A branch combined with `%new`/`getfield` (both new mechanisms exercised together, reusing
        # `V2` — already defined above for `rstruct`).
        function branch_struct(a, b)
            v = V2(a, b)
            return a > b ? v.a * v.b : v.a + v.b
        end
        for (a, b) in ((5.0, 2.0), (1.0, 4.0))
            _, da, db = gradient(branch_struct, a, b)
            h = 1e-6
            @test da ≈ (branch_struct(a + h, b) - branch_struct(a - h, b)) / 2h rtol = 1e-5
            @test db ≈ (branch_struct(a, b + h) - branch_struct(a, b - h)) / 2h rtol = 1e-5
        end

        # loops: A loop body may execute an unknown number of times, so this is
        # the first place the block stack and per-block comms `Stack`s are actually needed
        # (not just degenerate 0-or-1-entry stacks, as in the branch-only cases above) — and the
        # first place rdata `Ref`s must correctly reset/accumulate across repeated visits in exact
        # LIFO order. `sumk`/`sumk2`/`sumk_multi` are the existing forward-mode loop fixtures (a
        # single while-loop, nested while-loops, and two live loop-carried accumulators in one block,
        # respectively) — reused here and cross-checked against the already-trusted `frule!!`.
        _, dx_sumk = gradient(sumk, 3.0, 4)
        @test dx_sumk ≈ frule!!(Dual(sumk, NoTangent()), Dual(3.0, 1.0), Dual(4, 0)).dx
        _, dx_sumk2 = gradient(sumk2, 2.0, 3, 5)
        @test dx_sumk2 ≈ frule!!(Dual(sumk2, NoTangent()), Dual(2.0, 1.0), Dual(3, 0), Dual(5, 0)).dx
        _, dx_multi, dy_multi = gradient(sumk_multi, 2.0, 3.0, 4)
        @test dx_multi ≈ frule!!(Dual(sumk_multi, NoTangent()), Dual(2.0, 1.0), Dual(3.0, 0.0), Dual(4, 0)).dx
        @test dy_multi ≈ frule!!(Dual(sumk_multi, NoTangent()), Dual(2.0, 0.0), Dual(3.0, 1.0), Dual(4, 0)).dx
        # A zero-iteration loop (the loop-carried accumulator never updates) is a good edge case.
        _, dx_zero = gradient(sumk, 3.0, 0)
        @test dx_zero == 0.0

        # A surviving high-level call now differentiates via Part 1's recursive `rrule` support
        # (`sin` specifically resolves to the hand-written rule in `src/rrules.jl`, not raw recursion
        # into `Base.Math.sin`'s internals — see that file's header).
        _, dx_plus1 = gradient(plus1, 1.3)
        @test dx_plus1 ≈ cos(1.3)

        # `code_reverse_fwds_ircode`/`code_reverse_pullback_ircode` mirror `code_dual_ircode`:
        # inspect each carrier's generated IR directly and confirm it passes `Core.Compiler.verify_ir`
        # — the same debugging workflow forward mode uses.
        # `code_reverse_fwds_ircode` inspects the tape-*allocating* carrier shape (`Ctx{Nothing}`). A
        # `build_ctx(...; prealloc=true)` context compiles a *different* prologue — one that reads the
        # caller's stacks out of the ctx and resets them instead of constructing them — so verify that
        # shape too, or the pre-allocated path goes unchecked.
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
        checkverify_rev(rprod, (Float64, Float64))
        checkverify_rev(rquot, (Float64, Float64))
        checkverify_rev(rstruct, (Float64, Float64))
        checkverify_rev(relu, (Float64,))
        checkverify_rev(branch3, (Float64,))
        checkverify_rev(branch_struct, (Float64, Float64))
        checkverify_rev(sumk, (Float64, Int))
        checkverify_rev(sumk2, (Float64, Int, Int))
        checkverify_rev(sumk_multi, (Float64, Float64, Int))
        checkverify_rev(plus1, (Float64,))   # recursion into a hand-written reverse-mode rule (sin)
        checkverify_rev(nest, (Float64,))    # composed hand rules: sin(cos(x))

        # Reverse-mode conversion intrinsics. `sitofp` (Int->Float promotion) is the INACTIVE bucket:
        # its result carries a real tangent but its operands don't, so its pullback consumes the seed
        # and contributes `NoRData` — d/dx (x·(1+2+3)) = 6. `fpext`/`fptrunc` (Float32<->Float64) is
        # the LINEAR bucket: genuinely differentiable, d/dx (Float64(Float32(x)·2) + x) = 3. Before
        # these rules existed, either bailed with "no reverse rule for intrinsic `sitofp`/`fptrunc`".
        # Cross-checked against forward mode and finite differences.
        _, dx_si = gradient(sitofp_ctl, 2.0)
        @test dx_si == 6.0
        @test dx_si == frule!!(Dual(sitofp_ctl, NoTangent()), Dual(2.0, 1.0)).dx
        _, dx_mx = gradient(mix32_ctl, 1.0)
        @test dx_mx == 3.0
        @test dx_mx == frule!!(Dual(mix32_ctl, NoTangent()), Dual(1.0, 1.0)).dx
        hh = 1e-5
        @test dx_si ≈ (sitofp_ctl(2.0 + hh) - sitofp_ctl(2.0 - hh)) / 2hh rtol = 1e-5
        @test dx_mx ≈ (mix32_ctl(1.0 + hh) - mix32_ctl(1.0 - hh)) / 2hh rtol = 1e-2  # Float32 FD is noisy
        checkverify_rev(sitofp_ctl, (Float64,))
        checkverify_rev(mix32_ctl, (Float64,))

        # Dynamic (non-literal) `getfield` index (Phase B): `for i in 1:2` does not unroll, so `t[i]`
        # reaches the pullback as a genuine dynamic index. Before the fix, the pullback resolved the
        # field to a raw, unresolved `SSAValue` instead of its runtime value, silently degenerating
        # `increment_field!!`'s `Val`-based dispatch into a no-op — the gradient came back `(0.0,0.0)`
        # instead of `(1.0,1.0)`, with no error at all. `d/dt_1 = d/dt_2 = 1`.
        _, dt_dyn = gradient(tupsum_dyn, (3.0, 4.0))
        @test dt_dyn == (1.0, 1.0)
        @test dt_dyn[1] == frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), (1.0, 0.0))).dx
        @test dt_dyn[2] == frule!!(Dual(tupsum_dyn, NoTangent()), Dual((3.0, 4.0), (0.0, 1.0))).dx
        h = 1e-6
        @test dt_dyn[1] ≈ (tupsum_dyn((3.0 + h, 4.0)) - tupsum_dyn((3.0 - h, 4.0))) / 2h rtol = 1e-5
        @test dt_dyn[2] ≈ (tupsum_dyn((3.0, 4.0 + h)) - tupsum_dyn((3.0, 4.0 - h))) / 2h rtol = 1e-5
        checkverify_rev(tupsum_dyn, (Tuple{Float64,Float64},))

        # same, over a homogeneous NamedTuple.
        _, dnt_dyn = gradient(ntupsum_dyn, (a=3.0, b=4.0))
        @test dnt_dyn == (a=1.0, b=1.0)
        checkverify_rev(ntupsum_dyn, (@NamedTuple{a::Float64,b::Float64},))

        # dynamic getfield index into a homogeneous MUTABLE struct (Part 2b): the field's rdata
        # contribution routes into the object's own `MutableTangent` via the runtime-`Int`
        # `increment_field_rdata!` (not an object-level `RData`, as a mutable struct has none). The
        # gradient w.r.t. the struct is a one-hot `MutableTangent` for a single selected field, and
        # all-ones when summed over both. Index is a genuine `Argument`/`SSAValue`, not const-folded.
        _, dmp2_r, _ = gradient(mp2get, MP2(3.0, 4.0), 2)
        @test get_tangent_field(dmp2_r, :x) == 0.0 && get_tangent_field(dmp2_r, :y) == 1.0
        _, dmp1_r, _ = gradient(mp2get, MP2(3.0, 4.0), 1)
        @test get_tangent_field(dmp1_r, :x) == 1.0 && get_tangent_field(dmp1_r, :y) == 0.0
        _, dms_r = gradient(mp2sum_dyn, MP2(3.0, 4.0))
        @test get_tangent_field(dms_r, :x) == 1.0 && get_tangent_field(dms_r, :y) == 1.0
        checkverify_rev(mp2get, (MP2, Int))
        checkverify_rev(mp2sum_dyn, (MP2,))

        # regression: a dynamic getfield index into a HETEROGENEOUS struct is the genuinely hard case
        # — must bail with a located error, never crash or silently return a wrong/zero gradient.
        @test_throws "dynamic (non-literal) field index" gradient(hetdyn, Het2(1.0, 2), 1)

        # regression: a dynamic setfield! index — Phase A only, always bails.
        @test_throws "dynamic (non-literal) field index" gradient(setdyn!, MP2(1.0, 2.0), 1, 5.0)

        # Phase D (unique-predecessor optimization): every push must still be matched by exactly one
        # pop across a full rule+pullback round trip — the class of bug ("accidentally
        # under/over-pushing") that optimization risks introducing. The pullback *is* the tape, so
        # this just calls it and then confirms every `Stack`'s `position` (block stack, and every
        # non-singleton per-block comms stack) is back to 0.
        #
        # Doubly load-bearing now that a `build_ctx(...; prealloc=true)` context *reuses* its tape
        # across calls: balance is what makes reuse correct, so this also runs each case twice
        # through one pre-allocated context and checks the answers agree.
        function check_stack_balance(f, args...)
            ctx = build_ctx(f, map(Differ._typeof, args); prealloc=false)
            fcd, argcds = zero_fcodual(f), map(zero_fcodual, args)
            ycd, pb = rrule!!(fcd, ctx, argcds...)
            pb(one(primal(ycd)))
            @test pb.block_stack.position == 0
            @test all(s -> !(s isa Differ.Stack) || s.position == 0, pb.comms)

            # Same again through a pre-allocated (tape-reusing) context, twice.
            pctx = build_ctx(f, map(Differ._typeof, args))
            g1 = gradient!(pctx, zero_fcodual(f), map(zero_fcodual, args)...)
            g2 = gradient!(pctx, zero_fcodual(f), map(zero_fcodual, args)...)
            @test g1 == g2
            @test g1 == gradient(f, args...)
            @test pctx.tape.block_stack.position == 0
            @test all(s -> !(s isa Differ.Stack) || s.position == 0, pctx.tape.comms)
        end
        check_stack_balance(rprod, 2.0, 3.0)                # straight-line: no block-stack push at all
        check_stack_balance(relu, 2.0)
        check_stack_balance(relu, -2.0)
        check_stack_balance(branch3, 3.0)
        check_stack_balance(branch3, 1.0)
        check_stack_balance(branch3, -1.0)
        check_stack_balance(sumk, 2.0, 5)
        check_stack_balance(sumk, 2.0, 0)                   # zero iterations
        check_stack_balance(plus1, 1.3)
        check_stack_balance(nest, 0.4)
        check_stack_balance(sumk2, 1.5, 3, 4)

        # Tier 5: Part 1 — recursive `rrule` calls (statically-resolvable calls only).
        _, dx_rec = gradient(rec_call, 3.0)
        @test dx_rec ≈ 2*3.0 + 1
        @test dx_rec ≈ frule!!(Dual(rec_call, NoTangent()), Dual(3.0, 1.0)).dx

        for x in (2.0, -2.0)
            _, dx_rb = gradient(rec_branch, x)
            @test dx_rb ≈ 2x
        end

        _, dx_rl = gradient(rec_loop, 2.0, 3)
        @test dx_rl ≈ 3 * 2 * 2.0
        check_stack_balance(rec_loop, 2.0, 5)

        # Regression: self-recursion has no finite `Tape` type — must bail cleanly (the `in_progress`
        # cycle guard), not stack-overflow.
        @test_throws ErrorException gradient(rec_self, 1.0, 3)
        # Regression: a dynamic-dispatch callee (read from a non-`const` global, reusing the existing
        # forward-mode `dyncallee` fixture) is not statically recursible — must still bail.
        @test_throws ErrorException gradient(dyncallee, 1.0)

        checkverify_rev(rec_call, (Float64,))
        checkverify_rev(rec_branch, (Float64,))
        checkverify_rev(rec_loop, (Float64, Int))

        # Tier 6: Part 2 — read-only array indexing.
        x4 = [1.0, 2.0, 3.0, 4.0]
        _, dx_i3 = gradient(arr_idx3, x4)
        @test dx_i3 == [0.0, 0.0, 1.0, 0.0]

        x2 = [1.0, 2.0]
        _, dx_ib_t, dp_t = gradient(arr_idx_branch, x2, true)
        @test dx_ib_t == [1.0, 0.0]
        @test dp_t === NoTangent()
        _, dx_ib_f, = gradient(arr_idx_branch, x2, false)
        @test dx_ib_f == [0.0, 1.0]

        x3 = [1.0, 2.0, 3.0]
        _, dx_sum = gradient(arr_sum, x3)
        @test dx_sum == ones(3)
        # Cross-check every element individually against central differences.
        for k in eachindex(x3)
            xp = copy(x3); xp[k] += 1e-6
            xm = copy(x3); xm[k] -= 1e-6
            @test dx_sum[k] ≈ (arr_sum(xp) - arr_sum(xm)) / 2e-6 rtol = 1e-5
        end

        # Array mutation (Part 3, `memoryrefset!`): `arr_mutate!(x) = (x[1] = 2*x[1]; x[1])` — the
        # returned value only ever sees the *overwritten* x[1], so d/dx = [2.0, 0.0].
        _, dx_mut = gradient(arr_mutate!, [1.0, 2.0])
        @test dx_mut == [2.0, 0.0]
        h = 1e-6
        xp = [1.0 + h, 2.0]; xm = [1.0 - h, 2.0]
        @test dx_mut[1] ≈ (arr_mutate!(xp) - arr_mutate!(xm)) / 2h rtol = 1e-5

        checkverify_rev(arr_idx3, (Vector{Float64},))
        checkverify_rev(arr_idx_branch, (Vector{Float64}, Bool))
        checkverify_rev(arr_sum, (Vector{Float64},))

        check_stack_balance(arr_sum, [1.0, 2.0, 3.0])

        # Tier 7: Part 3 — recursive calls with an array argument.
        x5 = [3.0, 4.0]

        # The general engine path (no hand rule): a plain composite function taking the array
        # directly, one level and two levels of recursion.
        _, dx_outer = gradient(arr_outer, x5)
        @test dx_outer == [2 * x5[1], 2 * x5[2]]
        _, dx_nest = gradient(arr_nest, x5)
        @test dx_nest == [2 * x5[1], 2 * x5[2]]

        # Aliasing: the same array accumulated into by two separate recursive calls must double the
        # gradient, not overwrite or lose one contribution.
        _, dx_alias = gradient(arr_alias, x5)
        @test dx_alias == [4 * x5[1], 4 * x5[2]]

        # The user's original failing benchmark: `sum(v) do vi ... end`, routed through the
        # hand-written `sum(f, ·)` rule (`src/rrules.jl`) rather than the general engine path.
        x6 = [1.0, 2.0]
        _, dx_sumdo = gradient(f_sumdo, x6)
        @test dx_sumdo == 2 .* x6 .+ 2
        for k in eachindex(x6)
            xp = copy(x6); xp[k] += 1e-6
            xm = copy(x6); xm[k] -= 1e-6
            @test dx_sumdo[k] ≈ (f_sumdo(xp) - f_sumdo(xm)) / 2e-6 rtol = 1e-5
        end

        # Plain `sum(x)`, also via the hand-written rule.
        x7 = [1.0, 2.0, 3.0, 4.0]
        _, dx_plainsum = gradient(sum, x7)
        @test dx_plainsum == ones(4)

        # Piece 2: `vs[1]`'s identity is now tracked through the recursive call (see the comment on
        # `arr_via_box` above): d/dv[1] = 1, d/dv[2] = 1.
        _, dvs_avb = gradient(arr_via_box, [[1.0, 2.0]])
        @test dvs_avb == [[1.0, 1.0]]
        checkverify_rev(arr_via_box, (Vector{Vector{Float64}},))
        check_stack_balance(arr_via_box, [[1.0, 2.0]])

        # 3b: recursive call with a mutable-struct *argument* — the inner call's rule accumulates
        # straight into the caller's own shared `MutableTangent`, so both fields' gradients come back.
        _, dp_avm = gradient(arr_via_mut, MPoint(1.0, 2.0))
        @test dp_avm == MutableTangent{@NamedTuple{x::Float64,y::Float64}}((x=1.0, y=1.0))

        checkverify_rev(arr_outer, (Vector{Float64},))
        checkverify_rev(arr_nest, (Vector{Float64},))
        checkverify_rev(arr_alias, (Vector{Float64},))
        checkverify_rev(f_sumdo, (Vector{Float64},))
        checkverify_rev(arr_via_mut, (MPoint,))

        check_stack_balance(arr_outer, [3.0, 4.0])
        check_stack_balance(arr_alias, [3.0, 4.0])
        check_stack_balance(f_sumdo, [1.0, 2.0])
        check_stack_balance(arr_via_mut, MPoint(1.0, 2.0))

        # Tier 8: Part 3 — mutable-struct `getfield`/`setfield!` and array mutation (`memoryrefset!`).
        p0 = MPoint(2.0, 3.0)
        _, dp_read = gradient(mpoint_read, p0)
        @test dp_read == MutableTangent{@NamedTuple{x::Float64,y::Float64}}((x=1.0, y=1.0))

        p1 = MPoint(2.0, 3.0)
        _, dp_setx, dv_setx = gradient(mpoint_setx!, p1, 10.0)
        # p.x is overwritten before use, so its own gradient contribution is 0; p.y and v both flow
        # straight through to the `+`.
        @test dp_setx == MutableTangent{@NamedTuple{x::Float64,y::Float64}}((x=0.0, y=1.0))
        @test dv_setx == 1.0
        # Cross-check against the already-trusted forward-mode `frule!!` result for the same primal.
        p2 = MPoint(2.0, 3.0)
        fwd_setx = frule!!(Dual(mpoint_setx!, NoTangent()), Dual(p2, zero_tangent(p2)), Dual(10.0, 1.0))
        @test dv_setx ≈ fwd_setx.dx
        h = 1e-6
        fd_v = (mpoint_setx!(MPoint(2.0, 3.0), 10.0 + h) - mpoint_setx!(MPoint(2.0, 3.0), 10.0 - h)) / 2h
        @test dv_setx ≈ fd_v rtol = 1e-5

        # Array mutation: the mutated element and an untouched one are both exercised via `sum`.
        x8 = [3.0, 4.0]
        _, dx_mutsum = gradient(arr_mutate_sum!, x8)
        @test dx_mutsum == [2.0, 1.0]
        for k in eachindex(x8)
            xp = copy(x8); xp[k] += 1e-6
            xm = copy(x8); xm[k] -= 1e-6
            @test dx_mutsum[k] ≈ (arr_mutate_sum!(xp) - arr_mutate_sum!(xm)) / 2e-6 rtol = 1e-5
        end

        # Repeated mutation in a loop — the case the save/restore (`:old_primal`/`:old_tangent`)
        # machinery exists for; also the strongest available proof that the *same* `MutableTangent`
        # is shared across every iteration's separate `getfield` access (a broken-aliasing bug here
        # would give a numerically wrong answer, not an error): d(result)/dr = ys[1]*ys[2] = 12,
        # d(result)/dys[1] = r*ys[2] = 8, d(result)/dys[2] = r*ys[1] = 6.
        r0 = Ref(2.0)
        ys0 = [3.0, 4.0]
        _, dr_loop, dys_loop = gradient(refprod_loop!, r0, ys0)
        @test get_tangent_field(dr_loop, 1) ≈ 12.0
        @test dys_loop == [8.0, 6.0]
        @test r0[] == 2.0   # forward-replay mutates r0; the pullback's restore leaves it as found

        # A closure over a `Ref`, read via two separate `getfield` calls
        # (once before the mutation, once after) — must simply run without erroring, must leave `r`
        # restored after a full forward+pullback round trip, and (since the primal returns `nothing`)
        # contributes no gradient to `y` through the return value.
        g, r = make_refmul_closure()
        gcd, pb = rrule!!(zero_fcodual(g), Ctx(), CoDual(3.0, NoFData()))
        @test primal(gcd) === nothing
        _, dy_closure = pb(NoRData())
        @test dy_closure == 0.0
        @test r[] == 1.0

        # `setfield!` of an array-valued field (fdata-carrying): the field's shadow is aliased to
        # `w`'s own shadow, so `sum(b.v)`'s in-place accumulation into that shared array is `w`'s
        # actual gradient, not a fresh zero. `b`'s own gradient is zero — its original `.v` is
        # overwritten before ever being read, mirroring `mpoint_setx!` above.
        b1 = MBox([1.0, 2.0])
        w1 = [3.0, 4.0]
        _, db_setbox, dw_setbox = gradient(setbox_sum!, b1, w1)
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
        # `setbox!` (returns `nothing`, `b.v` never read back downstream) now runs to completion
        # instead of bailing — the aliasing mechanism handles it fine, there is just nothing
        # downstream to carry a gradient to `w`. `gradient`/`gradient!` both seed the pullback with
        # `one(y)`, which has no method for `y === nothing` (a pre-existing, unrelated restriction of
        # that convenience API to scalar-output primals) — so exercise `rrule!!` directly with an
        # explicit `NoRData()` seed, exactly as `make_refmul_closure` above does for the same reason.
        b0 = MBox([1.0, 2.0])
        bshadow0 = MutableTangent{@NamedTuple{v::Vector{Float64}}}((v=zeros(2),))
        w0 = [3.0, 4.0]
        wshadow0 = zeros(2)
        ycd0, pb0 = rrule!!(zero_fcodual(setbox!), Ctx(), CoDual(b0, bshadow0), CoDual(w0, wshadow0))
        @test primal(ycd0) === nothing
        pb0(NoRData())
        @test bshadow0 == MutableTangent{@NamedTuple{v::Vector{Float64}}}((v=[0.0, 0.0],))
        @test wshadow0 == [0.0, 0.0]

        # Array allocation (Case 2 final step): `mutate_nested!` returns `nothing`, so `gradient`'s
        # `one(y)` seeding doesn't apply — exercise `rrule!!` directly with an explicit `NoRData()`
        # seed, exactly as `setbox!` above. The freshly-allocated `[9.0, 9.0]` has no dependency on
        # `x`, so its aliased shadow is zero; the pullback's restore leaves `x`/its shadow as found.
        xm0 = [[1.0, 2.0], [3.0, 4.0]]
        xmshadow0 = [[0.0, 0.0], [0.0, 0.0]]
        ycdm, pbm = rrule!!(zero_fcodual(mutate_nested!), Ctx(), CoDual(xm0, xmshadow0))
        @test primal(ycdm) === nothing
        @test xm0 == [[9.0, 9.0], [3.0, 4.0]]        # forward replay mutated x[1] in place
        @test xmshadow0 == [[0.0, 0.0], [0.0, 0.0]]  # aliased shadow of the fresh array is zero
        @test pbm(NoRData()) == (NoRData(), NoRData())
        @test xm0 == [[1.0, 2.0], [3.0, 4.0]]        # pullback restored the overwritten slot
        @test xmshadow0 == [[0.0, 0.0], [0.0, 0.0]]
        checkverify_rev(mutate_nested!, (Vector{Vector{Float64}},))

        # Scalar-returning allocation tests (see the function definitions above for why `zeros` +
        # explicit writes, not a `[a,b]` literal): finite-difference-checked, `gradient`-based.
        _, dx_aas = gradient(alloc_and_sum, 3.0)
        @test dx_aas ≈ 3.0
        h = 1e-6
        @test dx_aas ≈ (alloc_and_sum(3.0 + h) - alloc_and_sum(3.0 - h)) / 2h rtol = 1e-5
        checkverify_rev(alloc_and_sum, (Float64,))
        check_stack_balance(alloc_and_sum, 3.0)

        # Allocation + 2b argument-array aliasing together: the freshly-allocated array is stored into
        # `x[1]` and read back *through `x`*, so the gradient w.r.t. `a` must flow through both the
        # allocation's own shadow chain and the argument-array aliasing machinery; `x`'s own gradient
        # is zero (its original `x[1]` is overwritten before ever being read).
        _, dx_asr, da_asr = gradient(alloc_store_read!, [[1.0, 2.0], [3.0, 4.0]], 5.0)
        @test dx_asr == [[0.0, 0.0], [0.0, 0.0]]
        @test da_asr ≈ 3.0
        @test da_asr ≈ (alloc_store_read!([[1.0, 2.0], [3.0, 4.0]], 5.0 + h) -
                        alloc_store_read!([[1.0, 2.0], [3.0, 4.0]], 5.0 - h)) / 2h rtol = 1e-5
        checkverify_rev(alloc_store_read!, (Vector{Vector{Float64}}, Float64))
        check_stack_balance(alloc_store_read!, [[1.0, 2.0], [3.0, 4.0]], 5.0)

        # Adversarial: reverse mode's shadow `memoryrefnew` now forces its own boundscheck flag `true`
        # (mirroring forward mode's identical safety note — `Dual`/`CoDual`'s constructor never checks
        # a caller-supplied tangent array's *length* against its primal's), so a too-short shadow
        # raises a catchable `BoundsError` instead of corrupting memory via an unchecked out-of-bounds
        # `MemoryRef`. Not allocation-specific (the checked ref is the *argument* array's own), but the
        # same `Base.memoryrefnew` rule allocation itself now depends on.
        @test_throws BoundsError rrule!!(zero_fcodual(arr_idx3), Ctx(),
                                         CoDual([1.0, 2.0, 3.0, 4.0], [1.0]))

        # Regression: still out of scope, must bail cleanly (a located reason, not a crash) — growing
        # an existing array (`push!`/`resize!`) routes through `Core.memoryrefoffset`, a distinct,
        # still-unhandled builtin (unrelated to allocation, which is now fully supported).
        @test_throws ErrorException gradient(growvec!, [1.0, 2.0], 3.0)

        # Piece 2 (`memoryrefget` provenance) / Piece 3 (`memoryrefset!` fdata-aliasing): nested-array
        # read and write-then-read, no allocation involved.
        _, dx_nr = gradient(nested_read, [[1.0, 2.0], [3.0, 4.0]])
        @test dx_nr == [[1.0, 1.0], [0.0, 0.0]]
        h = 1e-6
        for k in 1:2
            xp = [[1.0, 2.0], [3.0, 4.0]]; xp[1][k] += h
            xm = [[1.0, 2.0], [3.0, 4.0]]; xm[1][k] -= h
            @test dx_nr[1][k] ≈ (nested_read(xp) - nested_read(xm)) / 2h rtol = 1e-5
        end

        _, dx_nwe, dw_nwe = gradient(nested_write_existing, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
        @test dx_nwe == [[0.0, 0.0], [0.0, 0.0]]   # x[1] overwritten before use — zero, not aliased
        @test dw_nwe == [1.0, 1.0]
        for k in 1:2
            wp = [5.0, 6.0]; wp[k] += h
            wm = [5.0, 6.0]; wm[k] -= h
            fd = (nested_write_existing([[1.0, 2.0], [3.0, 4.0]], wp) -
                  nested_write_existing([[1.0, 2.0], [3.0, 4.0]], wm)) / 2h
            @test dw_nwe[k] ≈ fd rtol = 1e-5
        end

        # Adversarial: both aliases read back — the gradient must sum both contributions (a broken,
        # fresh-zero shadow would only ever show one of them).
        _, dx_rb, dw_rb = gradient(nested_write_read_both, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
        @test dx_rb == [[0.0, 0.0], [0.0, 0.0]]
        @test dw_rb == [2.0, 2.0]

        # Adversarial: mutate a scalar element through the alias — real array aliasing means this
        # also mutates `w` in the primal, and the gradient must track that: d/dw[1] = 3, d/dw[2] = 1.
        _, dx_mt, dw_mt = gradient(nested_write_mutate_through, [[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
        @test dx_mt == [[0.0, 0.0], [0.0, 0.0]]
        @test dw_mt == [3.0, 1.0]

        # Adversarial: a pre-seeded non-zero incoming shadow on `w` must accumulate, not overwrite —
        # exercise `rrule!!` directly with explicit shadows, mirroring `make_refmul_closure` above.
        x0 = [[1.0, 2.0], [3.0, 4.0]]
        xshadow0 = [[0.0, 0.0], [0.0, 0.0]]
        w0 = [5.0, 6.0]
        wshadow0 = [10.0, 20.0]
        ycd_nwe, pb_nwe = rrule!!(zero_fcodual(nested_write_existing), Ctx(),
                                  CoDual(x0, xshadow0), CoDual(w0, wshadow0))
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

        # Piece 2: `vs[1]`'s provenance is tracked, so aliasing it into `MArrBox.v` is safe too:
        # d/dvs[1] = [1.0, 1.0], d/dvs[2] = [0.0, 0.0] (untouched).
        _, dvs_abu = gradient(arrbox_untraced, [[1.0, 2.0], [3.0, 4.0]])
        @test dvs_abu == [[1.0, 1.0], [0.0, 0.0]]
        checkverify_rev(arrbox_untraced, (Vector{Vector{Float64}},))
        check_stack_balance(arrbox_untraced, [[1.0, 2.0], [3.0, 4.0]])

        # 3a: `%new` of a mutable struct, purely locally (no cross-call boundary) — checked against
        # finite differences.
        _, dx_nml = gradient(newmut_local, 3.0)
        @test dx_nml ≈ 1.0
        h = 1e-6
        @test dx_nml ≈ (newmut_local(3.0 + h) - newmut_local(3.0 - h)) / 2h rtol = 1e-5

        # 3a + 3b together (the original target case): `%new` of a mutable struct crossing a genuine
        # `@noinline` recursive-call boundary.
        _, dx_nm = gradient(newmut, 5.0)
        @test dx_nm ≈ 1.0
        @test dx_nm ≈ (newmut(5.0 + h) - newmut(5.0 - h)) / 2h rtol = 1e-5

        # 3a + 3b + mutation inside the callee: a locally-created mutable struct passed to a
        # recursive callee that mutates it in place before the caller ever reads it back.
        _, dx_nmr = gradient(newmut_recursive_mutate, 2.0)
        @test dx_nmr ≈ 1.0
        @test dx_nmr ≈ (newmut_recursive_mutate(2.0 + h) - newmut_recursive_mutate(2.0 - h)) / 2h rtol = 1e-5

        checkverify_rev(mpoint_read, (MPoint,))
        checkverify_rev(mpoint_setx!, (MPoint, Float64))
        checkverify_rev(arr_mutate!, (Vector{Float64},))
        checkverify_rev(arr_mutate_sum!, (Vector{Float64},))
        checkverify_rev(refprod_loop!, (Base.RefValue{Float64}, Vector{Float64}))
        checkverify_rev(setbox_sum!, (MBox, Vector{Float64}))
        checkverify_rev(newmut_local, (Float64,))
        checkverify_rev(newmut, (Float64,))
        checkverify_rev(newmut_recursive_mutate, (Float64,))

        check_stack_balance(mpoint_setx!, MPoint(2.0, 3.0), 10.0)
        check_stack_balance(arr_mutate!, [1.0, 2.0])
        check_stack_balance(arr_mutate_sum!, [3.0, 4.0])
        check_stack_balance(refprod_loop!, Ref(2.0), [3.0, 4.0])
        check_stack_balance(setbox_sum!, MBox([1.0, 2.0]), [3.0, 4.0])
        check_stack_balance(newmut_local, 3.0)
        check_stack_balance(newmut, 5.0)
        check_stack_balance(newmut_recursive_mutate, 2.0)
    end

    include("test_intrinsic_dispatch.jl")      # dispatch-based intrinsic handling (add_float)
    include("test_backedges.jl")               # derivative invalidation on primal redefinition
    include("test_cfg_ir.jl")                  # CFGBlock/ID working-IR round-trip (Phase A, no AD)

end
