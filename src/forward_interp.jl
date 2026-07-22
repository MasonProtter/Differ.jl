# Forward-mode AD: the `frule` entry point, the mode-specific glue that plugs into the
# `ADInterpreter` seams (`contextual.jl`), and the split-shadow dualization engine itself.
#
# `frule(dualargs::Dual...)` is a `@generated` fallback: for a composite primal `g` with no
# hand-written rule, it compiles a *dualized* version of `g`'s body under
# `ADInterpreter{Forward}` and invokes the resulting `CodeInstance`. The dualization is a
# split-shadow transform on `g`'s post-optimization `IRCode`, spliced into the typeinf pipeline at
# the `finishinfer!` (return type) / `optimize` (install) seams via the `build_contextual_ir`
# hook below.

# Carrier stub: gives a MethodInstance whose specTypes is the *dual* signature.
# `ADInterpreter{Forward}` replaces its source with the dualized primal body, so this
# body must never actually run.
dualized_impl(dualargs::Dual...) =
    error("ADNext.dualized_impl ran directly: ADInterpreter could not dualize the primal ",
          "(likely unsupported IR — e.g. array indexing / a builtin with no rule, or a vararg call).")


# ---------------------------------------------------------------------------
# Forward-mode transform hook.
#
# The generated `frule` fallback asks the interpreter to compile a `dualized_impl` MethodInstance
# whose `specTypes` is the *dual* signature. We compile it by transforming the corresponding primal
# method's post-optimization `IRCode` into a dualized `IRCode` (`build_dual_ir`). Non-`dualized_impl`
# MethodInstances return `nothing` here and flow through the ordinary pipeline (see the seam
# in `contextual.jl`).
# ---------------------------------------------------------------------------

function build_contextual_ir(interp::ADInterpreter{Forward}, mi::MethodInstance)
    is_dualized_impl(mi) || return nothing
    return build_dual_ir(interp, mi)
end

is_dualized_impl(mi) = isa(mi.def, Method) && !isempty(mi.specTypes.parameters) &&
                       mi.specTypes.parameters[1] === typeof(dualized_impl)

# Resolve the primal MethodInstance and dual arity for a `dualized_impl` specialization.
function primal_of_impl(interp::ADInterpreter, impl_mi::MethodInstance)
    dualparams = impl_mi.specTypes.parameters[2:end]
    all(P -> P isa Type && P <: Dual, dualparams) || return nothing
    primal_tt = Base.to_tuple_type(Any[_dual_primal_type(P) for P in dualparams])
    pmatch, _ = CC.findsup(primal_tt, CC.method_table(interp))
    pmatch === nothing && return nothing
    isa(pmatch.method, Method) || return nothing
    pmatch.method.isva && return nothing
    primal_mi = specialize_method(pmatch.method, pmatch.spec_types, pmatch.sparams)::MethodInstance
    return (primal_mi, length(dualparams))
end

# Build the dualized `IRCode` for a `dualized_impl` specialization from the primal's optimized
# `IRCode`. Returns the dual `IRCode` or `nothing` (unsupported IR → bail).
#
# Two cases, distinguished by whether any *value* argument's primal is itself a `Dual`:
#
#  * First order (base case): the primal is an ordinary user method (or a hand-written `frule`),
#    found via `primal_of_impl`, and dualized directly. `pir` has scalar positional arguments.
#  * Higher order (Option A — compose the transform): a request whose value args are nested Duals
#    (e.g. `Dual{Dual{F,F},Dual{F,F}}`) is differentiating the *order-(k-1) dualized function*. That
#    function's optimized dual IR is obtained by peeling one `Dual` level off each value arg,
#    resolving the inner `dualized_impl` carrier, and recursing through `optimized_dual_ir`; the
#    result is then re-dualized. That inner dual IR is vararg-shaped (`dualized_impl(dualargs...)`),
#    so `pir_is_vararg=true` selects the tuple-reconstruction prologue in `dualize_to_ircode`.
function build_dual_ir(interp::ADInterpreter, impl_mi::MethodInstance)
    dualparams = impl_mi.specTypes.parameters[2:end]
    all(P -> P isa Type && P <: Dual, dualparams) || return nothing
    n = length(dualparams)
    # All arguments — the function included — are nested uniformly to the differentiation order (a
    # constant/function nests too, carrying `NoTangent` at the leaf), so an order-≥2 request is one
    # where *any* argument's primal is itself a `Dual`. The `frule` guard is the one exception:
    # `frule` is the only Dual-consuming function in the system, so a Dual-valued arg *under `frule`*
    # means "differentiate a hand rule once" (the base case — the rule is an ordinary method taking
    # Duals), not "differentiate the derivative". That guard is what lets the recursion peel down to
    # `frule` and hand off to `primal_of_impl`.
    higher_order = n >= 1 && _dual_primal_type(dualparams[1]) !== typeof(frule) &&
                   any(i -> _dual_primal_type(dualparams[i]) <: Dual, 1:n)

    if higher_order
        # Peel exactly one Dual level off every arg uniformly (the function like any other value):
        # `Dual{Dual{f,NoTangent},…}` → `Dual{f,NoTangent}`, `Dual{Dual{F,F},Dual{F,F}}` → `Dual{F,F}`.
        # That yields the order-(k-1) carrier signature.
        inner_dualparams = Any[_dual_primal_type(P) for P in dualparams]
        inner_tt = Tuple{typeof(dualized_impl), inner_dualparams...}
        m, _ = CC.findsup(inner_tt, CC.method_table(interp))
        m === nothing && return nothing
        inner_mi = specialize_method(m.method, m.spec_types, m.sparams)::MethodInstance
        pir = optimized_dual_ir(interp, inner_mi)
        pir === nothing && return nothing
        return dualize_to_ircode(interp, impl_mi, pir, n; pir_is_vararg=true)
    end

    info = primal_of_impl(interp, impl_mi)
    info === nothing && return nothing
    primal_mi, _ = info
    world = CC.get_inference_world(interp)
    # Optimized primal IR via the internal `typeinf_ircode`. A NativeInterpreter is used so that
    # `sin`/`cos` and other hand-ruled functions survive as `:invoke`s (routed through `frule`).
    pir, _ = CC.typeinf_ircode(CC.NativeInterpreter(world), primal_mi, nothing)
    pir === nothing && return nothing
    return dualize_to_ircode(interp, impl_mi, pir, n; pir_is_vararg=false)
end

# The optimized dual `IRCode` for a `dualized_impl` carrier: exactly what the `optimize` seam
# installs (and what `code_dual_ircode` returns) — `build_dual_ir` followed by the IPO-safe passes.
# Used both by the higher-order recursion above (to obtain the order-(k-1) dual IR as a primal) and
# by the reflection entry point.
function optimized_dual_ir(interp::ADInterpreter, impl_mi::MethodInstance)
    ir = build_dual_ir(interp, impl_mi)
    ir === nothing && return nothing
    world = CC.get_inference_world(interp)
    opt = CC.OptimizationState(impl_mi, CC.retrieve_code_info(impl_mi, world), interp)
    return run_ipo_passes!(ir, opt)
end

# Resolve and compile the `frule(Dual{typeof(f),NoTangent}, dualargs...)` rule for a surviving
# high-level call to an *invoke-able `CodeInstance`*, so the dualized IR can emit a static
# `:invoke` (mirroring how the primal IR keeps `sin`/`cos` as `:invoke`s to a `CodeInstance`).
# `:invoke` targets *must* be `CodeInstance`s: `collectinvokes!` only JITs those, so a bare
# `MethodInstance` would fall back to a boxed dynamic call. Returns `nothing` if unresolved.
function frule_codeinstance(interp::ADInterpreter, @nospecialize(ftype), dual_argtypes)
    frule_tt = Tuple{typeof(frule), Dual{ftype,NoTangent}, dual_argtypes...}
    fm, _ = CC.findsup(frule_tt, CC.method_table(interp))
    fm === nothing && return nothing
    isa(fm.method, Method) || return nothing
    frule_mi = specialize_method(fm.method, fm.spec_types, fm.sparams)::MethodInstance
    world = CC.get_inference_world(interp)
    return CC.typeinf_ext_toplevel(CC.NativeInterpreter(world), frule_mi, CC.SOURCE_MODE_ABI)::CodeInstance
end


# ===========================================================================
# The split-shadow dualization engine: post-optimization forward-AD on typed `IRCode`.
#
# `dualize_to_ircode` (called by `build_dual_ir` above) transforms a primal method's fully
# optimized `IRCode` into a *dualized* `IRCode` using the "split-shadow" scheme: the primal
# computation is reconstructed and a parallel tangent computation is emitted beside it, then packed
# into a `Dual`. Low-level rules describe how each intrinsic (`add_float`, `mul_float`, …), builtin
# (`getfield`), and `%new` propagates tangents; surviving `:invoke`/`:call`s (e.g. `sin`/`cos`) go
# through `frule` dispatch, which picks up hand-written rules.
#
# Types are derived directly and exactly from the primal IR, not guessed: every shadow (tangent)
# statement shares its primal statement's type `Ti`; a `Dual{R,R}` wrapper uses `R = Ti`; a
# surviving `frule` result is `Dual{R,R}` with `R` the primal call's result type. The result is
# therefore a fully typed `IRCode` that installs as the optimization result and whose return type
# `finishinfer!` reads off via `compute_ir_rettype` — no re-inference. Branches and loops
# (`GotoNode`/`GotoIfNot`/`PhiNode`) are supported: block topology is preserved 1:1 from the primal,
# so only within-block instruction counts change. Throwing error paths are supported too: a block
# terminating in an unreachable `ReturnNode` (a domain/bounds check's throw target) is reconstructed
# primal-only (the derivative reproduces the same throw) — see `unreachable_block` below. Exception
# *handling* (`try`/`catch`) is supported as well: `UpsilonNode`/`PhiCNode` are duplicated into
# primal + shadow copies and `EnterNode`/`:leave`/`:pop_exception` carry over as control markers.
# Returns `nothing` to bail on still-unsupported constructs — e.g. a `Core.Builtin` with no rule
# (array/memoryref indexing) or a vararg call — and the caller then bails gracefully.
# ===========================================================================

const _Intr = Core.Intrinsics

_calleeval(@nospecialize(x)) =
    isa(x, GlobalRef) ? (isdefined(x.mod, x.name) ? getglobal(x.mod, x.name) : nothing) :
    isa(x, QuoteNode) ? x.value : x

# Build the dualized IRCode for `impl_mi` (a `dualized_impl` specialization) from the primal's
# optimized IRCode `pir`. `n` = number of dual arguments (= primal args incl. #self#).
# `pir_is_vararg` selects the argument-unpacking prologue: `false` for an ordinary primal (scalar
# positional args, first-order/base case), `true` when `pir` is itself an order-(k-1) dual IR (a
# vararg `dualized_impl` taking one tuple of Duals — the higher-order case; see the prologue below).
# Returns `ir::IRCode` or `nothing`.
function dualize_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int; pir_is_vararg::Bool=false)
    pstmts = pir.stmts
    N = length(pstmts)
    # Defensive: a leading PhiNode in block 1 (predecessor-free) would collide with the
    # unconditional arg-extraction prologue below, which must land before any real phi. Not
    # observed in practice (slot2ssa! always keeps block 1 phi-free), but bail rather than emit
    # invalid IR if it ever occurs.
    isa(pstmts[1][:stmt], Core.PhiNode) && return nothing

    # Embed the actual (stable, singleton) function objects as literals rather than `GlobalRef`s
    # to a non-Core/Base module: `Core.Compiler.verify_ir` rejects a bare `GlobalRef` used directly
    # as a value unless its binding is proven constant across the IR's valid worlds, which these
    # ADNext-module bindings aren't considered to be even though they never change identity.
    # Embed the actual functions as literals (see comment above re: value-position GlobalRefs).
    zerotang_g   = zero_tangent       # runtime zero-tangent fallback for non-diff results
    buildtang_g  = build_tangent      # construct a struct's `Tangent`/`MutableTangent` shadow
    gettfield_g  = get_tangent_field  # read a field's tangent out of a `Tangent`/`MutableTangent`
    fruleg = frule
    getf   = GlobalRef(Core, :getfield)
    intrg(name) = GlobalRef(Core.Intrinsics, name)

    # Tangent type of a primal type — drives every shadow SSA's declared type. For scalars
    # (`Float64`, `Float32`, `Complex`, …) `tt(T) == T`, so scalar shadows are unchanged from the
    # old same-typed scheme; for general structs it is a `Tangent`/`MutableTangent`, for tuples a
    # per-field tangent tuple, and for a `Dual` carrier the `Dual` itself (see `dual.jl`).
    tt(@nospecialize T) = tangent_type(T)

    code = Any[]; types = Any[]
    emit!(ex, @nospecialize(ty)) = (push!(code, ex); push!(types, ty); Core.SSAValue(length(code)))
    opf(name, ty, args...) = emit!(Expr(:call, intrg(name), args...), ty)
    # Construct a `Dual{P,T}` directly with `%new` (an immutable-struct construction, no dispatch /
    # allocation) rather than a dynamic `Dual(...)` call the inliner couldn't reach in synthetic IR.
    dual!(@nospecialize(P), @nospecialize(T), @nospecialize(p), @nospecialize(t)) =
        emit!(Expr(:new, Dual{P,T}, p, t), Dual{P,T})
    # The tangent of a compile-time-constant primal is its zero tangent, computed now and embedded
    # as a literal so no call survives into the IR. `zero_tangent` handles singletons (-> the zero
    # tangent, e.g. `NoTangent()` for a function/`Int`), scalars (`0.0`), `Dual` carriers (a
    # same-typed zero), and composite constants (a `Tangent`/tuple of zeros).
    const_tangent(@nospecialize x) = zero_tangent(isa(x, QuoteNode) ? x.value : x)
    # Zero tangent for a *computed* primal value of type `Ti` (the tangent of a non-differentiable
    # operation's result). `NoTangent()` when the tangent type is trivial (`Int`, `Bool`, …), a
    # literal `zero(Ti)` for a concrete `Number`, otherwise a runtime `zero_tangent` on the primal.
    function zero_shadow(@nospecialize(Ti), @nospecialize(primal_ssa))
        T = tt(Ti)
        T === NoTangent && return NoTangent()
        (Ti isa DataType && isconcretetype(Ti) && Ti <: Number) && return zero(Ti)::Ti
        return emit!(Expr(:call, zerotang_g, primal_ssa), T)
    end

    dualparams = impl_mi.specTypes.parameters[2:end]     # the Dual{…} argument types
    vararg_tt = Tuple{dualparams...}                     # dualargs is one vararg tuple (Argument 2)

    primal = Vector{Any}(undef, N); shadow = Vector{Any}(undef, N)
    if !pir_is_vararg
        # Base case: `pir` has scalar positional args (`Argument(k)` is the k-th primal arg). Unpack
        # each incoming `Dual` into its primal/tangent, indexed by argument position so `presolve`/
        # `tresolve` can treat `Argument`s uniformly with `SSAValue`s.
        parg = Vector{Any}(undef, n); targ = Vector{Any}(undef, n)
        for i in 1:n
            Di = dualparams[i]
            di = emit!(Expr(:call, getf, Core.Argument(2), i), Di)
            parg[i] = emit!(Expr(:call, getf, di, 1), _dual_primal_type(Di))
            targ[i] = emit!(Expr(:call, getf, di, 2), _dual_tangent_type(Di))
        end
    else
        # Higher-order case: `pir` is a vararg `dualized_impl` whose only real argument is
        # `Argument(2)`, the tuple of order-(k-1) dual args (accessed as `getfield(Argument(2), i)`).
        # Every arg is nested uniformly (the function included), so each `Di` has `primal_type(Di) <:
        # Dual`. Peel one level off each: reconstruct a primal tuple (each element the order-(k-1) dual
        # value) and a tangent tuple (its new, k-th tangent direction), so the existing getfield branch
        # resolves `getfield(Argument(2), i)` with no changes. `pir` has exactly two arguments (#self#
        # and the tuple), so `parg`/`targ` are length 2 regardless of `n`.
        pelts = Vector{Any}(undef, n); telts = Vector{Any}(undef, n)
        ptys  = Vector{Any}(undef, n); ttys  = Vector{Any}(undef, n)
        for i in 1:n
            Di = dualparams[i]
            _dual_primal_type(Di) <: Dual || return nothing  # non-uniformly-nested seed at order ≥2 → bail
            di = emit!(Expr(:call, getf, Core.Argument(2), i), Di)
            pelts[i] = emit!(Expr(:call, getf, di, 1), _dual_primal_type(Di));  ptys[i] = _dual_primal_type(Di)
            telts[i] = emit!(Expr(:call, getf, di, 2), _dual_tangent_type(Di)); ttys[i] = _dual_tangent_type(Di)
        end
        ctuple = GlobalRef(Core, :tuple)
        ptuple = emit!(Expr(:call, ctuple, pelts...), Tuple{ptys...})
        ttuple = emit!(Expr(:call, ctuple, telts...), Tuple{ttys...})
        @assert Tuple{ptys...} === pir.argtypes[2]     # reconstructed tuple == pir's own arg type
        parg = Any[dualized_impl, ptuple]; targ = Any[NoTangent(), ttuple]
    end

    presolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? primal[x.id] : isa(x, Core.Argument) ? parg[x.n] : x
    tresolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? shadow[x.id] :
        isa(x, Core.Argument) ? targ[x.n] :
        const_tangent(x)                                 # constant tangent (literal)

    function frule_split!(fpos, actual, R)
        # Dual(callee, NoTangent()) and each Dual(arg_primal, arg_tangent), constructed via %new.
        # Embed the *resolved* callee value (e.g. the `sin` function object) rather than the raw
        # callee-position AST node: a bare `GlobalRef` outside Core/Base used directly as a value
        # is rejected by `Core.Compiler.verify_ir` unless its binding is proven constant across the
        # IR's valid worlds, which a plain `GlobalRef(Main, :sin)` isn't. Falls back to the raw node
        # (rare: a dynamic/unresolvable callee) when resolution fails, matching prior behavior.
        fval = _calleeval(fpos)
        fcallee = fval === nothing ? fpos : fval
        ftype = fval === nothing ? _valtype(fpos) : typeof(fval)
        fd = dual!(ftype, NoTangent, fcallee, NoTangent())
        # Each argument dual is `Dual{P, tangent_type(P)}` (the Mooncake invariant). For scalar
        # primals `tangent_type(P) == P`, so this matches the old `Dual{P,P}`.
        dualtys = Any[Dual{_optype(pir,a),tt(_optype(pir,a))} for a in actual]
        duals = Any[dual!(_optype(pir,a), tt(_optype(pir,a)), presolve(a), tresolve(a)) for a in actual]
        # Emit the surviving high-level rule as a static `:invoke` to a compiled `CodeInstance`
        # when we can resolve one (direct, unboxed call); otherwise a plain non-inlined `:call`.
        ci = frule_codeinstance(interp, ftype, dualtys)
        DR = Dual{R,tt(R)}
        dd = ci === nothing ? emit!(Expr(:call, fruleg, fd, duals...), DR) :
                              emit!(Expr(:invoke, ci, fruleg, fd, duals...), DR)
        return emit!(Expr(:call, getf, dd, 1), R), emit!(Expr(:call, getf, dd, 2), tt(R))
    end

    # Block topology (block count, order, preds/succs) is preserved 1:1 from the primal: this
    # transform only expands each original statement into more instructions, never splits, merges,
    # or reorders blocks. So `GotoNode.label`/`GotoIfNot.dest`/`PhiNode.edges` (already basic-block
    # numbers in post-optimization IRCode) carry over unchanged; only each block's `StmtRange` needs
    # recomputing, tracked live below as blocks are crossed during the single linear pass.
    nblocks = length(pir.cfg.blocks)
    block_start_new = Vector{Int}(undef, nblocks)
    block_start_new[1] = 1                          # includes the arg-extraction prologue above
    bidx = 1

    # A block that terminates in an *unreachable* `ReturnNode` (one with no `val`) is an error/throw
    # block: `GotoIfNot -> error block -> unreachable`. Every statement in it only ever leads to a
    # `throw` and never contributes to a returned value (nor reaches a live `PhiNode` — dominance
    # guarantees this), so such statements are reconstructed *primal-only* below (see the main loop):
    # the primal computation is rebuilt faithfully so the derivative raises the same error on the
    # same inputs, but no shadow/tangent is computed for it.
    unreachable_block = falses(nblocks)
    for b in 1:nblocks
        term = pstmts[pir.cfg.blocks[b].stmts.stop][:stmt]
        unreachable_block[b] = isa(term, Core.ReturnNode) && !isdefined(term, :val)
    end

    # Forward-reference patches for `PhiNode` operands not yet resolved when the phi is processed
    # (loop back-edges: the operand is defined later in the linear statement order). Keyed by the
    # referenced *original* SSA index; each entry is (target values-vector, slot, want_primal).
    # `PhiNode.values` is a plain `Vector`, mutable in place even though `PhiNode` is immutable.
    pending = Dict{Int,Vector{Tuple{Vector{Any},Int,Bool}}}()

    for i in 1:N
        while bidx < nblocks && i > pir.cfg.blocks[bidx].stmts.stop
            # A block whose every original statement is a pure alias (e.g. a bare `nothing`
            # fallthrough placeholder block, common between adjacent loops) emits no instructions
            # at all, which would leave it an empty `StmtRange`. Backfill a placeholder so every
            # block keeps at least one statement, matching the primal's own convention.
            if length(code) < block_start_new[bidx]
                push!(code, nothing); push!(types, Nothing)
            end
            bidx += 1
            block_start_new[bidx] = length(code) + 1
        end
        s = pstmts[i][:stmt]; Ti = pstmts[i][:type]
        if unreachable_block[bidx]
            # Primal-only reconstruction (see `unreachable_block` above). `shadow[i] = primal[i]` is
            # a never-consumed placeholder so `presolve`/`tresolve` on later error-path operands
            # still resolve, without emitting a bogus tangent computation.
            if isa(s, Core.ReturnNode)
                push!(code, Core.ReturnNode()); push!(types, Union{})   # unreachable terminator
            elseif isa(s, Expr) && s.head === :invoke
                # Resolve the display-callee to its value: a bare non-Core/Base `GlobalRef` in the
                # invoke's callee position is a value position `verify_ir` rejects (see frule_split!).
                fv = _calleeval(s.args[2])
                ex = Expr(:invoke, s.args[1], fv === nothing ? s.args[2] : fv,
                          (presolve(a) for a in s.args[3:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :call
                # Resolve the callee to its value too: `userefs` checks the callee operand, and a
                # bare non-Core/Base `GlobalRef` (e.g. `throw`) fails the const-binding check when
                # re-embedded in this synthetic IR's world range.
                fv = _calleeval(s.args[1])
                ex = Expr(:call, fv === nothing ? s.args[1] : fv,
                          (presolve(a) for a in s.args[2:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Expr) && s.head === :new
                ex = Expr(:new, s.args[1], (presolve(a) for a in s.args[2:end])...)
                primal[i] = emit!(ex, Ti); shadow[i] = primal[i]
            elseif isa(s, Core.PiNode)
                primal[i] = presolve(s.val); shadow[i] = primal[i]
            elseif isa(s, GlobalRef) || !isa(s, Expr)
                primal[i] = presolve(s); shadow[i] = primal[i]
            else
                return nothing                                          # unexpected in an error block
            end
        elseif isa(s, Core.ReturnNode)
            if !isdefined(s, :val)
                push!(code, Core.ReturnNode()); push!(types, Union{})   # unreachable terminator
            else
                p = presolve(s.val); t = tresolve(s.val)
                R = _optype(pir, s.val)
                res = dual!(R, tt(R), p, t)
                push!(code, Core.ReturnNode(res)); push!(types, Any)
            end
        elseif isa(s, Core.PiNode)
            primal[i] = presolve(s.val); shadow[i] = tresolve(s.val)
        elseif isa(s, Expr) && s.head === :new
            T = s.args[1]
            args = @view s.args[2:end]
            pf = Any[presolve(a) for a in args]
            primal[i] = emit!(Expr(:new, T, pf...), Ti)
            TT = tt(Ti)
            if T <: Dual
                # `Dual` is its own tangent type, so its shadow is a same-typed `Dual` built via
                # `%new`. Each field is its `tresolve`d tangent, except a *non-differentiable
                # singleton* field (a function/constant, whose tangent is `NoTangent` and can't fill
                # e.g. a `typeof(sin)` slot) which carries the primal value through unchanged. This is
                # what lets a `Dual{typeof(sin),NoTangent}` be re-dualized at higher order.
                tf = Any[_nondiff_singleton(fieldtype(T, j)) ? presolve(args[j]) : tresolve(args[j])
                         for j in eachindex(args)]
                shadow[i] = emit!(Expr(:new, T, tf...), Ti)
            elseif TT === NoTangent
                # Whole aggregate is non-differentiable (e.g. `Tuple{Int,Int}`, a fieldless struct).
                shadow[i] = NoTangent()
            elseif T <: Tuple || T <: NamedTuple
                # Tuple/NamedTuple tangents are same-shape but tangent-typed: a non-differentiable
                # slot holds `NoTangent()`, differentiable slots hold their tangent. Built via `%new`
                # of the (tangent-typed) aggregate.
                tf = Any[tt(fieldtype(T, j)) === NoTangent ? NoTangent() : tresolve(args[j])
                         for j in eachindex(args)]
                shadow[i] = emit!(Expr(:new, TT, tf...), TT)
            else
                # General (im)mutable user struct → `Tangent`/`MutableTangent` via `build_tangent`,
                # which wraps possibly-undef fields in `PossiblyUninitTangent` and fills the backing
                # `NamedTuple`. `tresolve`d field tangents already carry `NoTangent()` for non-diff
                # fields, which `build_tangent` places in the corresponding `NoTangent` slot.
                tf = Any[tresolve(args[j]) for j in eachindex(args)]
                shadow[i] = emit!(Expr(:call, buildtang_g, T, tf...), TT)
            end
        elseif isa(s, Expr) && (s.head === :call || s.head === :invoke)
            fpos = s.head === :invoke ? s.args[2] : s.args[1]
            actual = s.head === :invoke ? s.args[3:end] : s.args[2:end]
            f = _calleeval(fpos)
            if isa(f, Core.IntrinsicFunction)
                nm = nameof(f)
                if nm === :add_float || nm === :add_float_fast
                    primal[i] = opf(:add_float, Ti, presolve(actual[1]), presolve(actual[2]))
                    shadow[i] = opf(:add_float, Ti, tresolve(actual[1]), tresolve(actual[2]))
                elseif nm === :sub_float || nm === :sub_float_fast
                    primal[i] = opf(:sub_float, Ti, presolve(actual[1]), presolve(actual[2]))
                    shadow[i] = opf(:sub_float, Ti, tresolve(actual[1]), tresolve(actual[2]))
                elseif nm === :neg_float || nm === :neg_float_fast
                    primal[i] = opf(:neg_float, Ti, presolve(actual[1]))
                    shadow[i] = opf(:neg_float, Ti, tresolve(actual[1]))
                elseif nm === :mul_float || nm === :mul_float_fast
                    pa, pb = presolve(actual[1]), presolve(actual[2])
                    ta, tb = tresolve(actual[1]), tresolve(actual[2])
                    primal[i] = opf(:mul_float, Ti, pa, pb)
                    shadow[i] = opf(:add_float, Ti, opf(:mul_float, Ti, pa, tb), opf(:mul_float, Ti, ta, pb))
                elseif nm === :div_float || nm === :div_float_fast
                    pa, pb = presolve(actual[1]), presolve(actual[2])
                    ta, tb = tresolve(actual[1]), tresolve(actual[2])
                    primal[i] = opf(:div_float, Ti, pa, pb)
                    num = opf(:sub_float, Ti, opf(:mul_float, Ti, ta, pb), opf(:mul_float, Ti, pa, tb))
                    shadow[i] = opf(:div_float, Ti, num, opf(:mul_float, Ti, pb, pb))
                else
                    # non-differentiable intrinsic (int arithmetic, comparisons, conversions):
                    # primal computed, tangent is the zero tangent of the primal result (see
                    # `zero_shadow`: `NoTangent()` for `Int`/`Bool`, a literal `zero` for a concrete
                    # Number, else a runtime `zero_tangent`).
                    primal[i] = opf(nm, Ti, (presolve(a) for a in actual)...)
                    shadow[i] = zero_shadow(Ti, primal[i])
                end
            elseif f === Core.getfield
                Pobj = _optype(pir, actual[1])
                primal[i] = emit!(Expr(:call, getf, presolve(actual[1]), actual[2]), Ti)
                TT = tt(Ti)
                if TT === NoTangent
                    shadow[i] = NoTangent()
                elseif Pobj <: Dual || Pobj <: Tuple || Pobj <: NamedTuple
                    # Same-shape tangents: index/name the shadow aggregate directly.
                    shadow[i] = emit!(Expr(:call, getf, tresolve(actual[1]), actual[2]), TT)
                else
                    # General struct: read the field's tangent out of the `Tangent`/`MutableTangent`.
                    shadow[i] = emit!(Expr(:call, gettfield_g, tresolve(actual[1]), actual[2]), TT)
                end
            elseif isa(f, Core.Builtin)
                return nothing                              # other builtins: bail
            else
                primal[i], shadow[i] = frule_split!(fpos, actual, Ti)
            end
        elseif isa(s, Core.GotoNode)
            push!(code, Core.GotoNode(s.label)); push!(types, Any)
        elseif isa(s, Core.GotoIfNot)
            push!(code, Core.GotoIfNot(presolve(s.cond), s.dest)); push!(types, Any)
        elseif isa(s, Core.PhiNode)
            k = length(s.values)
            pvals = Vector{Any}(undef, k); tvals = Vector{Any}(undef, k)
            for j in 1:k
                isassigned(s.values, j) || continue    # mirror the primal's own unassigned slot
                v = s.values[j]
                if isa(v, Core.SSAValue) && !isassigned(primal, v.id)
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (pvals, j, true))
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (tvals, j, false))
                else
                    pvals[j] = presolve(v); tvals[j] = tresolve(v)
                end
            end
            primal[i] = emit!(Core.PhiNode(s.edges, pvals), Ti)
            shadow[i] = emit!(Core.PhiNode(s.edges, tvals), tt(Ti))
        elseif isa(s, Core.UpsilonNode)
            # try/catch: a value live into a handler is captured by an `UpsilonNode` (at a def or
            # right before the `:enter`) and collected by a `PhiCNode` at the handler top. Duplicate
            # into a primal + shadow upsilon, exactly like a `PhiNode`. An unassigned `ϒ ()` slot
            # (a live-but-undefined capture) is mirrored as an empty upsilon in both copies.
            if isdefined(s, :val)
                primal[i] = emit!(Core.UpsilonNode(presolve(s.val)), Ti)
                shadow[i] = emit!(Core.UpsilonNode(tresolve(s.val)), tt(Ti))
            else
                primal[i] = emit!(Core.UpsilonNode(), Ti)
                shadow[i] = emit!(Core.UpsilonNode(), tt(Ti))
            end
        elseif isa(s, Core.PhiCNode)
            # Collects `UpsilonNode`s at a handler entry. Its operands are `SSAValue`s that must
            # reference upsilons (`verify_ir`), so the primal-phic references the primal upsilons and
            # the shadow-phic the shadow ones — reusing `primal[v.id]`/`shadow[v.id]`. The same
            # `pending` forward-ref mechanism as `PhiNode` covers a capture defined later in linear
            # order (a try/catch inside a loop).
            k = length(s.values)
            pvals = Vector{Any}(undef, k); tvals = Vector{Any}(undef, k)
            for j in 1:k
                isassigned(s.values, j) || continue
                v = s.values[j]
                if isa(v, Core.SSAValue) && !isassigned(primal, v.id)
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (pvals, j, true))
                    push!(get!(() -> Tuple{Vector{Any},Int,Bool}[], pending, v.id), (tvals, j, false))
                else
                    pvals[j] = presolve(v); tvals[j] = tresolve(v)
                end
            end
            primal[i] = emit!(Core.PhiCNode(pvals), Ti)
            shadow[i] = emit!(Core.PhiCNode(tvals), tt(Ti))
        elseif isa(s, Core.EnterNode)
            # Control marker beginning a protected region. `catch_dest` is a basic-block number
            # (topology preserved 1:1) so it carries over unchanged; the token it defines is
            # referenced by `:leave`/`:pop_exception` via its primal SSA. It terminates its block, so
            # its type must be `Any` (verify_ir). `scope`, if present, is an ordinary value operand.
            ent = isdefined(s, :scope) ?
                    Core.EnterNode(s.catch_dest, presolve(s.scope)) : Core.EnterNode(s.catch_dest)
            primal[i] = emit!(ent, Any); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :leave
            # Pops one or more `:enter` scopes, referencing the `EnterNode`(s) by `SSAValue` (or
            # `nothing`); `presolve` remaps each to the enter's primal SSA. Pure control, no value.
            push!(code, Expr(:leave, Any[presolve(a) for a in s.args]...)); push!(types, Any)
        elseif isa(s, Expr) && s.head === :pop_exception
            primal[i] = emit!(Expr(:pop_exception, presolve(s.args[1])), Ti); shadow[i] = primal[i]
        elseif isa(s, Expr) && s.head === :the_exception
            # The caught exception object has no meaningful tangent (non-differentiable): keep the
            # primal, give it a zero tangent like other non-diff results.
            primal[i] = emit!(Expr(:the_exception), Ti)
            shadow[i] = zero_shadow(Ti, primal[i])
        elseif isa(s, GlobalRef) || !isa(s, Expr)
            primal[i] = presolve(s); shadow[i] = tresolve(s)
        else
            return nothing
        end
        if haskey(pending, i)
            for (arr, slot, wantp) in pending[i]
                arr[slot] = wantp ? primal[i] : shadow[i]
            end
            delete!(pending, i)
        end
    end
    if length(code) < block_start_new[nblocks]      # the final block emitted nothing; see above
        push!(code, nothing); push!(types, Nothing)
    end
    isempty(pending) || return nothing               # unreachable on well-formed IR; bail, don't emit invalid IR

    len = length(code)
    stream = CC.InstructionStream(len)
    for i in 1:len
        stream.stmt[i] = code[i]
        stream.type[i] = types[i]
        stream.flag[i] = CC.IR_FLAG_NULL
    end
    new_blocks = Vector{CC.BasicBlock}(undef, nblocks)
    for b in 1:nblocks
        lo = block_start_new[b]
        hi = b == nblocks ? len : block_start_new[b+1] - 1
        ob = pir.cfg.blocks[b]
        new_blocks[b] = CC.BasicBlock(CC.StmtRange(lo, hi), copy(ob.preds), copy(ob.succs))
    end
    cfg = CC.CFG(new_blocks, Int[bb.stmts.stop + 1 for bb in new_blocks])   # index[b] is one past block b's end
    di  = CC.DebugInfoStream(stream.line)
    di.def = impl_mi                                # required: Core.DebugInfo(di, n) does something(di.def)
    argtypes = Any[impl_mi.specTypes.parameters[1], vararg_tt]
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[])
    CC.verify_ir(ir)                                # a failure here is a bug in this transform, not
                                                     # unsupported input IR — let it throw plainly
    return ir
end

# Type helpers. Types are taken directly from the primal IR (`_optype`), so they are exact rather
# than guessed; the fallbacks only cover genuine constants/globals.
_valtype(@nospecialize x) = isa(x, GlobalRef) && isdefined(x.mod, x.name) ? typeof(getglobal(x.mod, x.name)) : Any
_optype(pir, @nospecialize x) = isa(x, Core.SSAValue) ? pir.stmts[x.id][:type] :
                                isa(x, Core.Argument) ? pir.argtypes[x.n] : typeof(x)
# A field is a "non-differentiable singleton" (a function/constant, tangent = `NoTangent`) when it is
# a non-`NoTangent` singleton type. Such a field's shadow can't hold a `NoTangent` value (its slot
# type would reject it), so the primal value is carried through instead. Differentiable structs are
# not singletons, so they still take their structural tangent.
_nondiff_singleton(@nospecialize T) = T isa DataType && Base.issingletontype(T) && T !== NoTangent


function frule_body(world::UInt, source, self, dual_argtypes)
    argnames = Any[Symbol("#self#"), :dualargs]

    # Resolve the `dualized_impl` specialization for these dual argument types.
    impl_tt = Tuple{typeof(dualized_impl), dual_argtypes...}
    interp = ADInterpreter{Forward}(; world)
    match, _ = Core.Compiler.findsup(impl_tt, Core.Compiler.method_table(interp))
    if match === nothing
        return expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                                :(error("ADNext: no dualized_impl match")), true)
    end
    impl_mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance

    # Compile the dualized body under ADInterpreter -> an invoke-able CodeInstance.
    # Call typeinf_ext_toplevel directly (not CompilerPlugins.typeinf, which would recreate the
    # interpreter at tls_world_age() — stale inside a generator) so it uses `interp`'s generation
    # world; the finishinfer!/optimize seams then build and install the dual IR (see contextual.jl).
    cinst = Compiler.typeinf_ext_toplevel(interp, impl_mi, Compiler.SOURCE_MODE_ABI)

    # Trivial generated body: return invoke(dualized_impl, cinst, dualargs...)
    ci = expr_to_codeinfo(@__MODULE__(), argnames, [], (),
                          :(return invoke(dualized_impl, $cinst, dualargs...)), true)

    ci.edges = Core.MethodInstance[impl_mi]
    return ci
end

function refresh_frule()
    @eval function frule(dualargs::Dual...)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, frule_body))
    end
end
refresh_frule()
