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
          "(likely unsupported IR — e.g. exception handling (try/catch), which is not handled yet).")


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
    primal_tt = Base.to_tuple_type(Any[primal_type(P) for P in dualparams])
    pmatch, _ = CC.findsup(primal_tt, CC.method_table(interp))
    pmatch === nothing && return nothing
    isa(pmatch.method, Method) || return nothing
    pmatch.method.isva && return nothing
    primal_mi = specialize_method(pmatch.method, pmatch.spec_types, pmatch.sparams)::MethodInstance
    return (primal_mi, length(dualparams))
end

# Build the dualized `IRCode` for a `dualized_impl` specialization from the primal's optimized
# `IRCode`. Returns the dual `IRCode` or `nothing` (unsupported IR → bail).
function build_dual_ir(interp::ADInterpreter, impl_mi::MethodInstance)
    info = primal_of_impl(interp, impl_mi)
    info === nothing && return nothing
    primal_mi, n = info
    world = CC.get_inference_world(interp)
    # Optimized primal IR via the internal `typeinf_ircode`. A NativeInterpreter is used so that
    # `sin`/`cos` and other hand-ruled functions survive as `:invoke`s (routed through `frule`).
    pir, _ = CC.typeinf_ircode(CC.NativeInterpreter(world), primal_mi, nothing)
    pir === nothing && return nothing
    return dualize_to_ircode(interp, impl_mi, pir, n)
end

# Resolve and compile the `frule(Dual{typeof(f),NoFData}, dualargs...)` rule for a surviving
# high-level call to an *invoke-able `CodeInstance`*, so the dualized IR can emit a static
# `:invoke` (mirroring how the primal IR keeps `sin`/`cos` as `:invoke`s to a `CodeInstance`).
# `:invoke` targets *must* be `CodeInstance`s: `collectinvokes!` only JITs those, so a bare
# `MethodInstance` would fall back to a boxed dynamic call. Returns `nothing` if unresolved.
function frule_codeinstance(interp::ADInterpreter, @nospecialize(ftype), dual_argtypes)
    frule_tt = Tuple{typeof(frule), Dual{ftype,NoFData}, dual_argtypes...}
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
# so only within-block instruction counts change. Returns `nothing` to bail on exception handling
# (`EnterNode`/`PhiCNode`/`UpsilonNode`) or other unsupported constructs (the caller then bails
# gracefully).
# ===========================================================================

const _Intr = Core.Intrinsics

_calleeval(@nospecialize(x)) =
    isa(x, GlobalRef) ? (isdefined(x.mod, x.name) ? getglobal(x.mod, x.name) : nothing) :
    isa(x, QuoteNode) ? x.value : x

# Build the dualized IRCode for `impl_mi` (a `dualized_impl` specialization) from the primal's
# optimized IRCode `pir`. `n` = number of dual arguments (= primal args incl. #self#).
# Returns `ir::IRCode` or `nothing`.
function dualize_to_ircode(interp, impl_mi::MethodInstance, pir, n::Int)
    pstmts = pir.stmts
    N = length(pstmts)
    for i in 1:N                                    # exception handling not yet supported
        s = pstmts[i][:stmt]
        if isa(s, Core.PhiCNode) || isa(s, Core.UpsilonNode) || isa(s, Core.EnterNode)
            return nothing
        end
    end
    # Defensive: a leading PhiNode in block 1 (predecessor-free) would collide with the
    # unconditional arg-extraction prologue below, which must land before any real phi. Not
    # observed in practice (slot2ssa! always keeps block 1 phi-free), but bail rather than emit
    # invalid IR if it ever occurs.
    isa(pstmts[1][:stmt], Core.PhiNode) && return nothing

    # Embed the actual (stable, singleton) function objects as literals rather than `GlobalRef`s
    # to a non-Core/Base module: `Core.Compiler.verify_ir` rejects a bare `GlobalRef` used directly
    # as a value unless its binding is proven constant across the IR's valid worlds, which these
    # ADNext-module bindings aren't considered to be even though they never change identity.
    zerog  = struct_zero
    fruleg = frule
    getf   = GlobalRef(Core, :getfield)
    intrg(name) = GlobalRef(Core.Intrinsics, name)

    code = Any[]; types = Any[]
    emit!(ex, @nospecialize(ty)) = (push!(code, ex); push!(types, ty); Core.SSAValue(length(code)))
    opf(name, ty, args...) = emit!(Expr(:call, intrg(name), args...), ty)
    # Construct a `Dual{P,T}` directly with `%new` (an immutable-struct construction, no dispatch /
    # allocation) rather than a dynamic `Dual(...)` call the inliner couldn't reach in synthetic IR.
    dual!(@nospecialize(P), @nospecialize(T), @nospecialize(p), @nospecialize(t)) =
        emit!(Expr(:new, Dual{P,T}, p, t), Dual{P,T})
    # The tangent of a compile-time-constant primal is itself a constant: compute it now and embed
    # it as a literal, so no `struct_zero` call survives into the IR. `struct_zero` already handles
    # arbitrary structs (recursing per field) and singletons (-> `NoFData()`), so this covers
    # composite constants too (needed so a constant struct arm of a `PhiNode` gets a same-shaped
    # zero tangent rather than a type-inconsistent `NoFData()`).
    const_tangent(@nospecialize x) = struct_zero(isa(x, QuoteNode) ? x.value : x)
    # The zero tangent for a value of concrete Number type is a literal; otherwise fall back.
    zero_of_type(@nospecialize T) = (T isa DataType && isconcretetype(T) && T <: Number) ?
                                    (zero(T)::T) : nothing

    dualparams = impl_mi.specTypes.parameters[2:end]     # the Dual{…} argument types
    vararg_tt = Tuple{dualparams...}                     # dualargs is one vararg tuple (Argument 2)

    primal = Vector{Any}(undef, N); shadow = Vector{Any}(undef, N)
    parg   = Vector{Any}(undef, n); targ   = Vector{Any}(undef, n)
    for i in 1:n
        Di = dualparams[i]
        di = emit!(Expr(:call, getf, Core.Argument(2), i), Di)
        parg[i] = emit!(Expr(:call, getf, di, 1), primal_type(Di))
        targ[i] = emit!(Expr(:call, getf, di, 2), tangent_type(Di))
    end

    presolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? primal[x.id] : isa(x, Core.Argument) ? parg[x.n] : x
    tresolve(@nospecialize x) =
        isa(x, Core.SSAValue) ? shadow[x.id] :
        isa(x, Core.Argument) ? targ[x.n] :
        const_tangent(x)                                 # constant tangent (literal)

    function frule_split!(fpos, actual, R)
        # Dual(callee, NoFData()) and each Dual(arg_primal, arg_tangent), constructed via %new.
        # Embed the *resolved* callee value (e.g. the `sin` function object) rather than the raw
        # callee-position AST node: a bare `GlobalRef` outside Core/Base used directly as a value
        # is rejected by `Core.Compiler.verify_ir` unless its binding is proven constant across the
        # IR's valid worlds, which a plain `GlobalRef(Main, :sin)` isn't. Falls back to the raw node
        # (rare: a dynamic/unresolvable callee) when resolution fails, matching prior behavior.
        fval = _calleeval(fpos)
        fcallee = fval === nothing ? fpos : fval
        ftype = fval === nothing ? _valtype(fpos) : typeof(fval)
        fd = dual!(ftype, NoFData, fcallee, NoFData())
        dualtys = Any[Dual{_optype(pir,a),_optype(pir,a)} for a in actual]
        duals = Any[dual!(_optype(pir,a), _optype(pir,a), presolve(a), tresolve(a)) for a in actual]
        # Emit the surviving high-level rule as a static `:invoke` to a compiled `CodeInstance`
        # when we can resolve one (direct, unboxed call); otherwise a plain non-inlined `:call`.
        ci = frule_codeinstance(interp, ftype, dualtys)
        dd = ci === nothing ? emit!(Expr(:call, fruleg, fd, duals...), Dual{R,R}) :
                              emit!(Expr(:invoke, ci, fruleg, fd, duals...), Dual{R,R})
        return emit!(Expr(:call, getf, dd, 1), R), emit!(Expr(:call, getf, dd, 2), R)
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
        if isa(s, Core.ReturnNode)
            isdefined(s, :val) || return nothing
            p = presolve(s.val); t = tresolve(s.val)
            R = _optype(pir, s.val)
            res = dual!(R, R, p, t)
            push!(code, Core.ReturnNode(res)); push!(types, Any)
        elseif isa(s, Core.PiNode)
            primal[i] = presolve(s.val); shadow[i] = tresolve(s.val)
        elseif isa(s, Expr) && s.head === :new
            T = s.args[1]
            pf = Any[presolve(a) for a in @view s.args[2:end]]
            tf = Any[tresolve(a) for a in @view s.args[2:end]]
            primal[i] = emit!(Expr(:new, T, pf...), Ti)
            shadow[i] = emit!(Expr(:new, T, tf...), Ti)
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
                    # primal computed, tangent is the structural zero of the primal result. When the
                    # result type is a concrete Number the zero is a literal; else fall back to a
                    # runtime `struct_zero` of the computed primal.
                    primal[i] = opf(nm, Ti, (presolve(a) for a in actual)...)
                    z = zero_of_type(Ti)
                    shadow[i] = z === nothing ? emit!(Expr(:call, zerog, primal[i]), _zerotype_of(Ti)) : z
                end
            elseif f === Core.getfield
                primal[i] = emit!(Expr(:call, getf, presolve(actual[1]), actual[2]), Ti)
                shadow[i] = emit!(Expr(:call, getf, tresolve(actual[1]), actual[2]), Ti)
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
            shadow[i] = emit!(Core.PhiNode(s.edges, tvals), Ti)
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
_zerotype_of(@nospecialize T) = (T <: Number ? T : NoFData)


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
