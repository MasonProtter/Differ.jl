# Tier 2: post-optimization forward-AD on typed `IRCode`.
#
# Transforms a primal method's fully optimized `IRCode` into a *dualized* `IRCode` using the
# "split-shadow" scheme: the primal computation is reconstructed and a parallel tangent
# computation is emitted beside it, then packed into a `Dual`. Low-level rules describe how
# each intrinsic (`add_float`, `mul_float`, …), builtin (`getfield`), and `%new` propagates
# tangents; surviving `:invoke`/`:call`s go through `frule` dispatch (Tier 1).
#
# The result is a *typed* IRCode (types are computed directly from the primal IR — shadow
# statements share their primal's type, a surviving call yields `Dual{R,R}`), so it can be
# installed as the optimization result and the `CodeInstance` return type *determined* from it,
# with no re-inference. Straight-line only; returns `nothing` to bail (caller falls back).

const _Intr = Core.Intrinsics

_calleeval(@nospecialize(x)) =
    isa(x, GlobalRef) ? (isdefined(x.mod, x.name) ? getglobal(x.mod, x.name) : nothing) :
    isa(x, QuoteNode) ? x.value : x

# Build the dualized IRCode for `impl_mi` (a `dualized_impl` specialization) from the primal's
# optimized IRCode `pir`. `n` = number of dual arguments (= primal args incl. #self#).
# Returns `(ir::IRCode, rettype)` or `nothing`.
function dualize_to_ircode(impl_mi::MethodInstance, pir, n::Int)
    pstmts = pir.stmts
    N = length(pstmts)
    for i in 1:N                                    # straight-line only
        s = pstmts[i][:stmt]
        if isa(s, Core.GotoNode) || isa(s, Core.GotoIfNot) || isa(s, Core.PhiNode) ||
           isa(s, Core.PhiCNode) || isa(s, Core.UpsilonNode) || isa(s, Core.EnterNode)
            return nothing
        end
    end

    dualg  = GlobalRef(@__MODULE__(), :Dual)
    zerog  = GlobalRef(@__MODULE__(), :struct_zero)
    fruleg = GlobalRef(@__MODULE__(), :frule)
    nofg   = GlobalRef(@__MODULE__(), :NoFData)
    getf   = GlobalRef(Core, :getfield)
    intrg(name) = GlobalRef(Core.Intrinsics, name)

    code = Any[]; types = Any[]
    emit!(ex, @nospecialize(ty)) = (push!(code, ex); push!(types, ty); Core.SSAValue(length(code)))
     opf(name, ty, args...) = emit!(Expr(:call, intrg(name), args...), ty)

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
        emit!(Expr(:call, zerog, x), _zerotype(x))       # constant tangent

    function frule_split!(fpos, actual, R)
        fz = emit!(Expr(:call, zerog, fpos), NoFData)
        fd = emit!(Expr(:call, dualg, fpos, fz), Dual{_valtype(fpos),NoFData})
        duals = Any[emit!(Expr(:call, dualg, presolve(a), tresolve(a)),
                          Dual{_optype(pir,a),_optype(pir,a)}) for a in actual]
        dd = emit!(Expr(:call, fruleg, fd, duals...), Dual{R,R})
        return emit!(Expr(:call, getf, dd, 1), R), emit!(Expr(:call, getf, dd, 2), R)
    end

    for i in 1:N
        s = pstmts[i][:stmt]; Ti = pstmts[i][:type]
        if isa(s, Core.ReturnNode)
            isdefined(s, :val) || return nothing
            p = presolve(s.val); t = tresolve(s.val)
            res = emit!(Expr(:call, dualg, p, t), Dual{_optype(pir,s.val), _optype(pir,s.val)})
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
                    # primal computed, tangent is the structural zero of the primal result.
                    primal[i] = opf(nm, Ti, (presolve(a) for a in actual)...)
                    shadow[i] = emit!(Expr(:call, zerog, primal[i]), _zerotype_of(Ti))
                end
            elseif f === Core.getfield
                primal[i] = emit!(Expr(:call, getf, presolve(actual[1]), actual[2]), Ti)
                shadow[i] = emit!(Expr(:call, getf, tresolve(actual[1]), actual[2]), Ti)
            elseif isa(f, Core.Builtin)
                return nothing                              # other builtins: bail
            else
                primal[i], shadow[i] = frule_split!(fpos, actual, Ti)
            end
        elseif isa(s, GlobalRef) || !isa(s, Expr)
            primal[i] = presolve(s); shadow[i] = tresolve(s)
        else
            return nothing
        end
    end

    len = length(code)
    stream = CC.InstructionStream(len)
    for i in 1:len
        stream.stmt[i] = code[i]
        stream.type[i] = types[i]
        stream.flag[i] = CC.IR_FLAG_NULL
    end
    cfg = CC.CFG([CC.BasicBlock(CC.StmtRange(1, len), Int[], Int[])], Int[1])
    di  = CC.DebugInfoStream(stream.line)
    di.def = impl_mi                                # required: Core.DebugInfo(di, n) does something(di.def)
    argtypes = Any[impl_mi.specTypes.parameters[1], vararg_tt]
    ir = CC.IRCode(stream, cfg, di, argtypes, Expr[], CC.VarState[])
    rettype = types[end-0]  # type of the returned Dual (second-to-last emitted, before ReturnNode)
    # the ReturnNode itself has type Any; the Dual it returns is the last SSAValue emitted
    retval = (code[end]::Core.ReturnNode).val
    rettype = isa(retval, Core.SSAValue) ? types[retval.id] : Any
    return ir, rettype
end

# Type helpers (best-effort; conservative fallbacks keep the IR well-formed).
_valtype(@nospecialize x) = isa(x, GlobalRef) && isdefined(x.mod, x.name) ? typeof(getglobal(x.mod, x.name)) : Any
_optype(pir, @nospecialize x) = isa(x, Core.SSAValue) ? pir.stmts[x.id][:type] :
                                isa(x, Core.Argument) ? pir.argtypes[x.n] : typeof(x)
_zerotype(@nospecialize x) = (T = typeof(x); T <: Number ? T : NoFData)
_zerotype_of(@nospecialize T) = (T <: Number ? T : NoFData)
