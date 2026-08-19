# Shared test infrastructure: IR-verification wrappers and finite-difference checks reused
# across many test files. Not test fixtures — nothing here is itself differentiated. Reverse-
# mode-specific helpers (checkverify_prealloc/checkverify_rev, tape-balance/traffic/size
# checks) live in DifferReverse.jl/test/testutils.jl instead.

using Test
using DifferForwards
using DifferForwards: code_dual_ircode, Dual, Inactive

# Central finite difference, one argument.
central_diff(f, x; h=1e-6) = (f(x + h) - f(x - h)) / 2h

# Central finite difference of a 2-argument function, w.r.t. argument `k` (1 or 2).
central_diff(f, x, y, k::Int; h=1e-6) =
    k == 1 ? (f(x+h, y) - f(x-h, y)) / 2h : (f(x, y+h) - f(x, y-h)) / 2h

# A `Dual` carrier for an argument the caller holds constant.
const_dual(x) = Dual(x, Inactive())

# Forward-mode dualized IR is legal (order-1). `inactive` names argument positions held constant.
checkverify(f, at; inactive=()) =
    Core.Compiler.verify_ir(code_dual_ircode(f, at; inactive)[1])

# The bail message for a function Differ declines to dualize, or `nothing` if it dualizes fine.
# Every graceful bail is supposed to name a *reason*, so tests assert on this rather than just on
# "it threw".
function bail_reason(f, at; inactive=())
    try
        code_dual_ircode(f, at; inactive)
        return nothing
    catch e
        return sprint(showerror, e)
    end
end

# Forward-mode dualized IR is legal at a given nesting order.
checkverify2(f, at; order=2, inactive=()) =
    Core.Compiler.verify_ir(code_dual_ircode(f, at; order, inactive)[1])
