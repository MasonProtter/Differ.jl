# Late extension load: the scenario `Contextual.at_world` exists for.
#
# `DifferForwardsOverReverseExt`'s methods — and `DifferReverse`'s own `tangent_type` overrides —
# are always defined at a strictly later world than `DifferForwards`' `frule!!`. A generator body
# dispatches at its method's `primary_world`, so before the `at_world` treatment those methods were
# invisible to the forward transform and forward-over-reverse hung (ISSUES #85).
#
# The ordinary suite loads everything up front, which is the *easy* order. What actually has to work
# is: compile and cache a derived forward carrier while only `DifferForwards` exists, then load
# `DifferReverse` (pulling in the extension), then compose. Each order runs in its own subprocess,
# since what is being tested is load order within a session.

using Test

const _PROJECT = Base.active_project()

# `h` is deliberately a composite (no hand `frule!!`), so the baseline call really does build and
# cache a `dualized_impl` carrier rather than dispatching straight to a rule.
const _COMMON = raw"""
    using DifferForwards: Dual, frule!!, NoTangent
    h(x) = x * x + 1.0
    check(r, x, dx, what) = (r.x == x && r.dx == dx) || error("$what wrong: got $r")
"""

const _FORWARDS_FIRST = """
    using DifferForwards
    $_COMMON
    check(frule!!(Dual(h, NoTangent()), Dual(2.0, 1.0)), 5.0, 4.0, "baseline forward mode")

    using DifferReverse
    using DifferReverse: rev_gradient
    g(x) = rev_gradient(h, x)[2]
    check(frule!!(Dual(g, NoTangent()), Dual(1.5, 1.0)), 3.0, 2.0, "forward-over-reverse")
    println("OK")
"""

const _REVERSE_FIRST = """
    using DifferReverse
    using DifferReverse: rev_gradient
    using DifferForwards
    $_COMMON
    check(frule!!(Dual(h, NoTangent()), Dual(2.0, 1.0)), 5.0, 4.0, "baseline forward mode")
    g(x) = rev_gradient(h, x)[2]
    check(frule!!(Dual(g, NoTangent()), Dual(1.5, 1.0)), 3.0, 2.0, "forward-over-reverse")
    println("OK")
"""

function run_in_fresh_session(script::String)
    out = IOBuffer()
    err = IOBuffer()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$_PROJECT -e $script`
    ok = success(pipeline(cmd; stdout=out, stderr=err))
    return ok, String(take!(out)), String(take!(err))
end

@testset "forward-over-reverse after a late extension load" begin
    # The regression order: `frule!!` carriers already compiled and cached before the extension's
    # methods exist. Before the fix this hung rather than failing, so it could not be caught by a
    # plain `@test` in-process — hence the subprocess.
    ok, out, err = run_in_fresh_session(_FORWARDS_FIRST)
    @test ok
    @test occursin("OK", out)
    ok || @info "forwards-first subprocess stderr" err

    # The reverse order must keep working too; nothing here should depend on which sibling package
    # happens to load first.
    ok2, out2, err2 = run_in_fresh_session(_REVERSE_FIRST)
    @test ok2
    @test occursin("OK", out2)
    ok2 || @info "reverse-first subprocess stderr" err2
end
