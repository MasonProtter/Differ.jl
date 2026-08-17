# Writing custom rules for Differ.jl

These docs are currently under construction, and need a lot more attention to be properly pedagogical. For now, I will attempt to show you the basics that you need to know to write custom rules for Differ.jl forwards and reverse mode.

The rule system is heavily based on Mooncake's rule system, with so if you are confused by something you see here, it may be good to go read https://chalk-lab.github.io/Mooncake.jl/stable/understanding_mooncake/rule_system/ . It might also be a good idea if you're new to AD in general to read a bit about ChainRules.jl https://juliadiff.org/ChainRulesCore.jl/stable/ since the documentation there is excellent, though note that ChainRules does have a slightly different model from Differ and Mooncake, and lacks mutation support.

## Writing an `frule!!` for Forwards mode AD

Forwards mode AD is a bit simpler than reverse. Essentially, if we have some function 

```julia
function foo(x::Float64)
    y = sin(x)
    z = y^2
    return z + 1
end
```
then forwards mode automatic differentiation is concerned with calculating `dfoo/dx` by re-writing this into a new function of the form
```julia
function DifferForwards.frule!!(::Dual{typeof(foo)}, (;x, dx)::Dual{Float64})
    (; y, dy) = Dual(sin(x), var"d(sin(x))/dx" * dx)
    (; z, dz) = Dual(y^2, var"d(y^2)/dx" * dy)
    return Dual(z+1, var"d(z+1)/dz" * dz)
end
```
A `Dual` holds two objests: one represents the "primal" value in the regular function evaluation, and the other represents the derivative value. 

Differ will automatically generate `frule!!` methods for a function like `foo` at compile time, but if there is a part of your calculation which Differ doesn't understand, but you know the derivative of, you can write a custom `frule!!` method to allow it to differentiate your function. 

For example, if `Differ` didn't know the derivative of `sin`, you could write a custom method
```julia
function DifferForwards.frule!!(::Dual{typeof(sin)}, (; x,dx)::Dual{Float64})
    Dual(sin(x), cos(x)*dx)
end
```
and then it should automatically use your custom rule inside of the automatic derivative of `foo`. 


This is harder to see with scalar code, but `frule!!` on `Dual`s is essentially concerned with calculating a Jacobian-vector-product, i.e.
```julia
frule!!(::Dual(f, NoTangent()), Dual(a, b))
```
will calculate ``J[f](a) \cdot b`` where ``J[f](a)`` is the Jacobian of `f` evaluated at `a`. By evaulating the `frule!!` across a whole basis, we can generate the whole Jacobian.

Handling of mutation in forwards mode is also easy: if you mutate `x`, then you do the equivalent derivative mutations to `dx`. Suppose we have the function
```julia
function f!(u, v) 
    u[1] = 2*v[1]^2
    nothing
end
```
then the appropriate `frule!!` (and indeed the one Differ would generate automatically) is
```julia
function DifferForwards.frule!!(::Dual{f!}, udu::Dual, vdv::Dual)
    u, du = udu.x, udu.dx
    v, dv = vdv.x, vdv.dx
    du[1] = 2 * (2v[1]) * dv[1]
    nothing
end
```

## Writing a `rrule!!` for reverse mode AD

Reverse mode AD is the *transpose* of forwards mode AD. Instead of calculating a JVP (Jacobian-vector-product) ``J[f](a) \cdot b``, reverse mode calculates the VJP (vector-jacobian product) ``b^\dagger J[f](a)``. This is a trivial difference in simple programs, but it's enormously consequential for complicated programs.

For a function with many inputs and few outputs, the full jacobian can be reconstructed with fewer VJP evaluations than JVP evaluations, which is the reason that this technique is so impactful in fields like optimization where there is a scalar cost function, and unboundedly large parameter sets one may wish to differentiate with respect to.

What does it mean to transpose a sequential program like `foo`? 

To borrow some terminology from ChainRules, forwards mode AD is about taking a *wiggle* in the input space and asking how much that wiggle makes the output space *wobble*. Conversely, reverse mode AD is concerned with taking an already determined *wobble* in the output space, and asking how much *wiggle* that corresponds to in the input space. 

In fancier language, if forwards mode calculates the [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)) on a function, then reverse mode is concerned with the [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)) of that function.

So, what does reverse mode AD of `foo` look like?

We take our regular primal function, and have to run it *backwards* in order to propagate output *wobbles* into input *wiggles*.

In the simplest case with a function like `sin`, that just looks like

```julia
function DifferReverse.rrule!!(::CoDual{typeof(sin),NoFData}, ctx::AbstractCtx, (; x)::CoDual{Float64,NoFData})
    sinx, cosx = sincos(x)
    fwd_result = CoDual(sinx, NoFData())
    sin_pullback(dy) = (NoRData(), dy * cosx) # dy * d/dx sin(x)

    fwd_result, sin_pullback
end
```
This `rrule!!` takes in bunch of `CoDual` arguments (and a context argument `ctx` that we won't worry about for now),
and then computes the primal result `sin(x)`, and returns that together with a pullback function `sin_pullback` which computes
the derivative of sin, multiplied by the *wobble* `dy`.

What are these funky types `NoFData` and `NoRData`? Differ (like Mooncake which is where this convention comes from), splits information about derivatives into information that needs to be present during the *forwards pass* (evaluating the primal): FData, and information that's only needed during the *reverse pass* (evaluating the pullback): RData. Generally speaking, FData is where mutable structs and arrays go, and RData is where immutable objects go. This distinction is important when mutation comes into play later.

Note that `sin_pullback` returns a `Tuple` with two members: `NoRData()` and the derivative w.r.t. x. A pullback must return a `Tuple` of outputs, one for each input to `rrule!!` including the *function* being differentiated. In this case, that's `sin` which has `NoFData` *and* `NoRData` so we know it's derivative is a `NoTangent()`. That's because `sin` is just a constant singleton and is not a differentiable argument. However, objects like closures *can* carry `FData` or `RData` in general, so this needs to be tracked. 

### `rrule!!` for a function with multiple steps

Okay, so that's a one-line program. What about something with multiple steps like `foo`? 

Repeating again our primal,
```julia
function foo(x::Float64)
    y = sin(x)
    z = y^2
    return z + x
end
```

An `rrule!!` for this function would look like

```julia
function DifferReverse.rrule!!(::CoDual{typeof(foo), NoFData}, ctx::AbstractCtx, (; x)::CoDual{Float64, NoFData})
    # First we run the function fowards, evaluating the primal and collecting pullbacks:
    (; y), y_pullback = rrule!!(CoDual(sin, NoFData()), ctx, CoDual(x, NoFData()))
    (; z), z_pullback = rrule!!(CoDual(^,   NoFData()), ctx, CoDual(y, NoFData()), CoDual(2, NoFData()))
    (; w), w_pullback = rrule!!(CoDual(+,   NoFData()), ctx, CoDual(z, NoFData()), CoDual(1, NoFData()))
    
    # and we combine this with a function that generates the pullback, running through the program backwards
    function foo_pullback(dw)
        (_, dz, _) = w_pullback(dw)
        (_, dy, _) = z_pullback(dz)
        (_, dx)    = y_pullback(dy)
        return (NoRData(), dx)
    end
    return CoDual(w, NoFData()), foo_pullback # rrule!! returns the result of foo(x) (as a CoDual), together with the pullback closure
end
```

Here it becomes a bit more clear *why* it's called reverse mode AD. The pullback essentially runs the function *backwards*. 

Of course, if someone were to actually write a rrule!! for `foo` by hand, this could all be simplified significantly to a one-liner, but I split everything to show what Differ itself does when it encounters this function.

### `rrule!!` for a mutating function

What about mutation? Well, bringing back our friend `f!`, it's time to meet `FData` for the first time:
```julia
function f!(u, v) 
    u[1] = 2*v[1]^2
    nothing
end
```

This function has a `rrule!!` that looks like

```julia
function rrule!!(::CoDual{typeof(f!),NoFData}, ::AbstractCtx,
                 ucd::CoDual{Vector{Float64},Vector{Float64}},
                 vcd::CoDual{Vector{Float64},Vector{Float64}})
    u, du = primal(ucd), tangent(ucd)
    v, dv = primal(vcd), tangent(vcd)

    old_u1  = u[1]   # save the initial value of u
    old_du1 = du[1]  # save the initial value of du
    
    u[1] = (2 * v[1])^2 # run the primal function
    
    du[1] = 0.0   # zero out the derivative of du, we're going to accumulate into this buffer

    function f!_pullback(::NoRData)          # f! returns `nothing`, so there is *no* wobble in the output
        dv[1] += 4 * v[1] * du[1]            # Derivative of the primal, mutating dv

        u[1]  = old_u1  # undo mutation of u
        du[1] = old_du1 # undo the mutation of du 

        return NoRData(), NoRData(), NoRData()   # f! is a constant, and u and v are tracked  purely through `FData`
    end
    return CoDual(nothing, NoFData()), f_pullback
end
```
