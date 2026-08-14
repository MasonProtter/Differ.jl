# Differ.jl

Differ is a (highly experimental) attempt at a Julia-level automatic differentiation engine over optimized, typed IR. It is partially descended from Mooncake, in particular it inherits Mooncake's `FData`/`RData`/`Tangent` system with minor tweaks, and while the actual compiler passes are not based on Mooncake's exact designs, a lot of inspiration and lessons were taken from how Mooncake's compiler passes were written. In future work, I hope to encorporate more of Enzyme's compiler-level analysis and optimization passes in order to better approach its performance.

That largest `Differ`ence between Mooncake and Differ, is that Differ's code-generation phase happens at compile-time inside a `@generated function`, and hooks into the relatively [new mechanism](https://github.com/JuliaLang/julia/pull/56660) for `invoke`ing `CodeInstance`s, as such it currently only supports julia version 1.13 currently. This builds on and extends my previous stalled attempts to move Mooncake's rule-generation phase to compile time, https://github.com/chalk-lab/Mooncake.jl/pull/593 and https://github.com/chalk-lab/Mooncake.jl/pull/900 .

Differ is generally slower than Enzyme, and faster than Mooncake, though there are exceptions. It is far less mature and less tested than either. 

This is a project mostly made for fun, and is not something serious to be relied upon.

## Assorted things to know

- A minority of this code is hand written by me, I mostly worked on the main compiler mechanism and entrypoints, but large sections of the overall codebase are on various points along the spectrum from "vibe-coded" to "LLM-assisted". Skepticism is very warranted.
- Differ has forwards and reverse mode.
  - Both modes support mutation (though lack supprot for things like `push!`/`pop!`)
  - Forwards mode is simpler and more robust
  - Reverse mode works on a lot of common programs, but is more liable to choke. It will try to tell you what part of your program it doesn't understand.
- Nested forwards-mode differentiation usually works. Forwards-over-Reverse differentiation works in limited case. 

## Using via DifferentiationInterface.jl 

The most ergonomic way to Differ is with DifferentiationInterface.jl, I have not yet upstreamed dispatch structs to ADTypes.jl, but Differ exports it's own structs `AutoDifferForwards()` and `AutoDifferReverse()`.


``` julia
julia> using DifferentiationInterface, Differ

julia> derivative(AutoDifferForwards(), 1.0) do x
           sin(x)^2 + 1
       end
0.9092974268256818
```

Because Differ.jl does compile-time rule generation, these derivatives can be fast even without `prepare_gradient` machinery:

``` julia
julia> @btime derivative(AutoDifferForwards(), x) do x
           sin(x)^2 + 1
       end setup=(x=rand())
  19.868 ns (0 allocations: 0 bytes)
0.9927783746649702
```

though it's still mildly helpful for reverse mode where we allocate a `Tape` value:

``` julia
julia> f(v) = sum(sin, v);

julia> v = rand(100);

julia> @btime gradient(f, AutoDifferReverse(), $v);
  3.244 μs (54 allocations: 9.59 KiB)
```

versus

```julia
julia> prep = prepare_gradient(f, AutoDifferReverse(), v);

julia> @btime gradient(f, $prep, AutoDifferReverse(), $v);
  2.356 μs (2 allocations: 928 bytes)
```

## More native 'API'

Differ.jl's internal entrypoints are invoked by calling `frule!!` with `Dual` arguments, or `rrule!!` with `CoDual` arguments.

This program computes `1.0 * d/dx sin(x)` at `x=5.0` using forwards mode AD:

``` julia
julia> f(x) = sin(x)^2 + 1;

julia> fdf = Dual(f, zero_tangent(f))
Dual{typeof(f), NoTangent}(f, NoTangent())

julia> xdx = Dual(5.0, 1.0)
Dual{Float64, Float64}(5.0, 1.0)

julia> ydy = frule!!(fdf, xdx)
Dual{Float64, Float64}(1.9195357645382263, -0.5440211108893698)
```

This computes the `1.0 *  ∇g(v)` at `v = [1.0, 2.0, 3.0]`:

``` julia
julia> g(v) = sum(sin, v);

julia> gdg = CoDual(g, NoFData()); # g and any mutable data needed to calculate its derivative

julia> v = [1.0, 2.0, 3.0] # where we differentiate
       dv = [0.0, 0.0, 0.0]# accumulator for gradient
       vdv = CoDual(v, dv) # v together with gradient accumulator 
CoDual{Vector{Float64}, Vector{Float64}}([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])

julia> res, pullback = rrule!!(gdg, ctx, vdv);

julia> res # this is just g(v)
CoDual{Float64, NoFData}(1.8918884196934453, NoFData())

julia> pb(1.0) # compute the pullback, modifying vdv
(NoRData(), NoRData())

julia> dv # dv was mutated in place by `pullback`
3-element Vector{Float64}:
  0.5403023058681398
 -0.4161468365471424
 -0.9899924966004454
```
