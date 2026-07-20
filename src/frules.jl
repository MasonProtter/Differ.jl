struct NoFData end
struct Dual{T, U <: Union{T, NoFData}}
    x::T
    dx::U
end

function Base.getproperty(d::Dual, s::Symbol)
    if s ∈ (:x, :y, :z)
        getfield(d, :x)
    elseif s ∈ (:dx, :dy, :dz)
        getfield(d, :dx)
    else
        getfield(d, s)
    end
end
Base.propertynames(d::Dual) = (:x, :y, :z, :dx, :dy, :dz)

primal_type(::Type{Dual{T, U}}) where {T, U} = T
tangent_type(::Type{Dual{T, U}}) where {T, U} = U

struct_zero(x::Number) = zero(x)::typeof(x)
struct_zero(x::Union{Memory, Array}) = map(struct_zero, x)

@generated function struct_zero(x::T) where {T}
    if Base.issingletontype(T)
        :(NoFData())
    else
        Expr(:new, :T, (:(struct_zero(getfield(x, $i))) for i ∈ 1:fieldcount(T))...)
    end
end

function frule(::Dual{typeof(sin)}, (; x, dx)::Dual)
    sinx, cosx = sincos(x)
    Dual(sinx, cosx*dx)
end
function frule(::Dual{typeof(cos)}, (; x, dx)::Dual)
    sinx, cosx = sincos(x)
    Dual(cosx, -sinx*dx)
end

# NOTE: arithmetic (`+`, `-`, `*`, `/`) and comparisons are intentionally NOT given `frule`
# methods here. They inline to intrinsics (`add_float`, `mul_float`, `lt_float`, …), which the
# Tier-2 post-optimization IRCode pass (`ir_dualize.jl`) differentiates directly. A bare
# `frule(Dual(+), …)` therefore routes through the generated fallback into that pass, so `+`/`*`
# work for any type (Complex, Float32, …) without a per-type rule. Keep `frule` methods only for
# functions we'd rather not differentiate through (transcendentals like `sin`/`cos` above).

