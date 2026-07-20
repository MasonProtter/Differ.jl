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
infini_type(::Type{Dual{T, U}}) where {T, U} = U

struct_zero(x::Number) = zero(x)::typeof(x)
struct_zero(x::Union{Memory, Array}) = map(struct_zero, x)

@generated function struct_zero(x::T) where {T}
    if fieldcount(T) == 0
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

function frule(::Dual{typeof(+)}, xdx::Dual)
    xdx
end
function frule(::Dual{typeof(+)}, (; x, dx)::Dual, (; y, dy)::Dual)
    Dual(x + y, dx + dy)
end


function frule(::Dual{typeof(-)}, (; x, dx)::Dual)
    Dual(-x, -dx)
end
function frule(::Dual{typeof(-)}, (; x, dx)::Dual, (; y, dy)::Dual)
    Dual(x - y, dx - dy)
end

function frule(::Dual{typeof(*)}, (; x, dx)::Dual, (; y, dy)::Dual)
    Dual(x * y, x*dy + dx * y)
end

