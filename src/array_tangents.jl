# Array tangent *value* operations, ported from Mooncake's `src/rules/array_legacy.jl`
# (lines 1-80 — the element-wise, rule-system-free part). These build a plain
# `Array{tangent_type(P),N}` tangent element-wise and are aliasing/circular-reference aware via
# the cache, matching the generic scalar/struct implementations in `tangents.jl`.
#
# NOTE (Differ port): Mooncake's Julia-1.13 array path (`src/rules/memory.jl`) instead recurses
# through `Memory`/`MemoryRef` tangent internals and is fused with the reverse-mode rule system
# (`frule!!`/`rrule!!`/`@is_primitive`), which is out of scope here. The element-wise version below
# is semantically identical for the tangent/fdata/rdata system: the tangent of an `Array{P,N}` is
# an `Array{tangent_type(P),N}`, its fdata is itself, and its rdata is `NoRData` (handled generically
# in `fwds_rvs_data.jl`). Memory/MemoryRef primals themselves are not covered.

@inline function zero_tangent_internal(x::Array{P,N}, dict::MaybeCache) where {P,N}
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))

    zt = Array{tangent_type(P),N}(undef, size(x)...)
    dict[x] = zt
    return _map_if_assigned!(
        Base.Fix2(zero_tangent_internal, dict), zt, x
    )::Array{tangent_type(P),N}
end

function randn_tangent_internal(
    rng::AbstractRNG, x::Array{T,N}, dict::MaybeCache
) where {T,N}
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))

    dx = Array{tangent_type(T),N}(undef, size(x)...)
    dict[x] = dx
    return _map_if_assigned!(x -> randn_tangent_internal(rng, x, dict), dx, x)
end

function increment_internal!!(c::IncCache, x::T, y::T) where {P,N,T<:Array{P,N}}
    (haskey(c, x) || x === y) && return x
    c[x] = true
    return _map_if_assigned!((x, y) -> increment_internal!!(c, x, y), x, x, y)
end

function set_to_zero_internal!!(c::SetToZeroCache, x::Array)
    _already_tracked!(c, x) && return x
    return _map_if_assigned!(Base.Fix1(set_to_zero_internal!!, c), x, x)
end

function _scale_internal(c::MaybeCache, a::Float64, t::Array{T,N}) where {T,N}
    haskey(c, t) && return c[t]::Array{T,N}
    t′ = Array{T,N}(undef, size(t)...)
    c[t] = t′
    return _map_if_assigned!(t -> _scale_internal(c, a, t), t′, t)
end

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:Array}
    key = (t, s)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    bitstype = Val(isbitstype(eltype(T)))
    return sum(eachindex(t, s); init=0.0) do i
        if bitstype isa Val{true} || (isassigned(t, i) && isassigned(s, i))
            _dot_internal(c, t[i], s[i])::Float64
        else
            0.0
        end
    end
end

function _add_to_primal_internal(
    c::MaybeCache, x::Array{P,N}, t::Array{<:Any,N}, unsafe::Bool
) where {P,N}
    key = (x, t, unsafe)
    haskey(c, key) && return c[key]::Array{P,N}
    x′ = Array{P,N}(undef, size(x)...)
    c[key] = x′
    return _map_if_assigned!((x, t) -> _add_to_primal_internal(c, x, t, unsafe), x′, x, t)
end

function tangent_to_primal_internal!!(
    x::Array{P,N}, t::Array{<:Any,N}, c::MaybeCache
) where {P,N}
    haskey(c, x) && return c[x]::Array{P,N}
    c[x] = x
    return _map_if_assigned!(x, x, t) do xn, tn
        return tangent_to_primal_internal!!(xn, tn, c)
    end
end
function primal_to_tangent_internal!!(
    t::Array{<:Any,N}, x::Array{P,N}, c::MaybeCache
) where {P,N}
    haskey(c, x) && return c[x]::Array{tangent_type(P),N}
    c[x] = t
    return _map_if_assigned!(t, t, x) do txn, xn
        return primal_to_tangent_internal!!(txn, xn, c)
    end
end
