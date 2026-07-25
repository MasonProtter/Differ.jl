# Utility helpers ported from Mooncake's `src/utils.jl`, trimmed to what the tangent /
# fdata / rdata system needs. See `/project/Mooncake.jl/src/utils.jl` for the originals.

const CC = Core.Compiler

const IEEEFloat = Base.IEEEFloat
const bitcast = Core.Intrinsics.bitcast
using Core: svec, SimpleVector

# Mooncake gets `@unstable`/`@stable` from DispatchDoctor; they are type-stability
# *assertions* used when the whole module is wrapped in `@stable`. Differ is not, so these
# are safe no-op pass-throughs — they keep the ported source verbatim without a new dep.
macro unstable(expr)
    return esc(expr)
end
macro stable(expr)
    return esc(expr)
end

"""
    @foldable def

Shorthand for `Base.@assume_effects :foldable function f(x)...`.
"""
macro foldable(expr)
    return esc(:(Base.@assume_effects :foldable $expr))
end

"""
    _typeof(x)

Central definition of typeof, specific to the use required in this package.
"""
@unstable _typeof(x) = Base._stable_typeof(x)
@unstable _typeof(x::Tuple) = Tuple{tuple_map(_typeof, x)...}
@unstable _typeof(x::NamedTuple{names}) where {names} = NamedTuple{names,_typeof(Tuple(x))}

# ---------------------------------------------------------------------------
# tuple_map / tuple_fill / _findall / stable_all
# ---------------------------------------------------------------------------

"""
    tuple_map(f::F, x::Tuple) where {F}

Like `map(f, x)` but always specialises on all element types of `x`, regardless of length.
"""
@inline @generated function tuple_map(f::F, x::Tuple) where {F}
    return Expr(:call, :tuple, map(n -> :(f(getfield(x, $n))), 1:fieldcount(x))...)
end

@inline @generated function tuple_map(f::F, x::Tuple, y::Tuple) where {F}
    if length(x.parameters) != length(y.parameters)
        return :(throw(ArgumentError("length(x) != length(y)")))
    else
        stmts = map(n -> :(f(getfield(x, $n), getfield(y, $n))), 1:fieldcount(x))
        return Expr(:call, :tuple, stmts...)
    end
end

@inline @generated function tuple_map(f::F, x::Tuple, y::Tuple, z::Tuple) where {F}
    if length(x.parameters) != length(y.parameters) ||
        length(x.parameters) != length(z.parameters)
        return :(throw(ArgumentError("x, y, and z must have the same length")))
    else
        stmts = map(
            n -> :(f(getfield(x, $n), getfield(y, $n), getfield(z, $n))), 1:fieldcount(x)
        )
        return Expr(:call, :tuple, stmts...)
    end
end

@generated function tuple_map(f, x::NamedTuple{names}) where {names}
    getfield_exprs = map(n -> :(f(getfield(x, $n))), 1:fieldcount(x))
    return :(NamedTuple{names}($(Expr(:call, :tuple, getfield_exprs...))))
end

@generated function tuple_map(f, x::NamedTuple{names}, y::NamedTuple{names}) where {names}
    if fieldcount(x) != fieldcount(y)
        return :(throw(ArgumentError("length(x) != length(y)")))
    end
    getfield_exprs = map(n -> :(f(getfield(x, $n), getfield(y, $n))), 1:fieldcount(x))
    return :(NamedTuple{names}($(Expr(:call, :tuple, getfield_exprs...))))
end

@inline @generated function tuple_fill(val, ::Val{N}) where {N}
    return Expr(:call, :tuple, map(_ -> :val, 1:N)...)
end

"""
    _findall(cond, x::Tuple)

Type-stable version of `findall` for `Tuple`s.
"""
@inline @generated function _findall(cond, x::Tuple)
    y = :(y = ())
    exprs = map(n -> :(y = cond(x[$n]) ? ($n, y...) : y), 1:fieldcount(x))
    return Expr(:block, y, exprs...)
end

"""
    stable_all(x::NTuple{N, Bool}) where {N}

`all` variant that constant-folds when the values of `x` are known statically.
"""
@generated function stable_all(x::NTuple{N,Bool}) where {N}
    exprs = map(n -> :(x[$n] || return false), 1:N)
    return Expr(:block, exprs..., :(return true))
end

"""
    _map(f, x...)

Same as `map`, but requires all elements of `x` to have equal length.
"""
@unstable @inline function _map(f::F, x::Vararg{Any,N}) where {F,N}
    @assert allequal(map(length, x))
    return map(f, x...)
end

"""
    _map_if_assigned!(f, y, x)

For all `n`, if `x[n]` is assigned, write `f(x[n])` to `y[n]`, else leave `y[n]` unchanged.
"""
function _map_if_assigned!(f::F, y::DenseArray, x::DenseArray{P}) where {F,P}
    @assert size(y) == size(x)
    @inbounds for n in eachindex(y)
        if isbitstype(P) || isassigned(x, n)
            y[n] = f(x[n])
        end
    end
    return y
end

function _map_if_assigned!(
    f::F, y::DenseArray, x1::DenseArray{P}, x2::DenseArray
) where {F,P}
    @assert size(y) == size(x1)
    @assert size(y) == size(x2)
    @inbounds for n in eachindex(y)
        if isbitstype(P) || isassigned(x1, n)
            y[n] = f(x1[n], x2[n])
        end
    end
    return y
end

# ---------------------------------------------------------------------------
# Field-initialisation queries
# ---------------------------------------------------------------------------

"""
    always_initialised(::Type{P}) where {P}

Tuple of `Bool`s, one per field of `P`: `true` if the nth field is always initialised.
"""
@generated function always_initialised(::Type{P}) where {P}
    P isa DataType || return :(error("$P is not a DataType."))
    num_init = CC.datatype_min_ninitialized(P)
    return (map(n -> n <= num_init, 1:fieldcount(P))...,)
end

"""
    is_always_initialised(P::DataType, n::Int)::Bool

True if the `n`th field of `P` is always initialised.
"""
function is_always_initialised(P::DataType, n::Int)::Bool
    return n <= CC.datatype_min_ninitialized(P)
end

"""
    is_always_fully_initialised(P::DataType)::Bool

True if all fields in `P` are always initialised.
"""
function is_always_fully_initialised(P::DataType)::Bool
    return CC.datatype_min_ninitialized(P) == fieldcount(P)
end

"""
    _new_(::Type{T}, x::Vararg{Any, N}) where {T, N}

Calls the `:new` instruction with type `T` and arguments `x`.
"""
@inline @generated function _new_(::Type{T}, x::Vararg{Any,N}) where {T,N}
    return Expr(:new, :T, map(n -> :(x[$n]), 1:N)...)
end

# ---------------------------------------------------------------------------
# Boxed error printing (used by Invalid{F,R}DataException show methods)
# ---------------------------------------------------------------------------

function _print_boxed_block(io::IO, first_prefix::AbstractString, lines; footer=nothing)
    first_item = iterate(lines)
    isnothing(first_item) && return nothing
    line, state = first_item
    rest_prefix = "│ "
    first_width = _boxed_message_width(io, first_prefix)
    rest_width = _boxed_message_width(io, rest_prefix)
    first_wrapped = _wrap_boxed_line(line, first_width)
    println(io, first_prefix, first(first_wrapped))
    for wrapped_line in Base.tail(first_wrapped)
        println(io, rest_prefix, wrapped_line)
    end
    while true
        item = iterate(lines, state)
        isnothing(item) && break
        line, state = item
        for wrapped_line in _wrap_boxed_line(line, rest_width)
            println(io, rest_prefix, wrapped_line)
        end
    end
    return isnothing(footer) ? println(io, "└") : println(io, "└ ", footer)
end

@inline function _boxed_message_width(io::IO, prefix::AbstractString)
    cols = get(io, :displaysize, displaysize(io))[2]
    return max(20, cols - textwidth(prefix))
end

function _wrap_boxed_line(line, width::Int)
    text = string(line)
    isempty(text) && return (text,)
    width < 1 && return (text,)
    textwidth(text) <= width && return (text,)

    wrapped = String[]
    remaining = text
    while textwidth(remaining) > width
        split_idx = nothing
        for idx in eachindex(remaining)
            textwidth(SubString(remaining, 1, idx)) > width && break
            remaining[idx] == ' ' && (split_idx = idx)
        end
        if isnothing(split_idx)
            split_idx = firstindex(remaining)
            for idx in eachindex(remaining)
                textwidth(SubString(remaining, firstindex(remaining), idx)) > width && break
                split_idx = idx
            end
        end
        push!(wrapped, rstrip(SubString(remaining, firstindex(remaining), split_idx)))
        remaining = lstrip(SubString(remaining, nextind(remaining, split_idx)))
        isempty(remaining) && break
    end
    isempty(remaining) || push!(wrapped, remaining)
    return Tuple(wrapped)
end

function _print_boxed_error(io::IO, lines; footer=nothing)
    _print_boxed_block(io, "", lines; footer)
end

# ---------------------------------------------------------------------------
# _copy: structural copy used by tangent / fdata / rdata types. Base methods here; the
# type-specific overloads live alongside their types (tangents.jl, fwds_rvs_data.jl, etc).
# ---------------------------------------------------------------------------

_copy(x) = copy(x)
_copy(::Nothing) = nothing
_copy(x::Symbol) = x
_copy(x::Tuple) = map(_copy, x)
_copy(x::NamedTuple) = map(_copy, x)
_copy(x::Ref{T}) where {T} = isassigned(x) ? Ref{T}(_copy(x[])) : Ref{T}()
_copy(x::Type) = x
