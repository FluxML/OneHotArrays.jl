"""
    OneHotArray{T,L,N,M,I} <: AbstractArray{Bool,M}

These are constructed by [`onehot`](@ref) and [`onehotbatch`](@ref).
Parameter `I` is the type of the underlying storage, and `T` its eltype.
"""
struct OneHotArray{T<:Integer, L, N, var"N+1", I<:Union{T, AbstractArray{T, N}}} <: AbstractArray{Bool, var"N+1"}
  indices::I
end
OneHotArray{T, L, N, I}(indices) where {T, L, N, I} = OneHotArray{T, L, N, N+1, I}(indices)
OneHotArray(indices::T, L::Integer) where {T<:Integer} = OneHotArray{T, L, 0, 1, T}(indices)
OneHotArray(indices::I, L::Integer) where {T, N, I<:AbstractArray{T, N}} = OneHotArray{T, L, N, N+1, I}(indices)

_indices(x::OneHotArray) = x.indices
_indices(x::Base.ReshapedArray{<: Any, <: Any, <: OneHotArray}) =
  reshape(parent(x).indices, x.dims[2:end])

const OneHotVector{T, L} = OneHotArray{T, L, 0, 1, T}
const OneHotMatrix{T, L, I} = OneHotArray{T, L, 1, 2, I}

@doc @doc(OneHotArray)
OneHotVector(idx, L) = OneHotArray(idx, L)
@doc @doc(OneHotArray)
OneHotMatrix(indices, L) = OneHotArray(indices, L)

# use this type so reshaped arrays hit fast paths
# e.g. argmax
const OneHotLike{T, L, N, var"N+1", I} =
  Union{OneHotArray{T, L, N, var"N+1", I},
        Base.ReshapedArray{Bool, var"N+1", <:OneHotArray{T, L, <:Any, <:Any, I}}}

_isonehot(x::OneHotArray) = true
_isonehot(x::Base.ReshapedArray{<:Any, <:Any, <:OneHotArray{<:Any, L}}) where L = (size(x, 1) == L)

Base.size(x::OneHotArray{<:Any, L}) where L = (Int(L), size(x.indices)...)

function Base.getindex(x::OneHotArray{<:Any, <:Any, N}, i::Integer, I::Vararg{Any, N}) where N
  @boundscheck checkbounds(x, i, I...)
  return x.indices[I...] .== i
end

function Base.getindex(x::OneHotArray{<:Any, L}, ::Colon, I...) where L
  @boundscheck checkbounds(x, :, I...)
  return OneHotArray(x.indices[I...], L)
end

Base.getindex(x::OneHotArray, ::Colon) = BitVector(reshape(x, :))
Base.getindex(x::OneHotArray{<:Any, <:Any, N}, ::Colon, ::Vararg{Colon, N}) where N = x

function Base.showarg(io::IO, x::OneHotArray, toplevel)
  print(io, ndims(x) == 1 ? "OneHotVector(" : ndims(x) == 2 ? "OneHotMatrix(" : "OneHotArray(")
  Base.showarg(io, x.indices, false)
  print(io, ')')
  toplevel && print(io, " with eltype Bool")
  return nothing
end

# this is from /LinearAlgebra/src/diagonal.jl, official way to print the dots:
function Base.replace_in_print_matrix(x::OneHotLike, i::Integer, j::Integer, s::AbstractString)
  x[i,j] ? s : _isonehot(x) ? Base.replace_with_centered_mark(s) : s
end

# copy CuArray versions back before trying to print them:
Base.print_array(io::IO, X::OneHotLike{T, L, N, var"N+1", <:AbstractGPUArray}) where {T, L, N, var"N+1"} = 
  Base.print_array(io, adapt(Array, X))
Base.print_array(io::IO, X::LinearAlgebra.AdjOrTrans{Bool, <:OneHotLike{T, L, N, var"N+1", <:AbstractGPUArray}}) where {T, L, N, var"N+1"} = 
  Base.print_array(io, adapt(Array, X))

_onehot_bool_type(::OneHotLike{<:Any, <:Any, <:Any, N, <:Union{Integer, AbstractArray}}) where N = Array{Bool, N}
_onehot_bool_type(::OneHotLike{<:Any, <:Any, <:Any, N, <:AbstractGPUArray}) where N = AbstractGPUArray{Bool, N}

function Base.cat(x::OneHotLike{<:Any, L}, xs::OneHotLike{<:Any, L}...; dims::Int) where L
  if isone(dims) || any(x -> !_isonehot(x), (x, xs...))
    return cat(map(x -> convert(_onehot_bool_type(x), x), (x, xs...))...; dims = dims)
  else
    return OneHotArray(cat(_indices(x), _indices.(xs)...; dims = dims - 1), L)
  end
end

Base.hcat(x::OneHotLike, xs::OneHotLike...) = cat(x, xs...; dims = 2)
Base.vcat(x::OneHotLike, xs::OneHotLike...) = cat(x, xs...; dims = 1)

# optimized concatenation for matrices and vectors of same parameters
Base.hcat(x::T, xs::T...) where {L, T <: OneHotLike{<:Any, L, <:Any, 2}} =
  OneHotMatrix(reduce(vcat, _indices.(xs); init = _indices(x)), L)
Base.hcat(x::T, xs::T...) where {L, T <: OneHotLike{<:Any, L, <:Any, 1}} =
  OneHotMatrix(reduce(vcat, _indices.(xs); init = _indices(x)), L)

MLUtils.batch(xs::AbstractArray{<:OneHotVector{<:Any, L}}) where L = OneHotMatrix(_indices.(xs), L)

Adapt.adapt_structure(T, x::OneHotArray{<:Any, L}) where L = OneHotArray(adapt(T, _indices(x)), L)

function Base.BroadcastStyle(::Type{<:OneHotArray{<: Any, <: Any, <: Any, var"N+1", T}}) where {var"N+1", T <: AbstractGPUArray}
  # We want CuArrayStyle{N+1}(). There's an AbstractGPUArrayStyle but it doesn't do what we need. 
  S = Base.BroadcastStyle(T)
  # S has dim N not N+1. The following hack to fix it relies on the arraystyle having N as its first type parameter, which
  # isn't guaranteed, but there are not so many GPU broadcasting styles in the wild. (Far fewer than there are array wrappers.)
  (typeof(S).name.wrapper){var"N+1"}()
end

Base.map(f, x::OneHotLike) = Base.broadcast(f, x)

Base.argmax(x::OneHotLike; dims = Colon()) =
  (_isonehot(x) && dims == 1) ?
    reshape(CartesianIndex.(_indices(x), CartesianIndices(_indices(x))), 1, size(_indices(x))...) :
    invoke(argmax, Tuple{AbstractArray}, x; dims = dims)
