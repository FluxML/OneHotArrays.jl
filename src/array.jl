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

function Base.similar(::OneHotArray{T, L}, ::Type{Bool}, dims::Dims) where {T, L}
  if first(dims) == L
    indices = ones(T, Base.tail(dims))
    return OneHotArray(indices, first(dims))
  else
    return BitArray(undef, dims)
  end
end

function Base.setindex!(x::OneHotLike{<:Any, <:Any, N}, v::Bool, i::Integer, I::Vararg{Integer, N}) where N
  @boundscheck checkbounds(x, i, I...)
  if v
    _indices(x)[I...] = i
  else
    error("OneHotArray cannot be set with false values")
  end
end

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
Base.print_array(io::IO, X::OneHotLike{T, L, N, var"N+1", <:CuArray}) where {T, L, N, var"N+1"} = 
  Base.print_array(io, adapt(Array, X))
Base.print_array(io::IO, X::LinearAlgebra.AdjOrTrans{Bool, <:OneHotLike{T, L, N, var"N+1", <:CuArray}}) where {T, L, N, var"N+1"} = 
  Base.print_array(io, adapt(Array, X))

_onehot_bool_type(::OneHotLike{<:Any, <:Any, <:Any, N, <:Union{Integer, AbstractArray}}) where N = Array{Bool, N}
_onehot_bool_type(::OneHotLike{<:Any, <:Any, <:Any, N, <:CuArray}) where N = CuArray{Bool, N}

_onehot_compatible(x::OneHotLike) = _isonehot(x)
_onehot_compatible(x::AbstractVector{Bool}) = count(x) == 1
_onehot_compatible(x::AbstractArray{Bool}) = all(isone, reduce(+, x; dims=1))
_onehot_compatible(x::AbstractArray) = _onehot_compatible(BitArray(x))

function OneHotArray(x::OneHotLike)
  !_onehot_compatible(x) && error("Array is not onehot compatible")
  return x
end

function OneHotArray(x::AbstractVector)
  !_onehot_compatible(x) && error("Array is not onehot compatible")
  return OneHotArray(findfirst(x), length(x))
end

function OneHotArray(x::AbstractArray)
  !_onehot_compatible(x) && error("Array is not onehot compatible")
  dims = size(x)
  dim1, dim2 = dims[1], reduce(*, Base.tail(dims))
  rx = reshape(x, (dim1, dim2))
  indices = UInt32[findfirst(==(true), col) for col in eachcol(rx)]
  return OneHotArray(reshape(indices, Base.tail(dims)), dim1)
end

function Base.cat(x::OneHotLike{<:Any, L}, xs::OneHotLike{<:Any, L}...; dims::Int) where L
  if isone(dims) || any(x -> !_isonehot(x), (x, xs...))
    return cat(map(x -> convert(_onehot_bool_type(x), x), (x, xs...))...; dims = dims)
  else
    return OneHotArray(cat(_indices(x), _indices.(xs)...; dims = dims - 1), L)
  end
end

Base.hcat(x::OneHotLike, xs::OneHotLike...) = cat(x, xs...; dims = 2)
Base.vcat(x::OneHotLike, xs::OneHotLike...) = cat(x, xs...; dims = 1)

# optimized concatenation for arrays of same parameters
Base.hcat(x::T, xs::T...) where {L, T <: OneHotLike{<:Any, L, <:Any}} =
  OneHotArray(reduce(vcat, _indices.(xs); init = _indices(x)), L)

MLUtils.batch(xs::AbstractArray{<:OneHotVector{<:Any, L}}) where L = OneHotMatrix(_indices.(xs), L)

Adapt.adapt_structure(T, x::OneHotArray{<:Any, L}) where L = OneHotArray(adapt(T, _indices(x)), L)

Base.BroadcastStyle(::Type{<:OneHotArray{<: Any, <: Any, <: Any, N, <: CuArray}}) where N = CUDA.CuArrayStyle{N}()

Base.map(f, x::OneHotLike) = Base.broadcast(f, x)

function Base.argmax(x::OneHotLike; dims = Colon())
  if _isonehot(x) && dims == 1
    cart_inds = CartesianIndex.(_indices(x), CartesianIndices(_indices(x)))
    return reshape(cart_inds, (1, size(_indices(x))...))
  else
    return argmax(BitArray(x); dims=dims)
  end
end

function Base.argmin(x::OneHotLike; dims = Colon())
  if _isonehot(x) && dims == 1
    labelargs = ifelse.(_indices(x) .== 1, 2, 1)
    cart_inds = CartesianIndex.(labelargs, CartesianIndices(_indices(x)))
    return reshape(cart_inds, (1, size(_indices(x))...))
  else
    return argmin(BitArray(x); dims=dims)
  end
end
