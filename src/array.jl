"""
    OneHotArray{T, N, M, I} <: AbstractArray{Bool, M}
    OneHotArray(indices, L)

A one-hot `M`-dimensional array with `L` labels (i.e. `size(A, 1) == L` and `sum(A, dims=1) == 1`)
stored as a compact `N == M-1`-dimensional array of indices.

Typically constructed by [`onehot`](@ref) and [`onehotbatch`](@ref).
Parameter `I` is the type of the underlying storage, and `T` its eltype.
"""
struct OneHotArray{T<:Integer, N, var"N+1", I<:Union{T, AbstractArray{T, N}}} <: AbstractArray{Bool, var"N+1"}
  indices::I
  nlabels::Int
end
OneHotArray{T, N, I}(indices, L::Int) where {T, N, I} = OneHotArray{T, N, N+1, I}(indices, L)
OneHotArray(indices::T, L::Int) where {T<:Integer} = OneHotArray{T, 0, 1, T}(indices, L)
OneHotArray(indices::I, L::Int) where {T, N, I<:AbstractArray{T, N}} = OneHotArray{T, N, N+1, I}(indices, L)

_indices(x::OneHotArray) = x.indices
_indices(x::Base.ReshapedArray{<:Any, <:Any, <:OneHotArray}) =
  reshape(parent(x).indices, x.dims[2:end])

"""
    OneHotVector{T} = OneHotArray{T, 0, 1, T}
    OneHotVector(indices, L)

A one-hot vector with `L` labels (i.e. `length(A) == L` and `count(A) == 1`) typically constructed by [`onehot`](@ref).
Stored efficiently as a single index of type `T`, usually `UInt32`.
"""
const OneHotVector{T} = OneHotArray{T, 0, 1, T}
OneHotVector(idx, L) = OneHotArray(idx, L)

"""
    OneHotMatrix{T, I} = OneHotArray{T, 1, 2, I}
    OneHotMatrix(indices, L)

A one-hot matrix (with `L` labels) typically constructed using [`onehotbatch`](@ref).
Stored efficiently as a vector of indices with type `I` and eltype `T`.
"""
const OneHotMatrix{T, I} = OneHotArray{T, 1, 2, I}
OneHotMatrix(indices, L) = OneHotArray(indices, L)

# use this type so reshaped arrays hit fast paths
# e.g. argmax
const OneHotLike{T, N, var"N+1", I} =
  Union{OneHotArray{T, N, var"N+1", I},
        Base.ReshapedArray{Bool, var"N+1", <:OneHotArray{T, <:Any, <:Any, I}}}

_isonehot(x::OneHotArray) = true
_isonehot(x::Base.ReshapedArray{<:Any, <:Any, <:OneHotArray}) = (size(x, 1) == parent(x).nlabels)

_check_nlabels(L, xs::OneHotLike...) = all(size.(xs, 1) .== L)

_nlabels(x::OneHotArray) = size(x, 1)
function _nlabels(x::OneHotLike, xs::OneHotLike...)
  L = size(x, 1)
  _check_nlabels(L, xs...) ||
    throw(DimensionMismatch("The number of labels are not the same for all one-hot arrays."))

  return L
end

Base.size(x::OneHotArray) = (x.nlabels, size(x.indices)...)

function Base.getindex(x::OneHotArray{<:Any, N}, i::Int, I::Vararg{Int, N}) where N
  @boundscheck (1 <= i <= x.nlabels) || throw(BoundsError(x, (i, I...)))
  return x.indices[I...] .== i
end
# the method above is faster on the CPU but will scalar index on the GPU
# so we define the method below to pass the extra indices directly to GPU array
function Base.getindex(x::OneHotArray{<:Any, N, <:Any, <:AbstractGPUArray},
                       i::Int, 
                       I::Vararg{Any, N}) where N
  @boundscheck (1 <= i <= x.nlabels) || throw(BoundsError(x, (i, I...)))
  return x.indices[I...] .== i
end
function Base.getindex(x::OneHotArray{<:Any, N}, ::Colon, I::Vararg{Any, N}) where N
  return OneHotArray(x.indices[I...], x.nlabels)
end
Base.getindex(x::OneHotArray, ::Colon) = BitVector(reshape(x, :))
Base.getindex(x::OneHotArray{<:Any, N}, ::Colon, ::Vararg{Colon, N}) where N = x

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
for fun in (:show, :print_array)  # print_array is used by 3-arg show
  @eval begin
    Base.$fun(io::IO, X::OneHotLike{T, N, var"N+1", <:AbstractGPUArray}) where {T, N, var"N+1"} = 
      Base.$fun(io, adapt(Array, X))
    Base.$fun(io::IO, X::LinearAlgebra.AdjOrTrans{Bool, <:OneHotLike{T, N, <:Any, <:AbstractGPUArray}}) where {T, N} = 
      Base.$fun(io, adapt(Array, X))
  end
end

_onehot_bool_type(::OneHotLike{<:Any, <:Any, var"N+1", <:Union{Integer, AbstractArray}}) where {var"N+1"} = Array{Bool, var"N+1"}
_onehot_bool_type(::OneHotLike{<:Any, <:Any, var"N+1", <:AbstractGPUArray}) where {var"N+1"} = AbstractGPUArray{Bool, var"N+1"}

_notall_onehot(x::OneHotArray, xs::OneHotArray...) = false
_notall_onehot(x::OneHotLike, xs::OneHotLike...) = any(x -> !_isonehot(x), (x, xs...))

function Base.cat(x::OneHotLike{<:Any, <:Any, N}, xs::OneHotLike...; dims::Int) where N
  if isone(dims) || _notall_onehot(x, xs...)
    return cat(map(x -> convert(_onehot_bool_type(x), x), (x, xs...))...; dims = dims)
  else
    L = _nlabels(x, xs...)

    return OneHotArray(cat(_indices(x), _indices.(xs)...; dims = dims - 1), L)
  end
end

Base.hcat(x::OneHotLike, xs::OneHotLike...) = cat(x, xs...; dims = 2)
Base.vcat(x::OneHotLike, xs::OneHotLike...) =
  vcat(map(x -> convert(_onehot_bool_type(x), x), (x, xs...))...)

# optimized concatenation for matrices and vectors of same parameters
Base.hcat(x::OneHotMatrix, xs::OneHotMatrix...) =
  OneHotMatrix(reduce(vcat, _indices.(xs); init = _indices(x)), _nlabels(x, xs...))
Base.hcat(x::OneHotVector, xs::OneHotVector...) =
  OneHotMatrix(reduce(vcat, _indices.(xs); init = _indices(x)), _nlabels(x, xs...))

if isdefined(Base, :stack)
  import Base: _stack
else
  import Compat: _stack
end
function _stack(::Colon, xs::AbstractArray{<:OneHotArray})
  n = _nlabels(first(xs))
  all(x -> _nlabels(x)==n, xs) || throw(DimensionMismatch("The number of labels are not the same for all one-hot arrays."))
  OneHotArray(Compat.stack(_indices, xs), n)
end

Adapt.adapt_structure(T, x::OneHotArray) = OneHotArray(adapt(T, _indices(x)), x.nlabels)

function Base.BroadcastStyle(::Type{<:OneHotArray{<:Any, <:Any, var"N+1", T}}) where {var"N+1", T <: AbstractGPUArray}
  # We want CuArrayStyle{N+1}(). There's an AbstractGPUArrayStyle but it doesn't do what we need.
  S = Base.BroadcastStyle(T)
  typeof(S)(Val{var"N+1"}())
end

Base.map(f, x::OneHotLike) = Base.broadcast(f, x)

Base.argmax(x::OneHotLike; dims = Colon()) =
  (_isonehot(x) && dims == 1) ?
    reshape(CartesianIndex.(_indices(x), CartesianIndices(_indices(x))), 1, size(_indices(x))...) :
    invoke(argmax, Tuple{AbstractArray}, x; dims = dims)
