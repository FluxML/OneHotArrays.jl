"""
    OneHotVector{T,L}
    OneHotVector(index, L)

A one-hot vector with `L` labels (i.e. `length(A) == L` and `count(A) == 1`).
"""
struct OneHotVector{T,L} # <: AbstractVector{Bool} # this does everything but is too limiting
  index::Integer
  function OneHotVector{T,L}(index) where {T,L}
    @assert 1 <= index <= L "OneHotVector index $(index) out of range [1,$(L)]"
    return new(index)
  end
end

OneHotVector(t::Type, index, nlabels) = OneHotVector{t,nlabels}(index)
OneHotVector(index, nlabels) = OneHotVector{Float32,nlabels}(index)
Base.size(x::OneHotVector{T,L}) where {T,L} = (L,)
function Base.getindex(x::OneHotVector{T,L}, i::Integer) where {T,L}
  @boundscheck 1 <= i <= L
  i == x.index || throw(BoundsError(x, i))
end
Base.show(io::IO, x::OneHotVector{T,L}) where {T,L} = Base.show(io, setindex!(zeros(T, L), convert(T, 1), x.index))
Base.argmax(x::OneHotVector; dims = Colon()) = x.index

struct OneHotArray{T,N,M,L,A} <: AbstractArray{T,N}
  onehotvectors::AbstractArray{OneHotVector{T,L},M}
  function OneHotArray(onehotaxis, onehotvectors::AbstractArray{OneHotVector{T,L},M}) where {T,L,M}
    N = M+1
    @assert onehotaxis isa Integer "onehot axis must be integer"
    @assert 1 <= onehotaxis <= N "onehot axis out of range [1,$N]"
    new{T,N,M,L,onehotaxis}(onehotvectors)
  end
end
onehotaxis(x::OneHotArray{T,N,M,L,A}) where {T,N,M,L,A} = A
function size_selector(i, shape, onehotaxis, L)
  if i < onehotaxis
    shape[i]
  elseif i > onehotaxis
    shape[i-1]
  else
    L
  end
end

Base.size(x::OneHotArray{T,N,M,L,A}) where {T,N,M,L,A} = ntuple(i -> size_selector(i, size(x.onehotvectors), onehotaxis(x), L), N)
function Base.getindex(x::OneHotArray{T,N,M,L}, i::Integer) where {T,N,M,L}
  flat_onehotvectors = x.onehotvectors[:]
  h_ind, L_ind = fldmod(i, L)
  @boundscheck 1 <= h_ind <= length(flat_onehotvectors) || throw(BoundsError(x, i))
  @boundscheck 1 <= L_ind <= L                          || throw(BoundsError(x, i))
  return flat_onehotvectors[h_ind].index == L_ind
end
function Base.getindex(x::OneHotArray{T,N,M,L}, i::Vararg{Integer,N}) where {T,N,M,L}
  @boundscheck all(1 .<= i .<= size(x)) || throw(BoundsError(x, (i...)))
  index_pre = i[1:onehotaxis(x)-1]
  index_post = i[onehotaxis(x)+1:end]
  intern_ind = x.onehotvectors[index_pre..., index_post...].index
  return convert(T, intern_ind == i[onehotaxis(x)])
end

function Base.show(io::IO, x::OneHotArray{T,N,M,L}) where {T,N,M,L}
  z = zeros(Int32, size(x))
  # loop efficiently over only the ones
  for ext_ind in eachindex(IndexCartesian(), x.onehotvectors)
    index_pre = Tuple(ext_ind)[1:onehotaxis(x)]
    index_post = Tuple(ext_ind)[onehotaxis(x)+1:end]
    intern_ind = x.onehotvectors[ext_ind].index
    ind = CartesianIndex(CartesianIndex(index_pre..., intern_ind, index_post...))
    setindex!(z, convert(Int32, 1), ind)
  end
  Base.show(io, z)
end

function Base.showarg(io::IO, x::OneHotArray{T,N,M,L,A}, toplevel) where {T,N,M,L,A}
  print(io, "$(size(x)) OneHotArray")
  toplevel && print(io, " with one hot axis $A and eltype $T")
  return nothing
end

# this is from /LinearAlgebra/src/diagonal.jl, official way to print the dots:
function Base.replace_in_print_matrix(x::OneHotArray, i::Integer, j::Integer, s::AbstractString)
  x[i,j] > 0 ? s : Base.replace_with_centered_mark(s)
end

# copy CuArray versions back before trying to print them:
for fun in (:show, :print_array)  # print_array is used by 3-arg show
  @eval begin
    Base.$fun(io::IO, X::OneHotArray{T,N,M,L, <:AbstractGPUArray}) where {T,N,M,L} = 
      Base.$fun(io, adapt(Array, X))
    Base.$fun(io::IO, X::LinearAlgebra.AdjOrTrans{Bool, <:OneHotArray{T,N,M,L,<:AbstractGPUArray}}) where {T,N,M,L} = 
      Base.$fun(io, adapt(Array, X))
  end
end

# Adapt.adapt_structure(T, x::OneHotArray) = OneHotArray(adapt(T, _indices(x)), x.nlabels) # TODO: edit this

# function Base.BroadcastStyle(::Type{<:OneHotArray{<:Any, <:Any, var"N+1", T}}) where {var"N+1", T <: AbstractGPUArray}  # TODO: edit this
#   # We want CuArrayStyle{N+1}(). There's an AbstractGPUArrayStyle but it doesn't do what we need. 
#   S = Base.BroadcastStyle(T)
#   # S has dim N not N+1. The following hack to fix it relies on the arraystyle having N as its first type parameter, which
#   # isn't guaranteed, but there are not so many GPU broadcasting styles in the wild. (Far fewer than there are array wrappers.)
#   (typeof(S).name.wrapper){var"N+1"}()
# end

# Base.map(f, x::OneHotLike) = Base.broadcast(f, x)

Base.argmax(x::OneHotArray; dims = Colon()) =
  dims == onehotaxis(x) ?
    argmax.(x.onehotvectors) :
    invoke(argmax, Tuple{AbstractArray}, x; dims = dims)

"""
    OneHotMatrix{T, L, A} = OneHotArray{T,2,1,L,1}
    OneHotMatrix(indices, L)

A one-hot matrix (with `L` labels) typically constructed using [`onehotbatch`](@ref).
"""
const OneHotMatrix{T, L} = OneHotArray{T, 2,1,L,1}
OneHotMatrix(indices, L) = OneHotArray(1, [OneHotVector(index,L) for index in indices])