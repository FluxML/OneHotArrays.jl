function Base.:(*)(A::AbstractMatrix, B::OneHotLike)
  _isonehot(B) || return invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(size(B, 1))"))
  return A[:, onecold(B)]
end
  
function Base.:(*)(A::AbstractMatrix, B::OneHotLike{<:Any, 1})
  _isonehot(B) || return invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(size(B, 1))"))
  return NNlib.gather(A, _indices(B))
end

function Base.:(*)(A::AbstractMatrix, B::Adjoint{Bool, <:OneHotMatrix})
  B_dim = length(_indices(parent(B)))
  size(A, 2) == B_dim || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $B_dim"))
  return NNlib.scatter(+, A, _indices(parent(B)), dstsize=(size(A,1), size(B,2)))
end

for wrapper in [:Adjoint, :Transpose]
  @eval begin
    function Base.:*(A::$wrapper{<:Any, <:AbstractMatrix{T}}, b::OneHotVector) where T
      size(A, 2) == length(b) ||
          throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(length(b))"))

      return A[:, onecold(b)]
    end

    function Base.:*(A::$wrapper{<:Number, <:AbstractVector{T}}, b::OneHotVector) where T
      size(A, 2) == length(b) ||
          throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(length(b))"))

      return A[onecold(b)]
    end
  end
end

function LinearAlgebra.mul!(Y::AbstractVecOrMat, A::AbstractMatrix, B::OneHotLike)
  _isonehot(B) || return invoke(mul!, Tuple{AbstractArray,AbstractMatrix,AbstractMatrix}, Y, A, B)
  if size(A,2) ≠ size(B,1)
    throw(DimensionMismatch("Matrix column must correspond with the OneHot Size $(size(A,2)) ≠ $(size(B,1))"))
  end
  if !(size(Y,1) == size(A,1) && size(Y,2) == size(B,2))
    throw(DimensionMismatch("Invalid output matrix size for multiplication of matrix sizes $(size(A)) and $(size(B))"))
  end
  # matmul sometimes wraps in ReshapedArray, taking parent is a simple way to handle that case
  idxs = onecold(parent(B))
  if idxs isa Integer  # occurs whe B is AbstractVector
    copyto!(Y, view(A, :, idxs))
  else
    NNlib.gather!(Y, A, idxs)
  end
end

