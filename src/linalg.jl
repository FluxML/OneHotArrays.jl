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

function Base.:(*)(A::OneHotMatrix, B::AbstractMatrix{<:Number})
  size(A, 2) == size(B, 1) || throw(DimensionMismatch("A has dimensions $(size(A)) but B has dimensions $(size(B))"))  # probably caught later anyway
  T = promote_type(eltype(B), Int)
  Y = similar(B, T, size(A, 1), size(B, 2))
  LinearAlgebra.mul!(transpose(Y), transpose(B), transpose(A))  # uses matrix * wrapper(OneHotMatrix) method below
  return Y
end

for wrapper in [:Adjoint, :Transpose]
  @eval begin
    # Adjoint * OneHotVector cases
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

    # Matrix * Adjoint(OneHotVector)
    function LinearAlgebra.mul!(Y::AbstractMatrix, A::AbstractMatrix, B::$wrapper{Bool,<:OneHotMatrix})
      if size(A,2) ≠ size(B,1)
        throw(DimensionMismatch("Matrix column must correspond with the OneHot Size $(size(A,2)) ≠ $(size(B,1))"))
      end
      if !(size(Y,1) == size(A,1) && size(Y,2) == size(B,2))
        throw(DimensionMismatch("Invalid output matrix size for multiplication of matrix sizes $(size(A)) and $(size(B))"))
      end
      # note that the fill! is the same thing done by NNlib.scatter so it is not more expensive
      fill!(Y, zero(eltype(Y)))
      return NNlib.scatter!(+, Y, A, _indices(parent(B)))
    end

    # Adjoint(OneHotVector) * Matrix
    function LinearAlgebra.mul!(Y::AbstractVecOrMat, A::$wrapper{Bool,<:OneHotMatrix}, B::AbstractMatrix)
      op = LinearAlgebra.wrapperop(A)
      LinearAlgebra.mul!(op(Y), op(B), A.parent)  # uses OneHotMatrix method below
      return Y
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
  idxs =  _indices(B)
  if idxs isa Integer  # occurs whe B is AbstractVector
    copyto!(Y, view(A, :, idxs))
  else
    NNlib.gather!(Y, A, idxs)
  end
end

function LinearAlgebra.mul!(Y::AbstractVecOrMat{<:Real}, A::OneHotMatrix, B::AbstractVecOrMat)
  LinearAlgebra.mul!(transpose(Y), transpose(B), transpose(A))  #  uses matrix * wrapper(OneHotMatrix) method above
  return Y
end
