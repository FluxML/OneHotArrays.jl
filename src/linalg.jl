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
