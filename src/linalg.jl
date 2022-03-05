function Base.:(*)(A::AbstractMatrix, B::OneHotLike{<:Any, L}) where L
  _isonehot(B) || return invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
  size(A, 2) == L || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $L"))
  return A[:, onecold(B)]
end
  
function Base.:(*)(A::AbstractMatrix, B::OneHotLike{<:Any, L, 1}) where L
  _isonehot(B) || return invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
  size(A, 2) == L || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $L"))
  return NNlib.gather(A, _indices(B))
end

function Base.:(*)(A::AbstractMatrix, B::Adjoint{Bool, <:OneHotMatrix})
  B_dim = length(_indices(parent(B)))
  size(A, 2) == B_dim || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $B_dim"))
  return NNlib.scatter(+, A, _indices(parent(B)), dstsize=(size(A,1), size(B,2)))
end

for wrapper in [:Adjoint, :Transpose]
  @eval begin
    function Base.:*(A::$wrapper{<:Any, <:AbstractMatrix{T}}, b::OneHotVector{<:Any, L}) where {L, T}
      size(A, 2) == L ||
          throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $L"))

      return A[:, onecold(b)]
    end

    function Base.:*(A::$wrapper{<:Number, <:AbstractVector{T}}, b::OneHotVector{<:Any, L}) where {L, T}
      size(A, 2) == L ||
          throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $L"))

      return A[onecold(b)]
    end
  end
end
