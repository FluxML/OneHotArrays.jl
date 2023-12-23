function Base.:(*)(A::AbstractMatrix, B::OneHotArray)
  onehotaxis(B) == 1 || return invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(size(B, 1))"))
  return A[:, argmax(B;dims=1)]
end
  
function Base.:(*)(A::AbstractMatrix, B::OneHotArray{T,2,1,L,Ax}) where {T,L,Ax}
  onehotaxis(B) == 1 || return invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(size(B, 1))"))
  return NNlib.gather(A, [v.index for v in B.onehotvectors])
end

function Base.:(*)(A::AbstractMatrix, B::Adjoint{Bool, <:OneHotArray})
  B_dim = length(_indices(parent(B)))
  size(A, 2) == B_dim || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $B_dim"))
  return NNlib.scatter(+, A, _indices(parent(B)), dstsize=(size(A,1), size(B,2)))
end

for wrapper in [:Adjoint, :Transpose]
  @eval begin
    function Base.:*(A::$wrapper{<:Any, <:AbstractMatrix{T}}, b::OneHotVector{T,L}) where {T,L}
      size(A, 2) == L ||
          throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(L)"))
      return A[:, argmax(b)]
    end

    function Base.:*(A::$wrapper{<:Number, <:AbstractVector{T}}, b::OneHotVector{T,L}) where {T,L}
      size(A, 2) == L ||
          throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(L)"))
      return A[argmax(b)]
    end
  end
end
