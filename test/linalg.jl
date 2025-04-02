@testset "abstractmatrix onehotvector multiplication" begin
  A = [1 3 5; 2 4 6; 3 6 9]
  v = [1, 2, 3, 4, 5]
  X = reshape(v, (5, 1))
  b1 = OneHotVector(1, 3)
  b2 = OneHotVector(3, 5)

  @test A * b1 == A[:,1]
  @test b1' * A == Array(b1') * A
  @test A' * b1 == A' * Array(b1)
  @test v' * b2 == v' * Array(b2)
  @test transpose(X) * b2 == transpose(X) * Array(b2)
  @test transpose(v) * b2 == transpose(v) * Array(b2)
  @test_throws DimensionMismatch A*b2

  # in-place with mul!
  c1 = fill(NaN, 3)
  @test mul!(c1, A, b1) == A * Array(b1)
  @test c1 == A * Array(b1)
  @test mul!(c1, transpose(A), b1) == transpose(A) * Array(b1)
  @test mul!(zeros(3,1), A, b1) == reshape(A * b1, 3,1)
  @test mul!([NaN], transpose(v), b2) == mul!([NaN], transpose(v), Array(b2))
  
  @test_throws DimensionMismatch mul!(zeros(5), A, b1)
  @test_throws DimensionMismatch mul!(c1, X, b1)
end

@testset "AbstractMatrix-OneHotMatrix multiplication" begin
  A = [1 3 5; 2 4 6; 3 6 9]
  v = [1, 2, 3, 4, 5]
  X = reshape(v, (5, 1))
  b1 = OneHotMatrix([1, 1, 2, 2], 3)
  b2 = OneHotMatrix([2, 4, 1, 3], 5)
  b3 = OneHotMatrix([1, 1, 2], 4)
  b4 = reshape(OneHotMatrix([1 2 3; 2 2 1], 3), 3, :)
  @test OneHotArrays._isonehot(b4)
  b5 = reshape(b4, 6, :)
  b6 = reshape(OneHotMatrix([1 2 2; 2 2 1], 2), 3, :)
  @test !OneHotArrays._isonehot(b6)
  b7 = reshape(OneHotMatrix([1 2 3; 1 2 3], 3), 6, :)

  @test A * b1 == A[:,[1, 1, 2, 2]]
  @test b1' * A == Array(b1') * A
  @test A' * b1 == A' * Array(b1)
  @test A * b3' == A * Array(b3')
  @test transpose(X) * b2 == transpose(X) * Array(b2)
  @test A * b4 == A[:,[1, 2, 2, 2, 3, 1]]
  @test A * b5' == hcat(A[:,[1, 2, 3, 3]], A[:,1]+A[:,2], zeros(Int64, 3))
  @test A * b6 == hcat(A[:,1], 2*A[:,2], A[:,2], A[:,1]+A[:,2])
  @test A * b7' == A[:,[1, 2, 3, 1, 2, 3]]

  @test_throws DimensionMismatch A*b1'
  @test_throws DimensionMismatch A*b2
  @test_throws DimensionMismatch A*b2'
  @test_throws DimensionMismatch A*b6'
  @test_throws DimensionMismatch A*b7

  # in-place with mul!
  c1 = fill(NaN, 3, 4)
  @test mul!(c1, A, b1) == A * b1
  @test c1 == A * b1
  
  c4 = fill(NaN, 3, 6)
  @test mul!(c4, A, b4) == A * b4  # b4 is reshaped but still one-hot
  @test mul!(c4, A', b4) == A' * b4
  c6 = fill(NaN, 3, 4)
  @test mul!(c6, A, b6) == A * b6  # b4 is reshaped and not one-hot
  @test mul!(c6, A', b6) == A' * b6
  
  @test_throws DimensionMismatch mul!(c1, A, b2)
  @test_throws DimensionMismatch mul!(c1, A, b4)
  @test_throws DimensionMismatch mul!(c4, A, b1)
  @test_throws DimensionMismatch mul!(zeros(10, 3), A, b1)
end
