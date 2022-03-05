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
end

@testset "AbstractMatrix-OneHotMatrix multiplication" begin
  A = [1 3 5; 2 4 6; 3 6 9]
  v = [1, 2, 3, 4, 5]
  X = reshape(v, (5, 1))
  b1 = OneHotMatrix([1, 1, 2, 2], 3)
  b2 = OneHotMatrix([2, 4, 1, 3], 5)
  b3 = OneHotMatrix([1, 1, 2], 4)
  b4 = reshape(OneHotMatrix([1 2 3; 2 2 1], 3), 3, :)
  b5 = reshape(b4, 6, :)
  b6 = reshape(OneHotMatrix([1 2 2; 2 2 1], 2), 3, :)
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
end
