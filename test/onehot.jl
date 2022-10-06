@testset "onehot constructors" begin
  @test onehot(20, 10:10:30) == [false, true, false]
  @test onehot(20, (10,20,30)) == [false, true, false]
  @test onehot(40, (10,20,30), 20) == [false, true, false]

  @test_throws Exception onehot('d', 'a':'c')
  @test_throws Exception onehot(:d, (:a, :b, :c))
  @test_throws Exception onehot('d', 'a':'c', 'e')
  @test_throws Exception onehot(:d, (:a, :b, :c), :e)

  @test onehotbatch([20, 10], 10:10:30) == Bool[0 1; 1 0; 0 0]
  @test onehotbatch([20, 10], (10,20,30)) == Bool[0 1; 1 0; 0 0]
  @test onehotbatch([40, 10], (10,20,30), 20) == Bool[0 1; 1 0; 0 0]

  @test onehotbatch("abc", 'a':'c') == Bool[1 0 0; 0 1 0; 0 0 1]
  @test onehotbatch("zbc", ('a', 'b', 'c'), 'a') == Bool[1 0 0; 0 1 0; 0 0 1]

  @test onehotbatch([10, 20], [30, 40, 50], 30) == Bool[1 1; 0 0; 0 0]

  @test_throws Exception onehotbatch([:a, :d], [:a, :b, :c])
  @test_throws Exception onehotbatch([:a, :d], (:a, :b, :c))
  @test_throws Exception onehotbatch([:a, :d], [:a, :b, :c], :e)
  @test_throws Exception onehotbatch([:a, :d], (:a, :b, :c), :e)

  floats = (0.0, -0.0, NaN, -NaN, Inf, -Inf)
  @test onecold(onehot(0.0, floats)) == 1
  @test onecold(onehot(-0.0, floats)) == 2  # as it uses isequal
  @test onecold(onehot(Inf, floats)) == 5

  # inferrabiltiy tests
  @test @inferred(onehot(20, 10:10:30)) == [false, true, false]
  @test @inferred(onehot(40, (10,20,30), 20)) == [false, true, false]
  @test @inferred(onehotbatch([20, 10], 10:10:30)) == Bool[0 1; 1 0; 0 0]
  @test @inferred(onehotbatch([40, 10], (10,20,30), 20)) == Bool[0 1; 1 0; 0 0]
end

@testset "onecold" begin
  a = [1, 2, 5, 3.]
  A = [1 20 5; 2 7 6; 3 9 10; 2 1 14]
  labels = ['A', 'B', 'C', 'D']

  @test onecold(a) == 3
  @test onecold(A) == [3, 1, 4]
  @test onecold(a, labels) == 'C'
  @test onecold(a, Tuple(labels)) == 'C'
  @test onecold(A, labels) == ['C', 'A', 'D']
  @test onecold(A, Tuple(labels)) == ['C', 'A', 'D']

  data = [:b, :a, :c]
  labels = [:a, :b, :c]
  hot = onehotbatch(data, labels)
  cold = onecold(hot, labels)

  @test cold == data
end

@testset "onehotbatch indexing" begin
  y = onehotbatch(ones(3), 1:10)
  @test y[:,1] isa OneHotVector
  @test y[:,:] isa OneHotMatrix
end
