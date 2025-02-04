
# Tests from Flux, probably not the optimal testset organisation!

@testset "CUDA" begin
  x = randn(5, 5)
  cx = cu(x)
  @test cx isa CuArray

  @test_skip onecold(cu([1.0, 2.0, 3.0])) == 3  # passes with CuArray with Julia 1.6, but fails with JLArray

  x = onehotbatch([1, 2, 3], 1:3)
  cx = cu(x)
  @test cx isa OneHotMatrix && cx.indices isa CuArray
  @test (cx .+ 1) isa CuArray

  xs = rand(5, 5)
  ys = onehotbatch(1:5,1:5)
  @test collect(cu(xs) .+ cu(ys)) ≈ collect(xs .+ ys)
end

@testset "onehot gpu" begin
  y = onehotbatch(ones(3), 1:2) |> cu;
  @test (repr("text/plain", y); true)

  gA = rand(3, 2) |> cu;
  if VERSION >= v"1.9" && CUDA.functional()
    @test gradient(A -> sum(A * y), gA)[1] isa CuArray 
  else
    @test gradient(A -> sum(A * y), gA)[1] isa CuArray  # fails with JLArray, bug in Zygote?
  end

  # some specialized implementations call only mul! and not *, so we must ensure this works
  @test LinearAlgebra.mul!(similar(gA, 3, 3), gA, y) ≈ gA*y

  #TODO: the below fails due to method ambiguity and GPU scalar indexing
  y = reshape(y, 3, 2)
  gA = rand(2, 3) |> cu
  @test_broken LinearAlgebra.mul!(similar(gA, 2, 2), gA, y) ≈ gA*y
end

@testset "onehotbatch(::CuArray, ::UnitRange)" begin
  y1 = onehotbatch([1, 3, 0, 2], 0:9) |> cu
  y2 = onehotbatch([1, 3, 0, 2] |> cu, 0:9)
  @test y1.indices == y2.indices
  @test_broken y1 == y2  # issue 28

  if !CUDA.functional()
    # Here CUDA gives an error which @test_throws does not notice,
    # although with JLArrays @test_throws it's fine.
    @test_throws Exception onehotbatch([1, 3, 0, 2] |> cu, 1:10)
    @test_throws Exception onehotbatch([1, 3, 0, 2] |> cu, -2:2)
  end
end

@testset "onecold gpu" begin
  y = onehotbatch(ones(3), 1:10) |> cu;
  l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
  @test onecold(y) isa CuArray
  @test y[3,:] isa CuArray
  @test onecold(y, l) == ['a', 'a', 'a']
end

@testset "onehot forward map to broadcast" begin
  oa = OneHotArray(rand(1:10, 5, 5), 10) |> cu
  @test all(map(identity, oa) .== oa)
  @test all(map(x -> 2 * x, oa) .== 2 .* oa)
end

@testset "show gpu" begin
  x = onehotbatch([1, 2, 3], 1:3)
  cx = cu(x)
  # 3-arg show
  @test contains(repr("text/plain", cx), "1  ⋅  ⋅")
  @test contains(repr("text/plain", cx), string(typeof(cx.indices)))
  # 2-arg show, https://github.com/FluxML/Flux.jl/issues/1905
  @test repr(cx) == "Bool[1 0 0; 0 1 0; 0 0 1]"
end
