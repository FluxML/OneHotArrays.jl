
@testset "onehotbatch gpu" begin
  # move to GPU after construction
  x = onehotbatch([1, 2, 3, 2], 1:3)
  @test cu(x) isa OneHotMatrix
  @test cu(x).indices isa CuArray
  
  # broadcast style works:
  @test (cu(x) .+ 1) isa CuArray
  xs = rand(5, 5)
  ys = onehotbatch(rand(1:5, 5), 1:5)
  @test collect(cu(xs) .+ cu(ys)) ≈ collect(xs .+ ys)

  # move to GPU before construction
  z1 = onehotbatch(cu([3f0, 1f0, 2f0, 2f0]), (1.0, 2f0, 3))
  @test z1.indices isa CuArray
  z2 = onehotbatch(cu([3f0, 1f0, 2f0, 2f0]), [1, 2], 2)  # with default
  @test z2.indices isa CuArray
  @test_throws ArgumentError onehotbatch(cu([1, 2, 3]), [1, 2])  # friendly error, not scalar indexing
  @test_throws ArgumentError onehotbatch(cu([1, 2, 3]), [1, 2], 5)
end

@testset "onecold gpu" begin
  y = onehotbatch(ones(3), 1:10) |> cu;
  l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
  @test onecold(y) isa CuArray
  @test y[3,:] isa CuArray
  @test onecold(y, l) == ['a', 'a', 'a']

  @test_skip onecold(cu([1.0, 2.0, 3.0])) == 3  # passes with CuArray with Julia 1.6, but fails with JLArray
end

@testset "matrix multiplication gpu" begin
  y = onehotbatch([1, 2, 1], [1, 2]) |> cu;
  A = rand(3, 2) |> cu;
  
  @test_broken collect(A * y) ≈ collect(A) * collect(y)
  
  @test_broken gradient(A -> sum(abs, A * y), A)[1] isa CuArray  # gather!(dst::JLArray, ...) fails
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
