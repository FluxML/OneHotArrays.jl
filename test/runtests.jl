using OneHotArrays
using Test

@testset verbose=true "OneHotArrays" begin
  @testset "Array" begin
    include("array.jl")
  end

  @testset "Constructors" begin
    include("onehot.jl")
  end

  @testset "Linear Algebra" begin
    include("linalg.jl")
  end
end
