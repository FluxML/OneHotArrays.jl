using OneHotArrays
using Test

@testset "OneHotArray" begin
  include("array.jl")
end

@testset "Constructors" begin
  include("onehot.jl")
end

@testset "Linear Algebra" begin
  include("linalg.jl")
end

using Zygote
import CUDA
if CUDA.functional()
  using CUDA  # exports CuArray, etc
  CUDA.allowscalar(false)
  @info "starting CUDA tests"
else
  @info "CUDA not functional, testing with JLArrays instead"
  using JLArrays  # fake GPU array, for testing
  JLArrays.allowscalar(false)
  cu = jl
  CuArray{T,N} = JLArray{T,N}
end

@testset "GPUArrays" begin
  @test cu(rand(3)) .+ 1 isa CuArray
  include("gpu.jl")
end
