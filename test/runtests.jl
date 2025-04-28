using OneHotArrays
using Test, LinearAlgebra
using Compat: stack

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
  using CUDA: @allowscalar
  @info "starting CUDA tests"
else
  @info "CUDA not functional, testing with JLArrays instead"
  using JLArrays  # fake GPU array, for testing
  JLArrays.allowscalar(false)
  using JLArrays: @allowscalar
  cu = jl
  CuArray{T,N} = JLArray{T,N}
end

@testset "GPUArrays" begin
  include("gpu.jl")
end
