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
  @info "starting CUDA tests"
else
  @info "CUDA not functional, testing via GPUArrays"
  using GPUArrays
  GPUArrays.allowscalar(false)

  # GPUArrays provides a fake GPU array, for testing
  jl_file = normpath(joinpath(pathof(GPUArrays), "..", "..", "test", "jlarray.jl"))
  using Random  # loaded within jl_file
  include(jl_file)
  using .JLArrays
  cu = jl
  CuArray{T,N} = JLArray{T,N}
end

@test cu(rand(3)) .+ 1 isa CuArray

@testset "GPUArrays" begin
  include("gpu.jl")
end
