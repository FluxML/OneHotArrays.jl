module OneHotArrays

using Adapt
using ChainRulesCore
using GPUArrays
using LinearAlgebra
using MLUtils 
using NNlib

export onehot, onehotbatch, onecold,
       OneHotArray, OneHotVector, OneHotMatrix, OneHotLike

include("array.jl")
include("onehot.jl")
include("linalg.jl")

end
