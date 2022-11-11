module OneHotArrays

using Adapt
using ChainRulesCore
using GPUArraysCore
using LinearAlgebra
using Compat: Compat
using NNlib

export onehot, onehotbatch, onecold,
       OneHotArray, OneHotVector, OneHotMatrix, OneHotLike

include("array.jl")
include("onehot.jl")
include("linalg.jl")

end
