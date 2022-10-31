<img align="right" width="200px" src="https://github.com/FluxML/OneHotArrays.jl/raw/main/docs/src/assets/logo.png">

# OneHotArrays.jl

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://fluxml.ai/OneHotArrays.jl/dev/)
[![Tests](https://github.com/FluxML/OneHotArrays.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FluxML/OneHotArrays.jl/actions/workflows/CI.yml)

This package provides memory efficient one-hot array encodings.
It was originally part of [Flux.jl](https://github.com/FluxML/Flux.jl).

```julia
julia> using OneHotArrays

julia> m = onehotbatch([10, 20, 30, 10, 10], 10:10:40)
4×5 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 1  ⋅  ⋅  1  1
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> dump(m)
OneHotMatrix{UInt32, 4, Vector{UInt32}}
  indices: Array{UInt32}((5,)) UInt32[0x00000001, 0x00000002, 0x00000003, 0x00000001, 0x00000001]

julia> @which rand(100, 4) * m
*(A::AbstractMatrix, B::Union{OneHotArray{var"#s14", L, 1, var"N+1", I}, Base.ReshapedArray{Bool, var"N+1", <:OneHotArray{var"#s14", L, <:Any, <:Any, I}}} where {var"#s14", var"N+1", I}) where L
     @ OneHotArrays ~/.julia/dev/OneHotArrays/src/linalg.jl:7
```
