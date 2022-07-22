# OneHotArrays.jl

[![Documentation][https://img.shields.io/badge/docs-latest-blue.svg]][https://fluxml.ai/OneHotArrays.jl/dev/]
[![Tests](https://github.com/FluxML/OneHotArrays.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FluxML/OneHotArrays.jl/actions/workflows/CI.yml)

Memory efficient one-hot array encodings. Originally part of [Flux.jl](https://github.com/FluxML/Flux.jl).

```julia
julia> using OneHotArrays

julia> m = onehotbatch("abracadabra", 'a':'e', 'e')  # stores only a vector of indices
5×11 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 1  ⋅  ⋅  1  ⋅  1  ⋅  1  ⋅  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅

julia> @which rand(3,5) * m  # this can be done efficiently
*(A::AbstractMatrix, B::Union{OneHotArray{var"#s14", L, 1, var"N+1", I}, Base.ReshapedArray{Bool, var"N+1", <:OneHotArray{var"#s14", L, <:Any, <:Any, I}}} where {var"#s14", var"N+1", I}) where L
     @ OneHotArrays ~/.julia/dev/OneHotArrays/src/linalg.jl:7
```
