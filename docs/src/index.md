# OneHotArrays.jl

[![CI](https://github.com/FluxML/OneHotArrays.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FluxML/OneHotArrays.jl/actions/workflows/CI.yml)

Memory efficient one-hot array encodings (primarily for use in machine-learning contexts like Flux.jl).

## Usage

One-hot arrays are boolean arrays where only a single element in the first dimension is `true` (i.e. "hot"). OneHotArrays.jl stores such arrays efficiently by encoding a N-dimensional array of booleans as a (N - 1)-dimensional array of integers. For example, the one-hot vector below only uses a single `UInt32` for storage.

```julia
julia> β = onehot(:b, (:a, :b, :c))
3-element OneHotVector(::UInt32) with eltype Bool:
 ⋅
 1
 ⋅
```

As seen above, the one-hot encoding can be useful for representing labeled data. The label `:b` is encoded into a 3-element vector where the "hot" element indicates the label from the set `(:a, :b, :c)`.

We can also encode a batch of one-hot vectors or reverse the encoding.

```julia
julia> oh = onehotbatch("abracadabra", 'a':'e', 'e')
5×11 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 1  ⋅  ⋅  1  ⋅  1  ⋅  1  ⋅  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅

julia> Flux.onecold(β, (:a, :b, :c))
:b

julia> Flux.onecold([0.3, 0.2, 0.5], (:a, :b, :c))
:c
```

In addition to functions for encoding and decoding data as one-hot, this package provides numerous "fast-paths" for linear algebraic operations with one-hot arrays. For example, multiplying by a matrix by a one-hot vector triggers an indexing operation instead of a matrix multiplication.
