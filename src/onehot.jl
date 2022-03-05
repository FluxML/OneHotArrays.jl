"""
    onehot(x, labels, [default])

Return a `OneHotVector` which is roughly a sparse representation of `x .== labels`.

Instead of storing say `Vector{Bool}`, it stores the index of the first occurrence 
of `x` in `labels`. If `x` is not found in labels, then it either returns `onehot(default, labels)`,
or gives an error if no default is given.

See also [`onehotbatch`](@ref) to apply this to many `x`s, 
and [`onecold`](@ref) to reverse either of these, as well as to generalise `argmax`.

# Examples
```jldoctest
julia> β = Flux.onehot(:b, (:a, :b, :c))
3-element OneHotVector(::UInt32) with eltype Bool:
 ⋅
 1
 ⋅

julia> αβγ = (Flux.onehot(0, 0:2), β, Flux.onehot(:z, [:a, :b, :c], :c))  # uses default
(Bool[1, 0, 0], Bool[0, 1, 0], Bool[0, 0, 1])

julia> hcat(αβγ...)  # preserves sparsity
3×3 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
```
"""
function onehot(x, labels)
  i = _findval(x, labels)
  isnothing(i) && error("Value $x is not in labels")
  OneHotVector{UInt32, length(labels)}(i)
end

function onehot(x, labels, default)
  i = _findval(x, labels)
  isnothing(i) && return onehot(default, labels)
  OneHotVector{UInt32, length(labels)}(i)
end

_findval(val, labels) = findfirst(isequal(val), labels)
# Fast unrolled method for tuples:
function _findval(val, labels::Tuple, i::Integer=1)
  ifelse(isequal(val, first(labels)), i, _findval(val, Base.tail(labels), i+1))
end
_findval(val, labels::Tuple{}, i::Integer) = nothing

"""
    onehotbatch(xs, labels, [default])

Returns a `OneHotMatrix` where `k`th column of the matrix is [`onehot(xs[k], labels)`](@ref onehot).
This is a sparse matrix, which stores just a `Vector{UInt32}` containing the indices of the
nonzero elements.

If one of the inputs in `xs` is not found in `labels`, that column is `onehot(default, labels)`
if `default` is given, else an error.

If `xs` has more dimensions, `M = ndims(xs) > 1`, then the result is an 
`AbstractArray{Bool, M+1}` which is one-hot along the first dimension, 
i.e. `result[:, k...] == onehot(xs[k...], labels)`.

Note that `xs` can be any iterable, such as a string. And that using a tuple
for `labels` will often speed up construction, certainly for less than 32 classes.

# Examples
```jldoctest
julia> oh = Flux.onehotbatch("abracadabra", 'a':'e', 'e')
5×11 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 1  ⋅  ⋅  1  ⋅  1  ⋅  1  ⋅  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅

julia> reshape(1:15, 3, 5) * oh  # this matrix multiplication is done efficiently
3×11 Matrix{Int64}:
 1  4  13  1  7  1  10  1  4  13  1
 2  5  14  2  8  2  11  2  5  14  2
 3  6  15  3  9  3  12  3  6  15  3
```
"""
onehotbatch(ls, labels, default...) = _onehotbatch(ls, length(labels) < 32 ? Tuple(labels) : labels, default...)
# NB function barier:
_onehotbatch(ls, labels, default...) = batch([onehot(l, labels, default...) for l in ls])

"""
    onecold(y::AbstractArray, labels = 1:size(y,1))

Roughly the inverse operation of [`onehot`](@ref) or [`onehotbatch`](@ref): 
This finds the index of the largest element of `y`, or each column of `y`, 
and looks them up in `labels`.

If `labels` are not specified, the default is integers `1:size(y,1)` --
the same operation as `argmax(y, dims=1)` but sometimes a different return type.

# Examples
```jldoctest
julia> Flux.onecold([false, true, false])
2

julia> Flux.onecold([0.3, 0.2, 0.5], (:a, :b, :c))
:c

julia> Flux.onecold([ 1  0  0  1  0  1  0  1  0  0  1
                      0  1  0  0  0  0  0  0  1  0  0
                      0  0  0  0  1  0  0  0  0  0  0
                      0  0  0  0  0  0  1  0  0  0  0
                      0  0  1  0  0  0  0  0  0  1  0 ], 'a':'e') |> String
"abeacadabea"
```
"""
onecold(y::AbstractVector, labels = 1:length(y)) = labels[argmax(y)]
function onecold(y::AbstractArray, labels = 1:size(y, 1))
  indices = _fast_argmax(y)
  xs = isbits(labels) ? indices : collect(indices) # non-bit type cannot be handled by CUDA

  return map(xi -> labels[xi[1]], xs)
end

_fast_argmax(x::AbstractArray) = dropdims(argmax(x; dims = 1); dims = 1)
function _fast_argmax(x::OneHotLike)
  if _isonehot(x)
    return _indices(x)
  else
    return _fast_argmax(convert(_onehot_bool_type(x), x))
  end
end

ChainRulesCore.@non_differentiable onehot(::Any...)
ChainRulesCore.@non_differentiable onehotbatch(::Any...)
ChainRulesCore.@non_differentiable onecold(::Any...)

ChainRulesCore.@non_differentiable (::Type{<:OneHotArray})(indices::Any, L::Integer)
