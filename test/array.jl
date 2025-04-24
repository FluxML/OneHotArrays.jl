ov = OneHotVector(rand(1:10), 10)
ov2 = OneHotVector(rand(1:11), 11)
om = OneHotMatrix(rand(1:10, 5), 10)
om2 = OneHotMatrix(rand(1:11, 5), 11)
oa = OneHotArray(rand(1:10, 5, 5), 10)

# sizes
@testset "Base.size" begin
  @test size(ov) == (10,)
  @test size(om) == (10, 5)
  @test size(oa) == (10, 5, 5)
end

@testset "Indexing" begin
  # vector indexing
  @test ov[3] == (ov.indices == 3)
  @test ov[:] == ov

  # matrix indexing
  @test om[3, 3] == (om.indices[3] == 3)
  @test om[:, 3] == OneHotVector(om.indices[3], 10)
  @test om[3, :] == (om.indices .== 3)
  @test om[:, :] == om
  @test om[:] == reshape(om, :)

  # array indexing
  @test oa[3, 3, 3] == (oa.indices[3, 3] == 3)
  @test oa[:, 3, 3] == OneHotVector(oa.indices[3, 3], 10)
  @test oa[3, :, 3] == (oa.indices[:, 3] .== 3)
  @test oa[3, :, :] == (oa.indices .== 3)
  @test oa[:, 3, :] == OneHotMatrix(oa.indices[3, :], 10)
  @test oa[:, :, :] == oa
  @test oa[:] == reshape(oa, :)

  # cartesian indexing
  @test oa[CartesianIndex(3, 3, 3)] == oa[3, 3, 3]

  # linear indexing
  @test om[11] == om[1, 2]
  @test oa[52] == oa[2, 1, 2]
  @test copyto!(rand(50,1), om) == reshape(om,:,1)  # hits invoke path
  @test copyto!(rand(51,1), om)[1:50] == vec(om)
  @test_throws BoundsError copyto!(rand(49,1), om)

  # bounds checks
  @test_throws BoundsError ov[0]
  @test_throws BoundsError om[2, -1]
  @test_throws BoundsError oa[11, 5, 5]
  @test_throws BoundsError oa[:, :]
end

@testset "Concatenating" begin
  # vector cat
  @test hcat(ov, ov) == OneHotMatrix(vcat(ov.indices, ov.indices), 10)
  @test hcat(ov, ov) isa OneHotMatrix
  @test vcat(ov, ov) == vcat(collect(ov), collect(ov))
  @test cat(ov, ov; dims = 3) == OneHotArray(cat(ov.indices, ov.indices; dims = 2), 10)
  @test cat(ov, ov; dims = 3) isa OneHotArray
  @test cat(ov, ov; dims = Val(3)) == OneHotArray(cat(ov.indices, ov.indices; dims = 2), 10)
  @test cat(ov, ov; dims = Val(3)) isa OneHotArray
  @test cat(ov, ov; dims = (1, 2)) == cat(collect(ov), collect(ov); dims = (1, 2))

  # matrix cat
  @test hcat(om, om) == OneHotMatrix(vcat(om.indices, om.indices), 10)
  @test hcat(om, om) isa OneHotMatrix
  @test vcat(om, om) == vcat(collect(om), collect(om))  # not one-hot!
  @test cat(om, om; dims = 1) == vcat(collect(om), collect(om))
  @test cat(om, om; dims = Val(1)) == vcat(collect(om), collect(om))
  @test cat(om, om; dims = 3) == OneHotArray(cat(om.indices, om.indices; dims = 2), 10)
  @test cat(om, om; dims = 3) isa OneHotArray
  @test cat(om, om; dims = Val(3)) == OneHotArray(cat(om.indices, om.indices; dims = 2), 10)
  @test cat(om, om; dims = Val(3)) isa OneHotArray
  @test cat(om, om; dims = (1, 2)) == cat(collect(om), collect(om); dims = (1, 2))

  # array cat
  @test cat(oa, oa; dims = 3) == OneHotArray(cat(oa.indices, oa.indices; dims = 2), 10)
  @test cat(oa, oa; dims = 3) isa OneHotArray
  @test cat(oa, oa; dims = 1) == cat(collect(oa), collect(oa); dims = 1)

  # stack
  @test stack([ov, ov]) == hcat(ov, ov)
  @test stack([ov, ov, ov]) isa OneHotMatrix
  @test stack([om, om]) == cat(om, om; dims = 3)
  @test stack([om, om]) isa OneHotArray
  @test stack([oa, oa, oa, oa]) isa OneHotArray

  # reduce(hcat)
  @test reduce(hcat, [ov, ov]) == hcat(ov, ov)
  @test reduce(hcat, [ov, ov]) isa OneHotMatrix
  @test reduce(hcat, [onehotbatch(1, 1:3), onehotbatch(1, 1:3)]) == [1 1; 0 0; 0 0]
  @test reduce(hcat, [onehotbatch(1, 1:3), onehotbatch(1, 1:3)]) isa OneHotMatrix
  @test reduce(hcat, [om, om]) == hcat(om, om)
  @test reduce(hcat, [om, om]) isa OneHotMatrix

  # proper error handling of inconsistent sizes
  @test_throws DimensionMismatch hcat(ov, ov2)
  @test_throws DimensionMismatch hcat(om, om2)
  @test_throws DimensionMismatch stack([om, om2])
  @test_throws DimensionMismatch reduce(hcat, [om, om2])
end

@testset "Base.reshape" begin
  # reshape test
  @test reshape(oa, 10, 25) isa OneHotLike
  @test reshape(oa, 10, :) isa OneHotLike
  @test reshape(oa, :, 25) isa OneHotLike
  @test reshape(oa, 50, :) isa OneHotLike
  @test reshape(oa, 5, 10, 5) isa OneHotLike
  @test reshape(oa, (10, 25)) isa OneHotLike

  @testset "w/ cat" begin
    r = reshape(oa, 10, :)
    @test hcat(r, r) isa OneHotArray
    @test cat(r, r; dims = 2) isa OneHotArray
    @test cat(r, r; dims = Val(2)) isa OneHotArray
    @test vcat(r, r) isa Array{Bool}
  end

  @testset "w/ argmax" begin
    r = reshape(oa, 10, :)
    @test argmax(r) == argmax(OneHotMatrix(reshape(oa.indices, :), 10))
    @test OneHotArrays._fast_argmax(r) == collect(reshape(oa.indices, :))
  end
end

@testset "Base.argmax" begin
  # argmax test
  @test argmax(ov) == argmax(convert(Array{Bool}, ov))
  @test argmax(om) == argmax(convert(Array{Bool}, om))
  @test argmax(om; dims = 1) == argmax(convert(Array{Bool}, om); dims = 1)
  @test argmax(om; dims = 2) == argmax(convert(Array{Bool}, om); dims = 2)
  @test argmax(oa; dims = 1) == argmax(convert(Array{Bool}, oa); dims = 1)
  @test argmax(oa; dims = 3) == argmax(convert(Array{Bool}, oa); dims = 3)
end

@testset "Forward map to broadcast" begin
  @test map(identity, oa) == oa
  @test map(x -> 2 * x, oa) == 2 .* oa
end
