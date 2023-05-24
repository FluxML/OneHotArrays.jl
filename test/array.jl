ov = OneHotVector(rand(1:10), 10)
ov2 = OneHotVector(rand(1:11), 11)
om = OneHotMatrix(rand(1:10, 5), 10)
om2 = OneHotMatrix(rand(1:11, 5), 11)
oa = OneHotArray(rand(1:10, 5, 5), 10)
oa2 = OneHotArray(rand(1:10, 5, 5), 10, 2)

# sizes
@testset "Base.size" begin
  @test size(ov) == (10,)
  @test size(om) == (10, 5)
  @test size(oa) == (10, 5, 5)
  @test size(oa2) == (5, 10, 5)
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

  @test oa2[3, 3, 3] == (oa2.parent.indices[3, 3] == 3)
  @test oa2[3, :, 3] == OneHotVector(oa2.parent.indices[3, 3], 10)
  @test oa2[:, 3, 3] == (oa2.parent.indices[:, 3] .== 3)
  @test oa2[:, 3, :] == (oa2.parent.indices .== 3)
  @test oa2[3, :, :] == OneHotMatrix(oa2.parent.indices[3, :], 10)
  @test oa2[:, :, :] == oa2
  @test oa2[:] == reshape(oa2, :)

  # cartesian indexing
  @test oa[CartesianIndex(3, 3, 3)] == oa[3, 3, 3]
  @test oa2[CartesianIndex(3, 3, 3)] == oa2[3, 3, 3]

  # linear indexing
  @test om[11] == om[1, 2]
  @test oa[52] == oa[2, 1, 2]
  @test oa2[55] == oa2[1, 2, 2]

  # bounds checks
  @test_throws BoundsError ov[0]
  @test_throws BoundsError om[2, -1]
  @test_throws BoundsError oa[11, 5, 5]
  @test_throws BoundsError oa[:, :]
  @test_throws BoundsError oa2[5, 11, 5]
  @test_throws BoundsError oa2[:, :]
end

@testset "Concatenating" begin
  # vector cat
  @test hcat(ov, ov) == OneHotMatrix(vcat(ov.indices, ov.indices), 10)
  @test hcat(ov, ov) isa OneHotMatrix
  @test vcat(ov, ov) == vcat(collect(ov), collect(ov))
  @test cat(ov, ov; dims = 3) == OneHotArray(cat(ov.indices, ov.indices; dims = 2), 10)

  # matrix cat
  @test hcat(om, om) == OneHotMatrix(vcat(om.indices, om.indices), 10)
  @test hcat(om, om) isa OneHotMatrix
  @test vcat(om, om) == vcat(collect(om), collect(om))
  @test cat(om, om; dims = 3) == OneHotArray(cat(om.indices, om.indices; dims = 2), 10)

  # array cat
  @test cat(oa, oa; dims = 3) == OneHotArray(cat(oa.indices, oa.indices; dims = 2), 10)
  @test cat(oa, oa; dims = 3) isa OneHotArray
  @test cat(oa, oa; dims = 1) == cat(collect(oa), collect(oa); dims = 1)

  @test cat(oa2, oa2; dims = 3) == OneHotArray(cat(oa2.parent.indices, oa2.parent.indices; dims = 2), 10, 2)
  @test cat(oa2, oa2; dims = 2) == cat(collect(oa2), collect(oa2); dims = 2)

  # stack
  @test stack([ov, ov]) == hcat(ov, ov)
  @test stack([ov, ov, ov]) isa OneHotMatrix
  @test stack([om, om]) == cat(om, om; dims = 3)
  @test stack([om, om]) isa OneHotArray
  @test stack([oa, oa, oa, oa]) isa OneHotArray

  # proper error handling of inconsistent sizes
  @test_throws DimensionMismatch hcat(ov, ov2)
  @test_throws DimensionMismatch hcat(om, om2)
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
    @test vcat(r, r) isa Array{Bool}
  end

  @testset "w/ argmax" begin
    r = reshape(oa, 10, :)
    @test argmax(r) == argmax(OneHotMatrix(reshape(oa.indices, :), 10))
    @test OneHotArrays._fast_argmax(r) == collect(reshape(oa.indices, :))
  end

  @testset "w/ cat" begin
    r = reshape(oa2, 10, :)
    @test vcat(r, r) isa Array{Bool}
  end

  @testset "w/ argmax" begin
    oa2p = PermutedDimsArray(oa2, [2,1,3])
    r = reshape(oa2p, 10, :)
    @test argmax(r) == argmax(OneHotMatrix(reshape(oa2p.parent.parent.indices, :), 10))
    @test stack(collect(Tuple.(OneHotArrays._fast_argmax(r))))[1,:] == collect(reshape(oa2p.parent.parent.indices, :))
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
  @test argmax(oa2; dims = 2) == argmax(convert(Array{Bool}, oa2); dims = 2)
  @test argmax(oa2; dims = 3) == argmax(convert(Array{Bool}, oa2); dims = 3)
end

@testset "Forward map to broadcast" begin
  @test map(identity, oa) == oa
  @test map(x -> 2 * x, oa) == 2 .* oa
  @test map(identity, oa2) == oa2
  @test map(x -> 2 * x, oa2) == 2 .* oa2
end
