using Test, OneHotArrays, CategoricalArrays

@testset "CategoricalArrays -> OneHotArrays" begin
    cval = CategoricalArrays.CategoricalValue('b', CategoricalArray('a':'z'))

    @test OneHotArray(cval) isa OneHotVector
    @test OneHotArray(cval) == (('a':'z') .== 'b')

    @test_broken OneHotVector(cval) isa OneHotVector  # surely if OneHotArray works, subtypes should too
    @test_broken convert(OneHotArray, cval) isa OneHotVector
    @test_broken onehot(cval) isa OneHotVector  # possibly we should define this? Instead?

    cvec = categorical(string.([:a, :b, :b, :c, :d, :e]))

    @test OneHotArray(cvec) isa OneHotMatrix
    @test size(OneHotArray(cvec)) == (5, 6)
    @test onecold(OneHotArray(cvec)) == [1, 2, 2, 3, 4, 5]

    @test_broken onehotbatch(cvec) isa OneHotMatrix  # possibly we should define this? Instead?
end
