module OneHotArraysCategoricalArraysExt

println("loading?")

using OneHotArrays, CategoricalArrays

OneHotArrays.OneHotArray(cv::CategoricalValue) = OneHotVector(cv.ref, length(cv.pool.levels))

OneHotArrays.OneHotArray(ca::CategoricalArray) = OneHotArray(ca.refs, length(ca.pool))

end  # module
