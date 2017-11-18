using Base.Test

@testset "LMCLUS" begin
    include("types_test.jl")
    include("kittler_test.jl")
    include("lmclus_test.jl")
    include("lmclus_p_test.jl")
    include("mdl_test.jl")
end
