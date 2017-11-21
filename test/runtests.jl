using Base.Test

@testset "LMCLUS" begin
    include("utils_test.jl")
    include("types_test.jl")
    include("separation_test.jl")
    include("lmclus_test.jl")
    include("lmclus_p_test.jl")
    include("mdl_test.jl")
end
