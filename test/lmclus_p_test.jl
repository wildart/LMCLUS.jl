addprocs(1)

import LMCLUS
@everywhere using LMCLUS
using Base.Test

@testset "LMCLUS Parallel" begin
    @test nprocs() == 2

    p = LMCLUSParameters(5)
    p.log_level = 0
    p.random_seed = 4572489057
    testDataFile = joinpath(dirname(@__FILE__),"testData")
    ds = readdlm(testDataFile, ',')
    manifolds1 = lmclus(ds[:,1:end-1]', p)
    @test length(manifolds1) >= 3
    manifolds2 = lmclus(ds[:,1:end-1]', p, 3)
    @test length(manifolds2) >= 3

end