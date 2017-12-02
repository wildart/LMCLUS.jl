addprocs(1)

import LMCLUS
@everywhere using LMCLUS
using Base.Test

@testset "Clustering (Parallel)" begin
    @test nprocs() == 2

    p = LMCLUS.Parameters(5)
    p.random_seed = 4572489057
    testDataFile = joinpath(dirname(@__FILE__),"testData")
    ds = readdlm(testDataFile, ',')
    res1 = lmclus(ds[:,1:end-1]', p)
    @test nclusters(res1) >= 3
    res2 = lmclus(ds[:,1:end-1]', p, 2*nprocs())
    @test nclusters(res2) >= 3

end