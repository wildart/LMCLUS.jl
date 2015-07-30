addprocs(1)

@everywhere using LMCLUS
using Base.Test

p = LMCLUSParameters(5)
p.log_level = 2
p.random_seed = 4572489057
testDataFile = joinpath("/",split(string(lmclus.env.defs.func.code.file),'/')[1:end-2]...,"test","testData")
ds = readdlm(testDataFile, ',')
manifolds1 = lmclus(ds[:,1:end-1]', p)
@test length(manifolds1) >= 3
manifolds2 = lmclus(ds[:,1:end-1]', p, 3)
@test length(manifolds2) >= 3
