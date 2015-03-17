addprocs(1)

@everywhere using LMCLUS
using Base.Test

p = LMCLUSParameters(5)
p.log_level = 2
p.random_seed = 4572489057
ds = readdlm(Pkg.dir("LMCLUS", "test", "testData"), ',')
manifolds = lmclus(ds[:,1:end-1]',p)
@test length(manifolds) >= 3