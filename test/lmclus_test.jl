using LMCLUS
using Base.Test

ds = eye(5)
p = LMCLUSParameters(5, 3, 0, 20, 1.0, 0.0001, 0.1, 0, 3, 0.003, false)
@test lmclus(ds, p) == 2
