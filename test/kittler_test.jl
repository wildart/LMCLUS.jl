using LMCLUS
using Distributions
using Base.Test

N = 5000
bins = 100
generate_sample(N, μ1, σ1, μ2, σ2) = vcat(rand(Normal(μ1,σ1), N),rand(Normal(μ2,σ2), N))

# Following tests' parameters are taken from origin work:
# J. Kittler & J. Illingworth: "Minimum Error Thresholding" Pattern Recognition, Vol 19, nr 1. 1986, pp. 41-47.
# See fig.1 and fih.2

# Test 1
res = kittler(generate_sample(N, 50, 15, 150, 15), bins=bins)
println("Threshold: ", res[3])#, " (",res[5],")")
@test_approx_eq_eps res[3] 102.0 5

# Test 2
res = kittler(generate_sample(N, 38, 9, 121, 44), bins=bins)
println("Threshold: ", res[3])#, " (",res[5],")")
@test_approx_eq_eps res[3] 65.0 5

# Test 3
res = kittler(generate_sample(N, 47, 13, 144, 25), bins=bins)
println("Threshold: ", res[3])#, " (",res[5],")")
@test_approx_eq_eps res[3] 85.0 5

# Test 4
res = kittler(generate_sample(N, 50, 4, 150, 30), bins=bins)
println("Threshold: ", res[3])#, " (",res[5],")")
@test_approx_eq_eps res[3] 64.0 5

# Try unimodal histogram
@test_throws kittler(rand(Normal(1, 10), N), bins=bins)
