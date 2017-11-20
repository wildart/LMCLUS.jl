using LMCLUS
using Distributions
using Base.Test

@testset "Thresholding" begin

	srand(283739285)
	N = 5000
	bins = 100
	generate_sample(N, μ1, σ1, μ2, σ2) = vcat(rand(Normal(μ1,σ1), N),rand(Normal(μ2,σ2), N))

	# Following tests' parameters are taken from origin work:
	# J. Kittler & J. Illingworth: "Minimum Error Thresholding" Pattern Recognition, Vol 19, nr 1. 1986, pp. 41-47.
	# See fig.1 and fih.2

	# Test 1
	res = separation(LMCLUS.Kittler, generate_sample(N, 50, 15, 150, 15), bins=bins)
	@test threshold(res) ≈ 100.0 atol=1.0

	# Test 2
	res = separation(LMCLUS.Kittler, generate_sample(N, 38, 9, 121, 44), bins=bins)
	@test threshold(res) ≈ 64.0 atol=1.0

	# Test 3
	res = separation(LMCLUS.Kittler, generate_sample(N, 47, 13, 144, 25), bins=bins)
	@test threshold(res) ≈ 84.0 atol=1.0

	# Test 4
	res = separation(LMCLUS.Kittler, generate_sample(N, 50, 4, 150, 30), bins=bins)
	@test threshold(res) ≈ 64.0 atol=1.

	# Try unimodal histogram
	@test_throws LMCLUS.LMCLUSException separation(LMCLUS.Kittler, rand(Normal(1, 10), N), bins=10)

end
