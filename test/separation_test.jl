using LMCLUS
using Distributions
using Test
using Random

@testset "Thresholding" begin

	Random.seed!(283739285);
	N = 10000
	bins = 200
	generate_sample(N, P1, μ1, σ1, P2, μ2, σ2) = vcat(rand(Normal(μ1,σ1), round(Int, N*P1)),rand(Normal(μ2,σ2), round(Int, N*P2)))

	# Following tests' parameters are taken from origin work:
	# J. Kittler & J. Illingworth: "Minimum Error Thresholding" Pattern Recognition, Vol 19, nr 1. 1986, pp. 41-47.
	# See fig.1 and fih.2

	# Test 1
	D = generate_sample(N, 0.5, 50, 15, 0.5, 150, 15)
	mindint, maxdist = extrema(D)
	res = separation(LMCLUS.Kittler, D, bins=bins)
	@test threshold(res) ≈ 100.0 atol=3.0
	@test res.mindist == mindint
	@test res.maxdist == maxdist
	@test res.bins == bins

	# Test 2
	res = separation(LMCLUS.Kittler, generate_sample(N, 0.25, 38, 9, 0.75, 121, 44), bins=bins)
	@test threshold(res) ≈ 65.0 atol=2.0

	# Test 3
	res = separation(LMCLUS.Kittler, generate_sample(N, 0.45, 47, 13, 0.55, 144, 25), bins=bins)
	@test threshold(res) ≈ 85.0 atol=2.0

	# Test 4
	res = separation(LMCLUS.Kittler, generate_sample(N, 0.5, 50, 4, 0.5, 150, 30), bins=bins)
	@test threshold(res) ≈ 65.0 atol=2.

	# Try unimodal histogram
	D = rand(Normal(1, 10), N)
	mindint, maxdist = extrema(D)
	res = separation(LMCLUS.Kittler, D, bins=bins)
	@test criteria(res) == 0.0
	@test threshold(res) == maxdist # threshold is set to maximal element
	@test res.mindist == mindint
	@test res.maxdist == maxdist
	@test res.bins == bins

	## Otsu ##
	# Test 1
	res = separation(LMCLUS.Otsu, generate_sample(N, 0.5, 50, 15, 0.5, 150, 15), bins=bins)
	@test threshold(res) ≈ 100.0 atol=5.0

	# Test 2
	res = separation(LMCLUS.Otsu, generate_sample(N, 0.25, 38, 9, 0.75, 121, 44), bins=bins)
	@test threshold(res) ≈ 108.0 atol=2.0

	# Test 3
	res = separation(LMCLUS.Otsu, generate_sample(N, 0.45, 47, 13, 0.55, 144, 25), bins=bins)
	@test threshold(res) ≈ 112.0 atol=2.0

	# Test 4
	res = separation(LMCLUS.Otsu, generate_sample(N, 0.5, 50, 4, 0.5, 150, 30), bins=bins)
	@test threshold(res) ≈ 121.0 atol=2.

	# Try unimodal histogram
	res = separation(LMCLUS.Otsu, rand(Normal(1, 10), N), bins=bins)
	@test threshold(res) ≈ 1.0 atol=1.

end
