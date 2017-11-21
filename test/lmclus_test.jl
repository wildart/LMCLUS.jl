using Base.Test
using LMCLUS
using Combinatorics

@testset "Clustering" begin

	# Initialize parameters
	p = LMCLUS.Parameters(3)

	# Test sampling parameters
	p.number_of_clusters = 1
	@test LMCLUS.sample_quantity(1,3,1000,p,1) == 1
	p.number_of_clusters = 2
	p.sampling_heuristic = 1
	@test LMCLUS.sample_quantity(1,3,1000,p,1) == 13
	p.sampling_heuristic = 2
	@test LMCLUS.sample_quantity(1,3,1000,p,1) == 10
	p.sampling_heuristic = 3
	@test LMCLUS.sample_quantity(1,3,1000,p,1) == 10

	# Test distance calculation
	basis = reshape([1.,0.,0.,0.,1.,0.],3,2)
	point = [1.0,1.0,1.0]
	@test distance_to_manifold(point, basis) ≈ 1.0

	# Test parameters
	l = 1000
	x = rand(l)
	p = LMCLUS.Parameters(5)
	@test LMCLUS.hist_bin_size(x, p) == round(Int, l*p.max_bin_portion)
	p.hist_bin_size = 20
	@test LMCLUS.hist_bin_size(x, p) == p.hist_bin_size

	# Test clustering
	p = LMCLUS.Parameters(5)
	p.random_seed = 4572489057
	p.log_level = 0
	# println(p) # test show()

	testDataFile = joinpath(dirname(@__FILE__),"testData")
	ds = readdlm(testDataFile, ',')
	data = ds[:,1:end-1]'

	# test separation calculations
	s = LMCLUS.calculate_separation(data, [1,2,3], p)
	@test typeof(s[1]) == LMCLUS.Separation
	@test typeof(s[2]) == Vector{Float64}
	@test typeof(s[3]) == Matrix{Float64}

	# run clustering
	res = lmclus(data,p)
	@test nclusters(res) >= 3
	@test sum(counts(res)) == size(ds, 1)

	cnts = counts(res)
	@testset "Label Match" for idxs in combinations(1:nclusters(res),2)
		i = idxs[1]
		j = idxs[2]
		@test length(symdiff(labels(manifold(res,i)), labels(manifold(res, j)))) == cnts[i] + cnts[j]
	end

	# Sampling
	p.histogram_sampling = true
	p.number_of_clusters = 3
	res = lmclus(data,p)
	@test nclusters(res) >= 3
	p.histogram_sampling = false

	# RNG seed
	p.random_seed = 0
	res = lmclus(data,p)
	@test nclusters(res) >= 3

	# MDL
	LMCLUS.MDL.type!(LMCLUS.MDL.SizeIndependent)
	p.mdl = true
	res = lmclus(data,p)
	@test nclusters(res) >= 3
	p.mdl = false

	# adjust cluster bases
	p.basis_alignment = true
	res = lmclus(data,p)
	@test nclusters(res) >= 3
	p.basis_alignment = false

	# calculate cluster bounsds
	p.basis_alignment = true
	p.bounded_cluster = true
	res = lmclus(data,p)
	@test nclusters(res) >= 3
	p.bounded_cluster = false
	p.basis_alignment = false

	# Otsu
	p.sep_algo = LMCLUS.Otsu
	manifolds = lmclus(data,p)
	@test length(manifolds) >= 3
	p.sep_algo = LMCLUS.Kittler

end
