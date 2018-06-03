using Base.Test
using LMCLUS
using Combinatorics

@testset "Clustering" begin

	# Test histogram size calculations
	srand(24975754857)
	x = rand(1000)
	@test LMCLUS.gethistogrambins(x, 0.0, 10, 5) == 10
	@test LMCLUS.gethistogrambins(x, 0.0, 10, 15) == 15
	@test LMCLUS.gethistogrambins(x, 0.01, 10, 15) == 233

	# Test distance calculation
	basis = reshape([1.,0.,0.,0.,1.,0.],3,2)
	point = [1.0,1.0,1.0]
	@test distance_to_manifold(point, basis)[1] ≈ 1.0
	@test distance_to_manifold(point, basis)[2] ≈ sqrt(2.0)

	# Test sampling parameters
	p = LMCLUS.Parameters(3)
	p.number_of_clusters = 1
	@test LMCLUS.sample_quantity(1,3,1000,p,1) == 1
	p.number_of_clusters = 2
	p.sampling_heuristic = 1
	@test LMCLUS.sample_quantity(1,3,1000,p,1) == 13
	p.sampling_heuristic = 2
	@test LMCLUS.sample_quantity(1,3,1000,p,1) == 10
	p.sampling_heuristic = 3
	@test LMCLUS.sample_quantity(1,3,1000,p,1) == 10

	# Test clustering
	p = LMCLUS.Parameters(5)
	p.random_seed = 4572489057

	testDataFile = joinpath(dirname(@__FILE__),"testData")
	ds = readdlm(testDataFile, ',')
	data = ds[:,1:end-1]'

	# test separation calculations
	M = [297.654, -183.908, -164.718, -339.345, -53.5994, -142.06, -207.939, -180.871, 469.81, 190.212]
	B = [0.0114666 0.20954 -0.371882 0.0527596 0.429496 -0.0887774 0.0872738 0.734395 0.121809 -0.246463]'
	s = LMCLUS.find_separation(data, M, B, p)
	@test typeof(s) == LMCLUS.Separation
	@test criteria(s) ≈ 7.285358462818518
	@test threshold(s) ≈ 876.2634381305518

	# run clustering
	res = lmclus(data, p)
	@test nclusters(res) >= 3
	@test sum(counts(res)) == size(ds, 1)

	cnts = counts(res)
	@testset "Label Match" for idxs in combinations(1:nclusters(res),2)
		i = idxs[1]
		j = idxs[2]
		@test length(symdiff(points(manifold(res,i)), points(manifold(res, j)))) == cnts[i] + cnts[j]
	end

	# Sampling
	p.histogram_sampling = true
	p.number_of_clusters = 3
	res = lmclus(data, p)
	@test nclusters(res) >= 3
	p.histogram_sampling = false

	# RNG seed
	p.random_seed = 0
	res = lmclus(data, p)
	@test nclusters(res) >= 3

	# MDL
	p.mdl_algo = LMCLUS.MDL.SizeIndependent
	p.mdl = true
	res = lmclus(data, p)
	@test nclusters(res) >= 3
	p.mdl_algo = LMCLUS.MDL.OptimalQuant
	p.mdl = false

	# adjust cluster bases
	p.basis_alignment = true
	res = lmclus(data, p)
	@test nclusters(res) >= 3
	@test threshold(manifold(res,1))[1] > 0.0
	@test threshold(manifold(res,1))[2] == Inf
	p.basis_alignment = false

	# calculate cluster bounsds
	p.basis_alignment = true
	p.bounded_cluster = true
	res = lmclus(data, p)
	@test nclusters(res) >= 3
	@test threshold(manifold(res,1))[1] > 0.0
	@test threshold(manifold(res,1))[2] > 0.0
	p.bounded_cluster = false
	p.basis_alignment = false

	# Otsu
	p.sep_algo = LMCLUS.Otsu
	p.min_cluster_size = 200
	p.basis_alignment = true
	res = lmclus(data, p)
	@test nclusters(res) >= 3
	p.sep_algo = LMCLUS.Kittler
	p.min_cluster_size = 20

	# Iterative refinement
	dfun = (X,m)  -> distance_to_manifold(X, mean(m), projection(m))
	efun = (X,ms) -> LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, ms, X, 32, 16)
	res2 = refine(res, data, dfun, efun, bounds=true)
	@test nclusters(res) >= nclusters(res2)
	@test threshold(manifold(res2,1))[1] > 0.0
	@test threshold(manifold(res2,1))[2] > 0.0
	# no changes to original clustering
	@testset for (m1, m2) in zip(manifolds(res), manifolds(res))
		@test all(points(m1) .== points(m2))
	end

end
