module TestLMCLUS
	using LMCLUS
	using Base.Test

	# Initialize parameters
	p = LMCLUSParameters(3)

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

	# Test data sampling
	@test size(unique(LMCLUS.randperm2(10,5)),1) == 5
	@test size(LMCLUS.sample_points(eye(5, 3),3),1) == 3
	@test_throws ErrorException size(LMCLUS.sample_points(eye(5, 3),5),1) == 4

	# Test basis forming
	data = float(reshape([1,1,0, 1,2,2, -1,0,2, 0,0,1],3,4))
	basis = data[:, 2:end]
	result = float(reshape([1,2,2,-2,-1,2, 2,-2,1],3,3))/3.0
	@test_approx_eq LMCLUS.orthogonalize(basis) result
	data = float(reshape([1,1,0, 2,3,2, 0,1,2, 1,1,1],3,4))
	@test_approx_eq LMCLUS.form_basis(data)[2] result

	# Test distance calculation
	basis = reshape([1.,0.,0.,0.,1.,0.],3,2)
	point = [1.0,1.0,1.0]
	@test_approx_eq distance_to_manifold(point, basis) 1.0

	# Test parameters
	l = 1000
	x = rand(l)
	p = LMCLUSParameters(5)
	@test LMCLUS.hist_bin_size(x, p) == round(Int, l*p.max_bin_portion)
	p.hist_bin_size = 20
	@test LMCLUS.hist_bin_size(x, p) == p.hist_bin_size

	# Test clustering
	p = LMCLUSParameters(5)
	p.basis_alignment = true
	p.log_level = 2
	p.dim_adjustment = true
	p.dim_adjustment_ratio = 0.95
	p.random_seed = 4572489057
	println(p) # test show()

	testDataFile = joinpath(splitdir(string(lmclus.env.defs.func.code.file))[1],"..","test","testData")
	ds = readdlm(testDataFile, ',')
	data = ds[:,1:end-1]'

	# test separation calculations
	s = LMCLUS.calculate_separation(data, [1,2,3], p)
	@test typeof(s[1]) == Separation
	@test typeof(s[2]) == Vector{Float64}
	@test typeof(s[3]) == Matrix{Float64}

	# run clustering
	manifolds = lmclus(data,p)
	@test length(manifolds) >= 3
	@test sum(map(m->length(m.points), manifolds)) == size(ds, 1)
	print("Comparing indexes of manifolds: ")
	for (i,j) in combinations(1:length(manifolds),2)
		print("($(i), $(j))")
		@test length(symdiff(labels(manifolds[i]), labels(manifolds[j]))) ==
				length(labels(manifolds[i])) + length(labels(manifolds[j]))
	end
	println()

	# Sampling
	p.histogram_sampling = true
	p.number_of_clusters = 3
	manifolds = lmclus(data,p)
	@test length(manifolds) >= 3
	p.histogram_sampling = false

	# MDL
	p.mdl = true
	manifolds = lmclus(data,p)
	@test length(manifolds) >= 3
	p.mdl = false

	# RNG seed
	p.random_seed = 0
	manifolds = lmclus(data,p)
	@test length(manifolds) >= 3
end
