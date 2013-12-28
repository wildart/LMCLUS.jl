using LMCLUS
using Base.Test

# Initialize parameters
p = LMCLUSParameters(3)

# Test sampling parameters
p.cluster_number = 1
@test LMCLUS.sample_quantity(1,3,1000,p) == 1
p.cluster_number = 2
p.heuristic = 1
@test LMCLUS.sample_quantity(1,3,1000,p) == 13
p.heuristic = 2
@test LMCLUS.sample_quantity(1,3,1000,p) == 10
p.heuristic = 3
@test LMCLUS.sample_quantity(1,3,1000,p) == 10

# Test data sampling
@test size(unique(LMCLUS.randperm2(10,5)),1) == 5
@test size(LMCLUS.sample_points(eye(5, 3),3),1) == 3
@test_throws size(LMCLUS.sample_points(eye(5, 3),5),1) == 4

# Test basis forming
data = float(reshape([1,1,0, 1,2,2, -1,0,2, 0,0,1],3,4)')
basis = data[2:,:]
result = float(reshape([1,2,2,-2,-1,2, 2,-2,1],3,3)')/3.0
@test_approx_eq LMCLUS.orthogonalize(basis) result
data = float(reshape([1,1,0, 2,3,2, 0,1,2, 1,1,1],3,4)')
@test_approx_eq LMCLUS.form_basis(data)[2] result

# Test distance calculation
basis = float(reshape([1,0,0, 0,1,0],3,2)')
point = vec([1.0,1.0,1.0])
@test_approx_eq distance_to_manifold(point, basis) 1.0

# Test separation
#data = rand(1000,6)
#p = LMCLUSParameters(5)
#LMCLUS.find_best_separation(data, params, 1)

# Main test
p = LMCLUSParameters(5)
ds = readdlm("test/testData", ',')
manifolds = lmclus(ds[:,1:end-1],p)
@test length(manifolds) == 3

