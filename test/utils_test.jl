using LMCLUS
using LinearAlgebra
using Test
import Random

@testset "Utils" begin

	Random.seed!(123894)
    S = hcat(fill(1,2,2), fill(2,2,2))
    @test LMCLUS.sample_points(S, 2) == [4, 1]
    @test LMCLUS.sample_points(S, 3) == Int[]
	@test_throws ErrorException LMCLUS.sample_points(S, -3)

    # Test data sampling
	@test size(unique(LMCLUS.randperm2(10,5)),1) == 5
	@test size(LMCLUS.sample_points(Matrix(I, 5, 3),3),1) == 3
	@test_throws ErrorException size(LMCLUS.sample_points(Matrix(I, 5, 3),5),1) == 4

	# Test basis forming
	data = float(reshape([1,1,0, 1,2,2, -1,0,2, 0,0,1],3,4))
	basis = data[:, 2:end]
	result = float(reshape([1,2,2,-2,-1,2, 2,-2,1],3,3))/3.0
	@test LMCLUS.orthogonalize(basis) ≈ result
	data = float(reshape([1,1,0, 2,3,2, 0,1,2, 1,1,1],3,4))
	@test LMCLUS.form_basis(data, collect(1:4))[2] ≈ result

end
