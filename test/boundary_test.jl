using LMCLUS
using MultivariateStats
using Distances
using Base.Test

@testset "Boundary" begin

    srand(29348039483948)
    Csize = 200
    d = 2
    X = hcat(randn(d, Csize), randn(d, Csize).-5., 3.*randn(d, Csize).+10.)
    L = vcat(fill(1,Csize), fill(2,Csize), fill(3,Csize))

    # Construct manifold cluster
    P = fit(MultivariateStats.PCA, X[:,1:Csize], pratio = 0.5)
    θ = distance_to_manifold(X[:,1:Csize], mean(P), projection(P)) |> maximum
    SO = LMCLUS.Separation(0.0, 0.0, θ, 0, Int[])
    M = Manifold(1, mean(P), projection(P), collect(1:Csize), SO)

    # test quantile boundary construction
    q = 0.05
    B = fit(LMCLUS.QuantileInteriorBoundary, q, M, X)
    @test B.q == q
    @test length(LMCLUS.index(B)) == Csize*q
    @test all(i ∈ labels(M) for i in LMCLUS.index(B)) # all point are in first clusters

    # test separation relation boundary construction with distance matrix
    D = pairwise(Euclidean(), X)
    S(D,A,a) = any(D[a,A] .<= 3.3) # distance to cluster compliment less the 3.3
    B = fit(LMCLUS.SeparationInteriorBoundary, S, M, D)
    @test length(LMCLUS.index(B)) == 15
    @test all(i ∈ labels(M) for i in LMCLUS.index(B)) # all point are in first clusters

    # test separation relation boundary construction with data
    S(D,A,a) = any(colwise(Euclidean(), X[:,A], X[:,a]) .<= 3.3) # distance to cluster compliment less the 3.3
    B = fit(LMCLUS.SeparationInteriorBoundary, S, M, X)
    @test length(LMCLUS.index(B)) == 15
    @test all(i ∈ labels(M) for i in LMCLUS.index(B)) # all point are in first clusters
    @test B.SR === S

end
