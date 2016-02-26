module TestMDL
    using LMCLUS
    using Distributions
    using Base.Test

    function generate_lm(N::Int, M::Int, C::Int,
                    B::Matrix{Float64},
                    bounds::Matrix{Float64}, θ::Float64,
                    D::Symbol = :Uniform;  σs::Vector{Float64} = ones(N))
        @assert size(bounds) == (N,2) "Define bounds for every dimension"
        @assert size(B) == (N,M) "Define bounds for every dimension"
        S = Separation()
        S.threshold = θ
        manifold = Manifold(M, zeros(N), B, round(Int, linspace(1, C, C)), S)

        c = 1
        X = zeros(N,C)
        while c <= C
            if D == :Gaussian
                for i in 1:N
                    R = abs(bounds[i,2]-bounds[i,1])
                    X[i,c] = rand(Normal(0.,σs[i]))
                end
            else
                for i in 1:N
                    R = abs(bounds[i,2]-bounds[i,1])
                    X[i,c] = rand()*R - R/2.
                end
            end
            if distance_to_manifold(X[:,c], B) < θ
                c += 1
            end
        end

        return X, manifold
    end

    Pm = 32      # Precision encoding constant for model
    Pd = 16      # Precision encoding constant for data
    N = 2        # Space dimension
    M = 1        # Linear manifold dimension
    C = 100      # Size of a LM cluster
    B = eye(N,M) # Basis vectors
    bounds = [-1. 1.; -1. 1.] # LM cluster bounds
    θ = 0.8      # distance threshold
    σs = [1.0, 0.25] # diag covariances

    srand(923487298)
    Xg, Mg = generate_lm(N, M, C, B, bounds, θ, :Gausian; σs = σs)
    @test LMCLUS.mdl(Mg, Xg, Pm = Pm, Pd = Pd, dist=:Uniform)  == 1741
    @test LMCLUS.mdl(Mg, Xg, Pm = Pm, Pd = Pd, dist=:Gaussian) == 1838
    @test LMCLUS.mdl(Mg, Xg, Pm = Pm, Pd = Pd, dist=:Empirical, ɛ = 1e-2) == 2162 # quantization
    @test LMCLUS.mdl(Mg, Xg, Pm = Pm, Pd = Pd, dist=:Empirical, ɛ = 20.0) == 2116 # bin # fixed
    @test LMCLUS.mdl(Mg, Xg, Pm = Pm, Pd = Pd, dist=:OptimalQuant, ɛ = 1e-2) == 2324  # optimal quantizing
    Mg.d = 0
    @test LMCLUS.mdl(Mg, Xg, Pm = Pm, Pd = Pd, dist=:None)   == 3264
    @test LMCLUS.mdl(Mg, Xg, Pm = Pm, Pd = Pd, dist=:Center) == 170

    # Quantization
    @test_approx_eq LMCLUS.univar([1.]) [1./12.]
    @test_approx_eq LMCLUS.opt_bins([1.], 1.) 1.
    bins, ɛ, c, itr = LMCLUS.opt_quant([1.], 1e-2)
    @test bins[1] == 29
    @test ɛ < 1e-2
    bins, ɛ, c, itr = LMCLUS.opt_quant([Inf], 1e-2)
    @test bins[1] == 29
    @test ɛ < 1e-2
    bins, ɛ, c, itr = LMCLUS.opt_quant([NaN], 1e-2)
    @test bins[1] == 1
    @test ɛ < 1e-2
end
