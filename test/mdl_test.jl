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
        manifold = Manifold(M, zeros(N), B, int(linspace(1, C, C)), S)

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

    P = 32       # Precision encoding constant
    N = 2        # Space dimension
    M = 1        # Linear manifold dimension
    C = 100      # Size of a LM cluster
    B = eye(N,M) # Basis vectors
    bounds = [-1. 1.; -1. 1.] # LM cluster bounds
    θ = 0.8      # distance threshold
    σs = [1.0, 0.25] # diag covariances

    srand(923487298)
    Xg, Mg = generate_lm(N, M, C, B, bounds, θ, :Gausian; σs = σs)
    @test_approx_eq_eps LMCLUS.MDLength(Mg, Xg, P = 32., T=:Uniform)   3296 1
    @test_approx_eq_eps LMCLUS.MDLength(Mg, Xg, P = 32., T=:Gausian)   9632 1
    @test_approx_eq_eps LMCLUS.MDLength(Mg, Xg, P = 32., T=:Empirical) 4124 1
    Mg.d = 0
    @test_approx_eq     LMCLUS.MDLength(Mg, Xg, P = 32., T=:None)      6400
    @test_approx_eq_eps LMCLUS.MDLength(Mg, Xg, P = 32., T=:Center)    169  1
end