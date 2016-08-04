module TestMDL
    using LMCLUS
    using Distributions
    using Base.Test

    @testset "Quantization" begin
        @test_approx_eq LMCLUS.MDL.univar([1.]) [1./12.]
        @test_approx_eq LMCLUS.MDL.optbins([1.], 1.) 1.
        bins, ɛ, c, itr = LMCLUS.MDL.optquant([1.], 1e-2)
        @test bins[1] == 29
        @test ɛ < 1e-2
        bins, ɛ, c, itr = LMCLUS.MDL.optquant([Inf], 1e-2)
        @test bins[1] == 29
        @test ɛ < 1e-2
        bins, ɛ, c, itr = LMCLUS.MDL.optquant([NaN], 1e-2)
        @test bins[1] == 1
        @test ɛ < 1e-2
    end

    @testset "MDL calculations" begin
        # Generate manifold data
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

        srand(923487298)

        Pm = 64      # Precision encoding constant for model
        Pd = 32      # Precision encoding constant for data
        N = 2        # Space dimension
        M = 1        # Linear manifold dimension
        C = 100      # Size of a LM cluster
        B = eye(N,M) # Basis vectors
        bounds = [-1. 1.; -1. 1.] # LM cluster bounds
        θ = 0.8      # distance threshold
        σs = [1.0, 0.25] # diag covariances
        ɛ = 1e-2

        B *= rand()
        Xg, Mg = generate_lm(N, M, C, B, bounds, θ, :Gausian; σs = σs)

        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Raw, Mg, Xg, Pm, Pd) == Pm*N*C
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.ZeroDim, Mg, Xg, Pm, Pd) == 222

        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Uniform, Mg, Xg, Pm, Pd)  == 5623
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Gaussian, Mg, Xg, Pm, Pd) == 3741

        # Empirical entropy from optimal quantization
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Empirical, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 3894
        # Empirical entropy from fixed bin size quantization
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Empirical, Mg, Xg, Pm, Pd, ɛ = 20.0) == 3812
        # Optimal quantizing
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 4097

        Mg.d = 0
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.SizeIndependent, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 9344
        Mg.d = 1
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.SizeIndependent, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 3136

        srand(923487298)
        Xg, Mg = generate_lm(N, M, 10*C, B, bounds, θ, :Gausian; σs = σs)
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.SizeIndependent, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 3200
        # Optimal quantizing
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 39306
    end
end
