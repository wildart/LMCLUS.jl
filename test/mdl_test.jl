using LinearAlgebra
using LMCLUS
using Statistics
using Distributions
using Test
import Random

@testset "MDL" begin

    @testset "Quantization" begin
        @test LMCLUS.MDL.univar([1.0]) ≈ [1 ./ 12.0]
        @test LMCLUS.MDL.optbins([1.], 1.) == [one(UInt64)]
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

    @testset "Calculations" begin
        # Generate manifold data
        function generate_lm(N::Int, M::Int, C::Int,
                        B::Matrix{Float64},
                        bounds::Matrix{Float64}, θ::Float64,
                        D::Symbol = :Uniform;  σs::Vector{Float64} = ones(N))
            @assert size(bounds) == (N,2) "Define bounds for every dimension"
            @assert size(B) == (N,M) "Define bounds for every dimension"
            manifold = Manifold(M, zeros(N), B, round.(Int, range(1, stop=C, length=C)), θ, 0.0)

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
                if distance_to_manifold(X[:,c], B)[1] < θ
                    c += 1
                end
            end

            return X, manifold
        end

        Pm = 64      # Precision encoding constant for model
        Pd = 32      # Precision encoding constant for data
        N = 2        # Space dimension
        M = 1        # Linear manifold dimension
        C = 100      # Size of a LM cluster
        B = Matrix(I,N,M) # Basis vectors
        bounds = hcat(fill(-1.,N), fill(1., N)) # LM cluster bounds
        θ = 0.8      # distance threshold
        σs = [1.0, 0.25] # diag covariances
        ɛ = 1e-2
        tot = 1000
        tol = 1e-8

        Random.seed!(923487298)
        B *= rand()

        Random.seed!(923487298)
        Xg, Mg = generate_lm(N, M, C, B, bounds, θ, :Gausian; σs = σs)
        LMCLUS.adjustbasis!(Mg, Xg)
        @test LMCLUS.hasfullbasis(Mg)

        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Raw, Mg, Xg, Pm, Pd) == Pm*N*C
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.ZeroDim, Mg, Xg, Pm, Pd) == 221

        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Uniform, Mg, Xg, Pm, Pd)  == 5623
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Gaussian, Mg, Xg, Pm, Pd) == 1739

        # Empirical entropy from optimal quantization
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Empirical, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 3889
        # Empirical entropy from fixed bin size quantization
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.Empirical, Mg, Xg, Pm, Pd, ɛ = 20.0) == 3808

        # Optimal quantizing
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 4093
        @test LMCLUS.mdl(Mg, Xg, Pm, Pd, ɛ = 1e-2) == 4093

        Mg.d = 0
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.SizeIndependent, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 9152
        Mg.d = 1
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.SizeIndependent, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 3008

        # Test size dependence
        Random.seed!(923487298)
        Xg, Mg = generate_lm(N, M, 10*C, B, bounds, θ, :Gausian; σs = σs)
        LMCLUS.adjustbasis!(Mg, Xg)
        @test LMCLUS.hasfullbasis(Mg)
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 39306
        @test LMCLUS.MDL.calculate(LMCLUS.MDL.SizeIndependent, Mg, Xg, Pm, Pd, ɛ = 1e-2) == 3136

    end

end
