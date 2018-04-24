#### Interface to Clustering package

struct LMCLUSResult <: ClusteringResult
    manifolds::Vector{Manifold}
    separations::Vector{Separation}
end

nclusters(R::LMCLUSResult) = length(R.manifolds)

counts(R::LMCLUSResult) = map(outdim, R.manifolds)

assignments(R::LMCLUSResult) = assignments(R.manifolds)

"""Get linear manifold cluster"""
manifold(R::LMCLUSResult, idx::Int) = R.manifolds[idx]
manifolds(R::LMCLUSResult) = R.manifolds

"""Iterative refinement of LMCLUS clustering

    refine(res::LMCLUSResult, data::AbstractMatrix, dfun::Function, efun::Function; tol::Real = 10.0, maxiter::Integer = 100)

Updates assignments to linear manifold clusters `res` on dataset `data` by evaluating similarity from dataset points to each cluster by

    dfun::(AbstractMatrix, Manifold) -> Vector{<:Real}

with following evaluation of the refined clustering by

    efun::(AbstractMatrix, Manifold) -> Real

Function returns refined `LMCLUSResult` object, after algorithm converges with tolerance, `tol`, or after specified number of iterations, `maxiter`.
"""
function refine(res::LMCLUSResult, data::AbstractMatrix,
                dfun::Function, efun::Function; bounds = false,
                tol::Real = 10.0, maxiter::Integer = 100, debug = false,
                drop_last = true, min_cluster_size = 20)
    M = manifolds(res)[1:(drop_last ? end-1 : end)]
    Δ = efun(data, M)

    # main loop
    c = 0
    converged = false
    while !converged && c < maxiter
        c += 1

        D = map(m->outdim(m) > 0 ? dfun(data, m) : fill(Inf, size(data,2)), M)
        DM = hcat(D...)
        A = mapslices(d->last(findmin(d)), DM, 2)

        # update assignments
        M⁺ = Manifold[]
        for (i,m) in enumerate(M)
            I = find(A .== i)
            if length(I) < min_cluster_size #TODO: reassign points before modifing cluster parameters
                for ii in I
                    A[ii] = sortperm(DM[ii,:])[2]
                end
                continue # reassign cluster points, effectivly removing it
            end
            if length(I) > 0
                X = data[:, I]
                R = fit(PCA, X; maxoutdim = indim(m))
                μ = mean(R)
                B = projection(R)
                θ = σ = 0.0
                if bounds
                    θ = maximum(distance_to_manifold(X, μ, B))
                    σ = maximum(distance_to_manifold(X, μ, B, ocss=true))
                end
                push!(M⁺, Manifold(indim(m), μ, B, I, θ, σ))
            end
        end
        filter!(m->outdim(m) != 0, M⁺)

        # evaluate new clustering
        Δ⁺ = sum(efun(data, m) for m in M⁺)
        debug && println("$c: Δ = ($Δ - $Δ⁺) = $(Δ - Δ⁺)")
        M = M⁺

        # check convergence
        converged = abs(Δ - Δ⁺) < tol
        Δ = Δ⁺
    end

    return LMCLUSResult(M, LMCLUS.Separation[])
end

"""Reassign outliers to the closes cluster in the clustering."""
function clearoutliers(res::LMCLUSResult, data::AbstractMatrix,
                       dfun::Function; debug = false)
    M = manifolds(res)
    indim(M[end]) != 0 && return res
    O = pop!(M)

    # evaluate distances to outliers
    D = map(m->dfun(data, m), M)
    OI = labels(O)
    DM = hcat(D...)[OI, :]

    # reassing outliers to available clusters
    A = mapslices(d->last(findmin(d)), DM, 2)
    for (i,a) in enumerate(A)
        push!(labels(M[a]), OI[i])
    end

    # refine parameters of manifolds
    for m in M
        X = data[:, labels(m)]
        R = fit(PCA, X; maxoutdim = indim(m))
        m.μ = mean(R)
        m.proj = projection(R)
        m.θ = 0.0
        m.σ = 0.0
    end

    return LMCLUSResult(M, LMCLUS.Separation[])
end
