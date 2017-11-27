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
