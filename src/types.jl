"""Abstract histogram thresholding algorithm type"""
abstract type Thresholding end

## histogram separation type
"Cluster separation parameters"
struct Separation{T<:AbstractFloat}
    "Separation depth (depth between separated histogram modes)"
    depth::T
    "Separation discriminability (width between separated histogram modes)"
    discriminability::T
    "Distance threshold value"
    threshold::T
    "Global minimum as histogram bin index"
    globalmin::Int
    "Minimal distance"
    mindist::T
    "Maximal distance"
    maxdist::T
    "Number of bins in the histogram"
    bins::Int
end
Separation() = Separation{Float64}(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0)

# properties
"Returns separation criteria value which is product of depth and discriminability."
criteria(sep::Separation) = sep.discriminability*sep.depth
"Returns distance threshold value for separation calculated on histogram of distances. It is used to determine which points belong to formed cluster."
threshold(sep::Separation) = sep.threshold
Base.extrema(sep::Separation) = (sep.mindist, sep.maxdist)

function Base.show(io::IO, S::Separation)
    print(io, "Separation($(criteria(S)), θ=$(threshold(S)))")
end

"Linear manifold cluster"
mutable struct Manifold{T<:AbstractFloat}
    "Dimension of the manifold"
    d::Int
    "Translation vector that contains coordinates of the linear subspace origin"
    μ::Vector{T}
    "Matrix of basis vectors that span the linear manifold"
    basis::Matrix{T}
    "Indexes of points associated with this cluster"
    points::Vector{Int}
    "Orthogonal subspace distance threshold"
    θ::T
    "Linear manifold subspace distance threshold"
    σ::T
end
Manifold(d::Int, μ::Vector{T}, basis::Matrix{T}, pnts::Vector{Int}) where T<:AbstractFloat =
    Manifold(d, μ, basis, pnts, zero(T), zero(T))
Manifold{T}(d::Int, pnts::Vector{Int}) where T<:AbstractFloat = Manifold(d, zeros(T,d), zeros(T,d,d), pnts)
Manifold{T}(d::Int) where T<:AbstractFloat = Manifold(d, Int[])
Manifold() = Manifold{Float64}(0)

Base.copy(M::Manifold) = Manifold(outdim(M),mean(M),projection(M),points(M),threshold(M)...)

# properties
"Returns the dimension of the linear manifold cluster."
outdim(M::Manifold) = M.d
"Returns the dimension of the observation space."
indim(M::Manifold) = length(M.μ)
"Return an array of cluster assignments."
points(M::Manifold) = M.points
"Returns the cluster thresholds."
threshold(M::Manifold) = (M.θ, M.σ)
"Returns the matrix with columns corresponding to orthonormal vectors that span the linear manifold."
projection(M::Manifold) = M.basis[:,1:M.d]
"Returns the translation vector `μ` which contains coordinates of the linear manifold origin."
mean(M::Manifold) = M.μ
"Returns the cluster size - a number of points assosiated with the cluster."
Base.size(M::Manifold) = length(points(M))
"""Checks if the manifold `M` contains a full basis."""
hasfullbasis(M::Manifold) = size(M.basis,1) == size(M.basis,2)

function Base.show(io::IO, M::Manifold)
    print(io, "Manifold (dim = $(outdim(M)), size = $(size(M)), (θ,σ)=$(threshold(M)))")
end
function Base.dump(io::IO, M::Manifold)
    show(io, M)
    println(io)
    println(io, "thresholds (θ,σ): $(threshold(M)) ")
    println(io, "translation (μ): ")
    Base.showarray(io, mean(M)', header=false, repr=false)
    println(io)
    println(io, "basis: ")
    Base.showarray(io, projection(M), header=false, repr=false)
end

"""Algorithm specific exception"""
struct LMCLUSException <: Exception
    msg::String
end
