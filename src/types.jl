## histogram separation type
"Cluster separation parameters"
mutable struct Separation
    "Separation depth (depth between separated histogram modes)"
    depth::Float64
    "Separation discriminability (width between separated histogram modes)"
    discriminability::Float64
    "Distance threshold value"
    threshold::Float64
    "Global minimum as histogram bin index"
    globalmin::Int
    "Minimal distance"
    mindist::Float64
    "Maximal distance"
    maxdist::Float64
    "Number of bins in the histogram"
    bins::Int
end
# Separation() = Separation(-Inf, eps(), Inf, -1, Float64[])
Separation() = Separation(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0)

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
mutable struct Manifold
    "Dimension of the manifold"
    d::Int
    "Translation vector that contains coordinates of the linear subspace origin"
    μ::Vector{Float64}
    "Matrix of basis vectors that span the linear manifold"
    proj::Matrix{Float64}
    "Indexes of points associated with this cluster"
    points::Vector{Int}
    "Orthogonal subspace distance threshold"
    θ::Float64
    "Linear manifold subspace distance threshold"
    σ::Float64
end
Manifold(d::Int, μ::Vector{Float64}, proj::Matrix{Float64}, pnts::Vector{Int}) =
    Manifold(d, μ, proj, pnts, 0.0, 0.0)
Manifold() = Manifold(0, zeros(0), zeros(0,0), Int[])

# properties
"Returns a dimension of the linear manifold cluster."
indim(M::Manifold) = M.d
"Returns the number of points in the cluster."
outdim(M::Manifold) = length(M.points)
"Return an array of cluster assignments."
labels(M::Manifold) = M.points
"Returns the cluster thresholds."
threshold(M::Manifold) = (M.θ, M.σ)
"Returns the matrix with columns corresponding to orthonormal vectors that span the linear manifold."
projection(M::Manifold) = M.proj
"Returns the translation vector `μ` which contains coordinates of the linear manifold origin."
Base.mean(M::Manifold) = M.μ
Base.copy(M::Manifold) = Manifold(indim(M),mean(M),projection(M),labels(M),threshold(M)...)

function Base.show(io::IO, M::Manifold)
    print(io, "Manifold (dim = $(indim(M)), size = $(outdim(M)))")
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
