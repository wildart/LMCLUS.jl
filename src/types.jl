import Base: show, dump, mean, copy, serialize, deserialize
import MultivariateStats: indim, outdim, projection

## histogram separation type
type Separation
    depth::Float64
    discriminability::Float64
    threshold::Float64
    globalmin::Int
    hist_range::Vector{Float64}
    hist_count::Vector{Int}
    bin_index::Vector{Int}
end
Separation() = Separation(-Inf, eps(), Inf, -1, Float64[], Int[], Int[])

# properties
criteria(sep::Separation) = sep.discriminability*sep.depth
threshold(sep::Separation) = sep.threshold

function show(io::IO, S::Separation)
    print(io, "Separation($(criteria(S)), θ=$(threshold(S)))")
end

## manifold type
type Manifold
    d::Int
    μ::Vector{Float64}
    proj::Matrix{Float64}
    points::Vector{Int}
    separation::Separation
end
Manifold() = Manifold(0,Float64[],zeros(0,0),Int[],Separation())

# properties
indim(M::Manifold) = M.d
outdim(M::Manifold) = length(M.points)
labels(M::Manifold) = M.points
separation(M::Manifold) = M.separation
mean(M::Manifold) = M.μ
projection(M::Manifold) = M.proj
copy(M::Manifold) = Manifold(indim(M),mean(M),projection(M),labels(M),separation(M))

function show(io::IO, M::Manifold)
    print(io, "Manifold (dim = $(indim(M)), size = $(outdim(M)))")
end
function dump(io::IO, M::Manifold)
    show(io, M)
    println(io)
    println(io, "threshold (θ): $(threshold(separation(M))) ")
    println(io, "translation (μ): ")
    Base.showarray(io, mean(M)', header=false, repr=false)
    println(io)
    println(io, "basis: ")
    Base.showarray(io, projection(M), header=false, repr=false)
end