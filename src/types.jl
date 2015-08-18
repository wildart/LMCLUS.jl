## histogram separation type
type Separation
    depth::Float64
    discriminability::Float64
    threshold::Float64
    globalmin::Int
    hist_range::Vector{Float64}
    hist_count::Vector{UInt32}
    bin_index::Vector{UInt32}
end
Separation() = Separation(-Inf, eps(), Inf, -1, Float64[], UInt32[], UInt32[])

# properties
criteria(sep::Separation) = sep.discriminability*sep.depth
threshold(sep::Separation) = sep.threshold

function Base.show(io::IO, S::Separation)
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
projection(M::Manifold) = M.proj
Base.mean(M::Manifold) = M.μ
Base.copy(M::Manifold) = Manifold(indim(M),mean(M),projection(M),labels(M),separation(M))

function Base.show(io::IO, M::Manifold)
    print(io, "Manifold (dim = $(indim(M)), size = $(outdim(M)))")
end
function Base.dump(io::IO, M::Manifold)
    show(io, M)
    println(io)
    println(io, "threshold (θ): $(threshold(separation(M))) ")
    println(io, "translation (μ): ")
    Base.showarray(io, mean(M)', header=false, repr=false)
    println(io)
    println(io, "basis: ")
    Base.showarray(io, projection(M), header=false, repr=false)
end

function assignments(Ms::Vector{Manifold})
    lbls = zeros(Int, sum(map(m->outdim(m), Ms)))
    for (i,m) in enumerate(Ms)
        lbls[labels(m)] = i
    end
    return lbls
end

# function save(io::IO, m::Manifold)
#     serialize(io, m.d)
#     serialize(io, m.μ)
#     serialize(io, m.proj)
#     serialize(io, m.points)
#     serialize(io, m.separation)
# end

# function load(io::IO)
#     d = deserialize(io)
#     μ = deserialize(io)
#     proj = deserialize(io)
#     points = deserialize(io)
#     separation = deserialize(io)
#     return Manifold(d, μ, proj, points, separation)
# end