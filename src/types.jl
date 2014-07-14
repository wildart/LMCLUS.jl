import Base.show

type Separation
    origin::Vector{Float64}
    basis::Matrix{Float64}
    depth::Float64
    discriminability::Float64
    threshold::Float64
    globalmin::Int
    hist_range::Vector{Float64}
    hist_count::Vector{Int}
end
Separation() = Separation(Float64[], Array(Float64,0,0), -Inf, eps(), Inf, -1, Float64[], Int[])
separation_criteria(sep::Separation) = sep.depth*sep.discriminability

type Manifold
    dimension::Int
    points::Vector{Int}
    separation::Separation
end
show(io::IO, m::Manifold) =
    print(io, """Manifold:
        Dimension: $(m.dimension)
        Size: $(length(m.points))
        θ: $(m.separation.threshold)
        μ: $(m.separation.origin')
    """)