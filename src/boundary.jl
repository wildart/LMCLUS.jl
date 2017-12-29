"""Abstract type for the cluster boundary"""
abstract type AbstractBoundary end
"""Get element index of the boundary"""
index(b::AbstractBoundary) = b.index

struct QuantileInteriorBoundary <: AbstractBoundary
    index::IntSet
    q::Float64
end

"""Construction of the interior boundary as quantile of the LM cluster"""
function fit(::Type{QuantileInteriorBoundary}, q::Float64, M::Manifold, D::Vector{T}) where T <: Real
    L = labels(M)
    csize = length(L)

    # calculate internal boundary quantile (w.r.t. cluster size)
    Iᵦ = IntSet()
    if q > 0.0
        cps = sortperm(D[L], rev=true)[1:ceil(Int, q*csize)]
        union!(Iᵦ, L[cps])
    end

    return QuantileInteriorBoundary(Iᵦ, q)
end

function fit(::Type{QuantileInteriorBoundary}, q::Float64, M::Manifold, X::Matrix{T}) where T <: Real
    return fit(QuantileInteriorBoundary, q, M, distance_to_manifold(X, mean(M), projection(M)))
end


struct SeparationInteriorBoundary <: AbstractBoundary
    index::IntSet
    SR::Function   # S(X::Matrix{T}, Vector{Int}, Int) -> bool
end

"""Construction of the interior boundary of the LM cluster from separation relation

The interior boundary of A, denoted is the subset of all members of A connected to Aᶜ: Iᵦ(A) = {a ∈ A | ({a}, Aᶜ) ∉ S}
"""
function fit(::Type{SeparationInteriorBoundary}, SR::Function, M::Manifold, X::AbstractMatrix{T}) where T <: Real
    L = labels(M)
    # construct complement
    Aᶜ = symdiff(collect(1:size(X,2)), L)
    # calculate internal boundary from separation relation
    Iᵦ = IntSet(i for i in L if SR(X,Aᶜ,i))
    return SeparationInteriorBoundary(Iᵦ, SR)
end

# struct ExteriorBoundary <: AbstractBoundary end
