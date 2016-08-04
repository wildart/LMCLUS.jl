module MDL

import ..LMCLUS: Manifold, indim, outdim, mean, projection, separation, histogram

# Various types for MDL calculation
abstract MethodType

type Raw <: MethodType end
type ZeroDim <: MethodType end

type Uniform <: MethodType end
type Gaussian <: MethodType end
type Empirical <: MethodType end
type OptimalQuant <: MethodType end
type SizeIndependent <: MethodType end

"""Optimal number of bins given constant C over intervals"""
function optbins{T<:AbstractFloat}(intervals::Vector{T}, C::T)
    ns = ceil(intervals * exp( (C - sum(intervals))/length(intervals) ), 0 )
    nsi = zeros(UInt, size(ns))
    for i in 1:length(nsi)
        nsi[i] = if ns[i] >= typemax(UInt)
            typemax(UInt)
        else
            round(UInt, ns[i])
        end
    end
    return nsi
end

"""Variance of a uniform distribution over interval"""
univar{T<:AbstractFloat}(interval::Vector{T}) = (1//12)*(interval).^2

"""Quantizaton error"""
quanterror(intervals, N_k_opt) = sum(univar(intervals./N_k_opt))

"""Optimal quantization of the interval"""
function optquant{T<:AbstractFloat}(intervals::Vector{T}, ɛ::T; tot::Int = 1000, tol=1e-6)
    intervals[isnan(intervals)] = eps() # remove nans
    intervals[intervals .< eps()] = eps() # remove 0s
    intervals[isinf(intervals)] = 1. # remove inf

    # Setup C bounds
    K = length(intervals)
    C = 0.
    Cmin = 0.
    Cmax = K*log(typemax(UInt)./intervals)+sum(log(intervals)) |> minimum

    i = 1
    N_k_opt = ones(UInt, K)
    err_opt = Inf
    while i < tot
        C = Cmax - (Cmax - Cmin)/2.
        N_k_opt = optbins(intervals, C)
        err_opt = quanterror(intervals, N_k_opt)
        err_diff = err_opt - ɛ^2
        if abs(err_diff) < tol
            break # error is small enough to stop search
        elseif err_diff > 0.
            # error is too big
            Cmin = C
        elseif err_diff < 0.
            # error is too small
            Cmax = C
        end
        i += 1
    end
    return N_k_opt, err_opt, C, i
end

function boundingbox(X, m)
    n, l = size(X)

    # PCA data
    F = svdfact(X'/sqrt(n))

    # Take an orthogonal compliment subspace (OCS) basis
    r = (m+1):min(n,l)
    BC = F[:V][:,r]

    # Project data to OCS
    Y = BC'*X

    # Calculate data spread in OCS
    Ymin = vec(minimum(Y, 2))
    Ymax = vec(maximum(Y, 2))
    intervals = abs(Ymax - Ymin)

    return intervals, Y, Ymin, Ymax
end


#=
    Minimum Description Length Encoding
=#

"""Raw representation of the data"""
modeldl(::Type{Raw}, M::Manifold, X::Matrix, P::Int) = P*length(X)
datadl(::Type{Raw}, aggs...) = 0

function raw(M::Manifold, Pm::Int)
    return Pm*outdim(M)*length(mean(M))
end

"""Zero dimensional model description length: L(ZD)

    From: O. Georgieva, K. Tschumitschew, and F. Klawonn,
    “Cluster validity measures based on the minimum description length principle”
"""
function modeldl(::Type{ZeroDim}, M::Manifold, X::Matrix, P::Int)
    return P*size(X,1)
end

"""Zero dimensional data description length: L(X|ZD)

    Ref: O. Georgieva, K. Tschumitschew, and F. Klawonn, “Cluster validity
    measures based on the minimum description length principle”
"""
function datadl{T<:AbstractFloat}(::Type{ZeroDim}, C::Manifold, X::Matrix{T},
                                  P::Int, ɛ::T, tot::Int, tol::T)
    N = size(X,1)  # space dimension
    μ = mean(C)    # manifold translation
    n = outdim(C)  # size of cluster

    E = 0.0
    for i in 1:n
        for j in 1:N
            E += log2(abs(X[j,i] - μ[j]) + 1)
        end
    end
    return round(Int, E)
end

#=
    Various models of MDL calculation (model MDL)
=#

"""General model description length (v1): L(ML)"""
function modeldl{MT<:MethodType}(::Type{MT}, C::Manifold, X::Matrix, P::Int)
    N = size(X,1)  # space dimension
    M = indim(C)   # manifold dimension
    return P*(N + M*(N - (M+1)>>1))
end

"""General model description length (v2)

    Size independent model description length: L(SI)
"""
function modeldl(::Type{SizeIndependent}, M::Manifold, X::Matrix, P::Int)
    N = size(X,1)  # space dimension
    return P*(N*(N+3)>>1)
end


"""Size independent data description length: L(X|SI)
"""
function datadl{T<:AbstractFloat}(::Type{SizeIndependent}, C::Manifold, X::Matrix{T},
                                  P::Int, ɛ::T, tot::Int, tol::T)
    M = indim(C)   # manifold dimension
    μ = mean(C)    # manifold translation

    intervals, _ = boundingbox(X.-μ, M)
    bins, _ = optquant(intervals, ɛ)

    return convert(Int, P*2*sum(bins))
end

#=
    Various models of MDL calculation (data MDL)
=#

""" Data MDL: Uniform encoding

    Points in the orthogonal complement space of the linear manifold cluster
    are uniformly distributed around the cluster manifold.
"""
function datadl{T<:AbstractFloat}(::Type{Uniform}, C::Manifold, X::Matrix{T},
                                  P::Int, ɛ::T, tot::Int, tol::T)
    N = size(X,1)  # space dimension
    M = indim(C)   # manifold dimension
    n = outdim(C)  # size of cluster

    # Point projected on manifold
    PR = P*M

    # Entropy of point in the orthogonal complement space
    E = -float(n)*log(separation(C).threshold)

    # Number of bits of two parts for every point
    return round(Int, (PR + E)*n)
end


""" Data MDL: Gaussian encoding

    Points in the orthogonal complement space of the linear manifold cluster
    are normally distributed around the cluster manifold.
"""
function datadl{T<:AbstractFloat}(::Type{Gaussian}, C::Manifold, X::Matrix{T},
                                  P::Int, ɛ::T, tot::Int, tol::T)
    N = size(X,1)  # space dimension
    M = indim(C)   # manifold dimension
    n = outdim(C)  # size of cluster
    μ = mean(C)    # manifold translation
    B = projection(C) # manifold basis

    # Point projected on manifold
    PR = P*M

    # project points to orthogonal complement subspace
    OP = (eye(N) - B*B')*(X.-μ)

    # multivariate normal distribution entropy
    Σ = OP*OP'
    Σd = det(Σ)
    E = (M/2)*(1+log(2π)) + (Σd > 0.0 ? log(Σd) : 0.0)/2

    # Number of bits of two parts for every point
    return round(Int, (PR + E)*n)
end


""" Data MDL: Empirical entropy encoding

    Distribution of points in the orthogonal complement space of the linear manifold cluster,
    constructed by optimal quantizing with a fixed length bin, is used to
    calculate entropy, thus number of bits.
"""
function datadl{T<:AbstractFloat}(::Type{Empirical}, C::Manifold, X::Matrix{T},
                                  P::Int, ɛ::T, tot::Int, tol::T)
    N = size(X,1)  # space dimension
    M = indim(C)   # manifold dimension
    n = outdim(C)   # manifold dimension
    μ = mean(C)    # manifold translation

    # Point projected on manifold
    PR = P*M

    # Get orthogonal compliment space bounding box
    intervals, Y, Ymin, Ymax = boundingbox(X.-μ, M)

    # Estimate number of bins required for each competent of OSC
    if ɛ < 1. # using optimal quantization
        bins, _ = optquant(intervals, ɛ, tot=tot, tol=tol)
    else      # constant number of bins per component
        bins = fill(round(Int,ɛ), N-M)
    end

    # Cumulative entropy of data spread in OSC (calculated from quantization)
    E = 0.0
    for i in 1:N-M
        Yb = linspace(Ymin[i], Ymax[i], bins[i]+1)
        h = histogram(vec(Y[i,:]),Yb)[2]
        h /= sum(h)
        E += -sum(map(x-> x > 0. ? x*log2(x) : 0., h))
    end

    # Number of bits of two parts for every point
    return round(Int, (PR + E)*n)
end


""" Data MDL: Optimal quantization encoding

    Distribution of points in the orthogonal complement space of the linear manifold cluster,
    constructed by optimal quantizing with a fixed length bin, is used to
    calculate entropy, thus number of bits.
"""
function datadl{T<:AbstractFloat}(::Type{OptimalQuant}, C::Manifold, X::Matrix{T},
                                  P::Int, ɛ::T, tot::Int, tol::T)
    N = size(X,1)  # space dimension
    M = indim(C)   # manifold dimension
    n = outdim(C)  # size of cluster
    μ = mean(C)    # manifold translation

    # Point projected on manifold
    PR = P*M

    # Get orthogonal compliment space bounding box
    intervals, _ = boundingbox(X.-μ, M)

    # Estimate number of bins required for each competent of OSC
    bins, sqerr, C = optquant(intervals, ɛ, tot=tot, tol=tol)

    # Number of bins required to encode data spread in OSC given optimal quantization
    # C = Σ_k log(N_k)
    # where N_k number of bins in quantized interval of component K of OSC
    E = C/log(2)

    # Number of bits of two parts for every point
    return round(Int, (PR + E)*n)
end


#=
    Main MDL call
=#

"Calculate MDL for the manifold"
function calculate{MT<:MethodType, T<:AbstractFloat}(::Type{MT},
             M::Manifold, X::Matrix{T}, Pm::Int, Pd::Int;
             ɛ::T=1e-2, tot::Int = 1000, tol=1e-8)
    return modeldl(MT, M, X, Pm) + datadl(MT, M, X, Pd, ɛ, tot, tol)
end

"Calculate MDL for the clustering"
function calculate{MT<:MethodType, T<:AbstractFloat}(::Type{MT},
             Ms::Vector{Manifold}, X::Matrix{T}, Pm::Int, Pd::Int;
             ɛ::T=1e-2, tot::Int = 1000, tol=1e-8)
    return sum([calculate(MT,m,X,Pm,Pd,ɛ=ɛ,tot=tot,tol=tol) for m in Ms])
end


end