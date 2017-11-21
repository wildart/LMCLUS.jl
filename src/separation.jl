using StatsBase

"""Perform the dataset `xs` separation by thresholding a dataset `bins`ed histogram based on the algorith `T`"""
function separation(::Type{T}, xs::Vector{S};
                    bins = 20, tol = 1.0e-5,
                    debug = false) where {T <: Thresholding, S <: Real}
    minX, maxX = extrema(xs)

    # get histogram
    r = linspace(minX, maxX, bins+1)
    H = fit(Histogram, xs, r, closed=:left)
    N = length(H.weights)

    thresh = fit(T, H.weights, length(xs)-1)

    midx = minimum(thresh)
    s = stats(thresh)
    threshold = minX + ( midx * (maxX - minX) / N )
    discriminability = (abs(s[midx,3]-s[midx,4]))/(sqrt(s[midx,5]+s[midx,6]))

    return Separation(depth(thresh), discriminability, threshold, midx)
end

"""Kittler algorithm thresholding type"""
struct Kittler <: Thresholding
    statistics::Matrix{Float64}
    depth::Float64
    minindex::Int
end
depth(t::Kittler) = t.depth
stats(t::Kittler) = t.statistics
Base.minimum(t::Kittler) = t.minindex

"""Performs Kittler's minimal thresholding algorithm

J. Kittler & J. Illingworth: "Minimum Error Thresholding", Pattern Recognition, Vol 19, nr 1. 1986, pp. 41-47.
"""
function fit(::Type{Kittler}, H::Vector{Int}, n::Int; tol=1.0e-5)
    N = length(H)

    # calculate stats
    S = histstats(H, n)

    # Compute criterion function
    J = fill(-Inf, N-1)
    for t=1:(N-1)
        if S[t,1]!=0 && S[t,2]!=0 && S[t,5]>0 && S[t,6]>0
            J[t] = 1 + 2*(S[t,1]*log(sqrt(S[t,5])) + S[t,2]*log(sqrt(S[t,6]))) - 2*(S[t,1]*log(S[t,1]) + S[t,2]*log(S[t,2]))
        end
    end

    # Global minimum parameters
    depth, global_min = find_global_min(J, tol)

    return Kittler(S, depth, round(Int, global_min))
end

"""Otsu algorith thresholding type"""
struct Otsu <: Thresholding
    statistics::Matrix{Float64}
    depth::Float64
    minindex::Int
end
depth(t::LMCLUS.Otsu) = t.depth
stats(t::LMCLUS.Otsu) = t.statistics
Base.minimum(t::LMCLUS.Otsu) = t.minindex

"""Performs Otsu thresholding algorithm

N. Otsu: "A threshold selection method from gray-level histograms", Automatica, 1975, 11, 285-296.
"""
function fit(::Type{Otsu}, H::Vector{Int}, n::Int; tol=1.0e-5)
    N = length(H)

    # calculate stats
    S = histstats(H, n)

    # Compute criterion function
    J = zeros(N)
    for t in 1:N
        varb = S[t,1]*S[t,2]*(S[t,4]-S[t,3])^2
        varw = S[t,1]*S[t,5] - S[t,2]*S[t,6]
        J[t] = varb/varw
    end

    # Global minimum parameters
    depth, global_min = find_global_min(J, tol)

    return Otsu(S, depth, round(Int, global_min))
end