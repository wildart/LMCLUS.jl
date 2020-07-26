"""Perform the dataset `xs` separation by thresholding a dataset `bins`ed histogram based on the algorith `T`"""
function separation(::Type{T}, xs::Vector{S};
                    bins = 20, tol::Real = 1.0e-5) where {T <: Thresholding, S <: Real}
    minX, maxX = extrema(xs)

    # get histogram
    r = range(minX, stop=maxX, length=bins+1)
    H = fit(Histogram, xs, r, closed=:left)
    N = length(H.weights)
    @logmsg DEBUG_SEPARATION "Separation histogram" H

    return try
        thresh = fit(T, H; tol=tol)

        minIdx = minimum(thresh)
        s = stats(thresh)
        threshold = minX + (minIdx)*(maxX - minX)/N
        # println(s[minIdx,:])
        discriminability = (abs(s[minIdx,3]-s[minIdx,4]))/(sqrt(s[minIdx,5]+s[minIdx,6]))

        Separation(convert(S, depth(thresh)), convert(S, discriminability), threshold, minIdx, minX, maxX, bins)
    catch ex
        # rethrow(ex)
        @logmsg TRACE "Separation error" ex
        Separation(zero(S), zero(S), maxX, 0, minX, maxX, bins) # threshold is set to maximal element
    end
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
function fit(::Type{Kittler}, H::Histogram; tol::Real=1.0e-5)
    N = length(H.weights)

    # calculate stats
    S = histstats(H.weights./sum(H.weights))

    # Compute criterion function
    J = fill(2*log(eps()), N-1)
    for t=1:(N-1)
        @logmsg TRACE "Statistics" P₁=S[t,1] P₂=S[t,2] σ₁=S[t,5] σ₂=S[t,6]
        if S[t,1]!=0 && S[t,2]!=0 && S[t,5]>0 && S[t,6]>0
            σ₁ = sqrt(max(0, S[t,5]))
            logσ₁ = log(σ₁ == 0 ? eps() : σ₁)
            σ₂ = sqrt(max(0, S[t,6]))
            logσ₂ = log(σ₂ == 0 ? eps() : σ₂)
            ses = S[t,1]*logσ₁ + S[t,2]*logσ₂
            se = S[t,1]*log(S[t,1]) + S[t,2]*log(S[t,2])
            J[t] = 1 + 2*ses - 2*se
        end
    end
    @logmsg DEBUG_SEPARATION "Kittler criterion function" J=J' extrema=extrema(J)

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
depth(t::Otsu) = t.depth
stats(t::Otsu) = t.statistics
Base.minimum(t::Otsu) = t.minindex

"""Performs Otsu thresholding algorithm

N. Otsu: "A threshold selection method from gray-level histograms", Automatica, 1975, 11, 285-296.
"""
function fit(::Type{Otsu}, H::Histogram; tol=1.0e-5)
    N = length(H.weights)

    # calculate stats
    S = histstats(H.weights./sum(H.weights))

    # Compute criterion function
    J = zeros(N-1)
    for t in 1:N-1
        σb = S[t,1]*S[t,2]*(S[t,4]-S[t,3])^2
        σw = S[t,1]*S[t,5]^2 - S[t,2]*S[t,6]^2
        J[t] = σb/σw
    end
    @logmsg DEBUG_SEPARATION "Otsu criterion function" J=J'

    # Global minimum parameters
    depth, global_min = find_global_min(J, tol)

    return Otsu(S, depth, round(Int, global_min))
end
