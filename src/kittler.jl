using StatsBase

"""Abstract histogram thresholding algorithm type"""
abstract type Thresholding end

"""Perform the dataset `xs` separation by thresholding a dataset `bins`ed histogram based on the algorith `T`"""
function separation(::Type{T}, xs::Vector{S};
                    bins = 20, tol = 1.0e-5,
                    debug = false) where {T <: Thresholding, S <: Real}
    minX, maxX = extrema(xs)

    # get histogram
    r = linspace(minX,maxX,bins+1)
    H = fit(Histogram, xs, r, closed=:left)
    N = length(H.weights)

    return try
        thresh = fit(T, H.weights, length(xs)-1)

        minIdx = minimum(thresh)
        s = stats(thresh)
        threshold = minX + ( midx * (maxX - minX) / N )
        discriminability = (abs(s[midx,3]-s[midx,4]))/(sqrt(s[midx,5]+s[midx,6]))

        Separation(depth(thresh), discriminability, threshold, minIdx, minX, maxX, bins)
    catch
        Separation(0.0, 0.0, maxX, 0, minX, maxX, bins)
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
function fit(::Type{Kittler}, H::Vector{Int}, n::Int; tol=1.0e-5)
    N = length(H)

    # calculate stats
    S = recurstats(H, n)

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

function find_global_min(J::Vector{T}, tol::T) where {T<:Real}
    N = length(J)

    # Mark minima
    M = zeros(Bool,N)
    if N-1 >= 1
        prev = J[2] - J[1]
        curr = 0.0
        for i=2:(N-1)
            curr = J[i+1] - J[i]
            M[i] = prev<=0 && curr>=0
            prev=curr
        end
    end

    # Find global minima of criterion funtion if exists
    # find first minimum
    lmin = 1
    while lmin<N && !M[lmin]
        lmin += 1
    end

    depth = 0
    global_min = 0
    if lmin == N
        throw(LMCLUSException("No minimum found, unimode histogram"))
    else
        while lmin < N
            # Detect flat
            rmin = lmin
            while rmin<N && M[rmin]
                rmin += 1
            end
            loc_min=( lmin + rmin - 1 ) / 2

            # Monotonically ascend to the left
            lheight = round(Int, loc_min)
            while lheight > 1 && J[lheight-1] >= J[lheight]
                lheight -= 1
            end

            # Monotonically ascend to the right
            rheight = round(Int, loc_min)
            while rheight < N && J[rheight] <= J[rheight+1]
                rheight += 1
            end

            # Compute depth
            local_depth = 0
            local_depth = (J[lheight] < J[rheight] ? J[lheight] : J[rheight]) - J[round(Int, loc_min)]

            if local_depth > depth
                depth = local_depth
                global_min = loc_min
            end

            lmin = rmin
            while lmin<N && !M[lmin]
                lmin += 1
            end
        end
    end

    if depth < tol
        throw(LMCLUSException("No minimum found, unimode histogram"))
    end

    depth, global_min
end

"""Otsu algorith thresholding type"""
struct Otsu <: Thresholding
    statistics::Matrix{Float64}
    minindex::Int
end
depth(t::Otsu) = t.depth
stats(t::Otsu) = t.statistics
Base.minimum(t::Otsu) = t.minindex
