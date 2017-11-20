# PERFORMS KITTLER'S MINIMAL THRESHOLDING ALGORITH
# REFERENCES:   J. Kittler & J. Illingworth: "Minimum Error Thresholding"
#                Pattern Recognition, Vol 19, nr 1. 1986, pp. 41-47.
using StatsBase

function kittler(xs::Vector{T}; bins = 20, tol = 1.0e-5, debug = false) where {T<:Real}
    # find maximum and minimum
    minX, maxX = extrema(xs)

    # get normalized histogram
    r = linspace(minX,maxX,bins+1)
    H = fit(Histogram, xs, r, closed=:left)
    Hw = H.weights
    Hn = Hw/convert(T, length(xs)-1)

    depth, discriminability, threshold, min_index, criterion_func = kittler(Hn, minX, maxX, tol=tol, debug=debug)
    # depth, discriminability, threshold, min_index, r, c
    Separation(depth, discriminability, threshold, min_index, collect(r))
end

function kittler(H::Vector{T}, minX::T, maxX::T;  tol=1.0e-5, debug = false) where {T<:Real}
    N = length(H)

    # calculate stats
    S = origstats(H)

    # Compute criterion function
    J = fill(-Inf, N-1)
    for t=1:(N-1)
        if S[t,1]!=0 && S[t,2]!=0
            J[t] = 1 + 2*(S[t,1]*log(sqrt(S[t,5])) + S[t,2]*log(sqrt(S[t,6]))) - 2*(S[t,1]*log(S[t,1]) + S[t,2]*log(S[t,2]))
        end
    end
    debug && println("H: $(H)")
    debug && println("J: $(J)")

    # Global minimum parameters
    depth, global_min = find_global_min(J, tol)
    min_index = round(Int, global_min)
    threshold = minX + ( global_min * (maxX - minX) / N )
    discriminability = (abs(S[min_index,3]-S[min_index,4]))/(sqrt(S[min_index,5]+S[min_index,6]))

    depth, discriminability, threshold, min_index, J
end

function find_global_min(J::Vector{T}, tol) where {T<:Real}
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
