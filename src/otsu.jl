# OTSU THRESHOLDING ALGORITH
# REFERENCES:   N. Otsu: "A threshold selection method from gray-level histograms"
#               Automatica, 1975, 11, 285-296.
using StatsBase

function otsu(xs::Vector{T}; bins = 20, debug = false) where {T<:Real}
    # find maximum and minimum
    minX, maxX = extrema(xs)

    # get normalized histogram
    r = linspace(minX, maxX, bins+1)
    H = fit(Histogram, xs, r)
    Hw = H.weights
    Hn = Hw/convert(T, length(xs)-1)

    threshold, min_index, varmax = otsu(Hn, r, debug=debug)
    Separation(varmax, 1., threshold, min_index, collect(r))
end

function otsu(H::Vector{T}, hrange::AbstractVector{Float64}; debug = false) where {T<:Real}
    N = length(H)

    hsum = sum((1:N).*H)
    bsum = 0.
    bw = 1
    fw = 1
    varmax = 0.
    thr = 1
    total = N

    for t in 1:N
        bw += H[t] # Weight background
        if bw == 0
            continue
        end

        fw = total - bw # Weight foreground
        if fw == 0
            break
        end

        bsum += t * H[t]

        bm = bsum/bw # Mean background
        fm = (hsum - bsum)/fw # Mean foreground

        # Calculate between class variance
        btwvar = bw * fw * (bm-fm) * (bm-fm)

        # Check if new maximum found
        if btwvar > varmax
            varmax = btwvar
            thr = t
            debug && println("BEST: ($(thr), $(varmax)), CURRENT: ($(t), $(btwvar))")
        end
    end
    hrange[thr], thr, varmax
end
