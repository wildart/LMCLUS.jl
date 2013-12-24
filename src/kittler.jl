function kittler(xs::Vector{Float}; bins=20)
    # find maximum and minimum
    maxX = maximum(xs)
    minX = minimum(xs)

    # get normalized histogram
    range, counts = hist(xs, bins)
    normH = counts/sum(counts)

    # calculate threshold
    P1 = 0
    P2 = 0
    μ1 = 0
    μ2 = 0
    σ1 = 0
    σ2 = 0

end
