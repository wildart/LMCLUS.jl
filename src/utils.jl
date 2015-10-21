"""Empty manifold"""
emptymanifold(N::Int, points::Vector{Int}=Int[]) = Manifold(N, zeros(N), eye(N,N), points, Separation())

"""Returns seed from parameters"""
getseed(params::LMCLUSParameters) = params.random_seed == 0 ? time_ns() : params.random_seed

"""
Sample uniformly k integers from the integer range 1:n, making sure that
the same integer is not sampled twice. the function returns an integer vector
containing the sampled integers.
"""
function randperm2(n, k)
    if n < k
        error("Sample size cannot be grater then range")
    end
    sample = Set()
    sample_size = 0
    while sample_size < k
        selected = rand(1:n, k-sample_size)
        union!(sample, selected)
        sample_size = length(sample)
    end
    collect(sample)
end

"""
Sample randomly lm_dim+1 points from the dataset, making sure
that the same point is not sampled twice. the function will return
a index vector, specifying the index of the sampled points.
"""
function sample_points{T<:AbstractFloat}(X::Matrix{T}, n::Int)
    if n <= 0
        error("Sample size must be positive")
    end
    data_rows, data_cols = size(X)
    I = Int[]
    hashes = Set{UInt}()
    hashes_size = 0

    # get sample indexes
    J = randperm2(data_cols, n-size(I,1))
    for idx in J
        h = hash(X[:, idx])
        push!(hashes, h)
        hashes_size += 1
        # Find duplicate row in sample and delete it
        if length(hashes) != hashes_size
            hashes_size -= 1
        else
            push!(I, idx)
        end
    end

    # Sample is smaller then expected
    if length(I) != n
        # Resample from the rest of data
        for idx in setdiff(randperm(data_cols),I)
            h = hash(X[:, idx])
            push!(hashes, h)
            hashes_size += 1
            # Find same rows and delete them
            if length(hashes) != hashes_size
                hashes_size -= 1
            else
                push!(I, idx)
            end
        end
    end

    # If at this point we do not have proper sample
    # then our dataset doesn't have enough unique rows
    if length(I) < n
        warn("Dataset doesn't have enough unique points, decrease number of points")
        I = Int[]
    end
    return I
end

"""Reservoir sampling"""
function sample_points{T<:AbstractFloat}(X::Matrix{T}, k::Int, r::MersenneTwister)
    N, n = size(X)
    if n < k
        warn("Not enough samples to construct manifold")
        return Int[]
    end

    I = collect(1:k)
    for i in (k+1):n
        j = trunc(Int, rand(r)*i)+1
        if j <= k
            I[j] = i
        end
    end

    for i in 1:(k-1)
        for j in (i+1):k
            if all([X[k,I[i]] == X[k,I[j]] for k in 1:N])
                warn("Sample is not unique: X[:,$(I[i])] == X[:,$(I[j])]")
                return Int[]
            end
        end
    end

    return I
end

"""
V-measure of contingency table
Accepts contingency table with row as classes, c, and columns as clusters, k.
"""
function V_measure(A::Matrix; β = 1.0)
    C, K = size(A)
    N = sum(A)

    # Homogeneity
    hck = 0.0
    for k in 1:K
        d = sum(A[:,k])
        for c in 1:C
            if A[c,k] != 0 && d != 0
                hck += log(A[c,k]/d) * A[c,k]/N
            end
        end
    end
    hck = -hck

    hc = 0.0
    for c in 1:C
        n = sum(A[c,:]) / N
        if n != 0.0
            hc += log(n) * n
        end
    end
    hc = -hc

    h = (hc == 0.0 || hck == 0.0) ? 1 : 1 - hck/hc

    # Completeness
    hkc = 0.0
    for c in 1:C
        d = sum(A[c,:])
        for k in 1:K
            if A[c,k] != 0 && d != 0
                hkc += log(A[c,k]/d) * A[c,k]/N
            end
        end
    end
    hkc = -hkc

    hk = 0.0
    for k in 1:K
        n = sum(A[:,k]) / N
        if n != 0.0
            hc += log(n) * n
        end
    end
    hk = -hk

    c = (hk == 0.0 || hkc == 0.0) ? 1 : 1 - hkc/hk

    # V-measure
    V_β = (1 + β)*h*c/(β*h + c)
    return V_β
end

function histogram3{T<:AbstractFloat}(V::Vector{T}, edgs)
    VI = sortperm(V, alg=Base.Sort.MergeSort)
    counts = zeros(Int,length(edgs)-1)
    b = 1
    nb = b+1
    for i in 1:length(V)
        if V[VI[i]] > edgs[nb]
            b += 1
            nb = b+1
        end
        counts[b] += 1
    end
    return edgs, counts, VI
end

function histogram2{T<:AbstractFloat}(V::Vector{T}, edgs)
    n = length(edgs)-1
    counts = zeros(Int,n)
    cindex = zeros(Int,length(V))
    @inbounds for i in 1:length(V)
        x = V[i]
        j = 1
        while j <= n
            edgs[j] > x && break
            j+=1
        end
        counts[j-1] += 1
        cindex[i] = j
    end
    return edgs, counts, cindex
end

function histogram{T<:AbstractFloat}(V::Vector{T}, edgs)
    n = length(edgs)-1
    counts = zeros(Int32,n)
    cindex = zeros(UInt32,length(V))
    @inbounds for i in 1:length(V)
        x = V[i]
        lo = 0
        hi = n+2
        while lo < hi-1
            m = (lo+hi)>>>1
            if edgs[m] < x
                lo = m
            else
                hi = m
            end
        end
        if lo > 0
            hi -= 1
            counts[hi] += 1
            cindex[i] = hi
        end
    end
    return edgs, counts, cindex
end

# r = linspace(0.,1.,51)
# h1 = map(x -> begin
#     srand(x)
#     xs = rand(10000)
#     tic()
#     LMCLUS.histogram(xs, r)
#     toq()
# end, 1:1000)

# r = linspace(0.,1.,51)
# h2 = map(x -> begin
#     srand(x)
#     xs = rand(10000)
#     tic()
#     LMCLUS.histogram2(xs, r)
#     toq()
# end, 1:1000)

# r = linspace(0.,1.,51)
# h3 = map(x -> begin
#     srand(x)
#     xs = rand(10000)
#     tic()
#     LMCLUS.histogram3(xs, r)
#     toq()
# end, 1:1000)

# r = linspace(0.,1.,51)
# h4 = map(x -> begin
#     srand(x)
#     xs = rand(10000)
#     tic()
#     hist(xs, r)
#     toq()
# end, 1:1000)

# htimes = hcat(h1,h2,h3,h4)
# hcat([:mean, :std, :median, :min, :max],
#     vcat(mean(htimes, 1),
#          std(htimes, 1),
#          median(htimes, 1),
#          minimum(htimes, 1),
#          maximum(htimes, 1))
# )