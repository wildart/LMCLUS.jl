# Sample uniformly k integers from the integer range 1:n, making sure that
# the same integer is not sampled twice. the function returns an integer vector
# containing the sampled integers.
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

# Sample randomly lm_dim+1 points from the dataset, making sure
# that the same point is not sampled twice. the function will return
# a index vector, specifying the index of the sampled points.
function sample_points{T<:FloatingPoint}(X::Matrix{T}, n::Int)
    if n <= 0
        error("Sample size must be positive")
    end
    data_rows, data_cols = size(X)
    I = Int[]
    hashes = Set{Uint}()
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

# Accepts contingency table with row as classes, c, and columns as clusters, k.
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
