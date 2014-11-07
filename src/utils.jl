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
function sample_points(X::Matrix{Float64}, n::Int)
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
    if length(I) != n
        warn("Dataset doesn't have enough unique points, decrease number of points")
        I = Int[]
    end
    I
end

function sample_points2(X::Matrix{Float64}, n::Int, trycount::Int = 10)
    if n <= 0
        error("Sample size must be positive")
    end
    data_rows, data_cols = size(X)
    I = Int[]

    tries = 0
    while length(I) < n
        idx = rand(1:data_cols)
        if idx ∉ I
            if length(I) > 0  # Check points coordinates
                found = false
                for i in I
                    if X[:, i] == X[:, idx]
                        tries += 1
                        found = true
                        break
                    end
                end
                if found
                    continue
                end
                push!(I, idx)
                tries = 0
            else
                push!(I, idx)
            end
        end
        if tries > trycount
            error("Dataset doesn't have enough unique points, decrease number of points")
        end
    end
    return I
end

# Sum of squares of vector components
function sumsq{T<:FloatingPoint}(x::Vector{T})
    sum = 0.0
    @inbounds for i = 1:length(x)
        v = x[i]
        sum += v*v
    end
    return sum
end

# Bootstrap histogram
function histbst{T}(x::Vector{T}; bins::Int = 10)
    x = sort(x)
    counts = Int[]

    # count points
    p = x[1]
    push!(counts, 1)
    i = 2
    while i <= length(x)
        if (x[i] - p) < eps()
            counts[end] += 1
            deleteat!(x, i)
        else
            p = x[i]
            i+=1
            push!(counts, 1)
        end
    end
    @assert length(x) == length(counts)

    S = length(x)
    L = int(sqrt(S)/2.)
    if (L < 1)
        return zeros(T, 0, 0)
    end

    # generate a mass function
    emf_x = Array(T, S-2*L)
    emf_y = Array(T, S-2*L)
    for i in L+1:S-L
        emf_x[i-L] = x[i]
        c = 0 # Cout all points in interval
        for j in -L:L
            c += counts[i+j]
        end
        emf_y[i-L] = c/(x[i+L]-x[i-L])
    end

    # generate bins boundaries
    min_d, max_d = emf_x[1], emf_x[end]
    bbins = [ min_d + i*(max_d-min_d)/bins for i in 1:bins+1]

    # interpolate and integrate linear piecewise PDF
    lppdf = Array(T, bins)
    lppdf[1] = emf_y[1]
    ilppdf = emf_y[1]
    for i in 1:bins
        tail = sum(emf_x .< bbins[i])
        ly = emf_y[tail]
        lx = emf_x[tail]
        gy = emf_y[tail+1]
        gx = emf_x[tail+1]
        lppdf[i] = (bbins[i] - lx)*(gy-ly)/(gx-lx)+ly
        ilppdf += 2*lppdf[i]
    end
    lppdf[bins] = emf_y[end]
    ilppdf += lppdf[bins]
    ilppdf *=(max_d-min_d)/(2*bins)
    lppdf /= ilppdf
    lppdf /= sum(lppdf)

    lppdf
end

function MDLength(M::Manifold, X::Matrix; P::Float64 = 32.0, T::Symbol = :Gausian, bins::Int = 20)
    n = size(X,1)
    m = indim(M)
    L = m == 0 ? P*n : P*(m*(n-1) + n) + m*outdim(M)

    E = 0.0
    if T == :Uniform && m > 0
        E = log(separation(M).threshold)
    elseif T == :Gausian && m > 0
        B = projection(M)
        OP = (eye(n) - B*B')*X
        Σ = OP*OP'
        E = n*(1+log(2π))/2 + log(det(Σ))
    elseif T == :Empirical # for 0D manifold only empirical estimate is avalible
        F = svdfact(X)
        BC = F[:U][:,(m+1):end]
        Y = BC'*X
        for i in 1:n-m
            C = vec(Y[i,:])
            Cmin, Cmax = extrema(C)
            Cx = linspace(Cmin, Cmax, bins)
            h = hist(C,Cx)[2]
            h /= sum(h)
            E += -sum(map(x-> x == 0. ? 0. : x*log2(x), h))
        end
    else
        E = outdim(M)*n
    end
    #warn("H:$(L) + D:$(E)")
    return L+E
end