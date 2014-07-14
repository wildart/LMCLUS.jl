module LMCLUS

export  lmclus,
        kittler,
        distance_to_manifold,
        LMCLUSParameters,
        Separation

include("types.jl")
include("params.jl")
include("kittler.jl")

#
# Linear Manifold Clustering
#
function lmclus(data::Matrix{Float64}, params::LMCLUSParameters)
    # Setup RNG
    if params.random_seed == 0
        srand(time_ns())
    else
        srand(params.random_seed)
    end

    data_rows, data_cols = size(data)
    data_index = [1:data_rows]
    cluster_number = 0
    manifolds = Manifold[]

    # Main loop through dataset
    while length(data_index) > params.noise_size
        # Find one manifold
        manifold_points, best_separation, separation_dimension, rest_points, noise = find_manifold(data, data_index, params)
        cluster_number += 1

        if params.zero_d_search
            # Look for small dimensional embedding in a found manifold
        end

        info(@sprintf("found cluster # %d, size=%d, dim=%d", cluster_number, length(manifold_points), separation_dimension))

        if !(noise || separation_dimension == 0)
            m = Manifold(separation_dimension, manifold_points, best_separation)
            push!(manifolds, m)
        else
            # Form from noise manifold of 0-dimension
            noise = Manifold(0, manifold_points, Separation())
            push!(manifolds, noise)
        end

        # Stop clustering if found specified number of clusters
        if length(manifolds) == params.cluster_number
            break
        end

        # Look at the rest of the dataset
        data_index = rest_points
    end
    manifolds
end

# Find manifold in multiple dimensions
function find_manifold(data::Matrix{Float64}, index::Array{Int,1}, params::LMCLUSParameters)
    noise = false
    manifold_points = index
    data_points = Int[]
    best_sep = Separation()
    best_dim = 0  # dimension in which separation was found

    for separation_dimension = 1:params.max_dim
        if noise
            break
        end

        while true
            separation = find_best_separation(data[manifold_points,:], separation_dimension, params)
            criteria = separation_criteria(separation)
            println("BEST_BOUND: ", criteria, " (", params.best_bound, ")")
            if criteria < params.best_bound
                if separation_dimension == params.max_dim
                    println("no separation")
                else
                    println("no separation, increasing dimension ...")
                end
                break
            end

            best_sep = separation
            best_dim = separation_dimension
            best_points = Int[]
            manifold_points_size = length(manifold_points)
            for i=1:manifold_points_size
                # point i has distances less than the threshold value
                d = distance_to_manifold(vec(data[manifold_points[i],:]), best_sep.origin, best_sep.basis)
                if d < best_sep.threshold
                    push!(best_points, i)
                else
                    push!(data_points, i)
                end
            end

            manifold_points = best_points
            manifold_points_size = length(manifold_points)
            println("Separated points: ", manifold_points_size)

            # small amount of points is considered noise
            if manifold_points_size < params.noise_size
                noise = true
                println("noise less than ", params.noise_size," points")
                break
            end
        end
    end

    manifold_points, best_sep, best_dim, data_points, noise
end

# LMCLUS main function:
# 1- sample trial linear manifolds by sampling points from the data
# 2- create distance histograms of the data points to each trial linear manifold
# 3- of all the linear manifolds sampled select the one whose associated distance histogram
#    shows the best separation between to modes.
function find_best_separation(data::Matrix{Float64}, lm_dim::Int, params::LMCLUSParameters)
    data_size, full_space_dim = size(data)

    println("---------------------------------------------------------------------------------")
    println("data size=", data_size,"   linear manifold dim=",
            lm_dim,"   space dim=", full_space_dim,"   searching for separation ...")
    println("---------------------------------------------------------------------------------")

    # determine number of samples of lm_dim+1 points
    Q = sample_quantity( lm_dim, full_space_dim, data_size, params)

    # sample Q times SubSpaceDim+1 points
    best_sep = Separation()
    for i = 1:Q
        try
            sep = find_separation(data, lm_dim, params)
            #println("SEP: ", separation_criteria(sep), ", BSEP:", separation_criteria(best_sep))
            if separation_criteria(sep) > separation_criteria(best_sep)
                best_sep = sep
            end
        catch
            continue
        end
    end

    criteria = separation_criteria(best_sep)
    if criteria <= 0.
        println("no good histograms to separate data !!!")
    else
        println("Separation: width=", best_sep.discriminability,
        "  depth=", best_sep.depth, "  criteria=", criteria)
    end
    best_sep
end

# Find separation criteria
function find_separation(data::Matrix{Float64}, lm_dim::Int, params::LMCLUSParameters)
    # Sample LM_Dim+1 points
    sample = sample_points(data, lm_dim+1)
    origin, basis = form_basis(data[sample,:])

    # Define sample for distance calculation
    if params.histogram_sampling
        Z_01=2.576  # Z random variable, confidence interval 0.99
        delta_p=0.2
        delta_mu=0.1
        P=1.0/params.cluster_number
        Q=1-P
        n1=(Q/P)*((Z_01*Z_01)/(delta_p*delta_p))
        p=( P<=Q ? P : Q )
        n2=(Z_01*Z_01)/(delta_mu*delta_mu*p)
        n3= ( n1 >= n2 ? n1 : n2 )
        n4= int(n3)
        n= ( size(data, 1) <= n4 ? size(data, 1)-1 : n4 )

        sampleIndex = sample_points(data, n)
        sample_data = data[sampleIndex, :]
    else
        sample_data = data
    end

    distances = distance_to_manifold(sample_data, origin, basis)
    # Define histogram size
    bins = hist_bin_size(distances, params)
    sep = kittler(distances, bins=bins)
    Separation(origin, basis, sep[1], sep[2], sep[3], sep[4], sep[5], sep[6])
end

# Determine the number of times to sample the data in order to guaranty
# that the points sampled are from the same cluster with probability
# of error that does not exceed an error bound.
# Three different types of heuristics may be used depending on LMCLUS's input parameters.
function sample_quantity(lm_dim::Int, full_space_dim::Int, data_size::Int, params::LMCLUSParameters)

    k = params.cluster_number
    if k == 1
        return 1 # case where there is only one cluster
    end

    p = 1.0 / k        # p = probability that 1 point comes from a certain cluster
    P = p^lm_dim       # P = probability that "k+1" points are from the same cluster
    N = log10(params.error_bound)/log10(1-P)
    num_samples = 0

    println("number of samples by first heuristic=", N, ", by second heuristic=", data_size*params.sampling_factor)

    if params.heuristic == 1
        num_samples = int(N)
    elseif params.heuristic == 2
        num_samples = int(data_size*params.sampling_factor)
    elseif params.heuristic == 3
        if N < (data_size*params.sampling_factor)
            num_samples = int(N)
        else
            num_samples = int(data_size*params.sampling_factor)
        end
    end

    println("number of samples=", num_samples)

    num_samples
end

# Sample randomly lm_dim+1 points from the dataset, making sure
# that the same point is not sampled twice. the function will return
# a index vector, specifying the index of the sampled points.
function sample_points(data::Matrix{Float64}, n::Int)
    if n <= 0
        error("Sample size must be positive")
    end
    data_rows, data_cols = size(data)
    I = Int[]
    hashes = Set{Uint}()
    hashes_size = 0

    # get sample indexes
    J = randperm2(data_rows, n-size(I,1))
    for idx in J
        h = hash(data[idx,:])
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
    if size(I,1) != n
        # Resample from the rest of data
        for idx in setdiff(randperm(data_rows),I)
            h = hash(data[idx,:])
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
    if size(I,1) != n
        error("Dataset doesn't have enough unique points, decrease number of points")
    end
    I
end

# Sample uniformly k integers from the integer range 1:n, making sure that
# the same integer is not sampled twice. the function returns an integer vector
# containing the sampled integers.
function randperm2(n, k)
    if n < k
        error("Sample size cannot be grater then range")
    end
    sample = Array(Int,0)
    sample_size = 0
    while sample_size < k
        selected = rand(1:n, k-sample_size)
        sample = unique(append!(sample, selected))
        sample_size = size(sample, 1)
    end
    sample
end

# Forming basis from sample. The idea is to pick a point (origin) from the sampled points
# and generate the basis vectors by subtracting all other points from the origin,
# creating a basis matrix with one less vector than the number of sampled points.
# Then perform orthogonalization through Gram-Schmidt process.
# Note: Resulting basis is transposed.
function form_basis(data::Matrix{Float64})
    origin = data[1,:]
    basis = data[2:end,:] .- origin
    vec(origin), orthogonalize(basis)
end


# Modified Gram-Schmidt orthogonalization algorithm
function orthogonalize(vecs::Matrix{Float64})
    n, m = size(vecs)
    basis = zeros(n, m)
    for j = 1:n
        v_j = vec(vecs[j,:])
        for i = 1:(j-1)
            q_i = vec(basis[i,:])
            r_ij = dot(q_i, v_j)
            v_j -= q_i*r_ij
        end
        r_jj = norm(v_j)
        basis[j,:] = r_jj != 0.0 ? v_j/r_jj : v_j
    end
    basis
end

# Calculate histogram size
function hist_bin_size{T}(xs::T, params::LMCLUSParameters)
    bins = params.hist_bin_size
    if bins == 0
        bins = int(size(xs, 1) * params.max_bin_portion)
    end
    bins
end

# Calculates distance from point to manifold defined by basis
# Note: point should be translated wrt manifold origin
function distance_to_manifold(point::Vector{Float64}, basis::Matrix{Float64})
    d_v = basis * point
    c = norm(point)
    b = norm(d_v)
    d_n = 0.0
    try
        d_n = sqrt(c*c-b*b)
        if d_n>1000000000
            d_n = 0.0
        end
    end
    d_n
end

distance_to_manifold(point::Vector{Float64},
    origin::Vector{Float64},
    basis::Matrix{Float64}) = distance_to_manifold(point - origin, basis)

# Determine the distance of each point in the dataset from to a linear manifold
function distance_to_manifold(data::Matrix{Float64},
    origin::Vector{Float64}, basis::Matrix{Float64})

    #dist(x::Vector{Float64}) = distance_to_manifold(x, origin, basis)
    #distances = mapslices(dist, sample_data, 2)
    #vec(distances)

    data_size = size(data, 1)
    # vector to hold distances of points from basis
    distances = zeros(Float64,data_size)
    for i=1:data_size
        distances[i] = distance_to_manifold(vec(data[i,:]), origin, basis)
    end
    distances
end


end # module
