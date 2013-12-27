module LMCLUS

export  lmclus,
        kittler,
        distance_to_manifold,
        LMCLUSParameters

include("kittler.jl")

type LMCLUSParameters
    max_dim::Int
    cluster_number::Int
    hist_bin_size::Int
    noise_size::Int
    best_bound::Float64
    error_bound::Float64
    max_bin_portion::Float64
    random_seed::Int64
    heuristic::Int
    sampling_factor::Float64
    histogram_sampling::Bool

    LMCLUSParameters(dims)=new(dims, 100, 0, 20, 1.0, 0.0001, 0.1, 0, 3, 0.003, false)
end

import Base.show
show(io::IO, p::LMCLUSParameters) =
    print(io, """Linear Manifold Clustering parameters:
    Max dimension (max_dim): $(p.max_dim)
    Number of clusters (cluster_number): $(p.cluster_number)
    Noise size (noise_size): $(p.noise_size)
    Best bound (best_bound): $(p.best_bound)
    Error bound (error_bound): $(p.error_bound)
    Sample points for distance histogram (histogram_sampling): $(p.histogram_sampling)
    Histogram bins (hist_bin_size): $(p.hist_bin_size)
    Maximum histogram bin size (max_bin_portion): $(p.max_bin_portion)
    Sampling heuristic (heuristic): $(p.heuristic)
    Sampling factor (sampling_factor): $(p.sampling_factor)
    Random seed (random_seed): $(p.random_seed) (0 - random seed)
    """)

# Linear Manifold Clustering
function lmclus(data::Matrix{Float64}, params::LMCLUSParameters)
    # Setup RNG
    if params.random_seed == 0
        srand(time_ns())
    else
        srand(params.random_seed)
    end
    
    data_rows, data_cols = size(data)
    data_index =1:data_rows
    ClusterNum = 0
    
    # Main loop through dataset
    while length(data_index) > params.noise_size
        nonClusterPoints, separations, Noise, SepDim = find_manifold(ds, params, data_index)
        
    end
    
    ndims(ds)
end

function find_manifold(data::Matrix{Float64}, params::LMCLUSParameters, index::Array{Int64,1})
    Noise = false
    nonClusterPoints = Array(Int, 0)
    SepDim = 0  # dimension in which separation was found
    
    for lm_dim = 1:params.max_dim+1
        if Noise
            break
        end
        
        while true
            best_sep = find_best_separation(data, params, index, lm_dim)
            info("BEST_BOUND:" )
        end
    
    end
    
    Noise, nonClusterPoints, SepDim
end

function find_best_separation(data::Matrix{Float64}, params::LMCLUSParameters, lm_dim::Int)
    data_size, full_space_dim = size(data)

    println("data size=", data_size,"   linear manifold dim=", lm_dim,"   space dim=", full_space_dim,"   searching for separation ...")
    println("---------------------------------------------------------------------------------")

    # determine number of samples of lm_dim+1 points
    Q = sample_quantity( lm_dim, full_space_dim, data_size, params)
    for i = 1:Q
        try
            sample = sample_points(data, lm_dim)
            origin, basis = form_basis(sample)
            distances = calculate_distance_to_manifold(data, basis, origin, params)
            # Define histogram size
            bins = hist_bin_size(distances, params)
            depth, discriminability, threshold, globalmin = kittler(distances)
        catch
            continue
        end
    end

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

# Sample uniformly k integers from the integer range 1:n, making sure that
# the same integer is not sampled twice. the function returns an intger vector
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

# Sample randomly lm_dim+1 points from the dataset, making sure
# that the same point is not sampled twice. the function will return
# a index vector, specifying the index of the sampled points.
function sample_points(data::Matrix{Float64}, d::Int)
    if d <= 0
        error("Dimension size must be positive")
    end
    data_rows, data_cols = size(data)
    total = d+1
    I = Int[]
    hashes = Set{Uint}()
    hashes_size = 0

    # get sample indexes
    J = randperm2(data_rows, total-size(I,1))
    for idx in J
        h = hash(A[idx,:])
        push!(hashes, h)
        hashes_size += 1
        # Find duplicate row in sample and delete it
        if length(hashes) != hashes_size
            hashes_size -= 1
        else
            push!(I, idx)
        end
    end

    # Sample is smaller then expeced
    if size(I,1) != total
        # Resample from the rest of data
        for idx in setdiff(randperm(data_rows),I)
            h = hash(A[idx,:])
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
    if size(I,1) != total
        error("Dataset doesn't have enough unique points, decrease number of points")
    end
    I
end

# Modified Gram-Schmidt orthogonalization algorithm
function orthogonalize(vecs::Matrix{Float64})
    n = size(vecs,1)
    basis = zeros(n,n)
    for j = 1:n
        v_j = vec(vecs[j,:])
        for i = 1:(j-1)
            q_i = vec(basis[i,:])
            r_ij = dot(q_i, v_j)
            v_j -= q_i*r_ij
        end
        r_jj = norm(v_j)
        basis[j,:] = v_j/r_jj
    end
    d = det(basis)
    if det == 0 || isnan(d)
        error("Basis vectors are linearly dependent.")
    end
    basis
end

# Forming basis from sample. The idea is to pick a point (origin) from the sampled points
# and generate the basis vectors by subtracting all other points from the origin,
# creating a basis matrix with one less vector than the number of sampled points.
# Then perform orthagonalization through Gramm-Schmidt process.
# Note: Resulting basis is transposed.
function form_basis(data::Matrix{Float64})
    origin = data[1,:]
    basis = broadcast(-, data[2:,:], origin)
    origin', orthogonalize(basis)
end


# Calculate histogram size
function hist_bin_size{T}(xs::T, params::LMCLUSParameters)
    bins = params.hist_bin_size
    if bins == 0
        bins = int(size(xs, 1) * params.max_bin_portion)
    end
    bins
end

# Determine the distance of each point in the data set from to a linear manifold,
function calculate_distance_to_manifold(data::Matrix{Float64}, basis::Matrix{Float64},
    oring::Vector{Float64}, params::LMCLUSParameters)

end

# Calculates distance from point to manifold defined by basis
# Note: point should be translated wrt manifold origin
function distance_to_manifold(point::Vector{Float64}, basis::Matrix{Float64})
    d_v = basis * point
    c = norm(point)
    b = norm(d_v)
    try
        d_n = sqrt(c*c-b*b)
        if d_n>1000000000
            d_n = 0.0
        end
    catch
        d_n = 0
    end
    d_n
end

end # module
