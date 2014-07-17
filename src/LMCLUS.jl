module LMCLUS

export  lmclus,
        kittler,
        distance_to_manifold,
        LMCLUSParameters,
        Separation

include("params.jl")
include("utils.jl")
include("types.jl")
include("kittler.jl")

#
# Linear Manifold Clustering
#
function lmclus(X::Matrix{Float64}, params::LMCLUSParameters)
    # Setup RNG
    if params.random_seed == 0
        srand(time_ns())
    else
        srand(params.random_seed)
    end

    d, n = size(X)
    index = [1:n]
    cluster_number = 0
    manifolds = Manifold[]

    # Main loop through dataset
    while length(index) > params.noise_size
        # Find one manifold
        manifold_points, best_separation, separation_dimension, remains, noise = find_manifold(X, index, params)
        cluster_number += 1

        if params.zero_d_search
            # Look for small dimensional embedding in a found manifold
        end

        LOG(params, 1, @sprintf("found cluster # %d, size=%d, dim=%d", cluster_number, length(manifold_points), separation_dimension))

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
        index = remains
    end
    manifolds
end

# Find manifold in multiple dimensions
function find_manifold(X::Matrix{Float64}, index::Array{Int,1}, params::LMCLUSParameters)
    noise = false
    filtered = Int[]
    selected = Array(Int, length(index))
    copy!(selected, index)
    best_sep = Separation()
    best_dim = 0  # dimension in which separation was found

    for separation_dimension = params.min_dim:params.max_dim
        if noise
            break
        end

        while true
            separation = find_best_separation(X[:, selected], separation_dimension, params)
            criteria = separation_criteria(separation)
            LOG(params, 2, "BEST_BOUND: ", criteria, " (", params.best_bound, ")")
            if criteria < params.best_bound
                if separation_dimension == params.max_dim
                    LOG(params, 2, "no separation")
                else
                    LOG(params, 2, "no separation, increasing dimension ...")
                end
                break
            end

            best_sep = separation
            best_dim = separation_dimension
            best_points = Int[]
            for i=1:length(selected)
                idx = selected[i]
                # point i has distances less than the threshold value
                d = distance_to_manifold(X[:, idx], best_sep.origin, best_sep.basis)
                if d < best_sep.threshold
                    push!(best_points, idx)
                else
                    push!(filtered, idx)
                end
            end
            selected = best_points

            # small amount of points is considered noise
            if length(selected) < params.noise_size
                noise = true
                LOG(params, 2, "noise less than ", params.noise_size," points")
                break
            else
                LOG(params, 2, "Separated points: ", length(selected))
            end
        end
    end

    selected, best_sep, best_dim, filtered, noise
end

# LMCLUS main function:
# 1- sample trial linear manifolds by sampling points from the data
# 2- create distance histograms of the data points to each trial linear manifold
# 3- of all the linear manifolds sampled select the one whose associated distance histogram
#    shows the best separation between to modes.
function find_best_separation(X::Matrix{Float64}, lm_dim::Int, params::LMCLUSParameters)
    full_space_dim, data_size = size(X)

    LOG(params, 2, "---------------------------------------------------------------------------------")
    LOG(params, 2, "data size=", data_size,"   linear manifold dim=",
            lm_dim,"   space dim=", full_space_dim,"   searching for separation ...")
    LOG(params, 2, "---------------------------------------------------------------------------------")

    # determine number of samples of lm_dim+1 points
    Q = sample_quantity( lm_dim, full_space_dim, data_size, params)

    # sample Q times SubSpaceDim+1 points
    best_sep = Separation()
    LOG(params, 3, "Start sampling: ", Q)
    for i = 1:Q
        try
            sep = find_separation(X, lm_dim, params)
            LOG(params, 3, "SEP: ", separation_criteria(sep), ", BSEP:", separation_criteria(best_sep))
            if separation_criteria(sep) > separation_criteria(best_sep)
                best_sep = sep
            end
        catch e
            LOG(params, 4, e.msg)
            continue
        end
    end

    criteria = separation_criteria(best_sep)
    if criteria <= 0.
        LOG(params, 2, "no good histograms to separate data !!!")
    else
        LOG(params, 2, "Separation: width=", best_sep.discriminability,
        "  depth=", best_sep.depth, "  criteria=", criteria)
    end
    return best_sep
end

# Find separation criteria
function find_separation(X::Matrix{Float64}, lm_dim::Int, params::LMCLUSParameters)
    # Sample LM_Dim+1 points
    sample = sample_points(X, lm_dim+1)
    origin, basis = form_basis(X[:, sample])

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
        n= ( size(X, 1) <= n4 ? size(X, 1)-1 : n4 )

        sampleIndex = sample_points(X, n)
        X = X[:, sampleIndex]
    end

    distances = distance_to_manifold(X, origin, basis)
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

    LOG(params, 2, "number of samples by first heuristic=", N, ", by second heuristic=", data_size*params.sampling_factor)

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

    LOG(params, 2, "number of samples=", num_samples)

    num_samples
end

# Forming basis from sample. The idea is to pick a point (origin) from the sampled points
# and generate the basis vectors by subtracting all other points from the origin,
# creating a basis matrix with one less vector than the number of sampled points.
# Then perform orthogonalization through Gram-Schmidt process.
# Note: Resulting basis is transposed.
function form_basis(X::Matrix{Float64})
    origin = X[:,1]
    basis = X[:,2:end] .- origin
    vec(origin), orthogonalize(basis)
end

# Modified Gram-Schmidt orthogonalization algorithm
function orthogonalize(vecs::Matrix{Float64})
    m, n = size(vecs)
    basis = zeros(m, n)
    for j = 1:n
        v_j = vecs[:,j]
        for i = 1:(j-1)
            q_i = basis[:,i]
            r_ij = dot(q_i, v_j)
            v_j -= q_i*r_ij
        end
        r_jj = norm(v_j)
        basis[:,j] = r_jj != 0.0 ? v_j/r_jj : v_j
    end
    basis
end

# Calculate histogram size
function hist_bin_size(xs::Vector, params::LMCLUSParameters)
    params.hist_bin_size == 0 ? int(length(xs) * params.max_bin_portion) : params.hist_bin_size
end

# Calculates distance from point to manifold defined by basis
# Note: point should be translated wrt manifold origin
function distance_to_manifold{T<:FloatingPoint}(point::Vector{T}, basis::Matrix{T})
    d_n = 0.0
    d_v = basis' * point
    c = sumsq(point)
    b = sumsq(d_v)
    if c >= b
        d_n = sqrt(c-b)
        if d_n > 1e10
            warn("Distance is too large: $(point) -> $(d_v) = $(d_n)")
            d_n = 0.0
        end
    end
    return d_n
end

distance_to_manifold{T<:FloatingPoint}(
    point::Vector{T}, origin::Vector{T}, basis::Matrix{T}) = distance_to_manifold(point - origin, basis)

# Determine the distance of each point in the dataset from to a linear manifold
function distance_to_manifold{T<:FloatingPoint}(
    X::Matrix{T}, origin::Vector{T}, basis::Matrix{T})

    data_size = size(X, 2)
    # vector to hold distances of points from basis
    distances = zeros(Float64, data_size)
    for i=1:data_size
        @inbounds distances[i] = distance_to_manifold(X[:,i], origin, basis)
    end
    return distances
end

end # module
