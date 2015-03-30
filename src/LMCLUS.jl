module LMCLUS

using MultivariateStats

export  lmclus,
        LMCLUSParameters,

        kittler, otsu,
        distance_to_manifold,
        histbst,

        Separation,
        criteria,
        threshold,

        Manifold,
        indim,
        outdim,
        labels,
        separation,
        mean,
        projection

include("types.jl")
include("params.jl")
include("utils.jl")
include("kittler.jl")
include("otsu.jl")
include("mdl.jl")

#
# Linear Manifold Clustering
#
function lmclus{T<:FloatingPoint}(X::Matrix{T}, params::LMCLUSParameters)
    # Setup RNG
    if nprocs() == 1
        if params.random_seed == 0
            srand(time_ns())
        else
            srand(params.random_seed)
        end
    else
        if params.random_seed == 0
            seeds = [time_ns() for i in 1:nprocs()]
        else
            seeds = [params.random_seed+10*i for i in 1:nprocs()]
        end
        @parallel for i=1:nprocs()
            srand(seeds[i])
        end
    end

    d, n = size(X)
    index = [1:n]
    cluster_number = 0
    manifolds = Manifold[]

    # Check if manifold maximum dimension is less then full dimension
    if d <= params.max_dim
        params.max_dim = d - 1
        LOG(params, 1, "Adjusting maximum manifold dimension to $(params.max_dim)")
    end

    # Main loop through dataset
    while length(index) > params.noise_size
        # Find one manifold
        best_manifold, remains = find_manifold(X, index, params)
        cluster_number += 1

        # Perform dimensioality regression
        if params.zero_d_search && indim(best_manifold) == 1
            LOG(params, 4, "Searching zero dimensional manifold")
            # TODO: Look for small dimensional embedding in a found manifold
        end

        # Perform basis alignment through PCA on found cluster
        if params.basis_alignment
            if indim(best_manifold) > 0 && !params.dim_adjustment
                R = fit(PCA, X[:, labels(best_manifold)];
                        method=:svd,
                        maxoutdim=indim(best_manifold))
            else
                R = fit(PCA, X[:, labels(best_manifold)];
                        method=:svd,
                        pratio = params.dim_adjustment_ratio > 0.0 ? params.dim_adjustment_ratio : 0.99)
            end
            pr = @sprintf("%.5f", principalratio(R))
            LOG(params, 3, "aligning manifold basis: $pr")
            best_manifold = Manifold(outdim(R), mean(R), projection(R),
                                    labels(best_manifold), separation(best_manifold))
        end

        # Add a new manifold cluster to collection
        LOG(params, 2, @sprintf("found cluster # %d, size=%d, dim=%d",
                cluster_number, length(labels(best_manifold)), indim(best_manifold)))
        push!(manifolds, best_manifold)

        # Stop clustering if found specified number of clusters
        if length(manifolds) == params.cluster_number
            break
        end

        # Look at the rest of the dataset
        index = remains
    end

    # Rest of the points considered as noise
    if length(index) > 0
        LOG(params, 2, "outliers: $(length(index)), 0D cluster formed")
        push!(manifolds, Manifold(0, zeros(d), eye(d,d), index, Separation()))
    end

    return manifolds
end

# Find manifold in multiple dimensions
function find_manifold{T<:FloatingPoint}(X::Matrix{T}, index::Array{Int,1}, params::LMCLUSParameters)
    filtered = Int[]
    selected = Array(Int, length(index))
    copy!(selected, index)

    best_manifold = Manifold()

    mdl = Inf
    mdl_manifold = Manifold()
    mdl_filtered = Int[]

    for sep_dim in params.min_dim:params.max_dim
        noise = false
        separations = 0

        while true
            origin, basis, sep = find_best_separation(X[:, selected], sep_dim, params)
            LOG(params, 4, "best bound: ", criteria(sep), " (", params.best_bound, ")")

            # No good separation found
            if criteria(sep) < params.best_bound
                if sep_dim == params.max_dim
                    LOG(params, 3, "no separation, forming cluster...")
                else
                    LOG(params, 3, "no separation, increasing dimension...")
                end
                break
            end

            # filter separated points
            manifold_points = Int[]
            removed_points  = Int[]
            for i=1:length(selected)
                idx = selected[i]
                # point i has distances less than the threshold value
                d = distance_to_manifold(X[:, idx], origin, basis)
                if d < threshold(sep)
                    push!(manifold_points, idx)
                else
                    push!(removed_points, idx)
                end
            end

            # small amount of points is considered noise, try to bump dimension
            if length(manifold_points) <= params.noise_size
                noise = true
                LOG(params, 3, "noise: cluster size < ", params.noise_size," points")
                break
            end

            # Create manifold cluster from good separation found
            best_manifold = Manifold(sep_dim, origin, basis, manifold_points, sep)
            selected = manifold_points
            append!(filtered, removed_points)
            LOG(params, 3, "separated points: ", length(manifold_points))
            separations += 1
        end

        if length(selected) <= params.noise_size # no more points left - finish clustering
            LOG(params, 3, "noise: dataset size < ", params.noise_size," points")
            break
        end

        if params.mdl && !noise && indim(best_manifold) > 0 && separations > 0
            l = MDLength(best_manifold, X[:, selected];
                        P = params.mdl_precision, dist = :OptimalQuant, #Empirical
                        ɛ = params.mdl_quant_error)
            cfl = params.mdl_precision*indim(best_manifold)*length(selected)
            if cfl/l < params.mdl_compres_ratio
                if l < mdl
                    LOG(params, 4, "MDL improved: $l < $mdl (C: $(outdim(mdl_manifold)), D: $(indim(mdl_manifold)), R: $cfl)")
                    mdl = l
                    mdl_manifold = copy(best_manifold)
                    mdl_filtered = copy(filtered)
                else
                    LOG(params, 4, "MDL is not improved: $(l) >= $(mdl) (C: $(outdim(mdl_manifold)), D: $(indim(mdl_manifold)))")
                end
            else
                mdl = cfl
                LOG(params, 4, "MDL does not provide improvement over raw data encoding: $cfl/$l >= $(cfl/l)")
            end
        end
    end

    # Cannot find any manifold in data then form 0D cluster
    if indim(best_manifold) == 0
        n = size(X, 1)
        best_manifold = Manifold(0, zeros(n), eye(n,n), selected, Separation())
        LOG(params, 3, "no linear manifolds found in data, 0D cluster formed")
    end

    # Check final best manifold MDL score
    #tmp_manifold = Manifold(best_dim, best_origin, best_basis, selected, best_sep)
    if params.mdl
        l = MDLength(best_manifold, X[:, selected];
                    P = params.mdl_precision, dist = :Empirical,
                    ɛ = params.mdl_quant_error)
        if l > mdl
            best_manifold = mdl_manifold
            filtered = mdl_filtered
            LOG(params, 4, "MDL degraded: $(l) > $(mdl) (Rollback to C: $(outdim(best_manifold)), D: $(indim(best_manifold)), F: $(length(filtered)))")
        end
    end

    best_manifold, filtered
end

#function best_manifold

best_separation(t1, t2) = criteria(t1[1]) > criteria(t2[1]) ? t1 : t2

# LMCLUS main function:
# 1- sample trial linear manifolds by sampling points from the data
# 2- create distance histograms of the data points to each trial linear manifold
# 3- of all the linear manifolds sampled select the one whose associated distance histogram
#    shows the best separation between to modes.
function find_best_separation{T<:FloatingPoint}(X::Matrix{T}, lm_dim::Int, params::LMCLUSParameters)
    full_space_dim, data_size = size(X)

    LOG(params, 3, "---------------------------------------------------------------------------------")
    LOG(params, 3, "data size=", data_size,"   linear manifold dim=",
            lm_dim,"   space dim=", full_space_dim,"   searching for separation")
    LOG(params, 3, "---------------------------------------------------------------------------------")

    # determine number of samples of lm_dim+1 points
    Q = sample_quantity( lm_dim, full_space_dim, data_size, params)

    # sample Q times SubSpaceDim+1 points
    best_sep = Separation()
    best_origin = Float64[]
    best_basis = zeros(0, 0)
    LOG(params, 5, "start sampling: ", Q)

    if nprocs() > 1
        # Parallel implementation
        best_sep, best_origin, best_basis = @parallel (best_separation) for i = 1:Q
            sample = sample_points(X, lm_dim+1)
            if length(sample) == 0
                (Separation(), Float64[], zeros(0, 0))
            else
                origin, basis = form_basis(X[:, sample])
                sep = try
                    find_separation(X, origin, basis, params)
                catch e
                    Separation()
                end
                (sep, origin, basis)
            end
        end
    else
        # Single thread implementation
        for i = 1:Q
            # Sample LM_Dim+1 points
            sample = sample_points(X, lm_dim+1)
            if length(sample) == 0
                continue
            end
            origin, basis = form_basis(X[:, sample])
            try
                sep = find_separation(X, origin, basis, params)
                LOG(params, 5, "SEP: ", criteria(sep), ", BSEP:", criteria(best_sep))
                if criteria(sep) > criteria(best_sep)
                    best_sep = sep
                    best_origin = origin
                    best_basis = basis
                end
            catch e
                LOG(params, 5, e)
                continue
            end
        end
    end

    cr = criteria(best_sep)
    if cr <= 0.
        LOG(params, 4, "no good histograms to separate data !!!")
    else
        LOG(params, 4, "separation: width=", best_sep.discriminability,
        "  depth=", best_sep.depth, "  criteria=", cr)
    end
    return best_origin, best_basis, best_sep
end

# Find separation criteria
function find_separation{T<:FloatingPoint}(X::Matrix{T}, origin::Vector{T},
                        basis::Matrix{T}, params::LMCLUSParameters)
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
        n= ( size(X, 2) <= n4 ? size(X, 2)-1 : n4 )

        LOG(params, 4, "find_separation: try to find $n samples")
        sampleIndex = sample_points(X, n)
    else
        sampleIndex = 1:size(X,2)
    end

    distances = distance_to_manifold(X[:,sampleIndex] , origin, basis)
    # Define histogram size
    bins = hist_bin_size(distances, params)
    return kittler(distances, bins=bins)
end

# Determine the number of times to sample the data in order to guaranty
# that the points sampled are from the same cluster with probability
# of error that does not exceed an error bound.
# Three different types of heuristics may be used depending on LMCLUS's input parameters.
function sample_quantity(lm_dim::Int, full_space_dim::Int, data_size::Int, params::LMCLUSParameters)

    k = params.cluster_number
    if k <= 1
        return 1 # case where there is only one cluster
    end

    p = 1.0 / k        # p = probability that 1 point comes from a certain cluster
    P = p^lm_dim       # P = probability that "k+1" points are from the same cluster
    N = abs(log10(params.error_bound)/log10(1-P))
    num_samples = 0

    LOG(params, 4, "number of samples by first heuristic=", N, ", by second heuristic=", data_size*params.sampling_factor)

    if params.sampling_heuristic == 1
        num_samples = int(N)
    elseif params.sampling_heuristic == 2
        num_samples = int(data_size*params.sampling_factor)
    elseif params.sampling_heuristic == 3
        num_samples = int(data_size*params.sampling_factor)
        if N < num_samples
            num_samples = int(N)
        end
    end
    num_samples = num_samples > 1 ? num_samples : 1

    LOG(params, 3, "number of samples=", num_samples)

    num_samples
end

# Forming basis from sample. The idea is to pick a point (origin) from the sampled points
# and generate the basis vectors by subtracting all other points from the origin,
# creating a basis matrix with one less vector than the number of sampled points.
# Then perform orthogonalization through Gram-Schmidt process.
# Note: Resulting basis is transposed.
function form_basis{T<:FloatingPoint}(X::Matrix{T})
    origin = X[:,1]
    basis = X[:,2:end] .- origin
    vec(origin), orthogonalize(basis)
end

# Modified Gram-Schmidt orthogonalization algorithm
function orthogonalize{T<:FloatingPoint}(vecs::Matrix{T})
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
    return params.hist_bin_size == 0 ? int(length(xs) * params.max_bin_portion) : params.hist_bin_size
end

# Calculates distance from point to manifold defined by basis
# Note: point should be translated wrt manifold origin
function distance_to_manifold{T<:FloatingPoint}(point::Vector{T}, basis::Matrix{T})
    d_n = 0.0
    d_v = basis' * point
    c = sumabs2(point)
    b = sumabs2(d_v)
    # @inbounds for j = 1:length(point)
    #     c += point[j]*point[j]
    #     b += d_v[j]*d_v[j]
    # end
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