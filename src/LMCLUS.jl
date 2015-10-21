module LMCLUS

import MultivariateStats: MultivariateStats, PCA, fit, principalratio

export  lmclus,
        LMCLUSParameters, Diagnostic,

        kittler, otsu,
        distance_to_manifold,

        Separation,
        criteria,
        threshold,

        Manifold,
        indim,
        outdim,
        labels,
        separation,
        mean,
        projection,
        assignments

include("types.jl")
include("params.jl")
include("utils.jl")
include("kittler.jl")
include("otsu.jl")
include("mdl.jl")

#
# Linear Manifold Clustering
#
function lmclus{T<:AbstractFloat}(X::Matrix{T}, params::LMCLUSParameters, np::Int=nprocs())
    # Setup RNG
    seed = getseed(params)
    mts = if np == 1
        [MersenneTwister(seed)]
    else
        isdefined(:randjump) ? randjump(MersenneTwister(seed), np) : [MersenneTwister(seed+10*i) for i in 1:np]
    end
    return lmclus(X, params, mts)
end

function lmclus{T<:AbstractFloat}(X::Matrix{T},
                params::LMCLUSParameters,
                prngs::Vector{MersenneTwister})

    @assert length(prngs) >= nprocs() "Number of PRNGS cannot be less then processes."

    N, n = size(X)
    index = collect(1:n)
    cluster_number = 0
    manifolds = Manifold[]

    # Check if manifold maximum dimension is less then full dimension
    if N <= params.max_dim
        params.max_dim = N - 1
        LOG(params, 1, "Adjusting maximum manifold dimension to $(params.max_dim)")
    end

    # Main loop through dataset
    while length(index) > params.noise_size
        # Find one manifold
        best_manifold, remains = find_manifold(X, index, params, prngs, length(manifolds))
        cluster_number += 1

        # Perform dimensioality regression
        if params.zero_d_search && indim(best_manifold) == 1
            LOG(params, 4, "Searching zero dimensional manifold")
            # TODO: Look for small dimensional embedding in a found manifold
        end

        # Perform basis alignment through PCA on found cluster
        if params.basis_alignment
            if indim(best_manifold) > 0 && !params.dim_adjustment
                R = fit(PCA,
                        X[:, labels(best_manifold)];
                        method=:svd,
                        maxoutdim=indim(best_manifold))
            else
                R = fit(PCA,
                        X[:, labels(best_manifold)];
                        method=:svd,
                        pratio = params.dim_adjustment_ratio > 0.0 ? params.dim_adjustment_ratio : 0.99)
            end
            pr = @sprintf("%.5f", principalratio(R))
            LOG(params, 3, "aligning manifold basis: $pr")
            best_manifold = Manifold(MultivariateStats.outdim(R),
                                     MultivariateStats.mean(R),
                                     MultivariateStats.projection(R),
                                     labels(best_manifold), separation(best_manifold))
        end

        # Add a new manifold cluster to collection
        LOG(params, 2, @sprintf("found cluster # %d, size=%d, dim=%d",
                cluster_number, length(labels(best_manifold)), indim(best_manifold)))
        push!(manifolds, best_manifold)

        # Stop clustering if found specified number of clusters
        if length(manifolds) == params.stop_after_cluster
            break
        end

        # Look at the rest of the dataset
        index = remains
    end

    # Rest of the points considered as noise
    if length(index) > 0
        LOG(params, 2, "outliers: $(length(index)), 0D cluster formed")
        em = emptymanifold(N, index)
        push!(manifolds, em)
    end

    return manifolds
end

# Find manifold in multiple dimensions
function find_manifold{T<:AbstractFloat}(X::Matrix{T}, index::Array{Int,1},
                                         params::LMCLUSParameters,
                                         prngs::Vector{MersenneTwister},
                                         found::Int=0)
    filtered = Int[]
    selected = copy(index)
    N = size(X,1) # full space dimension
    best_manifold = emptymanifold(N)

    for sep_dim in params.min_dim:params.max_dim
        separations = 0
        ns = 0

        while true
            sep, origin, basis, ns = find_best_separation(X[:, selected], sep_dim, params, prngs, found)
            LOG(params, 4, "best bound: ", criteria(sep), " (", params.best_bound, ")")
            LOG(params, 4, "threshold: ", threshold(sep))

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
                LOG(params, 3, "noise: cluster size < ", params.noise_size," points")
                break
            end

            # Create manifold cluster from good separation found
            selected = manifold_points
            best_manifold = Manifold(sep_dim, origin, basis, selected, sep)
            append!(filtered, removed_points)

            LOG(params, 3, "separated points: ", length(selected))
            separations += 1
        end

        if length(selected) <= params.noise_size # no more points left - finish clustering
            LOG(params, 3, "noise: dataset size < ", params.noise_size," points")
            break
        end

        # check compression ratio
        if params.mdl && indim(best_manifold) > 0 && separations > 0
            mmdl = mdl(best_manifold, X[:, selected];
                        Pm = params.mdl_model_precision,
                        Pd = params.mdl_data_precision,
                        dist = :OptimalQuant, #Empirical
                        ɛ = params.mdl_quant_error)
            mraw = raw(best_manifold, params.mdl_data_precision)
            cratio = mraw/mmdl
            LOG(params, 4, "MDL: $mmdl, RAW: $mraw, COMPRESS: $cratio")
            if cratio < params.mdl_compres_ratio
                LOG(params, 3, "MDL: low compression ration $cratio, required $(params.mdl_compres_ratio). Reject manifold... ")
                best_manifold.d = N
                !params.force_max_dim && break #to higher dimensions
            end
        end

        #diagnostic(sep_dim, best_manifold, selected, params, mdl, ns, length(filtered))
        !params.force_max_dim && separations > 0 && break
    end

    # Cannot find any manifold in data then form 0D cluster
    if indim(best_manifold) == N
        if outdim(best_manifold) == 0
            best_manifold.points = selected
        end
        LOG(params, 3, "no linear manifolds found in data, noise cluster formed")
    end

    best_manifold, filtered
end

# LMCLUS main function:
# 1- sample trial linear manifolds by sampling points from the data
# 2- create distance histograms of the data points to each trial linear manifold
# 3- of all the linear manifolds sampled select the one whose associated distance histogram
#    shows the best separation between to modes.
function find_best_separation{T<:AbstractFloat}(X::Matrix{T}, lm_dim::Int,
                              params::LMCLUSParameters,
                              prngs::Vector{MersenneTwister},
                              found::Int=0)
    full_space_dim, data_size = size(X)

    LOG(params, 3, "---------------------------------------------------------------------------------")
    LOG(params, 3, "data size=", data_size,"   linear manifold dim=",
            lm_dim,"   space dim=", full_space_dim,"   searching for separation")
    LOG(params, 3, "---------------------------------------------------------------------------------")

    # determine number of samples of lm_dim+1 points
    Q = sample_quantity( lm_dim, full_space_dim, data_size, params, found)

    # divide samples between PRNGs
    samples_proc = round(Int, Q / length(prngs))

    arr = Array(RemoteRef, length(prngs))
    np = nprocs()
    for i in 1:length(prngs)
        arr[i] = remotecall((i%np)+1, sample_manifold, X, lm_dim+1, params, prngs[i], samples_proc)
    end

    # Reduce values of manifolds from remote sources
    best_sep = Separation()
    best_origin = Float64[]
    best_basis = zeros(0, 0)
    for (i, rr) in enumerate(arr)
        sep, origin, basis, mt = fetch(rr)
        if criteria(sep) > criteria(best_sep)
            best_sep = sep
            best_origin = origin
            best_basis = basis
        end
        prngs[i] = mt
    end

    cr = criteria(best_sep)
    if cr <= 0.
        LOG(params, 4, "no good histograms to separate data !!!")
    else
        LOG(params, 4, "separation: width=", best_sep.discriminability,
        "  depth=", best_sep.depth, "  criteria=", cr)
    end
    return best_sep, best_origin, best_basis, Q
end

function sample_manifold{T<:AbstractFloat}(X::Matrix{T}, lm_dim::Int,
                        params::LMCLUSParameters, prng::MersenneTwister, num_samples::Int)
    best_sep = Separation()
    best_origin = Float64[]
    best_basis = zeros(0, 0)

    for i in 1:num_samples
        sample = sample_points(X, lm_dim, prng)
        if length(sample) == 0
            continue
        end
        sep, origin, basis = calculate_separation(X, sample, params)
        if criteria(sep) > criteria(best_sep)
            best_sep = sep
            best_origin = origin
            best_basis = basis
        end
    end

    return best_sep, best_origin, best_basis, prng
end

function calculate_separation{T<:AbstractFloat}(X::Matrix{T}, sample::Vector{Int}, params::LMCLUSParameters)
    origin, basis = form_basis(X[:, sample])
    sep = try
        find_separation(X, origin, basis, params)
    catch ex
        LOG(params, 5, string(ex))
        Separation()
    end
    return (sep, origin, basis)
end

# Find separation criteria
function find_separation{T<:AbstractFloat}(X::Matrix{T}, origin::Vector{T},
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
        n4= round(Int, n3)
        n= ( size(X, 2) <= n4 ? size(X, 2)-1 : n4 )

        LOG(params, 4, "find_separation: try to find $n samples")
        sampleIndex = sample_points(X, n)
    end

    distances = distance_to_manifold(params.histogram_sampling ? X[:,sampleIndex] : X , origin, basis)
    # Define histogram size
    bins = hist_bin_size(distances, params)
    return kittler(distances, bins=bins)
end

# Determine the number of times to sample the data in order to guaranty
# that the points sampled are from the same cluster with probability
# of error that does not exceed an error bound.
# Three different types of heuristics may be used depending on LMCLUS's input parameters.
function sample_quantity(k::Int, full_space_dim::Int, data_size::Int,
                         params::LMCLUSParameters, S_found::Int)

    S_max = params.cluster_number
    if S_max <= 1
        return 1 # case where there is only one cluster
    end

    #p = 1.0 / S_max    # p = probability that 1 point comes from a certain cluster
    p = 1.0 / max(2, S_max-S_found)
    P = p^k            # P = probability that "k+1" points are from the same cluster
    N = abs(log10(params.error_bound)/log10(1-P))
    num_samples = 0

    LOG(params, 4, "number of samples by first heuristic=", N, ", by second heuristic=", data_size*params.sampling_factor)

    if params.sampling_heuristic == 1
        num_samples = isinf(N) ? typemax(Int) : round(Int, N)
    elseif params.sampling_heuristic == 2
        NN = data_size*params.sampling_factor
        num_samples = isinf(NN) ? typemax(Int) : round(Int, NN)
    elseif params.sampling_heuristic == 3
        NN = min(N, data_size*params.sampling_factor)
        num_samples = isinf(NN) ? typemax(Int) : round(Int, NN)
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
function form_basis{T<:AbstractFloat}(X::Matrix{T})
    origin = X[:,1]
    basis = X[:,2:end] .- origin
    vec(origin), orthogonalize(basis)
end

# Modified Gram-Schmidt orthogonalization algorithm
function orthogonalize{T<:AbstractFloat}(vecs::Matrix{T})
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

function form_basis_svd{T<:AbstractFloat}(X::Matrix{T})
    n = size(X,1)
    origin = mean(X,2)
    vec(origin), svdfact((X.-origin)'/sqrt(n))[:V][:,1:end-1]
end

# Calculate histogram size
function hist_bin_size(xs::Vector, params::LMCLUSParameters)
    return params.hist_bin_size == 0 ? (round(Int, length(xs) * params.max_bin_portion)) : params.hist_bin_size
end

# Calculates distance from point to manifold defined by basis
# Note: point should be translated wrt manifold origin
function distance_to_manifold{T<:AbstractFloat}(point::Vector{T}, basis::Matrix{T})
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

distance_to_manifold{T<:AbstractFloat}(
    point::Vector{T}, origin::Vector{T}, basis::Matrix{T}) = distance_to_manifold(point - origin, basis)

# Determine the distance of each point in the dataset from to a linear manifold
function distance_to_manifold{T<:AbstractFloat}(
    X::Matrix{T}, origin::Vector{T}, basis::Matrix{T})

    N, n = size(X)
    M = size(basis,2)
    # vector to hold distances of points from basis
    distances = zeros(T, n)
    tran = zeros(X)
    @simd for i in 1:n
        @simd for j in 1:N
            c = X[j,i] - origin[j]
            @inbounds tran[j,i] = c
            @inbounds distances[i] += c*c
        end
    end
    proj = At_mul_B(basis,tran)
    @simd for i in 1:n
        b = 0.0
        @simd for j in 1:M
            @inbounds b += proj[j,i]*proj[j,i]
        end
        @inbounds distances[i] = sqrt(abs(distances[i]-b))
    end
    return distances
end

end # module