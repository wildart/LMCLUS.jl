module LMCLUS

import MultivariateStats: MultivariateStats, PCA, fit, principalratio, indim, outdim, projection
import Clustering: ClusteringResult, assignments, counts, nclusters

export  lmclus,

        separation,
        distance_to_manifold,

        Manifold,
        indim,
        outdim,
        labels,
        threshold,
        mean,
        projection,
        criteria,

        nclusters,
        counts,
        assignments,
        manifold,
        manifolds

include("types.jl")
include("params.jl")
include("results.jl")
include("utils.jl")
include("separation.jl")
include("mdl.jl")
include("zerodim.jl")
include("deprecates.jl")

#
# Linear Manifold Clustering
#
function lmclus(X::Matrix{T}, params::Parameters, np::Int=nprocs()) where {T<:Real}
    # Setup RNG
    seed = getseed(params)
    mts = if np == 1
        [MersenneTwister(seed)]
    else
        randjump(MersenneTwister(seed), np)
    end
    return LMCLUSResult(lmclus(X, params, mts)...)
end

function lmclus(X::Matrix{T}, params::Parameters, prngs::Vector{MersenneTwister}) where {T<:Real}

    @assert length(prngs) >= nprocs() "Number of PRNGS cannot be less then processes."

    N, n = size(X)
    index = collect(1:n)
    number_of_clusters = 0
    manifolds = Manifold[]
    separations = Separation[]

    # Check if manifold maximum dimension is less then full dimension
    if N <= params.max_dim
        params.max_dim = N - 1
        LOG(params, 1, "Adjusting maximum manifold dimension to $(params.max_dim)")
    end

    # Main loop through dataset
    while length(index) > params.min_cluster_size

        # Find one manifold
        best_manifold, best_separation, remains = find_manifold(X, index, params, prngs, length(manifolds))

        # Add a new manifold cluster to collection
        push!(manifolds, best_manifold)
        push!(separations, best_separation)

        number_of_clusters += 1
        LOG(params, 2, @sprintf("found cluster #%d, size=%d, dim=%d",
            number_of_clusters, length(labels(best_manifold)), indim(best_manifold)))

        # Stop clustering if found specified number of clusters
        length(manifolds) == params.stop_after_cluster && break

        # Look at the rest of the dataset
        index = remains
    end

    # Rest of the points considered as noise
    if length(index) > 0
        LOG(params, 2, "outliers: $(length(index)), 0D cluster formed")
        em = emptymanifold(0, index)
        em.μ = zeros(T, N)
        em.proj = zeros(T, N, 0)
        push!(manifolds, em)
        push!(separations, Separation())
    end

    return manifolds, separations
end

# Find manifold in multiple dimensions
function find_manifold(X::Matrix{T}, index::Array{Int,1},
                       params::Parameters,
                       prngs::Vector{MersenneTwister},
                       found::Int=0) where {T<:Real}
    filtered = Int[]
    selected = copy(index)
    N = size(X,1) # full space dimension
    best_manifold = emptymanifold(N)
    best_separation = Separation()

    sep_dim = params.min_dim
    while sep_dim <= params.max_dim
        separations = 0

        # search appropriate linear manifold subspace for best distance separation
        while true
            sep, origin, basis = find_separation_basis(X[:, selected], sep_dim, params, prngs, found)
            log_separation(sep, params)

            # No good separation found
            check_separation(sep, params) && break

            # filter separated points
            selected, removed = filter_separated(selected, X, origin, basis, sep)

            # small amount of points is considered noise, try to bump dimension
            length(selected) <= params.min_cluster_size && break

            # create manifold cluster from good separation found
            best_manifold = Manifold(sep_dim, origin, basis, selected, threshold(sep), 0.0)
            best_separation = sep

            # partition cluster from the dataset
            append!(filtered, removed)

            LOG(params, 3, "separated points: ", length(selected))
            separations += 1
        end

        if length(selected) <= params.min_cluster_size # no more points left - finish clustering
            LOG(params, 3, "noise: dataset size < ", params.min_cluster_size," points")
            break
        end

        # perform basis adjustent
        if params.basis_alignment && outdim(best_manifold) > 0
            LOG(params, 3, "manifold: perform basis adjustment...")
            adjustbasis!(best_manifold, X)
            origin = mean(best_manifold)
            basis  = projection(best_manifold)

            # check if the adjusted basis provides better separation
            idxs = labels(best_manifold)
            adj_basis_separation = find_separation(view(X, :, idxs), origin, basis, params, debug=true)
            log_separation(adj_basis_separation, params)

            # check separation threshold
            if check_separation(adj_basis_separation, params)
                maxdist = extrema(adj_basis_separation)[2]
                if best_manifold.θ > maxdist
                    best_manifold.θ = maxdist
                end
            else
                # if found good separation, filter data and restart basis search
                selected, removed = filter_separated(idxs, X, origin, basis, adj_basis_separation)
                best_manifold.points = selected
                best_manifold.θ = threshold(adj_basis_separation)
                best_separation = adj_basis_separation
                append!(filtered, removed)
                continue
            end
        end

        # Estimate second bound for the cluster
        if params.bounded_cluster && outdim(best_manifold) > 0
            LOG(params, 3, "manifold: separating within manifold subspace...")
            origin = mean(best_manifold)
            basis  = projection(best_manifold)
            orth_separation = find_separation(view(X, :, labels(best_manifold)),
                                              origin, basis, params, ocss = true, debug=true)
            log_separation(orth_separation, params)

            # check separation threshold
            if check_separation(orth_separation, params)
                maxdist = extrema(orth_separation)[2]
                if best_manifold.σ > maxdist
                    best_manifold.σ = maxdist
                end
            else
                # if found good separation, filter data and restart basis search
                selected, removed = filter_separated(idxs, X, origin, basis, orth_separation)
                best_manifold.points = selected
                best_manifold.σ = threshold(orth_separation)
                # ???? what do we do with this separation
                #best_separation = adj_basis_separation
                append!(filtered, removed)
                continue
            end
        end

        LOG(params, 3, "no separation, ", sep_dim == params.max_dim ? "forming cluster..." : "increasing dimension...")

        # check compression ratio
        if params.mdl && indim(best_manifold) > 0 && separations > 0
            BM = best_manifold
            BMdata = X[:, selected]
            Pm = params.mdl_model_precision
            Pd = params.mdl_data_precision
            mmdl = MDL.calculate(MDL.DefaultType, BM, BMdata, Pm, Pd, ɛ = params.mdl_quant_error)
            mraw = MDL.calculate(MDL.Raw, BM, BMdata, Pm, Pd)

            cratio = mraw/mmdl
            LOG(params, 4, "MDL: $mmdl, RAW: $mraw, COMPRESS: $cratio")
            if cratio < params.mdl_compres_ratio
                LOG(params, 3, "MDL: low compression ration $cratio, required $(params.mdl_compres_ratio). Reject manifold... ")
                LOG(params, 4, "$(length(selected))  $(length(filtered))  $(length(labels(best_manifold)))")

                # reset dataset to original state
                append!(selected, filtered)
                filtered = Int[]
                separations = 0
            end
        end

        sep_dim += 1
        !params.force_max_dim && separations > 0 && break
    end

    # Cannot find any manifold in data then form 0D cluster
    if indim(best_manifold) == N
        if outdim(best_manifold) == 0
            best_manifold.points = selected
        end
        LOG(params, 3, "no linear manifolds found in data, noise cluster formed")
        best_manifold.d = 0
    end

    best_manifold, best_separation, filtered
end

# LMCLUS main function:
# 1- sample trial linear manifolds by sampling points from the data
# 2- create distance histograms of the data points to each trial linear manifold
# 3- of all the linear manifolds sampled select the one whose associated distance histogram
#    shows the best separation between to modes.
function find_separation_basis(X::Matrix{T}, lm_dim::Int,
                               params::Parameters,
                               prngs::Vector{MersenneTwister},
                               found::Int=0) where {T<:Real}
    full_space_dim, data_size = size(X)

    LOG(params, 3, "---------------------------------------------------------------------------------")
    LOG(params, 3, "data size=", data_size,"   linear manifold dim=",
            lm_dim,"   space dim=", full_space_dim,"   searching for separation")
    LOG(params, 3, "---------------------------------------------------------------------------------")

    # determine number of samples of lm_dim+1 points
    Q = sample_quantity( lm_dim, full_space_dim, data_size, params, found)

    # divide samples between PRNGs
    samples_proc = round(Int, Q / length(prngs))

    # enable parallel sampling
    arr = Array{Future}(length(prngs))
    np = nprocs()
    nodeid = myid()
    for i in 1:length(prngs)
        pid = nodeid == 1 ? (i%np)+1 : nodeid #  if running from node 1
        arr[i] = remotecall(sample_manifold, pid, X, lm_dim+1, params, prngs[i], samples_proc)
    end

    # reduce values of manifolds from remote sources
    best_sep = Separation()
    best_origin = T[]
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

    # bad sampling
    cr = criteria(best_sep)
    if cr <= 0.
        LOG(params, 4, "no good histograms to separate data !!!")
    end
    return best_sep, best_origin, best_basis
end

function sample_manifold(X::Matrix{T}, lm_dim::Int,
                         params::Parameters, prng::MersenneTwister,
                         num_samples::Int) where {T<:Real}
    best_sep = Separation()
    best_origin = T[]
    best_basis = zeros(0, 0)

    for i in 1:num_samples
        sample = sample_points(X, lm_dim, prng)
        if length(sample) == 0
            continue
        end
        origin, basis = form_basis(X, sample)
        sep = find_separation(X, origin, basis, params)
        if criteria(sep) > criteria(best_sep)
            best_sep = sep
            best_origin = origin
            best_basis = basis
        end
    end

    return best_sep, best_origin, best_basis, prng
end

"""Find separation criteria

Given a dataset `X`, an tanslation vector `origin` and set of basis vectors `basis` of the linear manifold.

`ocss` parameter enables separation for the orthogonal complement subspace of the linera manifold
"""
function find_separation(X::AbstractMatrix, origin::AbstractVector,
                         basis::AbstractMatrix, params::Parameters;
                         ocss::Bool = false, debug::Bool=false)
    # Define sample for distance calculation
    if params.histogram_sampling
        Z_01=2.576  # Z random variable, confidence interval 0.99
        delta_p=0.2
        delta_mu=0.1
        P=1.0/params.number_of_clusters
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

    # calculate distances to the manifold
    distances = distance_to_manifold(params.histogram_sampling ? view(X, :,sampleIndex) : X ,
                                     origin, basis, ocss = ocss)
    # define histogram size
    bins = hist_bin_size(distances, params)
    return separation(LMCLUS.Kittler, distances, bins=bins)
end

# Determine the number of times to sample the data in order to guaranty
# that the points sampled are from the same cluster with probability
# of error that does not exceed an error bound.
# Three different types of heuristics may be used depending on LMCLUS's input parameters.
function sample_quantity(k::Int, full_space_dim::Int, data_size::Int,
                         params::Parameters, S_found::Int)

    S_max = params.number_of_clusters
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

# Calculate histogram size
function hist_bin_size(xs::Vector, params::Parameters)
    return params.hist_bin_size == 0 ?
        round(Int, length(xs) * params.max_bin_portion) :
        params.hist_bin_size
end

# Determine the distance of each point in the dataset from to a linear manifold (batch-function)
function distance_to_manifold(X::AbstractMatrix{T}, origin::Vector{T}, basis::Matrix{T};
                              ocss::Bool = false) where {T<:Real}
    N, n = size(X)
    M = size(basis,2)
    # vector to hold distances of points from basis
    distances = zeros(T, n)
    tran = similar(X)
    @fastmath @inbounds for i in 1:n
        @simd for j in 1:N
            tran[j,i] = X[j,i] - origin[j]
            if !ocss
                distances[i] += tran[j,i]*tran[j,i]
            end
        end
    end
    proj = At_mul_B(basis,tran)
    @fastmath @inbounds for i in 1:n
        b = 0.0
        @simd for j in 1:M
            b += proj[j,i]*proj[j,i]
        end
        distances[i] = sqrt(abs(distances[i]-b))
    end
    return distances
end

end # module
