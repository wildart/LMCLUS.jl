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
include("logger.jl")
include("params.jl")
include("results.jl")
include("utils.jl")
include("separation.jl")
include("mdl.jl")

global DEBUG = false
function debug!()
    global DEBUG
    DEBUG = !DEBUG
end

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
        LOG(1, "Adjusting maximum manifold dimension to $(params.max_dim)")
    end

    # Main loop through dataset
    while length(index) > params.min_cluster_size

        # Find one manifold
        best_manifold, best_separation, remains = find_manifold(X, index, params, prngs, length(manifolds))

        # Add a new manifold cluster to collection
        push!(manifolds, best_manifold)
        push!(separations, best_separation)

        number_of_clusters += 1
        LOG(2, @sprintf("found cluster #%d, size=%d, dim=%d",
            number_of_clusters, length(labels(best_manifold)), indim(best_manifold)))

        # Stop clustering if found specified number of clusters
        length(manifolds) == params.stop_after_cluster && break

        # Look at the rest of the dataset
        index = remains
    end

    # Rest of the points considered as noise
    if length(index) > 0
        LOG(2, "outliers: $(length(index)), 0D cluster formed")
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
            criteria(sep) < params.best_bound && break

            # filter separated points
            selected, removed = filter_separated(selected, X, origin, basis, sep)

            # small amount of points is considered noise, try to bump dimension
            length(selected) <= params.min_cluster_size && break

            # create manifold cluster from good separation found
            best_manifold = Manifold(sep_dim, origin, basis, selected, threshold(sep), 0.0)
            best_separation = sep

            # partition cluster from the dataset
            append!(filtered, removed)

            LOG(3, "separated points: ", length(selected))
            separations += 1
        end

        if length(selected) <= params.min_cluster_size # no more points left - finish clustering
            LOG(3, "noise: dataset size < ", params.min_cluster_size," points")
            break
        end

        # perform basis adjustent
        if params.basis_alignment && outdim(best_manifold) > 0
            LOG(3, "manifold: perform basis adjustment...")
            adjustbasis!(best_manifold, X, adjust_dim=params.dim_adjustment)
            origin = mean(best_manifold)
            basis  = projection(best_manifold)

            # check if the adjusted basis provides better separation
            idxs = labels(best_manifold)
            adj_basis_separation = find_separation(view(X, :, idxs), origin, basis, params)
            log_separation(adj_basis_separation, params)

            # check separation threshold
            if criteria(adj_basis_separation) < params.best_bound
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
            LOG(3, "manifold: separating within manifold subspace...")
            origin = mean(best_manifold)
            basis  = projection(best_manifold)
            idxs = labels(best_manifold)
            orth_separation = find_separation(view(X, :, idxs),
                                              origin, basis, params, ocss = true)
            log_separation(orth_separation, params)

            # check separation threshold
            if criteria(orth_separation) < params.best_bound
                maxdist = extrema(orth_separation)[2]
                if best_manifold.σ <= maxdist
                    best_manifold.σ = maxdist
                end
            else
                # if found good separation, filter data and restart basis search
                selected, removed = filter_separated(idxs, X, origin, basis, orth_separation, ocss=true)
                best_manifold.points = selected
                best_manifold.σ = threshold(orth_separation)
                # ???? what do we do with this separation
                #best_separation = adj_basis_separation
                append!(filtered, removed)
                continue
            end
        end

        LOG(3, "no separation, ", sep_dim == params.max_dim ? "forming cluster..." : "increasing dimension...")

        # check compression ratio
        if params.mdl && indim(best_manifold) > 0 && separations > 0
            BM = best_manifold
            BMdata = X[:, selected]
            Pm = params.mdl_model_precision
            Pd = params.mdl_data_precision
            mmdl = MDL.calculate(MDL.DefaultType, BM, BMdata, Pm, Pd, ɛ = params.mdl_quant_error)
            mraw = MDL.calculate(MDL.Raw, BM, BMdata, Pm, Pd)

            cratio = mraw/mmdl
            LOG(4, "MDL: $mmdl, RAW: $mraw, COMPRESS: $cratio")
            if cratio < params.mdl_compres_ratio
                LOG(3, "MDL: low compression ration $cratio, required $(params.mdl_compres_ratio). Reject manifold... ")
                LOG(4, "$(length(selected))  $(length(filtered))  $(length(labels(best_manifold)))")

                # reset dataset to original state
                append!(selected, filtered)
                filtered = Int[]
                separations = 0
            end
        end

        sep_dim += 1
        !params.force_max_dim && separations > 0 && break
    end

    # Cannot find any manifold in data then form last cluster
    if indim(best_manifold) == N
        if outdim(best_manifold) == 0
            best_manifold.points = selected
        end
        LOG(3, "forming a cluster from the rest of $(outdim(best_manifold)) points")

        # calculate basis from the left points
        best_manifold.d = 1
        adjustbasis!(best_manifold, X, adjust_dim=params.dim_adjustment)

        # calculate bounds
        best_manifold.θ = maximum(distance_to_manifold(view(X, :,selected), best_manifold))
        if params.bounded_cluster
            best_manifold.σ = maximum(distance_to_manifold(view(X, :,selected), best_manifold, ocss=true))
        end
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

    LOG(3, "---------------------------------------------------------------------------------")
    LOG(3, "data size=", data_size,"   linear manifold dim=",
            lm_dim,"   space dim=", full_space_dim,"   searching for separation")
    LOG(3, "---------------------------------------------------------------------------------")

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
        LOG(4, "no good histograms to separate data !!!")
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
                         ocss::Bool = false)
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

        LOG(4, "sample distances for histogram: $n samples")
        sampleIndex = sample_points(X, n)
    end

    # calculate distances to the manifold
    distances = distance_to_manifold(params.histogram_sampling ? view(X, :,sampleIndex) : X ,
                                     origin, basis, ocss = ocss)
    # define histogram size
    bins = gethistogrambins(distances, params.max_bin_portion, params.hist_bin_size, params.min_bin_num)
    # perform separation
    return separation(params.sep_algo, distances, bins=bins, debug=DEBUG)
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

    LOG(4, "number of samples by first heuristic=", N, ", by second heuristic=", data_size*params.sampling_factor)

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

    LOG(3, "number of samples=", num_samples)

    num_samples
end

"""Calculate histogram size

Given a `x` value vector, the number of bins for the histogram is `hist_bin_size`.
If `max_bin_portion` value is in closed unit range, then the number of bins estimated from the bin size determined as `x_n+i - x_i`,
i.e. find the smallest difference between i successive points, where `x_n` is a point with index `n`,
and `i` is the max number of points we allow to be stored in a single bin.

*Note: the number of bins cannot be less then `min_bin_num`*
"""
function gethistogrambins(x::Vector, max_bin_portion::Float64, hist_bin_size::Int, min_bin_num::Int)
    bns = hist_bin_size
    if max_bin_portion > 0.0
        l = length(x)
        sort!(x)
        xmin, xmax = x[1], x[end]
        xrng = xmax - xmin
        mbp = round(Int, l * max_bin_portion)

        binwidth = xmax
        for i in mbp:mbp:l
            diff = x[i] - xmin
            if binwidth > diff && diff > 0.0
                binwidth = diff
            end
            xmin = x[i]
        end
        bns = round(Int, xrng/binwidth)

        # deal with special cases
        if bns > l
            bns = ceil(Int, log2(l))+1 # Sturges' formula
        end
    end
    return max(min_bin_num, bns) # special cases: use max bin size
end

"""Calculate the distances from points, `X`, to a linear manifold with `basis` vectors, translated to `origin` point.

`ocss` parameter turns on the distance calculatation to orthogonal complement subspace of the given manifold.
"""
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
distance_to_manifold(X::AbstractMatrix{T}, M::Manifold; ocss::Bool = false) where T<:Real =
    distance_to_manifold(X, mean(M), projection(M), ocss=ocss)

"""Calculate the distance from the `point` to the manifold, described by a `basis` matrix, and its orthoganal complement."""
function distance_to_manifold(point::AbstractVector, basis::AbstractMatrix)
    dpnt = sum(abs2, point)
    dprj = sum(abs2, basis' * point)
    return sqrt(abs(dpnt-dprj)), sqrt(dprj)
end
distance_to_manifold(point::AbstractVector, origin::AbstractVector, basis::AbstractMatrix) = distance_to_manifold(point - origin, basis)
distance_to_manifold(point::AbstractVector, M::Manifold) = distance_to_manifold(point - mean(M), projection(M))

end # module
