module LMCLUS

import MultivariateStats: MultivariateStats, PCA, fit, principalratio, indim, outdim, projection

export  lmclus,

        separation,
        distance_to_manifold,

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
include("separation.jl")
include("mdl.jl")
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
    return lmclus(X, params, mts)
end

function lmclus(X::Matrix{T}, params::Parameters, prngs::Vector{MersenneTwister}) where {T<:Real}

    @assert length(prngs) >= nprocs() "Number of PRNGS cannot be less then processes."

    N, n = size(X)
    index = collect(1:n)
    number_of_clusters = 0
    manifolds = Manifold[]

    # Check if manifold maximum dimension is less then full dimension
    if N <= params.max_dim
        params.max_dim = N - 1
        LOG(params, 1, "Adjusting maximum manifold dimension to $(params.max_dim)")
    end

    # Main loop through dataset
    while length(index) > params.min_cluster_size
        # Find one manifold
        best_manifold, remains = find_manifold(X, index, params, prngs, length(manifolds))
        number_of_clusters += 1

        # Perform basis alignment through PCA on found cluster
        params.basis_alignment && adjustbasis!(best_manifold, X, params)

        # Perform dimensioality regression
        if params.zero_d_search && indim(best_manifold) <= 1
            LOG(params, 3, "Searching zero dimensional manifolds...")

            # adjust noise cluster to form 1D manifold
            if indim(best_manifold) == 0
                R = fit(PCA, X[:, labels(best_manifold)], maxoutdim = 1)
                best_manifold.d = 0
                best_manifold.μ = MultivariateStats.mean(R)
                best_manifold.proj = MultivariateStats.projection(R)
                LOG(params, 4, @sprintf("Adjusted noise cluster size=%d, dim=%d",
                               length(labels(best_manifold)), indim(best_manifold)))
            end

            while true
                L = labels(best_manifold)
                # project data to manifold subspace
                Z = project(best_manifold, X[:,L]) |> vec
                # determine separation parameters
                ZDsep = try
                    separation(LMCLUS.Kittler, Z)
                catch ex
                    if isa(ex, LMCLUSException)
                        LOG(params, 5, ex.msg)
                    else
                        LOG(params, 5, string(ex))
                    end
                    Separation()
                end
                # stop search if we cannot separate manifold
                criteria(ZDsep) < params.best_bound && break
                LOG(params, 4, "Found zero-dimensional manifold. Separating...")

                # separate cluster points
                cluster_points = L[find(p->p<threshold(ZDsep), Z)]
                removed_points = setdiff(L, cluster_points)

                # cannot form small clusters, stop searching
                length(cluster_points) <= params.min_cluster_size && break

                # update manifold description
                R = fit(PCA, X[:, removed_points], maxoutdim = 1)
                best_manifold.points = removed_points # its labels
                best_manifold.μ = MultivariateStats.mean(R)
                best_manifold.proj = MultivariateStats.projection(R)
                # best_manifold.d = MultivariateStats.outdim(R) == MultivariateStats.indim(R) ? 0 : MultivariateStats.outdim(R)
                LOG(params, 4, "Shunk cluster to $(length(removed_points)) points.")

                # form 0D cluster
                R = fit(PCA, X[:, cluster_points])
                zd_manifold = Manifold(0, MultivariateStats.mean(R),
                                       MultivariateStats.projection(R),
                                       cluster_points, ZDsep)
                LOG(params, 4, "0D cluster formed with $(length(cluster_points)) points.")

                # add new manifold to output
                push!(manifolds, zd_manifold)
                LOG(params, 2, @sprintf("found cluster #%d, size=%d, dim=%d",
                    number_of_clusters, length(labels(zd_manifold)), indim(zd_manifold)))
                number_of_clusters += 1

                # main body of the noise cluster is form, stop further search (???)
                indim(best_manifold) == 0 && break
            end
        end

        # Add a new manifold cluster to collection
        LOG(params, 2, @sprintf("found cluster #%d, size=%d, dim=%d",
                number_of_clusters, length(labels(best_manifold)), indim(best_manifold)))
        push!(manifolds, best_manifold)

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
    end

    return manifolds
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
            cluster_points, removed_points = filter_separeted(selected, X, origin, basis, sep)

            # small amount of points is considered noise, try to bump dimension
            if length(cluster_points) <= params.min_cluster_size
                LOG(params, 3, "noise: cluster size < ", params.min_cluster_size," points")
                break
            end

            # Create manifold cluster from good separation found
            selected = cluster_points
            best_manifold = Manifold(sep_dim, origin, basis, selected, sep)
            append!(filtered, removed_points)

            LOG(params, 3, "separated points: ", length(selected))
            separations += 1
        end

        if length(selected) <= params.min_cluster_size # no more points left - finish clustering
            LOG(params, 3, "noise: dataset size < ", params.min_cluster_size," points")
            break
        end

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

        #diagnostic(sep_dim, best_manifold, selected, params, mdl, ns, length(filtered))
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

    best_manifold, filtered
end

# LMCLUS main function:
# 1- sample trial linear manifolds by sampling points from the data
# 2- create distance histograms of the data points to each trial linear manifold
# 3- of all the linear manifolds sampled select the one whose associated distance histogram
#    shows the best separation between to modes.
function find_best_separation(X::Matrix{T}, lm_dim::Int,
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
    else
        LOG(params, 4, "separation: width=", best_sep.discriminability,
        "  depth=", best_sep.depth, "  criteria=", cr)
    end
    return best_sep, best_origin, best_basis, Q
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
        sep, origin, basis = calculate_separation(X, sample, params)
        if criteria(sep) > criteria(best_sep)
            best_sep = sep
            best_origin = origin
            best_basis = basis
        end
    end

    return best_sep, best_origin, best_basis, prng
end

function calculate_separation(X::Matrix{T}, sample::Vector{Int}, params::Parameters) where {T<:Real}
    origin, basis = form_basis(X[:, sample])
    sep = try
        find_separation(X, origin, basis, params)
    catch ex
        if isa(ex, LMCLUSException)
            LOG(params, 5, ex.msg)
        else
            LOG(params, 5, string(ex))
        end
        Separation()
    end
    return (sep, origin, basis)
end

# Find separation criteria
function find_separation(X::Matrix{T}, origin::Vector{T},
                         basis::Matrix{T}, params::Parameters) where {T<:Real}
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

    distances = distance_to_manifold(params.histogram_sampling ? X[:,sampleIndex] : X , origin, basis)
    # Define histogram size
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
    return params.hist_bin_size == 0 ? (round(Int, length(xs) * params.max_bin_portion)) : params.hist_bin_size
end

# Calculates distance from point to manifold defined by basis
# Note: point should be translated wrt manifold origin
function distance_to_manifold(point::Vector{T}, basis::Matrix{T}) where {T<:Real}
    d_n = 0.0
    d_v = basis' * point
    c = sum(abs2, point)
    b = sum(abs2, d_v)
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

distance_to_manifold(point::Vector{T}, origin::Vector{T}, basis::Matrix{T}) where {T<:Real} =
    distance_to_manifold(point - origin, basis)

# Determine the distance of each point in the dataset from to a linear manifold
function distance_to_manifold(X::Matrix{T}, origin::Vector{T}, basis::Matrix{T}) where {T<:Real}

    N, n = size(X)
    M = size(basis,2)
    # vector to hold distances of points from basis
    distances = zeros(T, n)
    tran = similar(X)
    @fastmath @inbounds for i in 1:n
        @simd for j in 1:N
            tran[j,i] = X[j,i] - origin[j]
            distances[i] += tran[j,i]*tran[j,i]
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
