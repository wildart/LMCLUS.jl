module LMCLUS

using LinearAlgebra
using Logging
import StatsBase: fit, Histogram, sturges
import Statistics: mean
import MultivariateStats: PCA, fit, indim, outdim, projection
import Clustering: ClusteringResult, assignments, counts, nclusters
import Distributed: Future, remotecall, nprocs, myid
import Random: _randjump, MersenneTwister, randperm, DSFMT, GLOBAL_RNG, AbstractRNG

export  lmclus,

        separation,
        distance_to_manifold,

        Manifold,
        indim,
        outdim,
        points,
        threshold,
        projection,
        criteria,

        nclusters,
        counts,
        assignments,
        manifold,
        manifolds,
        LMCLUSResult,
        refine

include("types.jl")
include("params.jl")
include("results.jl")
include("utils.jl")
include("separation.jl")
include("mdl.jl")

const TRACE = LogLevel(Base.CoreLogging.Debug.level-1000)
const DEBUG_SAMPLING = LogLevel(Base.CoreLogging.Debug.level-900)
const DEBUG_SEPARATION = LogLevel(Base.CoreLogging.Debug.level-800)

#
# Linear Manifold Clustering
#
function lmclus(X::AbstractMatrix{T}, params::Parameters, np::Int=nprocs()) where T<:Real
    # Setup RNG
    seed = getseed(params)
    mts = if np == 1
        [MersenneTwister(seed)]
    else
        #randjump(MersenneTwister(seed), np)
        mt = MersenneTwister(seed)
        jumps = fill(2^30, np-1)
        rngs = accumulate((mt, j)->_randjump(mt, DSFMT.calc_jump(j)), jumps, init=mt)
        pushfirst!(rngs, mt)
    end
    return LMCLUSResult(lmclus(X, params, mts)...)
end

function lmclus(X::AbstractMatrix{T}, params::Parameters, prngs::Vector{MersenneTwister}) where T<:Real

    @assert length(prngs) >= nprocs() "Number of PRNGS cannot be less then processes."

    N, n = size(X)
    index = collect(1:n)
    number_of_clusters = 0
    manifolds = Manifold[]
    separations = Separation[]

    # Check if manifold maximum dimension is less then full dimension
    if N <= params.max_dim
        params.max_dim = N - 1
        @debug "Adjusting maximum manifold dimension to $(params.max_dim)"
    end

    # Main loop through dataset
    while length(index) > params.min_cluster_size

        # Find one manifold
        best_manifold, best_separation, remains = find_manifold(X, index, params, prngs, length(manifolds))

        # Add a new manifold cluster to collection
        push!(manifolds, best_manifold)
        push!(separations, best_separation)

        number_of_clusters += 1
        @debug "Found cluster" number_of_clusters size=size(best_manifold) dimension=outdim(best_manifold)

        # Stop clustering if found specified number of clusters
        length(manifolds) == params.stop_after_cluster && break

        # Look at the rest of the dataset
        index = remains
    end

    # Rest of the points considered as noise
    if length(index) > 0
        @debug "Outliers" number=length(index)
        outliers = Manifold(0, zeros(T, N), zeros(T, N, 0), index)
        if params.basis_alignment
            adjustbasis!(outliers, X, adjust_dim=params.dim_adjustment, adjust_dim_ratio=params.dim_adjustment_ratio)
        end
        push!(manifolds, outliers)
        push!(separations, Separation())
    end

    return manifolds, separations
end

# Find manifold in multiple dimensions
# The algorithm contains following state machine:
# 0. Set inspected subspace dimension to 1.
# 1. Find linear subspace (translation & basis) with best possible separation
# 2. Evaluate the separation criteria
# 3. If separation criterion is met then
# 3.1. Filter out points within separation threshold
# 3.2. Form cluster from the points and remove them from the dataset, go to 5.
# 4. If separation criterion not is met then
# 4.1. If the option is set and the manifold basis wasn't adjusted before then
#      adjust the linear manifold basis of the cluster subspace and go to 2.
# 4.2. If the option is set and the separation search within the manifold subspace wasn't performed then
#      search for the separation within the manifold subspace and go to 2.
# 5. If the number of points in the cluster is less then manimal acceptable size
#    then go to 9.
# 6. If the option is set and the separation was found then
#    perform MDL evaluation of the former cluster.
# 6.1. If compression ration is not met, return cluster points to the dataset
#      then go to 8.
# 7. Return the formed cluster.
# 8. Increase the dimension of the inspected subspace
# 8.1. If the subspace dimension equals to N-1, go to 9.
# 8.2. Otherwise, go to 1.
# 9. Form cluster from the rest of the dataset points.
function find_manifold(X::AbstractMatrix{T}, index::Vector{Int},
                       params::Parameters,
                       prngs::Vector{MersenneTwister},
                       found::Int=0) where T<:AbstractFloat
    filtered = Int[]
    selected = copy(index)
    N = size(X,1) # full space dimension
    best_manifold = Manifold{T}(params.min_dim, index)
    best_separation = Separation()

    sep_dim = params.min_dim
    while sep_dim <= params.max_dim
        separations = 0
        state = :SEPARATION

        # get thresholds
        θ, σ = T(Inf), T(Inf)

        # search appropriate linear manifold subspace for best distance separation
        while true
            @debug "Algorithm State" state best_manifold
            if state == :SEPARATION
                # get separation manifold
                @debug("manifold: find suitable subspace...",
                    data_size=length(selected),
                    linear_manifold_dimension=sep_dim,
                    space_dim=N
                )
                sep, origin, basis = find_separation_basis(X[:, selected], sep_dim, params, prngs, found)
            elseif state == :ALIGNMENT && params.basis_alignment && size(best_manifold) > 0
                # check if the adjusted basis provides better separation
                previous_outdim = outdim(best_manifold)
                adjustbasis!(best_manifold, X, adjust_dim=params.dim_adjustment, adjust_dim_ratio=params.dim_adjustment_ratio)
                origin, basis = mean(best_manifold), projection(best_manifold)
                @debug "manifold: perform basis adjustment..." outdim=outdim(best_manifold) previous_outdim origin basis
                idxs = points(best_manifold)
                sep = find_separation(view(X, :, idxs), origin, basis, params, prngs[1])
            elseif state == :BOUND && params.bounded_cluster && size(best_manifold) > 0
                # check if the bounded cluster provides better separation
                origin, basis = mean(best_manifold), projection(best_manifold)
                @debug "manifold: separating within manifold subspace..." origin basis
                idxs = points(best_manifold)
                sep = find_separation(view(X, :, idxs), origin, basis, params, prngs[1], ocss = true)
            else
                break
            end
            @debug("Separation",
                width=sep.discriminability,
                depth=sep.depth,
                threshold=threshold(sep),
                extrema=extrema(sep),
                global_minimum=sep.globalmin,
                total_bins=sep.bins,
                criteria=criteria(sep),
                best_bound=params.best_bound
            )

            # No good separation found
            if criteria(sep) < params.best_bound
                currthr  = state == :BOUND ? σ : θ
                thr = extrema(sep)[2]
                if currthr > 0.0 && thr > 0.0
                    currthr = min(currthr, thr)
                end
                if state == :BOUND
                    σ = best_manifold.σ = currthr
                else
                    θ = best_manifold.θ = currthr
                end

                # swith to next state
                state = (state == :SEPARATION) ? :ALIGNMENT :
                        (state == :ALIGNMENT) ?  :BOUND     :
                        (state == :BOUND) ?      :FINISHED  : :NONE
            else
                thr = threshold(sep)
                θ, σ  = state == :BOUND ? (θ, thr) : (thr, σ)

                # filter separated points
                separated, removed = filter_separated(selected, X, origin, basis, sep,
                                                    ocss=(state == :BOUND))

                # small amount of points is considered noise, try to bump dimension
                if length(separated) <= params.min_cluster_size
                    @debug "Separated points cannot form cluster. Its size is too small ($(length(separated)))."
                    state = :FINISHED
                else
                    # create manifold cluster from good separation found
                    best_manifold = Manifold(sep_dim, origin, basis, separated, θ, σ)
                    best_separation = sep

                    # partition cluster from the dataset
                    append!(filtered, removed)

                    # refine dataset
                    selected = separated
                    @debug "Separated points" size=length(selected)
                    state = :SEPARATION
                    separations += 1
                end
            end
            @debug("Best manifold",
                dim=outdim(best_manifold),
                origin=mean(best_manifold),
                basis=projection(best_manifold),
                threshold=threshold(best_manifold),
            )
        end

        if length(selected) <= params.min_cluster_size # no more points left - finish clustering
            @debug "Noise" minimum_cluster_size=params.min_cluster_size current_cluster_size=length(selected)
            break
        end

        @debug "Separation not found" action=(sep_dim == params.max_dim ? "forming cluster" : "increasing dimension")

        # check compression ratio
        if params.mdl && outdim(best_manifold) > 0 && separations > 0
            BM = best_manifold
            BMdata = X[:, selected]
            Pm = params.mdl_model_precision
            Pd = params.mdl_data_precision
            adjustbasis!(BM, X) # generate full basis - needed to calculate MDL
            mmdl = MDL.calculate(params.mdl_algo, BM, BMdata, Pm, Pd, ɛ = params.mdl_quant_error)
            mraw = MDL.calculate(MDL.Raw, BM, BMdata, Pm, Pd)

            cratio = mraw/mmdl
            @debug("Minimal Description Length",
                mdl_value=mmdl,
                raw_value=mraw,
                compression=cratio,
                compression_threshold=params.mdl_compres_ratio,
                action=(cratio < params.mdl_compres_ratio ? :reject : :accept)
            )
            if cratio < params.mdl_compres_ratio
                # reset dataset to original state
                append!(selected, filtered)
                filtered = Int[]
                separations = 0
            end
        end

        sep_dim += 1
        @debug "Increasing search dimension" dim=sep_dim
        !params.force_max_dim && separations > 0 && break
    end

    # Cannot find any manifold in data then form last cluster
    require_alignment = false
    if outdim(best_manifold) == N
        if size(best_manifold) == 0
            best_manifold.points = selected
        end
        @debug "Forming a cluster from the rest of $(size(best_manifold)) points"
        best_manifold.d = 1

        # calculate bounds
        best_manifold.θ = maximum(distance_to_manifold(view(X, :,selected), best_manifold))
        best_manifold.σ = if params.bounded_cluster
            maximum(distance_to_manifold(view(X, :,selected), best_manifold, ocss=true))
        else
            Inf
        end
        require_alignment = true
    end

    # Adjust cluster basis & dimension if it is the last clusters or a corresponding setting is set
    if params.basis_alignment || require_alignment
        adjustbasis!(best_manifold, X, adjust_dim=params.dim_adjustment, adjust_dim_ratio=params.dim_adjustment_ratio)
    end

    best_manifold, best_separation, filtered
end

# LMCLUS main function:
# 1- sample trial linear manifolds by sampling points from the data
# 2- create distance histograms of the data points to each trial linear manifold
# 3- of all the linear manifolds sampled select the one whose associated distance histogram
#    shows the best separation between to modes.
function find_separation_basis(X::AbstractMatrix{T}, lm_dim::Int,
                               params::Parameters,
                               prngs::Vector{MersenneTwister},
                               found::Int=0) where {T<:Real}
    full_space_dim, data_size = size(X)

    # determine number of samples of lm_dim+1 points
    Q = sample_quantity( lm_dim, full_space_dim, data_size, params, found)

    # divide samples between PRNGs
    samples_proc = round(Int, Q / length(prngs))

    # enable parallel sampling
    bases = Array{Future}(undef, length(prngs))
    np = nprocs()
    nodeid = myid()
    for i in 1:length(prngs)
        pid = nodeid == 1 ? (i%np)+1 : nodeid #  if running from node 1
        bases[i] = remotecall(sample_manifold, pid, X, lm_dim+1, params, prngs[i], samples_proc)
    end

    # reduce values of manifolds from remote sources
    best_sep, best_origin, best_basis = Separation(), zeros(T, 0), zeros(T, 0, 0)
    for (i, rr) in enumerate(bases)
        sep, origin, basis, mt = fetch(rr)
        if criteria(sep) > criteria(best_sep) # get largest separation
            best_sep = sep
            best_origin = origin
            best_basis = basis
        end
        prngs[i] = mt # save state of RNG
    end

    # bad sampling
    criteria(best_sep) <= 0. &&  @debug "no good histograms to separate data !!!"
    return best_sep, best_origin, best_basis
end

function sample_manifold(X::Matrix{T}, lm_dim::Int,
                         params::Parameters, prng::MersenneTwister,
                         num_samples::Int) where {T<:Real}
    best_sep = Separation()
    best_origin = T[]
    best_basis = zeros(0, 0)

    for i in 1:num_samples
        sample = sample_points(X, lm_dim, rng=prng)
        length(sample) == 0 && continue

        origin, basis = form_basis(X, sample)
        sep = find_separation(X, origin, basis, params, prng)
        if criteria(sep) > criteria(best_sep)
            best_sep = sep
            best_origin = origin
            best_basis = basis
            @logmsg TRACE "Found new separation" separation=best_sep origin=best_origin basis=best_basis
        end
    end

    return best_sep, best_origin, best_basis, prng
end

"""Find separation criteria

Given a dataset `X`, an tanslation vector `origin` and set of basis vectors `basis` of the linear manifold.

`ocss` parameter enables separation for the orthogonal complement subspace of the linera manifold
"""
function find_separation(X::AbstractMatrix, origin::AbstractVector,
                         basis::AbstractMatrix, params::Parameters,
                         prng::MersenneTwister;
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

        @logmsg TRACE "sample distances for histogram" samples=n
        sampleIndex = sample_points(X, n, rng=prng)
    end

    # calculate distances to the manifold
    distances = distance_to_manifold(params.histogram_sampling ? view(X, :,sampleIndex) : X ,
                                     origin, basis, ocss = ocss)
    # define histogram size
    bins = gethistogrambins(distances, params.max_bin_portion, params.hist_bin_size, params.min_bin_num)
    # perform separation
    return separation(params.sep_algo, distances, bins=bins)
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

    @debug "number of samples" first_heuristic=N second_heuristic=data_size*params.sampling_factor

    return num_samples
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
        bns = bns > l ? sturges(l) : bns # deal with special cases
    end
    return max(min_bin_num, bns) # special cases: use max bin size
end

"""Calculate the distances from points, `X`, to a linear manifold with `basis` vectors, translated to `origin` point.

`ocss` parameter turns on the distance calculatation to orthogonal complement subspace of the given manifold.
"""
function distance_to_manifold(X::AbstractMatrix{T}, origin::AbstractVector{T}, basis::AbstractMatrix{T};
                              ocss::Bool = false) where T<:Real
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
    proj = transpose(basis) * tran
    @fastmath @inbounds for i in 1:n
        b = 0.0
        @simd for j in 1:M
            b += proj[j,i]*proj[j,i]
        end
        distances[i] = sqrt(abs(distances[i]-b))
    end
    return distances
end
distance_to_manifold(X::AbstractMatrix{<:Real}, M::Manifold; ocss::Bool = false) =
    distance_to_manifold(X, mean(M), projection(M), ocss=ocss)

"""Calculate the distance from the `point` to the manifold, described by a `basis` matrix, and its orthoganal complement."""
function distance_to_manifold(point::AbstractVector{T}, basis::AbstractMatrix{T}) where T<:Real
    dpnt = sum(abs2, point)
    dprj = sum(abs2, transpose(basis) * point)
    return sqrt(abs(dpnt-dprj)), sqrt(dprj)
end
distance_to_manifold(point::AbstractVector, origin::AbstractVector, basis::AbstractMatrix) = distance_to_manifold(point - origin, basis)
distance_to_manifold(point::AbstractVector, M::Manifold) = distance_to_manifold(point - mean(M), projection(M))

end # module
