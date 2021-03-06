"""Returns seed from parameters"""
getseed(params::Parameters) = params.random_seed == 0 ? time_ns() : params.random_seed

"""
Sample uniformly k integers from the integer range 1:n, making sure that
the same integer is not sampled twice. the function returns an integer vector
containing the sampled integers.
"""
function randpermset(n::Int, k::Int, rng::AbstractRNG)
    if n < k
        error("Sample size cannot be grater then range")
    end
    sample = Set{UInt}()
    sample_size = 0
    while sample_size < k
        selected = rand(rng, 1:n, k-sample_size)
        union!(sample, selected)
        sample_size = length(sample)
    end
    return sample
end

"""
Sample randomly lm_dim+1 points from the dataset, making sure
that the same point is not sampled twice.
The function will return a index vector, specifying the index of the sampled points.
"""
function sample_points(X::AbstractMatrix{T}, n::Int;
                       rng::AbstractRNG=GLOBAL_RNG,
                       resample::Int=3) where T <: Real
    if n <= 0
        error("Sample size must be positive")
    end
    data_rows, data_cols = size(X)
    idxs = Set{UInt}()
    hashes = Set{UInt}()
    hashes_size = 0
    resample_count = 0

    # get sample indexes
    while length(idxs) != n && resample_count < resample
        J = randpermset(data_cols, n, rng)
        for idx in setdiff(J, idxs)
            h = hash(view(X, :, idx))
            push!(hashes, h)
            hashes_size += 1
            # Find duplicate row in sample and delete it
            if length(hashes) != hashes_size
                hashes_size -= 1
            else
                push!(idxs, idx)
            end
        end
        resample_count += 1
        @logmsg(DEBUG_SAMPLING,
            "Sampling iteration $resample_count",
            sample_indexes = idxs,
            sampled =J
        )
    end

    # If at this point we do not have proper sample
    # then our dataset doesn't have enough unique rows
    if length(idxs) < n
        @logmsg(DEBUG_SAMPLING,
            "Dataset doesn't have enough unique points, decrease number of sampled points",
            sample_size = n,
            sampled = length(idxs)
        )
        return UInt[]
    end
    return collect(idxs)
end

"""Reservoir sampling"""
function reservoir_sample_points(X::AbstractMatrix{T}, k::Int, r::MersenneTwister) where T <: Real
    N, n = size(X)
    if n < k
        @warn "Not enough samples to construct manifold" maxlog=5
        return Int[]
    end

    idxs = collect(1:k)
    for i in (k+1):n
        j = trunc(Int, rand(r)*i)+1
        if j <= k
            idxs[j] = i
        end
    end

    for i in 1:(k-1)
        for j in (i+1):k
            if all([X[k,idxs[i]] == X[k,idxs[j]] for k in 1:N])
                @warn "Sample is not unique: X[:,$(idxs[i])] == X[:,$(idxs[j])]"
                return Int[]
            end
        end
    end

    return idxs
end

"""Modified Gram-Schmidt orthogonalization algorithm"""
function orthogonalize(vecs::AbstractMatrix{T}) where T<:Real
    m, n = size(vecs)
    basis = zeros(T, m, n)
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

# Forming basis from sample. The idea is to pick a point (origin) from the sampled points
# and generate the basis vectors by subtracting all other points from the origin,
# creating a basis matrix with one less vector than the number of sampled points.
# Then perform orthogonalization through Gram-Schmidt process.
# Note: Resulting basis is transposed.
function form_basis(X::AbstractMatrix{T}, sample_indices::Vector{S}) where {T<:AbstractFloat, S<:Integer}
    origin = X[:,sample_indices[1]]
    basis = X[:,sample_indices[2:end]] .- origin
    vec(origin), orthogonalize(basis)
end

function filter_separated(selected_points, X, O, B, S; ocss=false)
    cluster_points = Int[]
    removed_points  = Int[]
    θ = threshold(S)
    ii = ocss ? 2 : 1

    for i in eachindex(selected_points)
        idx = selected_points[i]
        # point i has distances less than the threshold value
        d = distance_to_manifold(X[:, idx], O, B)
        if d[ii] < θ
            push!(cluster_points, idx)
        else
            push!(removed_points, idx)
        end
    end

    return cluster_points, removed_points
end

function adjustbasis!(M::Manifold, X::AbstractMatrix{T};
                      adjust_dim::Bool=false, adjust_dim_ratio::Real=0.99) where {T<:AbstractFloat}
    pts = points(M)
    MX = view(X, :, pts)
    d,n = size(MX)
    μ = mean(MX, dims=2)
    Svd = svd(MX .- μ)
    v = Svd.S
    U = Svd.U
    map!(x->abs2(x)/length(pts),v,v)
    if adjust_dim
        thres = sum(v)*convert(T,adjust_dim_ratio)
        k = findfirst(v->v > thres, cumsum(v))
        M.d = k === nothing ? d : k
    end
    M.μ = vec(μ)
    M.basis =  U
    return M
end

function origstats(H::Vector{T}) where T<:Real
    N = length(H)

    # calculate threshold
    S = zeros(N,6)

    # recursive defintions
    S[1,1] = H[1]
    S[N-1,2] = H[N]
    S[1,3] = 1.0
    S[N-1,4] = (H[N] == 0 ? 0 : N-1)
    S[1,5] = 0
    S[N-1,6] = 0
    i = 2
    j = N-2
    while i <= N-1
        S[i,1] = S[i-1,1] + H[i]
        if S[i,1] != 0
            S[i,3] = ((S[i-1,3] * S[i-1,1]) + ((i-1) * H[i])) / S[i,1]
            S[i,5] = (S[i-1,1] *
                       (S[i-1,5] + (S[i-1,3]-S[i,3]) * (S[i-1,3]-S[i,3])) +
                        H[i] * ((i-1) - S[i,3]) * ((i-1) - S[i,3]) ) / S[i,1]
        end

        S[j,2] = S[j+1,2] + H[j+1]
        if S[j+1,2] != 0
            S[j,4] = ((S[j+1,4] * S[j+1,2]) + (j * H[j+1])) / S[j,2]
            S[j,6] = (S[j+1,2] *
                       (S[j+1,6] + (S[j+1,4]-S[j,4]) * (S[j+1,4]-S[j,4])) +
                        H[j+1] * (j - S[j,4]) * (j - S[j,4]) ) / S[j,2]
        end

        i += 1
        j -= 1
    end

    return S
end

function refstats(H::Vector{T}) where T<:Real
    x = 1:length(H)
	A=cumsum(H)
	B=cumsum(H.*x)
	C=cumsum(H.*x.^2)

	p=A./A[end]
	q=(A[end] .- A)./A[end]

	u=B./A
    v=(B[end] .- B)./(A[end] .- A)
    v[isnan.(v)] .= 0.0

	s2=C./A - u.^2
    t2=(C[end] .- C)./(A[end].-A) - v.^2
    t2[isnan.(t2)] .= 0.0

	return hcat(p, q, u, v, s2, t2)
end

function histstats(H::Vector{T}) where T<:Real
    N = length(H)
    S = zeros(N,6)

    S[1, 1] = H[1]
    S[1, 3] = 1.0
    S[1, 2] = 1.0 - S[1, 1]
    S[N-1, 2] = H[N]
    S[N-1, 4] = N
    j = N
    for i in 2:N
        P = H[i]
        S[i, 1] = S[i-1, 1] + P
        A = S[i-1, 1]
        if (A + H[i]) != 0
            S[i, 3] = (S[i-1, 3]*A + H[i]*i)/(A + H[i])
            S[i, 5] = ((S[i-1, 5] + S[i-1, 3]*S[i-1, 3])*A + H[i]*i*i)/(A + H[i]) - S[i, 3]*S[i, 3]
        end

        S[j-1, 2] = S[j, 2] + H[j]
        ΔA = S[j-1, 2]
        if ΔA != 0
            S[j-1, 4] = (S[j, 4]*S[j, 2] + H[j]*j)/ΔA
            S[j-1, 6] = ((S[j, 6] + S[j, 4]*S[j, 4])*S[j, 2] + H[j]*j*j)/ΔA - S[j-1, 4]*S[j-1, 4]
        end
        j -= 1
    end

	return S
end

function find_global_min(J::Vector{T}, tol::T) where T<:Real
    N = length(J)

    # Mark minima
    M = zeros(Bool,N)
    if N > 1
        prev = J[2] - J[1]
        curr = 0.0
        for i=2:(N-1)
            curr = J[i+1] - J[i]
            M[i] = prev<=0 && curr>=0
            prev=curr
        end
    end
    @logmsg DEBUG_SEPARATION "Local minima" M=M'

    # Special case: flat minimum so separate at the first bin
    all(M[2:end-1]) && length(M)>2 && return (Inf, 1)

    # Find global minima of criterion funtion if exists
    lmin = findfirst(isequal(true),M)
    lmin === nothing && throw(LMCLUSException("No minimum found. Presume unimode histogram."))

    depth = 0.0
    global_min = 0
    if lmin == N
        throw(LMCLUSException("No minimum found, unimode histogram"))
    else
        while lmin < N
            # Detect flat
            rmin = lmin
            while rmin<N && M[rmin]
                rmin += 1
            end
            loc_min=( lmin + rmin - 1 ) >> 1

            # Monotonically ascend to the left
            lheight = loc_min
            while lheight > 1 && J[lheight-1] >= J[lheight]
                lheight -= 1
            end

            # Monotonically ascend to the right
            rheight = loc_min
            while rheight < N && J[rheight] <= J[rheight+1]
                rheight += 1
            end

            # Compute depth
            local_depth = 0
            local_depth = (J[lheight] < J[rheight] ? J[lheight] : J[rheight]) - J[loc_min]

            if local_depth > depth
                depth = local_depth
                global_min = loc_min
            end

            lmin = rmin
            while lmin<N && !M[lmin]
                lmin += 1
            end
        end
    end

    if depth < tol
        throw(LMCLUSException("Separation depth ($depth) beyond the tolerance level ($tol). Presume unimode histogram."))
    end

    depth, global_min
end

"Projection of the data to the manifold or its orthoganal comppliment"
function project(MC::Manifold{T}, X::AbstractMatrix{T}; ocs = true) where T<:Real
    M = outdim(MC)
    N = min(size(X)...)

    @assert hasfullbasis(MC) "$MC must have full basis"
    @assert M < N "Manifold dimension must be less then of the full space"

    # change basis for data
    r = ocs ? ((M+1):N) : (1:M)
    BC = MC.basis[:,r]
    Y = transpose(BC) * (X.-mean(MC))

    return Y
end
