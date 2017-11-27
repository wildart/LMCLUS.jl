"""Empty manifold"""
emptymanifold(N::Int, points::Vector{Int}=Int[]) = Manifold(N, zeros(N), eye(N,N), points, 0.0, 0.0)

"""Returns seed from parameters"""
getseed(params::LMCLUS.Parameters) = params.random_seed == 0 ? time_ns() : params.random_seed

"""
Sample uniformly k integers from the integer range 1:n, making sure that
the same integer is not sampled twice. the function returns an integer vector
containing the sampled integers.
"""
function randperm2(n, k)
    if n < k
        error("Sample size cannot be grater then range")
    end
    sample = Set()
    sample_size = 0
    while sample_size < k
        selected = rand(1:n, k-sample_size)
        union!(sample, selected)
        sample_size = length(sample)
    end
    collect(sample)
end

"""
Sample randomly lm_dim+1 points from the dataset, making sure
that the same point is not sampled twice. the function will return
a index vector, specifying the index of the sampled points.
"""
function sample_points(X::Matrix{T}, n::Int) where T <: Real
    if n <= 0
        error("Sample size must be positive")
    end
    data_rows, data_cols = size(X)
    I = Int[]
    hashes = Set{UInt}()
    hashes_size = 0

    # get sample indexes
    J = randperm2(data_cols, n-size(I,1))
    for idx in J
        h = hash(X[:, idx])
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
    if length(I) != n
        # Resample from the rest of data
        for idx in setdiff(randperm(data_cols),I)
            h = hash(X[:, idx])
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
    if length(I) < n
        warn("Dataset doesn't have enough unique points, decrease number of points")
        I = Int[]
    end
    return I
end

"""Reservoir sampling"""
function sample_points(X::Matrix{T}, k::Int, r::MersenneTwister) where T <: Real
    N, n = size(X)
    if n < k
        warn("Not enough samples to construct manifold")
        return Int[]
    end

    I = collect(1:k)
    for i in (k+1):n
        j = trunc(Int, rand(r)*i)+1
        if j <= k
            I[j] = i
        end
    end

    for i in 1:(k-1)
        for j in (i+1):k
            if all([X[k,I[i]] == X[k,I[j]] for k in 1:N])
                warn("Sample is not unique: X[:,$(I[i])] == X[:,$(I[j])]")
                return Int[]
            end
        end
    end

    return I
end

"""Modified Gram-Schmidt orthogonalization algorithm"""
function orthogonalize(vecs::AbstractMatrix{T}) where {T<:Real}
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
function form_basis(X::AbstractMatrix, sample::Vector{Int})
    origin = X[:,sample[1]]
    basis = X[:,sample[2:end]] .- origin
    vec(origin), orthogonalize(basis)
end

"Generate points-to-cluster assignments identifiers"
function assignments(Ms::Vector{Manifold})
    lbls = zeros(Int, sum(map(m->outdim(m), Ms)))
    for (i,m) in enumerate(Ms)
        lbls[labels(m)] = i
    end
    return lbls
end

"Projection of the data to the manifold"
function project(m::Manifold, X::Matrix{T}) where T <: Real
    proj = projection(m)'*(X.-mean(m))
    return proj
end

function filter_separeted(selected_points, X, O, B, S)
    cluster_points = Int[]
    removed_points  = Int[]
    θ = threshold(S)

    for i in eachindex(selected_points)
        idx = selected_points[i]
        # point i has distances less than the threshold value
        d = distance_to_manifold(X[:, idx], O, B)
        if d < θ
            push!(cluster_points, idx)
        else
            push!(removed_points, idx)
        end
    end

    return cluster_points, removed_points
end

function adjustbasis!(M::Manifold, X::AbstractMatrix;
                      adjust_dim::Bool=false, adjust_dim_ratio::Float64=0.99)
    R = if indim(M) > 0 && !adjust_dim
        fit(PCA, X[:, labels(M)]; maxoutdim=indim(M))
    else
        fit(PCA, X[:, labels(M)]; pratio = adjust_dim_ratio)
    end
    if adjust_dim
        M.d = MultivariateStats.outdim(R)
    end
    M.μ = MultivariateStats.mean(R)
    M.proj = MultivariateStats.projection(R)
    return M
end

check_separation(sep::Separation, params::Parameters) = criteria(sep) < params.best_bound

function log_separation(sep::Separation, params::Parameters)
    LOG(params, 4, "separation: width=", sep.discriminability, "  depth=", sep.depth)
    LOG(params, 4, "  criteria: $(criteria(sep)) (best bound=$(params.best_bound))")
    LOG(params, 4, " threshold: $(threshold(sep))")
end
