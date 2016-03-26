# Variance of a uniform distribution over interval
univar{T<:AbstractFloat}(interval::Vector{T}) = (1//12)*(interval).^2

# Optimal number of bins given constant C over interval
opt_bins{T<:AbstractFloat}(intervals::Vector{T}, C::T) =
    round(Int, ceil(intervals * exp( (C - sum(intervals))/length(intervals) ), 0 ) )

# Quantizing error
quant_error(intervals, N_k_opt) = sum(univar(intervals./N_k_opt))

# Optimal quantization of the interval
function opt_quant{T<:AbstractFloat}(intervals::Vector{T}, ɛ::T; tot::Int = 1000, tol=1e-6)
    # Clear interval values
    intervals[isnan(intervals)] = eps() # remove nans
    intervals[isinf(intervals)] = 1 # remove nans

    # Setup C bounds
    K = length(intervals)
    C = 0.
    Cmin = 0.
    Cmax = K*log(typemax(Int)./intervals)+sum(log(intervals)) |> minimum

    i = 1
    N_k_opt = ones(Int, K)
    err_opt = Inf
    while i < tot
        C = Cmax - (Cmax - Cmin)/2.
        N_k_opt = opt_bins(intervals, C)
        err_opt = quant_error(intervals, N_k_opt)
        err_diff = err_opt - ɛ^2
        if abs(err_diff) < tol
            break # error is small enough to stop search
        elseif err_diff > 0.
            # error is too big
            Cmin = C
        elseif err_diff < 0.
            # error is too small
            Cmax = C
        end
        i += 1
    end
    return N_k_opt, err_opt, C, i
end

# Multivariate normal distribution entropy
mvd_entropy(n, Σd) = n*(1+log(2π))/2 + (Σd == 0.0 ? 0.0 : log(Σd))

# the description length of the model: L(H)
function model_dl(M::Manifold, X::Matrix, P::Int)
    n = size(X,1) # space dimension
    m = indim(M)  # manifold dimension
    return m == n ? P*n : round(Int, P*(n + m*(n - (m+1)/2.)))
end

# the description length of the dataset encoded with the provided mode: L(D|H)
function data_dl{T<:AbstractFloat}(M::Manifold, X::Matrix{T}, P::Int, dist::Symbol, ɛ::T;
                                   tot::Int = 1000, tol=1e-6)
    D = 0.0
    if dist == :None
        D = P*length(X)
    elseif dist == :Center
        D = entropy_dl(M, X, dist, ɛ, tot=tot, tol=tol)
    else
        D = (P*indim(M) + entropy_dl(M, X, dist, ɛ, tot=tot, tol=tol))*outdim(M)
    end
    return round(Int, ceil(D))
end

# entropy of the orthoganal complement part of the data
function entropy_dl{T<:AbstractFloat}(M::Manifold, X::Matrix{T}, dist::Symbol, ɛ::T;
                                      tot::Int = 1000, tol=1e-6)
    n, l = size(X) # space dimension
    m = indim(M)  # manifold dimension
    Xtr = X .- mean(M)

    E = 0.0
    if dist == :Uniform && m > 0
        E = -float(n)*log(separation(M).threshold)
    elseif dist == :Gaussian && m > 0
        B = projection(M)
        OP = (eye(n) - B*B')*Xtr # points projected to orthoganal complement subspace
        Σ = OP*OP'
        E += mvd_entropy(m, det(Σ))
    elseif dist == :Empirical # for 0D manifold only empirical estimate is avalible
        F = svdfact(Xtr'/sqrt(n))
        r = (m+1):min(n,l)
        ri = 1:length(r)
        BC = F[:V][:,r]
        Y = BC'*Xtr
        Ymin = vec(minimum(Y, 2))
        Ymax = vec(maximum(Y, 2))
        intervals = abs(Ymax - Ymin) #TODO: normalize intervals to unit length
        if ɛ < 1.
            bins, _ = opt_quant(intervals/maximum(intervals), ɛ)
        else
            bins = fill(round(Int,ɛ), length(r))
        end
        for i in ri
            Yb = linspace(Ymin[i], Ymax[i], bins[i]+1)
            h = hist(vec(Y[i,:]),Yb)[2]
            h /= sum(h)
            E += -sum(map(x-> x == 0. ? 0. : x*log2(x), h))
        end
    elseif dist == :OptimalQuant
        F = svdfact(Xtr'/sqrt(n))
        r = (m+1):min(n,l)
        BC = F[:V][:,r]
        Y = BC'*Xtr
        Ymin = vec(minimum(Y, 2))
        Ymax = vec(maximum(Y, 2))
        intervals = abs(Ymax - Ymin)
        bins, sqerr, C = opt_quant(intervals, ɛ, tot=tot, tol=tol)
        E = C/log(2)
    elseif dist == :Center
        for i in 1:outdim(M)
            for p in 1:n
                E += log2(abs(Xtr[p,i]) + 1)
            end
        end
    end
    return E
end

function mdl{T<:AbstractFloat}(M::Manifold, X::Matrix{T};
            Pm::Int = 32, Pd::Int=16, dist::Symbol = :OptimalQuant, ɛ::T = 1e-2,
            tot::Int = 1000, tol=1e-6)
    return model_dl(M, X, Pm) + data_dl(M, X, Pd, dist, ɛ, tot=tot, tol=tol)
end

function raw(M::Manifold, Pm::Int)
    return Pm*outdim(M)*length(mean(M))
end