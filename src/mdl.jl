# Variance of a uniform distribution over interval
univar{T<:FloatingPoint}(interval::Vector{T}) = (1//12)*(interval).^2

# Optimal number of bins given constant C over interval
opt_bins{T<:FloatingPoint}(interval::Vector{T}, C::T) = int( interval * exp( (C - sum(interval))/length(interval) ) )

# Optimal quantization of the interval
function opt_quant{T<:FloatingPoint}(interval::Vector{T}, ɛ::T; tot::Int = 1000, α::T = 0.5)
    C = 1.0
    i = 1
    N_k_opt = opt_bins(interval, C)
    err_opt = sum(univar(interval./N_k_opt))
    while err_opt > ɛ && i < tot
        C += α
        i += 1
        N_k_opt = opt_bins(interval, C)
        err_opt = sum(univar(interval./N_k_opt))
    end
    return N_k_opt, err_opt, C, i
end

# Multivariate normal distribution entropy
mvd_entropy(n, Σd) = n*(1+log(2π))/2 + (Σd == 0.0 ? 0.0 : log(Σd))

# the description length of the model: L(H)
function model_dl(M::Manifold, X::Matrix, P::Int)
    n = size(X,1) # space dimension
    m = indim(M)  # manifold dimension
    return m == 0 ? P*n : P*(n + m*(n - (m+1)/2.))
end

# the description length of the dataset encoded with the provided mode: L(D|H)
function data_dl{T<:FloatingPoint}(M::Manifold, X::Matrix{T}, P::Int, dist::Symbol, ɛ::T)
    D = 0.0
    if dist == :None
        D = P*(length(X)-1)
    elseif dist == :Center
        D = entropy_dl(M, X, dist, ɛ)
    else
        D = (P*indim(M) + entropy_dl(M, X, dist, ɛ))*outdim(M)
    end
    return D
end

# entropy of the orthoganal complement part of the data
function entropy_dl{T<:FloatingPoint}(M::Manifold, X::Matrix{T}, dist::Symbol, ɛ::T)
    n = size(X,1) # space dimension
    m = indim(M)  # manifold dimension

    E = 0.0
    if dist == :Uniform && m > 0
        E = -float(n)*log(separation(M).threshold)
    elseif dist == :Gaussian && m > 0
        B = projection(M)
        Xtr = X .- mean(M)
        OP = (eye(n) - B*B')*Xtr # points projected to orthoganal complement subspace
        Σ = OP*OP'
        E += mvd_entropy(m, det(Σ))
    elseif dist == :Empirical # for 0D manifold only empirical estimate is avalible
        Xtr = X .- mean(M)
        F = svdfact(Xtr)
        r = (m+1):n
        ri = 1:length(r)
        BC = F[:U][:,r]
        Y = BC*BC'*Xtr
        Ymin = minimum(Y, 2)
        Ymax = maximum(Y, 2)
        intervals = 2*max(abs(Ymin), abs(Ymax))[:]
        if ɛ < 1.
            bins, _ = opt_quant(intervals[ri], ɛ)
        else
            bins = fill(int(ɛ), length(r))
        end
        for i in ri
            Yb = linspace(Ymin[i], Ymax[i], bins[i]+1)
            h = hist(vec(Y[i,:]),Yb)[2]
            h /= sum(h)
            E += -sum(map(x-> x == 0. ? 0. : x*log2(x), h))
        end
    elseif dist == :Center
        Xtr = X .- mean(M)
        for i in 1:outdim(M)
            for p in 1:n
                E += log2(abs(Xtr[p,i]) + 1)
            end
        end
    end
    return E
end

function MDLength{T<:FloatingPoint}(M::Manifold, X::Matrix{T};
            P::Int = 32, dist::Symbol = :Gaussian, ɛ::T = 1e-3)
    return model_dl(M, X, P) + data_dl(M, X, P, dist, ɛ)
end