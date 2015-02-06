function MDLength(M::Manifold, X::Matrix; P::Float64 = 32.0, T::Symbol = :Gausian, bins::Int = 20)
    n = size(X,1)
    m = indim(M)
    L = m == 0 ? P*n : P*(m*n - (m-1)*(m-2)/2.)

    gusent(n, Σd) = n*(1+log(2π))/2 + (Σd == 0.0 ? 0.0 : log(Σd))

    E = 0.0
    if T == :Uniform && m > 0
        E = -float(n)*log(separation(M).threshold)
    elseif T == :Gausian && m > 0
        B = projection(M)
        Xtr = X .- mean(M)
        OP = B*B'*Xtr # points projected to manifold subspace
        Σ = OP*OP'
        E += gusent(m, det(Σ)) * outdim(M)
        OP = (eye(n) - B*B')*Xtr # points projected to orthoganal complement subspace
        Σ = OP*OP'
        E += gusent(m, det(Σ)) * outdim(M)
    elseif T == :Empirical # for 0D manifold only empirical estimate is avalible
        Xtr = X .- mean(M)
        F = svdfact(Xtr)
        idx = [1 m; m+1 n]
        for s in 1:2
            r = idx[s,1]:idx[s,2]
            BC = F[:U][:,r]
            Y = BC*BC'*Xtr
            for i in 1:length(r)
                C = vec(Y[i,:])
                Cmin, Cmax = extrema(C)
                Cx = linspace(Cmin, Cmax, bins)
                h = hist(C,Cx)[2]
                h /= sum(h)
                E += -sum(map(x-> x == 0. ? 0. : x*log2(x), h)) * outdim(M)
            end
        end
    elseif T == :Center
        Xtr = X .- mean(M)
        for i in 1:outdim(M)
            for p in 1:n
                E += log2(abs(Xtr[p,i]) + 1)
            end
        end
    else
        E = outdim(M)*n
    end
    #warn("H:$(L) + D:$(E)")
    return L+E
end