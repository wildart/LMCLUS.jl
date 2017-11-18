function form_basis_svd(X::Matrix{T}) where {T<:Real}
    n = size(X,1)
    origin = mean(X,2)
    vec(origin), svdfact((X.-origin)'/sqrt(n))[:V][:,1:end-1]
end

"Fast histogram calculation"
function histogram(V::Vector{T}, edgs) where T <: Real
    n = length(edgs)-1
    counts = zeros(Int32,n)
    cindex = zeros(UInt32,length(V))
    @inbounds for i in 1:length(V)
        x = V[i]
        lo = 0
        hi = n+2
        while lo < hi-1
            m = (lo+hi)>>>1
            if edgs[m] < x
                lo = m
            else
                hi = m
            end
        end
        if lo > 0
            hi -= 1
            counts[hi] += 1
            cindex[i] = hi
        end
    end
    return edgs, counts, cindex
end

"""
V-measure of contingency table
Accepts contingency table with row as classes, c, and columns as clusters, k.
"""
function V_measure(A::Matrix; β = 1.0)
    C, K = size(A)
    N = sum(A)

    # Homogeneity
    hck = 0.0
    for k in 1:K
        d = sum(A[:,k])
        for c in 1:C
            if A[c,k] != 0 && d != 0
                hck += log(A[c,k]/d) * A[c,k]/N
            end
        end
    end
    hck = -hck

    hc = 0.0
    for c in 1:C
        n = sum(A[c,:]) / N
        if n != 0.0
            hc += log(n) * n
        end
    end
    hc = -hc

    h = (hc == 0.0 || hck == 0.0) ? 1 : 1 - hck/hc

    # Completeness
    hkc = 0.0
    for c in 1:C
        d = sum(A[c,:])
        for k in 1:K
            if A[c,k] != 0 && d != 0
                hkc += log(A[c,k]/d) * A[c,k]/N
            end
        end
    end
    hkc = -hkc

    hk = 0.0
    for k in 1:K
        n = sum(A[:,k]) / N
        if n != 0.0
            hc += log(n) * n
        end
    end
    hk = -hk

    c = (hk == 0.0 || hkc == 0.0) ? 1 : 1 - hkc/hk

    # V-measure
    V_β = (1 + β)*h*c/(β*h + c)
    return V_β
end
