# PERFORMS KITTLER'S MINIMAL THRESHOLDING ALGORITH
# REFERENCES:   J. Kittler & J. Illingworth: "Minimum Error Thresholding" 
#                Pattern Recognition, Vol 19, nr 1. 1986, pp. 41-47.
function kittler(xs::Vector{Float64}; bins=20, tol=1.0e-5)
    # find maximum and minimum
    maxX = maximum(xs)
    minX = minimum(xs)

    # get normalized histogram
    r = linspace(minX,maxX,bins)
    r, c = hist(xs, r)
    H = c/sum(c)
    N = length(H)

    # calculate threshold
    P1 = zeros(Float64, N)
    P2 = zeros(Float64, N)
    μ1 = zeros(Float64, N)
    μ2 = zeros(Float64, N)
    Var1 = zeros(Float64, N)
    Var2 = zeros(Float64, N)

    # recursive defintions
    P1[1] = H[1]
    P2[N] = H[N]
    i = 2
    j = N-1
    while i < N
        P1[i] = P1[i-1] + H[i]
        if P1[i] != 0
            μ1[i] = ( (μ1[i-1] * P1[i-1]) + (i * H[i]) ) / P1[i]
            Var1[i]= ( P1[i-1] * ( Var1[i-1] + (μ1[i-1]-μ1[i]) * (μ1[i-1]-μ1[i]) ) 
                + H[i] * (i - μ1[i]) * (i - μ1[i]) ) / P1[i]
        end

        P2[j] = P2[j+1] + H[j+1]
        if P2[j] != 0
            μ2[j] = ( (μ2[j+1] * P2[j+1]) + ((j+1) * H[j+1]) ) / P2[j];
            Var2[j]= ( P2[j+1] * ( Var2[j+1] + (μ2[j+1]-μ2[j]) * (μ2[j+1]-μ2[j]) ) 
                + H[j+1] * (j+1 - μ2[j]) * (j+1 - μ2[j]) ) / P2[j]
        end
        
        i += 1
        j -= 1
    end
    
    #r1(T)=[1:T]
    #r2(T)=[T+1:N]
    #for T=1:N
    #    P1[T] = sum(H[r1(T)])
    #    if P1[T] != 0.0
    #        μ1[T] = sum( H[r1(T)].*r1(T) ) / P1[T]
    #        Var1[T] = sum( ((r1(T)-μ1[T]).^2).*H[r1(T)] ) / P1[T]
    #    end
    #    P1
    #    P2[T] = sum(H[r2(T)])
    #    if P2[T] != 0.0
    #        μ2[T] = sum( H[r2(T)].*r2(T) ) / P2[T]
    #        Var2[T] = sum( ((r2(T)-μ2[T]).^2).*H[r2(T)] ) / P2[T]
    #    end
    #end

    # Compute criterion function
    J = fill(-Inf, N)
    for T=1:N
        if P1[T]!=0 && P2[T]!=0
            J[T] = 1 + 2*( P1[T]*log(sqrt(Var1[T])) + P2[T]*log(sqrt(Var2[T]))) - 2*(P1[T]*log(P1[T]) + P2[T]*log(P2[T]))
        end
    end
    
    # Global minimum parameters
    depth, global_min = find_global_min(J, tol)
    min_index = int(global_min)
    threshold = minX + ( (global_min+1) * (maxX - minX) / N )
    #threshold2 = (r[min_index] + r[min_index+1])/2 # or from histogram bins' edges
    discriminability = (abs(μ1[min_index]-μ2[min_index]))/(sqrt(Var1[min_index]+Var2[min_index]))

    depth, discriminability, threshold, min_index, r, c#, threshold2
end

function find_global_min(J::Vector{Float64}, tol)
    N = length(J)
    
    # Mark minima
    M = zeros(Bool,N)
    if N-1 >= 1
        prev = J[2] - J[1]
        curr = 0.0
        for i=2:(N-1)
            curr = J[i+1] - J[i]
            M[i] = prev<=0 && curr>=0
            prev=curr
        end
    end
    
    # Find global minima of criterion funtion if exists
    # find first minimum
    lmin = 1
    while lmin<N && !M[lmin]
        lmin += 1
    end

    depth = 0
    global_min = 0
    if lmin == N
        error("No minimum found, unimode histogram")
    else
        while lmin < N
            # Dedect flat
            rmin = lmin
            while rmin<N && M[rmin]
                rmin += 1
            end
            loc_min=( lmin + rmin - 1 ) / 2
            
            # Monotonically ascend to the left
            lheight = int(loc_min)
            while lheight > 1 && J[lheight-1] >= J[lheight] 
                lheight -= 1
            end
            
            # Monotonically ascend to the right
            rheight = int(loc_min)
            while rheight < N && J[rheight] <= J[rheight+1] 
                rheight += 1
            end
            
            # Compute depth
            local_depth = 0
            local_depth = (J[lheight] < J[rheight] ? J[lheight] : J[rheight]) - J[int(loc_min)] 
            
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
        error("No minimum found, unimode histogram")
    end
    
    depth, global_min
end
