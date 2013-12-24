module LMCLUS

export  lmclus,
        LMCLUSParameters

type LMCLUSParameters
    max_dim::Int
    clust_num::Int
    hist_bin_size::Int
    noise_size::Int
    best_bound::Float64
    error_bound::Float64
    max_bin_portion::Float64
    random_seed::Int64
    heuristic::Int
    sampling_factor::Float64
    histogram_sampling::Bool
end

# Linear Manifold Clustering
function lmclus(data::Array{Float64,2}, params::LMCLUSParameters)
    # Setup RNG
    if params.random_seed == 0
        srand(time_ns())
    else
        srand(params.random_seed)
    end
    
    data_rows, data_cols = size(data)
    data_index = iround(linspace(1, data_rows, data_rows))
    ClusterNum = 0
    
    # Main loop through dataset
    while length(data_index) > params.noise_size
        nonClusterPoints, separations, Noise, SepDim = find_manifold(ds, params, data_index)
        
    end
    
    ndims(ds)
end

function find_manifold(data::Array{Float64,2}, params::LMCLUSParameters, index::Array{Int64,1})
    Noise = false
    nonClusterPoints = Array(Int, 0)
    SepDim = 0  # dimension in which separation was found
    
    for lm_dim = 1:params.max_dim+1
        if Noise
            break
        end
        
        while true
            best_sep = find_best_separation(data, params, index, lm_dim)
            info("BEST_BOUND: )";
        end
    
    end
    
    Noise, nonClusterPoints, SepDim
end

function find_best_separation(data::Array{Float64,2}, params::LMCLUSParameters, index::Array{Int64,1}, lm_dim::Int)
    data_size, full_space_dim = size(data)

    info("data size=", data_size,"   linear manifold dim=",lm_dim,"   space dim=",full_space_dim,"   searching for separation ...")
    info("---------------------------------------------------------------------------------")

end

# Modified Gram-Schmidt Algorithm
function gramSchmidtOrthogonalization()
    
end

end # module
