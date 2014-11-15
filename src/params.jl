import Base.show

type LMCLUSParameters
    min_dim::Int
    max_dim::Int
    cluster_number::Int
    hist_bin_size::Int
    noise_size::Int
    best_bound::Float64
    error_bound::Float64
    max_bin_portion::Float64
    random_seed::Int64
    heuristic::Int
    sampling_factor::Float64
    histogram_sampling::Bool
    zero_d_search::Bool
    basis_alignment::Bool
    dim_adjustment::Bool
    dim_adjustment_ratio::Float64
    mdl_heuristic::Bool
    mdl_coding_value::Float64
    log_level::Int

    LMCLUSParameters(max_dim) =
        new(1, max_dim, 100, 0, 20, 1., 0.0001, 0.1, 0, 3, 0.01, false, false, false, false, 0.99, false, 32., 0)
end

show(io::IO, p::LMCLUSParameters) =
    print(io, """Linear Manifold Clustering parameters:
    Min dimension (min_dim): $(p.min_dim)
    Max dimension (max_dim): $(p.max_dim)
    Number of clusters (cluster_number): $(p.cluster_number)
    Noise size (noise_size): $(p.noise_size)
    Best bound (best_bound): $(p.best_bound)
    Error bound (error_bound): $(p.error_bound)
    Sample points for distance histogram (histogram_sampling): $(p.histogram_sampling)
    Histogram bins (hist_bin_size): $(p.hist_bin_size)
    Maximum histogram bin size (max_bin_portion): $(p.max_bin_portion)
    Sampling heuristic (heuristic): $(p.heuristic)
    Sampling factor (sampling_factor): $(p.sampling_factor)
    Random seed (random_seed): $(p.random_seed) (0 - random seed)
    0D manifold search (zero_d_search): $(p.zero_d_search)
    Manifold cluster basis alignment (basis_alignment): $(p.basis_alignment)
    Manifold dimensionality adjustment (dim_adjustment): $(p.dim_adjustment)
    Ratio of manifold principal subspace variance (dim_adjustment_ratio): $(p.dim_adjustment_ratio)
    Use MDL heuristic (mdl_heuristic): $(p.mdl_heuristic)
    MDL encoding parameter (mdl_heuristic): $(p.mdl_coding_value)
    Log level (log_level): $(p.log_level)""")

# Logger
# RED (31): Error (1)
# BLUE (34): Info (2)
# DEBUG (32): Debug (3)
# DEV (36): Development (4)
# Trace (33): Trace (5)
function LOG(p::LMCLUSParameters, lvl, msg...)
    if p.log_level < lvl
        return
    else
        #prefix = lvl == 1 ? "\e[1;34mINFO" : (lvl == 2 ? "\e[1;32mDEBUG" : "\e[1;33mTRACE")
        #println(prefix, ": ", msg..., "\e[0m")
        prefix = lvl == 1 ? "\e[1;31m" : (lvl == 2 ? "\e[1;34m" :
            (lvl == 3 ? "\e[1;32m" : (lvl == 4 ? "\e[1;36m" : "\e[1;33m")))
        println(prefix, msg..., "\e[0m")
    end
end