import Base.show

type LMCLUSParameters
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

    LMCLUSParameters(dims) = new(dims, 100, 0, 20, 1.0, 0.0001, 0.1, 0, 3, 0.01, false, true)
end


show(io::IO, p::LMCLUSParameters) =
    print(io, """Linear Manifold Clustering parameters:
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
    """)
