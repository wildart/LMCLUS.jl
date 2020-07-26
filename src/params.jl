import Base.show

"LMCLUS algorithm parameters"
mutable struct Parameters
    "Minimum dimension of the cluster"
    min_dim::Int
    "Maximum dimension of the cluster"
    max_dim::Int
    "Nominal number of resulting clusters"
    number_of_clusters::Int
    "Terminate algorithm upon founding specified number of clusters"
    stop_after_cluster::Int
    "Force to search clusters in high subspaces"
    force_max_dim::Bool
    "Fixed number of bins for the distance histogram"
    hist_bin_size::Int
    "Minimum number of bins for the distance histogram"
    min_bin_num::Int
    "Minimum cluster size (or noise size) in order to prevent generation of small clusters"
    min_cluster_size::Int
    "Separation best bound value is used for evaluating a goodness of separation characterized by a discriminability and a depth between modes of a distance histogram."
    best_bound::Float64
    "Sampling error bound determines a minimal number of samples required to correctly identify a linear manifold cluster."
    error_bound::Float64
    "Maximum histogram bin size"
    max_bin_portion::Float64
    "RNG seed"
    random_seed::Int64
    "Sampling heuristic (1-3)"
    sampling_heuristic::Int
    "Sampling factor used in one of sampling heuristics"
    sampling_factor::Float64
    "Enables a sampling for a distance histogram"
    histogram_sampling::Bool
    "Enables an alignment of a manifold cluster basis"
    basis_alignment::Bool
    "Enables a linear manifold cluster dimensionality detection"
    dim_adjustment::Bool
    "Ratio of manifold principal subspace variance"
    dim_adjustment_ratio::Float64
    "Enables the minimum description length heuristic for a complexity validation of a generated cluster"
    mdl::Bool
    "MDL model precision encoding constant"
    mdl_model_precision::Int
    "MDL data precision encoding constant"
    mdl_data_precision::Int
    "Quantization error of a bin size calculation"
    mdl_quant_error::Float64
    "Compression threshold value for discarding candidate clusters"
    mdl_compres_ratio::Float64
    "MDL algorithm"
    mdl_algo::DataType
    "Enable creation of bounded linear manifold clusters"
    bounded_cluster::Bool
    "Separation threshold algorithm"
    sep_algo::Type{<:Thresholding}

    Parameters(max_dim) = new(
        1,        # min_dim
        max_dim,  # max_dim
        10,       # number_of_clusters
        1000,     # stop_after_cluster
        false,    # force_max_dim
        0,        # hist_bin_size
        7,        # min_bin_num
        20,       # min_cluster_size
        1.0,      # best_bound
        0.0001,   # error_bound
        0.1,      # max_bin_portion
        0,        # random_seed
        3,        # sampling_heuristic
        0.01,     # sampling_factor
        false,    # histogram_sampling
        false,    # basis_alignment
        false,    # dim_adjustment
        0.99,     # dim_adjustment_ratio
        false,    # mdl
        32,       # mdl_model_precision
        16,       # mdl_data_precision
        0.001,    # mdl_quant_error
        1.05,     # mdl_compres_ratio
        MDL.OptimalQuant, # mdl_algo
        false,    # bounded_cluster
        Kittler   # sep_algo
    )
end

show(io::IO, p::Parameters) =
    print(io, """Linear Manifold Clustering parameters:
    Min dimension (min_dim): $(p.min_dim)
    Max dimension (max_dim): $(p.max_dim)
    Nominal number of clusters (number_of_clusters): $(p.number_of_clusters)
    Stop searching after number for clusters found (stop_after_cluster): $(p.stop_after_cluster)
    Force algorithm to search in higher dimensions (force_max_dim): $(p.force_max_dim)
    Minimum cluster size (min_cluster_size): $(p.min_cluster_size)
    Best bound (best_bound): $(p.best_bound)
    Error bound (error_bound): $(p.error_bound)
    Sample points for distance histogram (histogram_sampling): $(p.histogram_sampling)
    Histogram bins (hist_bin_size): $(p.hist_bin_size)
    Minimum number of histogram bins (min_bin_num): $(p.min_bin_num)
    Maximum histogram bin size (max_bin_portion): $(p.max_bin_portion)
    Sampling heuristic (sampling_heuristic): $(p.sampling_heuristic)
    Sampling factor (sampling_factor): $(p.sampling_factor)
    Random seed (random_seed): $(p.random_seed) (0 - random seed)
    Manifold cluster basis alignment (basis_alignment): $(p.basis_alignment)
    Manifold dimensionality adjustment (dim_adjustment): $(p.dim_adjustment)
    Ratio of manifold principal subspace variance (dim_adjustment_ratio): $(p.dim_adjustment_ratio)
    Use MDL heuristic (mdl): $(p.mdl)
    MDL model precision encoding (mdl_model_precision): $(p.mdl_model_precision)
    MDL data precision encoding (mdl_data_precision): $(p.mdl_data_precision)
    MDL quantizing error (mdl_quant_error): $(p.mdl_quant_error)
    MDL compression ratio threshold (mdl_compres_ratio): $(p.mdl_compres_ratio)
    MDL algorithm (mdl_algo): $(p.mdl_algo)
    Creation of bounded linear manifold clusters (bounded_cluster): $(p.bounded_cluster)
    Separation algorithm (sep_algo): $(p.sep_algo)""")
