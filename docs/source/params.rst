Parameters
==========

The clustering properties set in ``LMCLUSParameters`` instance, which is defined as follows:

    .. code-block:: julia

        type LMCLUSParameters
            min_dim::Int                     # Min dimension
            max_dim::Int                     # Max dimension
            cluster_number::Int              # Number of clusters
            noise_size::Int                  # Noise size
            best_bound::Float64              # Best bound
            error_bound::Float64             # Error bound
            hist_bin_size::Int               # Histogram bins
            max_bin_portion::Float64         # Maximum histogram bin size
            random_seed::Int64               # Random seed
            heuristic::Int                   # Sampling heuristic
            sampling_factor::Float64         # Sampling factor
            histogram_sampling::Bool         # Sample points for distance histogram
            zero_d_search::Bool              # Enable 0D manifold search
            basis_alignment::Bool            # Manifold cluster basis alignment
            dim_adjustment::Bool             # Manifold dimensionality adjustment
            dim_adjustment_ratio::Float64    # Ratio of manifold principal subspace variance
            mdl_heuristic::Bool              # Enable MDL heuristic
            mdl_coding_value::Float64        # MDL encoding parameter
            log_level::Int                   # Log level (0-5)
        end