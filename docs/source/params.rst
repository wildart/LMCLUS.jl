Parameters
==========

The clustering properties set in ``LMCLUSParameters`` instance, which is defined as follows:

    .. code-block:: julia

        type LMCLUSParameters
            min_dim::Int                     # Min dimension
            max_dim::Int                     # Max dimension
            cluster_number::Int              # Number of clusters
            hist_bin_size::Int               # Histogram bins
            noise_size::Int                  # Noise size
            best_bound::Float64              # Best bound
            error_bound::Float64             # Error bound
            max_bin_portion::Float64         # Maximum histogram bin size
            random_seed::Int64               # Random seed
            sampling_heuristic::Int          # Sampling heuristic
            sampling_factor::Float64         # Sampling factor
            histogram_sampling::Bool         # Sample points for distance histogram
            zero_d_search::Bool              # Enable 0D manifold search
            basis_alignment::Bool            # Manifold cluster basis alignment
            dim_adjustment::Bool             # Manifold dimensionality adjustment
            dim_adjustment_ratio::Float64    # Ratio of manifold principal subspace variance
            mdl_heuristic::Bool              # Enable MDL heuristic
            mdl_precision::Float64           # MDL encoding parameter
            mdl_quant_error::Float64         # Quantization error
            log_level::Int                   # Log level (0-5)
        end

Here is a description of algorithm parameters and their default values:

====================  ===============================================================  ===============
  name                 description                                                      default
====================  ===============================================================  ===============
min_dim               Low bound of a cluster manifold dimension.                       ``1``
--------------------  ---------------------------------------------------------------  ---------------
max_dim               High bound of a cluster manifold dimension.
                      *It cannot be larger then a dimensionality of a dataset.*
--------------------  ---------------------------------------------------------------  ---------------
cluster_number        Expected number of clusters.                                     ``10``
                      *Requred for the sampling heuristics.*
--------------------  ---------------------------------------------------------------  ---------------
hist_bin_size         Number of bins for a distance histogram.                         ``0``
                      *If this parameter is set to zero, the number of bins in
                      the distance histogram determined by parameter*
                      ``max_bin_portion``.
--------------------  ---------------------------------------------------------------  ---------------
noise_size            Minimum size of a collection of data points to be considered as  ``20``
                      a proper cluster.
--------------------  ---------------------------------------------------------------  ---------------
best_bound            Separation best bound value is used for evaluating a goodness    ``1.0``
                      of separation characterized by a discriminability and a depth
                      between modes of a distance histogram.
--------------------  ---------------------------------------------------------------  ---------------
error_bound           Sampling error bound determines a minimal number of samples      ``1e-4``
                      required to correctly identify a linear manifold cluster.
--------------------  ---------------------------------------------------------------  ---------------
max_bin_portion       Sampling error bound determines a minimal number of samples      ``0.1``
                      required to correctly identify a linear manifold cluster.
                      *Value should be selected from a (0,1) range.*
--------------------  ---------------------------------------------------------------  ---------------
random_seed           Random number generator seed.                                    ``0``
                      *If not specified then RNG will be reinitialized at every run.*
--------------------  ---------------------------------------------------------------  ---------------
sampling_heuristic    The choice of heuristic method:                                  ``3``

                      1) algorithm will use a probabilistic heuristic which will
                         sample a quantity exponential in ``max_dim`` and
                         ``cluster_number`` parameters

                      2) will sample fixed number of points

                      3) the lesser of the previous two

--------------------  ---------------------------------------------------------------  ---------------
sampling_factor       Sampling factor used in the sampling heuristics                  ``0.01``
                      (see above, options 2 & 3) to determine a number of samples
                      as a percentage from a total dataset size.
--------------------  ---------------------------------------------------------------  ---------------
histogram_sampling    Turns on a sampling for a distance histogram.                    ``false``
                      Instead of computing the distance histogram from
                      a whole dataset, the algorithm draws a small sample for
                      the histogram construction, thus improving a its performance.
                      This parameter depends on a ``cluster_number`` value.
--------------------  ---------------------------------------------------------------  ---------------
zero_d_search         Turn on/off zero dimensional manifold search.                    ``false``
--------------------  ---------------------------------------------------------------  ---------------
basis_alignment       Turn of/off an alignment of a manifold cluster basis.            ``false``
                      *If it's on, a manifold basis of the generated cluster is
                      aligned along the direction of the maximum variance
                      (by performing PCA).
--------------------  ---------------------------------------------------------------  ---------------
dim_adjustment        Turn of/off a linear manifold cluster dimensionality detection   ``false``
                      by looking for portion of a variance associated with
                      principal components.
--------------------  ---------------------------------------------------------------  ---------------
dim_adjustment_ratio  Ratio of manifold principal subspace variance.                   ``0.99``
--------------------  ---------------------------------------------------------------  ---------------
mdl                   Turn on/off minimum description length heuristic for             ``false``
                      a complexity validation of a generated cluster.
--------------------  ---------------------------------------------------------------  ---------------
mdl_precision         Precision encoding value.                                        ``16``
--------------------  ---------------------------------------------------------------  ---------------
mdl_quant_error       Quantization error of a bin size calculation for a histogram     ``1e-4``
                      which used in determining entropy value of
                      the empirical distance distribution.
--------------------  ---------------------------------------------------------------  ---------------
log_level             Logging level (ranges from 0 to 5).                      ``0``
====================  ===============================================================  ===============

Suggestions
-----------
Particular settings could impact performance of the algorithm:

- If you want a persistent clustering results fix a ``random_seed`` parameter.
  By default, RNG is reinitialized every time when algorithm runs.

- If a dimensionality of the data is low, a histogram sampling could speeding up calculations.

- Value ``1`` of ``sampling_heuristic`` parameter should not be used if parameter ``max_dim`` is large,
  as it will generate a very large number of samples.

- Increasing value of ``max_bin_portion`` parameter could improve an efficiency of
  the clustering partitioning, but as well could degrade overall performance of the algorithm.


Parallelization
---------------
This implementation of LMCLUS algorithm uses parallel computations during a manifold sampling stage.
You need add additional workers before executing the algorithm.