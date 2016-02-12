Separations
===========

When linear manifold is formed, a distance from every point of dataset to the manifold is calculated, and a histograms of point distances to each trial manifold are computed.
If the resulting histogram contains multiple modes then the mode near zero is isolated in histogram [#R1]_. The isolated part of histogram is used to determine a separation criteria, and the data points are partitioned from the rest of the dataset on the basis of such separation.

The separation properties defined in ``Separation`` instance, which is defined as follows:

    .. code-block:: julia

        type Separation
            depth::Float64              # Separation depth (depth between separated histogram modes)
            discriminability::Float64   # Separation discriminability (width between separated histogram modes)
            threshold::Float64          # Distance threshold value
            globalmin::Int              # Global minimum as histogram bin index
            hist_range::Vector{Float64} # Histogram ranges
            hist_count::Vector{UInt32}  # Histogram counts
            bin_index::Vector{UInt32}   # Point to bin assignments
        end

Separation criteria and distance threshold value can be accessed through following functions:

.. function:: criteria(S)

    Returns separation criteria value which is product of depth and discriminability.

.. function:: threshold(S)

    Returns distance threshold value for separation calculated on histogram of distances. It is used to determine which points belong to formed cluster.

.. rubric:: References
.. [#R1] J. Kittler & J. Illingworth: "Minimum Error Thresholding", Pattern Recognition, Vol 19, nr 1. 1986, pp. 41-47, DOI:`10.1016/0031-3203(86)90030-0 <http://dx.doi.org/10.1016/0031-3203(86)90030-0>`_


