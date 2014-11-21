Separations
===========

When linear manifold is formed, a distance from every point of dataset to the manifold is calculated.
Histograms of the distances of the points to each trial manifold are computed.
The histogram having the best separation between a mode near zero and the rest is selected and the data points are partitioned on the basis of the best separation.

The separation properties defined in ``Separation`` instance, which is defined as follows:

    .. code-block:: julia

        type Separation
            depth::Float64              # Separation depth (depth between separated histogram modes)
            discriminability::Float64   # Separation discriminability (width between separated histogram modes)
            threshold::Float64          # Distance threshold value
            globalmin::Int              # Global minimum as histogram bin index
            hist_range::Vector{Float64} # Histogram ranges
            hist_count::Vector{Int}     # Histogram counts
        end

Separation criteria and distance threshold value can be accessed through following functions:

.. function:: criteria(S)

    Returns separation criteria value which is product of depth and discriminability.

.. function:: threshold(S)

    Returns distance threshold value for separation calculated on histogram of distances. It is used to determine which points belong to formed cluster.
