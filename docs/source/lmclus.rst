Linear Manifold CLUStering
==========================

LMCLUS is a linear manifold clustering algorithm which defines cluster through the following model:

.. math::

    x = \mu^{N \times 1} + B^{N \times K} \phi^{K \times 1} +
        \bar{B}^{N \times N-K} \epsilon^{N-K \times 1}

where :math:`N` is a dimension of the dataset, :math:`K` is dimension of the manifold,
:math:`\mu \in \mathbb{R}^N` is a linear manifold translation vector,
:math:`B` is a matrix whose columns are orthonormal vectors that span :math:`\mathbb{R}^K`,
:math:`\bar{B}` is a matrix whose columns span subspace orthogonal to spanned by columns of :math:`B`,
:math:`\phi` is a 0-mean random vector whose entries are i.i.d. from a support of linear manifold,
:math:`\epsilon` is a zero mean random vector with small variance independent of :math:`\phi`.

Clustering
----------

This package implements the *LMCLUS* algorithm in the ``lmclus`` function:

.. function:: lmclus(X, p)

    Performs linear manifold clustering over the given dataset.

    :param X:   The given sample matrix. Each column of ``X`` is a sample.
    :param p:   The clustering parameters as instance of :doc:`LMCLUSParameters </params>`.

    This function returns an array of ``Manifold`` instances, which is defined as follows:

    .. code-block:: julia

        type Manifold
            d::Int                     # Dimension of the manifold
            mu::Vector{Float64}        # Translation vector
            proj::Matrix{Float64}      # Matrix of basis vectors that span manifold
            points::Vector{Int}        # Indexes of points associated with this cluster
            separation::Separation     # Separation parameters
        end

Results
-------

Let ``M`` be an instance of ``Manifold``, ``n`` be the number of observations, and ``d`` be the dimension of the linear manifold cluster.

.. function:: indim(M)

    Get the the linear manifold cluster dimension ``d``, *i.e* the dimension of the subspace.

.. function:: outdim(M)

    Get the number of points in the cluster ``n``, *i.e* the size of the cluster.

.. function:: mean(M)

    Get the translation vector :math:`\mu`, *i.e* the coordinates of the linear subspace origin.

.. function:: projection(M)

    Get the matrix with columns corresponding to orthonormal vectors that span the linear manifold.

.. function:: separation(M)

    Get the instance of :doc:`Separation </separation>` object.

Example
---------

.. code-block:: julia

    using LMCLUS

    # Load test data, remove label column and flip
    X = readdlm(Pkg.dir("LMCLUS", "test", "testData"), ',')[:,1:end-1]'

    # Initialize clustering parameters with
    # maximum dimensionality for clusters.
    # I should be less then original space dimension.
    params = LMCLUSParameters(5)

    # perform clustering
    Ms = lmclus(X, params)

    # pick the first cluster
    M = Ms[1]

    # obtain indexes of points assigned to the cluster
    l = labels(M)

    # obtain the linear manifold cluster translation vector
    mu = mean(M)

    # get basis vectors that span manifold as columns of matrix
    B = projection(M)

    # get separation properties
    S = separation(M)
