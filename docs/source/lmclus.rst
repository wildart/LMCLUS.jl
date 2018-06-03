Linear Manifold Clustering
==========================

Linear manifold clustering algorithm (LMCLUS) discovers clusters which are described by a following model:

.. math::

    x = \mu^{N \times 1} + B^{N \times K} \phi^{K \times 1} +
        \bar{B}^{N \times N-K} \epsilon^{N-K \times 1}

where :math:`N` is a dimension of the dataset, :math:`K` is dimension of the manifold,
:math:`\mu \in \mathbb{R}^N` is a linear manifold translation vector,
:math:`B` is a matrix whose columns are orthonormal vectors that span :math:`\mathbb{R}^K`,
:math:`\bar{B}` is a matrix whose columns span subspace orthogonal to spanned by columns of :math:`B`,
:math:`\phi` is a zero-mean random vector whose entries are i.i.d. from a support of linear manifold,
:math:`\epsilon` is a zero-mean random vector with small variance independent of :math:`\phi`.

Clustering
----------

This package implements the *LMCLUS* algorithm in the ``lmclus`` function:

.. function:: lmclus(X, p)

    Performs linear manifold clustering over the given dataset.

    :param X:   The given sample matrix. Each column of ``X`` is a sample.
    :param p:   The clustering parameters as instance of :doc:`LMCLUSParameters </params>`.

    This function returns an ``LMCLUSResult`` instance.

Results
-------

Let ``M`` be an instance of ``Manifold``, ``n`` be the number of observations, and ``d`` be the dimension of the linear manifold cluster.

.. function:: indim(M)

    Returns a dimension of the observation space.

.. function:: outdim(M)

    Returns a dimension of the linear manifold cluster which is the dimension of the subspace.

.. function:: size(M)

    Returns the number of points in the cluster which is the size of the cluster.

.. function:: points(M)

    Returns indexes of points assigned to the cluster.

.. function:: mean(M)

    Returns the translation vector :math:`\mu` which contains coordinates of the linear manifold origin.

.. function:: projection(M)

    Returns the basis matrix with columns corresponding to orthonormal vectors that span the linear manifold."

.. function:: separation(M)

    Returns the instance of :doc:`Separation </separation>` object.

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

    # perform clustering and returns a collection of clusters
    clust = lmclus(X, params)

    # pick the first cluster
    M = manifold(clust, 1)

    # obtain indexes of points assigned to the cluster
    l = points(M)

    # obtain the linear manifold cluster translation vector
    mu = mean(M)

    # get basis vectors that span manifold as columns of the returned matrix
    B = projection(M)

    # get separation properties
    S = separation(M)
