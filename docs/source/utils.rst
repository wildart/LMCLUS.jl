Utilities
=========

Linear Manifold Clustering Algorithm relies on multiple search and optimization methods:

.. function:: kittler(X, bins, tol)

    A minimum error thresholding method for multimodal histograms [#R1]_.

.. function:: otsu(X, bins)

    A gray-level thresholding method for multimodal histograms [#R2]_.

.. function:: mdl(M, X; Pm = 32, Pd = 16, T = :Empirical, ɛ = 1e-4)

    Performs calculation of the minimum description length for the linear manifold cluster.

    :param M:   Linear manifold cluster description as ``Manifold`` type instance.
    :param X:   Linear manifold cluster data as ``Matrix`` with points as its columns.
    :param Pm:  Precision encoding constant for the model, i.e. number of bits
                required for encoding on element of the model description.
                Default value is 32 which corresponds to ``Float32``.
    :param Pd:  Precision encoding constant for the data.
    :param T:   Type of a dataset encoding model as symbol:
                ``:Gausian``, ``:Uniform``, ``:Empirical``.
    :param ɛ:   Error tolerance for bin quantization used in an empirical model encoding

    Returns number of bits required to encode linear manifold cluster with the MDL schema.

.. rubric:: References
.. [#R1] J. Kittler & J. Illingworth: "Minimum Error Thresholding", Pattern Recognition, Vol 19, nr 1. 1986, pp. 41-47, DOI:`10.1016/0031-3203(86)90030-0 <http://dx.doi.org/10.1016/0031-3203(86)90030-0>`_
.. [#R2] N. Otsu: "A threshold selection method from gray-level histograms", Automatica, 1975, 11, 285-296, DOI:`10.1109/TSMC.1979.4310076 <http://dx.doi.org/10.1109/TSMC.1979.4310076>`_

