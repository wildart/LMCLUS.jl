Utilities
=========

Linear Manifold Clustering Algorithm relies on multiple search and optimization methods:

.. function:: kittler(X, bins, tol)

    A minimum error thresholding method for multimodal histograms.

.. function:: otsu(X, bins)

    A gray-level thresholding method for multimodal histograms.

.. function:: MDLength(M, X; P = 16, T = :Gausian, ɛ = 1e-4)

    Performs calculation of linear manifold cluster minimum description length.

    :param M:   Linear manifold cluster description as ``Manifold`` type instance.
    :param X:   Linear manifold cluster data as ``Matrix`` with points as its columns.
    :param P:   Precision encoding constant, i.e. number of bits required for
                calculating size of a number, if MDL schema contains one, in
                certain floating point precession. Default value is 32 which correspond
                to ``Float32``.
    :param T:   Type of a dataset encoding model as ``Symbol``:
                ``:Gausian``, ``:Uniform ``, ``:Empirical``.
    :param ɛ:   Error tolerance for bin quantization used in an empirical model encoding

    Returns number of bits required to encode linear manifold cluster with the MDL schema.
