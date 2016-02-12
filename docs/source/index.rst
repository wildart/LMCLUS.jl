
Linear Manifold Clustering (LMCLUS)
====================================

Many clustering algorithms are based on the concept that a cluster has a a single center point.
Clusters could be considered as groups of points compact around a linear manifold. A linear manifold of dimension 0 is a point. So clustering around a center point is a special case of linear manifold clustering.

Linear manifold clustering algorithm identifies subsets of the data which are embedded in arbitrary oriented lower dimensional linear manifolds, not nessesaraly zero dimensional. Minimal subsets of points are repeatedly sampled to construct trial a linear manifold and isolete points around it based of the proximity of points to the found manifold. Using top-down aproach, the linear manifold clustering alogorithm iterativly partitions dataset and discovers clusters embedded into low-dimensioanl linear subspaces [#R1]_.

*LMCLUS.jl* is a Julia package for linear manifold clustering.


**Contents:**

.. toctree::
   :maxdepth: 1

   lmclus.rst
   params.rst
   separation.rst
   utils.rst

**Notes:**

All methods implemented in this package adopt the column-major convention: in a data matrix, each column corresponds to a sample/observation, while each row corresponds to a feature (variable or attribute).

.. rubric:: References
.. [#R1] Haralick, R. & Harpaz, R. "Linear manifold clustering in high dimensional spaces by stochastic search", Pattern recognition, Elsevier, 2007, 40, 2672-2684, DOI:`10.1016/j.patcog.2007.01.020 <http://dx.doi.org/10.1016/j.patcog.2007.01.020>`_
