# LMCLUS

A Julia package for linear manifold clustering

[![Build Status](https://travis-ci.org/wildart/LMCLUS.jl.svg?branch=master)](https://travis-ci.org/wildart/LMCLUS.jl) [![Coverage Status](https://coveralls.io/repos/wildart/LMCLUS.jl/badge.png?branch=master)](https://coveralls.io/r/wildart/LMCLUS.jl)

-------

# Demo
```
julia> using LMCLUS

julia> p = LMCLUSParameters(5)
Linear Manifold Clustering parameters:
Min dimension (min_dim): 1
Max dimension (max_dim): 5
Number of clusters (cluster_number): 100
Noise size (noise_size): 20
Best bound (best_bound): 1.0
Error bound (error_bound): 0.0001
Sample points for distance histogram (histogram_sampling): false
Histogram bins (hist_bin_size): 0
Maximum histogram bin size (max_bin_portion): 0.1
Sampling heuristic (heuristic): 3
Sampling factor (sampling_factor): 0.01
Random seed (random_seed): 0 (0 - random seed)
0D manifold search (zero_d_search): false
Manifold cluster basis alignment (basis_alignment): false
Log level (log_level): 0

julia> ds = readdlm(Pkg.dir("LMCLUS", "test", "testData"), ',')

julia> manifolds = lmclus(ds[:,1:end-1]',p) # remove last index column
3-element Array{Manifold,1}:
 Manifold (dim = 1, size = 1000)
 Manifold (dim = 1, size = 1000)
 Manifold (dim = 0, size = 1000)

julia> dump(manifolds[1])
Manifold (dim = 1, size = 1000)
threshold (θ): 709.1368999082067 
translation (μ): 
 -85.8847  541.302  355.883  -94.3428  167.501  -239.952  -51.7843  -215.585  -158.391  247.486
basis: 
 -0.246314 
 -0.139219 
 -0.485083 
 -0.573303 
 -0.407966 
  0.0484066
  0.106572 
  0.140955 
 -0.291805 
  0.2661  
```

# TODO
    * Zero-dimensional manifold search
    * Documentation
