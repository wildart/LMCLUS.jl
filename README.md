LMCLUS.jl
=========
Version: 0.0.1

Julia package for Linear Manifold Clustering

# Documentation

* [Library-style function reference](https://github.com/wildart/LMCLUS.jl/tree/master/doc/Reference.md)

# Demo
```
julia> using LMCLUS

julia> p = LMCLUSParameters(5)
Linear Manifold Clustering parameters:
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

julia> ds = readdlm("test/testData", ',')[:,1:end-1] # remove last index column

julia> manifolds = lmclus(ds,p) 
3-element Array{Manifold,1}:
 Manifold: 
    Dimension: 1
    Size: 1000
    θ: 601.1803043130371

 Manifold: 
    Dimension: 1
    Size: 1000
    θ: 884.7436215451837

 Manifold: 
    Dimension: 0
    Size: 1000
    θ: Inf
```

# TODO
    * Documentation
