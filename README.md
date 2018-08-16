# LMCLUS

A Julia package for linear manifold clustering.

[![Build Status](https://travis-ci.org/wildart/LMCLUS.jl.svg?branch=master)](https://travis-ci.org/wildart/LMCLUS.jl)
[![Coverage Status](https://coveralls.io/repos/wildart/LMCLUS.jl/badge.png?branch=master)](https://coveralls.io/r/wildart/LMCLUS.jl)
[![DOI](https://zenodo.org/badge/19164/wildart/LMCLUS.jl.svg)](https://zenodo.org/badge/latestdoi/19164/wildart/LMCLUS.jl)

## Installation

Prior to Julia v0.7.0

```julia
Pkg.clone("https://github.com/wildart/LMCLUS.jl.git")
```

for Julia v0.7.0/1.0.0
```
pkg> add https://github.com/wildart/LMCLUS.jl.git#0.4.0
```

### Julia Compatibility
| Julia Version | LMCLUS version |
|---------------|----------------|
|v0.3.*|v0.0.2|
|v0.4.*|v0.1.2|
|v0.5.*|v0.2.0|
|v0.6.*|v0.3.0|
|≥v0.7.*|≥v0.4.0|


## Resources
- **Documentation:** <http://lmclusjl.readthedocs.org/en/latest/index.html>
- **Papers:**
    - Haralick, R. & Harpaz, R., "Linear manifold clustering in high dimensional spaces by stochastic search", Pattern recognition, Elsevier, 2007, 40, 2672-2684, DOI:[10.1016/j.patcog.2007.01.020](http://dx.doi.org/10.1016/j.patcog.2007.01.020)
    - Haralick et al., "Inexact MDL for Linear Manifold Clusters", ICPR-2016, DOI:[10.1109/ICPR.2016.7899824](http://dx.doi.org/10.1109/ICPR.2016.7899824)