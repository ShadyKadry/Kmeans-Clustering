```@meta
CurrentModule = KMeansClustering
```

# KMeansClustering

Documentation for [KMeansClustering](https://github.com/ShadyKadry/Kmeans-Clustering).

The module exports a single function [`KMeansClustering.kmeans`](@ref).
To use this function, a settings struct needs to be created, that is then passed on to the [`KMeansClustering.kmeans`](@ref) function using multiple dispatch. More information about the available algorithms can be found in the specific algorithm desciptions.

A simple step-by-step description can be found in the [Getting Started Guide](getting_started.md).
More usage example can be found in the examples repo folder.

```@docs
KMeansClustering.KMeansClustering
KMeansClustering.kmeans
KMeansClustering.KMeansResult
```
