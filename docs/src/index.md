```@meta
CurrentModule = KMeansClustering
```

# KMeansClustering

Documentation for [KMeansClustering](https://github.com/ShadyKadry/Kmeans-Clustering).

The module exports a single function [`KMeansClustering.kmeans`](@ref).
There are two ways to use this function. Either by passing all settings to the [`KMeansClustering.kmeans`](@ref) function directly
or by creating a settings struct specific to the algorithm and passing it to the [`KMeansClustering.kmeans`](@ref) function using multiple dispatch.
The latter option allows for more detailed settings. More information about the available settings can be found in the specific algorithm desciptions.

A simple step-by-step description can be found in the [Getting Started Guide](getting_started.md).
More usage example can be found in the examples repo folder.

```@docs
KMeansClustering.KMeansClustering
KMeansClustering.kmeans
KMeansClustering.KMeansResult
```
