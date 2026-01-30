"""
    KMeansClustering

A Julia package for clustering algorithms, including K-Means, K-Medoids, K-Means++, BKmeans, and CKmeans.

# Exported Functions
- [`kmeans`](@ref): Perform K-Means clustering.
# Usage
julia> using KMeansClustering

# Implemented algorithms

- K-Medoids (method=:kmedoids):
    As described by [TU Dortmund: Partitioning Around Medoids (k-Medoids)](https://dm.cs.tu-dortmund.de/mlbits/cluster-kmedoids-intro/)
    Unlike typical K-Means, K-Medoids chooses its cluster centers from the given points X instead of calculating
    artificial ones.

- Bisecting K-Means (method=:bkmeans):
    A hierarchical, divisive variant of K-Means.
    The algorithm starts with a single cluster and repeatedly splits the cluster with the largest
    within-cluster sum of squared errors (SSE) into two sub-clusters, until `k` clusters are reached.
    Each split is performed by running a 2-means sub-problem (optionally with multiple restarts via `nstart`).
"""
module KMeansClustering
using Random: AbstractRNG, GLOBAL_RNG, randperm

include("types.jl")
include("algorithms/kmeans.jl")
include("algorithms/kmeanspp.jl")
include("algorithms/kmedoids.jl")
include("algorithms/bkmeans.jl")
include("algorithms/kmeanslog.jl")

using .AlgorithmsKMeansPP: kmeanspp_init

export kmeans, KMeansResult, KMedoidsAlgorithm, SimpleKMeansAlgorithm, BKMeansAlgorithm, simplekmeans, KMeansLogAlgorithm

end # module
