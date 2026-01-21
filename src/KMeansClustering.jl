"""
    KMeansClustering

A Julia package for clustering algorithms, including K-Means, K-Medoids, K-Means++, BKmeans, and CKmeans.

# Exported Functions
- [`kmeans`](@ref): Perform K-Means clustering.
# Usage
julia> using KMeansClustering
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
using .KMeansLog: kmeanslog

"""
    kmeans(X, k; method=:kmeans, init=:random, maxiter=100, tol=1e-4, nstart=10, rng=Random.GLOBAL_RNG)

High-level entry point for k-means clustering.

Arguments
- `X`: data matrix with features in rows and observations in columns.
- `k`: number of clusters.

Keyword arguments
- `method`: algorithm selector, see below (:kmedoids)
- `init`: initialization strategy (:random, :kmeanspp).
- `maxiter`: maximum number of Lloyd iterations.
- `tol`: tolerance for convergence.
- `nstart`: number of random restarts for Bisecting K-Means (only used when `method=:bkmeans`).
- `rng`: random number generator.

Returns a `KMeansResult`.

Available algorithms:

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
function kmeans(
    X::AbstractMatrix{<:Real},
    k::Integer;
    method::Symbol=:kmeans,
    init::Symbol=:random,
    maxiter::Int=100,
    tol::Real=1e-4,
    nstart::Int=10,
    rng::AbstractRNG=GLOBAL_RNG
)

    if method == :kmedoids
        return kmedoids_fit(X, k, max_iter=maxiter, tol=tol, rng=rng)
    elseif method == :kmeans
        if init == :random
            idx = randperm(rng, size(X, 2))[1:k]
        elseif init == :kmeanspp
            idx = kmeanspp_init(X, k, rng=rng)
        else
            error("initialization strategy '$init' is not implemented")
        end
        return simplekmeans(X, X[:, idx], init_method=init, maxiter=maxiter, tol=tol)
    elseif method == :bkmeans
        return bkmeans_fit(X, k; max_iter=maxiter, tol=tol, nstart=nstart, rng=rng)
    elseif method == :kmeanslog
        idx = randperm(rng, size(X, 2))[1:k]
        return kmeanslog(X, X[:, idx], init, maxiter, maxiter, tol)
    else
        error("method '$method' is not implemented.")
    end
end

export kmeans, KMeansResult, KMedoidsAlgorithm, SimpleKMeansAlgorithm, BKMeansAlgorithm, simplekmeans

end # module