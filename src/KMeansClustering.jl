
"""
    KMeansClustering

A Julia package for clustering algorithms, including K-Means, K-Medoids, K-Means++, BKmeans, and CKmeans.

# Exported Functions
- [`kmeans`](@ref): Perform K-Means clustering.
# Usage
julia> using KMeansClustering
"""
module KMeansClustering

include("types.jl")
include("utils.jl")
include("algorithms/kmeans.jl")
include("algorithms/kmeanspp.jl")
include("algorithms/kmedoids.jl")
include("algorithms/bkmeans.jl")
include("algorithms/ckmeans.jl")

using Random

using .KMedoids: KMedoidsAlgorithm, kmedoids_fit
using .KMeans: SimpleKMeansAlgorithm, simplekmeans
using .BKMeans: bkmeans
using .AlgorithmsKMeansPP: kmeanspp_init


export kmeans, KMeansResult

"""
    kmeans(X, k; method=:kmeans, init=:random, maxiter=100, tol=1e-4, rng=Random.GLOBAL_RNG)

High-level entry point for k-means clustering.

Arguments
- `X`: data matrix with features in rows and observations in columns.
- `k`: number of clusters.

Keyword arguments
- `method`: algorithm selector, see below (:kmedoids)
- `init`: initialization strategy (:random, :kmeanspp).
- `maxiter`: maximum number of Lloyd iterations.
- `tol`: tolerance for convergence.
- `rng`: random number generator.

Returns a `KMeansResult`.

Available algorithms:

- K-Medoids (method=:kmedoids):
    As described by [E.M. Mirkes, K-means and K-medoids applet. University of Leicester, 2011](http://leicestermath.org.uk/KmeansKmedoids/Kmeans_Kmedoids.html)
    Unlike typical K-Means, K-Medoids chooses its cluster centers from the given points X instead of calculating
    artificial ones.

"""
function kmeans(X::AbstractMatrix{<:Real},
    k::Integer;
    method::Symbol=:kmeans,
    init::Symbol=:random,
    maxiter::Int=1000,
    tol::Real=1e-4,
    rng::AbstractRNG=Random.GLOBAL_RNG
)

    if method == :kmedoids
        return kmedoids_fit(X, k, init_method=init, max_iter=maxiter, tol=tol, rng=rng)
    elseif method == :kmeans
        if init == :random
            idx = randperm(rng, size(X, 2))[1:k]
            return simplekmeans(X, X[:, idx], init_method=init, maxiter=maxiter, tol=tol)
        else
            error("initialization strategy '$init' is not implemented")
        end
    elseif method == :bkmeans
        ce, as, to, co = bkmeans(Float64.(X), k, maxiter, tol)
        return KMeansResult(ce, as, to, co)
    else
        error("method '$method' is not implemented.")
    end
end


"""
    kmeans(settings::KMedoidsAlgorithm)

    Entry point for K-Medoids clustering using a settings object instead.

# Arguments
- `settings::KMedoidsAlgorithm`: Settings object. See object description for more information

# Returns
A `KMeansResult` containing the clustering results.

# Example
```julia
settings = KMeansClustering.KMedoidsAlgorithm(
    X,                  # Points, column-wise: rows are the features, cols are the points
    cluster_count;
    init_method=:random,
    max_iter=50,
)
result = KMeansClustering.kmeans(settings)
```

See also: [`kmeans(X, k; method=:kmeans, init=:random, maxiter=100, tol=1e-4, rng=Random.GLOBAL_RNG)`](@ref)
"""
function kmeans(
    settings::KMedoidsAlgorithm
)
    kmedoids_fit(settings)
end

function kmeans(
    settings::SimpleKMeansAlgorithm
)
    X = settings.data
    k = settings.n_clusters
    if settings.init_method == :random
        idx = randperm(settings.rng, size(X, 2))[1:k]
    elseif settings.init_method == :kmeanspp
        idx = kmeanspp_init(X, k; rng=settings.rng)
    else
        error("initialization strategy '$settings.init_method' is not implemented")
    end

    simplekmeans(settings, X[:, idx])

end

end # module
