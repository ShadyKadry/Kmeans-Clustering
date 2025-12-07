module KMeansClustering

using Random

include("types.jl")
include("utils.jl")
include("algorithms/kmeans.jl")
include("algorithms/kmeanspp.jl")
include("algorithms/bkmeans.jl")
include("algorithms/ckmeans.jl")

export kmeans, KMeansResult

"""
    kmeans(X, k; init=:random, maxiter=100, tol=1e-4, rng=Random.GLOBAL_RNG)

High-level entry point for k-means clustering.

Arguments
- X: data matrix with features in rows and observations in columns.
- k: number of clusters.

Keyword arguments
- init: initialization strategy (:random, :kmeanspp).
- maxiter: maximum number of Lloyd iterations.
- tol: tolerance for convergence.
- rng: random number generator.

Returns a `KMeansResult`.
"""
function kmeans(X::AbstractMatrix{<:Real}, k::Integer;
                init::Symbol = :random,
                maxiter::Int = 100,
                tol::Real = 1e-4,
                rng::AbstractRNG = Random.GLOBAL_RNG)
    error("kmeans is not implemented yet. Implement Lloyd's algorithm in algorithms/kmeans.jl and call it from here.")
end

end # module
