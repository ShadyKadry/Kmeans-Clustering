module KMeansClustering

using Random

include("types.jl")
include("utils.jl")
include("algorithms/kmeans.jl")
include("algorithms/kmeanspp.jl")
include("algorithms/kmedoids.jl")
include("algorithms/bkmeans.jl")
include("algorithms/ckmeans.jl")

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

- K-Medoids (method=:kmedoids)
    As described by [E.M. Mirkes, K-means and K-medoids applet. University of Leicester, 2011](http://leicestermath.org.uk/KmeansKmedoids/Kmeans_Kmedoids.html)
    Unlike typical K-Means, K-Medoids chooses its cluster centers from the given points X instead of calculating 
    artificial ones.

"""
function kmeans(X::AbstractMatrix{<:Real}, 
                k::Integer;
                method::Symbol = :kmeans,
                init::Symbol = :random,
                maxiter::Int = 1000,
                tol::Real = 1e-4,
                rng::AbstractRNG = Random.GLOBAL_RNG
        )

    if method == :kmedoids
        return KMedoids.KMedoids_fit(X, k, init_method=init, max_iter=maxiter, tol=tol, rng=rng)
    else
        error("method '$method' is not implemented.")
    end
end

end # module