module AlgorithmsKMeansPP

using Random

"""
    kmeanspp_init(X, k; rng=Random.GLOBAL_RNG)

Select `k` initial centers using the k-means++ heuristic.

Arguments
- X: data matrix with features in rows and observations in columns.
- k: number of clusters.

Keyword arguments
- rng: random number generator.

Returns
A vector of length `k` with indices into the columns of `X`, indicating which
points are chosen as initial centers.
"""
function kmeanspp_init(X::AbstractMatrix{<:Real}, k::Integer;
                       rng::AbstractRNG = Random.GLOBAL_RNG)
    error("kmeanspp_init is not implemented yet")
end

end # module
