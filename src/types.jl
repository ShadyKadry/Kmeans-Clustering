"""
KMeansResult

This type stores the outcome of a k-means clustering run.

Conventions:
- The data matrix X is assumed to have observations in columns and features in rows:
    size(X, 1) = number of features
    size(X, 2) = number of points
- The centers matrix follows the same convention: each column is a cluster center.

Fields:
- centers::Matrix{T}
    The final cluster centers as a d×k matrix, where d is the number of features
    and k is the number of clusters.

- assignments::Vector{Int}
    Cluster assignment for each data point as a vector of length n, where
    n is the number of points (columns of X). The i-th entry is an integer
    in 1:k indicating the cluster index of point i.

- inertia::T
    The sum of squared distances of each point to its assigned center
    (within-cluster sum of squares), used as a measure of cluster quality.

- iterations::Int
    The number of iterations of the k-means update loop that were performed.

- converged::Bool
    Indicates whether the algorithm stopped because it met the convergence
    criterion (true) or because it hit the maximum number of iterations (false).

- init_method::Symbol
    The initialization method used for the run, e.g. :random or :kmeanspp.
"""
struct KMeansResult{T<:Real}
    centers::Matrix{T}
    assignments::Vector{Int}
    inertia::T
    iterations::Int
    converged::Bool
    init_method::Symbol
end
