module KMeans

using LinearAlgebra: norm
using Statistics: mean
using Random

using ..KMeansClustering: KMeansResult, KMeansAlgorithm

struct SimpleKMeansAlgorithm <: KMeansAlgorithm
    data::AbstractMatrix
    n_clusters::Integer
    init_method::Symbol
    max_iter::Integer
    tol::Real
    rng::AbstractRNG

    function SimpleKMeansAlgorithm(
        data::AbstractMatrix,
        n_clusters::Integer;
        init_method::Symbol=:random,
        max_iter::Integer=100,
        tol::Real=10e-4,
        rng::AbstractRNG=Random.GLOBAL_RNG
    )
        new(data, n_clusters, init_method, max_iter, tol, rng)
    end
end

"""
    simplekmeans(dataset::AbstractMatrix{<:Real}, 
                 initialcentroids::AbstractMatrix{<:Real}; 
                 init_method::Symbol=:random,
                 maxiter::Int=100,
                 tol::Real=10e-4)

Perform k-means clustering on a dataset following Lloyd's algorithm.
In each iteration step, the mean of each cluster becomes the new centroid.

# Arguments
- `dataset::AbstractMatrix{<:Real}`  
    A `dxn` matrix where each column is a point and each row is a feature.
- `initialcentroids::AbstractMatrix{<:Real}`  
    A `dxk` matrix containing the starting `k` centroids.

# Keyword Arguments
- `init_method::Symbol`  
    Method for choosing initial medoids, e.g. :random, :kmeans++
- `maxiter::Int`  
    Maximum number of iterations.
- `tol::Real`  
    tolerance threshold to determine convergence.

Returns a `KMeansResult`
"""
function simplekmeans(dataset::AbstractMatrix{<:Real},
    initialcentroids::AbstractMatrix{<:Real};
    init_method::Symbol=:random,
    maxiter::Int=100,
    tol::Real=10e-4)

    d, N = size(dataset)
    k = size(initialcentroids, 2)

    if d != size(initialcentroids, 1)
        error("dimensions of data and centroids do not match")
    end

    assignedto = Vector{Int}(undef, N)
    centroids = initialcentroids
    converged = false
    lastiter = 0

    for iter in 1:maxiter

        # assign points to nearest centroid

        for i in 1:N
            point = dataset[:, i]
            closestindex = 1
            mindist = norm(point - centroids[:, 1])

            for j in 2:k
                otherdist = norm(point - centroids[:, j])
                if otherdist < mindist
                    mindist = otherdist
                    closestindex = j
                end
            end
            assignedto[i] = closestindex
        end

        # calculate new centroid of each cluster

        newcentroids = Matrix{Float64}(undef, d, k)

        for i in 1:k
            # check for empty cluster
            indices = findall(==(i), assignedto)
            if isempty(indices)
                newcentroids[:, i] = centroids[:, i]
            else
                newcentroids[:, i] = vec(mean(dataset[:, indices], dims=2))
            end
        end

        # check for convergence

        if norm(newcentroids - centroids) < tol * sqrt(k * d)
            converged = true
            lastiter = iter
            break
        else
            centroids = newcentroids
        end

    end

    # calculate inertia

    dist = 0

    for i in 1:k
        dist += sum(norm.(eachcol(dataset[:, assignedto.==i] .- centroids[:, i])) .^ 2)
    end

    return KMeansResult(
        centroids,
        assignedto,
        dist,
        converged ? lastiter : maxiter,
        converged,
        init_method)
end

# Single struct overload
function simplekmeans(
    settings::SimpleKMeansAlgorithm,
    initialcentroids::AbstractMatrix
)
    simplekmeans(
        settings.data,
        initialcentroids;
        init_method=settings.init_method,
        maxiter=settings.max_iter,
        tol=settings.tol)
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

end