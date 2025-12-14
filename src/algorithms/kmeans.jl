module KMeans

using LinearAlgebra: norm
using Statistics: mean

using ..KMeansClustering: KMeansResult

"""
    simplekmeans(dataset::Matrix{Float64}, initialcentroids::Matrix{Float64}; init_method::Symbol, maxiter::Int, tol::Real)

Perform k-means clustering on a dataset following Lloyd's algorithm.
In each iteration step, the mean of each cluster becomes the new centroid.

# Arguments
- `dataset::Matrix{Float64}`  
    A `dxn` matrix where each column is a point and each row is a feature.
- `initialcentroids::Matrix{Float64}`  
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

function simplekmeans(dataset::Matrix{Float64},
    initialcentroids::Matrix{Float64};
    init_method::Symbol,
    maxiter::Int,
    tol::Real)

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
                newcentroids[:, i] = mean(dataset[:, indices], dims=2)
            end
        end

        # check for convergence

        if norm(newcentroids - centroids) < tol
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

end