module KMeans

using LinearAlgebra: norm
using Statistics: mean

using ..KMeansClustering: KMeansResult

function simplekmeans(dataset::Matrix{Float64},
    initialcentroids::Matrix{Float64},
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

    return KMeansResult(
        centroids,
        assignedto,
        1.0,
        converged ? lastiter : maxiter,
        converged,
        init_method)
end

end