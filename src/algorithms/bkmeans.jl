using LinearAlgebra: norm
using Statistics: mean
using Random: AbstractRNG, GLOBAL_RNG, rand

using ..KMeansClustering: KMeansResult, KMeansAlgorithm

# Implementation for the Bisecting K-Means (BKMeans) algorithm.
#
# Bisecting K-Means is a divisive (top-down) clustering method:
# 1) Start with one cluster containing all points.
# 2) Repeatedly split the "worst" cluster (largest SSE) into two sub-clusters
#    using a 2-means run.
# 3) Continue until the requested number of clusters is reached.

# AI Note:
# - Parts of the documentation and comments in the code were written with the help of AI
# - AI was used to validate parts of the code
# - Recomomendations from AI were I could improve the code structure
# The code itself was not written by any form of AI

"""
    BKMeansAlgorithm(
        data::AbstractMatrix{<:Real},
        n_clusters::Integer;
        max_iter::Integer = 100,
        tol::Real = 1e-4,
        nstart::Integer = 10,
        rng::AbstractRNG = Random.GLOBAL_RNG
    )

Settings specific to the **Bisecting K-Means (BKMeans)** algorithm.

Use this settings struct with [`kmeans(::BKMeansAlgorithm)`](@ref) to run
bisecting k-means via multiple dispatch.

Fields:
- `data`: Data matrix with features in rows and observations in columns
- `n_clusters`: Number of clusters to produce
- `max_iter`: Maximum number of Lloyd iterations for each 2-means split
- `tol`: Convergence tolerance for each 2-means split
- `nstart`: Number of random restarts for each 2-means split (best split wins)
- `rng`: Random Number Generator used for split initialization
"""
struct BKMeansAlgorithm{R<:AbstractMatrix{<:Real}, K<:AbstractRNG} <: KMeansAlgorithm
    data::R
    n_clusters::UInt32
    max_iter::UInt32
    tol::Float64
    nstart::UInt32
    rng::K

    function BKMeansAlgorithm(
        data::R,
        n_clusters::Integer;
        max_iter::Integer = 100,
        tol::Real = 1e-4,
        nstart::Integer = 10,
        rng::K = GLOBAL_RNG
    ) where {R<:AbstractMatrix{<:Real}, K<:AbstractRNG}

        n_points = size(data, 2)
        n_clusters >= 1 || throw(ArgumentError("Number of clusters must be at least 1, got $(n_clusters)"))
        n_clusters <= n_points || throw(ArgumentError("Number of clusters must be <= number of data points (data points == $(n_points)), got $(n_clusters) requested clusters"))
        max_iter >= 1 || throw(ArgumentError("max_iter must be at least 1, got $(max_iter)"))
        nstart >= 1 || throw(ArgumentError("nstart must be at least 1, got $(nstart)"))

        new{R, K}(data, UInt32(n_clusters), UInt32(max_iter), Float64(tol), UInt32(nstart), rng)
    end
end

"""Squared Euclidean distance between two vectors."""
@inline function _sqdist(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})::Float64
    s = 0.0
    @inbounds for i in eachindex(x, y)
        d = x[i] - y[i]
        s += d * d
    end
    return s
end

"""SSE (sum of squared errors) of a cluster given its member indices and centroid."""
function _cluster_sse(dataset::AbstractMatrix{<:Real}, idxs::Vector{Int}, centroid::AbstractVector{<:Real})::Float64
    s = 0.0
    @inbounds for id in idxs
        s += _sqdist(view(dataset, :, id), centroid)
    end
    return s
end

"""
Initialize 2 centroids for a 2-means split of the `subset`.

Strategy:
1) pick one random point as the first centroid
2) pick the point farthest away from it as the second centroid
"""
function _init_two_centroids(self::BKMeansAlgorithm, subset::AbstractMatrix{<:Real})::Matrix{Float64}
    d, n = size(subset)
    n >= 2 || error("Can't initialize 2 centroids from a cluster with fewer than 2 points")

    i1 = rand(self.rng, 1:n)
    p1 = view(subset, :, i1)

    besti = (i1 == 1 ? 2 : 1)
    bestd = -1.0
    @inbounds for i in 1:n
        i == i1 && continue
        dist = _sqdist(view(subset, :, i), p1)
        if dist > bestd
            bestd = dist
            besti = i
        end
    end

    init = Matrix{Float64}(undef, d, 2)
    init[:, 1] = Float64.(subset[:, i1])
    init[:, 2] = Float64.(subset[:, besti])
    return init
end

"""
Internal 2-means (Lloyd) used for bisection.

Conventions:
- dataset is `d×N` and centroids are `d×k` (points and centroids are columns)

Returns `(centroids, assignments, iterations, converged)`.
"""
function _simplekmeans(
    dataset::Matrix{Float64},
    initialCentroids::Matrix{Float64},
    maxiter::Int,
    tol::Real
)
    d, N = size(dataset)
    k = size(initialCentroids, 2)
    d == size(initialCentroids, 1) || error("Dimensions of data and centroids do not match")

    assignedto = Vector{Int}(undef, N)
    centroids = copy(initialCentroids)
    converged = false

    for iter in 1:maxiter
        # 1) assign each point to the closest centroid
        @inbounds for i in 1:N
            point = view(dataset, :, i)
            closestindex = 1
            mindist = _sqdist(point, view(centroids, :, 1))

            for j in 2:k
                otherdist = _sqdist(point, view(centroids, :, j))
                if otherdist < mindist
                    mindist = otherdist
                    closestindex = j
                end
            end
            assignedto[i] = closestindex
        end

        # 2) recompute centroids
        newcentroids = Matrix{Float64}(undef, d, k)
        for j in 1:k
            indices = findall(==(j), assignedto)
            if isempty(indices)
                # safeguard: keep old centroid if the cluster is empty
                newcentroids[:, j] = centroids[:, j]
            else
                newcentroids[:, j] = vec(mean(dataset[:, indices], dims=2))
            end
        end

        # 3) stop if centroids barely move
        if norm(newcentroids - centroids) < tol
            centroids = newcentroids
            converged = true
            return centroids, assignedto, iter, converged
        else
            centroids = newcentroids
        end
    end

    return centroids, assignedto, maxiter, converged
end

"""
Core BKMeans routine.

Returns `(centroids, assignments, total_iters, converged, inertia)`.
"""
function _bkmeans(
    self::BKMeansAlgorithm,
    dataset::Matrix{Float64},
    k::Int,
    maxiter::Int,
    tol::Real;
    nstart::Int
)
    d, N = size(dataset)

    # Start with a single cluster containing all points
    assignedto = ones(Int, N)
    centroids = reshape(vec(mean(dataset, dims=2)), d, 1)

    # clusters[i] stores the global point indices that belong to cluster i
    clusters = Vector{Vector{Int}}(undef, 1)
    clusters[1] = collect(1:N)

    # sse[i] stores the SSE value of cluster i
    sse = Vector{Float64}(undef, 1)
    sse[1] = _cluster_sse(dataset, clusters[1], centroids[:, 1])

    totaliters = 0
    converged = true
    current_k = 1

    while current_k < k
        # Choose the cluster to split: the one with the largest SSE (and size > 1)
        splitidx = 0
        bestval = -1.0
        for i in 1:current_k
            if length(clusters[i]) > 1 && sse[i] > bestval
                bestval = sse[i]
                splitidx = i
            end
        end
        splitidx != 0 || error("Cannot bisect further: all clusters have size 1, but requested k=$k")

        # Extract the points of the chosen cluster into a smaller matrix subset
        idxs = clusters[splitidx]
        subset = dataset[:, idxs]

        # Run multiple 2-means attempts and keep the best split (lowest SSE)
        best_split_sse = Inf
        best_centroids2 = Matrix{Float64}(undef, d, 2)
        best_assign2 = Int[]
        best_iter = 0
        best_conv = false

        for _ in 1:nstart
            init = _init_two_centroids(self, subset)
            c2, a2, it2, conv2 = _simplekmeans(subset, init, maxiter, tol)

            # If one side becomes empty, this split is invalid
            n1 = count(==(1), a2)
            n2 = count(==(2), a2)
            if n1 == 0 || n2 == 0
                continue
            end

            # Compute SSE for the two new clusters
            idx1_local = findall(==(1), a2)
            idx2_local = findall(==(2), a2)
            sse1 = _cluster_sse(subset, idx1_local, c2[:, 1])
            sse2 = _cluster_sse(subset, idx2_local, c2[:, 2])
            split_sse = sse1 + sse2

            if split_sse < best_split_sse
                best_split_sse = split_sse
                best_centroids2 = c2
                best_assign2 = a2
                best_iter = it2
                best_conv = conv2
            end
        end

        # Fallback
        if isempty(best_assign2)
            m = length(idxs)
            half = m ÷ 2
            best_assign2 = vcat(fill(1, half), fill(2, m - half))
            best_centroids2[:, 1] = vec(mean(subset[:, 1:half], dims=2))
            best_centroids2[:, 2] = vec(mean(subset[:, (half + 1):m], dims=2))
            best_iter = 0
            best_conv = false
        end

        # Convert local split labels (1/2 within subset) into global point index lists
        idx1_global = Int[]
        idx2_global = Int[]
        for (localpos, globalid) in enumerate(idxs)
            if best_assign2[localpos] == 1
                push!(idx1_global, globalid)
            else
                push!(idx2_global, globalid)
            end
        end

        # The new cluster will get the next label number
        newlabel = current_k + 1

        # Update global assignment vector assignedto
        for id in idx1_global
            assignedto[id] = splitidx
        end
        for id in idx2_global
            assignedto[id] = newlabel
        end

        # Update stored cluster member lists
        clusters[splitidx] = idx1_global
        push!(clusters, idx2_global)

        # Update the centroid matrix
        centroids[:, splitidx] = best_centroids2[:, 1]
        centroids = hcat(centroids, best_centroids2[:, 2])

        # Recompute SSE values for the affected clusters
        sse[splitidx] = _cluster_sse(dataset, clusters[splitidx], centroids[:, splitidx])
        push!(sse, _cluster_sse(dataset, clusters[end], centroids[:, end]))

        # Track total work and convergence status
        totaliters += best_iter
        converged &= best_conv
        current_k += 1
    end

    inertia = sum(sse)
    return centroids, assignedto, totaliters, converged, inertia
end

# Perform BKMeans clustering on a dataset.
#
# Returns a KMeansResult.
function bkmeans_fit(
    data::AbstractMatrix{<:Real},
    n_clusters::Integer;
    max_iter::Integer = 100,
    tol::Real = 1e-4,
    nstart::Integer = 10,
    rng::AbstractRNG = GLOBAL_RNG
)
    return kmeans(
        BKMeansAlgorithm(
            data,
            n_clusters;
            max_iter,
            tol,
            nstart,
            rng
        )
    )
end

"""
    kmeans(::BKMeansAlgorithm)

Entry point for Bisecting K-Means clustering using a settings object.

Returns a [`KMeansResult`](@ref) containing the clustering results.

See also: [`kmeans(X, k; method=:bkmeans, maxiter=100, tol=1e-4, rng=GLOBAL_RNG)`](@ref)
"""
function kmeans(self::BKMeansAlgorithm)
    dataset = Float64.(self.data)
    k = Int(self.n_clusters)
    maxiter = Int(self.max_iter)
    tol = self.tol
    nstart = Int(self.nstart)

    centroids, assignedto, totaliters, converged, inertia = _bkmeans(self, dataset, k, maxiter, tol; nstart)

    return KMeansResult(
        centroids,
        assignedto,
        Float64(inertia),
        Int(totaliters),
        Bool(converged),
        :bkmeans
    )
end