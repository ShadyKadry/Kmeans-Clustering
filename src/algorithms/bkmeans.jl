module BKMeans

using LinearAlgebra: norm
using Statistics: mean
using Random: rand

"""
Helper: squared Euclidean distance between two vectors.
"""
function _sqdist(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})::Float64
    s = 0.0
    @inbounds for i in 1:length(x)
        d = x[i] - y[i]
        s += d * d
    end
    return s
end

"""
Helper: SSE (Sum of Squared Errors) for one cluster.
SSE measures how "spread out" a cluster is:
  SSE = sum over points in the cluster of squared distance to the centroid.
Larger SSE means the cluster is more "messy" and is a good candidate to split.
"""
function _cluster_sse(dataset::Matrix{Float64}, idxs::Vector{Int}, centroid::Vector{Float64})::Float64
    s = 0.0
    @inbounds for id in idxs
        s += _sqdist(view(dataset, :, id), centroid)
    end
    return s
end

"""
Helper: initialization for a 2-means split.
We pick:
  1) one random point as the first centroid
  2) the point farthest away from it as the second centroid
Input subset is a d×n matrix (each column is a point).
"""
function _init_two_centroids(subset::Matrix{Float64})::Matrix{Float64}
    d, n = size(subset)
    if n < 2
        error("Can't initialize 2 centroids from a cluster with fewer than 2 points")
    end

    i1 = rand(1:n)
    p1 = view(subset, :, i1)

    besti = (i1 == 1 ? 2 : 1)
    bestd = -1.0
    @inbounds for i in 1:n
        if i == i1
            continue
        end
        dist = _sqdist(view(subset, :, i), p1)
        if dist > bestd
            bestd = dist
            besti = i
        end
    end

    init = Matrix{Float64}(undef, d, 2)
    init[:, 1] = subset[:, i1]
    init[:, 2] = subset[:, besti]
    return init
end

"""
Internal simple k-means.
NOTE: dataset is d×N and centroids are d×k (points and centroids are columns).
"""
function _simplekmeans(dataset::Matrix{Float64}, initialCentroids::Matrix{Float64}, maxiter::Int, tol::Real)
    d, N = size(dataset)
    k = size(initialCentroids, 2)

    if d != size(initialCentroids, 1)
        error("Dimensions of data and centroids do not match")
    end

    assignedto = Vector{Int}(undef, N)
    centroids = copy(initialCentroids)
    converged = false

    for iter in 1:maxiter

        # 1. assign each point to the closest centroid
        for i in 1:N
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

        """
        2. Recompute each centroid as the mean of its assigned points
        """
        newcentroids = Matrix{Float64}(undef, d, k)

        for j in 1:k
            indices = findall(==(j), assignedto)
            if isempty(indices)
                # Safeguard: if a cluster is empty, keep the old centroid.
                newcentroids[:, j] = centroids[:, j]
            else
                newcentroids[:, j] = vec(mean(dataset[:, indices], dims=2))
            end
        end

        """
        3. Stop if centroids barely move
        """
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


function bkmeans(dataset::Matrix{Float64}, k::Int, maxiter::Int, tol::Real; nstart::Int=10)
    d, N = size(dataset)

    if k < 1
        error("k must be >= 1")
    end
    if k > N
        error("k cannot be larger than the number of points (N=$N)")
    end

    """
    Start with a single cluster containing all points
    """
    assignedto = ones(Int, N)
    centroids = reshape(vec(mean(dataset, dims=2)), d, 1)

    """
    clusters[i] stores the global point indices that belong to cluster i
    """
    clusters = Vector{Vector{Int}}(undef, 1)
    clusters[1] = collect(1:N)

    """
    sse[i] stores the SSE value of cluster i (used to decide which cluster to split next)
    """
    sse = Vector{Float64}(undef, 1)
    sse[1] = _cluster_sse(dataset, clusters[1], centroids[:, 1])

    totaliters = 0
    converged = true
    current_k = 1

    while current_k < k

        """
        Choose the cluster to split: the one with the largest SSE (and size > 1)
        """
        splitidx = 0
        bestval = -1.0
        for i in 1:current_k
            if length(clusters[i]) > 1 && sse[i] > bestval
                bestval = sse[i]
                splitidx = i
            end
        end
        if splitidx == 0
            """
            This happens if all clusters have exactly one point.
            """
            error("Cannot bisect further: all clusters have size 1, but requested k=$k")
        end

        """
        Extract the points of the chosen cluster into a smaller matrix subset
        """
        idxs = clusters[splitidx]
        subset = dataset[:, idxs]

        """
        Run multiple 2-means attempts and keep the best split (lowest SSE)
        """
        best_split_sse = Inf
        best_centroids2 = Matrix{Float64}(undef, d, 2)
        best_assign2 = Int[]
        best_iter = 0
        best_conv = false

        for trial in 1:nstart
            init = _init_two_centroids(subset)
            c2, a2, it2, conv2 = _simplekmeans(subset, init, maxiter, tol)

            """
            If one side becomes empty, this split is invalid: skip it.
            """
            n1 = count(==(1), a2)
            n2 = count(==(2), a2)
            if n1 == 0 || n2 == 0
                continue
            end

            """
            Compute SSE for the two new clusters
            """
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

        """
        Fallback if all tries failed
        """
        if isempty(best_assign2)
            """
            Split the points into two halves
            """
            m = length(idxs)
            half = m ÷ 2
            best_assign2 = vcat(fill(1, half), fill(2, m - half))
            best_centroids2[:, 1] = vec(mean(subset[:, 1:half], dims=2))
            best_centroids2[:, 2] = vec(mean(subset[:, (half+1):m], dims=2))
            best_iter = 0
            best_conv = false
        end

        """
        Convert local split labels (1/2 within subset) into global point index lists
        """
        idx1_global = Int[]
        idx2_global = Int[]
        for (localpos, globalid) in enumerate(idxs)
            if best_assign2[localpos] == 1
                push!(idx1_global, globalid)
            else
                push!(idx2_global, globalid)
            end
        end

        """
        The new cluster will get the next label number
        """
        newlabel = current_k + 1

        """
        Update global assignment vector assignedto
        """
        for id in idx1_global
            assignedto[id] = splitidx
        end
        for id in idx2_global
            assignedto[id] = newlabel
        end

        """
        Update the stored cluster member lists
        """
        clusters[splitidx] = idx1_global
        push!(clusters, idx2_global)

        """
        Update the centroid matrix
        Replace centroid of the split cluster and append the new one
        """
        centroids[:, splitidx] = best_centroids2[:, 1]
        centroids = hcat(centroids, best_centroids2[:, 2])

        """
        Recompute SSE values for the affected clusters
        """
        sse[splitidx] = _cluster_sse(dataset, clusters[splitidx], centroids[:, splitidx])
        push!(sse, _cluster_sse(dataset, clusters[end], centroids[:, end]))

        # Track total work and convergence status
        totaliters += best_iter
        converged &= best_conv
        current_k += 1
    end

    return centroids, assignedto, totaliters, converged
end

end
