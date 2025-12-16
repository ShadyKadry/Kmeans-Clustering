# https://arxiv.org/abs/1911.05940
# https://github.com/DPanknin/modelagnostic_superior_training/blob/main/modelagnostic_superior_training/myKMeans.py

module CKMeans
using ..KMeansClustering: KMeansResult
using LinearAlgebra: norm

function ckmeans(dataset::Matrix{Float64},
  initialcentroids::Matrix{Float64},
  init_method::Symbol,
  maxiter::Int,
  tol::Real)

  d, N = size(dataset)
  k = size(initialcentroids, 2)

  if d != size(initialcentroids, 1)
    error("dimensions of data and centroids do not match")
  end
  error("not implemented")
end
"""
Find clusters that minimize the sum of the log of the Euclidean norm.
"""
function dc_asymp(dataset::Matrix{Float64},
  init_method::Symbol,
  initialcentroids::Matrix{Float64},
  maxiter::Int,
  maxinneriter::Int,
  tol::Real)
  N = size(dataset, 2)
  # a mapping from an index in the data set to an index in the clusters.
  cluster_map = Vector{Int}(undef, N)
  # an array of centroids
  centroids = initialcentroids
  # a mapping from an index in the data set to its weight value in its cluster.
  weights = Vector{Real}(undef, N)
  n_outer_iter::Int = maxiter
  converged = false
  for iter in 1:maxiter
    # Step 1: assign points to nearest centroid
    for i in 1:N
      point = dataset[:, i]
      closest_i = 1
      min_dist = norm(point - centroids[:, 1])
      for j in 2:size(centroids, 2)
        other_dist = norm(point - centroids[:, j])
        if other_dist < min_dist
          min_dist = other_dist
          closest_i = j
        end
      end
      cluster_map[i] = closest_i
    end
    # Step 2: compute weights
    for point_i in 1:N
      point = dataset[:, point_i]
      cluster_i = cluster_map[point_i]
      weights[point_i] = 1 / norm(point - centroids[:, cluster_i])
    end
    # Step 3: update clusters
    for cluster_i in 1:size(centroids, 2)
      # minimize log 2-norm via iterative reweighted least squares (IRLS) for each cluster.
      inner_flag = true
      curr_iter:Int = 1
      while inner_flag && curr_iter <= maxinneriter
        new_centroid = sum(weights[:, point_i] .* dataset[:, points_i] for i in 1:N if cluster_map[point_i] == cluster_i)
        if norm(new_centroid - centroids[:, cluster_i]) < tol
          inner_flag = false
        end
        centroids[:, cluster_i] = new_centroid
        for point_i in 1:N
          if cluster_map[point_i] == cluster_i
            point = dataset[:, point_i]
            weights[point_i] = 1 / norm(point - centroids[:, cluster_i])
          end
        end
        curr_iter += 1
      end
    end
    # break the loop if the max error is under the threshold.
    if max(norm(dataset[:, point_i] - centroids[:, cluster_map[point_i]]) for point_i in 1:N) < tol
      converged = true
      n_outer_iter = iter
      break
    end
  end
  # calculate inertia

  dist::Real = 0
  for i in 1:size(initialcentroids, 2)
    dist += sum(norm.(eachcol(dataset[:, cluster_map.==i] .- centroids[:, i])) .^ 2)
  end
  return KMeansResult(
    centroids,
    cluster_map,
    dist,
    n_outer_iter,
    converged,
    init_method)
end
export ckmeans
export dc_asymp
end
