using LinearAlgebra: norm

"""
    kmeanslog{T<:AbstractMatrix{<:Real}}(
        dataset::T,
        initialcentroids::T,
        init_method::Symbol,
        tol::Real;
        maxiter::Int=100,
        maxinneriter::Int=100,
        eps::Real=1e-12)
Find clusters that minimize the sum of the log of the Euclidean norm.

# Arguments
- `dataset::Matrix{Float64}`  
    A `dxn` matrix where each column is a point and each row is a feature.
- `initialcentroids::Matrix{Float64}`  
    A `dxk` matrix containing the starting `k` centroids.
- `init_method::Symbol`  
    Method for choosing initial medoids, e.g. :random, :kmeans++
- `tol::Real`  
    tolerance threshold to determine convergence. Note that this number is the `log` of the distance from the cluster point.
# Keyword Arguments
- `maxiter::Int`  
    Maximum number of iterations.
- `maxinneriter::Int`
    Maximum number of iterations for iterative reweighted least squares, which is used to compute the new cluster point.
- `eps`
    epsilon to avoid `log(0)`
Returns a `KMeansResult`
"""
function kmeanslog(
  dataset::T,
  initialcentroids::T,
  init_method::Symbol,
  tol::Real;
  maxiter::Int=100,
  maxinneriter::Int=100,
  eps::Real=1e-12) where {T<:AbstractMatrix{<:Real}}
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
      weights[point_i] = 1 / (norm(point - centroids[:, cluster_i]) + eps)
    end
    # Step 3: update clusters
    for cluster_i in 1:size(centroids, 2)
      # minimize log 2-norm via iterative reweighted least squares (IRLS) for each cluster.
      # an LLM was used to get the pseudocode for the IRLS algorithm. 
      inner_flag = true
      curr_iter = 1
      while inner_flag && curr_iter <= maxinneriter
        idx = [i for i in 1:N if cluster_map[i] == cluster_i]
        if isempty(idx)
          # the cluster is empty. just ignore.
          break
        end
        new_centroid = sum(weights[i] .* dataset[:, i] for i in idx) / sum(weights[i] for i in idx)
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
    maxerr = -Inf
    for i in 1:N
      c = cluster_map[i]
      err = log(norm(dataset[:, i] - centroids[:, c]) + eps)
      maxerr = max(maxerr, err)
    end

    if maxerr < tol
      converged = true
      n_outer_iter = iter
      break
    end
  end

  # calculate inertia
  dist::Real = 0
  for i in 1:size(initialcentroids, 2)
    dist += sum((log ∘ norm).(eachcol(dataset[:, cluster_map.==i] .- centroids[:, i])) .^ 2)
  end
  return KMeansResult(
    centroids,
    cluster_map,
    dist,
    n_outer_iter,
    converged,
    init_method)
end

"""
    KMeansLogAlgorithm(
        dataset::T,
        n_clusters::Int,
        tol::Real;
        maxiter::Int=100,
        maxinneriter::Int=100,
        eps::Real=1e-12
    )

Fields:
- `dataset::Matrix{Float64}`  
    A `dxn` matrix where each column is a point and each row is a feature.
- `initialcentroids::Matrix{Float64}`  
    A `dxk` matrix containing the starting `k` centroids.
- `tol::Real`  
    tolerance threshold to determine convergence. Note that this number is the `log` of the distance from the cluster point.
- `maxiter::Int`  
    Maximum number of iterations.
- `maxinneriter::Int`
    Maximum number of iterations for iterative reweighted least squares(IRLS), which is used to compute the new cluster point.
- `eps`
    epsilon to avoid `log(0)`
"""
struct KMeansLogAlgorithm{T<:AbstractMatrix{<:Real}} <: KMeansAlgorithm
  dataset::T
  n_clusters::Int
  tol::Real
  maxiter::Int
  maxinneriter::Int
  eps::Real

  function KMeansLogAlgorithm(
    dataset::T,
    n_clusters::Int,
    tol::Real;
    maxiter::Int=100,
    maxinneriter::Int=100,
    eps::Real=1e-12
  ) where {T<:AbstractMatrix{<:Real}}
    new{T}(dataset, n_clusters, tol, maxiter, maxinneriter, eps)
  end
end

function kmeans(self::KMeansLogAlgorithm)
  n_points = size(self.dataset, 2)
  if self.n_clusters < 1
    throw(ArgumentError("Number of clusters must be at least 1, currently $(self.n_clusters)"))
  end
  if self.n_clusters > n_points
    throw(ArgumentError("Number of clusters must be less than than the number of data points (data points == $(n_points)), got $(self.n_clusters) requested clusters"))
  end
  idx = randperm(Random.default_rng(), size(self.dataset, 2))[1:self.n_clusters]
  initialcentroids = self.dataset[:, idx]
  return kmeanslog(self.dataset, initialcentroids, :random, self.tol, maxiter=self.maxiter, maxinneriter=self.maxinneriter, eps=self.eps)
end