using Random: AbstractRNG, GLOBAL_RNG, randperm
using DataStructures: DefaultDict

using ..KMeansClustering: KMeansResult, KMeansAlgorithm


"""
    KMedoidsAlgorithm

    Settings specific to the KMedoids algorithm

    Fields:
    - `data`: Data matrix with features in rows and observations in columns
    - `n_clusters`: Number of clusters that the dataset should be split up into
    - `max_iter`: Maximum number of iterations to run before aborting
    - `tol`: Tolerance for abortion. If the improvement between iterations is smaller than `tol`, the algorithm aborts
    - `rng`: Random Number Generator to use for generating the initial medoid centers
    - `distance_fun`: Cost function to calculate the distance between two points. This function must take two pairs of coordinates and return a number
"""
struct KMedoidsAlgorithm{T<:Function, R <: AbstractMatrix{<:Real}, K <: AbstractRNG} <: KMeansAlgorithm
    data::R
    n_clusters::UInt32
    max_iter::UInt32
    tol::Float64
    rng::K
    distance_fun::T

    function KMedoidsAlgorithm(
        data::R,
        n_clusters::Integer;
        max_iter::Integer = 100,
        tol::Real = 10e-4,
        rng::K = GLOBAL_RNG,
        distance_fun::T = (a::AbstractVector, b::AbstractVector) -> sum((a .- b).^2)
    ) where {T<:Function, R <: AbstractMatrix{<:Real}, K <: AbstractRNG}
        new{T, R, K}(data, UInt32(n_clusters), UInt32(max_iter), Float64(tol), rng, distance_fun)
    end
end

t_Medoid_Idx = UInt32
t_Medoid_Array = Array{t_Medoid_Idx}

t_Cluster_Map = DefaultDict{t_Medoid_Idx,t_Medoid_Array}
t_Cluster_Weights = DefaultDict{t_Medoid_Idx,Float64}

# Find the nearest medoid and its distance for a specific data point
function get_shortest_distance_to_medoid(
    self::KMedoidsAlgorithm,
    col_index::Integer,
    medoids::t_Medoid_Array
)
    min_distance = Inf
    current_medoid = 0

    for medoid in medoids
        current_distance = self.distance_fun(self.data[:, medoid], self.data[:, col_index])
        if current_distance < min_distance
            min_distance = current_distance
            current_medoid = medoid
        end
    end

    return current_medoid, min_distance
end

# Randomly select initial medoid indices
function initialize_medoids(
    self::KMedoidsAlgorithm
)
    return t_Medoid_Idx.(randperm(self.rng, size(self.data, 2))[1:self.n_clusters])
end

# Assign all data points to their closest medoids and calculate the cluster internal distance for each
function calculate_clusters(
    self::KMedoidsAlgorithm,
    medoids::t_Medoid_Array
)
    clusters = t_Cluster_Map(() -> t_Medoid_Array[])
    cluster_distances = t_Cluster_Weights(0.0)

    for col in axes(self.data, 2)
        nearest_medoid, nearest_distance = get_shortest_distance_to_medoid(self, col, medoids)
        push!(clusters[nearest_medoid], col)
        cluster_distances[nearest_medoid] += nearest_distance
    end

    return clusters, cluster_distances
end

# Calculate the sum of distances from a newly selected medoid to all points assigned to its cluster
function calculate_cluster_distance(
    self::KMedoidsAlgorithm,
    medoid::t_Medoid_Idx,       # Medoid of cluster
    point_list::t_Medoid_Array  # Points in cluster
)
    distance = 0.0
    for idx in point_list
        distance += self.distance_fun(self.data[:, medoid], self.data[:, idx])
    end

    return distance
end

# Iterate through clusters and swap current medoids with cluster points if it reduces the total distance
function swap_and_recalculate_clusters(
    self::KMedoidsAlgorithm,
    medoids::t_Medoid_Array,
    clusters::t_Cluster_Map,
    weights::t_Cluster_Weights
)
    new_cluster_dist = t_Cluster_Weights(0.0)

    for medoid in medoids
        best_medoid = medoid
        best_distance = weights[medoid]

        for data_index in clusters[medoid]
            if data_index != medoid
                new_distance = calculate_cluster_distance(
                    self, data_index, clusters[medoid]
                )

                if new_distance < best_distance
                    best_medoid = data_index
                    best_distance = new_distance
                end
            end
        end

        new_cluster_dist[best_medoid] = best_distance
    end

    return new_cluster_dist
end

# Calculates a metric that describes the quality of our current clustering
function sum_cluster_distances(
    cluster_dist::t_Cluster_Weights
)
    return sum(values(cluster_dist))
end

# Perform the iterative loop to update medoids and cluster assignments until we are happy with the result
function update_clusters(
    self::KMedoidsAlgorithm,
    medoids::t_Medoid_Array,
    clusters::t_Cluster_Map,
    weights::t_Cluster_Weights
)
    oidx = 0
    final_sum = sum_cluster_distances(weights)
    for idx in 1:self.max_iter
        oidx = idx

        cluster_dist_with_new_medoids = swap_and_recalculate_clusters(self, medoids, clusters, weights)

        old_sum = final_sum
        new_sum = sum_cluster_distances(cluster_dist_with_new_medoids)

        # Scale the tolerance with the size of the data points
        if new_sum < old_sum && (old_sum - new_sum) > self.tol * size(self.data, 2)
            medoids = collect(t_Medoid_Idx, keys(cluster_dist_with_new_medoids))
            clusters, weights = calculate_clusters(self, medoids)
            final_sum = sum_cluster_distances(weights)
        else
            break
        end
    end

    return clusters, oidx, self.max_iter != oidx, final_sum
end

#     kmedoids_fit(data, n_clusters; max_iter=100, tol=1e-4, rng=GLOBAL_RNG, distance_fun=(a,b)->sum((a .- b).^2))
#
# Perform K-Medoids clustering on a dataset.
#
# K-Medoids is a clustering algorithm similar to k-means, but cluster centers
# ("medoids") are always chosen from actual data points, making the algorithm more
# robust to noise and outliers.
#
# Implementation is based on the description from:
# <http://leicestermath.org.uk/KmeansKmedoids/Kmeans_Kmedoids.html>
#
# This implementation chooses the initial medoids at random.
#
# # Arguments
# - `data::AbstractMatrix`
#     A matrix of size `(n_features, n_samples)` where **columns are data points**
#     and **rows are features**.
# - `n_clusters::Integer`
#     Number of clusters (i.e. number of medoids to compute).
#
# # Keyword Arguments
# - `max_iter::Integer = 100`
#     Maximum number of refinement iterations.
# - `tol::Real = 1e-4`
#     Minimum improvement required for convergence.
# - `rng::AbstractRNG = Random.GLOBAL_RNG`
#     Random number generator.
# - `distance_fun::Function`
#     A function `dist(a, b)` returning the distance between two sample vectors.
#     Default is squared Euclidean distance. Must return a single real number where
#     greater values represent greater distnaces
#
# Returns a `KMeansResult`
function kmedoids_fit(
    data::AbstractMatrix{<:Real},
    n_clusters::Integer;
    max_iter::Integer=100,
    tol::Real=1e-4,
    rng::AbstractRNG=GLOBAL_RNG,
    distance_fun::T=(a::AbstractVector, b::AbstractVector) -> sum((a .- b) .^ 2)
) where {T<:Function}
    return kmeans(
        KMedoidsAlgorithm(
            data,
            n_clusters;
            max_iter,
            tol,
            rng,
            distance_fun
        )
    )
end

"""
    kmeans(KMedoidsAlgorithm)

    Entry point for K-Medoids clustering using a settings object instead.

# Arguments
- `KMedoidsAlgorithm`: Settings object. See object description for more information

# Returns
A `KMeansResult` containing the clustering results.

# Example
```julia
settings = KMeansClustering.KMedoidsAlgorithm(
    X,                  # Points, column-wise: rows are the features, cols are the points
    cluster_count;
    init_method=:random,
    max_iter=50,
)
result = KMeansClustering.kmeans(settings)
```

See also: [`kmeans(X, k; method=:kmeans, init=:random, maxiter=100, tol=1e-4, rng=GLOBAL_RNG)`](@ref)
"""
function kmeans(
    self::KMedoidsAlgorithm
)
    n_points = size(self.data, 2)
    if self.n_clusters < 1
        throw(ArgumentError("Number of clusters must be at least 1, currently $(self.n_clusters)"))
    end
    if self.n_clusters > n_points
        throw(ArgumentError("Number of clusters must be less than than the number of data points (data points == $(n_points)), got $(self.n_clusters) requested clusters"))
    end

    initial_medoids = initialize_medoids(self)

    clusters, weights = calculate_clusters(self, initial_medoids)

    clusters, iterations, converged, inertia = update_clusters(self, initial_medoids, clusters, weights)


    labels = zeros(Int, size(self.data, 2))

    medoid_to_id = Dict(medoid => i for (i, medoid) in enumerate(keys(clusters)))
    for (cluster_id, rows) in clusters
        for r in rows
            labels[r] = medoid_to_id[cluster_id]
        end
    end

    return KMeansResult(
        self.data[:, collect(keys(clusters))],
        labels,
        Float64(inertia),
        iterations,
        converged,
        :random
    )
end
