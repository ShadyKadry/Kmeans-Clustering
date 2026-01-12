module KMedoids

using Random
using DataStructures: DefaultDict

using ..KMeansClustering: KMeansResult, KMeansAlgorithm


"""
    KMedoidsAlgorithm

    Settings specific to the KMedoids algorithm

    Fields:
    - `data`: Data matrix with features in rows and observations in columns
    - `n_clusters`: Number of clusters that the dataset should be split up into
    - `init_method`: Initialization method for selecting initial medoids (e.g., :random)
    - `max_iter`: Maximum number of iterations to run before aborting
    - `tol`: Tolerance for abortion. If the improvement between iterations is smaller than `tol`, the algorithm aborts
    - `rng`: Random Number Generator to use for generating the initial medoid centers
    - `distance_fun`: Cost function to calculate the distance between two points. This function must take two pairs of coordinates and return a number
"""
struct KMedoidsAlgorithm{T<:Function} <: KMeansAlgorithm
    data::AbstractMatrix
    n_clusters::Integer
    init_method::Symbol
    max_iter::Integer
    tol::Real
    rng::AbstractRNG
    distance_fun::T

    function KMedoidsAlgorithm(
        data::AbstractMatrix,
        n_clusters::Integer;
        init_method::Symbol = :random,
        max_iter::Integer = 100,
        tol::Real = 10e-4,
        rng::AbstractRNG = Random.GLOBAL_RNG,
        distance_fun::T = (a::AbstractVector, b::AbstractVector) -> sum((a .- b).^2)
    ) where {T<:Function}
        new{T}(data, n_clusters, init_method, max_iter, tol, rng, distance_fun)
    end
end

# Internal
struct KMedoids_Settings{T<:Function}
    n_clusters::UInt32
    max_iter::UInt32
    tol::Float32
    rng::AbstractRNG
    distance_fun::T
end

t_Medoid_Idx = UInt32
t_Medoid_Array = Array{t_Medoid_Idx}

t_Cluster_Map = DefaultDict{t_Medoid_Idx,t_Medoid_Array}
t_Cluster_Weights = DefaultDict{t_Medoid_Idx,Float64}


function get_distance(
    self::KMedoids_Settings,
    a::AbstractArray,
    b::AbstractArray
)
    return self.distance_fun(a, b)
end

function get_shortest_distance_to_medoid(
    self::KMedoids_Settings,
    data::AbstractMatrix,
    col_index::Integer,
    medoids::t_Medoid_Array
)
    min_distance = Inf
    current_medoid = 0

    for medoid in medoids
        current_distance = get_distance(self, data[:, medoid], data[:, col_index])
        if current_distance < min_distance
            min_distance = current_distance
            current_medoid = medoid
        end
    end

    return current_medoid, min_distance
end

function initialize_medoids(
    data::AbstractMatrix,
    n_clusters::UInt32,
    rng::AbstractRNG
)
    cols = size(data, 2)

    return t_Medoid_Idx.(randperm(rng, cols)[1:n_clusters])
end

function calculate_clusters(self::KMedoids_Settings, data::AbstractMatrix, medoids::t_Medoid_Array)
    cols = size(data, 2)

    clusters = t_Cluster_Map(() -> t_Medoid_Array[])
    cluster_distances = t_Cluster_Weights(0.0)

    for col in 1:cols
        nearest_medoid, nearest_distance = get_shortest_distance_to_medoid(self, data, col, medoids)
        push!(clusters[nearest_medoid], col)
        cluster_distances[nearest_medoid] += nearest_distance
    end

    return clusters, cluster_distances
end

function calculate_cluster_distance(
    self::KMedoids_Settings,
    data::AbstractMatrix,
    medoid::t_Medoid_Idx,       # Medoid of cluster
    point_list::t_Medoid_Array  # Points in cluster
)
    distance = 0.0
    for idx in point_list
        distance += get_distance(self, data[:, medoid], data[:, idx])
    end

    return distance
end

function swap_and_recalculate_clusters(
    self::KMedoids_Settings,
    data::AbstractMatrix,
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
                    self, data, data_index, clusters[medoid]
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

function sum_cluster_distances(
    cluster_dist::t_Cluster_Weights
)
    return sum(values(cluster_dist))
end

function update_clusters(
    self::KMedoids_Settings,
    data::AbstractMatrix,
    medoids::t_Medoid_Array,
    clusters::t_Cluster_Map,
    weights::t_Cluster_Weights
)
    oidx = 0
    final_sum = sum_cluster_distances(weights)
    for idx in 1:self.max_iter
        oidx = idx

        cluster_dist_with_new_medoids = swap_and_recalculate_clusters(self, data, medoids, clusters, weights)

        old_sum = final_sum
        new_sum = sum_cluster_distances(cluster_dist_with_new_medoids)

        if new_sum < old_sum && (old_sum - new_sum) > self.tol * size(data, 2)
            medoids = collect(t_Medoid_Idx, keys(cluster_dist_with_new_medoids))
            clusters, weights = calculate_clusters(self, data, medoids)
            final_sum = sum_cluster_distances(weights)
        else
            break
        end
    end

    return clusters, oidx, self.max_iter != oidx, final_sum
end

function kmedoids_fit(
    data::AbstractMatrix,
    initial_medoids::t_Medoid_Array;
    init_method::Symbol=:random,
    max_iter::Integer=100,
    tol::Real=10e-4,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    distance_fun::T=(a::AbstractVector, b::AbstractVector) -> sum((a .- b) .^ 2)
) where {T<:Function}
    self = KMedoids_Settings(
        UInt32(length(initial_medoids)),
        UInt32(max_iter),
        Float32(tol),
        rng,
        distance_fun,
    )

    clusters, weights = calculate_clusters(self, data, initial_medoids)

    clusters, iterations, converged, inertia = update_clusters(self, data, initial_medoids, clusters, weights)

    labels = zeros(Int, size(data, 2))

    medoid_to_id = Dict(medoid => i for (i, medoid) in enumerate(keys(clusters)))
    for (cluster_id, rows) in clusters
        for r in rows
            labels[r] = medoid_to_id[cluster_id]
        end
    end

    return KMeansResult(
        data[:, collect(keys(clusters))],
        labels,
        Float64(inertia),
        iterations,
        converged,
        init_method
    )
end

"""
    KMedoids_fit(data, n_clusters; init_method=:random, max_iter=100,
                 tol=1e-4, rng=Random.GLOBAL_RNG, distance_fun=(a,b)->sum((a .- b).^2))

Perform K-Medoids clustering on a dataset.

K-Medoids is a clustering algorithm similar to k-means, but cluster centers
(*medoids*) are always chosen from actual data points, making the algorithm more
robust to noise and outliers.

Implementation is based on the description from:
<http://leicestermath.org.uk/KmeansKmedoids/Kmeans_Kmedoids.html>

# Arguments
- `data::AbstractMatrix`
    A matrix of size `(n_features, n_samples)` where **columns are data points**
    and **rows are features**.
- `n_clusters::Integer`
    Number of clusters (i.e. number of medoids to compute).

# Keyword Arguments
- `init_method::Symbol = :random`
    Method for choosing initial medoids. Currently supported: `:random`.
- `max_iter::Integer = 100`
    Maximum number of refinement iterations.
- `tol::Real = 1e-4`
    Minimum improvement required for convergence.
- `rng::AbstractRNG = Random.GLOBAL_RNG`
    Random number generator.
- `distance_fun::Function`
    A function `dist(a, b)` returning the distance between two sample vectors.
    Default is squared Euclidean distance. Must return a single real number where
    greater values represent greater distnaces

Returns a `KMeansResult`
"""
function kmedoids_fit(
    data::AbstractMatrix,
    n_clusters::Integer;
    init_method::Symbol=:random,
    max_iter::Integer=100,
    tol::Real=1e-4,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    distance_fun::T=(a::AbstractVector, b::AbstractVector) -> sum((a .- b) .^ 2)
) where {T<:Function}
    medoids = initialize_medoids(data, UInt32(n_clusters), rng)

    return kmedoids_fit(
        data,
        medoids,
        init_method=init_method,
        max_iter=max_iter,
        tol=tol,
        rng=rng,
        distance_fun=distance_fun
    )
end

# Single struct overload
function kmedoids_fit(
    settings::KMedoidsAlgorithm
)
    kmedoids_fit(
        settings.data,
        settings.n_clusters,
        init_method=settings.init_method,
        max_iter=settings.max_iter,
        tol=settings.tol,
        rng=settings.rng,
        distance_fun=settings.distance_fun
    )
end

export KMedoidsAlgorithm, kmedoids_fit

end
