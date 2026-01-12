module KMedoids

using Random
using DataStructures: DefaultDict

using ..KMeansClustering: KMeansResult, KMeansAlgorithm, kmeans

"""
    KMedoids_Settings

    Settings specific to the KMedoids algorithm

    Fields:
    - `n_clusters`: Number of clusters that the dataset should be split up into
    - `max_iter`: Maximum number of iterations to run before aborting
    - `tol`: Tolerance for abortion. If the improvement between iterations is smaller than `tol`, the algorithm aborts
    - `rng`: Random Number Generator to use for generating the initial medoid centers
    - `distance_fun`: Cost function to calculate the distance between two points. This function must take two pairs of coordinates and return a number
"""
struct KMedoids_Settings{T<:Function}
    n_clusters::UInt32
    max_iter::UInt32
    tol::Float32
    rng::AbstractRNG
    distance_fun::T
end

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

t_Medoid_Idx = UInt32
t_Medoid_Array = Array{t_Medoid_Idx}

t_Cluster_Map = DefaultDict{t_Medoid_Idx, t_Medoid_Array}
t_Cluster_Weights = DefaultDict{t_Medoid_Idx, Float64}


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

    medoids = t_Medoid_Idx[]

    for _ in 1:n_clusters
        n_med = t_Medoid_Idx(rand(rng, 1:cols))
        if !(n_med in medoids)
            push!(medoids, n_med)
        end
    end

    return medoids
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

    for medoid in medoids
        cluster_distances[medoid] /= length(clusters[medoid])
    end

    return clusters, cluster_distances
end

function calculate_inter_cluster_distance(
    self::KMedoids_Settings,
    data::AbstractMatrix,
    medoid::t_Medoid_Idx,
    cluster_list::t_Medoid_Array
)
    distance = 0.0
    for idx in cluster_list
        distance += get_distance(self, data[:, medoid], data[:, idx])
    end

    return distance / length(cluster_list)
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
        shortest_found = false

        for data_index in clusters[medoid]
            if data_index != medoid
                cluster_list = copy(clusters[medoid])
                idx = findfirst(==(data_index), cluster_list)
                if idx !== nothing
                    cluster_list[idx] = medoid
                end

                new_distance = calculate_inter_cluster_distance(
                    self, data, data_index, cluster_list
                )

                if new_distance < weights[medoid]
                    new_cluster_dist[data_index] = new_distance
                    shortest_found = true
                    break
                end
            end
        end

        if !shortest_found
            new_cluster_dist[medoid] = weights[medoid]
        end
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
    for idx in 1:self.max_iter
        cluster_dist_with_new_medoids = swap_and_recalculate_clusters(self, data, medoids, clusters, weights)

        old_sum = sum_cluster_distances(weights)
        new_sum = sum_cluster_distances(cluster_dist_with_new_medoids)

        if new_sum < old_sum && (old_sum - new_sum) > self.tol
            medoids = collect(t_Medoid_Idx, keys(cluster_dist_with_new_medoids))
            clusters, weights = calculate_clusters(self, data, medoids)
        elseif new_sum < old_sum
            break
        end

        oidx = idx
    end

    return clusters, oidx, self.max_iter != oidx
end

function KMedoids_fit(
    data::AbstractMatrix,
    initial_medoids::t_Medoid_Array;
    init_method::Symbol = :random,
    max_iter::Integer = 100,
    tol::Real = 10e-4,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    distance_fun::T = (a::AbstractVector, b::AbstractVector) -> sum((a .- b).^2)
) where {T<:Function}
    self = KMedoids_Settings(
        UInt32(length(initial_medoids)),
        UInt32(max_iter),
        Float32(tol),
        rng,
        distance_fun,
    )

    clusters, weights = calculate_clusters(self, data, initial_medoids)

    clusters, iterations, converged = update_clusters(self, data, initial_medoids, clusters, weights)

    labels = zeros(Int, size(data, 2))

    for (cluster_id, rows) in clusters
        for r in rows
            labels[r] = cluster_id
        end
    end

    return KMeansResult(
        data[:, collect(keys(clusters))],
        labels,
        -1.0,
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
    Default is squared Euclidean distance.

Returns a `KMeansResult`
"""
function KMedoids_fit(
    data::AbstractMatrix,
    n_clusters::Integer;
    init_method::Symbol = :random,
    max_iter::Integer = 100,
    tol::Real = 10e-4,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    distance_fun::T = (a::AbstractVector, b::AbstractVector) -> sum((a .- b).^2)
) where {T<:Function}
    medoids = initialize_medoids(data, UInt32(n_clusters), rng)

    return KMedoids_fit(
        data,
        medoids,
        init_method=init_method,
        max_iter=max_iter,
        tol=tol,
        rng=rng,
        distance_fun=distance_fun
    )
end

function kmeans(
    settings::KMedoidsAlgorithm
)
    KMedoids_fit(
        settings.data,
        settings.medoids,
        init_method=settings.init_method,
        max_iter=settings.max_iter,
        tol=settings.tol,
        rng=settings.rng,
        distance_fun=settings.distance_fun
    )
end

export KMedoidsAlgorithm, KMedoids_fit

end
