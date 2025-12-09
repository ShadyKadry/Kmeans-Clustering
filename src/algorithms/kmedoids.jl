module KMedoids

using Random
using DataStructures

struct KMedoids_Settings{T<:Function}
    n_clusters::UInt32
    max_iter::UInt32
    tol::Float32
    rng::AbstractRNG
    distance_fun::T
end

t_Medoid_Idx = UInt32
t_Medoid_Array = Array{t_Medoid_Idx}

t_Cluster_Map = DefaultDict{t_Medoid_Idx, t_Medoid_Array}
t_Cluster_Weights = DefaultDict{t_Medoid_Idx, Float64}

function KMedoids_init(
    n_clusters::UInt32,
    max_iter::UInt32,
    tol::Float32,
    rng::AbstractRNG,
    distance_fun::T
) where {T<:Function}
    return KMedoids_Settings(
        n_clusters,
        max_iter,
        tol,
        rng,
        distance_fun,
    )
end

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
    row_index::Integer,
    medoids::t_Medoid_Array
)
    min_distance = Inf
    current_medoid = 0

    for medoid in medoids
        current_distance = get_distance(self, data[medoid, :], data[row_index, :])
        if current_distance < min_distance
            min_distance = current_distance
            current_medoid = medoid
        end
    end

    return current_medoid, min_distance
end

function initialize_medoids(
    self::KMedoids_Settings,
    data::AbstractMatrix
)
    rows, _ = size(data)

    medoids = t_Medoid_Idx[]

    for _ in 1:self.n_clusters
        n_med = t_Medoid_Idx(rand(self.rng, 1:rows))
        if !(n_med in medoids)
            push!(medoids, n_med)
        end
    end

    return medoids
end

function calculate_clusters(self::KMedoids_Settings, data::AbstractMatrix, medoids::t_Medoid_Array)
    rows, _ = size(data)

    clusters = t_Cluster_Map(() -> t_Medoid_Array[])
    cluster_distances = t_Cluster_Weights(0.0)

    for row in 1:rows
        nearest_medoid, nearest_distance = get_shortest_distance_to_medoid(self, data, row, medoids)
        push!(clusters[nearest_medoid], row)
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
        distance += get_distance(self, data[medoid, :], data[idx, :])
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
    self::KMedoids_Settings,
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
    for idx in 1:self.max_iter
        cluster_dist_with_new_medoids = swap_and_recalculate_clusters(self, data, medoids, clusters, weights)
        
        old_sum = sum_cluster_distances(self, weights)
        new_sum = sum_cluster_distances(self, cluster_dist_with_new_medoids)

        # @info old_sum, new_sum
        if new_sum < old_sum && (old_sum - new_sum) > self.tol
            medoids = collect(t_Medoid_Idx, keys(cluster_dist_with_new_medoids))
            clusters, weights = calculate_clusters(self, data, medoids)
        else
            @info idx
            break
        end
    end

    return medoids, clusters, weights
end

function KMedoids_fit(
    self::KMedoids_Settings,
    data::AbstractMatrix
)
    medoids = initialize_medoids(self, data)

    clusters, weights = calculate_clusters(self, data, medoids)

    return update_clusters(self, data, medoids, clusters, weights)
end

export KMedoids_init
export KMedoids_fit

end
