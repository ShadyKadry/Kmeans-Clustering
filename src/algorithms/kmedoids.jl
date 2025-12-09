module KMedoids

using Random

struct KMedoids_Settings{T<:Function}
    n_clusters::UInt32
    max_iter::UInt32
    tol::Float32
    rng::AbstractRNG
    start_prob::Float32
    end_prob::Float32
    distance_fun::T
end



function KMedoids_init(
    n_clusters::UInt32,
    max_iter::UInt32,
    tol::Float32,
    rng::AbstractRNG,
    start_prob::Float32,
    end_prob::Float32,
    distance_fun::T
) where {T<:Function}
    return KMedoids_Settings(
        n_clusters,
        max_iter,
        tol,
        rng,
        start_prob,
        end_prob,
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
    medoids::Vector{UInt32}
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

    return min_distance
end


function select_distant_medoid(
    distance_indices::Vector{Int64},
    start_prob::AbstractFloat,
    end_prob::AbstractFloat
)

    n = length(distance_indices)

    start_index = max(1, round(UInt32, start_prob * n))
    end_index = max(start_index, round(UInt32, end_prob * n))

    return distance_indices[rand(start_index:end_index)]
end

function find_distant_medoid(
    self::KMedoids_Settings,
    data::AbstractMatrix,
    medoids::Vector{UInt32}
)
    rows, cols = size(data)

    distances = Vector{Float64}(undef, rows)

    for row in 1:rows
        dist = get_shortest_distance_to_medoid(self, data, row, medoids)
        distances[row] = dist
    end

    distance_indices = sortperm(distances)   # indices sorted by distance
    chosen = select_distant_medoid(distance_indices, self.start_prob, self.end_prob)

    return chosen
end

function initialize_medoids(
    self::KMedoids_Settings,
    data::AbstractMatrix
)
    rows, cols = size(data)

    medoids = UInt32[]

    push!(medoids, UInt32(rand(self.rng, 1:rows)))

    for i = 2:self.n_clusters
        push!(medoids, find_distant_medoid(self, data, medoids))
    end

    return medoids
end

function calculate_clusters()

end

function update_clusters()


end

function KMedoids_fit(
    self::KMedoids_Settings,
    data::AbstractMatrix
)
    medoids = initialize_medoids(self, data)

    return medoids
end

export KMedoids_init
export KMedoids_fit

end

using .KMedoids
using Random

test_dist(a::AbstractVector, b::AbstractVector) = sum((a .- b).^2)

# Dummy data
X = rand(20, 3)

settings = KMedoids.KMedoids_init(
    UInt32(4),
    UInt32(20),
    Float32(1e-2),
    MersenneTwister(1234),
    Float32(0.2),
    Float32(0.8),
    test_dist
)

medoids = KMedoids.KMedoids_fit(settings, X)

@info medoids