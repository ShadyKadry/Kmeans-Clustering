using KMeansClustering
using Test

include("../src/algorithms/kmedoids.jl")

@testset "KMeansClustering.jl" begin
    # Write your tests here.
end



# A simple test distance function
test_dist(a::AbstractVector, b::AbstractVector) = sum((a .- b).^2)

# Dummy data
X = rand(20, 3)

@testset "KMedoids Fit" begin

    settings = KMedoids_init(
        UInt32(4),
        UInt32(20),
        Float32(1e-2),
        MersenneTwister(1234),
        Float32(0.2),
        Float32(0.8),
        test_dist
    )

    medoids = KMedoids_fit(settings, X)

    @test medoids isa Vector{Int}
    @test length(medoids) == Int(settings.n_clusters)

    # Valid index check
    for m in medoids
        @test 1 ≤ m ≤ size(X, 1)
    end
end