using KMeansClustering
using Test


@testset "KMeansClustering.jl" begin
    # Write your tests here.
    include("test_kmeanspp.jl")
    include("test_kmedoids.jl")
end
