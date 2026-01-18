using KMeansClustering
using Test

@testset "simplekmeans" begin

    data = [1.0 1.5 1.5 6.5 5.0 2.0 6.0 2.5 5.0 5.5;
        1.0 3.5 0.5 1.0 1.5 4.0 2.0 3.5 2.0 2.5]
    centroids = [1.5 2.0 6.0;
        0.5 4.0 2.0]
    k = 3
    expected_assign = [1, 2, 1, 3, 3, 2, 3, 2, 3, 3]

    settings = SimpleKMeansAlgorithm(data, k)

    @testset "SimpleKMeansAlgorithm construction" begin
        @test settings.data == data
        @test settings.n_clusters == k
        @test_throws ArgumentError SimpleKMeansAlgorithm(data, 0)
        @test_throws ArgumentError SimpleKMeansAlgorithm(data, size(data, 2) + 1)
        @test_throws ArgumentError SimpleKMeansAlgorithm(data, k, init_method=:km)
    end

    @testset "simplekmeans" begin
        result = simplekmeans(data, centroids)
        @test result.assignments == expected_assign
        @test size(result.centers, 2) == k
        @test result.init_method == :random
    end

    @testset "kmeans" begin
        result = kmeans(settings)
        @test result.assignments == expected_assign
        @test size(result.centers, 2) == k
        @test result.init_method == :random
    end
end