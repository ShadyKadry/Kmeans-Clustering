using KMeansClustering
using Test

@testset "simplekmeans" begin

    data = [1.0 1.5 1.5 6.5 5.0 2.0 6.0 2.5 5.0 5.5;
        1.0 3.5 0.5 1.0 1.5 4.0 2.0 3.5 2.0 2.5]
    k = 3

    settings_rand = SimpleKMeansAlgorithm(data, k)
    settings_kmpp = SimpleKMeansAlgorithm(data, k, init_method=:kmeanspp)

    @testset "SimpleKMeansAlgorithm construction" begin
        @test settings_rand.data == data
        @test settings_rand.n_clusters == k
        @test_throws ArgumentError SimpleKMeansAlgorithm(data, 0)
        # more wanted clusters than points 
        @test_throws ArgumentError SimpleKMeansAlgorithm(data, size(data, 2) + 1)
        # invalid init_method
        @test_throws ArgumentError SimpleKMeansAlgorithm(data, k, init_method=:km)
    end

    @testset "simplekmeans function" begin
        centroids = [1.5 2.0 6.0;
            0.5 4.0 2.0]
        expected_assign = [1, 2, 1, 3, 3, 2, 3, 2, 3, 3]
        result = simplekmeans(data, centroids)
        @test all(1 .<= result.assignments .<= k)
        @test result.assignments == expected_assign
        @test size(result.centers, 2) == k
        @test result.init_method == :random
        @test result.converged == true
        # data points and center points have different dimensions
        @test_throws DimensionMismatch simplekmeans(data, centroids[1:end-1, :])
    end

    @testset "kmeans" begin
        @testset "random" begin
            result = kmeans(settings_rand)
            @test size(result.centers, 2) == k
            @test all(1 .<= result.assignments .<= k)
            @test result.init_method == :random
            @test result.converged == true
        end
        @testset "kmeanspp" begin
            result = kmeans(settings_kmpp)
            @test all(1 .<= result.assignments .<= k)
            @test result.init_method == :kmeanspp
            @test result.converged == true
        end
    end
end