using KMeansClustering
using Test

@testset "simplekmeans" begin

    Random.seed!(42)
    # create two clusters that are definitely far apart
    cluster1 = rand(2, 20) .- 5.0
    cluster2 = rand(2, 20) .+ 5.0
    data = hcat(cluster1, cluster2)
    k = 2

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
        # check if clusters are separated correctly 
        cols = shuffle(1:size(cluster1, 2))[1:2]
        centroids = hcat(cluster1[:, cols[1]:cols[1]], cluster2[:, cols[2]:cols[2]])

        result = simplekmeans(data, centroids)
        @test all(1 .<= result.assignments .<= k)
        # check that first 20 points are all assigned to same cluster
        @test length(unique(result.assignments[1:20])) == 1
        # check that last 20 points are all assigned to same cluster
        @test length(unique(result.assignments[21:end])) == 1
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

    @testset "edge cases" begin
        # only one cluster
        result1 = kmeans(data, 1)
        @test all(result1.assignments .== 1)
        @test size(result1.centers, 2) == 1
        # every point is a cluster
        result2 = kmeans(data, size(data, 2))
        @test length(unique(result2.assignments)) == size(data, 2)
        @test size(result2.centers, 2) == size(data, 2)
    end
end