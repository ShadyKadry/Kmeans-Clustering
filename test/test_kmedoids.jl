using Test
using Random
using KMeansClustering

@testset "KMedoids" begin

    @testset "Basic functionality - well-separated clusters" begin
        # Create 3 well-separated clusters
        Random.seed!(42)
        cluster1 = [randn(2) .+ [0.0, 0.0] for _ in 1:10]
        cluster2 = [randn(2) .+ [10.0, 10.0] for _ in 1:10]
        cluster3 = [randn(2) .+ [0.0, 10.0] for _ in 1:10]

        data = hcat(cluster1..., cluster2..., cluster3...)

        result = kmeans(data, 3, method=:kmedoids, rng=MersenneTwister(123))

        @test length(result.assignments) == 30
        @test all(1 .<= result.assignments .<= 3)
        @test size(result.centers, 2) == 3
        @test size(result.centers, 1) == 2
        @test result.iterations >= 1
    end



    @testset "Medoids are actual data points" begin
        # Key property of K-Medoids: centers must be from the dataset
        data = [1.0 2.0 3.0 10.0 11.0 12.0;
            1.0 1.5 1.2 10.0 10.5 10.2]

        result = kmeans(data, 2, method=:kmedoids, rng=MersenneTwister(42))

        # Check that each center column appears in the original data
        for i in 1:size(result.centers, 2)
            center = result.centers[:, i]
            found = false
            for j in 1:size(data, 2)
                if all(center .≈ data[:, j])
                    found = true
                    break
                end
            end
            @test found
        end
    end

    @testset "Single cluster" begin
        data = randn(3, 20)
        result = kmeans(data, 1, method=:kmedoids, rng=MersenneTwister(1))

        @test all(result.assignments .== 1)
        @test size(result.centers, 2) == 1
        @test length(result.assignments) == 20
    end

    @testset "k equals number of points" begin
        data = [1.0 2.0 3.0 4.0;
            5.0 6.0 7.0 8.0]
        k = 4

        result = kmeans(data, k, method=:kmedoids, rng=MersenneTwister(2))

        @test length(unique(result.assignments)) == k
        @test size(result.centers, 2) == k
    end

    @testset "Deterministic with fixed RNG" begin
        data = randn(MersenneTwister(100), 4, 50)

        result1 = kmeans(data, 3, method=:kmedoids, rng=MersenneTwister(123))
        result2 = kmeans(data, 3, method=:kmedoids, rng=MersenneTwister(123))

        @test result1.assignments == result2.assignments
        @test result1.centers ≈ result2.centers
        @test result1.iterations == result2.iterations
    end

    @testset "All assignments are valid" begin
        data = randn(MersenneTwister(50), 5, 100)
        k = 4

        result = kmeans(data, k, method=:kmedoids, rng=MersenneTwister(10))

        @test length(result.assignments) == 100
        @test all(result.assignments .>= 1)
        @test all(result.assignments .<= k)
        @test minimum(result.assignments) == 1
        @test maximum(result.assignments) == k
    end

    @testset "Inertia is non-negative" begin
        data = randn(MersenneTwister(77), 3, 30)
        result = kmeans(data, 3, method=:kmedoids, rng=MersenneTwister(88))

        @test result.inertia >= 0.0
        @test !isnan(result.inertia)
        @test !isinf(result.inertia)
    end

    @testset "Convergence behavior" begin
        # Simple data that should converge quickly
        data = [0.0 0.1 0.2 5.0 5.1 5.2;
            0.0 0.1 0.0 5.0 5.1 5.0]

        result = kmeans(data, 2, method=:kmedoids, maxiter=100, tol=1e-4, rng=MersenneTwister(99))

        @test result.converged || result.iterations < 100
    end

    @testset "Max iterations respected" begin
        data = randn(MersenneTwister(55), 2, 50)
        max_iter = 5

        result = kmeans(data, 3, method=:kmedoids, maxiter=max_iter, rng=MersenneTwister(66))

        @test result.iterations <= max_iter
    end

    @testset "Custom distance function - Manhattan" begin
        # Use direct kmedoids_fit call to access distance_fun parameter
        using KMeansClustering.KMedoids: kmedoids_fit

        data = [0.0 1.0 2.0 10.0 11.0;
            0.0 0.0 0.0 5.0 5.0]

        manhattan_dist(a, b) = sum(abs.(a .- b))

        result = kmedoids_fit(
            data, 2,
            distance_fun=manhattan_dist,
            rng=MersenneTwister(111)
        )

        @test length(result.assignments) == 5
        @test size(result.centers, 2) == 2
    end

    @testset "Identical points" begin
        # All points are the same
        data = repeat([5.0, 3.0], 1, 10)

        result = kmeans(data, 2, method=:kmedoids, rng=MersenneTwister(200))

        @test length(result.assignments) == 10
        @test all(1 .<= result.assignments .<= 2)
    end

    @testset "High-dimensional data" begin
        data = randn(MersenneTwister(300), 20, 50)
        k = 5

        result = kmeans(data, k, method=:kmedoids, rng=MersenneTwister(301))

        @test size(result.centers, 1) == 20
        @test size(result.centers, 2) == k
        @test length(result.assignments) == 50
    end

    @testset "Two-point dataset" begin
        data = [1.0 2.0;
            1.0 2.0]

        result = kmeans(data, 2, method=:kmedoids, rng=MersenneTwister(400))

        @test length(unique(result.assignments)) == 2
        @test size(result.centers, 2) == 2
    end

    @testset "Result structure completeness" begin
        data = randn(MersenneTwister(500), 3, 25)
        result = kmeans(data, 4, method=:kmedoids, init=:random, rng=MersenneTwister(501))

        @test isa(result.centers, Matrix)
        @test isa(result.assignments, Vector{Int})
        @test isa(result.inertia, Real)
        @test isa(result.iterations, Int)
        @test isa(result.converged, Bool)
        @test result.init_method == :random
    end

    @testset "Each cluster has at least one point" begin
        # Create data where we know k clusters should form
        data = hcat(
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],  # cluster 1
            [5.0, 5.0], [5.1, 5.1], [5.0, 5.2],  # cluster 2
            [10.0, 0.0], [10.1, 0.1]              # cluster 3
        )

        result = kmeans(data, 3, method=:kmedoids, rng=MersenneTwister(600))

        # Check that all cluster IDs from 1 to k appear
        cluster_counts = [count(==(i), result.assignments) for i in 1:3]
        @test all(cluster_counts .> 0)
    end

    @testset "Tolerance affects convergence" begin
        data = randn(MersenneTwister(700), 2, 30)

        # Very loose tolerance - should converge faster
        result_loose = kmeans(data, 3, method=:kmedoids, tol=1.0, maxiter=100, rng=MersenneTwister(701))

        # Tight tolerance - may need more iterations
        result_tight = kmeans(data, 3, method=:kmedoids, tol=1e-8, maxiter=100, rng=MersenneTwister(701))

        # Loose tolerance typically converges in fewer iterations
        @test result_loose.iterations <= result_tight.iterations || result_tight.iterations == 100
    end

end
