using Test
using Random
using KMeansClustering

@testset "Basic KMedoids Tests" begin
    # Synthetic data
    Random.seed!(42) # Pin random values to be reproducible

    cluster1 = randn(2, 20) .- 5.0
    cluster2 = randn(2, 20) .+ 5.0
    X = hcat(cluster1, cluster2)

    clusters = 2

    # Check that the distinct clusters were detected properly
    @testset "Check cluster separation" begin
        settings = KMedoidsAlgorithm(
            X,
            clusters;
            max_iter=100,
            tol=1e-4,
            rng=MersenneTwister(42),
            distance_fun=(a, b) -> sum((a .- b).^2)
        )

        result = kmeans(settings)

        @test result isa KMeansResult
        @test size(result.centers, 2) == clusters
        @test length(result.assignments) == size(X, 2)
        @test result.converged == true
        # Verify that centers are actually points from the original dataset
        for i in 1:clusters
            center = result.centers[:, i]
            @test any(all(center .== X[:, j]) for j in 1:size(X, 2))
        end
    end

    # Check that the single dispatch yields the same results as the multiple dispatch version
    @testset "single dispatch vs multiple dispatch" begin
        base = kmeans(
            X,
            clusters;
            method=:kmedoids,
            maxiter=100,
            tol=1e-4,
            rng=MersenneTwister(42)
        )

        multiple_dispatch = KMedoidsAlgorithm(
            X,
            clusters;
            max_iter=100,
            tol=1e-4,
            rng=MersenneTwister(42)
        )
        multiple_dispatch = kmeans(multiple_dispatch)


        @test base.centers == multiple_dispatch.centers
        @test base.assignments == multiple_dispatch.assignments
        @test base.inertia == multiple_dispatch.inertia
        @test base.iterations == multiple_dispatch.iterations
        @test base.converged == multiple_dispatch.converged
        @test base.init_method == multiple_dispatch.init_method
    end

    # Ensure that the same seed produces the exact same results and we dont use some unknown rng inbetween
    @testset "Reproducibility" begin
        res1 = kmeans(X, 5, method=:kmedoids, rng=MersenneTwister(42))
        res2 = kmeans(X, 5, method=:kmedoids, rng=MersenneTwister(42))

        @test res1.assignments == res2.assignments
        @test res1.centers == res2.centers
        @test res1.inertia == res2.inertia
        @test res1.iterations == res2.iterations
    end

    @testset "Test custom Distance Metrics" begin
        clusters = 3

        settings = KMedoidsAlgorithm(
            X,
            clusters;
            distance_fun=(a, b) -> sum(abs.(a .- b)), # manhattan
            rng=MersenneTwister(42)
        )
        manhattan = kmeans(settings)

        @test manhattan isa KMeansResult
        @test manhattan.converged == true
        @test manhattan.inertia >= 0

        settings = KMedoidsAlgorithm(
            X,
            clusters;
            max_iter=100,
            tol=1e-4,
            rng=MersenneTwister(42)
        )
        euclidian = kmeans(settings)

        @test manhattan.converged == true
        @test euclidian.converged == true

        @test euclidian.centers != manhattan.centers # should be different
        @test euclidian.inertia != manhattan.inertia # should be different
    end

    @testset "Edge Case: Single Cluster" begin
        res_k1 = kmeans(X, 1, method=:kmedoids)
        @test size(res_k1.centers, 2) == 1
        @test all(a == 1 for a in res_k1.assignments)
    end

    @testset "Edge Case: Every point is its own medoid" begin
        res_kn = kmeans(X, size(X, 2), method=:kmedoids)
        @test size(res_kn.centers, 2) == size(X, 2)
        @test length(unique(res_kn.assignments)) == size(X, 2)
    end

    @testset "Bound Checks" begin
        X_small = rand(2, 5)

        @test_throws ArgumentError kmeans(X_small, size(X_small, 2) + 1, method=:kmedoids)

        # Test with settings struct as well
        settings_too_many = KMeansClustering.KMedoidsAlgorithm(
            X_small,
            size(X_small, 2) + 1;
            rng = MersenneTwister(1)
        )
        @test_throws ArgumentError kmeans(settings_too_many)

        # Test case: n_clusters < 1
        @test_throws ArgumentError kmeans(X_small, 0, method=:kmedoids)

        settings_too_few = KMeansClustering.KMedoidsAlgorithm(
            X_small,
            0;
            rng = MersenneTwister(1)
        )
        @test_throws ArgumentError kmeans(settings_too_few)
    end

    # max_iter = 1: Should stop after one iteration
    @testset "Iterations" begin
        res_limited = kmeans(X, 10, method=:kmedoids, maxiter=1)
        @test res_limited.iterations <= 1
    end
end
