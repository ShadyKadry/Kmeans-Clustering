using Test
using Random
using KMeansClustering

@testset "kmeans test" begin
  @testset "number of centers should be the same as the initial centroids" begin
    # Generate sample data (features in rows, observations in columns)
    Random.seed!(42)
    X = randn(2, 100)  # 2 features, 100 observations

    # Perform k-means clustering with 3 clusters
    k = 3

    setting = KMeansLogAlgorithm(X, k, 1)
    # tol = 1
    converged = KMeansClustering.kmeans(setting)
    @test size(converged.centers, 2) == k
  end

  @testset "inertia should be smaller if tolerance is much smaller and does not converge" begin
    # Generate sample data (features in rows, observations in columns)
    Random.seed!(42)
    X = randn(2, 100)  # 2 features, 100 observations

    # Perform k-means clustering with 3 clusters
    k = 3

    # tol = 1
    converged = KMeansClustering.kmeans(KMeansLogAlgorithm(X, k, 1))
    # tol = 0
    notconverged = KMeansClustering.kmeans(KMeansLogAlgorithm(X, k, 0))
    @test converged.converged
    @test !notconverged.converged
    @test converged.inertia > notconverged.inertia
  end

  @testset "inertia should be reasonably close to a non-converging result with an optimized tolerance parameter" begin
    # Generate sample data (features in rows, observations in columns)
    Random.seed!(42)
    X = randn(2, 100)  # 2 features, 100 observations

    # Perform k-means clustering with 3 clusters
    k = 3

    # tol = 0.75
    converged = KMeansClustering.kmeans(KMeansLogAlgorithm(X, k, 0.75))
    # tol = 0
    notconverged = KMeansClustering.kmeans(KMeansLogAlgorithm(X, k, 0))
    @test converged.converged
    @test !notconverged.converged
    @test notconverged.inertia < converged.inertia && converged.inertia < notconverged.inertia * 2
  end

  @testset "excessively large tolerance produces unusable result" begin
    # Generate sample data (features in rows, observations in columns)
    Random.seed!(42)
    X = randn(2, 100)  # 2 features, 100 observations

    # Perform k-means clustering with 3 clusters
    k = 3

    # tol = 1e10
    unusable = KMeansClustering.kmeans(KMeansLogAlgorithm(X, k, 1e10))
    print(unusable)
    # tol = 0
    notconverged = KMeansClustering.kmeans(KMeansLogAlgorithm(X, k, 0))
    @test unusable.converged
    @test !notconverged.converged
    @test unusable.inertia / notconverged.inertia > 5
  end
end