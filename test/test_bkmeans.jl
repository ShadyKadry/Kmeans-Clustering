using KMeansClustering
using Random
using Test

# AI Note:
# - The Naming of the tests and how to structure them were written with the help of AI
# The code itself was not written by any form of AI
@testset "BKMeans" begin

    @testset "BKMeansAlgorithm: construction & argument validation" begin
        X = rand(2, 5)

        s = BKMeansAlgorithm(X, 2; max_iter=10, tol=1e-4, nstart=3, rng=MersenneTwister(1))
        @test s.data == X
        @test s.n_clusters == UInt32(2)
        @test s.max_iter == UInt32(10)
        @test s.nstart == UInt32(3)
        @test isfinite(s.tol)

        @test_throws ArgumentError BKMeansAlgorithm(X, 0)
        @test_throws ArgumentError BKMeansAlgorithm(X, 6)            # k > N
        @test_throws ArgumentError BKMeansAlgorithm(X, 2; nstart=0)
        @test_throws ArgumentError BKMeansAlgorithm(X, 2; max_iter=0)
    end

    @testset "Low-level helpers: _sqdist and _cluster_sse" begin
        x = [1.0, 2.0]
        y = [4.0, 6.0]
        @test KMeansClustering._sqdist(x, y) == 25.0

        dataset = [0.0 1.0 2.0;
                   0.0 0.0 0.0]
        centroid = [1.0, 0.0]
        idxs = [1, 2, 3]
        @test KMeansClustering._cluster_sse(dataset, idxs, centroid) == 2.0
    end

    @testset "_init_two_centroids: normal behavior & error path" begin
        rng = MersenneTwister(7)
        X = [0.0  10.0  20.0;
             0.0   0.0   0.0]  # 2×3

        settings = BKMeansAlgorithm(X, 2; rng=rng, nstart=1)
        init = KMeansClustering._init_two_centroids(settings, X)

        @test size(init) == (2, 2)
        @test init[:, 1] != init[:, 2]   # two distinct points

        p1 = init[:, 1]
        dists = [KMeansClustering._sqdist(view(X, :, i), p1) for i in 1:size(X, 2)]
        maxd = maximum(dists)
        d2 = KMeansClustering._sqdist(init[:, 2], p1)
        @test isapprox(d2, maxd; atol=1e-12, rtol=0.0) || d2 > maxd - 1e-12

        # Error path: cannot initialize 2 centroids if n < 2
        X1 = reshape([1.0, 2.0], 2, 1) # 2×1
        @test_throws ErrorException KMeansClustering._init_two_centroids(settings, X1)
    end

    @testset "_simplekmeans: dimension error, empty-cluster safeguard, converged & non-converged paths" begin
        data = rand(2, 3)
        badC = rand(3, 2)
        @test_throws ErrorException KMeansClustering._simplekmeans(Float64.(data), Float64.(badC), 5, 1e-4)

        data2 = zeros(2, 6)
        C = [0.0 100.0;
             0.0 100.0]
        c2, a2, it2, conv2 = KMeansClustering._simplekmeans(Float64.(data2), Float64.(C), 10, 1e-12)

        @test conv2 == true
        @test it2 == 1
        @test all(a2 .== 1)
        @test c2[:, 2] == C[:, 2]  # empty cluster -> centroid stays unchanged

        data3 = [0.0  0.0  10.0  10.0;
                 0.0  1.0   0.0   1.0]
        C3 = [-5.0  20.0;
               0.0   0.0]
        c3, a3, it3, conv3 = KMeansClustering._simplekmeans(Float64.(data3), Float64.(C3), 1, 0.0)

        @test conv3 == false
        @test it3 == 1
        @test all(1 .<= a3 .<= 2)
    end

    @testset "_bkmeans: normal operation + wrapper + high-level API" begin
        rng = MersenneTwister(42)
        data = hcat(
            randn(rng, 2, 30) .+ [0.0, 0.0],
            randn(rng, 2, 30) .+ [5.0, 5.0],
            randn(rng, 2, 30) .+ [10.0, 0.0],
        )

        k = 3
        settings = BKMeansAlgorithm(data, k; max_iter=50, tol=1e-4, nstart=5, rng=MersenneTwister(123))

        res = kmeans(settings)
        @test size(res.centers, 2) == k
        @test length(res.assignments) == size(data, 2)
        @test all(1 .<= res.assignments .<= k)
        @test res.inertia >= 0
        @test res.init_method == :bkmeans

        res2 = KMeansClustering.bkmeans_fit(data, k; max_iter=30, tol=1e-4, nstart=3, rng=MersenneTwister(777))
        @test size(res2.centers, 2) == k
        @test res2.init_method == :bkmeans

        res3 = kmeans(data, k; method=:bkmeans, maxiter=30, tol=1e-4, nstart=3, rng=MersenneTwister(888))
        @test size(res3.centers, 2) == k
        @test res3.init_method == :bkmeans
    end

    @testset "_bkmeans: invalid splits -> continue + fallback path (isempty(best_assign2))" begin
        data = zeros(2, 8)
        settings = BKMeansAlgorithm(data, 3; max_iter=10, tol=1e-6, nstart=4, rng=MersenneTwister(999))

        res = kmeans(settings)
        @test size(res.centers, 2) == 3
        @test all(1 .<= res.assignments .<= 3)

        @test res.inertia == 0.0

        @test res.converged == false
        @test res.iterations == 0
        @test res.init_method == :bkmeans
    end

    @testset "_bkmeans: force 'Cannot bisect further' error branch via direct call" begin
        X = [0.0 1.0 2.0;
             0.0 0.0 0.0]  # N=3

        settings = BKMeansAlgorithm(X, 2; max_iter=5, tol=1e-6, nstart=1, rng=MersenneTwister(1))
        dataset = Float64.(X)

        @test_throws ErrorException KMeansClustering._bkmeans(settings, dataset, 4, 5, 1e-6; nstart=1)
    end
end