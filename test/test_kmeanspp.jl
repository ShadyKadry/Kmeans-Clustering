using Test
using Random
using KMeansClustering

const KMpp = KMeansClustering.AlgorithmsKMeansPP

@testset "kmeanspp_init" begin
    # Simple dataset: 2D points as columns
    X = [0.0  1.0  2.0  10.0  11.0;
         0.0  0.0  0.0   0.0   0.0]

    k = 3
    rng1 = MersenneTwister(1)
    rng2 = MersenneTwister(1)

    idxs1 = KMpp.kmeanspp_init(X, k; rng=rng1)
    idxs2 = KMpp.kmeanspp_init(X, k; rng=rng2)

    @test idxs1 == idxs2                      # deterministic with same seed
    @test length(idxs1) == k
    @test all(1 .<= idxs1 .<= size(X, 2))
    @test length(unique(idxs1)) == k          # no duplicates

    # Degenerate case: all columns identical should still return distinct indices.
    Xdeg = zeros(2, 10)
    idxs3 = KMpp.kmeanspp_init(Xdeg, 5; rng=MersenneTwister(42))
    @test length(unique(idxs3)) == 5

        # Edge case: k = 1 (single center)
    rng_single = MersenneTwister(7)
    idxs_single = KMpp.kmeanspp_init(X, 1; rng=rng_single)
    @test length(idxs_single) == 1
    @test 1 <= idxs_single[1] <= size(X, 2)

    # Edge case: k = number of observations (should select all columns)
    ncols = size(X, 2)
    rng_all = MersenneTwister(8)
    idxs_all = KMpp.kmeanspp_init(X, ncols; rng=rng_all)
    @test length(idxs_all) == ncols
    @test all(1 .<= idxs_all .<= ncols)
    @test length(unique(idxs_all)) == ncols

    # Invalid k: zero and too large should throw ArgumentError
    @test_throws ArgumentError KMpp.kmeanspp_init(X, 0; rng=MersenneTwister(9))
    @test_throws ArgumentError KMpp.kmeanspp_init(X, ncols + 1; rng=MersenneTwister(10))

end
