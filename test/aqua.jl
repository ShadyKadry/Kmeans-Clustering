using Aqua

@testset "Aqua Quality Checker" begin
    Aqua.test_all(KMeansClustering)
end
