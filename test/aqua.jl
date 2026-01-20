using Aqua

@testset "Aqua Quality Checker" begin
    using Aqua
    Aqua.test_all(KMeansClustering)
end
