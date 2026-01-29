# Copyright 2026
# This code was not generated using AI


using KMeansClustering
using Random
using Plots

my_rng = MersenneTwister(1234) # Number Generator with fixed seed

X = rand(my_rng, 2, 200) # Create an artificial dataset
cluster_count = 4 # Number of clusters to separate the dataset into

clustering_result = KMeansClustering.kmeans(
# The algorithm is chosen by selecting the appropriate struct.
# Options are:
# - SimpleKMeansAlgorithm
# - BKMeansAlgorithm
# - KMeansLogAlgorithm
# - KMedoidsAlgorithm

    KMeansClustering.SimpleKMeansAlgorithm(
    # Points, column-wise: rows are the features, cols are the points
    X,
    cluster_count;
    # Select, how the initial centroids should be chosen
    init_method=:random,
    # Maximum number of iterations before the algorithm is aborted
    max_iter=50,
    # Tolerance of improvement between each iteration
    tol=1e-4,
    # Random Number Generator to use
    rng=my_rng
    )
)

@info "Required Iterations: $(clustering_result.iterations)"
@info "Converged: $(clustering_result.converged)"

scatter(
    X[1, :],
    X[2, :],
    group=clustering_result.assignments,
    legend=false
)

scatter!(
    clustering_result.centers[1, :],
    clustering_result.centers[2, :],
    markersize=8,
    marker=:star,
    color=:black
)
