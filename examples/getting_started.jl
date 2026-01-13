using KMeansClustering
using Random
using Plots

my_rng = MersenneTwister(1234) # Number Generator with fixed seed

X = rand(my_rng, 2, 200) # Create an artificial dataset
cluster_count = 4 # Number of clusters to separate the dataset into

clustering_result = KMeansClustering.kmeans(
    X,                  # Points, column-wise: rows are the features, cols are the points
    cluster_count,
    method=:kmedoids,   # Select the KMeans-method to use
    init=:random,
    maxiter=50,
    tol=1e-4,           # Tolerance of improvement between each iteration.
    rng=my_rng          # Random Number Generator to use
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
