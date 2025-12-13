# Geting Started with KMeansClustering.jl

This is a short tutorial on how to use the KMeansClustering Library and visualize
the results.

## Installation

There are two options:
1. Installation using a local directory
```
    pkg> add <path to library/Kmeans-Clustering>
```
2. Installation using git
```
    pkg> add https://github.com/ShadyKadry/Kmeans-Clustering
```


## Basic API

Importing is as easy as with any other package:

```@example getting_started
    using KMeansClustering

    # Used to select a predictable Random Numbers generator
    using Random

    # For Plotting
    using Plots
```

Only one function is exposed:

```@example getting_started
    my_rng = MersenneTwister(1234) # Number Generator with fixed seed

    X = rand(my_rng, 2, 200) # Create an artificial dataset
    cluster_count = 4 # Number of clusters to separate the dataset into

    clustering_result = KMeansClustering.kmeans(
        X,                  # Points, column-wise: rows are the features, cols are the points
        cluster_count,
        method=:kmedoids,   # Select the KMeans-method to use
        init=:random,       # Select, how the initial centroids should be chosen
        maxiter=50,         # Maximum number of iterations before the algorithm is aborted
        tol=1e-4,           # Tolerance of improvement between each iteration.
        rng=my_rng          # Random Number Generator to use
    )

    @info "Required Iterations: $(clustering_result.iterations)"
    @info "Converged: $(clustering_result.converged)"
```

The return value contains the result:

- `clustering_result.centers`: Matrix of center points
- `clustering_result.assignments`: Vector, that maps each original point to one of the centers in clustering_result.centers
- `clustering_result.iterations`: Number of iterations that was required
- `clustering_result.converged`: If true, the algorithm finished successfully (tol was reached). If false, maxiter was reached and the algorithm aborted

## Plotting


The result of the `kmeans()` function can be directly plotted using `Plots.js`:

```@example getting_started
scatter(
    X[1, :], 
    X[2, :], 
    group=clustering_result.assignments, 
    legend=false
)
```

The centers can be additionally marked:

```@example getting_started
scatter!(
    clustering_result.centers[1, :], 
    clustering_result.centers[2, :],
    markersize=8,
    marker=:star,
    color=:black
)
```

## Full Script

```@example
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
```