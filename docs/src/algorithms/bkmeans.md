```@meta
CurrentModule = KMeansClustering
```

# Bisecting K-Means (BKMeans) Clustering

## Overview

The Bisecting K-Means (BKMeans) algorithm is a divisive hierarchical variant of the K-Means algorithm. Instead of solving the full k-means problem in one step, BKMeans starts with a single cluster containing all data points and repeatedly splits one cluster into two sub-clusters using a 2-means (Lloyd) procedure until the desired number of clusters is reached.

This approach often yields a more stable high-level structure than a single K-Means run, because new clusters are created by refining an existing partition rather than by re-optimizing all k centers at once.

Comparison to standard K-Means:

- **Advantages:**
  - Often less sensitive than one-shot K-Means initialization (especially with multiple restarts per split)
  - Produces an implicit hierarchical structure (sequence of splits)
  - Can yield better local optima in practice by repeatedly refining the "worst" cluster

- **Disadvantages:**
  - Typically more computationally expensive than a single K-Means run (multiple 2-means subproblems)
  - Still depends on initialization inside each 2-means split (mitigated by nstart)

## Implementation Details
This implementation follows a common BKMeans strategy:
  - Which cluster is split: the cluster with the largest SSE (sum of squared errors), provided it contains at least 2 points.
  - How a split is performed: solve a 2-means clustering problem (Lloyd iterations).
  - Multiple restarts per split (nstart): run several 2-means attempts and keep the split with the lowest SSE.
  - 2-means initialization: choose one random point as centroid 1; choose the point farthest from it as centroid 2.


### Data Format

The algorithm expects data in column-major format:
- **Rows** represent features/dimensions
- **Columns** represent individual data points/observations

## Usage

### Basic Usage with `kmeans` Function

```@example
using KMeansClustering
using Random

# Generate sample data
rng = MersenneTwister(42)
X = rand(rng, 2, 100)  # 2 features, 100 observations

# Perform BKMeans clustering with 3 clusters
result = kmeans(X, 3, method=:bkmeans, maxiter=50, tol=1e-4, nstart=10, rng=rng)

println("Cluster assignments: ", result.assignments)
println("Centers: ", result.centers)
println("Total inertia (SSE): ", result.inertia)
println("Converged: ", result.converged)
println("Iterations (sum over all 2-means splits): ", result.iterations)
```

### Advanced Usage with Settings Object

For more control over the algorithm, use the BKMeansAlgorithm settings object:

```@example
using KMeansClustering
using Random

rng = MersenneTwister(123)
X = rand(rng, 2, 120)

settings = KMeansClustering.BKMeansAlgorithm(
    X,                       # Data matrix
    4;                       # Number of clusters
    max_iter=60,             # Maximum iterations per 2-means split
    tol=1e-4,                # Convergence tolerance per 2-means split
    nstart=8,                # Restarts per split (best split wins)
    rng=rng                  # Random number generator
)

# Run clustering using multiple dispatch
result = kmeans(settings)

println("Cluster assignments: ", result.assignments)
println("Centers: ", result.centers)
println("Total inertia (SSE): ", result.inertia)
println("Converged: ", result.converged)
```

## Parameters

> For the non-overloaded version, see the [main documentation page](../index.md).

```@docs
KMeansClustering.kmeans(::BKMeansAlgorithm)
KMeansClustering.BKMeansAlgorithm
```

## Examples

### Example 1: Clustering Three Well-Separated Groups

```@example
using KMeansClustering
using Random

rng = MersenneTwister(7)

# Create sample data: 3 Gaussian clusters
data = hcat(
    randn(rng, 2, 30) .+ [0.0, 0.0],
    randn(rng, 2, 30) .+ [5.0, 5.0],
    randn(rng, 2, 30) .+ [10.0, 0.0]
)

# Cluster the data
result = kmeans(data, 3, method=:bkmeans, maxiter=60, tol=1e-4, nstart=10, rng=rng)

println("Number of iterations: ", result.iterations)
println("Converged: ", result.converged)
println("Inertia (SSE): ", result.inertia)
```

### Example 2: Effect of nstart (More Restarts per Split)

```@example
using KMeansClustering
using Random

rng = MersenneTwister(99)

# Slightly elongated cloud (2×200)
X = vcat(randn(rng, 1, 200), randn(rng, 1, 200) .* 0.3)

res_low  = kmeans(X, 4, method=:bkmeans, maxiter=40, tol=1e-4, nstart=1,  rng=MersenneTwister(1))
res_high = kmeans(X, 4, method=:bkmeans, maxiter=40, tol=1e-4, nstart=20, rng=MersenneTwister(1))

println("Inertia with nstart=1:  ", res_low.inertia)
println("Inertia with nstart=20: ", res_high.inertia)
```

## References
---

## AI Note
- I took the structure from kmedoids.md 
- I wrote the text myself first, then translated and corrected it with AI
- The comments in the examples are also AI-generated.