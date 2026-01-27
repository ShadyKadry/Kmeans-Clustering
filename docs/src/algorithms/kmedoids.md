```@meta
CurrentModule = KMeansClustering
```

# K-Medoids Clustering

## Overview

The K-Medoids algorithm is a robust variant of the K-Means algorithm, that, instead of creating artificial cluster centers, uses actual data points as centers. Just as calculating the median is more robust to outliers as the average is, the K-Medoids algorithm is more resistant to noise and outliers than K-Means.
This implementation uses the Partitioning Around Medoids (PAM) approach. See the references below for an explanation.

Comparison to K-Means:

- **Advantages:**
  - Medoids are actual data points, making results directly interpretable
  - More robust to outliers and noise compared to K-Means
  - Works with any distance metric (not limited to Euclidean distance)

- **Disadvantages:**
  - Computationally more expensive than K-Means ($O(k(n-k)^2)$ > $O(n \cdot k \cdot d \cdot i)$) per iteration)
    - where:
      - $n$: number of data points
      - $k$: number of clusters
      - $d$: dimensionality
      - $i$: number of iterations until convergence

## Implementation Details

This implementation is based on the Partitioning Around Medoids (PAM) approach as described by [TU Dortmund: Partitioning Around Medoids (k-Medoids)](https://dm.cs.tu-dortmund.de/mlbits/cluster-kmedoids-intro/).

### Data Format

The algorithm expects data in column-major format:
- **Rows** represent features/dimensions
- **Columns** represent individual data points/observations

## Usage

### Basic Usage with `kmeans` Function

```@example kmedoids_1
using KMeansClustering

# Generate sample data
X = rand(2, 100)  # 2 features, 100 observations

# Perform K-Medoids clustering with 3 clusters
# defaults to euclidian distance
result = kmeans(X, 3, method=:kmedoids)

println("Cluster assignments: ", result.assignments)
println("Medoids: ", result.centers)
println("Total inertia: ", result.inertia)
println("Converged: ", result.converged)
```

### Advanced Usage with Settings Object

For more control over the algorithm, use the `KMedoidsAlgorithm` settings object:

```@example kmedoids_2
using KMeansClustering
using Random

X = rand(2, 100) # Again 2 x 100

settings = KMeansClustering.KMedoidsAlgorithm(
    X,                       # Data matrix
    3;                       # Number of clusters
    max_iter=100,            # Maximum iterations
    tol=1e-4,                # Convergence tolerance
    rng=MersenneTwister(42), # Random number generator
    distance_fun=(a, b) -> sum((a .- b).^2)  # Distance function
    # Alternative distance function examples:
    # distance_fun = (a, b) -> sum(abs.(a .- b))
    # distance_fun = (a, b) -> 1 - (dot(a, b) / (norm(a) * norm(b)))
)

# Run clustering using multiple dispatch
result = kmeans(settings)

println("Cluster assignments: ", result.assignments)
println("Medoids: ", result.centers)
println("Total inertia: ", result.inertia)
println("Converged: ", result.converged)
```

## Parameters

> For the non-overloaded version, see the [main documentation page](../index.md).

```@docs
KMeansClustering.kmeans(::KMedoidsAlgorithm)
KMeansClustering.KMedoidsAlgorithm
```

## Examples

### Example 1: Basic Clustering

```@example kmedoids_3
using KMeansClustering

# Create sample data: 3 Gaussian clusters
data = hcat(
    randn(2, 30) .+ [0.0, 0.0],
    randn(2, 30) .+ [5.0, 5.0],
    randn(2, 30) .+ [10.0, 0.0]
)

# Cluster the data
result = kmeans(data, 3, method=:kmedoids, maxiter=100, tol=1e-4)

println("Number of iterations: ", result.iterations)
println("Converged: ", result.converged)
println("Inertia: ", result.inertia)
```

### Example 3: Custom Distance Metric

```@example kmedoids_4
using KMeansClustering
using LinearAlgebra

X = rand(3, 100)

# Use cosine distance
cosine_distance(a, b) = begin
    dot_product = dot(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    return 1.0 - (dot_product / (norm_a * norm_b))
end

settings = KMedoidsAlgorithm(
    X,
    5;
    distance_fun=cosine_distance
)

result = kmeans(settings)

println("Number of iterations: ", result.iterations)
println("Converged: ", result.converged)
println("Inertia: ", result.inertia)
```

## References

- [TU Dortmund: Partitioning Around Medoids (k-Medoids)](https://dm.cs.tu-dortmund.de/mlbits/cluster-kmedoids-intro/)
- [TU Dortmund: k-means Clustering](https://dm.cs.tu-dortmund.de/mlbits/cluster-kmeans-intro/)


### AI Note
Parts of the text were compiled using generative AI, notably:
- A base text was produced for the Overview and then modified
