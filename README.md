# KMeansClustering

<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shadykadry.github.io/Kmeans-Clustering/stable/)
-->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shadykadry.github.io/Kmeans-Clustering/dev/)
[![Build Status](https://github.com/shadykadry/Kmeans-Clustering/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shadykadry/Kmeans-Clustering/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/shadykadry/Kmeans-Clustering/branch/main/graph/badge.svg)](https://codecov.io/gh/shadykadry/Kmeans-Clustering)

A Julia package for k-means clustering and related algorithms. This package provides implementations of several clustering algorithms including standard k-means, k-means++, k-medoids, bisecting k-means, and constrained k-means.

## Installation

To install KMeansClustering, use Julia's package manager:

1. Installation using a local directory
```
    pkg> add <path to library/Kmeans-Clustering>
```
2. Installation using git
```
    pkg> add https://github.com/ShadyKadry/Kmeans-Clustering
```

## Quick Start


```julia
using KMeansClustering
using Random

# Generate sample data (features in rows, observations in columns)
Random.seed!(42)
X = randn(2, 100)  # 2 features, 100 observations

# Perform k-means clustering with 3 clusters
result = kmeans(X, 3)

# Access results
println("Cluster centers: ", result.centers)
println("Assignments: ", result.assignments)
println("Inertia: ", result.inertia)
println("Converged: ", result.converged)
println("Iterations: ", result.iterations)
```

See the [examples](https://github.com/ShadyKadry/Kmeans-Clustering/tree/main/examples) directory for more detailed usage examples.

## Algorithms

### K-Means

The standard k-means algorithm using Lloyd's algorithm. Iteratively assigns points to the nearest centroid and updates centroids as the mean of assigned points.

**Method**: `:kmeans`

### K-Medoids

Based on [TU Dortmund: Partitioning Around Medoids (k-Medoids)](https://dm.cs.tu-dortmund.de/mlbits/cluster-kmedoids-intro/). Unlike typical k-means, k-medoids chooses cluster centers from the actual data points instead of calculating artificial centroids. This makes it more interpretable and robust to outliers.

**Method**: `:kmedoids`

### K-Means++

An improved initialization method that spreads out the initial cluster centers, leading to better and more consistent results.

**Initialization**: `:kmeanspp`

### Bisecting K-Means

A hierarchical variant that recursively splits clusters to achieve the desired number of clusters.

### Constrained K-Means

K-means with additional constraints on cluster assignments or properties.
