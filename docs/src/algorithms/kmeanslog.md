```@meta
CurrentModule = KMeansClustering
```

# K Means with Log Heuristic

## Overview

This is a variant of the [classical kmeans algorithm](simplekmeans.md) that minimizes the logarithm of the distance instead of the distance.

## Implementation Details

### Data Format

The algorithm expects data in column-major format:
- **Rows** represent features/dimensions
- **Columns** represent individual data points/observations

### Initialization

`k` random points are chosen from the dataset as initial centers

### Cluster Computation

The cluster centers are computed by [iteratively reweighted least squares(IRLS)](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares). The maximum number of iterations for each cluster computation can be set with the `maxinnteriter` attribute of `KMeansLogAlgorithm`. 

## Usage

### Basic Usage with `kmeans` Function

```@example kmeanslog_1
using KMeansClustering

# Generate sample data
X = rand(2, 100)  # 2 features, 100 observations

# Perform simple K-Means clustering with 3 clusters
# with tolerance = 0.8
result = kmeans(KMeansLogAlgorithm(X,3,0.8))

println("Cluster assignments: ", result.assignments)
println("Final centroids: ", result.centers)
println("Total inertia: ", result.inertia)
println("Converged: ", result.converged)
```

## Parameters

> For the non-overloaded version, see the [main documentation page](../index.md).

```@docs
KMeansClustering.kmeans(::KMeansLogAlgorithm)
KMeansClustering.KMeansLogAlgorithm
```