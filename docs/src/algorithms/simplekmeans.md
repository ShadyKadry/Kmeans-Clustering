```@meta
CurrentModule = KMeansClustering
```

# Simple KMeans Clustering

## Overview

The simple K-Means algorithm performs clustering on a dataset following Lloyd's algorithm. Starting with `k` initial centers, in each iteration step every data point gets assigned to a cluster based on the nearest given center and the centroid gets updated by calculating the mean of each cluster. 

## Implementation Details

### Data Format

The algorithm expects data in column-major format:
- **Rows** represent features/dimensions
- **Columns** represent individual data points/observations

### Initialization

The algorithm supports two kinds of initialization:
- `:random` chooses `k` random points from the dataset as initial centers, also called Forgy method
- `:kmeanspp` selects `k` initial centroids using the k-means++ heuristic

## Usage

### Basic Usage with `kmeans` Function

```@example simplekmeans_1
using KMeansClustering

# Generate sample data
X = rand(2, 100)  # 2 features, 100 observations

# Perform simple K-Means clustering with 3 clusters
# defaults to :random initialization
result = kmeans(X, 3)

println("Cluster assignments: ", result.assignments)
println("Medoids: ", result.centers)
println("Total inertia: ", result.inertia)
println("Converged: ", result.converged)
```

### Advanced Usage with Settings Object

For more control over the algorithm, use the `SimpleKMeansAlgorithm` settings object:

```@example simplekmeans_2
using KMeansClustering

X = rand(2, 100) 

settings = KMeansClustering.SimpleKMeansAlgorithm(X, 3)

# Run clustering using multiple dispatch
result = kmeans(settings)
```

## Parameters

> For the non-overloaded version, see the [main documentation page](../index.md).

```@docs
KMeansClustering.kmeans(::SimpleKMeansAlgorithm)
KMeansClustering.SimpleKMeansAlgorithm
```