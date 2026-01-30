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

### Basic Usage

```@example simplekmeans_2
using KMeansClustering

X = rand(2, 100) 

settings = KMeansClustering.SimpleKMeansAlgorithm(X, 3)

# Run clustering using multiple dispatch
result = kmeans(settings)
```

## Parameters

```@docs
KMeansClustering.kmeans(::SimpleKMeansAlgorithm)
KMeansClustering.SimpleKMeansAlgorithm
```
