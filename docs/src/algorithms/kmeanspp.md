```@meta
CurrentModule = KMeansClustering
```

# k-means++ Initialization

## Overview

The k-means++ algorithm is a smarter way to choose initial cluster centers for the classical k-means algorithm. Instead of picking all k centers uniformly at random, k-means++ spreads them out in the data: the first center is chosen at random, and each subsequent center is chosen with probability proportional to the squared distance to the nearest already chosen center. This reduces the chance of poor initializations and often leads to faster convergence and lower final inertia.

## Random vs k-means++ initialization

- **Random (Forgy) initialization:**
    - All k centers are chosen uniformly at random from the data points.
    - Very simple and cheap.
    - Can produce bad initial configurations where several centers start in the same dense region, leading to slow convergence or poor local optima.

  - **k-means++ initialization:**
    - First center is chosen uniformly at random.
    - Each additional center is chosen with probability proportional to the squared distance to the closest existing center.
    - Tends to place centers in different high-density regions of the space.
    - Typically improves stability of the solution and reduces the number of Lloyd iterations needed.

# Implementation Details

## Data Format

The implementation expects data in column-major format:
 
- Rows represent features/dimensions.
- Columns represent individual data points/observations.

The function

kmeanspp_init(X, k; rng=Random.GLOBAL_RNG)


returns a vector of k column indices into X that are selected as initial centers using the k-means++ rule

## Algorithm (high level)

Given a data matrix X with n columns:

1. Choose one column index uniformly at random as the first center.

2. For every point, compute the squared distance to its nearest chosen center.

3. Sample the next center with probability proportional to this squared distance.

4. Repeat steps 2–3 until k centers have been chosen.

5. Return the k selected column indices.

# Usage

## Direct use

```@example kmeanspp
using KMeansClustering, Random

rng = MersenneTwister(1)

X = [0.0  1.0  10.0 11.0;
     0.0  0.0  10.0 10.0]

idxs = kmeanspp_init(X, 2; rng=rng)

println("Initial center indices: ", idxs)

```

## As initialization for Simple K-Means

The k-means++ initializer is mainly intended to be used together with the simple k-means implementation:

- :random chooses k initial centers uniformly at random from the columns of X.
- :kmeanspp uses k-means++ to select better-spread initial centers.

A typical usage pattern is:

```@example kmeanspp2
using KMeansClustering, Random

rng = MersenneTwister(2)
X = rand(rng, 2, 200)

# Use k-means++ as initialization strategy
settings = SimpleKMeansAlgorithm(X, 3; init=:kmeanspp, rng=rng)

result = kmeans(settings)

println("Centers: ", result.centers)
println("Inertia: ", result.inertia)
println("Converged: ", result.converged)

```

## Parameters

KMeansClustering.kmeanspp_init

## AI Note


An initial draft of the k-means++ initializer and parts of this documentation were created with the help of a generative AI tool (ChatGPT). I then adapted the code and text to fit the project’s structure and style guidelines and verified that I understand the implementation and tests.