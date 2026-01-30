## Comparing clustering results

To compare two clustering assignments, we can use the [*Adjusted Rand Index (ARI)*](https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index) that corrects the simple Rand Index by subtracting the expected similarity of random clusterings and normalizing the result. With this we can check how similar two clustering results are.

### Example

```@setup ari
    using Random
    Random.seed!(42)
    # Load helper functions used only for examples
    include(joinpath(@__DIR__, "..", "..", "examples", "compare_results.jl"))
```
We will start by running two different K-means clusterings, e.g. the [Simple KMeans algorithm](algorithms/simplekmeans.md) and the [K-Medoids algorithm](algorithms/kmedoids.md).

```@example ari
    using KMeansClustering

    X = rand(2, 200) 
    cluster_count = 10

    simple_clustering_result = 
        KMeansClustering.kmeans(
            KMeansClustering.SimpleKMeansAlgorithm(X, cluster_count; init_method=:random)
        )
    kmeanspp_clustering_result = # hide
        KMeansClustering.kmeans( # hide
            KMeansClustering.SimpleKMeansAlgorithm(X, cluster_count; init_method=:kmeanspp) # hide
            ) # hide
    kmed_clustering_result = 
        KMeansClustering.kmeans( 
            KMeansClustering.KMedoidsAlgorithm(X, cluster_count)
        ) 
    logkm_clustering_result = # hide
        KMeansClustering.kmeans( # hide
            KMeansClustering.KMeansLogAlgorithm(X, cluster_count, 0.8) # hide
        ) # hide
    bk_clustering_result = # hide
        KMeansClustering.kmeans( # hide
            KMeansClustering.BKMeansAlgorithm(X, cluster_count) # hide  
        )  # hide

```

Now we will use our [helper function](../../examples/compare_results.jl) to check our two clusterings for **similarity** by comparing the assignment vectors of both results.

```@example ari
    label_km = simple_clustering_result.assignments
    label_kmpp = kmeanspp_clustering_result.assignments # hide
    label_kmed = kmed_clustering_result.assignments
    label_logkm = logkm_clustering_result.assignments # hide
    label_bkm = bk_clustering_result.assignments # hide

    adjusted_rand_index(label_km, label_kmed)
```

The closer the ARI is to 1 the better do the two clusterings match. Note that this does not say anything about cluster quality itself.
For this we could check the intertia.

```@example ari
    println("Inertia of Simple Kmeans result: ", simple_clustering_result.inertia)
    println("Inertia of K-Medoids result: ", kmed_clustering_result.inertia)
```
Smaller inertia stands for better cluster quality.
