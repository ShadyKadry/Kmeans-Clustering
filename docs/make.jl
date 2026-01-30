using KMeansClustering
using Documenter

DocMeta.setdocmeta!(KMeansClustering, :DocTestSetup, :(using KMeansClustering); recursive=true)

makedocs(;
    modules=[KMeansClustering],
    authors="Mark-André Schadow <m.schadow@campus.tu-berlin.de>",
    sitename="KMeansClustering",
    format=Documenter.HTML(;
        canonical="https://github.com/ShadyKadry/Kmeans-Clustering",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Comparing Results" => "compare_results.md",
        "Algorithms" => [
            "K-Medoids" => "algorithms/kmedoids.md",
            "Simple KMeans" => "algorithms/simplekmeans.md",
            "KMeans with Log Heuristic" => "algorithms/kmeanslog.md",
            "Bisecting K-Means" => "algorithms/bkmeans.md"
        ]
    ],
)

deploydocs(;
    repo="github.com/ShadyKadry/Kmeans-Clustering",
    devbranch="main"
)
