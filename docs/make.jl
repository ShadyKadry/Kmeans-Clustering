using KMeansClustering
using Documenter

DocMeta.setdocmeta!(KMeansClustering, :DocTestSetup, :(using KMeansClustering); recursive=true)

makedocs(;
    modules=[KMeansClustering],
    authors="Mark-André Schadow <m.schadow@campus.tu-berlin.de>",
    sitename="KMeansClustering.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/ShadyKadry/Kmeans-Clustering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Algorithms" => [
            "K-Medoids" => "algorithms/kmedoids.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/ShadyKadry/Kmeans-Clustering.jl",
    devbranch="main",
)
