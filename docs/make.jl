using KMeansClustering
using Documenter

DocMeta.setdocmeta!(KMeansClustering, :DocTestSetup, :(using KMeansClustering); recursive=true)

makedocs(;
    modules=[KMeansClustering],
    authors="Mark-André Schadow <m.schadow@campus.tu-berlin.de>",
    sitename="KMeans-Clustering.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/ShadyKadry/Kmeans-Clustering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ShadyKadry/Kmeans-Clustering.jl",
    devbranch="main",
)
