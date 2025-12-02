using KMeansClustering
using Documenter

DocMeta.setdocmeta!(KMeansClustering, :DocTestSetup, :(using KMeansClustering); recursive=true)

makedocs(;
    modules=[KMeansClustering],
    authors="Mark-André Schadow <m.schadow@campus.tu-berlin.de>",
    sitename="KMeansClustering.jl",
    format=Documenter.HTML(;
        canonical="https://markandre01.github.io/KMeansClustering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/markandre01/KMeansClustering.jl",
    devbranch="main",
)
