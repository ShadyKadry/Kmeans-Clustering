using PkgTemplates

template = Template(;
    user="markandre01",
    authors="Mark-André Schadow <m.schadow@campus.tu-berlin.de>",
    julia=v"1.11",
    plugins=[
        License(; name="MIT"),
        Git(; ssh=true, manifest=false),
        Tests(; project=true),
        GitHubActions(; x64=true, extra_versions=[v"1.11"]),
        Codecov(),
        Documenter{GitHubActions}(),
    ],
)

template("KMeansClustering")