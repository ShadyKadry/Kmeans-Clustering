#       function adjusted_rand_index(clusteringlabels1::Vector{Int}, clusteringlabels2::Vector{Int})
#
# Calculate the adjusted rand index to compare two clustering results.
# Implementation is based on the description from:
# <https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index>

# Arguments
# - `clusteringlabels1::Vector{Int}`
#     cluster assignments of KMeansResult from one algorithm 
# - `clusteringlabels2::Vector{Int}`
#     cluster assignments of KMeansResult from other algorithm 
# Returns
#   1 if clustering results are a perfect match
#   ~0  if the agreement is no better than random
#   <0  if the agreement is worse than random 
function adjusted_rand_index(clusteringlabels1::Vector{Int}, clusteringlabels2::Vector{Int})

    n = length(clusteringlabels1)

    if n != length(clusteringlabels2)
        throw(DimensionMismatch("clusterungs do not have the sme number of points"))
    end

    # map the cluster labels to integers
    indexmap1 = Dict{Int,Int}()
    indexmap2 = Dict{Int,Int}()

    for (i, label) in enumerate(unique(clusteringlabels1))
        indexmap1[label] = i
    end
    for (i, label) in enumerate(unique(clusteringlabels2))
        indexmap2[label] = i
    end

    # calculate contingency table
    k1 = length(indexmap1)
    k2 = length(indexmap2)
    contingency = zeros(Int, k1, k2)

    for i in 1:n
        x = indexmap1[clusteringlabels1[i]]
        y = indexmap2[clusteringlabels2[i]]
        contingency[x, y] += 1
    end

    # compute ari
    sum_binomial = x -> sum(x .* (x .- 1) .÷ 2)
    sum_a_i = sum_binomial(sum(contingency, dims=2))
    sum_b_i = sum_binomial(sum(contingency, dims=1))
    index = sum_binomial(contingency)

    expected_index = sum_a_i * sum_b_i / sum_binomial(n)
    max_index = (sum_a_i + sum_b_i) / 2
    ari = (index - expected_index) / (max_index - expected_index)
    return ari

end
