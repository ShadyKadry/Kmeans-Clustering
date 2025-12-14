module AlgorithmsKMeansPP

using Random

# Compute squared Euclidean distance between two observations (columns) in X.
# Returns Float64 to keep distance accumulation stable across integer/float inputs.
@inline function _sqeuclid_cols(X::AbstractMatrix{<:Real}, j::Int, c::Int)::Float64
    s = 0.0
    @inbounds for i in axes(X, 1)
        d = X[i, j] - X[i, c]
        s += d * d
    end
    return s
end

# Sample an index proportional to weights in `w` over the not-yet-chosen elements.
# Falls back to uniform sampling if all weights are zero (degenerate data).
function _weighted_sample_index(w::AbstractVector{<:Real}, chosen::AbstractVector{Bool}, rng::AbstractRNG)::Int
    n = length(w)
    total = 0.0
    @inbounds for i in 1:n
        if !chosen[i]
            total += float(w[i])
        end
    end

    if total == 0.0 || !isfinite(total)
        # Degenerate case (e.g., all points identical): choose uniformly among unchosen.
        while true
            idx = rand(rng, 1:n)
            if !chosen[idx]
                return idx
            end
        end
    end

    u = rand(rng) * total
    acc = 0.0
    @inbounds for i in 1:n
        if !chosen[i]
            acc += float(w[i])
            if u <= acc
                return i
            end
        end
    end

    # Numeric fallthrough: return the last unchosen index.
    @inbounds for i in n:-1:1
        if !chosen[i]
            return i
        end
    end

    error("weighted sampling failed: no unchosen indices found")
end

"""
    kmeanspp_init(X, k; rng=Random.GLOBAL_RNG)

Select `k` initial centers using the k-means++ heuristic.

Arguments
- X: data matrix with features in rows and observations in columns.
- k: number of clusters.

Keyword arguments
- rng: random number generator.

Returns
A vector of length `k` with indices into the columns of `X`, indicating which
points are chosen as initial centers.
"""
function kmeanspp_init(X::AbstractMatrix{<:Real}, k::Integer;
                       rng::AbstractRNG = Random.GLOBAL_RNG)

    n = size(X, 2)  # number of observations (columns)

    if k < 1
        throw(ArgumentError("k must be >= 1, got $k"))
    end
    if n == 0
        throw(ArgumentError("X has no observations (0 columns)"))
    end
    if k > n
        throw(ArgumentError("k must be <= number of observations (columns) = $n, got $k"))
    end

    kk = Int(k)

    # Track chosen indices and a mask for O(1) membership checks.
    chosen_idxs = Vector{Int}(undef, kk)
    chosen_mask = falses(n)

    # 1) Choose the first center uniformly at random.
    first = rand(rng, 1:n)
    chosen_idxs[1] = first
    chosen_mask[first] = true

    # mind2[j] stores D^2(x_j): squared distance from point j to nearest chosen center.
    mind2 = fill(Inf, n)
    mind2[first] = 0.0

    # Initialize distances using the first chosen center.
    @inbounds for j in 1:n
        if !chosen_mask[j]
            mind2[j] = _sqeuclid_cols(X, j, first)
        end
    end

    # 2) Repeatedly sample next center with probability proportional to mind2.
    for t in 2:kk
        next = _weighted_sample_index(mind2, chosen_mask, rng)

        chosen_idxs[t] = next
        chosen_mask[next] = true
        mind2[next] = 0.0

        # Update mind2 with the new center: mind2[j] = min(mind2[j], dist2(j, next)).
        @inbounds for j in 1:n
            if !chosen_mask[j]
                d2 = _sqeuclid_cols(X, j, next)
                if d2 < mind2[j]
                    mind2[j] = d2
                end
            end
        end
    end

    return chosen_idxs
end

end # module
