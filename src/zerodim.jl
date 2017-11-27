"Zero-dimensional manifold search"
function zerodimsearch(M::Manifold, X::AbstractMatrix, params::Parameters)
    manifolds = Manifold[]
    separations = Separation[]
    best_separation = Separation()

    # adjust noise cluster to form 1D manifold
    if indim(M) == 0
        M.d = 1
        adjustbasis!(M, X, adjust_dim = 1, adjust_dim_ratio = eps())
        LOG(params, 4, @sprintf("Adjusted noise cluster size=%d, dim=%d", length(labels(M)), indim(M)))
    end

    selected = labels(M)
    while length(selected) > 0
        best_separation = try
            find_separation(X[:,selected], mean(R), projection(M), params, true)
        catch ex
            if isa(ex, LMCLUSException)
                LOG(params, 5, ex.msg)
            else
                LOG(params, 5, string(ex))
            end
            Separation()
        end
        log_separation(best_separation, params)

        # stop search if we cannot separate manifold
        if check_separation(best_separation, params)
            LOG(params, 3, "no separation, cluster didn't change...")
            break
        end

        # otherwise separate points to 0D manifold
        LOG(params, 3, "Found zero-dimensional manifold. Separating...")

        # separate cluster points
        selected, removed_points = filter_separeted(selected, X, mean(M), projection(M), best_separation)

        # small amount of points is considered noise, stop searching
        if length(selected) <= params.min_cluster_size
            LOG(params, 3, "noise: cluster size < ", params.min_cluster_size," points")
            break
        end

        # update manifold description
        R = fit(PCA, X[:, removed_points], pratio = 1.0)
        M.points = removed_points # its labels
        M.μ = MultivariateStats.mean(R)
        M.proj = MultivariateStats.projection(R)
        # M.d = MultivariateStats.outdim(R) == MultivariateStats.indim(R) ? 0 : MultivariateStats.outdim(R)
        LOG(params, 4, "Reduce cluster to $(outdim(M)) points.")

        # form 0D cluster
        zd_manifold = Manifold(0, mean(X[:, selected],2)[:], zeros(0,0), selected, threshold(best_separation), 0.0)
        LOG(params, 3, "0D cluster formed with $(outdim(zd_manifold)) points.")

        # add new manifold to output
        push!(manifolds, zd_manifold)
        push!(separations, best_separation)
    end

    push!(manifolds, M)
    push!(separations, best_separation)
    return manifolds, separations
end
