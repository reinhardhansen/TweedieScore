# Gauss-Hermite rules and the stable posterior-mean quadrature.

"""
    hermite_rule(n) -> (x, w)

Physicists' Gauss-Hermite nodes and weights for the weight exp(-x^2), the rule
returned by numpy.polynomial.hermite.hermgauss(n). The weights sum to sqrt(pi).
"""
function hermite_rule(n::Integer)
    x, w = gausshermite(n)
    return collect(Float64, x), collect(Float64, w)
end

"""
    hermite_logweights(w)

log(w) - log(pi)/2, the log weights of the normalized rule used in the Python
code (`np.log(weights) - 0.5 * math.log(math.pi)`).
"""
hermite_logweights(w::AbstractVector) = log.(w) .- 0.5 * log(pi)

"""
    standard_normal_rule(n) -> (z, w)

Rule for E f(Z) with Z ~ N(0, 1): nodes sqrt(2) x and weights w / sqrt(pi).
This equals scipy.special.roots_hermitenorm(n) with the weights divided by
sqrt(2 pi), as in `gaussian_rule` of bayesian_risk_experiment.py.
"""
function standard_normal_rule(n::Integer)
    x, w = hermite_rule(n)
    return sqrt(2.0) .* x, w ./ sqrt(pi)
end

"""
    tensor_normal_rule_2d(n) -> (Z, w)

Tensor-product rule for a bivariate standard normal vector. Row r of `Z` is a
node and w[r] its weight. The second coordinate varies fastest, which is the
order of itertools.product in the Python code.
"""
function tensor_normal_rule_2d(n::Integer)
    z, w = standard_normal_rule(n)
    Z = Matrix{Float64}(undef, n * n, 2)
    W = Vector{Float64}(undef, n * n)
    row = 0
    for i in 1:n, j in 1:n
        row += 1
        Z[row, 1] = z[i]
        Z[row, 2] = z[j]
        W[row] = w[i] * w[j]
    end
    return Z, W
end

"""
    gaussian_posterior_correction(loglik, center, P, x, logw, buffer)

E(theta - center | y) when theta ~ N(center, P) a priori and `loglik(theta)` is
the conditional log likelihood of the observation. The integral is evaluated
by Gauss-Hermite quadrature with nodes theta_j = center + sqrt(2P) x_j. The log
weights are formed first and their maximum is subtracted before
exponentiation, as in `exact_correction` (static experiment) and
`gaussian_prior_correction` (recursive experiment) of the Python code. The
offsets sqrt(2P) x_j are averaged directly to avoid cancellation at small P.
`buffer` is a work vector with the length of `x`.
"""
function gaussian_posterior_correction(loglik::F, center::Real, P::Real,
                                       x::AbstractVector, logw::AbstractVector,
                                       buffer::AbstractVector) where {F}
    scale = sqrt(2.0 * P)
    m = -Inf
    @inbounds for j in eachindex(x)
        lw = loglik(center + scale * x[j]) + logw[j]
        buffer[j] = lw
        m = max(m, lw)
    end
    total = 0.0
    @inbounds for j in eachindex(x)
        e = exp(buffer[j] - m)
        buffer[j] = e
        total += e
    end
    out = 0.0
    @inbounds for j in eachindex(x)
        out += (buffer[j] / total) * (scale * x[j])
    end
    return out
end
