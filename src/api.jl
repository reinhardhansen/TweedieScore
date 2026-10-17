# Scalar interface: the posterior-mean correction under a Gaussian predictive
# law N(a, P) and its four score-driven approximations, for one observation.
#
# All functions return corrections, that is, the update minus the predictive
# mean a. They use the same formulas and solvers as the experiments.

# Fallbacks for models defined outside the package.
model_key(m::StaticModel) = string(nameof(typeof(m)))
model_label(m::StaticModel) = model_key(m)

function check_variance(P::Real)
    P > 0 || throw(ArgumentError("the predictive variance P must be positive, got $P"))
    return Float64(P)
end

"""
    esd_correction(model, y, a, P)

Explicit score-driven (ESD) correction `P s(a)`, where `s` is the score of the
conditional log likelihood of `y`.

# Example
```julia
esd_correction(PoissonLogIntensity(), 6, log(4), 0.1)   # 0.1 * (6 - 4) = 0.2
```
"""
function esd_correction(model::StaticModel, y::Real, a::Real, P::Real)
    P = check_variance(P)
    s, _, _ = derivatives(model, y, a)
    return esd_update(P, s)
end

# ISD by the safeguarded Newton solver of the static experiment.
function isd_scalar(model::StaticModel, y::Real, a::Real, P::Real)
    P = check_variance(P)
    d, W, T_at_mode, _, _ = isd_bracketed_newton(model, [Float64(y)], Float64(a), P)
    return d[1], W[1], T_at_mode[1]
end

"""
    isd_correction(model, y, a, P)

Implicit score-driven (ISD) correction: the solution `d` of `d = P s(a + d)`.
`a + d` is the posterior mode under the prior N(a, P). The equation is solved
by safeguarded Newton iterations inside a bracket, to an absolute residual of
at most 5e-13.
"""
isd_correction(model::StaticModel, y::Real, a::Real, P::Real) = isd_scalar(model, y, a, P)[1]

"""
    isdstar_correction(model, y, a, P)

Mean-corrected implicit correction (ISD*): `d + W^2 T(a + d) / 2`, where `d` is
the ISD correction, `W = P / (1 - P H(a + d))`, and `H` and `T` are the second
and third derivatives of the log likelihood. The added term is the leading
Laplace correction from the posterior mode to the posterior mean, so the error
of ISD* is of third order in `P`.
"""
function isdstar_correction(model::StaticModel, y::Real, a::Real, P::Real)
    d, W, T_at_mode = isd_scalar(model, y, a, P)
    return isdstar_update(d, W, T_at_mode)
end

"""
    explicit2_correction(model, y, a, P)

Explicit second-order correction `P s + P^2 (H s + T/2)`, with `s`, `H` and
`T` evaluated at `a`.
"""
function explicit2_correction(model::StaticModel, y::Real, a::Real, P::Real)
    P = check_variance(P)
    s, H, T = derivatives(model, y, a)
    return explicit2_update(P, s, H, T)
end

"""
    posterior_mean_correction(model, y, a, P; nodes = 80)

Exact posterior-mean correction `E(theta | y) - a` when `theta ~ N(a, P)` a
priori, computed by Gauss-Hermite quadrature with `nodes` nodes. The weights
are formed on the log scale, as in the experiments.
"""
function posterior_mean_correction(model::StaticModel, y::Real, a::Real, P::Real;
                                   nodes::Integer = 80)
    P = check_variance(P)
    x, w = hermite_rule(nodes)
    logw = hermite_logweights(w)
    return gaussian_posterior_correction(theta -> loglik(model, theta, y), Float64(a), P,
                                         x, logw, similar(x))
end

"""
    corrections(model, y, a, P; nodes = 80) -> NamedTuple

All five corrections for one observation, as the named tuple
`(posterior_mean, esd, isd, isdstar, explicit2)`.

# Example
```julia
c = corrections(PoissonLogIntensity(), 6, log(4), 0.1)
c.isdstar - c.posterior_mean   # error of third order in P
```
"""
function corrections(model::StaticModel, y::Real, a::Real, P::Real; nodes::Integer = 80)
    P = check_variance(P)
    s, H, T = derivatives(model, y, a)
    d, W, T_at_mode = isd_scalar(model, y, a, P)
    return (posterior_mean = posterior_mean_correction(model, y, a, P; nodes = nodes),
            esd = esd_update(P, s),
            isd = d,
            isdstar = isdstar_update(d, W, T_at_mode),
            explicit2 = explicit2_update(P, s, H, T))
end
