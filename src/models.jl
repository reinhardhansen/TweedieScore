# Observation models and the first three derivatives of the conditional
# log likelihood with respect to theta: the score s, the Hessian H, and the
# third derivative T.

# ---------------------------------------------------------------------------
# Static one-step models (Section 7.1; score_approximation_experiment.py)
# ---------------------------------------------------------------------------

abstract type StaticModel end

"""
    ObservationModel

Abstract type of scalar observation models `Y | theta`. A model `m` must
provide

  * `loglik(m, theta, y)`: the conditional log likelihood of `y` at `theta`,
    up to terms that do not depend on `theta`;
  * `derivatives(m, y, theta)`: the tuple `(s, H, T)` of its first three
    derivatives with respect to `theta`.

The implicit solver assumes that `1 - P H(theta) > 0` along its iterations,
which holds for every `P` when the log likelihood is concave. To add a model:

```julia
struct MyModel <: ObservationModel end
TweedieScore.loglik(::MyModel, theta::Real, y::Real) = ...
TweedieScore.derivatives(::MyModel, y::Real, theta::Real) = (s, H, T)
```

Models supplied: `PoissonLogIntensity`, `GaussianLogVariance`,
`StudentTLocation`, `BernoulliLogOdds`, and `GaussianLocation`.
"""
const ObservationModel = StaticModel

"""Poisson log intensity: Y | theta ~ Poisson(exp(theta)), prior center log 4."""
struct PoissonLogIntensity <: StaticModel end

"""Gaussian log variance: Y | theta ~ N(0, exp(theta)), prior center 0."""
struct GaussianLogVariance <: StaticModel end

"""
Unit-variance Student-t location: Y = theta + sigma E with E ~ t_df and
sigma^2 = (df - 2)/df, prior center 0.
"""
struct StudentTLocation <: StaticModel
    df::Float64
    scale::Float64
    residual_denominator::Float64
    key::String
    label::String
end

function StudentTLocation(df::Real; key::Union{Nothing,String} = nothing)
    df = Float64(df)
    scale_squared = (df - 2.0) / df
    residual_denominator = df * scale_squared   # as in the Python code
    display_df = isinteger(df) ? string(Int(df)) : string(df)
    k = key === nothing ? "student_t_" * display_df : key
    label = "Student-\$t_{" * display_df * "}\$ location"
    return StudentTLocation(df, sqrt(scale_squared), residual_denominator, k, label)
end

model_key(::PoissonLogIntensity) = "poisson"
model_key(::GaussianLogVariance) = "log_variance"
model_key(m::StudentTLocation) = m.key

model_label(::PoissonLogIntensity) = "Poisson log intensity"
model_label(::GaussianLogVariance) = "Gaussian log variance"
model_label(m::StudentTLocation) = m.label

prior_center(::PoissonLogIntensity) = log(4.0)
prior_center(::GaussianLogVariance) = 0.0
prior_center(::StudentTLocation) = 0.0

# Log likelihood up to terms that depend on y only; those cancel from the
# normalized quadrature weights.
loglik(::PoissonLogIntensity, theta::Real, y::Real) = y * theta - exp(theta)
loglik(::GaussianLogVariance, theta::Real, y::Real) = -0.5 * (theta + y * y * exp(-theta))
function loglik(m::StudentTLocation, theta::Real, y::Real)
    r = y - theta
    return -0.5 * (m.df + 1.0) * log1p(r * r / m.residual_denominator)
end

"""
    derivatives(model, y, theta) -> (s, H, T)

First three derivatives of the conditional log likelihood at theta.
"""
function derivatives(::PoissonLogIntensity, y::Real, theta::Real)
    intensity = exp(theta)
    return y - intensity, -intensity, -intensity
end

function derivatives(::GaussianLogVariance, y::Real, theta::Real)
    z = y * y * exp(-theta)
    return 0.5 * (z - 1.0), -0.5 * z, 0.5 * z
end

function derivatives(m::StudentTLocation, y::Real, theta::Real)
    r = y - theta
    k = m.residual_denominator
    denominator = k + r * r
    c = m.df + 1.0
    s = c * r / denominator
    h = c * (r * r - k) / denominator^2
    t = 2.0 * c * r * (r * r - 3.0 * k) / denominator^3
    return s, h, t
end

"""
    simulate(rng, model, theta) -> Vector{Float64}

One observation from the conditional law for each element of `theta`.
"""
simulate(rng::AbstractRNG, ::PoissonLogIntensity, theta::AbstractVector) =
    [Float64(rand(rng, Poisson(exp(t)))) for t in theta]

simulate(rng::AbstractRNG, ::GaussianLogVariance, theta::AbstractVector) =
    exp.(0.5 .* theta) .* randn(rng, length(theta))

simulate(rng::AbstractRNG, m::StudentTLocation, theta::AbstractVector) =
    theta .+ m.scale .* rand(rng, TDist(m.df), length(theta))

# ---------------------------------------------------------------------------
# Further models for the scalar interface in api.jl
# ---------------------------------------------------------------------------

"""Bernoulli log odds: Y | theta ~ Bernoulli(expit(theta)), with Y in {0, 1}."""
struct BernoulliLogOdds <: StaticModel end

"""
    GaussianLocation(variance)

Gaussian location with known variance: Y | theta ~ N(theta, variance). The log
likelihood is quadratic, so ISD and ISD* equal the exact posterior-mean
(Kalman) correction `P (y - a) / (variance + P)`.
"""
struct GaussianLocation <: StaticModel
    variance::Float64
    function GaussianLocation(variance::Real)
        variance > 0 || throw(ArgumentError("variance must be positive"))
        return new(Float64(variance))
    end
end

model_key(::BernoulliLogOdds) = "bernoulli"
model_key(::GaussianLocation) = "gaussian_location"
model_label(::BernoulliLogOdds) = "Bernoulli log odds"
model_label(::GaussianLocation) = "Gaussian location"

loglik(::BernoulliLogOdds, theta::Real, y::Real) = y * theta - log1pexp(theta)
loglik(m::GaussianLocation, theta::Real, y::Real) = -0.5 * (y - theta)^2 / m.variance

derivatives(::BernoulliLogOdds, y::Real, theta::Real) = bernoulli_logodds_derivatives(y, theta)
derivatives(m::GaussianLocation, y::Real, theta::Real) =
    ((y - theta) / m.variance, -1.0 / m.variance, 0.0)

simulate(rng::AbstractRNG, ::BernoulliLogOdds, theta::AbstractVector) =
    [Float64(rand(rng) < expit(t)) for t in theta]
simulate(rng::AbstractRNG, m::GaussianLocation, theta::AbstractVector) =
    theta .+ sqrt(m.variance) .* randn(rng, length(theta))

# ---------------------------------------------------------------------------
# Student-t location in residual form (Section 7.3; recursive_filter_experiment.py)
# ---------------------------------------------------------------------------
# Here r = y - theta and k = nu - 2, so that the observation error has unit
# variance. The derivatives are with respect to the location theta.

student_loglik_residual(r::Real, nu::Real, k::Real) = -0.5 * (nu + 1.0) * log1p(r * r / k)

function student_derivatives_residual(r::Real, nu::Real, k::Real)
    c = nu + 1.0
    denominator = k + r * r
    s = c * r / denominator
    h = c * (r * r - k) / denominator^2
    t = 2.0 * c * r * (r * r - 3.0 * k) / denominator^3
    return s, h, t
end

# ---------------------------------------------------------------------------
# Models of the deterministic risk experiment (Section 7.2; bayesian_risk_experiment.py)
# ---------------------------------------------------------------------------

"""
    bernoulli_logodds_derivatives(y, theta) -> (s, H, T)

Bernoulli log odds: s = y - p, H = -p(1 - p), T = H(1 - 2p), p = expit(theta).
"""
function bernoulli_logodds_derivatives(y::Real, theta::Real)
    p = expit(theta)
    h = -p * (1 - p)
    return y - p, h, h * (1 - 2 * p)
end

"""
    poisson_logintensity_derivatives(y, theta) -> (s, H, T)

Poisson log intensity: s = y - exp(theta), H = T = -exp(theta).
"""
function poisson_logintensity_derivatives(y::Real, theta::Real)
    rate = exp(theta)
    return y - rate, -rate, -rate
end

"""
Design of the bivariate logistic model in Appendix F.4: two conditionally
independent Bernoulli observations with success probabilities expit(x_i' theta).
"""
const BIVARIATE_X = [1.0 0.5; -0.4 1.0]
const BIVARIATE_A = [log(4.0), -0.4]
const BIVARIATE_SIGMA = [1.0 0.6; 0.6 1.4]
