"""
    TweedieScore

Score-driven updates and their Bayesian counterparts, from

> Hansen, P. R. and Tong, C. (2026). Tweedie's Formula and Score-Driven Updating.

The package computes, for an observation `y` and a Gaussian predictive law
`N(a, P)` for a scalar parameter, the posterior-mean correction and four
score-driven approximations to it:

  * `esd_correction`        explicit score-driven update (ESD), `P s(a)`
  * `isd_correction`        implicit score-driven update (ISD), `d = P s(a + d)`
  * `isdstar_correction`    mean-corrected implicit update (ISD*)
  * `explicit2_correction`  explicit second-order update
  * `posterior_mean_correction`  the exact correction `E(theta | y) - a`, by
    Gauss-Hermite quadrature

`corrections` returns all five at once. Observation models are subtypes of
`ObservationModel`; see its docstring for how to add a model.

The package also contains the three numerical experiments of the paper
(`run_static_experiment`, `run_risk_experiment`, `run_recursive_experiment`).
The `replication/` folder of the repository runs them and builds the tables
and the figure.

Source files:

  * `numerics.jl`    special functions and small helpers (logistic, Wright omega, ...)
  * `quadrature.jl`  Gauss-Hermite rules and the stable posterior-mean quadrature
  * `models.jl`      observation models and their first three log-likelihood derivatives
  * `updates.jl`     ESD, ISD, ISD* and explicit second-order updates, and the ISD solvers
  * `api.jl`         the scalar interface listed above
  * `io.jl`          CSV and JSON writers and a small command-line parser
  * `experiments/`   the three experiments (static, risk, recursive)

In output files the mean-corrected implicit update is labelled "ISD*". Some
file and column names contain `mcisd`, an earlier name for the same update.
"""
module TweedieScore

using CSV
using DataFrames
using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Printf
using Random
using Statistics

include("numerics.jl")
include("quadrature.jl")
include("models.jl")
include("updates.jl")
include("api.jl")
include("io.jl")
include("experiments/static.jl")
include("experiments/risk.jl")
include("experiments/recursive.jl")

# Observation models and their derivatives
export ObservationModel, PoissonLogIntensity, GaussianLogVariance, StudentTLocation,
       BernoulliLogOdds, GaussianLocation, loglik, derivatives

# Scalar corrections
export esd_correction, isd_correction, isdstar_correction, explicit2_correction,
       posterior_mean_correction, corrections

# Experiments
export run_static_experiment, run_risk_experiment, run_recursive_experiment,
       run_recursive_grid_checks, parse_cli

end # module
