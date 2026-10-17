# TweedieScore.jl

[![CI](https://github.com/reinhardhansen/TweedieScore/actions/workflows/CI.yml/badge.svg)](https://github.com/reinhardhansen/TweedieScore/actions/workflows/CI.yml)

Julia code for

> Hansen, P. R. and Tong, C. (2026). Tweedie's Formula and Score-Driven Updating.
> [arXiv:2605.15902](https://arxiv.org/abs/2605.15902).

Tweedie's formula expresses a posterior mean in terms of a score. The paper uses
it to connect score-driven updating with Bayesian updating. With a Gaussian
predictive law `N(a, P)` for a parameter `theta`, the posterior-mean correction
`E(theta | y) - a` is approximated by

* **ESD**, the explicit score-driven update `P s(a)`;
* **ISD**, the implicit score-driven update, the solution `d` of `d = P s(a + d)`;
* **ISD\***, the mean-corrected implicit update, which adds the leading Laplace
  correction from the posterior mode to the posterior mean;
* the explicit second-order update `P s + P^2 (H s + T/2)`.

Here `s`, `H` and `T` are the first three derivatives of the conditional log
likelihood. ESD and ISD have second-order errors in `P`; ISD\* and the explicit
second-order update have third-order errors.

The package has two parts: a small interface that computes these corrections for
a single observation, and the code of the numerical experiments in Section 7 of
the paper. The `replication/` folder runs the experiments and builds every table
and figure; see [replication/README.md](replication/README.md).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/reinhardhansen/TweedieScore")
```

Julia 1.10 or later is required.

## Example

```julia
using TweedieScore

model = PoissonLogIntensity()     # Y | theta ~ Poisson(exp(theta))
a, P, y = log(4), 0.1, 9          # predictive mean and variance, observation

c = corrections(model, y, a, P)
# c.posterior_mean   exact correction, by Gauss-Hermite quadrature
# c.esd, c.isd, c.isdstar, c.explicit2

c.isd - c.posterior_mean          # error of ISD
c.isdstar - c.posterior_mean      # error of ISD*, much smaller
```

Each correction is also available on its own: `esd_correction`,
`isd_correction`, `isdstar_correction`, `explicit2_correction`, and
`posterior_mean_correction(model, y, a, P; nodes = 80)`.

## Observation models

| Model | Conditional law |
|---|---|
| `PoissonLogIntensity()` | `Y ~ Poisson(exp(theta))` |
| `GaussianLogVariance()` | `Y ~ N(0, exp(theta))` |
| `StudentTLocation(df)` | `Y = theta + sigma E`, `E ~ t_df`, unit variance |
| `BernoulliLogOdds()` | `Y ~ Bernoulli(1 / (1 + exp(-theta)))` |
| `GaussianLocation(variance)` | `Y ~ N(theta, variance)` |

A new model is a subtype of `ObservationModel` with methods for `loglik` and
`derivatives`:

```julia
struct MyModel <: ObservationModel end
TweedieScore.loglik(::MyModel, theta::Real, y::Real) = ...           # up to terms free of theta
TweedieScore.derivatives(::MyModel, y::Real, theta::Real) = (s, H, T)
```

The implicit solver assumes `1 - P H > 0` along its iterations, which holds for
every `P` when the log likelihood is concave in `theta`.

## Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests check derivatives and solvers, exactness of ISD in the Gaussian
location model, the error orders of the four updates, small runs of the
experiments, and the reference output of the paper.

## Citation

Please cite the paper; see [CITATION.cff](CITATION.cff).

```bibtex
@article{HansenTong:TweedieScore,
  author  = {Hansen, Peter Reinhard and Tong, Chen},
  title   = {Tweedie's Formula and Score-Driven Updating},
  year    = {2026},
  journal = {arXiv:2605.15902}
}
```

## Notes

The experiments were first written in Python by Chen Tong and then ported to
Julia. The Julia results were checked against the Python implementation. Some
comments in the source refer to the Python functions that a routine reproduces.

## License

MIT; see [LICENSE](LICENSE).
