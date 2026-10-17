# Score-driven updates of the predictive mean a with predictive covariance P.
# With s, H and T the first three derivatives of the conditional log
# likelihood,
#
#   ESD:                    P s(a)
#   ISD:                    d_I = P s(a + d_I)
#   ISD*:                   d_I + (1/2) W_I (T(a + d_I) : W_I),  W_I = (P^{-1} - H(a + d_I))^{-1}
#   explicit second order:  P s + P H P s + (1/2) P (T : P)
#
# In the scalar case W_I = P / (1 - P H(a + d_I)), ISD* is d_I + W_I^2 T/2 and
# the explicit second-order update is P s + P^2 (H s + T/2).
#
# Each experiment in the Python code solves the implicit equation in its own
# way, and each solver is reproduced here with the same safeguards,
# tolerances and iteration limits.

# ---------------------------------------------------------------------------
# Scalar update formulas
# ---------------------------------------------------------------------------

"""ESD correction P s(a)."""
esd_update(P::Real, s::Real) = P * s

"""Explicit second-order correction P s + P^2 (H s + T/2)."""
explicit2_update(P::Real, s::Real, H::Real, T::Real) = P * s + P * P * (H * s + 0.5 * T)

"""ISD* correction d_I + W_I^2 T(a + d_I)/2 in the scalar case."""
isdstar_update(d_isd::Real, W::Real, T_at_mode::Real) = d_isd + 0.5 * W * W * T_at_mode

# ---------------------------------------------------------------------------
# ISD solver of the static experiment (implicit_correction in
# score_approximation_experiment.py)
# ---------------------------------------------------------------------------

"""
    isd_bracketed_newton(model, y, a, P; tolerance = 5e-13, max_iterations = 80)
        -> (d, W, T_at_mode, max_residual, min_curvature)

Solve d = P s(a + d) for every observation in `y` by safeguarded Newton
iterations inside a bracket, as in the Python code. For the Poisson and
log-variance models the root lies between zero and the ESD step P s(a); for
the Student-t location model it lies between zero and the innovation y - a.
A Newton candidate outside the current bracket is replaced by the bracket
midpoint. The iterations run jointly over all observations and stop when the
largest absolute residual is at most `tolerance`, which is the stopping rule
of the vectorized Python solver.

Returns the ISD correction, W_I = P/(1 - P H(a + d)), the third derivative at
the mode, the largest residual |d - P s(a + d)|, and the smallest curvature
denominator 1 - P H(a + d).
"""
function isd_bracketed_newton(model::StaticModel, y::AbstractVector, a::Real, P::Real;
                              tolerance::Real = 5.0e-13, max_iterations::Integer = 80)
    n = length(y)
    lower = Vector{Float64}(undef, n)
    upper = Vector{Float64}(undef, n)
    for i in 1:n
        s_a, _, _ = derivatives(model, y[i], a)
        endpoint = model isa StudentTLocation ? y[i] - a : P * s_a
        lower[i] = min(0.0, endpoint)
        upper[i] = max(0.0, endpoint)
    end

    equation(i, d) = begin
        s, h, t = derivatives(model, y[i], a + d)
        (d - P * s, 1.0 - P * h, t)
    end

    bracket_tolerance = 2.0e-11
    for i in 1:n
        f_lower = equation(i, lower[i])[1]
        f_upper = equation(i, upper[i])[1]
        if f_lower > bracket_tolerance || f_upper < -bracket_tolerance
            error("failed to bracket the implicit root for $(model_key(model))")
        end
    end

    d = 0.5 .* (lower .+ upper)
    converged = false
    for _ in 1:max_iterations
        max_abs_f = 0.0
        for i in 1:n
            f, curvature, _ = equation(i, d[i])
            if curvature <= 0.0
                error("non-positive posterior curvature for $(model_key(model)) at P=$(P)")
            end
            candidate = d[i] - f / curvature
            midpoint = 0.5 * (lower[i] + upper[i])
            if !isfinite(candidate) || candidate <= lower[i] || candidate >= upper[i]
                candidate = midpoint
            end
            fc = equation(i, candidate)[1]
            if fc < 0.0
                lower[i] = candidate
            end
            if fc >= 0.0
                upper[i] = candidate
            end
            d[i] = candidate
            max_abs_f = max(max_abs_f, abs(fc))
        end
        if max_abs_f <= tolerance
            converged = true
            break
        end
    end

    W = Vector{Float64}(undef, n)
    T_at_mode = Vector{Float64}(undef, n)
    max_residual = 0.0
    min_curvature = Inf
    for i in 1:n
        f, curvature, t = equation(i, d[i])
        max_residual = max(max_residual, abs(f))
        min_curvature = min(min_curvature, curvature)
        W[i] = P / curvature
        T_at_mode[i] = t
    end
    if !converged && max_residual > 5.0e-11
        error("implicit solver did not converge for $(model_key(model)) at P=$(P): " *
              "residual=$(max_residual)")
    end
    return d, W, T_at_mode, max_residual, min_curvature
end

# ---------------------------------------------------------------------------
# ISD solver of the recursive experiment (implicit_correction in
# recursive_filter_experiment.py)
# ---------------------------------------------------------------------------

"""
    isd_student_recursive(residual, P, nu, k; max_iter = 14) -> (correction, max_residual)

Proximal posterior mode argmax_d { l(a + d) - d^2/(2P) } for the Student-t
location likelihood, where `residual` is the innovation y - a. This mirrors
the Python solver exactly.

  1. Start from the ESD step clipped to the interval between zero and the
     innovation, and run `max_iter` safeguarded Newton steps on
     d - P s(a + d) = 0 (bisection when the Newton step leaves the bracket).
  2. If the largest residual over the whole vector exceeds 2e-9, replace
     every element by 30 bisection steps on its bracket (the vectorized
     Python fallback applies to all elements at once).
  3. Where P is at or above the global concavity bound 8k/(nu + 1), take the
     global maximizer among the real stationary points of the cubic
     r^3 - innovation r^2 + (k + P c) r - innovation k = 0, with r the residual
     at the mode and c = nu + 1.
"""
function isd_student_recursive(residual::AbstractVector, P::AbstractVector,
                               nu::Real, k::Real; max_iter::Integer = 14)
    n = length(residual)
    c = nu + 1.0
    concavity_bound = 8.0 * k / c
    low = Vector{Float64}(undef, n)
    high = Vector{Float64}(undef, n)
    correction = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        r = residual[i]
        low[i] = min(0.0, r)
        high[i] = max(0.0, r)
        s0 = student_derivatives_residual(r, nu, k)[1]
        correction[i] = clamp(P[i] * s0, low[i], high[i])
    end

    @inbounds for i in 1:n
        p = P[i]
        for _ in 1:max_iter
            s, h, _ = student_derivatives_residual(residual[i] - correction[i], nu, k)
            f = correction[i] - p * s
            if f <= 0.0
                low[i] = correction[i]
            end
            if f > 0.0
                high[i] = correction[i]
            end
            derivative = 1.0 - p * h
            newton = correction[i] - f / derivative
            midpoint = 0.5 * (low[i] + high[i])
            use_newton = isfinite(newton) && newton > low[i] && newton < high[i]
            correction[i] = use_newton ? newton : midpoint
        end
    end

    max_residual = equation_residual_max(correction, residual, P, nu, k)
    if max_residual > 2.0e-9
        # Rare fallback: bisection on the already-valid brackets.
        @inbounds for i in 1:n
            p = P[i]
            for _ in 1:30
                correction[i] = 0.5 * (low[i] + high[i])
                s = student_derivatives_residual(residual[i] - correction[i], nu, k)[1]
                f = correction[i] - p * s
                if f <= 0.0
                    low[i] = correction[i]
                end
                if f > 0.0
                    high[i] = correction[i]
                end
            end
            correction[i] = 0.5 * (low[i] + high[i])
        end
        max_residual = equation_residual_max(correction, residual, P, nu, k)
    end

    nonconcave = [P[i] >= concavity_bound * (1.0 - 1.0e-12) for i in 1:n]
    if any(nonconcave)
        for i in findall(nonconcave)
            innovation = residual[i]
            p = P[i]
            roots_r = cubic_real_roots(-innovation, k + p * c, -innovation * k)
            candidate_d = innovation .- roots_r
            objective = -0.5 .* candidate_d .* candidate_d ./ p .-
                        0.5 .* c .* log1p.(roots_r .* roots_r ./ k)
            correction[i] = candidate_d[argmax(objective)]
        end
        max_residual = equation_residual_max(correction, residual, P, nu, k)
    end
    return correction, max_residual
end

function equation_residual_max(correction, residual, P, nu, k)
    m = 0.0
    @inbounds for i in eachindex(correction)
        s = student_derivatives_residual(residual[i] - correction[i], nu, k)[1]
        m = max(m, abs(correction[i] - P[i] * s))
    end
    return m
end

"""
    cubic_real_roots(c2, c1, c0)

Real roots of x^3 + c2 x^2 + c1 x + c0 from the eigenvalues of the companion
matrix, keeping roots with |imag| < 1e-9 (numpy.roots followed by the filter
used in the Python code).
"""
function cubic_real_roots(c2::Real, c1::Real, c0::Real)
    companion = [-c2 -c1 -c0; 1.0 0.0 0.0; 0.0 1.0 0.0]
    ev = eigvals(companion)
    return [Float64(real(e)) for e in ev if abs(imag(e)) < 1.0e-9]
end

# ---------------------------------------------------------------------------
# ISD in the deterministic risk experiment (bayesian_risk_experiment.py)
# ---------------------------------------------------------------------------

"""
    isd_bernoulli(y, a, P) -> d

Root of d - P (y - expit(a + d)) = 0 on [-P, P]. The Python code uses
scipy.optimize.brentq with xtol 5e-16. The function is strictly increasing,
so safeguarded Newton iterations inside the bracket give the same root; they
stop when the step is within four units in the last place. The two solvers
agree to within about 5e-17 over the published grid.
"""
function isd_bernoulli(y::Real, a::Real, P::Real; max_iterations::Integer = 200)
    lo, hi = -P, P
    d = 0.0
    for _ in 1:max_iterations
        pm = expit(a + d)
        f = d - P * (y - pm)
        f == 0.0 && return d
        if f < 0.0
            lo = d
        else
            hi = d
        end
        candidate = d - f / (1.0 + P * pm * (1.0 - pm))
        if !(lo < candidate < hi)
            candidate = 0.5 * (lo + hi)
        end
        done = abs(candidate - d) <= 4.0 * eps(abs(candidate))
        d = candidate
        done && break
    end
    return d
end

"""
    isd_poisson(y, a, P) -> d

Closed-form ISD for the Poisson log intensity, d = P y - omega(a + P y + log P)
with omega the Wright omega function.
"""
isd_poisson(y::Real, a::Real, P::Real) = P * y - wrightomega(a + P * y + log(P))

"""
    isd_logistic(y, a, X, P, Pinv; max_iterations = 50) -> d

Newton iterations for the multivariate logistic ISD, P^{-1} d = X'(y - expit.(X(a + d))),
started at the ESD step and stopped when the Newton change has norm below
1e-15 (as in `multivariate` of the Python code).
"""
function isd_logistic(y::AbstractVector, a::AbstractVector, X::AbstractMatrix,
                      P::AbstractMatrix, Pinv::AbstractMatrix; max_iterations::Integer = 50)
    p0 = expit.(X * a)
    d = P * (X' * (y .- p0))
    for _ in 1:max_iterations
        pm = expit.(X * (a .+ d))
        Ji = Pinv .+ X' * Diagonal(pm .* (1 .- pm)) * X
        residual = Pinv * d .- X' * (y .- pm)
        change = Ji \ residual
        d = d .- change
        norm(change) < 1.0e-15 && break
    end
    return d
end

"""
    logistic_third_contraction(X, theta, M) -> Vector

T(theta) : M for the logistic log likelihood sum_i {y_i x_i'theta - log(1 + exp(x_i'theta))},
that is X' (t .* diag(X M X')) with t_i = -p_i (1 - p_i)(1 - 2 p_i).
"""
function logistic_third_contraction(X::AbstractMatrix, theta::AbstractVector, M::AbstractMatrix)
    p = expit.(X * theta)
    t = -p .* (1 .- p) .* (1 .- 2 .* p)
    quad = [dot(X[i, :], M * X[i, :]) for i in 1:size(X, 1)]
    return X' * (t .* quad)
end
