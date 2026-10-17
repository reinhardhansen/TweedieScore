# Section 7.3: a recursive non-Gaussian benchmark.
# Port of recursive_filter_experiment.py.
#
#   theta_t = phi theta_{t-1} + eta_t,   eta_t ~ N(0, q),
#   y_t     = theta_t + sigma_nu E_t,    E_t ~ t_nu,  sigma_nu^2 = (nu - 2)/nu.
#
# The stationary state variance equals the signal-to-noise ratio, so
# q = snr (1 - phi^2). A deterministic grid filter is the reference. Every
# recursive mean approximation uses the grid filter's predictive variance.
#
# Method labels follow the Python output files, except that the method called
# "Mean-corrected implicit" (MC-ISD) there is labelled "ISD*" here.

const RECURSIVE_APPROX_METHODS = ("First order", "Second order", "Implicit", "ISD*", "Kalman")
const RECURSIVE_METHODS = ("Exact Gaussian", RECURSIVE_APPROX_METHODS...)

struct Design
    nu::Float64
    snr::Float64
    phi::Float64
end

state_variance(d::Design) = d.snr
process_variance(d::Design) = state_variance(d) * (1.0 - d.phi * d.phi)
observation_scale(d::Design) = sqrt((d.nu - 2.0) / d.nu)
residual_denominator(d::Design) = d.nu - 2.0
design_key(d::Design) = "nu$(fmt_g(d.nu))_snr$(fmt_g(d.snr))_phi$(fmt_g(d.phi))"

"""
Grid, spacing and the propagation matrix of the grid filter. `propagator`
maps a filtered density on the grid to the (unnormalized) predictive density:
it first rescales by the AR coefficient with linear interpolation, then
convolves with the Gaussian innovation density. The Python code applies the
same two linear maps, using a zero-padded FFT for the convolution. The FFT
length exceeds 2G - 1, so that convolution is linear rather than circular and
the dense matrix gives the same result up to rounding error (about 1e-15).
"""
struct GridTransition
    grid::Vector{Float64}
    grid2::Vector{Float64}
    dx::Float64
    phi::Float64
    process_variance::Float64
    propagator::Matrix{Float64}
end

function make_grid_transition(design::Design, grid_points::Integer, grid_sd::Real)
    if grid_points < 101 || iseven(grid_points)
        throw(ArgumentError("grid_points must be an odd integer of at least 101"))
    end
    G = grid_points
    state_sd = sqrt(state_variance(design))
    half_width = grid_sd * state_sd
    grid = numpy_linspace(-half_width, half_width, G)
    dx = grid[2] - grid[1]

    # Deterministic AR scaling: p_{phi X}(z) = p_X(z/phi)/|phi|, by linear
    # interpolation. Indices `left` are zero-based as in the Python code.
    positions = (grid ./ design.phi .- grid[1]) ./ dx
    scaling = zeros(G, G)
    for j in 1:G
        pos = positions[j]
        (pos >= 0.0 && pos <= G - 1.0) || continue
        left = clamp(floor(Int, pos), 0, G - 2)
        weight = clamp(pos - left, 0.0, 1.0)
        scaling[j, left + 1] += (1.0 - weight) / abs(design.phi)
        scaling[j, left + 2] += weight / abs(design.phi)
    end

    # Centered Gaussian kernel on the grid. Discrete normalization removes the
    # small finite-grid quadrature error before convolution.
    half = G ÷ 2
    lag = ((0:(G - 1)) .- half) .* dx
    q = process_variance(design)
    kernel = exp.(-0.5 .* lag .* lag ./ q) ./ sqrt(2.0 * pi * q)
    kernel ./= sum(kernel) * dx
    # predicted[i] = dx * sum_j scaled[j] kernel[half + i - j] (zero-based).
    convolution = zeros(G, G)
    for i in 1:G, j in 1:G
        idx = half + i - j + 1
        if 1 <= idx <= G
            convolution[i, j] = dx * kernel[idx]
        end
    end
    return GridTransition(grid, grid .* grid, dx, design.phi, q, convolution * scaling)
end

"""Stationary Gaussian density on the grid, one column per path."""
function initial_grid_density(transition::GridTransition, variance::Real, n_rep::Integer)
    density = exp.(-0.5 .* transition.grid .* transition.grid ./ variance)
    density ./= sum(density) * transition.dx
    return repeat(density, 1, n_rep)
end

"""Clamp column `i` of `density` at zero and normalize it to integrate to one."""
function normalize_column!(density::AbstractMatrix, i::Integer, dx::Real)
    total = 0.0
    @inbounds for g in axes(density, 1)
        v = max(density[g, i], 0.0)
        density[g, i] = v
        total += v
    end
    normalizer = total * dx
    if !isfinite(normalizer) || normalizer <= 0.0
        error("A grid density has a nonpositive or nonfinite normalizer")
    end
    @inbounds for g in axes(density, 1)
        density[g, i] /= normalizer
    end
    return density
end

"""Mean and variance of every column of a grid density (variance floored at 1e-12)."""
function density_moments(density::AbstractMatrix, transition::GridTransition)
    n = size(density, 2)
    means = Vector{Float64}(undef, n)
    variances = Vector{Float64}(undef, n)
    grid = transition.grid
    grid2 = transition.grid2
    @inbounds for i in 1:n
        m1 = 0.0
        m2 = 0.0
        for g in eachindex(grid)
            m1 += density[g, i] * grid[g]
            m2 += density[g, i] * grid2[g]
        end
        m1 *= transition.dx
        m2 *= transition.dx
        means[i] = m1
        variances[i] = max(m2 - m1 * m1, 1.0e-12)
    end
    return means, variances
end

"""
    grid_update!(posterior, predictive, y, design, transition) -> (means, variances, max_edge_mass)

Multiply each predictive column by the Student-t likelihood of its
observation, normalize, and return the posterior moments and the largest
posterior mass in the four outermost cells at either end.
"""
function grid_update!(posterior::AbstractMatrix, predictive::AbstractMatrix,
                      y::AbstractVector, design::Design, transition::GridTransition)
    G, n = size(predictive)
    nu = design.nu
    k = residual_denominator(design)
    grid = transition.grid
    dx = transition.dx
    edge_cells = min(4, G ÷ 20)
    edge_mass = zeros(n)
    Threads.@threads for i in 1:n
        yi = y[i]
        @inbounds for g in 1:G
            posterior[g, i] = predictive[g, i] * exp(student_loglik_residual(yi - grid[g], nu, k))
        end
        normalize_column!(posterior, i, dx)
        e = 0.0
        @inbounds for g in 1:edge_cells
            e += posterior[g, i]
        end
        e2 = 0.0
        @inbounds for g in (G - edge_cells + 1):G
            e2 += posterior[g, i]
        end
        edge_mass[i] = (e + e2) * dx
    end
    means, variances = density_moments(posterior, transition)
    return means, variances, maximum(edge_mass)
end

"""Propagate filtered densities one period ahead and normalize (`grid_predict`)."""
function grid_predict!(predictive::AbstractMatrix, posterior::AbstractMatrix,
                       transition::GridTransition)
    mul!(predictive, transition.propagator, posterior)
    Threads.@threads for i in 1:size(predictive, 2)
        normalize_column!(predictive, i, transition.dx)
    end
    return predictive
end

"""
    recursive_gaussian_correction(y, centers, variances, design, x, logw)

Exact posterior-mean correction under the Gaussian prior N(center, variance)
for each path, by Gauss-Hermite quadrature (`gaussian_prior_correction`).
"""
function recursive_gaussian_correction(y::AbstractVector, centers::AbstractVector,
                                       variances::AbstractVector, design::Design,
                                       x::AbstractVector, logw::AbstractVector)
    nu = design.nu
    k = residual_denominator(design)
    out = Vector{Float64}(undef, length(y))
    buffer = Vector{Float64}(undef, length(x))
    for i in eachindex(y)
        yi = y[i]
        out[i] = gaussian_posterior_correction(theta -> student_loglik_residual(yi - theta, nu, k),
                                               centers[i], variances[i], x, logw, buffer)
    end
    return out
end

"""
    recursive_corrections(y, centers, variances, design) -> (Dict, max_implicit_residual)

The five mean corrections at predictive means `centers` and variances
`variances`: first order (ESD), second order (explicit), implicit (ISD),
ISD*, and the Gaussian-observation Kalman correction P (y - a)/(1 + P).
"""
function recursive_corrections(y::AbstractVector, centers::AbstractVector,
                               variances::AbstractVector, design::Design)
    nu = design.nu
    k = residual_denominator(design)
    n = length(y)
    residual = y .- centers
    p = variances
    first_order = Vector{Float64}(undef, n)
    second_order = Vector{Float64}(undef, n)
    for i in 1:n
        s, h, t = student_derivatives_residual(residual[i], nu, k)
        first_order[i] = esd_update(p[i], s)
        second_order[i] = explicit2_update(p[i], s, h, t)
    end
    implicit, implicit_residual = isd_student_recursive(residual, p, nu, k)
    isdstar = Vector{Float64}(undef, n)
    for i in 1:n
        _, mode_h, mode_t = student_derivatives_residual(residual[i] - implicit[i], nu, k)
        curvature = 1.0 - p[i] * mode_h
        if !isfinite(curvature) || curvature <= 0.0
            error("ISD* encountered a non-positive posterior curvature")
        end
        isdstar[i] = isdstar_update(implicit[i], p[i] / curvature, mode_t)
    end
    kalman = p ./ (p .+ 1.0) .* residual
    return Dict("First order" => first_order, "Second order" => second_order,
                "Implicit" => implicit, "ISD*" => isdstar, "Kalman" => kalman),
           implicit_residual
end

"""RMSE of path-level mean squared errors and its delta-method Monte Carlo standard error."""
function rmse_mcse(per_rep_mse::AbstractVector)
    rmse = sqrt(mean(per_rep_mse))
    (rmse == 0.0 || length(per_rep_mse) < 2) && return rmse, 0.0
    return rmse, std(per_rep_mse; corrected = true) / sqrt(length(per_rep_mse)) / (2.0 * rmse)
end

"""Mean across independent paths and its Monte Carlo standard error."""
function mean_mcse(values::AbstractVector)
    m = mean(values)
    length(values) < 2 && return m, 0.0
    return m, std(values; corrected = true) / sqrt(length(values))
end

"""Paired ISD versus ISD* comparison, with paths as independent clusters."""
function paired_loss_row(design::Design, target::AbstractString,
                         implicit_sse, isdstar_sse, implicit_sae, isdstar_sae, horizon)
    implicit_mse = implicit_sse ./ horizon
    isdstar_mse = isdstar_sse ./ horizon
    implicit_mae_path = implicit_sae ./ horizon
    isdstar_mae_path = isdstar_sae ./ horizon
    implicit_rmse, implicit_rmse_mcse = rmse_mcse(implicit_mse)
    isdstar_rmse, isdstar_rmse_mcse = rmse_mcse(isdstar_mse)
    mse_difference, mse_difference_mcse = mean_mcse(isdstar_mse .- implicit_mse)
    mae_difference, mae_difference_mcse = mean_mcse(isdstar_mae_path .- implicit_mae_path)
    implicit_mae = mean(implicit_mae_path)
    isdstar_mae = mean(isdstar_mae_path)
    # Column names keep the Python spelling "mcisd" so that the files are
    # column-compatible with the stored Python output.
    return (
        nu = design.nu,
        snr = design.snr,
        phi = design.phi,
        target = String(target),
        n_rep = length(implicit_mse),
        T = Int(horizon),
        implicit_rmse = implicit_rmse,
        implicit_rmse_mcse = implicit_rmse_mcse,
        mcisd_rmse = isdstar_rmse,
        mcisd_rmse_mcse = isdstar_rmse_mcse,
        rmse_reduction_percent = 100.0 * (1.0 - isdstar_rmse / implicit_rmse),
        paired_mse_difference = mse_difference,
        paired_mse_difference_mcse = mse_difference_mcse,
        paired_mse_difference_ci_low = mse_difference - 1.96 * mse_difference_mcse,
        paired_mse_difference_ci_high = mse_difference + 1.96 * mse_difference_mcse,
        implicit_mae = implicit_mae,
        mcisd_mae = isdstar_mae,
        mae_reduction_percent = 100.0 * (1.0 - isdstar_mae / implicit_mae),
        paired_mae_difference = mae_difference,
        paired_mae_difference_mcse = mae_difference_mcse,
        paired_mae_difference_ci_low = mae_difference - 1.96 * mae_difference_mcse,
        paired_mae_difference_ci_high = mae_difference + 1.96 * mae_difference_mcse,
        path_mse_win_share = mean(isdstar_mse .< implicit_mse),
        path_mae_win_share = mean(isdstar_mae_path .< implicit_mae_path),
    )
end

zero_dict(methods, n) = Dict(m => zeros(n) for m in methods)
matrix_dict(methods, horizon, n) = Dict(m => Matrix{Float64}(undef, horizon, n) for m in methods)

"""
    run_design(design, rng, n_rep, horizon, grid_points, grid_sd, x, logw, keep_paths)
        -> (path_rows, oracle_rows, paired_rows, diagnostics, history)

Simulate `n_rep` paths of length `horizon` for one design and evaluate the
grid filter, the Gaussian-reset recursion and the five mean recursions.
"""
function run_design(design::Design, rng::AbstractRNG, n_rep::Integer, horizon::Integer,
                    grid_points::Integer, grid_sd::Real, x::AbstractVector,
                    logw::AbstractVector, keep_paths::Bool)
    transition = make_grid_transition(design, grid_points, grid_sd)
    predictive_density = initial_grid_density(transition, state_variance(design), n_rep)
    posterior_density = similar(predictive_density)
    nu = design.nu
    k = residual_denominator(design)
    q = process_variance(design)
    obs_scale = observation_scale(design)
    tdist = TDist(nu)

    latent = sqrt(state_variance(design)) .* randn(rng, n_rep)
    approximate_predictive_means = zero_dict(RECURSIVE_METHODS, n_rep)

    path_sse_grid = zero_dict(RECURSIVE_METHODS, n_rep)
    path_sse_gaussian = zero_dict(RECURSIVE_METHODS, n_rep)
    path_sae_grid = zero_dict(RECURSIVE_METHODS, n_rep)
    path_sae_gaussian = zero_dict(RECURSIVE_METHODS, n_rep)
    path_bias_grid = zero_dict(RECURSIVE_METHODS, n_rep)
    path_bias_gaussian = zero_dict(RECURSIVE_METHODS, n_rep)
    path_sse_state = zero_dict(RECURSIVE_METHODS, n_rep)
    path_sae_state = zero_dict(RECURSIVE_METHODS, n_rep)
    path_errors_grid = matrix_dict(RECURSIVE_METHODS, horizon, n_rep)
    path_errors_gaussian = matrix_dict(RECURSIVE_METHODS, horizon, n_rep)
    exact_sse_state = zeros(n_rep)

    oracle_sse_local = zero_dict(RECURSIVE_APPROX_METHODS, n_rep)
    oracle_sse_total = zero_dict(RECURSIVE_APPROX_METHODS, n_rep)
    oracle_sae_local = zero_dict(RECURSIVE_APPROX_METHODS, n_rep)
    oracle_sae_total = zero_dict(RECURSIVE_APPROX_METHODS, n_rep)
    oracle_bias_local = zero_dict(RECURSIVE_APPROX_METHODS, n_rep)
    oracle_bias_total = zero_dict(RECURSIVE_APPROX_METHODS, n_rep)
    oracle_errors_local = matrix_dict(RECURSIVE_APPROX_METHODS, horizon, n_rep)
    oracle_errors_total = matrix_dict(RECURSIVE_APPROX_METHODS, horizon, n_rep)
    oracle_sse_closure = zeros(n_rep)
    oracle_sae_closure = zeros(n_rep)
    oracle_bias_closure = zeros(n_rep)
    oracle_errors_closure = Matrix{Float64}(undef, horizon, n_rep)
    oracle_sse_grid_correction = zeros(n_rep)

    max_edge_mass = 0.0
    max_implicit_residual = 0.0
    max_oracle_adjustment = 0.0
    max_recursive_adjustment = 0.0
    min_oracle_curvature = Inf
    min_recursive_curvature = Inf
    oracle_outside_count = 0
    recursive_outside_count = 0
    diagnostic_count = 0
    max_decomposition_residual = 0.0
    min_predictive_variance = Inf
    max_predictive_variance = 0.0

    history = nothing
    if keep_paths
        history = Dict{String,Matrix{Float64}}()
        for name in ("latent_state", "observation", "Exact grid", RECURSIVE_METHODS...)
            history[name] = Matrix{Float64}(undef, horizon, n_rep)
        end
    end

    for t in 1:horizon
        if t > 1
            latent = design.phi .* latent .+ sqrt(q) .* randn(rng, n_rep)
        end
        observation = latent .+ obs_scale .* rand(rng, tdist, n_rep)

        exact_predictive_mean, exact_predictive_variance =
            density_moments(predictive_density, transition)
        min_predictive_variance = min(min_predictive_variance, minimum(exact_predictive_variance))
        max_predictive_variance = max(max_predictive_variance, maximum(exact_predictive_variance))

        exact_posterior_mean, _, edge_mass =
            grid_update!(posterior_density, predictive_density, observation, design, transition)
        max_edge_mass = max(max_edge_mass, edge_mass)
        grid_correction = exact_posterior_mean .- exact_predictive_mean

        gaussian_correction = recursive_gaussian_correction(
            observation, exact_predictive_mean, exact_predictive_variance, design, x, logw)
        oracle, oracle_implicit_residual = recursive_corrections(
            observation, exact_predictive_mean, exact_predictive_variance, design)
        max_implicit_residual = max(max_implicit_residual, oracle_implicit_residual)
        max_oracle_adjustment = max(max_oracle_adjustment,
                                    maximum(abs.(oracle["ISD*"] .- oracle["Implicit"])))
        oracle_residual = observation .- exact_predictive_mean
        for i in 1:n_rep
            mode_h = student_derivatives_residual(oracle_residual[i] - oracle["Implicit"][i], nu, k)[2]
            min_oracle_curvature = min(min_oracle_curvature,
                                       1.0 - exact_predictive_variance[i] * mode_h)
            lower = min(0.0, oracle_residual[i])
            upper = max(0.0, oracle_residual[i])
            v = oracle["ISD*"][i]
            oracle_outside_count += (v < lower || v > upper) ? 1 : 0
        end
        closure_error = gaussian_correction .- grid_correction
        oracle_sse_closure .+= closure_error .* closure_error
        oracle_sae_closure .+= abs.(closure_error)
        oracle_bias_closure .+= closure_error
        oracle_errors_closure[t, :] = closure_error
        oracle_sse_grid_correction .+= grid_correction .* grid_correction
        for method in RECURSIVE_APPROX_METHODS
            local_error = oracle[method] .- gaussian_correction
            total_error = oracle[method] .- grid_correction
            max_decomposition_residual = max(max_decomposition_residual,
                maximum(abs.(total_error .- (local_error .+ closure_error))))
            oracle_sse_local[method] .+= local_error .* local_error
            oracle_sse_total[method] .+= total_error .* total_error
            oracle_sae_local[method] .+= abs.(local_error)
            oracle_sae_total[method] .+= abs.(total_error)
            oracle_bias_local[method] .+= local_error
            oracle_bias_total[method] .+= total_error
            oracle_errors_local[method][t, :] = local_error
            oracle_errors_total[method][t, :] = total_error
        end

        approximate_posteriors = Dict{String,Vector{Float64}}()
        gaussian_reset_correction = recursive_gaussian_correction(
            observation, approximate_predictive_means["Exact Gaussian"],
            exact_predictive_variance, design, x, logw)
        approximate_posteriors["Exact Gaussian"] =
            approximate_predictive_means["Exact Gaussian"] .+ gaussian_reset_correction
        for method in RECURSIVE_APPROX_METHODS
            method_corrections, method_implicit_residual = recursive_corrections(
                observation, approximate_predictive_means[method],
                exact_predictive_variance, design)
            max_implicit_residual = max(max_implicit_residual, method_implicit_residual)
            if method == "ISD*"
                max_recursive_adjustment = max(max_recursive_adjustment,
                    maximum(abs.(method_corrections["ISD*"] .- method_corrections["Implicit"])))
                method_residual = observation .- approximate_predictive_means[method]
                for i in 1:n_rep
                    mode_h = student_derivatives_residual(
                        method_residual[i] - method_corrections["Implicit"][i], nu, k)[2]
                    min_recursive_curvature = min(min_recursive_curvature,
                                                  1.0 - exact_predictive_variance[i] * mode_h)
                    lower = min(0.0, method_residual[i])
                    upper = max(0.0, method_residual[i])
                    v = method_corrections["ISD*"][i]
                    recursive_outside_count += (v < lower || v > upper) ? 1 : 0
                end
            end
            approximate_posteriors[method] =
                approximate_predictive_means[method] .+ method_corrections[method]
        end
        diagnostic_count += n_rep

        gaussian_reset_mean = approximate_posteriors["Exact Gaussian"]
        for method in RECURSIVE_METHODS
            grid_error = approximate_posteriors[method] .- exact_posterior_mean
            gaussian_error = approximate_posteriors[method] .- gaussian_reset_mean
            state_error = approximate_posteriors[method] .- latent
            path_sse_grid[method] .+= grid_error .* grid_error
            path_sse_gaussian[method] .+= gaussian_error .* gaussian_error
            path_sae_grid[method] .+= abs.(grid_error)
            path_sae_gaussian[method] .+= abs.(gaussian_error)
            path_bias_grid[method] .+= grid_error
            path_bias_gaussian[method] .+= gaussian_error
            path_sse_state[method] .+= state_error .* state_error
            path_sae_state[method] .+= abs.(state_error)
            path_errors_grid[method][t, :] = grid_error
            path_errors_gaussian[method][t, :] = gaussian_error
        end

        exact_state_error = exact_posterior_mean .- latent
        exact_sse_state .+= exact_state_error .* exact_state_error

        if history !== nothing
            history["latent_state"][t, :] = latent
            history["observation"][t, :] = observation
            history["Exact grid"][t, :] = exact_posterior_mean
            for method in RECURSIVE_METHODS
                history[method][t, :] = approximate_posteriors[method]
            end
        end

        if t < horizon
            grid_predict!(predictive_density, posterior_density, transition)
            for method in RECURSIVE_METHODS
                approximate_predictive_means[method] = design.phi .* approximate_posteriors[method]
            end
        end
    end

    pv = process_variance(design)
    path_rows = []
    exact_state_rmse, exact_state_mcse = rmse_mcse(exact_sse_state ./ horizon)
    push!(path_rows, (
        nu = design.nu, snr = design.snr, phi = design.phi, process_variance = pv,
        method = "Exact grid", n_rep = Int(n_rep), T = Int(horizon),
        rmse_vs_grid = 0.0, rmse_vs_grid_mcse = 0.0, mae_vs_grid = 0.0,
        median_ae_vs_grid = 0.0, bias_vs_grid = 0.0, median_path_rmse_vs_grid = 0.0,
        p90_path_rmse_vs_grid = 0.0,
        rmse_vs_gaussian_reset = NaN, rmse_vs_gaussian_reset_mcse = NaN,
        mae_vs_gaussian_reset = NaN, median_ae_vs_gaussian_reset = NaN,
        bias_vs_gaussian_reset = NaN,
        rmse_vs_state = exact_state_rmse, rmse_vs_state_mcse = exact_state_mcse,
    ))
    for method in RECURSIVE_METHODS
        per_rep_mse_grid = path_sse_grid[method] ./ horizon
        per_rep_mse_gaussian = path_sse_gaussian[method] ./ horizon
        per_rep_mse_state = path_sse_state[method] ./ horizon
        rmse_grid, mcse_grid = rmse_mcse(per_rep_mse_grid)
        rmse_gaussian, mcse_gaussian = rmse_mcse(per_rep_mse_gaussian)
        rmse_state, mcse_state = rmse_mcse(per_rep_mse_state)
        push!(path_rows, (
            nu = design.nu, snr = design.snr, phi = design.phi, process_variance = pv,
            method = method, n_rep = Int(n_rep), T = Int(horizon),
            rmse_vs_grid = rmse_grid,
            rmse_vs_grid_mcse = mcse_grid,
            mae_vs_grid = mean(path_sae_grid[method] ./ horizon),
            median_ae_vs_grid = median(abs.(path_errors_grid[method])),
            bias_vs_grid = mean(path_bias_grid[method] ./ horizon),
            median_path_rmse_vs_grid = median(sqrt.(per_rep_mse_grid)),
            p90_path_rmse_vs_grid = quantile(sqrt.(per_rep_mse_grid), 0.90),
            rmse_vs_gaussian_reset = rmse_gaussian,
            rmse_vs_gaussian_reset_mcse = mcse_gaussian,
            mae_vs_gaussian_reset = mean(path_sae_gaussian[method] ./ horizon),
            median_ae_vs_gaussian_reset = median(abs.(path_errors_gaussian[method])),
            bias_vs_gaussian_reset = mean(path_bias_gaussian[method] ./ horizon),
            rmse_vs_state = rmse_state,
            rmse_vs_state_mcse = mcse_state,
        ))
    end

    closure_rmse, closure_mcse = rmse_mcse(oracle_sse_closure ./ horizon)
    grid_correction_rms, _ = rmse_mcse(oracle_sse_grid_correction ./ horizon)
    oracle_rows = []
    push!(oracle_rows, (
        nu = design.nu, snr = design.snr, phi = design.phi, method = "Exact Gaussian",
        n_rep = Int(n_rep), T = Int(horizon),
        rmse_local_gaussian = 0.0, rmse_local_gaussian_mcse = 0.0, mae_local_gaussian = 0.0,
        median_ae_local_gaussian = 0.0, bias_local_gaussian = 0.0,
        rmse_total_grid = closure_rmse, rmse_total_grid_mcse = closure_mcse,
        mae_total_grid = mean(oracle_sae_closure ./ horizon),
        median_ae_total_grid = median(abs.(oracle_errors_closure)),
        bias_total_grid = mean(oracle_bias_closure ./ horizon),
        gaussian_closure_rmse = closure_rmse,
        grid_correction_rms = grid_correction_rms,
        relative_total_rmse = closure_rmse / grid_correction_rms,
    ))
    for method in RECURSIVE_APPROX_METHODS
        local_rmse, local_mcse = rmse_mcse(oracle_sse_local[method] ./ horizon)
        total_rmse, total_mcse = rmse_mcse(oracle_sse_total[method] ./ horizon)
        push!(oracle_rows, (
            nu = design.nu, snr = design.snr, phi = design.phi, method = method,
            n_rep = Int(n_rep), T = Int(horizon),
            rmse_local_gaussian = local_rmse,
            rmse_local_gaussian_mcse = local_mcse,
            mae_local_gaussian = mean(oracle_sae_local[method] ./ horizon),
            median_ae_local_gaussian = median(abs.(oracle_errors_local[method])),
            bias_local_gaussian = mean(oracle_bias_local[method] ./ horizon),
            rmse_total_grid = total_rmse,
            rmse_total_grid_mcse = total_mcse,
            mae_total_grid = mean(oracle_sae_total[method] ./ horizon),
            median_ae_total_grid = median(abs.(oracle_errors_total[method])),
            bias_total_grid = mean(oracle_bias_total[method] ./ horizon),
            gaussian_closure_rmse = closure_rmse,
            grid_correction_rms = grid_correction_rms,
            relative_total_rmse = total_rmse / grid_correction_rms,
        ))
    end

    paired_rows = [
        paired_loss_row(design, "one_step_gaussian",
                        oracle_sse_local["Implicit"], oracle_sse_local["ISD*"],
                        oracle_sae_local["Implicit"], oracle_sae_local["ISD*"], horizon),
        paired_loss_row(design, "one_step_grid",
                        oracle_sse_total["Implicit"], oracle_sse_total["ISD*"],
                        oracle_sae_total["Implicit"], oracle_sae_total["ISD*"], horizon),
        paired_loss_row(design, "recursive_gaussian_reset",
                        path_sse_gaussian["Implicit"], path_sse_gaussian["ISD*"],
                        path_sae_gaussian["Implicit"], path_sae_gaussian["ISD*"], horizon),
        paired_loss_row(design, "recursive_grid",
                        path_sse_grid["Implicit"], path_sse_grid["ISD*"],
                        path_sae_grid["Implicit"], path_sae_grid["ISD*"], horizon),
        paired_loss_row(design, "latent_state",
                        path_sse_state["Implicit"], path_sse_state["ISD*"],
                        path_sae_state["Implicit"], path_sae_state["ISD*"], horizon),
    ]

    diagnostics = (
        nu = design.nu,
        snr = design.snr,
        phi = design.phi,
        process_variance = pv,
        observation_scale = obs_scale,
        grid_points = Int(grid_points),
        grid_sd = Float64(grid_sd),
        grid_dx = transition.dx,
        max_posterior_edge_mass = max_edge_mass,
        max_implicit_equation_residual = max_implicit_residual,
        max_oracle_mean_mode_adjustment = max_oracle_adjustment,
        max_recursive_mcisd_mean_mode_adjustment = max_recursive_adjustment,
        min_oracle_mcisd_curvature = min_oracle_curvature,
        min_recursive_mcisd_curvature = min_recursive_curvature,
        oracle_mcisd_outside_innovation_share = oracle_outside_count / diagnostic_count,
        recursive_mcisd_outside_innovation_share = recursive_outside_count / diagnostic_count,
        max_error_decomposition_residual = max_decomposition_residual,
        min_exact_predictive_variance = min_predictive_variance,
        max_exact_predictive_variance = max_predictive_variance,
        approximate_covariance_source = "exact grid predictive variance",
    )
    return path_rows, oracle_rows, paired_rows, diagnostics, history
end

"""Index (one-based) of the path with the median total squared distance to the grid mean."""
function select_representative_path(history::AbstractDict)
    exact = history["Exact grid"]
    total_mse = zeros(size(exact, 2))
    for method in RECURSIVE_METHODS
        total_mse .+= vec(mean((history[method] .- exact) .^ 2; dims = 1))
    end
    rank = sortperm(total_mse)
    return rank[length(rank) ÷ 2 + 1]
end

"""Write representative_recursive_path.csv and return the (one-based) path index."""
function save_representative_path(history::AbstractDict, output_dir::AbstractString)
    index = select_representative_path(history)
    horizon = size(history["Exact grid"], 1)
    rows = [(
        t = t,
        latent_state = history["latent_state"][t, index],
        observation = history["observation"][t, index],
        exact_grid = history["Exact grid"][t, index],
        exact_gaussian = history["Exact Gaussian"][t, index],
        first_order = history["First order"][t, index],
        second_order = history["Second order"][t, index],
        implicit = history["Implicit"][t, index],
        mcisd = history["ISD*"][t, index],
        kalman = history["Kalman"][t, index],
    ) for t in 1:horizon]
    write_rows_csv(joinpath(output_dir, "representative_recursive_path.csv"), rows)
    return index
end

pooled_rmse(values) = sqrt(mean(abs2, values))

"""Pooled ISD versus ISD* summary (mcisd_vs_implicit_pooled.csv)."""
function pooled_paired_summary(paired_rows::AbstractVector)
    targets = unique([r.target for r in paired_rows])
    out = []
    for target in targets
        target_rows = [r for r in paired_rows if r.target == target]
        nus = sort(unique([r.nu for r in target_rows]))
        groups = [(fmt_g(nu), [r for r in target_rows if r.nu == nu]) for nu in nus]
        push!(groups, ("All", target_rows))
        for (nu_label, group) in groups
            g = length(group)
            implicit_rmse = pooled_rmse([r.implicit_rmse for r in group])
            isdstar_rmse = pooled_rmse([r.mcisd_rmse for r in group])
            mse_difference = mean([r.paired_mse_difference for r in group])
            mse_difference_mcse = sqrt(sum(abs2, [r.paired_mse_difference_mcse for r in group])) / g
            implicit_mae = mean([r.implicit_mae for r in group])
            isdstar_mae = mean([r.mcisd_mae for r in group])
            mae_difference = mean([r.paired_mae_difference for r in group])
            mae_difference_mcse = sqrt(sum(abs2, [r.paired_mae_difference_mcse for r in group])) / g
            push!(out, (
                target = target,
                nu = nu_label,
                design_cells = g,
                implicit_rmse = implicit_rmse,
                mcisd_rmse = isdstar_rmse,
                rmse_reduction_percent = 100.0 * (1.0 - isdstar_rmse / implicit_rmse),
                implicit_mae = implicit_mae,
                mcisd_mae = isdstar_mae,
                mae_reduction_percent = 100.0 * (1.0 - isdstar_mae / implicit_mae),
                paired_mse_difference = mse_difference,
                paired_mse_difference_mcse = mse_difference_mcse,
                paired_mse_difference_ci_low = mse_difference - 1.96 * mse_difference_mcse,
                paired_mse_difference_ci_high = mse_difference + 1.96 * mse_difference_mcse,
                paired_mae_difference = mae_difference,
                paired_mae_difference_mcse = mae_difference_mcse,
                paired_mae_difference_ci_low = mae_difference - 1.96 * mae_difference_mcse,
                paired_mae_difference_ci_high = mae_difference + 1.96 * mae_difference_mcse,
                path_mse_win_share = mean([r.path_mse_win_share for r in group]),
                path_mae_win_share = mean([r.path_mae_win_share for r in group]),
            ))
        end
    end
    return out
end

const ORACLE_POOLED_FIELDS = (:nu, :method, Symbol("local"), :total, :closure)

"""Pooled oracle errors by nu (recursive_oracle_pooled.csv)."""
function pooled_oracle_summary(oracle_rows::AbstractVector)
    out = []
    for nu in sort(unique([r.nu for r in oracle_rows])), method in RECURSIVE_METHODS
        group = [r for r in oracle_rows if r.nu == nu && r.method == method]
        # "local" is a Julia keyword, so this row is built with an explicit field list.
        push!(out, NamedTuple{ORACLE_POOLED_FIELDS}((
            nu,
            method,
            pooled_rmse([r.rmse_local_gaussian for r in group]),
            pooled_rmse([r.rmse_total_grid for r in group]),
            pooled_rmse([r.gaussian_closure_rmse for r in group]),
        )))
    end
    return out
end

"""Pooled one-step and recursive errors by nu (recursive_compact_summary.csv)."""
function compact_summary(path_rows::AbstractVector, oracle_rows::AbstractVector)
    out = []
    for nu in sort(unique([r.nu for r in oracle_rows])), method in RECURSIVE_METHODS
        oracle_group = [r for r in oracle_rows if r.nu == nu && r.method == method]
        path_group = [r for r in path_rows if r.nu == nu && r.method == method]
        push!(out, (
            nu = nu,
            method = method,
            one_step_gaussian = pooled_rmse([r.rmse_local_gaussian for r in oracle_group]),
            one_step_grid = pooled_rmse([r.rmse_total_grid for r in oracle_group]),
            recursive_gaussian = pooled_rmse([r.rmse_vs_gaussian_reset for r in path_group]),
            recursive_grid = pooled_rmse([r.rmse_vs_grid for r in path_group]),
        ))
    end
    return out
end

# ---------------------------------------------------------------------------
# Formula checks (validate_derivatives and validate_gaussian_location_identity)
# ---------------------------------------------------------------------------

allclose(a, b; rtol, atol) = all(abs.(a .- b) .<= atol .+ rtol .* abs.(b))

"""Finite-difference check of the analytic Student-t derivatives."""
function validate_student_derivatives()
    residuals = [-4.0, -1.2, -0.1, 0.0, 0.7, 2.5]
    theta = zeros(length(residuals))
    y = copy(residuals)
    h1 = 1.0e-5
    h3 = 2.0e-4
    max_s = 0.0
    max_h = 0.0
    max_t = 0.0
    for nu in (3.0, 5.0, 10.0)
        k = nu - 2.0
        ell = th -> student_loglik_residual.(y .- th, nu, k)
        derivs = th -> student_derivatives_residual.(y .- th, nu, k)
        analytic = derivs(theta)
        analytic_s = first.(analytic)
        analytic_h = getindex.(analytic, 2)
        analytic_t = last.(analytic)
        numeric_s = (ell(theta .+ h1) .- ell(theta .- h1)) ./ (2.0 * h1)
        numeric_h = (first.(derivs(theta .+ h1)) .- first.(derivs(theta .- h1))) ./ (2.0 * h1)
        numeric_t = (getindex.(derivs(theta .+ h3), 2) .- getindex.(derivs(theta .- h3), 2)) ./
                    (2.0 * h3)
        max_s = max(max_s, maximum(abs.(analytic_s .- numeric_s)))
        max_h = max(max_h, maximum(abs.(analytic_h .- numeric_h)))
        max_t = max(max_t, maximum(abs.(analytic_t .- numeric_t)))
        allclose(analytic_s, numeric_s; rtol = 2e-7, atol = 2e-8) ||
            error("Student-t score derivative check failed for nu=$(nu)")
        allclose(analytic_h, numeric_h; rtol = 2e-7, atol = 2e-8) ||
            error("Student-t Hessian check failed for nu=$(nu)")
        allclose(analytic_t, numeric_t; rtol = 3e-6, atol = 3e-7) ||
            error("Student-t third-derivative check failed for nu=$(nu)")
    end
    return jobj(
        "student_t_derivatives_passed" => true,
        "maximum_score_finite_difference_error" => max_s,
        "maximum_hessian_finite_difference_error" => max_h,
        "maximum_third_finite_difference_error" => max_t,
    )
end

"""End-to-end check of the Gaussian location limit with nu = 1e8."""
function validate_gaussian_location_identity()
    rng = Random.Xoshiro(20260815)
    n = 500
    center = randn(rng, n)
    observation = center .+ randn(rng, n)
    predictive_variance = exp.(log(0.005) .+ (log(1.5) - log(0.005)) .* rand(rng, n))
    design = Design(1.0e8, 1.0, 0.8)
    x, w = hermite_rule(80)
    logw = hermite_logweights(w)
    exact_limit = recursive_gaussian_correction(observation, center, predictive_variance,
                                                design, x, logw)
    implemented, implicit_residual = recursive_corrections(observation, center,
                                                           predictive_variance, design)
    kalman_answer = predictive_variance ./ (1.0 .+ predictive_variance) .* (observation .- center)
    exact_error = maximum(abs.(exact_limit .- kalman_answer))
    implicit_error = maximum(abs.(implemented["Implicit"] .- kalman_answer))
    isdstar_error = maximum(abs.(implemented["ISD*"] .- kalman_answer))
    kalman_error = maximum(abs.(implemented["Kalman"] .- kalman_answer))
    if max(exact_error, implicit_error, isdstar_error, kalman_error, implicit_residual) > 2.0e-7
        error("End-to-end Gaussian-limit check failed")
    end
    return jobj(
        "gaussian_location_identity_passed" => true,
        "large_nu" => design.nu,
        "maximum_large_nu_exact_kalman_error" => exact_error,
        "maximum_large_nu_implicit_kalman_error" => implicit_error,
        "maximum_large_nu_mcisd_kalman_error" => isdstar_error,
        "maximum_implemented_kalman_error" => kalman_error,
        "maximum_large_nu_implicit_equation_residual" => implicit_residual,
    )
end

function write_formula_validation(output_dir::AbstractString)
    checks = vcat(validate_student_derivatives(), validate_gaussian_location_identity())
    write_json(joinpath(output_dir, "formula_validation.json"), checks)
    return checks
end

# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

const RECURSIVE_DEFAULTS = (
    n_rep = 1_000, horizon = 500, nus = [3.0, 5.0, 10.0], snrs = [0.25, 1.0],
    phis = [0.80, 0.98], grid_points = 481, grid_sd = 8.0, gh_nodes = 32,
    seed = 20260815, representative_design = [3.0, 1.0, 0.98],
    grid_check_n_rep = 200, grid_check_T = 200, grid_check_coarse_points = 241,
    grid_check_fine_points = 961, grid_check_fine_gh_nodes = 64,
)

function validate_recursive_args(; n_rep, horizon, nus, snrs, phis, gh_nodes, grid_points,
                                 grid_check_n_rep, grid_check_T, grid_check_coarse_points,
                                 grid_check_fine_points, grid_check_fine_gh_nodes)
    (n_rep < 2 || horizon < 2) && throw(ArgumentError("n_rep and T must both be at least two"))
    any(nu -> nu <= 2.0, nus) && throw(ArgumentError("All Student-t degrees of freedom must exceed two"))
    any(s -> s <= 0.0, snrs) && throw(ArgumentError("All signal-to-noise ratios must be positive"))
    any(p -> p <= 0.0 || p >= 1.0, phis) &&
        throw(ArgumentError("All persistence parameters must lie strictly between zero and one"))
    gh_nodes < 12 && throw(ArgumentError("At least 12 Gauss-Hermite nodes are required"))
    (grid_check_n_rep < 2 || grid_check_T < 2) &&
        throw(ArgumentError("Grid-check n_rep and T must both be at least two"))
    points = (grid_check_coarse_points, grid_points, grid_check_fine_points)
    (points[1] < points[2] < points[3]) ||
        throw(ArgumentError("Grid-check point counts must be strictly increasing"))
    any(p -> iseven(p) || p < 101, points) &&
        throw(ArgumentError("Every grid-check point count must be odd and at least 101"))
    grid_check_fine_gh_nodes <= gh_nodes &&
        throw(ArgumentError("The refined quadrature must have more nodes than the default"))
    return nothing
end

"""
    run_recursive_experiment(; output_dir, kwargs...)

Run the twelve designs (phi outer, snr middle, nu inner, as in the Python
code) and write recursive_path_results.csv, recursive_oracle_results.csv,
mcisd_vs_implicit_results.csv, mcisd_vs_implicit_pooled.csv,
recursive_diagnostics.csv, recursive_oracle_pooled.csv,
recursive_compact_summary.csv, representative_recursive_path.csv,
formula_validation.json and recursive_run_metadata.json.
Keyword defaults are in `RECURSIVE_DEFAULTS`.
"""
function run_recursive_experiment(; output_dir::AbstractString = "output_recursive_mcisd",
                                  n_rep::Integer = RECURSIVE_DEFAULTS.n_rep,
                                  horizon::Integer = RECURSIVE_DEFAULTS.horizon,
                                  nus::AbstractVector = RECURSIVE_DEFAULTS.nus,
                                  snrs::AbstractVector = RECURSIVE_DEFAULTS.snrs,
                                  phis::AbstractVector = RECURSIVE_DEFAULTS.phis,
                                  grid_points::Integer = RECURSIVE_DEFAULTS.grid_points,
                                  grid_sd::Real = RECURSIVE_DEFAULTS.grid_sd,
                                  gh_nodes::Integer = RECURSIVE_DEFAULTS.gh_nodes,
                                  seed::Integer = RECURSIVE_DEFAULTS.seed,
                                  representative_design::AbstractVector =
                                      RECURSIVE_DEFAULTS.representative_design,
                                  verbose::Bool = true)
    validate_recursive_args(; n_rep, horizon, nus, snrs, phis, gh_nodes, grid_points,
                            grid_check_n_rep = RECURSIVE_DEFAULTS.grid_check_n_rep,
                            grid_check_T = RECURSIVE_DEFAULTS.grid_check_T,
                            grid_check_coarse_points = RECURSIVE_DEFAULTS.grid_check_coarse_points,
                            grid_check_fine_points = RECURSIVE_DEFAULTS.grid_check_fine_points,
                            grid_check_fine_gh_nodes = RECURSIVE_DEFAULTS.grid_check_fine_gh_nodes)
    mkpath(output_dir)
    write_formula_validation(output_dir)

    designs = [Design(nu, snr, phi) for phi in phis for snr in snrs for nu in nus]
    representative = Design(representative_design...)
    representative in designs ||
        throw(ArgumentError("The representative design must be included in nus, snrs, and phis"))

    x, w = hermite_rule(gh_nodes)
    logw = hermite_logweights(w)
    seeds = child_seeds(seed, length(designs))

    path_rows = []
    oracle_rows = []
    paired_rows = []
    diagnostic_rows = []
    representative_history = nothing
    start = time()
    for (idx, design) in enumerate(designs)
        design_start = time()
        paths, oracle, paired, diagnostics, history = run_design(
            design, Random.Xoshiro(seeds[idx]), n_rep, horizon, grid_points, grid_sd,
            x, logw, design == representative)
        elapsed = time() - design_start
        append!(path_rows, paths)
        append!(oracle_rows, oracle)
        append!(paired_rows, paired)
        push!(diagnostic_rows, merge(diagnostics, (runtime_seconds = elapsed,)))
        history === nothing || (representative_history = history)
        verbose && @printf("completed %-24s in %7.2f seconds\n", design_key(design), elapsed)
    end
    total_runtime = time() - start

    write_rows_csv(joinpath(output_dir, "recursive_path_results.csv"), path_rows)
    write_rows_csv(joinpath(output_dir, "recursive_oracle_results.csv"), oracle_rows)
    write_rows_csv(joinpath(output_dir, "mcisd_vs_implicit_results.csv"), paired_rows)
    write_rows_csv(joinpath(output_dir, "mcisd_vs_implicit_pooled.csv"),
                   pooled_paired_summary(paired_rows))
    write_rows_csv(joinpath(output_dir, "recursive_diagnostics.csv"), diagnostic_rows)
    write_rows_csv(joinpath(output_dir, "recursive_oracle_pooled.csv"),
                   pooled_oracle_summary(oracle_rows))
    write_rows_csv(joinpath(output_dir, "recursive_compact_summary.csv"),
                   compact_summary(path_rows, oracle_rows))

    representative_index = -1
    if representative_history !== nothing
        representative_index = save_representative_path(representative_history, output_dir) - 1
    end

    metadata = jobj(
        "seed" => seed,
        "n_rep" => n_rep,
        "T" => horizon,
        "nus" => collect(Float64, nus),
        "snrs" => collect(Float64, snrs),
        "phis" => collect(Float64, phis),
        "grid_points" => grid_points,
        "grid_sd" => grid_sd,
        "gh_nodes" => gh_nodes,
        "representative_design" => collect(Float64, representative_design),
        "representative_replication_zero_based" => representative_index,
        "runtime_seconds" => total_runtime,
        "session" => session_info(),
    )
    write_json(joinpath(output_dir, "recursive_run_metadata.json"), metadata)

    if verbose
        println("\nPooled recursive RMSE across the $(length(designs)) designs:")
        for method in RECURSIVE_METHODS
            values = [r.rmse_vs_grid for r in path_rows if r.method == method]
            @printf("%-16s %.5f\n", method, pooled_rmse(values))
        end
        @printf("\nTotal runtime: %.2f seconds\n", total_runtime)
    end
    return (paths = path_rows, oracle = oracle_rows, paired = paired_rows,
            diagnostics = diagnostic_rows)
end

"""
    run_recursive_grid_checks(; output_dir, kwargs...)

Formula checks and the nested-grid and quadrature refinement diagnostics
(`--grid-check-only` in the Python code). Two designs are checked: the
representative design and (max nu, min snr, min phi). Each design is simulated
once per grid size from the same seed, so the grids see identical data.
Writes formula_validation.json, grid_refinement_diagnostics.csv and
quadrature_refinement_diagnostics.csv.
"""
function run_recursive_grid_checks(; output_dir::AbstractString = "output_recursive_mcisd_checks",
                                   nus::AbstractVector = RECURSIVE_DEFAULTS.nus,
                                   snrs::AbstractVector = RECURSIVE_DEFAULTS.snrs,
                                   phis::AbstractVector = RECURSIVE_DEFAULTS.phis,
                                   grid_points::Integer = RECURSIVE_DEFAULTS.grid_points,
                                   grid_sd::Real = RECURSIVE_DEFAULTS.grid_sd,
                                   gh_nodes::Integer = RECURSIVE_DEFAULTS.gh_nodes,
                                   seed::Integer = RECURSIVE_DEFAULTS.seed,
                                   representative_design::AbstractVector =
                                       RECURSIVE_DEFAULTS.representative_design,
                                   grid_check_n_rep::Integer = RECURSIVE_DEFAULTS.grid_check_n_rep,
                                   grid_check_T::Integer = RECURSIVE_DEFAULTS.grid_check_T,
                                   grid_check_coarse_points::Integer =
                                       RECURSIVE_DEFAULTS.grid_check_coarse_points,
                                   grid_check_fine_points::Integer =
                                       RECURSIVE_DEFAULTS.grid_check_fine_points,
                                   grid_check_fine_gh_nodes::Integer =
                                       RECURSIVE_DEFAULTS.grid_check_fine_gh_nodes,
                                   verbose::Bool = true)
    validate_recursive_args(; n_rep = RECURSIVE_DEFAULTS.n_rep, horizon = RECURSIVE_DEFAULTS.horizon,
                            nus, snrs, phis, gh_nodes, grid_points, grid_check_n_rep,
                            grid_check_T, grid_check_coarse_points, grid_check_fine_points,
                            grid_check_fine_gh_nodes)
    mkpath(output_dir)
    write_formula_validation(output_dir)
    x, w = hermite_rule(gh_nodes)
    logw = hermite_logweights(w)
    candidates = [Design(representative_design...),
                  Design(maximum(nus), minimum(snrs), minimum(phis))]
    check_designs = unique(candidates)
    seeds = child_seeds(seed + 7_919, length(check_designs))

    path_columns = (:rmse_vs_grid, :mae_vs_grid, :median_ae_vs_grid,
                    :rmse_vs_gaussian_reset, :mae_vs_gaussian_reset, :rmse_vs_state)
    oracle_columns = (:rmse_local_gaussian, :mae_local_gaussian, :median_ae_local_gaussian,
                      :rmse_total_grid, :mae_total_grid, :median_ae_total_grid)

    # Largest absolute and relative differences between matching rows, ignoring NaN.
    function metric_differences(coarse_rows, fine_rows, columns)
        coarse = Dict(r.method => r for r in coarse_rows)
        max_abs = 0.0
        max_rel = 0.0
        for fine in fine_rows, col in columns
            a = getproperty(coarse[fine.method], col)
            b = getproperty(fine, col)
            diff = abs(a - b)
            isnan(diff) && continue
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, diff / max(abs(b), 1.0e-10))
        end
        return max_abs, max_rel
    end

    records = []
    quadrature_records = []
    for (design, design_seed) in zip(check_designs, seeds)
        point_sequence = (grid_check_coarse_points, grid_points, grid_check_fine_points)
        runs = Dict{Int,Any}()
        for points in point_sequence
            runs[points] = run_design(design, Random.Xoshiro(design_seed), grid_check_n_rep,
                                      grid_check_T, points, grid_sd, x, logw, true)
        end
        for (coarse_points, fine_points) in zip(point_sequence[1:end-1], point_sequence[2:end])
            coarse = runs[coarse_points]
            fine = runs[fine_points]
            exact_difference = coarse[5]["Exact grid"] .- fine[5]["Exact grid"]
            path_abs, path_rel = metric_differences(coarse[1], fine[1], path_columns)
            oracle_abs, oracle_rel = metric_differences(coarse[2], fine[2], oracle_columns)
            push!(records, (
                nu = design.nu,
                snr = design.snr,
                phi = design.phi,
                n_rep = Int(grid_check_n_rep),
                T = Int(grid_check_T),
                coarse_grid_points = Int(coarse_points),
                fine_grid_points = Int(fine_points),
                exact_grid_mean_difference_rmse = sqrt(mean(exact_difference .^ 2)),
                exact_grid_mean_difference_max_abs = maximum(abs.(exact_difference)),
                maximum_absolute_path_metric_difference = path_abs,
                maximum_relative_path_metric_difference = path_rel,
                maximum_absolute_oracle_metric_difference = oracle_abs,
                maximum_relative_oracle_metric_difference = oracle_rel,
                coarse_maximum_edge_mass = coarse[4].max_posterior_edge_mass,
                fine_maximum_edge_mass = fine[4].max_posterior_edge_mass,
            ))
            verbose && @printf("grid check %-24s %d->%d: mean RMSE=%.3e\n", design_key(design),
                               coarse_points, fine_points,
                               records[end].exact_grid_mean_difference_rmse)
        end

        fine_x, fine_w = hermite_rule(grid_check_fine_gh_nodes)
        refined = run_design(design, Random.Xoshiro(design_seed), grid_check_n_rep, grid_check_T,
                             grid_points, grid_sd, fine_x, hermite_logweights(fine_w), false)
        base = runs[grid_points]
        push!(quadrature_records, (
            nu = design.nu,
            snr = design.snr,
            phi = design.phi,
            n_rep = Int(grid_check_n_rep),
            T = Int(grid_check_T),
            grid_points = Int(grid_points),
            base_gh_nodes = Int(gh_nodes),
            refined_gh_nodes = Int(grid_check_fine_gh_nodes),
            maximum_absolute_path_metric_difference =
                metric_differences(base[1], refined[1], path_columns)[1],
            maximum_absolute_oracle_metric_difference =
                metric_differences(base[2], refined[2], oracle_columns)[1],
        ))
    end
    write_rows_csv(joinpath(output_dir, "grid_refinement_diagnostics.csv"), records)
    write_rows_csv(joinpath(output_dir, "quadrature_refinement_diagnostics.csv"), quadrature_records)
    return (grid = records, quadrature = quadrature_records)
end
