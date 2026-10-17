# Section 7.1: approximation under an exact Gaussian predictive law.
# Port of score_approximation_experiment.py.
#
# In each replication theta ~ N(a, P) and y ~ p(y | theta). The four updates
# are compared with the exact posterior-mean correction E(theta - a | y),
# evaluated by Gauss-Hermite quadrature. All methods use common draws.

const STATIC_METHODS = ("ESD", "ISD", "ISD*", "Explicit second order")
const STATIC_THEORETICAL_SLOPES = jobj("ESD" => 2, "ISD" => 2, "ISD*" => 3,
                                       "Explicit second order" => 3)
const STATIC_P_GRID = [0.00015625, 0.0003125, 0.000625, 0.00125, 0.0025, 0.005,
                       0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

static_main_models() = (PoissonLogIntensity(), GaussianLogVariance(),
                        StudentTLocation(5.0; key = "student_t"))
static_robustness_models() = (StudentTLocation(3.0), StudentTLocation(10.0))

"""
    static_exact_correction(model, y, a, P, x, logw) -> Vector

E(theta - a | y) for each observation, by Gauss-Hermite quadrature with the
physicists' nodes `x` and log weights `logw` of the normalized rule.
"""
function static_exact_correction(model::StaticModel, y::AbstractVector, a::Real, P::Real,
                                 x::AbstractVector, logw::AbstractVector)
    out = Vector{Float64}(undef, length(y))
    buffer = similar(x, Float64)
    for i in eachindex(y)
        yi = y[i]
        out[i] = gaussian_posterior_correction(theta -> loglik(model, theta, yi),
                                               a, P, x, logw, buffer)
    end
    return out
end

"""
    static_approximations(model, y, a, P) -> (Dict of corrections, diagnostics)

The four corrections of the paper and the solver diagnostics
(`compute_approximations` in the Python code).
"""
function static_approximations(model::StaticModel, y::AbstractVector, a::Real, P::Real)
    n = length(y)
    esd = Vector{Float64}(undef, n)
    explicit2 = Vector{Float64}(undef, n)
    for i in 1:n
        s, h, t = derivatives(model, y[i], a)
        esd[i] = esd_update(P, s)
        explicit2[i] = explicit2_update(P, s, h, t)
    end
    isd, W, T_at_mode, residual, min_curvature = isd_bracketed_newton(model, y, a, P)
    isdstar = [isdstar_update(isd[i], W[i], T_at_mode[i]) for i in 1:n]
    approximations = Dict("ESD" => esd, "ISD" => isd, "ISD*" => isdstar,
                          "Explicit second order" => explicit2)
    diagnostics = (max_implicit_residual = residual,
                   min_curvature_denominator = min_curvature,
                   max_abs_mean_mode_adjustment = maximum(abs.(0.5 .* W .* W .* T_at_mode)))
    return approximations, diagnostics
end

"""Rows of approximation_results.csv for one (model, P) cell (`summarize_errors`)."""
function static_summarize_errors(model::StaticModel, P::Real, exact::AbstractVector,
                                 approximations::AbstractDict, diagnostics::NamedTuple,
                                 n_rep::Integer)
    exact_rms = sqrt(mean(exact .* exact))
    rows = []
    for method in STATIC_METHODS
        err = approximations[method] .- exact
        sq = err .* err
        abs_err = abs.(err)
        rmse = sqrt(mean(sq))
        mcse = rmse > 0.0 ? std(sq; corrected = true) / sqrt(n_rep) / (2.0 * rmse) : 0.0
        push!(rows, (
            model = model_key(model),
            model_label = model_label(model),
            P = Float64(P),
            approximation = method,
            n_rep = Int(n_rep),
            bias = mean(err),
            mae = mean(abs_err),
            median_ae = median(abs_err),
            p95_ae = quantile(abs_err, 0.95),
            rmse = rmse,
            exact_correction_rms = exact_rms,
            relative_rmse = rmse / exact_rms,
            max_implicit_residual = diagnostics.max_implicit_residual,
            min_curvature_denominator = diagnostics.min_curvature_denominator,
            max_abs_mean_mode_adjustment = diagnostics.max_abs_mean_mode_adjustment,
            rmse_mcse = mcse,
        ))
    end
    return rows
end

"""
    static_slopes(rows, max_P)

Least-squares slopes of log RMSE on log P over the covariance values
P <= max_P, for each (model, method) in order of first appearance
(`estimate_slopes`).
"""
function static_slopes(rows::AbstractVector, max_P::Real)
    groups = Tuple{String,String,String}[]
    for r in rows
        g = (r.model, r.model_label, r.approximation)
        g in groups || push!(groups, g)
    end
    out = []
    for (model, label, method) in groups
        sel = [r for r in rows if r.model == model && r.approximation == method && r.P <= max_P]
        sort!(sel; by = r -> r.P)
        x = log.([r.P for r in sel])
        y = log.([r.rmse for r in sel])
        n = length(x)
        xm = mean(x)
        ym = mean(y)
        sxx = sum((x .- xm) .^ 2)
        slope = sum((x .- xm) .* (y .- ym)) / sxx
        intercept = ym - slope * xm
        resid = y .- (intercept .+ slope .* x)
        sse = sum(resid .* resid)
        sst = sum((y .- ym) .^ 2)
        push!(out, (
            model = model,
            model_label = label,
            approximation = method,
            max_P_in_fit = Float64(max_P),
            n_points = n,
            slope = slope,
            slope_se = sqrt(sse / (n - 2) / sxx),
            intercept = intercept,
            r_squared = 1.0 - sse / sst,
        ))
    end
    return out
end

"""Simulate one (model, P) cell and return its four result rows."""
function static_cell(model::StaticModel, P::Real, rng::AbstractRNG, n_rep::Integer,
                     x::AbstractVector, logw::AbstractVector)
    a = prior_center(model)
    theta = a .+ sqrt(P) .* randn(rng, n_rep)
    y = simulate(rng, model, theta)
    exact = static_exact_correction(model, y, a, P, x, logw)
    approximations, diagnostics = static_approximations(model, y, a, P)
    return static_summarize_errors(model, P, exact, approximations, diagnostics, n_rep)
end

"""
    run_static_experiment(; n_rep = 100_000, gh_nodes = 80, seed = 20260814,
                          p_grid = STATIC_P_GRID, slope_max_p = 0.0025, output_dir)

Run the static experiment and write approximation_results.csv,
solver_diagnostics.csv, convergence_slopes.csv, student_df_results.csv,
student_df_slopes.csv and metadata.json to `output_dir`. As in the Python
code, the main models use one stream per (model, P) cell derived from `seed`,
and the t_3 and t_10 robustness models use streams derived from `seed + 1`.
"""
function run_static_experiment(; n_rep::Integer = 100_000, gh_nodes::Integer = 80,
                               seed::Integer = 20260814,
                               p_grid::AbstractVector = STATIC_P_GRID,
                               slope_max_p::Real = 0.0025,
                               output_dir::AbstractString = "output_static_all_methods",
                               verbose::Bool = true)
    start_time = time()
    mkpath(output_dir)
    x, w = hermite_rule(gh_nodes)
    logw = hermite_logweights(w)

    models = static_main_models()
    seeds = child_seeds(seed, length(models) * length(p_grid))
    rows = []
    cell = 0
    for model in models, P in p_grid
        cell += 1
        append!(rows, static_cell(model, P, Random.Xoshiro(seeds[cell]), n_rep, x, logw))
        verbose && @printf("completed %-14s P=%s\n", model_key(model), fmt_g(P))
    end
    write_rows_csv(joinpath(output_dir, "approximation_results.csv"), rows)

    diagnostic_rows = []
    seen = Set{Tuple{String,Float64}}()
    for r in rows
        key = (r.model, r.P)
        key in seen && continue
        push!(seen, key)
        push!(diagnostic_rows, (model = r.model, model_label = r.model_label, P = r.P,
                                max_implicit_residual = r.max_implicit_residual,
                                min_curvature_denominator = r.min_curvature_denominator,
                                max_abs_mean_mode_adjustment = r.max_abs_mean_mode_adjustment))
    end
    write_rows_csv(joinpath(output_dir, "solver_diagnostics.csv"), diagnostic_rows)
    slopes = static_slopes(rows, slope_max_p)
    write_rows_csv(joinpath(output_dir, "convergence_slopes.csv"), slopes)

    # Tail-thickness robustness: the t_5 rows of the main run plus t_3 and t_10
    # from a separate seed stream, so that the main experiment is unchanged.
    robustness_rows = Any[r for r in rows if r.model == "student_t"]
    robustness_models = static_robustness_models()
    robustness_seeds = child_seeds(seed + 1, length(robustness_models) * length(p_grid))
    cell = 0
    for model in robustness_models, P in p_grid
        cell += 1
        append!(robustness_rows,
                static_cell(model, P, Random.Xoshiro(robustness_seeds[cell]), n_rep, x, logw))
        verbose && @printf("completed %-14s P=%s\n", model_key(model), fmt_g(P))
    end
    write_rows_csv(joinpath(output_dir, "student_df_results.csv"), robustness_rows)
    robustness_slopes = static_slopes(robustness_rows, slope_max_p)
    write_rows_csv(joinpath(output_dir, "student_df_slopes.csv"), robustness_slopes)

    metadata = jobj(
        "seed" => seed,
        "n_replications_per_cell" => n_rep,
        "gauss_hermite_nodes" => gh_nodes,
        "predictive_variance_grid" => collect(Float64, p_grid),
        "slope_max_predictive_variance" => slope_max_p,
        "methods" => collect(STATIC_METHODS),
        "theoretical_rmse_slopes" => STATIC_THEORETICAL_SLOPES,
        "main_models" => [model_key(m) for m in models],
        "student_robustness_models" => [model_key(m) for m in robustness_models],
        "max_implicit_equation_residual" => maximum(r.max_implicit_residual for r in diagnostic_rows),
        "min_curvature_denominator" => minimum(r.min_curvature_denominator for r in diagnostic_rows),
        "session" => session_info(),
        "runtime_seconds" => time() - start_time,
    )
    write_json(joinpath(output_dir, "metadata.json"), metadata)

    if verbose
        println("\nConvergence slopes (RMSE on P):")
        for s in slopes
            @printf("%-26s %-22s slope %.4f  se %.4f  R2 %.4f\n", s.model_label,
                    s.approximation, s.slope, s.slope_se, s.r_squared)
        end
    end
    return (results = rows, slopes = slopes, robustness = robustness_rows,
            robustness_slopes = robustness_slopes)
end
