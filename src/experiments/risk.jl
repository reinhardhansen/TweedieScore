# Section 7.2: conditional and average Bayesian risk (deterministic).
# Port of bayesian_risk_experiment.py.
#
# No random numbers are used. Every Bernoulli outcome, every Poisson count
# from 0 to `max_count`, and all four bivariate logistic outcome pairs are
# enumerated, and Gaussian integrals are evaluated by Gauss-Hermite quadrature.

const RISK_METHODS = ("ESD", "ISD", "ISD*", "Explicit 2nd")
const RISK_EPSILON_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.16, 0.32, 0.5]

"""
    risk_scalar_model(model, a, P, nodes, max_count) -> (rows, outcomes, certificate)

Bernoulli log odds (`model == "Bernoulli"`) or Poisson log intensity
(`model == "Poisson"`) with prior N(a, P). Returns the four method rows of
risk_results.csv, the per-outcome rows of scalar_outcomes.csv and the row of
certificate_results.csv (`scalar` in the Python code).
"""
function risk_scalar_model(model::AbstractString, a::Real, P::Real, nodes::Integer,
                           max_count::Integer)
    z, w = standard_normal_rule(nodes)
    logw = log.(w)
    ys = model == "Bernoulli" ? collect(0:1) : collect(0:max_count)
    m = length(ys)
    s = zeros(m); H = zeros(m); T = zeros(m)
    di = zeros(m); Wi = zeros(m); Ti = zeros(m)
    db = zeros(m); variance = zeros(m); logmass = zeros(m)
    low = zeros(m); high = zeros(m)
    lw = zeros(nodes)

    if model == "Bernoulli"
        p = expit(a)
        info = p * (1 - p)
        x = sqrt(P) .* z
        for (k, y) in enumerate(ys)
            s[k], H[k], T[k] = bernoulli_logodds_derivatives(y, a)
            di[k] = isd_bernoulli(y, a, P)
            pm = expit(a + di[k])
            Wi[k] = 1 / (1 / P + pm * (1 - pm))
            Ti[k] = -pm * (1 - pm) * (1 - 2 * pm)
            for j in 1:nodes
                lw[j] = logw[j] + (y * (a + x[j]) - log1pexp(a + x[j]))
            end
            logmass[k] = logsumexp(lw)
            ww = exp.(lw .- logmass[k])
            db[k] = dot(ww, x)
            second = dot(ww, x .* x)
            variance[k] = second - db[k] * db[k]
            B = Wi[k] * P / (12 * sqrt(3))
            low[k] = di[k] - B
            high[k] = di[k] + B
        end
    elseif model == "Poisson"
        rate = exp(a)
        info = rate
        logfact = logfactorials(max_count)
        for (k, y) in enumerate(ys)
            s[k], H[k], T[k] = poisson_logintensity_derivatives(y, a)
            di[k] = isd_poisson(y, a, P)
            mi = a + di[k]
            imode = exp(mi)
            Wi[k] = 1 / (1 / P + imode)
            Ti[k] = -imode
            # Mode-centered quadrature is stable even for very rare, large counts.
            sqW = sqrt(Wi[k])
            qm = y * mi - imode - di[k] * di[k] / (2 * P)
            xk = di[k] .+ sqW .* z
            for j in 1:nodes
                theta = a + xk[j]
                q = y * theta - exp(theta) - xk[j] * xk[j] / (2 * P)
                lw[j] = logw[j] + q - qm + z[j] * z[j] / 2
            end
            norm_k = logsumexp(lw)
            ww = exp.(lw .- norm_k)
            db[k] = sum(ww .* xk)
            variance[k] = sum(ww .* (xk .- db[k]) .^ 2)
            logmass[k] = 0.5 * log(Wi[k] / P) + qm - logfact[y + 1] + norm_k
            low[k] = P * y - wrightomega(a + P * y + log(P) + P / 2)
            high[k] = di[k]
        end
    else
        error("unknown scalar risk model $model")
    end

    mass = exp.(logmass)
    de = [esd_update(P, s[k]) for k in 1:m]
    dmc = [isdstar_update(di[k], Wi[k], Ti[k]) for k in 1:m]
    d2 = [explicit2_update(P, s[k], H[k], T[k]) for k in 1:m]
    estimates = (de, di, dmc, d2)
    gap = (db .- de) .^ 2 .- (db .- di) .^ 2
    sep = di .- de
    c1 = sep .* (2 .* low .- di .- de)
    c2 = sep .* (2 .* high .- di .- de)
    lower = min.(c1, c2)
    upper = max.(c1, c2)
    safety = 1.0e-14 .+ 1.0e-10 .* max.(abs.(lower), abs.(upper))
    certify_i = lower .> safety
    certify_e = upper .< -safety
    coef = info^3

    rows = []
    for (j, method) in enumerate(RISK_METHODS)
        err = estimates[j] .- db
        mse = dot(mass, err .^ 2)
        push!(rows, (
            model = String(model),
            epsilon = Float64(P),
            method = method,
            rmse = sqrt(mse),
            excess_bayes_risk = mse,
            total_bayes_risk = dot(mass, variance) + mse,
            risk_gap_E_I = dot(mass, gap),
            normalized_gap = dot(mass, gap) / P^4,
            theoretical_coefficient = coef,
            coefficient_ratio = dot(mass, gap) / P^4 / coef,
            predictive_mass = sum(mass),
        ))
    end

    outcomes = []
    for k in 1:m
        push!(outcomes, (
            model = String(model),
            epsilon = Float64(P),
            y = Int(ys[k]),
            predictive_probability = mass[k],
            bayes_correction = db[k],
            esd_correction = de[k],
            isd_correction = di[k],
            mcisd_correction = dmc[k],
            explicit2_correction = d2[k],
            posterior_variance = variance[k],
            mean_lower = low[k],
            mean_upper = high[k],
            risk_gap_E_I = gap[k],
            risk_gap_lower = lower[k],
            risk_gap_upper = upper[k],
            certified_ISD = Int(certify_i[k]),
            certified_ESD = Int(certify_e[k]),
        ))
    end

    certificate = (
        model = String(model),
        epsilon = Float64(P),
        certified_ISD_probability = dot(mass, Float64.(certify_i)),
        certified_ESD_probability = dot(mass, Float64.(certify_e)),
        predictive_mass = sum(mass),
        wrong_ISD_certificates = count(certify_i .& (gap .< -safety)),
        wrong_ESD_certificates = count(certify_e .& (gap .> safety)),
        max_enclosure_violation = maximum(max.(low .- db, db .- high)),
        weighted_enclosure_width = dot(mass, high .- low),
        weighted_mean_mode_gap = dot(mass, abs.(db .- di)),
    )
    # The bounds are mathematical guarantees; the checks allow floating-point error.
    certificate.max_enclosure_violation < 5.0e-10 ||
        error("posterior-mean enclosure violated: $(certificate)")
    certificate.wrong_ISD_certificates == certificate.wrong_ESD_certificates == 0 ||
        error("a declared risk comparison is wrong: $(certificate)")
    return rows, outcomes, certificate
end

"""
    risk_bivariate_logistic(epsilon, nodes) -> (rows, outcomes)

Bivariate logistic design of Appendix F.4 with prior N(a, epsilon Sigma)
(`multivariate` in the Python code).
"""
function risk_bivariate_logistic(epsilon::Real, nodes::Integer)
    X = BIVARIATE_X
    a = BIVARIATE_A
    Sigma = BIVARIATE_SIGMA
    P = epsilon .* Sigma
    Pinv = inv(P)
    Z, w = tensor_normal_rule_2d(nodes)
    L = cholesky(Symmetric(P)).L
    xs = Z * transpose(Matrix(L))            # row r is L z_r
    theta = xs .+ transpose(a)
    eta = theta * transpose(X)               # row r is (X theta_r)'
    softplus_sum = vec(sum(log1pexp.(eta); dims = 2))
    logw = log.(w)
    p0 = expit.(X * a)
    J = X' * Diagonal(p0 .* (1 .- p0)) * X
    A = Sigma * J * Sigma
    coefficient = tr(A * J * A)

    outcomes = []
    errs = Vector{Vector{Float64}}()
    masses = Float64[]
    max_residual = 0.0
    for (y1, y2) in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
        y = [y1, y2]
        lw = logw .+ (eta * y .- softplus_sum)
        logmass = logsumexp(lw)
        ww = exp.(lw .- logmass)
        db = transpose(xs) * ww
        s = X' * (y .- p0)
        de = P * s
        di = isd_logistic(y, a, X, P, Pinv)
        pm = expit.(X * (a .+ di))
        Wi = inv(Pinv .+ X' * Diagonal(pm .* (1 .- pm)) * X)
        dmc = di .+ (0.5 .* Wi) * logistic_third_contraction(X, a .+ di, Wi)
        d2 = de .- (P * J) * de .+ (0.5 .* P) * logistic_third_contraction(X, a, P)
        ds = (de, di, dmc, d2)
        push!(errs, [sum((ds[j] .- db) .^ 2) for j in 1:4])
        push!(masses, exp(logmass))
        residual = norm(di .- (P * X') * (y .- pm))
        max_residual = max(max_residual, residual)
        centered = xs .- transpose(db)
        covariance = transpose(centered) * (centered .* ww)
        push!(outcomes, (
            epsilon = Float64(epsilon),
            y1 = Int(y1),
            y2 = Int(y2),
            predictive_probability = exp(logmass),
            bayes_correction_1 = db[1],
            bayes_correction_2 = db[2],
            isd_correction_1 = di[1],
            isd_correction_2 = di[2],
            posterior_variance_trace = tr(covariance),
            risk_gap_E_I = errs[end][1] - errs[end][2],
            implicit_residual = residual,
        ))
    end
    mse = [sum(masses[k] * errs[k][j] for k in eachindex(masses)) for j in 1:4]
    gap = mse[1] - mse[2]
    trace_risk = 0.0
    for r in outcomes
        trace_risk += r.predictive_probability * r.posterior_variance_trace
    end
    rows = []
    for (j, method) in enumerate(RISK_METHODS)
        push!(rows, (
            model = "Bivariate logistic",
            epsilon = Float64(epsilon),
            method = method,
            rmse = sqrt(mse[j]),
            excess_bayes_risk = mse[j],
            total_bayes_risk = trace_risk + mse[j],
            risk_gap_E_I = gap,
            normalized_gap = gap / epsilon^4,
            theoretical_coefficient = coefficient,
            coefficient_ratio = gap / epsilon^4 / coefficient,
            predictive_mass = sum(masses),
        ))
    end
    max_residual < 1.0e-12 || error("bivariate ISD residual $(max_residual) exceeds 1e-12")
    return rows, outcomes
end

"""
    run_risk_experiment(; nodes = 80, max_count = 4096, output_dir)

Write risk_results.csv, scalar_outcomes.csv, certificate_results.csv,
bivariate_outcomes.csv and metadata.json. The refinement run of the paper
uses nodes = 120 and max_count = 8192.
"""
function run_risk_experiment(; nodes::Integer = 80, max_count::Integer = 4096,
                             epsilon_grid::AbstractVector = RISK_EPSILON_GRID,
                             output_dir::AbstractString = "output_bayesian_risk",
                             verbose::Bool = true)
    start_time = time()
    mkpath(output_dir)
    results = []
    outcomes = []
    certificates = []
    bivariate = []
    for model in ("Bernoulli", "Poisson"), P in epsilon_grid
        rr, oo, cc = risk_scalar_model(model, log(4.0), P, nodes, max_count)
        append!(results, rr)
        append!(outcomes, oo)
        push!(certificates, cc)
    end
    for P in epsilon_grid
        rr, oo = risk_bivariate_logistic(P, nodes)
        append!(results, rr)
        append!(bivariate, oo)
    end
    write_rows_csv(joinpath(output_dir, "risk_results.csv"), results)
    write_rows_csv(joinpath(output_dir, "scalar_outcomes.csv"), outcomes)
    write_rows_csv(joinpath(output_dir, "certificate_results.csv"), certificates)
    write_rows_csv(joinpath(output_dir, "bivariate_outcomes.csv"), bivariate)
    metadata = jobj(
        "nodes" => nodes,
        "max_count" => max_count,
        "covariance_grid" => collect(Float64, epsilon_grid),
        "integration" => "Gauss-Hermite; mode-centered for Poisson",
        "outcomes" => "all Bernoulli outcomes; Poisson 0 through max_count",
        "max_predictive_mass_error" => maximum(abs(r.predictive_mass - 1) for r in results),
        "bivariate_a" => collect(BIVARIATE_A),
        "bivariate_X" => [collect(BIVARIATE_X[i, :]) for i in 1:2],
        "bivariate_Sigma" => [collect(BIVARIATE_SIGMA[i, :]) for i in 1:2],
        "session" => session_info(),
        "runtime_seconds" => time() - start_time,
    )
    write_json(joinpath(output_dir, "metadata.json"), metadata)
    verbose && println("Bayesian risk experiment written to $(output_dir)")
    return (results = results, outcomes = outcomes, certificates = certificates,
            bivariate = bivariate)
end
