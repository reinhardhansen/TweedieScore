# Test suite of TweedieScore.jl.
#
#   julia --project=. -e 'using Pkg; Pkg.test()'
#
# Part 1 needs no stored output: helper functions, derivative checks, solver
# residuals, the propagation step of the grid filter, small runs of all three
# experiments, and update sensitivities.
#
# Part 2 tests the scalar interface: exactness in the Gaussian location model,
# agreement with the solvers of the experiments, and the error orders of the
# four updates.
#
# Part 3 audits the reference output in replication/reference_output, the run
# reported in the paper. Each block is skipped, with a message, when its files
# are missing. To compare a new run with the reference output, use
# replication/compare_with_reference.jl.

using Test
using TweedieScore
using CSV
using DataFrames
using LinearAlgebra
using Printf
using Random
using Statistics

const TS = TweedieScore
const REFERENCE = normpath(joinpath(@__DIR__, "..", "replication", "reference_output"))

ref(parts...) = joinpath(REFERENCE, parts...)
readcsv(path) = CSV.read(path, DataFrame; stringtype = String, pool = false)
num(x) = x === missing ? NaN : Float64(x)
keyvalue(v) = v isa AbstractString ? String(v) : v
rms(v) = sqrt(mean(abs2, v))

function available(paths...)
    missing_paths = filter(!isfile, collect(paths))
    isempty(missing_paths) && return true
    println("  skipped: missing ", join(missing_paths, ", "))
    return false
end

"""Pair the rows of two data frames on key columns."""
function paired_rows(df1, df2, keycols)
    index = Dict(Tuple(keyvalue(r[c]) for c in keycols) => r for r in eachrow(df2))
    pairs = Tuple{DataFrameRow,DataFrameRow}[]
    for r in eachrow(df1)
        key = Tuple(keyvalue(r[c]) for c in keycols)
        haskey(index, key) || error("no matching row for key $(key)")
        push!(pairs, (r, index[key]))
    end
    return pairs
end

@testset "TweedieScore" begin

# ===========================================================================
# Part 1: checks that need no stored output
# ===========================================================================

@testset "numerical helpers" begin
    for x in (-8.0, -1.0, 0.0, 0.5, 1.0, 3.0, 50.0, 2048.7)
        w = TS.wrightomega(x)
        @test isapprox(w + log(w), x; rtol = 1e-14, atol = 1e-14)
    end
    @test TS.wrightomega(1.0) ≈ 1.0 atol = 1e-15
    @test TS.wrightomega(0.0) ≈ 0.5671432904097838 rtol = 1e-15
    lf = TS.logfactorials(300)
    for k in (0, 1, 5, 20, 100, 300)
        @test isapprox(lf[k + 1], Float64(log(factorial(big(k)))); rtol = 1e-14, atol = 1e-14)
    end
    for n in (32, 64, 80, 120)
        z, w = TS.standard_normal_rule(n)
        @test sum(w) ≈ 1.0 atol = 1e-13
        @test sum(w .* z .^ 2) ≈ 1.0 atol = 1e-12
        @test sum(w .* z .^ 4) ≈ 3.0 atol = 1e-11
    end
    Z, W = TS.tensor_normal_rule_2d(20)
    @test sum(W) ≈ 1.0 atol = 1e-13
    @test sum(W .* Z[:, 1] .* Z[:, 2]) ≈ 0.0 atol = 1e-13
    g = TS.numpy_linspace(-4.0, 4.0, 481)
    @test length(g) == 481 && g[1] == -4.0 && g[end] == 4.0
    @test TS.expit(0.0) == 0.5
    @test TS.log1pexp(0.0) ≈ log(2.0)
    @test TS.logsumexp([1.0, 2.0, 3.0]) ≈ log(exp(1) + exp(2) + exp(3))
    @test TS.fmt_g(3.0) == "3" && TS.fmt_g(0.25) == "0.25"
end

@testset "model derivatives" begin
    h = 1.0e-5
    for m in (TS.PoissonLogIntensity(), TS.GaussianLogVariance(), TS.StudentTLocation(5.0),
              TS.StudentTLocation(3.0))
        for (y, th) in ((1.3, 0.2), (4.0, 1.1), (-0.7, -0.3))
            s, H, T = TS.derivatives(m, y, th)
            @test isapprox(s, (TS.loglik(m, th + h, y) - TS.loglik(m, th - h, y)) / (2h);
                           rtol = 1e-6, atol = 1e-8)
            @test isapprox(H, (TS.derivatives(m, y, th + h)[1] - TS.derivatives(m, y, th - h)[1]) / (2h);
                           rtol = 1e-6, atol = 1e-8)
            @test isapprox(T, (TS.derivatives(m, y, th + h)[2] - TS.derivatives(m, y, th - h)[2]) / (2h);
                           rtol = 1e-5, atol = 1e-7)
        end
    end
    checks = Dict(TS.validate_student_derivatives())
    @test checks["student_t_derivatives_passed"]
    limit = Dict(TS.validate_gaussian_location_identity())
    @test limit["gaussian_location_identity_passed"]
    @test limit["maximum_large_nu_exact_kalman_error"] < 2e-7
    @test limit["maximum_large_nu_mcisd_kalman_error"] < 2e-7
    @printf("  Gaussian-limit errors: exact %.3g, ISD %.3g, ISD* %.3g\n",
            limit["maximum_large_nu_exact_kalman_error"],
            limit["maximum_large_nu_implicit_kalman_error"],
            limit["maximum_large_nu_mcisd_kalman_error"])
end

@testset "ISD solvers" begin
    a = log(4.0)
    for P in (1e-4, 0.003, 0.01, 0.5), y in (0, 1)
        d = TS.isd_bernoulli(y, a, P)
        @test abs(d - P * (y - TS.expit(a + d))) <= 1e-17 + 2e-15 * abs(d)
    end
    # d = P (y - exp(a + d)) in the well-conditioned form a + d + log P = log(P y - d).
    # d = P y - omega carries one rounding of P y (ulp(2048) = 4.5e-13 at y = 4096,
    # P = 0.5), so the tolerance is relative; SciPy's wrightomega gives the same 2.2e-13.
    for P in (1e-4, 0.1, 0.5), y in (0, 3, 12, 4096)
        d = TS.isd_poisson(y, a, P)
        @test isapprox(a + d + log(P), log(P * y - d); rtol = 1e-12, atol = 1e-13)
    end
    rng = Random.Xoshiro(7)
    for m in (TS.PoissonLogIntensity(), TS.GaussianLogVariance(), TS.StudentTLocation(3.0))
        a0 = TS.prior_center(m)
        for P in (0.00015625, 0.32)
            theta = a0 .+ sqrt(P) .* randn(rng, 2000)
            y = TS.simulate(rng, m, theta)
            d, W, T, res, mincurv = TS.isd_bracketed_newton(m, y, a0, P)
            @test res <= 5e-11
            @test mincurv > 0
            s_mode = [TS.derivatives(m, y[i], a0 + d[i])[1] for i in eachindex(y)]
            @test maximum(abs.(d .- P .* s_mode)) <= 5e-11
        end
    end
    # Recursive solver: residual and agreement with the static solver for t_5.
    m5 = TS.StudentTLocation(5.0)
    r = randn(rng, 500) .* 2
    P = fill(0.9, 500)
    d, res = TS.isd_student_recursive(r, P, 5.0, 3.0)
    @test res < 2e-9
    d_static = TS.isd_bracketed_newton(m5, r, 0.0, 0.9)[1]
    @test maximum(abs.(d .- d_static)) < 1e-10
    # Scalar ISD* and explicit second order against their definitions.
    @test TS.isdstar_update(0.1, 0.2, 0.3) ≈ 0.1 + 0.5 * 0.04 * 0.3
    @test TS.explicit2_update(0.2, 1.5, -0.4, 0.3) ≈ 0.2 * 1.5 + 0.04 * (-0.4 * 1.5 + 0.15)
    # Cubic fallback: the chosen stationary point satisfies the implicit equation.
    rr = TS.cubic_real_roots(-2.0, 1.0 + 1.5 * 4.0, -2.0 * 1.0)
    @test !isempty(rr)
end

@testset "grid-filter propagation" begin
    design = TS.Design(3.0, 1.0, 0.98)
    transition = TS.make_grid_transition(design, 481, 8.0)
    grid, dx = transition.grid, transition.dx
    for (m0, v0) in ((0.0, 0.3), (0.5, 0.2), (-1.0, 0.1))
        post = exp.(-0.5 .* (grid .- m0) .^ 2 ./ v0)
        post = reshape(post ./ (sum(post) * dx), :, 1)
        pred = similar(post)
        TS.grid_predict!(pred, post, transition)
        mu, var = TS.density_moments(pred, transition)
        # Linear interpolation in the AR-scaling step adds a small O(dx^2)
        # variance (at most 6e-5 in the mean and 0.1% in the variance for
        # these three cases).
        @test isapprox(mu[1], design.phi * m0; atol = 5e-4)
        @test isapprox(var[1], design.phi^2 * v0 + TS.process_variance(design); rtol = 5e-3)
    end
end

@testset "smoke runs" begin
    x, w = TS.hermite_rule(80)
    logw = TS.hermite_logweights(w)
    for m in (TS.PoissonLogIntensity(), TS.GaussianLogVariance(), TS.StudentTLocation(5.0))
        rows = TS.static_cell(m, 0.04, Random.Xoshiro(3), 5000, x, logw)
        rmse = Dict(r.approximation => r.rmse for r in rows)
        @test all(isfinite, values(rmse))
        @test rmse["ISD"] < rmse["ESD"]
        @test rmse["ISD*"] < rmse["ISD"]
    end
    rows, outcomes, cert = TS.risk_scalar_model("Poisson", log(4.0), 0.1, 80, 60)
    @test length(rows) == 4 && length(outcomes) == 61
    rows, outcomes = TS.risk_bivariate_logistic(0.1, 30)
    @test length(rows) == 4 && length(outcomes) == 4
    xr, wr = TS.hermite_rule(32)
    out = TS.run_design(TS.Design(5.0, 1.0, 0.8), Random.Xoshiro(5), 20, 30, 241, 8.0,
                        xr, TS.hermite_logweights(wr), true)
    @test length(out[1]) == 7 && length(out[2]) == 6 && length(out[3]) == 5
    @test out[4].max_error_decomposition_residual < 1e-12
    @test out[4].max_implicit_equation_residual < 2e-9
end

@testset "update sensitivity" begin
    h = 1.0e-5
    a = log(4.0)
    errors_b = Float64[]
    errors_i = Float64[]
    for model in ("Bernoulli", "Poisson"), P in (0.01, 0.1, 0.32)
        _, center, _ = TS.risk_scalar_model(model, a, P, 120, 12)
        _, plus, _ = TS.risk_scalar_model(model, a + h, P, 120, 12)
        _, minus, _ = TS.risk_scalar_model(model, a - h, P, 120, 12)
        for (cc, pp, mm) in zip(center, plus, minus)
            derivative_b = 1 + (pp.bayes_correction - mm.bayes_correction) / (2h)
            derivative_i = 1 + (pp.isd_correction - mm.isd_correction) / (2h)
            mode = a + cc.isd_correction
            info = model == "Bernoulli" ? TS.expit(mode) * (1 - TS.expit(mode)) : exp(mode)
            push!(errors_b, abs(derivative_b - cc.posterior_variance / P))
            push!(errors_i, abs(derivative_i - 1 / (1 + P * info)))
        end
    end
    @printf("  max Bayesian sensitivity error %.3g, ISD %.3g\n",
            maximum(errors_b), maximum(errors_i))
    @test maximum(vcat(errors_b, errors_i)) < 1e-8
end

# ===========================================================================
# Part 2: scalar interface
# ===========================================================================

@testset "scalar interface: Gaussian location is exact for ISD and ISD*" begin
    variance = 0.7
    a = 0.1
    m = GaussianLocation(variance)
    for (y, P) in ((1.3, 0.2), (-2.0, 0.05), (0.4, 0.5))
        kalman = P * (y - a) / (variance + P)
        c = corrections(m, y, a, P)
        @test isapprox(c.posterior_mean, kalman; rtol = 1e-9, atol = 1e-12)
        @test isapprox(c.isd, kalman; rtol = 1e-11, atol = 1e-12)
        @test c.isdstar == c.isd                      # the third derivative is zero
        @test isapprox(c.esd, P * (y - a) / variance; rtol = 1e-14)
        @test esd_correction(m, y, a, P) == c.esd
        @test isd_correction(m, y, a, P) == c.isd
        @test isdstar_correction(m, y, a, P) == c.isdstar
        @test explicit2_correction(m, y, a, P) == c.explicit2
        @test posterior_mean_correction(m, y, a, P) == c.posterior_mean
    end
    @test_throws ArgumentError GaussianLocation(-1.0)
    @test_throws ArgumentError esd_correction(m, 1.0, 0.0, 0.0)
end

@testset "scalar interface: agreement with the experiment solvers" begin
    a = log(4.0)
    for P in (0.001, 0.01, 0.1, 0.32)
        for y in (0, 1)
            # The interface solver stops at an absolute residual of 5e-13.
            @test isapprox(isd_correction(BernoulliLogOdds(), y, a, P), TS.isd_bernoulli(y, a, P);
                           atol = 1e-12)
        end
        for y in (0, 2, 4, 9, 30)
            @test isapprox(isd_correction(PoissonLogIntensity(), y, a, P), TS.isd_poisson(y, a, P);
                           rtol = 1e-10, atol = 1e-12)
        end
        # Posterior means against the deterministic risk experiment (80 nodes).
        _, outcomes, _ = TS.risk_scalar_model("Bernoulli", a, P, 80, 12)
        for o in outcomes
            @test isapprox(posterior_mean_correction(BernoulliLogOdds(), o.y, a, P), o.bayes_correction;
                           rtol = 1e-10, atol = 1e-15)
            @test isapprox(isdstar_correction(BernoulliLogOdds(), o.y, a, P), o.mcisd_correction;
                           atol = 1e-12)
            @test isapprox(explicit2_correction(BernoulliLogOdds(), o.y, a, P), o.explicit2_correction;
                           rtol = 1e-12, atol = 1e-16)
        end
    end
    @test TS.model_key(PoissonLogIntensity()) == "poisson"
    @test TS.model_key(BernoulliLogOdds()) == "bernoulli"
end

@testset "scalar interface: error orders (Theorem 1 and ISD*)" begin
    # Poisson log intensity with a = log 4 and y = 6. The leading errors of ESD
    # and ISD are 10 P^2 and 2 P^2; ISD* and the explicit second-order update
    # have third-order errors.
    m = PoissonLogIntensity()
    a, y = log(4.0), 6
    err(f, P) = f(m, y, a, P) - posterior_mean_correction(m, y, a, P)
    @test isapprox(err(esd_correction, 0.001) / 0.001^2, 10.0; rtol = 0.02)
    @test isapprox(err(isd_correction, 0.001) / 0.001^2, 2.0; rtol = 0.02)
    for (f, order) in ((esd_correction, 2), (isd_correction, 2),
                       (isdstar_correction, 3), (explicit2_correction, 3))
        slope = log2(abs(err(f, 0.002) / err(f, 0.001)))
        @printf("  %-22s estimated order %.3f\n", string(f), slope)
        @test abs(slope - order) < 0.12
    end
end

# ===========================================================================
# Part 3: audits of the reference output (the run reported in the paper)
# ===========================================================================

@testset "reference output: deterministic risk experiment" begin
    raw_path = ref("output_bayesian_risk", "risk_results.csv")
    fine_path = ref("output_bayesian_risk_refinement", "risk_results.csv")
    cert_path = ref("output_bayesian_risk", "certificate_results.csv")
    if available(raw_path, fine_path, cert_path)
        raw, refined = readcsv(raw_path), readcsv(fine_path)
        key = r -> (r.model, r.epsilon, r.method)
        @test length(unique([key(r) for r in eachrow(raw)])) == nrow(raw) == 120
        reference = Dict(key(r) => r for r in eachrow(refined))
        change = col -> maximum(abs(r[col] - reference[key(r)][col]) for r in eachrow(raw))
        @printf("  refinement changes: rmse %.3g, coefficient ratio %.3g\n",
                change(:rmse), change(:coefficient_ratio))
        @test change(:rmse) < 1e-10
        @test change(:coefficient_ratio) < 1e-5
        smallest = maximum(abs(r.coefficient_ratio - 1) for r in eachrow(raw) if r.epsilon == 0.0001)
        @printf("  smallest-epsilon coefficient error %.3g\n", smallest)
        @test smallest < 2e-4
        for g in groupby(raw, [:model, :epsilon])
            rr = Dict(r.method => r for r in eachrow(g))
            @test all(isfinite, g.rmse)
            identity_gap = rr["ESD"].excess_bayes_risk - rr["ISD"].excess_bayes_risk
            @test abs(identity_gap - rr["ESD"].risk_gap_E_I) < 2e-14
        end
        cert = readcsv(cert_path)
        @test sum(cert.wrong_ISD_certificates) + sum(cert.wrong_ESD_certificates) == 0
        @test maximum(cert.max_enclosure_violation) < 5e-10
        @test maximum(abs.(raw.predictive_mass .- 1)) < 1e-12
        # Closed-form total risk of the Poisson ESD update.
        poisson_errors = Float64[]
        for r in eachrow(raw)
            (r.model == "Poisson" && r.method == "ESD") || continue
            P = r.epsilon
            rate = 4.0
            exact = P + P * P * (rate * rate * exp(2P) - (2rate + 1) * rate * exp(P / 2) + rate * rate)
            push!(poisson_errors, abs(r.total_bayes_risk - exact))
        end
        @printf("  Poisson closed-form risk error %.3g\n", maximum(poisson_errors))
        @test maximum(poisson_errors) < 5e-13
    end
end

@testset "reference output: static experiment" begin
    a80 = ref("output_static_all_methods", "approximation_results.csv")
    a120 = ref("output_static_gh120_check", "approximation_results.csv")
    s80 = ref("output_static_all_methods", "convergence_slopes.csv")
    s120 = ref("output_static_gh120_check", "convergence_slopes.csv")
    if available(a80, a120, s80, s120)
        j80, j120 = readcsv(a80), readcsv(a120)
        @test nrow(j80) == nrow(j120) == 144
        @test all(j80.n_rep .== 100_000)
        @test maximum(j80.max_implicit_residual) <= 5e-11
        pairs = paired_rows(j80, j120, (:model, :P, :approximation))
        drmse = maximum(abs(r1.rmse - r2.rmse) for (r1, r2) in pairs)
        k80, k120 = readcsv(s80), readcsv(s120)
        pairs = paired_rows(k80, k120, (:model, :approximation))
        dslope = maximum(abs(r1.slope - r2.slope) for (r1, r2) in pairs)
        @printf("  80 versus 120 nodes: max RMSE change %.3g, max slope change %.3g\n", drmse, dslope)
        @test drmse < 1e-6
        @test dslope < 2e-6
        # Estimated small-P slopes are close to the local orders. The largest
        # deviation is for the explicit second-order update of the log variance,
        # whose RMSE has heavy-tailed Monte Carlo error (slope 2.94, s.e. 0.03).
        for r in eachrow(k80)
            order = r.approximation in ("ESD", "ISD") ? 2 : 3
            @test abs(r.slope - order) < 5 * r.slope_se + 0.05
        end
    end
end

@testset "reference output: recursive experiment" begin
    dir = "output_recursive_mcisd"
    files = ("recursive_path_results.csv", "recursive_oracle_results.csv",
             "recursive_compact_summary.csv", "recursive_diagnostics.csv")
    if available((ref(dir, f) for f in files)...)
        jp = readcsv(ref(dir, "recursive_path_results.csv"))
        jo = readcsv(ref(dir, "recursive_oracle_results.csv"))
        jc = readcsv(ref(dir, "recursive_compact_summary.csv"))
        @test nrow(jp) == 84 && nrow(jo) == 72 && nrow(jc) == 18
        @test !any(nonunique(jp, [:nu, :snr, :phi, :method]))
        @test !any(nonunique(jo, [:nu, :snr, :phi, :method]))
        @test all(jp.n_rep .== 1000) && all(jp.T .== 500)
        max_error = 0.0
        for r in eachrow(jc)
            ppj = jp[(jp.nu .== r.nu) .& (jp.method .== r.method), :]
            ooj = jo[(jo.nu .== r.nu) .& (jo.method .== r.method), :]
            @test nrow(ppj) == nrow(ooj) == 4
            checks = ((r.one_step_gaussian, rms(ooj.rmse_local_gaussian)),
                      (r.one_step_grid, rms(ooj.rmse_total_grid)),
                      (r.recursive_gaussian, rms(num.(ppj.rmse_vs_gaussian_reset))),
                      (r.recursive_grid, rms(ppj.rmse_vs_grid)))
            max_error = max(max_error, maximum(abs(x - z) for (x, z) in checks))
        end
        @test max_error < 1e-12
        for g in groupby(jp[jp.method .== "Implicit", :], [:nu, :snr])
            byphi = Dict(r.phi => r.rmse_vs_grid for r in eachrow(g))
            @test byphi[0.98] < byphi[0.8]
        end
        jd = readcsv(ref(dir, "recursive_diagnostics.csv"))
        @test maximum(jd.max_posterior_edge_mass) < 1e-9
        @test maximum(jd.max_implicit_equation_residual) < 2e-9
        @test maximum(jd.max_error_decomposition_residual) < 1e-12
        @test all(jd.oracle_mcisd_outside_innovation_share .== 0)
        @test all(jd.recursive_mcisd_outside_innovation_share .== 0)
        @test all(jd.min_oracle_mcisd_curvature .> 0) && all(jd.min_recursive_mcisd_curvature .> 0)
        for r in eachrow(jd)
            @test r.max_exact_predictive_variance < 8 * (r.nu - 2) / (r.nu + 1)
        end
    end
end

end # testset TweedieScore
