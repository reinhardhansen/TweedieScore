# Compare a new run of the experiments with the reference output of the paper.
#
#   julia --project=replication replication/compare_with_reference.jl
#   julia --project=replication replication/compare_with_reference.jl --new-dir replication/output --reference-dir replication/reference_output
#
# Deterministic results (the risk experiment and the numerical checks) must
# agree to rounding. Monte Carlo results are reproduced exactly with the same
# Julia version and package versions as the reference run (Julia 1.13.2 and
# replication/Manifest.toml);
# otherwise they are compared through their Monte Carlo standard errors
# (|z| <= 5), since Julia does not promise identical random streams across
# versions. The reference copy of scalar_outcomes.csv covers the 80-node run
# only, so the comparison of the refinement run skips that file.

using Test
using TweedieScore
using CSV
using DataFrames
using Printf
using Statistics

opts = parse_cli(ARGS, Dict{String,Any}(
    "new-dir" => joinpath(@__DIR__, "output"),
    "reference-dir" => joinpath(@__DIR__, "reference_output"),
))
const NEW_ROOT = abspath(opts["new-dir"])
const REF_ROOT = abspath(opts["reference-dir"])

newp(parts...) = joinpath(NEW_ROOT, parts...)
refp(parts...) = joinpath(REF_ROOT, parts...)
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

"""Pair the rows of the new and the reference output on key columns."""
function paired_rows(jdf, pdf, keycols)
    index = Dict(Tuple(keyvalue(r[c]) for c in keycols) => r for r in eachrow(pdf))
    pairs = Tuple{DataFrameRow,DataFrameRow}[]
    for r in eachrow(jdf)
        key = Tuple(keyvalue(r[c]) for c in keycols)
        haskey(index, key) || error("no reference row for key $(key)")
        push!(pairs, (r, index[key]))
    end
    return pairs
end

"""
Worst ratio |new - reference| / (atol + rtol |reference|) over the pairs; values at
most one pass. Rows where both values are missing or NaN are skipped.
"""
function worst_ratio(pairs, col; rtol = 0.0, atol = 0.0)
    worst = 0.0
    for (rj, rp) in pairs
        a = num(rj[col])
        b = num(rp[col])
        (isnan(a) && isnan(b)) && continue
        ratio = abs(a - b) / (atol + rtol * abs(b))
        worst = max(worst, isnan(ratio) ? Inf : ratio)
    end
    return worst
end

function check_columns(pairs, specs, name)
    for (col, rtol, atol) in specs
        worst = worst_ratio(pairs, col; rtol = rtol, atol = atol)
        @printf("  %-34s %-30s worst |diff|/tol = %.3g\n", name, col, worst)
        @test worst <= 1.0
    end
end

"""Largest |z| for Monte Carlo quantities with standard-error columns."""
function max_abs_z(pairs, col, secol)
    worst = 0.0
    for (rj, rp) in pairs
        a, b = num(rj[col]), num(rp[col])
        se = sqrt(num(rj[secol])^2 + num(rp[secol])^2)
        (isnan(a) || isnan(b) || se == 0.0) && continue
        worst = max(worst, abs(a - b) / se)
    end
    return worst
end


@testset "new output against the reference output" begin


@testset "Bayesian risk: new vs reference ($(dir))" for dir in
        ("output_bayesian_risk", "output_bayesian_risk_refinement")
    files = ("risk_results.csv", "certificate_results.csv", "bivariate_outcomes.csv")
    if available((newp(dir, f) for f in files)..., (refp(dir, f) for f in files)...)
        jr, pr = readcsv(newp(dir, "risk_results.csv")), readcsv(refp(dir, "risk_results.csv"))
        @test nrow(jr) == nrow(pr) == 120
        pairs = paired_rows(jr, pr, (:model, :epsilon, :method))
        check_columns(pairs, (
            (:rmse, 1e-7, 1e-16),
            (:excess_bayes_risk, 2e-7, 1e-28),
            (:total_bayes_risk, 1e-11, 1e-16),
            (:risk_gap_E_I, 1e-6, 1e-26),
            (:normalized_gap, 1e-6, 0.0),
            (:theoretical_coefficient, 1e-12, 0.0),
            (:coefficient_ratio, 1e-6, 0.0),
            (:predictive_mass, 0.0, 1e-13),
        ), "$dir/risk_results")

        jc, pc = readcsv(newp(dir, "certificate_results.csv")), readcsv(refp(dir, "certificate_results.csv"))
        pairs = paired_rows(jc, pc, (:model, :epsilon))
        check_columns(pairs, (
            (:certified_ISD_probability, 0.0, 1e-12),
            (:certified_ESD_probability, 0.0, 1e-12),
            (:predictive_mass, 0.0, 1e-13),
            (:max_enclosure_violation, 1e-6, 1e-12),
            (:weighted_enclosure_width, 1e-9, 1e-20),
            (:weighted_mean_mode_gap, 1e-8, 1e-18),
        ), "$dir/certificate_results")
        @test all(jc.wrong_ISD_certificates .== 0) && all(jc.wrong_ESD_certificates .== 0)

        jb, pb = readcsv(newp(dir, "bivariate_outcomes.csv")), readcsv(refp(dir, "bivariate_outcomes.csv"))
        pairs = paired_rows(jb, pb, (:epsilon, :y1, :y2))
        check_columns(pairs, (
            (:predictive_probability, 0.0, 1e-14),
            (:bayes_correction_1, 1e-10, 1e-16),
            (:bayes_correction_2, 1e-10, 1e-16),
            (:isd_correction_1, 1e-12, 1e-18),
            (:isd_correction_2, 1e-12, 1e-18),
            (:posterior_variance_trace, 1e-10, 1e-16),
            (:risk_gap_E_I, 1e-6, 1e-24),
        ), "$dir/bivariate_outcomes")
        @test maximum(jb.implicit_residual) < 1e-12

    end
    # scalar_outcomes.csv is large; the reference copy covers the 80-node run only.
    if available(newp(dir, "scalar_outcomes.csv"), refp(dir, "scalar_outcomes.csv"))
        js, ps = readcsv(newp(dir, "scalar_outcomes.csv")), readcsv(refp(dir, "scalar_outcomes.csv"))
        @test nrow(js) == nrow(ps)
        pairs = paired_rows(js, ps, (:model, :epsilon, :y))
        check_columns(pairs, (
            (:predictive_probability, 0.0, 1e-14),
            (:bayes_correction, 1e-10, 1e-15),
            (:esd_correction, 1e-14, 1e-18),
            # Poisson d_I = P y - wrightomega(.) is exactly 0 at y = 4 = e^a, where the files
            # store rounding noise of a few ulp of P y; hence the 1e-15 floors on
            # isd_correction, mean_upper (= d_I) and the enclosure bound risk_gap_lower built from it.
            (:isd_correction, 1e-12, 1e-15),
            (:mcisd_correction, 1e-12, 1e-17),
            (:explicit2_correction, 1e-14, 1e-18),
            (:posterior_variance, 1e-10, 1e-16),
            (:mean_lower, 1e-12, 1e-17),
            (:mean_upper, 1e-12, 1e-15),
            (:risk_gap_lower, 1e-9, 1e-15),
            (:risk_gap_upper, 1e-9, 1e-20),
        ), "$dir/scalar_outcomes")
        # The risk gap is a difference of two squared errors; its tolerance is
        # relative to the larger of the two squares.
        worst = 0.0
        flag_mass = 0.0
        for (rj, rp) in pairs
            db, de, di = rp.bayes_correction, rp.esd_correction, rp.isd_correction
            scale = max((db - de)^2, (db - di)^2)
            worst = max(worst, abs(rj.risk_gap_E_I - rp.risk_gap_E_I) / (1e-6 * scale + 1e-30))
            if rj.certified_ISD != rp.certified_ISD || rj.certified_ESD != rp.certified_ESD
                flag_mass += rp.predictive_probability
            end
        end
        @printf("  %-34s %-30s worst |diff|/tol = %.3g\n", "$dir/scalar_outcomes", "risk_gap_E_I", worst)
        @printf("  %-34s predictive mass of outcomes whose certificate differs: %.3g\n",
                "$dir/scalar_outcomes", flag_mass)
        @test worst <= 1.0
        @test flag_mass < 1e-10
    end
end

@testset "Bayesian risk audit of the new output" begin
    raw_path = newp("output_bayesian_risk", "risk_results.csv")
    fine_path = newp("output_bayesian_risk_refinement", "risk_results.csv")
    cert_path = newp("output_bayesian_risk", "certificate_results.csv")
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


@testset "Static experiment: new vs reference ($(dir))" for dir in
        ("output_static_all_methods", "output_static_gh120_check")
    files = ("approximation_results.csv", "student_df_results.csv", "convergence_slopes.csv",
             "student_df_slopes.csv", "solver_diagnostics.csv")
    if available((newp(dir, f) for f in files)..., (refp(dir, f) for f in files)...)
        for f in ("approximation_results.csv", "student_df_results.csv")
            jdf, pdf = readcsv(newp(dir, f)), readcsv(refp(dir, f))
            @test nrow(jdf) == nrow(pdf) == 144
            @test all(jdf.n_rep .== 100_000)
            pairs = paired_rows(jdf, pdf, (:model, :P, :approximation))
            z = max_abs_z(pairs, :rmse, :rmse_mcse)
            @printf("  %-34s %-26s max |z| of RMSE = %.2f\n", dir, f, z)
            @test z <= 5.0
            @test maximum(jdf.max_implicit_residual) <= 5e-11
            # Curvature bound: 1 - P H >= 1 - P (nu + 1)/(8(nu - 2)) for Student-t, >= 1 otherwise.
            for r in eachrow(jdf)
                if startswith(r.model, "student_t")
                    nu = r.model == "student_t" ? 5.0 : parse(Float64, split(r.model, "_")[end])
                    @test r.min_curvature_denominator >= 1 - r.P * (nu + 1) / (8 * (nu - 2)) - 1e-12
                else
                    @test r.min_curvature_denominator >= 1 - 1e-12
                end
            end
        end
        for f in ("convergence_slopes.csv", "student_df_slopes.csv")
            jdf, pdf = readcsv(newp(dir, f)), readcsv(refp(dir, f))
            pairs = paired_rows(jdf, pdf, (:model, :approximation))
            worst = 0.0
            for (rj, rp) in pairs
                se = sqrt(rj.slope_se^2 + rp.slope_se^2)
                worst = max(worst, abs(rj.slope - rp.slope) / (5 * se + 0.002))
                @printf("    %-14s %-22s slope new %.3f  reference %.3f\n", rj.model,
                        rj.approximation, rj.slope, rp.slope)
            end
            @test worst <= 1.0
        end
    end
end

@testset "Static experiment: 80 versus 120 nodes (new run)" begin
    a80 = newp("output_static_all_methods", "approximation_results.csv")
    a120 = newp("output_static_gh120_check", "approximation_results.csv")
    s80 = newp("output_static_all_methods", "convergence_slopes.csv")
    s120 = newp("output_static_gh120_check", "convergence_slopes.csv")
    if available(a80, a120, s80, s120)
        j80, j120 = readcsv(a80), readcsv(a120)
        pairs = paired_rows(j80, j120, (:model, :P, :approximation))
        drmse = maximum(abs(rj.rmse - rp.rmse) for (rj, rp) in pairs)
        k80, k120 = readcsv(s80), readcsv(s120)
        pairs = paired_rows(k80, k120, (:model, :approximation))
        dslope = maximum(abs(rj.slope - rp.slope) for (rj, rp) in pairs)
        @printf("  max RMSE change %.3g, max slope change %.3g\n",
                drmse, dslope)
        # The largest change (about 1.5e-7) is at Poisson ESD, P = 0.32 (RMSE 0.79).
        @test drmse < 1e-6
        @test dslope < 2e-6
    end
end


@testset "Recursive experiment: new vs reference" begin
    dir = "output_recursive_mcisd"
    files = ("recursive_path_results.csv", "recursive_oracle_results.csv",
             "mcisd_vs_implicit_results.csv", "recursive_compact_summary.csv",
             "recursive_diagnostics.csv", "mcisd_vs_implicit_pooled.csv")
    if available((newp(dir, f) for f in files)..., (refp(dir, f) for f in files)...)
        jp, pp = readcsv(newp(dir, "recursive_path_results.csv")), readcsv(refp(dir, "recursive_path_results.csv"))
        jo, po = readcsv(newp(dir, "recursive_oracle_results.csv")), readcsv(refp(dir, "recursive_oracle_results.csv"))
        jc = readcsv(newp(dir, "recursive_compact_summary.csv"))

        # Consistency of the new output.
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
            max_error = max(max_error, maximum(abs(a - b) for (a, b) in checks))
        end
        @test max_error < 1e-12
        for g in groupby(jp[jp.method .== "Implicit", :], [:nu, :snr])
            byphi = Dict(r.phi => r.rmse_vs_grid for r in eachrow(g))
            @test byphi[0.98] < byphi[0.8]
        end

        # Monte Carlo agreement, design by design.
        pairs = paired_rows(jp, pp, (:nu, :snr, :phi, :method))
        for (col, se) in ((:rmse_vs_grid, :rmse_vs_grid_mcse),
                          (:rmse_vs_gaussian_reset, :rmse_vs_gaussian_reset_mcse),
                          (:rmse_vs_state, :rmse_vs_state_mcse))
            z = max_abs_z(pairs, col, se)
            @printf("  path results   %-28s max |z| = %.2f\n", col, z)
            @test z <= 5.0
        end
        pairs = paired_rows(jo, po, (:nu, :snr, :phi, :method))
        for (col, se) in ((:rmse_local_gaussian, :rmse_local_gaussian_mcse),
                          (:rmse_total_grid, :rmse_total_grid_mcse))
            z = max_abs_z(pairs, col, se)
            @printf("  oracle results %-28s max |z| = %.2f\n", col, z)
            @test z <= 5.0
        end
        jm, pm = readcsv(newp(dir, "mcisd_vs_implicit_results.csv")), readcsv(refp(dir, "mcisd_vs_implicit_results.csv"))
        pairs = paired_rows(jm, pm, (:nu, :snr, :phi, :target))
        z = max_abs_z(pairs, :paired_mse_difference, :paired_mse_difference_mcse)
        @printf("  paired ISD*-ISD MSE difference max |z| = %.2f\n", z)
        @test z <= 5.0

        # Pooled numbers quoted in Section 7.3, with a pooled Monte Carlo z-score.
        # Pooled RMSE sqrt(mean(r_d^2)) and its delta-method standard error.
        pooled = (df, method, col, secol) -> begin
            sel = df[df.method .== method, :]
            v = num.(sel[!, col])
            s = num.(sel[!, secol])
            value = rms(v)
            (value, sqrt(sum((v .* s) .^ 2)) / (length(v) * value))
        end
        println("  pooled over the 12 designs            new        reference")
        for (method, col, secol) in (("Implicit", :rmse_vs_grid, :rmse_vs_grid_mcse),
                                     ("ISD*", :rmse_vs_grid, :rmse_vs_grid_mcse),
                                     ("Exact Gaussian", :rmse_vs_grid, :rmse_vs_grid_mcse),
                                     ("First order", :rmse_vs_grid, :rmse_vs_grid_mcse),
                                     ("Implicit", :rmse_vs_state, :rmse_vs_state_mcse),
                                     ("ISD*", :rmse_vs_state, :rmse_vs_state_mcse))
            vj, sj = pooled(jp, method, col, secol)
            vp, sp = pooled(pp, method, col, secol)
            @printf("  %-15s %-20s %.6f   %.6f\n", method, col, vj, vp)
            @test abs(vj - vp) <= 5 * sqrt(sj^2 + sp^2)
        end
        jpool = readcsv(newp(dir, "mcisd_vs_implicit_pooled.csv"))
        ppool = readcsv(refp(dir, "mcisd_vs_implicit_pooled.csv"))
        rj = only(eachrow(jpool[(jpool.target .== "latent_state") .& (string.(jpool.nu) .== "All"), :]))
        rp = only(eachrow(ppool[(ppool.target .== "latent_state") .& (string.(ppool.nu) .== "All"), :]))
        @printf("  latent-state paired MSE difference: new %.5f [%.5f, %.5f], reference %.5f [%.5f, %.5f]\n",
                rj.paired_mse_difference, rj.paired_mse_difference_ci_low, rj.paired_mse_difference_ci_high,
                rp.paired_mse_difference, rp.paired_mse_difference_ci_low, rp.paired_mse_difference_ci_high)
        @test abs(rj.paired_mse_difference - rp.paired_mse_difference) <=
              5 * sqrt(rj.paired_mse_difference_mcse^2 + rp.paired_mse_difference_mcse^2)

        # Deterministic design quantities and numerical diagnostics.
        jd, pd = readcsv(newp(dir, "recursive_diagnostics.csv")), readcsv(refp(dir, "recursive_diagnostics.csv"))
        pairs = paired_rows(jd, pd, (:nu, :snr, :phi))
        check_columns(pairs, ((:process_variance, 1e-12, 0.0), (:observation_scale, 1e-12, 0.0),
                              (:grid_dx, 1e-12, 0.0)), "$dir/recursive_diagnostics")
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

@testset "Recursive grid and quadrature checks: new vs reference" begin
    dir = "output_recursive_mcisd_checks"
    files = ("grid_refinement_diagnostics.csv", "quadrature_refinement_diagnostics.csv")
    if available((newp(dir, f) for f in files)..., (refp(dir, f) for f in files)...)
        jg, pg = readcsv(newp(dir, files[1])), readcsv(refp(dir, files[1]))
        pairs = paired_rows(jg, pg, (:nu, :snr, :phi, :coarse_grid_points, :fine_grid_points))
        for (rj, rp) in pairs
            @printf("  %s->%s nu=%g: mean-difference RMSE new %.3e, reference %.3e\n",
                    rj.coarse_grid_points, rj.fine_grid_points, rj.nu,
                    rj.exact_grid_mean_difference_rmse, rp.exact_grid_mean_difference_rmse)
            @test abs(rj.exact_grid_mean_difference_rmse / rp.exact_grid_mean_difference_rmse - 1) < 0.3
            @test abs(log(rj.maximum_relative_path_metric_difference /
                          rp.maximum_relative_path_metric_difference)) < log(2.0)
            @test rj.fine_maximum_edge_mass < 1e-9
        end
        jq = readcsv(newp(dir, files[2]))
        @test maximum(jq.maximum_absolute_path_metric_difference) < 1e-5
        @test maximum(jq.maximum_absolute_oracle_metric_difference) < 1e-5
    end
end

end # testset
