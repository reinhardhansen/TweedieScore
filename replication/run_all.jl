# Run every experiment and refinement with the published settings, then build
# the tables, the quoted numbers, and Figure 1.
#
#   julia --project=replication -t auto replication/run_all.jl
#
# Output goes to replication/output, replication/tables and replication/figures.
# To compare the new output with the reference output of the paper, run
# replication/compare_with_reference.jl afterwards.

using TweedieScore

const ROOT = @__DIR__
out(name) = joinpath(ROOT, "output", name)

t0 = time()
println("== Static experiment, 80 nodes")
run_static_experiment(; gh_nodes = 80, output_dir = out("output_static_all_methods"))
println("\n== Static experiment, 120 nodes")
run_static_experiment(; gh_nodes = 120, output_dir = out("output_static_gh120_check"))
println("\n== Bayesian risk, 80 nodes and counts to 4096")
run_risk_experiment(; nodes = 80, max_count = 4096, output_dir = out("output_bayesian_risk"))
println("\n== Bayesian risk, 120 nodes and counts to 8192")
run_risk_experiment(; nodes = 120, max_count = 8192,
                    output_dir = out("output_bayesian_risk_refinement"))
println("\n== Recursive experiment")
run_recursive_experiment(; output_dir = out("output_recursive_mcisd"))
println("\n== Recursive grid and quadrature checks")
run_recursive_grid_checks(; output_dir = out("output_recursive_mcisd_checks"))

# The two build scripts run in fresh Julia processes with the same environment.
println("\n== Tables and quoted numbers")
run(`$(Base.julia_cmd()) --project=$(ROOT) $(joinpath(ROOT, "make_tables.jl"))`)
println("\n== Figure 1")
run(`$(Base.julia_cmd()) --project=$(ROOT) $(joinpath(ROOT, "make_figure.jl"))`)
println("\nDone in ", round(time() - t0; digits = 1), " seconds.")
