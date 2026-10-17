# Replication

This folder reproduces every table, figure, and quoted number in Section 7 of

> Hansen, P. R. and Tong, C. (2026). Tweedie's Formula and Score-Driven Updating.

and in Appendix F of its Supporting Information.

## Setup

The commands below are run from the root of the repository. Julia 1.11 or later
is needed for this environment, because `Project.toml` points to the package in
the parent folder through a `[sources]` entry. Install the dependencies once:

```
julia --project=replication -e 'using Pkg; Pkg.instantiate()'
```

On Julia 1.10, run `julia --project=replication -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'` instead.
The first installation takes a few minutes, most of it for CairoMakie, which is
used only for the figure.

## Running everything

```
julia --project=replication -t auto replication/run_all.jl
```

This runs the three experiments and their refinements, then builds the tables,
the list of quoted numbers, and Figure 1. Output goes to

* `replication/output/`: CSV and JSON files, one folder per run;
* `replication/tables/`: one `.tex` file per table, and `numbers.md`, which lists every number
  that the text quotes from the experiments;
* `replication/figures/`: `approximation_error.pdf` and `.png` (Figure 1).

On an Apple-silicon Mac with 10 threads and Julia 1.13.2, the reference run took
about 13 seconds for each static run, under a second for each risk run, and about
90 seconds for the recursive experiment, plus package compilation. The `-t auto`
flag lets the grid filter use all cores; the results do not depend on the number
of threads.

The individual steps are

```
julia --project=replication replication/run_static.jl
julia --project=replication replication/run_static.jl --gh-nodes 120 --output-dir replication/output/output_static_gh120_check
julia --project=replication replication/run_risk.jl
julia --project=replication replication/run_risk.jl --nodes 120 --max-count 8192 --output-dir replication/output/output_bayesian_risk_refinement
julia --project=replication -t auto replication/run_recursive.jl
julia --project=replication -t auto replication/run_recursive.jl --grid-check-only
julia --project=replication replication/make_tables.jl
julia --project=replication replication/make_figure.jl
```

The options of each script are listed at its top.

## Reference output

`reference_output/` holds the output of the run reported in the paper (Julia
1.13.2, macOS on Apple silicon, with the package versions in `Manifest.toml`). An
earlier run with Julia 1.13.1 gave identical CSV files. The tables and the figure can be rebuilt from
it without running the experiments:

```
julia --project=replication replication/make_tables.jl --input-dir replication/reference_output
julia --project=replication replication/make_figure.jl --input-dir replication/reference_output/output_static_all_methods
```

To compare a new run with the reference output:

```
julia --project=replication replication/compare_with_reference.jl
```

The deterministic results (the risk experiment and the numerical checks) agree
to rounding error. The Monte Carlo results are identical with the same Julia and
package versions. Julia does not promise identical random streams across versions,
so with other versions the Monte Carlo results are compared through their Monte
Carlo standard errors. The file `output_bayesian_risk_refinement/scalar_outcomes.csv`
(18 MB) is not included in the reference output.

## Where each result comes from

| In the paper | Output files | Built file |
|---|---|---|
| Table 3, RMSE of posterior-mean approximations | `output_static_all_methods/approximation_results.csv`, `convergence_slopes.csv` | `tables/simulation_summary.tex` |
| Figure 1 | `output_static_all_methods/approximation_results.csv` | `figures/approximation_error.pdf` |
| Section 7.1, quoted numbers | `output_static_all_methods/` | `tables/numbers.md` |
| Section 7.2, quoted numbers | `output_bayesian_risk/` | `tables/numbers.md` |
| Table 4, one-step and accumulated recursive errors | `output_recursive_mcisd/recursive_compact_summary.csv` | `tables/recursive_compact_table.tex` |
| Section 7.3, quoted numbers | `output_recursive_mcisd/` | `tables/numbers.md` |
| SI, static Student-t robustness across tail thickness | `output_static_all_methods/student_df_results.csv`, `student_df_slopes.csv`, `convergence_slopes.csv` | `tables/student_df_robustness.tex` |
| SI, average risk improvement normalized by epsilon^4 | `output_bayesian_risk/risk_results.csv` | `tables/risk_coefficient_table.tex` |
| SI, posterior-mean RMSE in the risk experiments | `output_bayesian_risk/risk_results.csv` | `tables/risk_rmse_table.tex` |
| SI, finite-covariance bounds establishing lower ISD risk | `output_bayesian_risk/certificate_results.csv` | `tables/risk_certificate_table.tex` |
| SI, recursive RMSE relative to the grid filter | `output_recursive_mcisd/recursive_path_results.csv` | `tables/recursive_path_table.tex` |
| SI, recursive errors under Student-t3 observations | `output_recursive_mcisd/recursive_path_results.csv` | `tables/recursive_t3_robustness_table.tex` |
| SI, Appendix F numerical checks | `output_static_gh120_check/`, `output_bayesian_risk_refinement/`, `output_recursive_mcisd_checks/`, diagnostics and `formula_validation.json` | `tables/numbers.md` |

Tables 1 and 2 of the paper are analytical and need no computation.

## Names in the output files

The mean-corrected implicit update is labelled `ISD*` in every output file. In
the recursive experiment the methods are labelled `First order` (ESD), `Second
order` (explicit second order), `Implicit` (ISD), `ISD*`, `Kalman`, and `Exact
Gaussian` (the Gaussian-reset recursion of the paper). Some file and column
names contain `mcisd`, an earlier name for ISD*.
