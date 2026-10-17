# Section 7.3: recursive Student-t filtering experiment.
#
#   julia --project=replication -t auto replication/run_recursive.jl
#   julia --project=replication -t auto replication/run_recursive.jl --grid-check-only --output-dir replication/output/output_recursive_mcisd_checks
#
# Options (defaults in parentheses): --n-rep (1000), --T (500), --nus (3,5,10),
# --snrs (0.25,1), --phis (0.8,0.98), --grid-points (481), --grid-sd (8),
# --gh-nodes (32), --seed (20260815), --representative-design (3,1,0.98),
# --grid-check-only, --grid-check-n-rep (200), --grid-check-T (200),
# --grid-check-coarse-points (241), --grid-check-fine-points (961),
# --grid-check-fine-gh-nodes (64), --output-dir (replication/output/output_recursive_mcisd, or
# replication/output/output_recursive_mcisd_checks with --grid-check-only).
# A relative --output-dir is taken relative to the current directory.
#
# The -t auto flag lets the per-path grid updates run on all cores; the
# results do not depend on the number of threads.

using TweedieScore

const ROOT = @__DIR__
const D = TweedieScore.RECURSIVE_DEFAULTS

opts = parse_cli(ARGS, Dict{String,Any}(
    "n-rep" => D.n_rep,
    "T" => D.horizon,
    "nus" => copy(D.nus),
    "snrs" => copy(D.snrs),
    "phis" => copy(D.phis),
    "grid-points" => D.grid_points,
    "grid-sd" => D.grid_sd,
    "gh-nodes" => D.gh_nodes,
    "seed" => D.seed,
    "representative-design" => copy(D.representative_design),
    "grid-check-only" => false,
    "grid-check-n-rep" => D.grid_check_n_rep,
    "grid-check-T" => D.grid_check_T,
    "grid-check-coarse-points" => D.grid_check_coarse_points,
    "grid-check-fine-points" => D.grid_check_fine_points,
    "grid-check-fine-gh-nodes" => D.grid_check_fine_gh_nodes,
    "output-dir" => "",
))

if opts["grid-check-only"]
    output_dir = isempty(opts["output-dir"]) ?
        joinpath(ROOT, "output", "output_recursive_mcisd_checks") : opts["output-dir"]
    run_recursive_grid_checks(;
        output_dir = output_dir,
        nus = opts["nus"],
        snrs = opts["snrs"],
        phis = opts["phis"],
        grid_points = opts["grid-points"],
        grid_sd = opts["grid-sd"],
        gh_nodes = opts["gh-nodes"],
        seed = opts["seed"],
        representative_design = opts["representative-design"],
        grid_check_n_rep = opts["grid-check-n-rep"],
        grid_check_T = opts["grid-check-T"],
        grid_check_coarse_points = opts["grid-check-coarse-points"],
        grid_check_fine_points = opts["grid-check-fine-points"],
        grid_check_fine_gh_nodes = opts["grid-check-fine-gh-nodes"],
    )
else
    output_dir = isempty(opts["output-dir"]) ?
        joinpath(ROOT, "output", "output_recursive_mcisd") : opts["output-dir"]
    run_recursive_experiment(;
        output_dir = output_dir,
        n_rep = opts["n-rep"],
        horizon = opts["T"],
        nus = opts["nus"],
        snrs = opts["snrs"],
        phis = opts["phis"],
        grid_points = opts["grid-points"],
        grid_sd = opts["grid-sd"],
        gh_nodes = opts["gh-nodes"],
        seed = opts["seed"],
        representative_design = opts["representative-design"],
    )
end
