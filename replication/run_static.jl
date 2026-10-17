# Section 7.1: static one-step experiment.
#
#   julia --project=replication replication/run_static.jl
#   julia --project=replication replication/run_static.jl --gh-nodes 120 --output-dir replication/output/output_static_gh120_check
#
# Options (defaults in parentheses): --n-rep (100000), --gh-nodes (80),
# --seed (20260814), --slope-max-p (0.0025), --p-grid (comma-separated list),
# --output-dir (replication/output/output_static_all_methods).

using TweedieScore

const ROOT = @__DIR__

opts = parse_cli(ARGS, Dict{String,Any}(
    "n-rep" => 100_000,
    "gh-nodes" => 80,
    "seed" => 20260814,
    "slope-max-p" => 0.0025,
    "p-grid" => copy(TweedieScore.STATIC_P_GRID),
    "output-dir" => joinpath(ROOT, "output", "output_static_all_methods"),
))

run_static_experiment(;
    n_rep = opts["n-rep"],
    gh_nodes = opts["gh-nodes"],
    seed = opts["seed"],
    p_grid = opts["p-grid"],
    slope_max_p = opts["slope-max-p"],
    output_dir = opts["output-dir"],
)
