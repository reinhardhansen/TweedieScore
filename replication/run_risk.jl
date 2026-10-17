# Section 7.2: deterministic Bayesian risk experiment.
#
#   julia --project=replication replication/run_risk.jl
#   julia --project=replication replication/run_risk.jl --nodes 120 --max-count 8192 --output-dir replication/output/output_bayesian_risk_refinement
#
# Options (defaults in parentheses): --nodes (80), --max-count (4096),
# --output-dir (replication/output/output_bayesian_risk).

using TweedieScore

const ROOT = @__DIR__

opts = parse_cli(ARGS, Dict{String,Any}(
    "nodes" => 80,
    "max-count" => 4096,
    "output-dir" => joinpath(ROOT, "output", "output_bayesian_risk"),
))

run_risk_experiment(;
    nodes = opts["nodes"],
    max_count = opts["max-count"],
    output_dir = opts["output-dir"],
)
