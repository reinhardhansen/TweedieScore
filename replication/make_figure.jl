# Figure 1 of the paper (Section 7.1): RMSE of the posterior-mean correction
# against the predictive variance P, on log-log axes, one panel per model.
# Only the three models and four methods of the paper's figure are drawn; the
# Student-t robustness runs (student_df_results.csv) are for a table.
#
#   julia --project=replication replication/make_figure.jl
#   julia --project=replication replication/make_figure.jl --input-dir replication/reference_output/output_static_all_methods
#
# Options (defaults in parentheses): --input-dir
# (replication/output/output_static_all_methods), --output-dir
# (replication/figures). A relative path given on the command line is taken
# relative to the current directory.
# Writes approximation_error.pdf and approximation_error.png (220 dpi), each
# 11.4 by 4.25 inches.

using TweedieScore
using CSV
using CairoMakie

const ROOT = @__DIR__

opts = parse_cli(ARGS, Dict{String,Any}(
    "input-dir" => joinpath(ROOT, "output", "output_static_all_methods"),
    "output-dir" => joinpath(ROOT, "figures"),
))

# Panels in the order of the paper, keyed by the `model` column of the CSV.
const PANELS = [
    ("poisson", "Poisson log intensity"),
    ("log_variance", "Gaussian log variance"),
    ("student_t", rich("Student-", rich("t"; font = :italic), subscript("5"), " location")),
]

# Series in legend order, keyed by the `approximation` column, with the
# colors, markers (o s D ^) and line styles (-- : - -.) of Figure 1.
# The asterisk of ISD* is set at full size; the text asterisk already sits at
# the cap height. (A superscript asterisk was too small to read in print.)
# `linestyle = nothing` is solid.
const SERIES = [
    (method = "ESD", label = "ESD", color = "#0072B2",
     marker = :circle, markersize = 4.6, linestyle = :dash),
    (method = "ISD", label = "ISD", color = "#009E73",
     marker = :rect, markersize = 5.0, linestyle = :dot),
    (method = "ISD*", label = "ISD*",
     color = "#CC79A7", marker = :diamond, markersize = 5.4, linestyle = nothing),
    (method = "Explicit second order", label = "Explicit second order", color = "#D55E00",
     marker = :utriangle, markersize = 5.0, linestyle = :dashdot),
]

const GRID_COLOR = "#D9D9D9"
const LINE_WIDTH = 1.6
const WIDTH_IN = 11.4
const HEIGHT_IN = 4.25
const PNG_DPI = 220

"""(P, rmse) of one model and method, sorted by P. Nonpositive values are dropped."""
function series_data(rows, model::AbstractString, method::AbstractString)
    sel = [r for r in rows if r.model == model && r.approximation == method]
    isempty(sel) && error("no rows with model = $model and approximation = $method")
    sort!(sel; by = r -> r.P)
    P = Float64[r.P for r in sel]
    rmse = Float64[coalesce(r.rmse, NaN) for r in sel]
    keep = (P .> 0) .& (rmse .> 0)   # false for NaN as well
    if !all(keep)
        @warn "dropping $(count(!, keep)) nonpositive or missing values for $model, $method"
    end
    return P[keep], rmse[keep]
end

"""log10 limits of `values`, widened by 5% of the range on each side as in matplotlib."""
function log_limits(values; margin = 0.05)
    lo, hi = extrema(log10.(values))
    pad = margin * (hi - lo)
    return lo - pad, hi + pad
end

"""
    log_ticks(lo, hi) -> ((values, labels), minor_values)

Ticks for a log10 axis from 10^lo to 10^hi. Major ticks are at every decade,
or at every second (third, ...) decade when the axis spans more than six
decades, aligned as matplotlib's LogLocator aligns them. Minor ticks are at
2, ..., 9 times each decade, or at the skipped decades. Labels read 10^k.
"""
function log_ticks(lo::Real, hi::Real)
    kfirst, klast = floor(Int, lo), ceil(Int, hi)
    ndec = floor(Int, hi) - ceil(Int, lo)
    stride = ndec <= 6 ? 1 : cld(ndec, 6)
    within(v) = 10.0^lo <= v <= 10.0^hi
    majors = [k for k in kfirst:stride:klast if within(10.0^k)]
    minors = if stride == 1
        [m * 10.0^k for k in kfirst:klast for m in 2:9]
    else
        [10.0^k for k in kfirst:klast if mod(k - kfirst, stride) != 0]
    end
    filter!(within, minors)
    power(k) = k < 0 ? string(Char(0x2212), -k) : string(k)   # U+2212 minus sign
    labels = [rich("10", superscript(power(k); offset = (0.1, 0.0))) for k in majors]
    return ([10.0^k for k in majors], labels), minors
end

input_file = joinpath(opts["input-dir"], "approximation_results.csv")
isfile(input_file) || error("$input_file not found; run replication/run_static.jl first")
rows = collect(CSV.File(input_file; stringtype = String))

# Sizes are in points (1 unit = 1 pt when saved with pt_per_unit = 1).
fig = Figure(size = (round(Int, 72 * WIDTH_IN), round(Int, 72 * HEIGHT_IN)),
             fontsize = 10, figure_padding = (4, 10, 4, 4))

legend_plots = Any[]
for (j, (model, title)) in enumerate(PANELS)
    data = [series_data(rows, model, s.method) for s in SERIES]
    xlo, xhi = log_limits(reduce(vcat, [d[1] for d in data]))
    ylo, yhi = log_limits(reduce(vcat, [d[2] for d in data]))
    xticks, xminor = log_ticks(xlo, xhi)
    yticks, yminor = log_ticks(ylo, yhi)

    ax = Axis(fig[2, j];
        title = title, titlefont = :regular, titlesize = 10, titlegap = 5,
        xlabel = rich("Predictive variance ", rich("P"; font = :italic)),
        ylabel = j == 1 ? "RMSE of posterior-mean correction" : "",
        xscale = log10, yscale = log10,
        limits = (10.0^xlo, 10.0^xhi, 10.0^ylo, 10.0^yhi),
        xticks = xticks, yticks = yticks,
        xminorticks = xminor, yminorticks = yminor,
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        xgridcolor = GRID_COLOR, ygridcolor = GRID_COLOR,
        xminorgridcolor = GRID_COLOR, yminorgridcolor = GRID_COLOR,
        xgridwidth = 0.55, ygridwidth = 0.55,
        xminorgridwidth = 0.55, yminorgridwidth = 0.55,
        spinewidth = 0.8,
        xtickwidth = 0.8, ytickwidth = 0.8, xticksize = 3.5, yticksize = 3.5,
        xminortickwidth = 0.6, yminortickwidth = 0.6,
        xminorticksize = 2.0, yminorticksize = 2.0,
    )

    for (s, (P, rmse)) in zip(SERIES, data)
        plt = scatterlines!(ax, P, rmse;
            color = s.color, linewidth = LINE_WIDTH, linestyle = s.linestyle,
            marker = s.marker, markersize = s.markersize)
        j == 1 && push!(legend_plots, plt)
    end
end

# Shared legend above the panels: one row of four entries, no frame.
Legend(fig[1, 1:length(PANELS)], legend_plots, [s.label for s in SERIES];
    orientation = :horizontal, framevisible = false, labelsize = 10,
    patchsize = (22, 10), patchlabelgap = 6, colgap = 20, padding = (0, 0, 0, 0))

rowgap!(fig.layout, 8)
colgap!(fig.layout, 12)

output_dir = opts["output-dir"]
mkpath(output_dir)
pdf_file = joinpath(output_dir, "approximation_error.pdf")
png_file = joinpath(output_dir, "approximation_error.png")
save(pdf_file, fig; pt_per_unit = 1)
save(png_file, fig; px_per_unit = PNG_DPI / 72)
println("Wrote $pdf_file")
println("Wrote $png_file")
