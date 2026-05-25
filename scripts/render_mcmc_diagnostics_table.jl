#=
Render MCMC diagnostics CSV → LaTeX tabular.

REPL: Run `# %%` cells in order (e.g. VS Code / Cursor: “Julia: Execute Cell”).
=#

# %%
using CSV
using DataFrames
using Printf

include(joinpath(@__DIR__, "_repo_paths.jl"))

# %%
"""
    _format_mcmc_numeric_for_table(x::Real) -> String

At most three digits after the decimal point; strip trailing zeros when not needed.
"""
function _format_mcmc_numeric_for_table(x::Real)
    xf = Float64(x)
    !isfinite(xf) && return string(xf)
    y = round(xf; digits = 3)
    s = @sprintf("%.3f", y)
    s = rstrip(rstrip(s, '0'), '.')
    return isempty(s) ? "0" : s
end

function _fmt_mcmc_cell(x)
    x isa Missing && return "missing"
    x isa Real && return _format_mcmc_numeric_for_table(x)
    return String(x)
end

# %%
"""
    render_mcmc_diagnostics_table(in_csv, out_tex)

Render a LaTeX `tabular` table from a CSV of MCMC diagnostics.

The expected columns are:
`model`, `parameters`, `mean`, `std`, `mcse`, `ess_bulk`, `ess_tail`, `rhat`, `ess_per_sec`.
"""
function render_mcmc_diagnostics_table(in_csv::AbstractString, out_tex::AbstractString)
    df = CSV.read(in_csv, DataFrame)

    open(out_tex, "w") do io
        println(io, raw"\begin{tabular}{llrrrrrrr}")
        println(io, raw"\toprule")
        println(
            io,
            raw"Model & Parameter & Mean & SD & MCSE & Bulk ESS & Tail ESS & \hat{R} & ESS/s \\",
        )
        println(io, raw"\midrule")

        for r in eachrow(df)
            @printf(
                io,
                "%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n",
                string(r.model),
                String(r.parameters),
                _fmt_mcmc_cell(r.mean),
                _fmt_mcmc_cell(r.std),
                _fmt_mcmc_cell(r.mcse),
                _fmt_mcmc_cell(r.ess_bulk),
                _fmt_mcmc_cell(r.ess_tail),
                _fmt_mcmc_cell(r.rhat),
                _fmt_mcmc_cell(r.ess_per_sec),
            )
        end

        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
end

# %%
if abspath(PROGRAM_FILE) == @__FILE__
    out_dir = results_tables_dir()
    stem = "SuppTable_MCMC_diagnostics_SCM_intervention"
    in_csv = joinpath(out_dir, "$(stem).csv")
    out_tex = joinpath(out_dir, "$(stem).tex")
    isfile(in_csv) ||
        error("Missing $in_csv — run `src/4_SCM_intervention.jl` first to export MCMC diagnostics.")
    render_mcmc_diagnostics_table(in_csv, out_tex)
end
