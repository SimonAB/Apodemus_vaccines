#=
Render arm-level IgG1 summary table for LaTeX.

REPL: Run `# %%` cells in order (e.g. VS Code / Cursor: “Julia: Execute Cell”).
=#

# %%
using CSV
using DataFrames
using Printf
using Statistics

include(joinpath(@__DIR__, "_repo_paths.jl"))

# %%
"""
    render_arm_level_igg1_summary_table(in_csv, out_tex; min_days_since_last_immunisation=7)

Render a LaTeX `tabular` for descriptive arm-level IgG1 OD summaries stratified by
immunisation history × habitat × diet.

The table matches the manuscript convention of showing only observations taken more than
`min_days_since_last_immunisation` days after the most recent immunisation visit.

Expected columns in `in_csv`:
- `vax_history`: one of A, D, AD, DA, DD
- `Env`: habitat (e.g. Lab, Wild)
- `Diet`: diet indicator (e.g. Low/High)
- `days_since_1st_trt`, `days_since_2nd_trt`: days since prime/boost
- `OD_avg`: raw IgG1 OD summary per observation (main read-out for plotting/analysis)
"""
function render_arm_level_igg1_summary_table(
    in_csv::AbstractString,
    out_tex::AbstractString;
    min_days_since_last_immunisation::Real = 7,
)
    df = CSV.read(in_csv, DataFrame; normalizenames = true)

    required = [
        :vax_history,
        :Env,
        :Diet,
        :days_since_1st_trt,
        :days_since_2nd_trt,
        :OD_avg,
    ]
    present = Set(Symbol.(names(df)))
    missing_cols = [c for c in required if !(c in present)]
    if !isempty(missing_cols)
        error("Missing required columns in input CSV: $(missing_cols)")
    end

    df.days_since_last_immunisation = coalesce.(df.days_since_2nd_trt, df.days_since_1st_trt)

    df_filt = filter(
        row ->
            !ismissing(row.days_since_last_immunisation) &&
            row.days_since_last_immunisation > min_days_since_last_immunisation &&
            !ismissing(row.OD_avg) &&
            !ismissing(row.vax_history) &&
            !ismissing(row.Env) &&
            !ismissing(row.Diet),
        df,
    )

    df_filt.arm = String.(df_filt.vax_history)
    df_filt.habitat = ifelse.(String.(df_filt.Env) .== "Lab", "laboratory", "wild")
    df_filt.diet = ifelse.(String.(df_filt.Diet) .== "High", "supplemented", "control")

    g = groupby(df_filt, [:arm, :habitat, :diet])
    out = combine(
        g,
        nrow => :n,
        :OD_avg => (x -> mean(skipmissing(x))) => :mean_od,
        :OD_avg => (x -> std(collect(skipmissing(x)))) => :sd_od,
    )

    arm_order = Dict("A" => 1, "D" => 2, "AD" => 3, "DA" => 4, "DD" => 5)
    habitat_order = Dict("laboratory" => 1, "wild" => 2)
    diet_order = Dict("control" => 1, "supplemented" => 2)
    out.arm_ord = [get(arm_order, a, 999) for a in out.arm]
    out.habitat_ord = [get(habitat_order, h, 999) for h in out.habitat]
    out.diet_ord = [get(diet_order, d, 999) for d in out.diet]
    sort!(out, [:arm_ord, :habitat_ord, :diet_ord])
    select!(out, Not([:arm_ord, :habitat_ord, :diet_ord]))

    open(out_tex, "w") do io
        println(io, raw"\begin{tabular}{lllrrr}")
        println(io, raw"\toprule")
        @printf(
            io,
            "%s & %s & %s & %s & %s & %s \\\\\n",
            raw"\textbf{Arm}",
            raw"\textbf{Habitat}",
            raw"\textbf{Diet}",
            raw"\textbf{n}",
            raw"\textbf{Mean IgG1 OD}",
            raw"\textbf{SD IgG1 OD}",
        )
        println(io, raw"\midrule")

        for r in eachrow(out)
            @printf(
                io,
                "%s & %s & %s & %d & %.3f & %.3f \\\\\n",
                r.arm,
                r.habitat,
                r.diet,
                r.n,
                r.mean_od,
                r.sd_od,
            )
        end

        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
end

# %%
if abspath(PROGRAM_FILE) == @__FILE__
    out_dir = results_tables_dir()
    in_csv = isfile(default_clean_csv()) ? default_clean_csv() : default_joint_csv()
    out_tex = joinpath(out_dir, "SuppTable_arm_level_IgG1_by_arm_habitat_diet.tex")
    render_arm_level_igg1_summary_table(in_csv, out_tex; min_days_since_last_immunisation = 7)
end
