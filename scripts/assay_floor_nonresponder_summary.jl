#=
Assay-floor / non-responder summary for IgG1 ELISA.

Uses the same observation window as `render_arm_level_igg1_summary_table.jl`:
samples from antigen-containing arms (>7 days after the most recent immunisation visit),
with assay-floor defined as OD_avg at or below the per-plate cut-off (`cutoff_new`).

Supports Supplementary Information on assay-floor and non-responder checks.

REPL: Run `# %%` cells in order, or execute as a script.
=#

# %%
using CSV
using DataFrames
using Printf
using Statistics

include(joinpath(@__DIR__, "_repo_paths.jl"))

const VACCINE_ARMS = ("D", "AD", "DA", "DD")

# %%
"""
    prepare_vaccinated_igg1_observations(df; min_days_since_last_immunisation=7)

Filter to antigen-containing immunisation histories with IgG1 OD and timing data,
using the most recent immunisation visit (boost when available, otherwise prime).
"""
function prepare_vaccinated_igg1_observations(
    df::DataFrame;
    min_days_since_last_immunisation::Real = 7,
)
    required = [
        :vax_history,
        :Env,
        :days_since_1st_trt,
        :days_since_2nd_trt,
        :OD_avg,
        :cutoff_new,
        :isvax,
        :ID,
    ]
    present = Set(Symbol.(names(df)))
    missing_cols = [c for c in required if !(c in present)]
    if !isempty(missing_cols)
        error("Missing required columns: $(missing_cols)")
    end

    df = copy(df)
    df.days_since_last_immunisation =
        coalesce.(df.days_since_2nd_trt, df.days_since_1st_trt)

    filter(
        row ->
            row.isvax == 1 &&
            !ismissing(row.days_since_last_immunisation) &&
            row.days_since_last_immunisation > min_days_since_last_immunisation &&
            !ismissing(row.OD_avg) &&
            !ismissing(row.cutoff_new) &&
            !ismissing(row.vax_history) &&
            !ismissing(row.Env) &&
            !ismissing(row.ID) &&
            String(row.vax_history) in VACCINE_ARMS,
        df,
    )
end

# %%
"""Return `true` when blank-centred OD is at or below the per-plate assay cut-off."""
is_assay_floor(row) = row.OD_avg <= row.cutoff_new

# %%
"""
    summarize_assay_floor_by_habitat_arm(df_obs)

Observation- and individual-level counts of assay-floor values by habitat and arm.
"""
function summarize_assay_floor_by_habitat_arm(df_obs::DataFrame)
    df_obs = copy(df_obs)
    df_obs.habitat = ifelse.(String.(df_obs.Env) .== "Lab", "laboratory", "wild")
    df_obs.arm = String.(df_obs.vax_history)
    df_obs.assay_floor = is_assay_floor.(eachrow(df_obs))

    g = groupby(df_obs, [:habitat, :arm])
    obs_summary = combine(
        g,
        nrow => :n_observations,
        :assay_floor => sum => :n_assay_floor,
        :ID => (ids -> length(unique(ids))) => :n_individuals,
    )

    indiv_flags = combine(
        groupby(df_obs, [:habitat, :arm, :ID]),
        :assay_floor => any => :individual_assay_floor,
    )
    indiv_floor = combine(
        groupby(indiv_flags, [:habitat, :arm]),
        :individual_assay_floor => sum => :n_individuals_assay_floor,
    )

    out = leftjoin(obs_summary, indiv_floor, on=[:habitat, :arm])
    out.n_individuals_assay_floor = coalesce.(out.n_individuals_assay_floor, 0)

    arm_order = Dict("D" => 1, "AD" => 2, "DA" => 3, "DD" => 4)
    habitat_order = Dict("laboratory" => 1, "wild" => 2)
    out.arm_ord = [get(arm_order, a, 999) for a in out.arm]
    out.habitat_ord = [get(habitat_order, h, 999) for h in out.habitat]
    sort!(out, [:habitat_ord, :arm_ord])
    select!(out, Not([:arm_ord, :habitat_ord]))

    return out
end

# %%
"""
    summarize_wild_d_individuals(df_obs)

Individual-level detail for wild habitat, arm D (repeat-capture check).
"""
function summarize_wild_d_individuals(df_obs::DataFrame)
    sub = filter(
        row -> String(row.Env) == "Wild" && String(row.vax_history) == "D",
        df_obs,
    )
    rows = NamedTuple[]
    for g in groupby(sub, :ID)
        push!(
            rows,
            (
                ID = first(g.ID),
                n_post_threshold_observations = nrow(g),
                n_assay_floor_obs = sum(is_assay_floor.(eachrow(g))),
            ),
        )
    end
    return DataFrame(rows)
end

# %%
"""
    habitat_od_contrast(df_obs; exclude_assay_floor=false)

Mean IgG1 OD (raw scale) for laboratory vs wild among all vaccinated arms.
"""
function habitat_od_contrast(df_obs::DataFrame; exclude_assay_floor::Bool = false)
    sub = exclude_assay_floor ? filter(row -> !is_assay_floor(row), df_obs) : df_obs
    lab = sub[String.(sub.Env) .== "Lab", :OD_avg]
    wild = sub[String.(sub.Env) .== "Wild", :OD_avg]
    return (
        lab_mean = mean(lab),
        wild_mean = mean(wild),
        difference = mean(lab) - mean(wild),
        n_lab = length(lab),
        n_wild = length(wild),
    )
end

# %%
"""
    render_assay_floor_table(summary, out_tex)

Write LaTeX tabular for supplementary assay-floor counts (habitat × arm).
"""
function render_assay_floor_table(summary::DataFrame, out_tex::AbstractString)
    open(out_tex, "w") do io
        println(io, raw"\begin{tabular}{llrrrr}")
        println(io, raw"\toprule")
        @printf(
            io,
            "%s & %s & %s & %s & %s & %s \\\\\n",
            raw"\textbf{Habitat}",
            raw"\textbf{Arm}",
            raw"\textbf{\$n\$ observations}",
            raw"\textbf{\$n\$ assay-floor}",
            raw"\textbf{\$n\$ individuals}",
            raw"\textbf{\$n\$ individuals assay-floor}",
        )
        println(io, raw"\midrule")
        for r in eachrow(summary)
            @printf(
                io,
                "%s & %s & %d & %d & %d & %d \\\\\n",
                r.habitat,
                r.arm,
                r.n_observations,
                r.n_assay_floor,
                r.n_individuals,
                r.n_individuals_assay_floor,
            )
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
end

# %%
"""
    run_assay_floor_nonresponder_summary(;
        in_csv,
        out_tex,
        out_csv=nothing,
        min_days_since_last_immunisation=7,
    )

Compute summaries, print to stdout, and write LaTeX (and optional CSV) outputs.
"""
function run_assay_floor_nonresponder_summary(;
    in_csv::AbstractString,
    out_tex::AbstractString,
    out_csv::Union{Nothing, AbstractString} = nothing,
    min_days_since_last_immunisation::Real = 7,
)
    df = CSV.read(in_csv, DataFrame; normalizenames = true)
    df_obs = prepare_vaccinated_igg1_observations(
        df;
        min_days_since_last_immunisation = min_days_since_last_immunisation,
    )
    summary = summarize_assay_floor_by_habitat_arm(df_obs)
    wild_d = summarize_wild_d_individuals(df_obs)

    contrast_all = habitat_od_contrast(df_obs; exclude_assay_floor = false)
    contrast_excl = habitat_od_contrast(df_obs; exclude_assay_floor = true)

    println("=== Assay-floor summary (>", min_days_since_last_immunisation, " days) ===")
    show(summary; allcols = true, allrows = true)
    println()
    println("=== Wild arm D: individual-level post-threshold captures ===")
    show(wild_d; allcols = true, allrows = true)
    println()
    n_wild_d_floor_ids = sum(wild_d.n_assay_floor_obs .> 0)
    println("Wild D subjects with ≥1 post-threshold observation: ", nrow(wild_d))
    println("Wild D subjects with ≥1 assay-floor observation: ", n_wild_d_floor_ids)
    for r in eachrow(wild_d)
        if r.n_assay_floor_obs > 0
            println(
                "  $(r.ID): $(r.n_assay_floor_obs)/$(r.n_post_threshold_observations) assay-floor observations",
            )
        end
    end
    println()
    println("=== Habitat mean IgG1 OD (all vaccinated arms) ===")
    @printf(
        "  All observations:    lab %.3f, wild %.3f, Δ = %.3f (n_lab=%d, n_wild=%d)\n",
        contrast_all.lab_mean,
        contrast_all.wild_mean,
        contrast_all.difference,
        contrast_all.n_lab,
        contrast_all.n_wild,
    )
    @printf(
        "  Excluding assay-floor: lab %.3f, wild %.3f, Δ = %.3f (n_lab=%d, n_wild=%d)\n",
        contrast_excl.lab_mean,
        contrast_excl.wild_mean,
        contrast_excl.difference,
        contrast_excl.n_lab,
        contrast_excl.n_wild,
    )

    render_assay_floor_table(summary, out_tex)
    if out_csv !== nothing
        CSV.write(out_csv, summary)
    end

    return (
        summary = summary,
        wild_d_individuals = wild_d,
        habitat_contrast_all = contrast_all,
        habitat_contrast_excluding_floor = contrast_excl,
    )
end

# %%
if abspath(PROGRAM_FILE) == @__FILE__
    out_dir = results_tables_dir()
    in_csv = isfile(default_clean_csv()) ? default_clean_csv() : default_joint_csv()
    out_tex = joinpath(out_dir, "SuppTable_assay_floor_nonresponders.tex")
    out_csv = joinpath(out_dir, "SuppTable_assay_floor_nonresponders.csv")
    run_assay_floor_nonresponder_summary(;
        in_csv,
        out_tex,
        out_csv,
        min_days_since_last_immunisation = 7,
    )
end
