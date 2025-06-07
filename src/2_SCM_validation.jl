#=
SCM validation
- Julia version: 1.11
- Author: Simon A Babayan
=#

#=
This script will validate the SCM using the data from the manuscript.
=#

## Import packages
print("Running on ", Threads.nthreads(), " threads.")
using CSV, DataFrames
using HypothesisTests
using GLM, MixedModels
using RCall
@rlibrary dagitty # we use the original dagitty from R

## Import data
if isdir("./src/")
    cd("./src/")
end
include("DataWrangler.jl")
# Data preparation - use efficient filtering with missing value handling
df = encode_df(df) # Choose between df and df_unique (the latter has no repeated measures)
df_sel = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df)
df_sel = filter(row -> !ismissing(row.Fat_Scores_Sum), df_sel) # Drop missing fat scores
# Optional filter (commented out)
# df_sel = filter(row -> row.vax_history != "DD", df_sel) # Keep only mice with single vaccination or adjuvant

dag_df = df_sel[!, [:E, :H, :V, :D, :R, :S, :M, :F, :T, :P, :nP, :Vidx, :vax_history, :ID]]
# Convert 1 / 0 to true / false
# dag_df[!, :V] = convert(Vector{Bool}, dag_df[!, :V])
# dag_df[!, :P] = convert(Vector{Bool}, dag_df[!, :P])

# standardize F
# dag_df.F = standardize(ZScoreTransform, convert(Vector{Float64}, df_sel.F), dims=1)
dag_df.F = standardize(ZScoreTransform, StatsBase.convert(Vector{Float64}, df_sel.F), dims=1)

describe(dag_df)

# DAG specification
#
dag = dagitty("dag{ D -> E; D -> F; D -> M; D -> P; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> P; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; R -> P; S -> E; S -> F; S -> M; S -> P; S -> R; V -> E; V -> F; V -> M; V -> P; V -> R; }")

impliedCovarianceMatrix(dag)

impliedConditionalIndependencies(dag)

# rough test of conditional independencies in light of observed data
localTests(dag, data=dag_df[!, [:E, :H, :V, :D, :R, :S, :M, :F, :P, :nP]])

# more sophisticated test of conditional independencies in light of observed data using mixed models
# D _||_ S
form = @formula(D ~ (1 | ID) + (1 | vax_history) + S)
fit(MixedModel, form, dag_df)

# D _||_ H
form = @formula(D ~ (1 | ID) + (1 | vax_history) + H)
fit(MixedModel, form, dag_df)

# D _||_ V
form = @formula(D ~ (1 | ID) + (1 | vax_history) + V)
fit(MixedModel, form, dag_df)

# H _||_ S
form = @formula(H ~ (1 | ID) + (1 | vax_history) + S)
fit(MixedModel, form, dag_df)

# H _||_ V
dag_df.Vbinom = dag_df.V .- 1;
form = @formula(Vbinom ~ (1 | ID) + (1 | vax_history) + H)
fit(MixedModel, form, dag_df, Bernoulli())

# S _||_ V
dag_df.Sbinom = dag_df.S .- 1;
form = @formula(Sbinom ~ (1 | ID) + (1 | vax_history) + V)
fit(MixedModel, form, dag_df, Bernoulli())
