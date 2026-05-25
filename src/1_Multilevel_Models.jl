#=
Multilevel Models
- Julia version: 1.12
- Author: Simon A Babayan

REPL: Run `# %%` cells in order (e.g. VS Code / Cursor: “Julia: Execute Cell”).
=#

#=
This script fits multilevel models to vaccine response data. It includes both classical
mixed-effects models and Bayesian hierarchical models with vaccination history × habitat
interactions. Figure export is omitted in the public methods release.
=#

# %%
## Import packages

print("Running on ", Threads.nthreads(), " threads.")
using CategoricalArrays, LazyArrays
using DataFrames, CSV
using Random
using Statistics, Distributions
using StatsBase, HypothesisTests
using MLDataUtils: shuffleobs, splitobs, rescale!
using LinearAlgebra
using MixedModels
using Turing
using MCMCChains


if isdir("./src/")
    cd("./src/")
end
include("TuringUtils.jl")

# %%
## Import data
include("DataWrangler.jl")

# Data preparation with missing value handling
df = encode_df(df) # Includes repeated measures
df = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 1, df)

df_unique = encode_df(df_unique) # No repeated measures
df_unique = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df_unique)

# Set vaccination history factor levels
levels!(df.vax_history, ["A", "D", "AD", "DA", "DD"])
df = sort(df, :vax_history)
df.IDidx = get_idx(:ID, df)[1]

println("vax_history counts: ", countmap(df.vax_history))

# %%
## Check distribution of transformed E
seroconv = standardize(ZScoreTransform, df.E[df[!, :logOD].>0], dims=1)

println("E standardised KS vs Normal: ", ExactOneSampleKSTest(seroconv, Normal()))

# %%
## Habitat-specific vaccine response comparisons

"""
    calculate_group_difference(data, condition1, condition2, variable)

Calculate mean difference and standard error between two groups.

# Arguments
- `data`: DataFrame containing the data
- `condition1`: Boolean mask for first group
- `condition2`: Boolean mask for second group
- `variable`: Column name for the response variable

# Returns
- Tuple of (mean_difference, standard_error)
"""
function calculate_group_difference(data, condition1, condition2, variable)
    mean1 = mean(data[condition1, variable])
    mean2 = mean(data[condition2, variable])
    std1 = std(data[condition1, variable])
    std2 = std(data[condition2, variable])
    n1 = sum(condition1)
    n2 = sum(condition2)

    mean_diff = mean1 - mean2
    se = sqrt((std1^2 / n1) + (std2^2 / n2))

    return round(mean_diff; digits=2), round(se; digits=2)
end

# Define comparison groups
boosted_lab = (df.vax_history .== "DD") .& (df.islab .== 1) .& (df.isvax .== 1)
boosted_wild = (df.vax_history .== "DD") .& (df.islab .== 0) .& (df.isvax .== 1)
nonboosted_lab = (df.vax_history .!= "DD") .& (df.islab .== 1) .& (df.isvax .== 1)
nonboosted_wild = (df.vax_history .!= "DD") .& (df.islab .== 0) .& (df.isvax .== 1)
vaccinated_lab = (df.islab .== 1) .& (df.isvax .== 1)
vaccinated_wild = (df.islab .== 0) .& (df.isvax .== 1)
da_lab = (df.vax_history .== "DA") .& (df.islab .== 1) .& (df.isvax .== 1)
da_wild = (df.vax_history .== "DA") .& (df.islab .== 0) .& (df.isvax .== 1)

# Calculate group comparisons
Δm, Δse = calculate_group_difference(df, boosted_wild, boosted_lab, :E)
println("Wild boosted vs. Lab boosted OD: $Δm ± $Δse")

Δm, Δse = calculate_group_difference(df, nonboosted_wild, nonboosted_lab, :E)
println("Wild non-boosted vs. Lab non-boosted OD: $Δm ± $Δse")

Δm, Δse = calculate_group_difference(df, vaccinated_wild, vaccinated_lab, :E)
println("Wild vaccinated vs. Lab vaccinated OD: $Δm ± $Δse")

Δm, Δse = calculate_group_difference(df, boosted_lab, da_lab, :E)
println("Lab boosted vs. Lab non-boosted OD: $Δm ± $Δse")

Δm, Δse = calculate_group_difference(df, boosted_wild, da_wild, :E)
println("Wild boosted vs. Wild non-boosted OD: $Δm ± $Δse")

# Percentage change in original OD scale
Δm_od, Δse_od = calculate_group_difference(df, vaccinated_wild, vaccinated_lab, :OD)
lab_mean = mean(df[vaccinated_lab, :OD])
Δpc = round((Δm_od / lab_mean) * 100; digits=1)
Δpc_se = round((Δse_od / lab_mean) * 100; digits=1)
println("Percentage drop in OD in wild vs lab: $Δpc ± $Δpc_se")

# %%
## Multilevel model selection for vaccination × habitat interactions

# Define candidate model formulae
vi_form_1 = @formula(E ~ (1 | ID) + vax_history + H + D + vax_history & H + H & D)
vi_form_2 = @formula(E ~ (1 | ID) + vax_history + H + D + vax_history & H)
vi_form_3 = @formula(E ~ (1 | ID) + vax_history + H + D + H & D)
vi_form_4 = @formula(E ~ (1 | ID) + vax_history + H + D)

# Fit candidate models
mm1 = fit(MixedModel, vi_form_1, df)
mm2 = fit(MixedModel, vi_form_2, df)
mm3 = fit(MixedModel, vi_form_3, df)
mm4 = fit(MixedModel, vi_form_4, df)

# Model comparison using likelihood ratio tests
println("Model selection via likelihood ratio tests:")
println("Full vs. vaccination×habitat only:")
MixedModels.likelihoodratiotest(mm1, mm2)

println("Full vs. habitat×diet only:")
MixedModels.likelihoodratiotest(mm1, mm3)

println("Vaccination×habitat vs. main effects only:")
MixedModels.likelihoodratiotest(mm2, mm4)

# Selected model (vaccination × habitat interaction)
mm = fit(MixedModel, vi_form_2, df)

# Model diagnostics
boot = parametricbootstrap(MersenneTwister(42), 3000, mm)

"""
    varying_intercept(IDidx, Vidx, H, D, E)

Bayesian hierarchical model for vaccine response with vaccination history × habitat interactions.

This model estimates the effect of vaccination history on immune response while accounting
for habitat differences and their interaction. Individual-level random intercepts capture
between-mouse variation not explained by measured covariates.

# Arguments
- `IDidx::Vector{Int}`: Individual mouse identifiers (indexed)
- `Vidx::Vector{Int}`: Vaccination history identifiers (indexed)
- `H::Vector{Int}`: Habitat (1=lab, 2=wild)
- `D::Vector{Int}`: Diet supplementation (1=low, 2=high)
- `E::Vector{Float64}`: Standardised vaccine response (outcome)

# Model Structure
- Population effects: vaccination history, habitat, diet
- Interaction: vaccination history × habitat
- Random effects: individual mouse intercepts
- Residual variation: normally distributed

# Prior Specification
Uses weakly informative priors scaled to the standardised outcome:
- Intercept: Normal(Ē, 2.5×SD) allowing reasonable deviations
- Effects: Normal(Ē, 2) for vaccination, Normal(0, 2) for others
- Random effects: Cauchy(0, 2) for moderate individual variation
- Residual: Exponential(SD) for positive variance

# Returns
Posterior samples for all model parameters enabling uncertainty quantification.
"""
@model function varying_intercept(IDidx, Vidx, H, D, E)

    n_id = length(unique(IDidx))
    n_vax = length(unique(Vidx))
    Ē = mean(E)

    # Population-level priors
    α ~ Normal(Ē, 2.5 * std(E))           # population-level intercept
    βv ~ filldist(Normal(Ē, 2), n_vax)    # population-level slopes relative to adjuvant control
    βh ~ Normal(0, 2)                     # population-level slopes
    βd ~ Normal(0, 2)                     # population-level slopes
    βvh ~ filldist(Normal(Ē, 2), n_vax)   # interaction term between Vidx and H
    σ ~ Exponential(std(E))               # residual SD

    # Random effects priors
    τ ~ truncated(Cauchy(0, 2); lower=0)  # group-level SDs intercepts
    α_ID ~ filldist(Normal(0, τ), n_id)   # group-level intercepts

    # Likelihood
    Ê = @. α + α_ID[IDidx] + βv[Vidx] + βh * H + βd * D + βvh[Vidx] * H
    E ~ MvNormal(Ê, σ^2 * I)

end

vi_model = varying_intercept(df.IDidx, df.Vidx, df.H, df.D, df.E)

vi_chn = sample(vi_model, NUTS(), MCMCThreads(), 3000, 4)


vi_chn_df = DataFrame(vi_chn)[!, r"α\b|β"]
precis(vi_chn_df)

# %%
# Manuscript figure generation omitted in this public methods release.
