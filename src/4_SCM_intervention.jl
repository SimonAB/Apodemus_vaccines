#=
SCM Intervention
- Julia version: 1.11
- Author: Simon A Babayan
=#

#=
This script implements counterfactual inference under parasite elimination interventions
using structural causal models. It compares vaccine efficacy in factual (with parasites)
vs counterfactual (without parasites) scenarios using Bayesian generative models.
=#

## Import packages

print("Running on ", Threads.nthreads(), " threads.")
# Data handling
using CSV, DataFrames
# Statistics
using Random
using Distributions
using HypothesisTests
using MixedModels
# Modelling
using LazyArrays
using LinearAlgebra: I
using MCMCChains
using Turing
using ReverseDiff
# Turing.setbackend(:reversediff)
# Turing.setrdcache(true)

using RCall
@rlibrary dagitty

# Plotting & diagnostics
using CairoMakie
CairoMakie.activate!(type="svg")
using MixedModelsMakie
using Colors

# Include modules
if isdir("./src/")
    cd("./src/")
end
include("TuringUtils.jl")
include("TuringPlots.jl")
include("PlottingUtils.jl")

# Import data
include("DataWrangler.jl")

## Data preparation

# All cases - use more efficient filtering
df = encode_df(df) # Choose between df and df_unique (the latter has no repeated measures)
df = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df)
df.IDidx = get_idx(:ID, df)[1]

# Set up intervention: post-intervention P, nP = 0 (no infection)
df.post_P .= 0
df.post_nP .= 0

# Restrict to unique cases (no repeated measures)
df_unique = encode_df(df_unique)
df_unique = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df_unique)

# Build DAG DataFrame - use select for efficiency
dag_df = select(df, :E, :H, :V, :D, :R, :S, :M, :Ḟ, :P, :nP, :ID, :IDidx, :vax_history, :Vidx, :post_P, :post_nP)
dag_df.lognP = log10.(1 .+ dag_df.nP)

# Select only infected mice for counterfactual analysis - use view for memory efficiency
infected_mask = dag_df.P .== 2
dag_df_infected = dag_df[infected_mask, :] # Keep as copy since we'll modify it

# Re-index IDidx for infected mice - optimized version
unique_ids_infected = unique(dag_df_infected.ID)
n_infected_ids = length(unique_ids_infected)
id_mapping = Dict{eltype(unique_ids_infected),Int}()  # Type-stable dictionary
for (i, id) in enumerate(unique_ids_infected)
    id_mapping[id] = i
end
dag_df_infected.IDidx_infected = [id_mapping[id] for id in dag_df_infected.ID]

## DAG specification

# Original DAG - includes P→E pathway
dag = dagitty("dag{ D -> E; D -> F; D -> M; D -> P; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> P; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; R -> P; S -> E; S -> F; S -> M; S -> P; S -> R; V -> E; V -> F; V -> M; V -> P; V -> R; }")

# Post-intervention DAG - {Pa} -> P -> E pathway removed
dag_m = dagitty("dag{ D -> E; D -> F; D -> M; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; S -> E; S -> F; S -> M; S -> R; V -> E; V -> F; V -> M; V -> R; }")

## Preliminary analysis: Effect of P on E

# Check adjustment sets
adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V}
adjustmentSets(dag_m, "P", "E", effect="total") # { }

# Mixed models to verify P→E relationship exists and is eliminated post-intervention
glmm_P_E_factual = fit(MixedModel, @formula(E ~ 1 + lognP + D + R + S + V + (1 | ID)), dag_df_infected)
glmm_P_E_counterfactual = fit(MixedModel, @formula(E ~ 1 + post_nP + (1 | ID)), dag_df_infected)

# Diagnostics for P→E models
qqnorm(glmm_P_E_factual; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E_factual)
coefplot(boot)

## Bayesian generative models for counterfactual inference

"""
    Counterfactual_E_Model(IDidx, E, V, D, Ḟ, H, M, P, R, S, nP, post_P, post_nP;
                          n_id=length(unique(IDidx)), intervention=false)

A Bayesian generative model for vaccine response (E) that enables counterfactual
inference under parasite elimination interventions.

This model implements the structural causal model framework by:
1. Pre-intervention: Models E as function of all relevant causes including P
2. Post-intervention: Models E under do(P=0) by removing P→E pathway
3. Counterfactual queries: Enables comparison between factual and counterfactual worlds

# Arguments
- `IDidx::Vector{Int}`: Mouse identifier indices
- `E::Vector{Float64}`: Standardised vaccine response (outcome)
- `V::Vector{Int}`: Vaccination status (1=adjuvant, 2=vaccine)
- `D::Vector{Int}`: Diet supplementation (1=low, 2=high)
- `Ḟ::Vector{Union{Missing,Float64}}`: Standardised fat scores (with missingness)
- `H::Vector{Int}`: Habitat (1=lab, 2=wild)
- `M::Vector{Float64}`: Standardised mass
- `P::Vector{Int}`: Parasite infection status (1=uninfected, 2=infected)
- `R::Vector{Int}`: Reproductive status (1=non-reproductive, 2=reproductive)
- `S::Vector{Int}`: Sex (1=male, 2=female)
- `nP::Vector{Float64}`: Parasite count (continuous)
- `post_P::Vector{Int}`: Post-intervention parasite status (always 0)
- `post_nP::Vector{Float64}`: Post-intervention parasite count (always 0)
- `n_id::Int`: Number of unique individuals
- `intervention::Bool`: Whether to model post-intervention structure

# Returns
- Samples from the posterior distribution of structural parameters
- Enables counterfactual queries via intervention programme transformations

# Model Structure
# Pre-intervention (original DAG):
- E ~ f(V, D, Ḟ, H, M, P, R, S, α_ID, ε_E)
- All pathways including P → E are active

# Post-intervention (manipulated DAG dag_m):
- E ~ f(V, D, Ḟ, H, M, R, S, α_ID, ε_E) [P pathway severed]
- P is set to 0 exogenously, breaking its causal influence on E

# Causal Assumptions
- Modularity: Each structural equation can be modified independently
- No unmeasured confounding given DAG structure
- SUTVA within individuals (but structured confounding via random effects)
"""
@model function Counterfactual_E_Model(IDidx, E, V, D, Ḟ, H, M, P, R, S, nP, post_P, post_nP;
    n_id=length(unique(IDidx)), intervention=false)

    # Pre-compute statistics for type stability
    E_mean = mean(E)
    E_std = std(E)

    # Population-level priors (weakly informative for standardised outcome)
    α ~ Normal(0, 1)  # Overall intercept - allows reasonable deviation from 0

    # Structural coefficients - weakly informative priors for standardised predictors/outcome
    βV ~ Normal(0, 0.5)   # Vaccination effect - allows small to large effect sizes
    βD ~ Normal(0, 0.5)   # Diet effect
    βḞ ~ Normal(0, 0.5)   # Fat effect
    βH ~ Normal(0, 1)     # Habitat effect - allow larger effect (lab vs wild)
    βM ~ Normal(0, 0.5)   # Mass effect
    βR ~ Normal(0, 0.5)   # Reproductive status effect
    βS ~ Normal(0, 0.5)   # Sex effect

    # Parasite effect - key parameter that changes under intervention
    if intervention
        # Post-intervention: P has no causal effect on E (pathway severed)
        βP ~ Normal(0, 0.001)  # Essentially constrained to 0
    else
        # Pre-intervention: P can causally influence E - allow moderate to large effect
        βP ~ Normal(0, 0.75)
    end

    # Residual variance - weakly informative for standardised outcome
    σ ~ Exponential(1)  # Allows reasonable residual variation

    # Random effects - weakly informative
    τ ~ Exponential(1)  # Individual-level SD - allows moderate between-individual variation
    α_ID ~ filldist(Normal(0, τ), n_id)   # Individual random intercepts

    # Handle missing fat scores with imputation
    N_missing = sum(ismissing.(Ḟ))
    if N_missing > 0
        F_impute ~ filldist(Normal(0, 1), N_missing)
        ν_F ~ Normal(0, 0.5)  # Imputation mean
        σ_F ~ Exponential(1)   # Imputation SD
    end

    # Likelihood for each observation
    i_missing = 1
    for i in eachindex(E)
        # Handle missing fat scores
        if ismissing(Ḟ[i])
            if N_missing > 0
                # Use the pre-sampled imputed value, applying transformation
                f_val = ν_F + σ_F * F_impute[i_missing]
                i_missing += 1
            else
                f_val = 0.0  # Fallback
            end
        else
            if N_missing > 0
                Ḟ[i] ~ Normal(ν_F, σ_F)
            end
            f_val = Ḟ[i]
        end

        # Structural equation for E
        if intervention
            # Post-intervention: use post_P (which should be 0) and remove P effect
            μ_E = α + α_ID[IDidx[i]] + βV * V[i] + βD * D[i] + βḞ * f_val +
                  βH * H[i] + βM * M[i] + βR * R[i] + βS * S[i]
            # Note: βP * post_P[i] would be ≈ 0 anyway since post_P[i] = 0
        else
            # Pre-intervention: use observed P and include P effect
            μ_E = α + α_ID[IDidx[i]] + βV * V[i] + βD * D[i] + βḞ * f_val +
                  βH * H[i] + βM * M[i] + βP * P[i] + βR * R[i] + βS * S[i]
        end

        E[i] ~ Normal(μ_E, σ)
    end
end

"""
    Twin_World_E_Model(IDidx, E_factual, E_counterfactual, V, D, Ḟ, H, M, P, R, S,
                       post_P, post_nP; n_id=length(unique(IDidx)))

A joint generative model for both factual and counterfactual vaccine responses,
implementing the "twin world" approach to counterfactual inference.

This model jointly samples both the observed world (with parasites) and the
counterfactual world (without parasites), sharing individual-level random effects
to maintain the identity of individuals across worlds.

# Arguments
- `E_factual::Vector{Float64}`: Observed vaccine response
- `E_counterfactual::Vector{Float64}`: Counterfactual vaccine response (under do(P=0))
- Other arguments as in `Counterfactual_E_Model`

# Returns
- Joint posterior over factual and counterfactual parameters
- Enables direct estimation of individual treatment effects
"""
@model function Twin_World_E_Model(IDidx, E_factual, E_counterfactual, V, D, Ḟ, H, M, P, R, S,
    post_P, post_nP; n_id=length(unique(IDidx)))

    # Pre-compute statistics for type stability
    E_mean = mean(E_factual)
    E_std = std(E_factual)

    # Shared population-level parameters (weakly informative)
    α ~ Normal(0, 1)  # Standardised outcome

    # Structural coefficients (shared across worlds) - weakly informative priors
    βV ~ Normal(0, 0.5)
    βD ~ Normal(0, 0.5)
    βḞ ~ Normal(0, 0.5)
    βH ~ Normal(0, 1)     # Allow larger habitat effect
    βM ~ Normal(0, 0.5)
    βP ~ Normal(0, 0.75)  # Allow moderate to large parasite effect in factual world
    βR ~ Normal(0, 0.5)
    βS ~ Normal(0, 0.5)

    # Shared residual variance - weakly informative
    σ ~ Exponential(1)

    # Shared random effects - weakly informative
    τ ~ Exponential(1)
    α_ID ~ filldist(Normal(0, τ), n_id)  # Same random intercepts in both worlds

    # Fat score imputation (shared parameters)
    N_missing = sum(ismissing.(Ḟ))
    if N_missing > 0
        F_impute ~ filldist(Normal(0, 1), N_missing)
        ν_F ~ Normal(0, 0.5)
        σ_F ~ Exponential(1)
    end

    # Likelihood for both factual and counterfactual observations
    i_missing = 1
    for i in eachindex(E_factual)
        # Handle missing fat scores
        if ismissing(Ḟ[i])
            if N_missing > 0
                # Use the pre-sampled imputed value, applying transformation
                f_val = ν_F + σ_F * F_impute[i_missing]
                i_missing += 1
            else
                f_val = 0.0
            end
        else
            if N_missing > 0
                Ḟ[i] ~ Normal(ν_F, σ_F)
            end
            f_val = Ḟ[i]
        end

        # Factual world (with parasites)
        μ_E_factual = α + α_ID[IDidx[i]] + βV * V[i] + βD * D[i] + βḞ * f_val +
                      βH * H[i] + βM * M[i] + βP * P[i] + βR * R[i] + βS * S[i]
        E_factual[i] ~ Normal(μ_E_factual, σ)

        # Counterfactual world (without parasites: P = 0)
        μ_E_counterfactual = α + α_ID[IDidx[i]] + βV * V[i] + βD * D[i] + βḞ * f_val +
                             βH * H[i] + βM * M[i] + βR * R[i] + βS * S[i]
        # Note: no βP term - this implements do(P=0)
        E_counterfactual[i] ~ Normal(μ_E_counterfactual, σ)
    end
end

## Prior predictive checks

# PPC functions moved to TuringUtils.jl for reusability

# Prior rationale: Weakly informative priors for standardised outcome (E)

# Generate prior predictive checks using improved approach
println("=== PRIOR PREDICTIVE CHECKS FOR INTERVENTION MODELS ===")

# Factual model (with parasites)
println("Checking priors for factual model...")
factual_model = Counterfactual_E_Model(
    dag_df_infected.IDidx_infected, dag_df_infected.E, dag_df_infected.V,
    dag_df_infected.D, dag_df_infected.Ḟ, dag_df_infected.H,
    dag_df_infected.M, dag_df_infected.P, dag_df_infected.R,
    dag_df_infected.S, dag_df_infected.nP, dag_df_infected.post_P,
    dag_df_infected.post_nP; n_id=n_infected_ids, intervention=false
)

factual_prior_samples = sample(factual_model, Prior(), 1000)
factual_prior_E = vec(Array(factual_prior_samples)[:, end])  # Extract E predictions

# Counterfactual model (without parasites)
println("Checking priors for counterfactual model...")
counterfactual_model = Counterfactual_E_Model(
    dag_df_infected.IDidx_infected, dag_df_infected.E, dag_df_infected.V,
    dag_df_infected.D, dag_df_infected.Ḟ, dag_df_infected.H,
    dag_df_infected.M, dag_df_infected.post_P, dag_df_infected.R,
    dag_df_infected.S, dag_df_infected.post_nP, dag_df_infected.post_P,
    dag_df_infected.post_nP; n_id=n_infected_ids, intervention=true
)

counterfactual_prior_samples = sample(counterfactual_model, Prior(), 1000)
counterfactual_prior_E = vec(Array(counterfactual_prior_samples)[:, end])  # Extract E predictions

# Create plots and assessments using TuringPlots functions
with_theme(theme_minimal()) do
    plot_prior_predictive_check(dag_df_infected.E, factual_prior_E;
        title_suffix="Factual Model", saveplot=true)

    plot_prior_predictive_check(dag_df_infected.E, counterfactual_prior_E;
        title_suffix="Counterfactual Model", saveplot=true)
end

# Assess prior adequacy
assess_prior_adequacy(dag_df_infected.E, factual_prior_E; model_name="Factual (With Parasites)")
assess_prior_adequacy(dag_df_infected.E, counterfactual_prior_E; model_name="Counterfactual (Without Parasites)")

## Model fitting

# Fit the pre-intervention model (reuse the factual_model from PPC)
println("\nFitting pre-intervention model...")
pre_intervention_chain = sample(factual_model, NUTS(), MCMCThreads(), 3000, 4)

# Fit the post-intervention model (reuse the counterfactual_model from PPC)
println("Fitting post-intervention model...")
post_intervention_chain = sample(counterfactual_model, NUTS(), MCMCThreads(), 3000, 4)

## Generate counterfactual predictions

"""
    generate_counterfactuals(pre_chain, df_infected; n_samples=1000)

Generate counterfactual vaccine responses under the intervention do(P=0).
Optimized version for better performance and type stability.

# Arguments
- `pre_chain::MCMCChains.Chains`: Posterior samples from pre-intervention model
- `df_infected::DataFrame`: Data for infected mice
- `n_samples::Int`: Number of posterior samples to use

# Returns
- `E_factual::Matrix{Float64}`: Factual vaccine responses (n_obs × n_samples)
- `E_counterfactual::Matrix{Float64}`: Counterfactual vaccine responses (n_obs × n_samples)
"""
function generate_counterfactuals(pre_chain, df_infected; n_samples::Int=1000)
    n_obs = nrow(df_infected)
    chain_samples = DataFrame(pre_chain)
    n_chain_samples = nrow(chain_samples)

    # Pre-allocate with specific types for better performance
    E_factual = Matrix{Float64}(undef, n_obs, n_samples)
    E_counterfactual = Matrix{Float64}(undef, n_obs, n_samples)

    # Sample from posterior
    sample_indices = rand(1:n_chain_samples, n_samples)

    # Pre-compute unique infected count to avoid repeated computation
    n_unique_infected = length(unique(df_infected.ID))

    # Extract data vectors once for efficiency
    V_vec = df_infected.V
    D_vec = df_infected.D
    H_vec = df_infected.H
    M_vec = df_infected.M
    P_vec = df_infected.P
    R_vec = df_infected.R
    S_vec = df_infected.S
    IDidx_vec = dag_df_infected.IDidx_infected
    Ḟ_vec = df_infected.Ḟ

    for (s, idx) in enumerate(sample_indices)
        # Extract parameters (type stable)
        α::Float64 = chain_samples.α[idx]
        βV::Float64 = chain_samples.βV[idx]
        βD::Float64 = chain_samples.βD[idx]
        βḞ::Float64 = chain_samples.βḞ[idx]
        βH::Float64 = chain_samples.βH[idx]
        βM::Float64 = chain_samples.βM[idx]
        βP::Float64 = chain_samples.βP[idx]
        βR::Float64 = chain_samples.βR[idx]
        βS::Float64 = chain_samples.βS[idx]
        σ::Float64 = chain_samples.σ[idx]

        # Extract random effects - pre-allocate for efficiency
        α_ID = Vector{Float64}(undef, n_unique_infected)
        for i in 1:n_unique_infected
            α_ID[i] = chain_samples[idx, Symbol("α_ID[$i]")]
        end

        # Vectorized computation where possible
        for i in 1:n_obs
            # Handle missing fat scores efficiently
            f_val::Float64 = ismissing(Ḟ_vec[i]) ? 0.0 : Ḟ_vec[i]

            # Pre-compute common terms
            base_μ = α + α_ID[IDidx_vec[i]] + βV * V_vec[i] + βD * D_vec[i] +
                     βḞ * f_val + βH * H_vec[i] + βM * M_vec[i] + βR * R_vec[i] + βS * S_vec[i]

            # Factual prediction (with parasites)
            μ_factual = base_μ + βP * P_vec[i]
            E_factual[i, s] = μ_factual + σ * randn()

            # Counterfactual prediction (without parasites: P = 0)
            E_counterfactual[i, s] = base_μ + σ * randn()
        end
    end

    return E_factual, E_counterfactual
end

# Generate counterfactual predictions
println("Generating counterfactual predictions...")
E_factual, E_counterfactual = generate_counterfactuals(pre_intervention_chain, dag_df_infected)

# Add predictions to dataframe - using views for efficiency where possible
dag_df_infected.E_factual_mean = vec(mean(E_factual, dims=2))
dag_df_infected.E_counterfactual_mean = vec(mean(E_counterfactual, dims=2))
dag_df_infected.E_diff_counterfactual = dag_df_infected.E_counterfactual_mean .- dag_df_infected.E_factual_mean

# Calculate uncertainty measures
dag_df_infected.E_factual_sd = vec(std(E_factual, dims=2))
dag_df_infected.E_counterfactual_sd = vec(std(E_counterfactual, dims=2))

# Calculate Cohen's d effect size for standardised assessment
dag_df_infected.E_cohens_d = begin
    # Pooled standard deviation for Cohen's d calculation
    pooled_sd = sqrt.((dag_df_infected.E_factual_sd .^ 2 .+ dag_df_infected.E_counterfactual_sd .^ 2) ./ 2)
    # Avoid division by very small numbers
    safe_pooled_sd = max.(pooled_sd, 0.1)  # Minimum threshold for stability
    dag_df_infected.E_diff_counterfactual ./ safe_pooled_sd
end

# Define clinical significance categories based on Cohen's d thresholds
dag_df_infected.E_clinical_significance = begin
    abs_cohens_d = abs.(dag_df_infected.E_cohens_d)
    ifelse.(abs_cohens_d .< 0.2, "Negligible",
        ifelse.(abs_cohens_d .< 0.5, "Small",
            ifelse.(abs_cohens_d .< 0.8, "Moderate", "Large")))
end

# Clinical significance with direction
dag_df_infected.E_clinical_direction = begin
    cohens_d = dag_df_infected.E_cohens_d
    abs_cohens_d = abs.(cohens_d)
    direction = ifelse.(cohens_d .>= 0, "Beneficial", "Detrimental")
    magnitude = ifelse.(abs_cohens_d .< 0.2, "Negligible",
        ifelse.(abs_cohens_d .< 0.5, "Small",
            ifelse.(abs_cohens_d .< 0.8, "Moderate", "Large")))
    magnitude .* " " .* direction
end

println("Summary of counterfactual effects:")
println("Mean counterfactual effect (E_counterfactual - E_factual): ",
    round(mean(dag_df_infected.E_diff_counterfactual), digits=3))
println("SD of counterfactual effects: ",
    round(std(dag_df_infected.E_diff_counterfactual), digits=3))
println("Mean Cohen's d: ", round(mean(dag_df_infected.E_cohens_d), digits=3))
println("SD Cohen's d: ", round(std(dag_df_infected.E_cohens_d), digits=3))

# Clinical significance summary
println("\nClinical significance distribution:")
for category in ["Negligible", "Small", "Moderate", "Large"]
    count = sum(dag_df_infected.E_clinical_significance .== category)
    percentage = round(100 * count / nrow(dag_df_infected), digits=1)
    println("$category: $count mice ($percentage%)")
end

# Summary diagnostics
precis(DataFrame(pre_intervention_chain)[!, r"α\b|β"])
println("\nPre-intervention model diagnostics:")
plot_chains_df(pre_intervention_chain; show_traces=false)

## Causal effect analysis using generative model predictions

# Check adjustment sets for vaccination effect
adjustmentSets(dag, "V", "E", effect="total") # { }
adjustmentSets(dag_m, "V", "E", effect="total") # { }

# Total effect of V on E (factual vs counterfactual)
glmm_V_E_total_factual = fit(MixedModel, @formula(E_factual_mean ~ V + (1 | ID)), dag_df_infected)
glmm_V_E_total_counterfactual = fit(MixedModel, @formula(E_counterfactual_mean ~ V + (1 | ID)), dag_df_infected)

# Diagnostics for vaccination effect model
qqnorm(glmm_V_E_total_counterfactual; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_total_counterfactual)
coefplot(boot)
ridgeplot(boot)

# Check adjustment sets for direct effects
adjustmentSets(dag, "V", "E", effect="direct") # { D, F, H, M, P, R, S }
adjustmentSets(dag_m, "V", "E", effect="direct") # { D, F, H, M, P, R, S }

# Direct effect of V on E (factual vs counterfactual)
glmm_V_E_direct_factual = fit(MixedModel, @formula(E_factual_mean ~ V + D + Ḟ + H + M + P + R + S + (1 | ID)), dag_df_infected)
glmm_V_E_direct_counterfactual = fit(MixedModel, @formula(E_counterfactual_mean ~ V + D + Ḟ + H + M + R + S + (1 | ID)), dag_df_infected)

# Diagnostics for direct vaccination effect model
qqnorm(glmm_V_E_direct_counterfactual; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_direct_counterfactual)
coefplot(boot)

## Generate enhanced plots with clinical significance

with_theme(theme_minimal()) do
    plot_E_factual_counterfactual(dag_df_infected, saveplot=true)
end

with_theme(theme_minimal()) do
    plot_counterfactual_effects_with_significance(dag_df_infected, saveplot=true)
end

with_theme(theme_minimal()) do
    plot_clinical_significance_summary(dag_df_infected, saveplot=true)
end

## Interaction analysis with Cohen's d

# Fit model using Cohen's d as outcome
glmm_cohens_d_interaction = fit(MixedModel, @formula(E_cohens_d ~ -1 + D + R + S + V + M + Ḟ + S & R + (1 | ID)), dag_df_infected)

# Generate Cohen's d interaction plots
with_theme(theme_minimal()) do
    plot_S_R_interaction_cohens_d(dag_df_infected, saveplot=true)
end

with_theme(theme_minimal()) do
    plot_S_R_interaction_factual_counterfactual(dag_df_infected, saveplot=true)
end

# Generate detailed Sex × Reproductive status interaction plots
with_theme(theme_minimal()) do
    plot_sex_reproductive_interaction_detailed(dag_df_infected, saveplot=true)
end

with_theme(theme_minimal()) do
    plot_sex_reproductive_heatmap(dag_df_infected, saveplot=true)
end

# Diagnostics for Cohen's d interaction model
boot_cohens_d = parametricbootstrap(MersenneTwister(1234), 10_000, glmm_cohens_d_interaction)
coefplot(boot_cohens_d)
