#=
SCM Intervention with Sex × Reproductive Status Interaction
- Julia version: 1.11
- Author: Simon A Babayan
- Enhanced version with S×R interaction terms
=#

#=
This script implements counterfactual inference under parasite elimination interventions
using structural causal models with Sex × Reproductive Status interactions. It compares
vaccine efficacy in factual (with parasites) vs counterfactual (without parasites)
scenarios using Bayesian generative models that account for differential effects across
sex-reproductive status combinations.

NOT USED IN THE MANUSCRIPT DUE TO SAMPLE SIZE LIMITATIONS.
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

## Configuration constants

# Set default plotting behaviour - change to false to disable plot saving
const SAVE_PLOTS = true

# Performance configuration following Turing.jl best practices
# For models with many parameters (>20), consider AutoZygote() or AutoReverseDiff()
# For models with few parameters (<20), AutoForwardDiff() is typically fastest
const AD_BACKEND = nothing  # Use default (AutoForwardDiff) for our model size

# Interaction coding method - set to true for effect coding (recommended)
const USE_EFFECT_CODING = true

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

# NEW: Add effect-coded variables for interaction analysis
if USE_EFFECT_CODING
    # Effect coding: Male=-1, Female=1; Non-reproductive=-1, Reproductive=1
    dag_df_infected.S_coded = ifelse.(dag_df_infected.S .== 1, -1, 1)  # Male=-1, Female=1
    dag_df_infected.R_coded = ifelse.(dag_df_infected.R .== 1, -1, 1)  # Non-repro=-1, Repro=1
    dag_df_infected.SR_interaction = dag_df_infected.S_coded .* dag_df_infected.R_coded

    println("Effect coding applied:")
    println("S_coded: Male=-1, Female=1")
    println("R_coded: Non-reproductive=-1, Reproductive=1")
    println("SR_interaction: ", unique(dag_df_infected.SR_interaction))
else
    # Raw coding for comparison
    dag_df_infected.SR_interaction = dag_df_infected.S .* dag_df_infected.R
    println("Raw interaction coding used (S × R)")
end

# Print interaction group summary
println("\nSex × Reproductive Status Groups:")
group_summary = combine(groupby(dag_df_infected, [:S, :R]), nrow => :count)
for row in eachrow(group_summary)
    sex_label = row.S == 1 ? "Male" : "Female"
    repro_label = row.R == 1 ? "Non-reproductive" : "Reproductive"
    println("$sex_label, $repro_label: $(row.count) mice")
end

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

# Test interaction in traditional mixed model for comparison
if USE_EFFECT_CODING
    glmm_P_E_interaction = fit(MixedModel, @formula(E ~ 1 + lognP + D + S_coded + R_coded + S_coded & R_coded + V + (1 | ID)), dag_df_infected)
else
    glmm_P_E_interaction = fit(MixedModel, @formula(E ~ 1 + lognP + D + S + R + S & R + V + (1 | ID)), dag_df_infected)
end

# Diagnostics for P→E models
qqnorm(glmm_P_E_factual; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E_factual)
coefplot(boot)

## Bayesian generative models for counterfactual inference with interaction

"""
    Counterfactual_E_Model_WithInteraction(IDidx, E, V, D, Ḟ, H, M, P, R, S, nP, post_P, post_nP;
                          n_id=length(unique(IDidx)), intervention=false, use_effect_coding=true)

Bayesian structural causal model for vaccine response with Sex × Reproductive Status interaction,
enabling counterfactual inference under parasite elimination.

This enhanced model implements the structural causal framework by:
1. Pre-intervention: Models E as function of all relevant causes including P and S×R interaction
2. Post-intervention: Models E under do(P=0) by removing P→E pathway while preserving interactions
3. Counterfactual queries: Enables comparison between factual and counterfactual worlds with interaction effects

# Arguments
- `IDidx`: Mouse identifier indices
- `E`: Standardised vaccine response (outcome)
- `V`: Vaccination status (1=adjuvant, 2=vaccine)
- `D`: Diet supplementation (1=low, 2=high)
- `Ḟ`: Standardised fat scores (with missingness)
- `H`: Habitat (1=lab, 2=wild)
- `M`: Standardised mass
- `P`: Parasite infection status (1=uninfected, 2=infected)
- `R`: Reproductive status (1=non-reproductive, 2=reproductive)
- `S`: Sex (1=male, 2=female)
- `intervention`: Whether to model post-intervention structure (P pathway removed)
- `use_effect_coding`: Whether to use effect coding (-1,1) instead of raw coding (1,2)

# Model Structure with Interaction
- Factual: E ~ f(V, D, Ḟ, H, M, P, R, S, S×R, α_ID) + ε
- Counterfactual: E ~ f(V, D, Ḟ, H, M, R, S, S×R, α_ID) + ε [P pathway severed]

# Interaction Interpretation (Effect Coding)
- βSR > 0: Female reproductive mice show enhanced response beyond additive effects
- βSR < 0: Female reproductive mice show diminished response
- βSR ≈ 0: No interaction - effects are purely additive

# Assumptions
- Modularity: Each structural equation can be modified independently
- No unmeasured confounding given DAG structure
- Individual-level confounding handled via random effects
- Interaction effects are consistent across intervention scenarios
"""
@model function Counterfactual_E_Model_WithInteraction(IDidx, E, V, D, Ḟ, H, M, P, R, S, nP, post_P, post_nP;
    n_id=length(unique(IDidx)), intervention=false, use_effect_coding=true)

    # Pre-compute statistics for type stability and performance
    N::Int = length(E)
    E_mean::Float64 = mean(E)
    E_std::Float64 = std(E)

    # Population-level priors (weakly informative for standardised outcome)
    α ~ Normal(0, 1)  # Overall intercept - allows reasonable deviation from 0

    # Main effect structural coefficients - weakly informative priors for standardised predictors/outcome
    βV ~ Normal(0, 0.5)   # Vaccination effect - allows small to large effect sizes
    βD ~ Normal(0, 0.5)   # Diet effect
    βḞ ~ Normal(0, 0.5)   # Fat effect
    βH ~ Normal(0, 1)     # Habitat effect - allow larger effect (lab vs wild)
    βM ~ Normal(0, 0.5)   # Mass effect
    βR ~ Normal(0, 0.5)   # Reproductive status main effect
    βS ~ Normal(0, 0.5)   # Sex main effect

    # NEW: Sex × Reproductive status interaction - smaller prior SD than main effects
    βSR ~ Normal(0, 0.3)  # Interaction effect - typically smaller than main effects

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

    # Random effects - weakly informative, using filldist for performance
    τ ~ Exponential(1)  # Individual-level SD - allows moderate between-individual variation
    α_ID ~ filldist(Normal(0, τ), n_id)   # Individual random intercepts

    # Pre-compute missing data information for efficiency
    missing_mask = ismissing.(Ḟ)
    N_missing::Int = sum(missing_mask)

    if N_missing > 0
        missing_indices = findall(missing_mask)
        F_impute ~ filldist(Normal(0, 1), N_missing)
        ν_F ~ Normal(0, 0.5)  # Imputation mean
        σ_F ~ Exponential(1)  # Imputation SD

        # Handle missing fat values by imputation
        f_vals = map(i -> missing_mask[i] ? (i in missing_indices ? ν_F + σ_F * F_impute[findfirst(==(i), missing_indices)] : zero(eltype(α))) : Ḟ[i], 1:N)

        # Add likelihood for observed fat values
        for i in 1:N
            if !missing_mask[i]
                Ḟ[i] ~ Normal(ν_F, σ_F)
            end
        end

        # Compute base effects for all individuals INCLUDING INTERACTION
        if use_effect_coding
            # Effect-coded interaction (recommended for interpretability)
            S_coded = @. ifelse(S == 1, -1, 1)  # Male=-1, Female=1
            R_coded = @. ifelse(R == 1, -1, 1)  # Non-repro=-1, Repro=1
            SR_interact = @. S_coded * R_coded

            base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M +
                              βR * R_coded + βS * S_coded + βSR * SR_interact
        else
            # Raw interaction with original coding
            SR_interact = @. S * R
            base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M +
                              βR * R + βS * S + βSR * SR_interact
        end

        # Main likelihood loop
        for i in 1:N
            μ_E = base_effects[i] + α_ID[IDidx[i]]

            # Add parasite effect only in factual (pre-intervention) scenario
            if !intervention
                μ_E += βP * P[i]
            end

            E[i] ~ Normal(μ_E, σ)
        end
    else
        # No missing data case
        f_vals = [ismissing(f) ? zero(eltype(α)) : f for f in Ḟ]

        if use_effect_coding
            S_coded = [s == 1 ? -1 : 1 for s in S]
            R_coded = [r == 1 ? -1 : 1 for r in R]
            SR_interact = [S_coded[i] * R_coded[i] for i in 1:N]

            base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M +
                              βR * R_coded + βS * S_coded + βSR * SR_interact
        else
            SR_interact = [S[i] * R[i] for i in 1:N]
            base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M +
                              βR * R + βS * S + βSR * SR_interact
        end

        for i in 1:N
            μ_E = base_effects[i] + α_ID[IDidx[i]]

            # Add parasite effect only in factual (pre-intervention) scenario
            if !intervention
                μ_E += βP * P[i]
            end

            E[i] ~ Normal(μ_E, σ)
        end
    end
end


"""
    Twin_World_E_Model_WithInteraction(IDidx, E_factual, E_counterfactual, V, D, Ḟ, H, M, P, R, S,
                       post_P, post_nP; n_id=length(unique(IDidx)), use_effect_coding=true)

Joint model for factual and counterfactual vaccine responses with Sex × Reproductive Status interaction
using the "twin world" approach.

This model simultaneously samples both the observed world (with parasites) and
counterfactual world (without parasites), sharing individual-level random effects
to maintain individual identity across worlds while accounting for interaction effects.

# Arguments
- `E_factual`: Observed vaccine response
- `E_counterfactual`: Counterfactual vaccine response (under do(P=0))
- `use_effect_coding`: Whether to use effect coding for interaction terms
- Other arguments as in `Counterfactual_E_Model_WithInteraction`

# Returns
Joint posterior enabling direct estimation of individual treatment effects with interaction.
"""
@model function Twin_World_E_Model_WithInteraction(IDidx, E_factual, E_counterfactual, V, D, Ḟ, H, M, P, R, S,
    post_P, post_nP; n_id=length(unique(IDidx)), use_effect_coding=true)

    # Pre-compute statistics for type stability and performance
    N::Int = length(E_factual)
    E_mean::Float64 = mean(E_factual)
    E_std::Float64 = std(E_factual)

    # Shared population-level parameters (weakly informative)
    α ~ Normal(0, 1)  # Standardised outcome

    # Structural coefficients (shared across worlds) - weakly informative priors
    βV ~ Normal(0, 0.5)
    βD ~ Normal(0, 0.5)
    βḞ ~ Normal(0, 0.5)
    βH ~ Normal(0, 1)     # Allow larger habitat effect
    βM ~ Normal(0, 0.5)
    βP ~ Normal(0, 0.75)  # Allow moderate to large parasite effect in factual world
    βR ~ Normal(0, 0.5)   # Reproductive status main effect
    βS ~ Normal(0, 0.5)   # Sex main effect
    βSR ~ Normal(0, 0.3)  # Sex × Reproductive status interaction

    # Shared residual variance - weakly informative
    σ ~ Exponential(1)

    # Shared random effects - weakly informative, using filldist for performance
    τ ~ Exponential(1)
    α_ID ~ filldist(Normal(0, τ), n_id)  # Same random intercepts in both worlds

    # Pre-compute missing data information for efficiency
    missing_mask = ismissing.(Ḟ)
    N_missing::Int = sum(missing_mask)

    if N_missing > 0
        missing_indices = findall(missing_mask)
        F_impute ~ filldist(Normal(0, 1), N_missing)
        ν_F ~ Normal(0, 0.5)
        σ_F ~ Exponential(1)

        # Handle missing fat values by imputation
        f_vals = map(i -> missing_mask[i] ? (i in missing_indices ? ν_F + σ_F * F_impute[findfirst(==(i), missing_indices)] : zero(eltype(α))) : Ḟ[i], 1:N)

        # Add likelihood for observed fat values
        for i in 1:N
            if !missing_mask[i]
                Ḟ[i] ~ Normal(ν_F, σ_F)
            end
        end

        # Compute base effects for all individuals INCLUDING INTERACTION
        if use_effect_coding
            S_coded = @. ifelse(S == 1, -1, 1)  # Male=-1, Female=1
            R_coded = @. ifelse(R == 1, -1, 1)  # Non-repro=-1, Repro=1
            SR_interact = @. S_coded * R_coded

            base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M +
                              βR * R_coded + βS * S_coded + βSR * SR_interact
        else
            SR_interact = @. S * R
            base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M +
                              βR * R + βS * S + βSR * SR_interact
        end

        # Likelihood for both factual and counterfactual observations
        for i in 1:N
            μ_base = base_effects[i] + α_ID[IDidx[i]]

            # Factual world (with parasites)
            μ_E_factual = μ_base + βP * P[i]
            E_factual[i] ~ Normal(μ_E_factual, σ)

            # Counterfactual world (without parasites)
            E_counterfactual[i] ~ Normal(μ_base, σ)
        end
    else
        # No missing data case
        f_vals = [ismissing(f) ? zero(eltype(α)) : f for f in Ḟ]

        if use_effect_coding
            S_coded = [s == 1 ? -1 : 1 for s in S]
            R_coded = [r == 1 ? -1 : 1 for r in R]
            SR_interact = [S_coded[i] * R_coded[i] for i in 1:N]

            base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M +
                              βR * R_coded + βS * S_coded + βSR * SR_interact
        else
            SR_interact = [S[i] * R[i] for i in 1:N]
            base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M +
                              βR * R + βS * S + βSR * SR_interact
        end

        for i in 1:N
            μ_base = base_effects[i] + α_ID[IDidx[i]]

            # Factual world (with parasites)
            μ_E_factual = μ_base + βP * P[i]
            E_factual[i] ~ Normal(μ_E_factual, σ)

            # Counterfactual world (without parasites)
            E_counterfactual[i] ~ Normal(μ_base, σ)
        end
    end
end


## Prior predictive checks

# Generate prior predictive checks to assess model priors including interaction
println("=== PRIOR PREDICTIVE CHECKS FOR INTERACTION MODELS ===")

# Test interaction prior adequacy
interaction_prior_samples = rand(Normal(0, 0.3), 10000)
println("Interaction prior (βSR) 95% interval: ", round.(quantile(interaction_prior_samples, [0.025, 0.975]), digits=3))
println("This represents interaction effects from $(round(quantile(interaction_prior_samples, 0.025), digits=2)) to $(round(quantile(interaction_prior_samples, 0.975), digits=2)) standardised units")

# Factual model with interaction (with parasites)
println("Checking priors for factual model with interaction...")
factual_model_interaction = Counterfactual_E_Model_WithInteraction(
    dag_df_infected.IDidx_infected, dag_df_infected.E, dag_df_infected.V,
    dag_df_infected.D, dag_df_infected.Ḟ, dag_df_infected.H,
    dag_df_infected.M, dag_df_infected.P, dag_df_infected.R,
    dag_df_infected.S, dag_df_infected.nP, dag_df_infected.post_P,
    dag_df_infected.post_nP; n_id=n_infected_ids, intervention=false, use_effect_coding=USE_EFFECT_CODING
)

factual_prior_samples_int = sample(factual_model_interaction, Prior(), 1000)
factual_prior_E_int = vec(Array(factual_prior_samples_int)[:, end])  # Extract E predictions

# Counterfactual model with interaction (without parasites)
println("Checking priors for counterfactual model with interaction...")
counterfactual_model_interaction = Counterfactual_E_Model_WithInteraction(
    dag_df_infected.IDidx_infected, dag_df_infected.E, dag_df_infected.V,
    dag_df_infected.D, dag_df_infected.Ḟ, dag_df_infected.H,
    dag_df_infected.M, dag_df_infected.post_P, dag_df_infected.R,
    dag_df_infected.S, dag_df_infected.post_nP, dag_df_infected.post_P,
    dag_df_infected.post_nP; n_id=n_infected_ids, intervention=true, use_effect_coding=USE_EFFECT_CODING
)

counterfactual_prior_samples_int = sample(counterfactual_model_interaction, Prior(), 1000)
counterfactual_prior_E_int = vec(Array(counterfactual_prior_samples_int)[:, end])  # Extract E predictions

# Create plots and assessments using TuringPlots functions
with_theme(theme_minimal()) do
    plot_prior_predictive_check(dag_df_infected.E, factual_prior_E_int;
        title_suffix="Factual Model with S×R Interaction", saveplot=SAVE_PLOTS)

    plot_prior_predictive_check(dag_df_infected.E, counterfactual_prior_E_int;
        title_suffix="Counterfactual Model with S×R Interaction", saveplot=SAVE_PLOTS)
end

# Assess prior adequacy
assess_prior_adequacy(dag_df_infected.E, factual_prior_E_int; model_name="Factual with Interaction (With Parasites)")
assess_prior_adequacy(dag_df_infected.E, counterfactual_prior_E_int; model_name="Counterfactual with Interaction (Without Parasites)")

## Model fitting

# Fit the pre-intervention model with interaction (reuse the factual_model_interaction from PPC)
println("\nFitting pre-intervention model with S×R interaction...")
if AD_BACKEND !== nothing
    pre_intervention_chain_interaction = sample(factual_model_interaction, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
    pre_intervention_chain_interaction = sample(factual_model_interaction, NUTS(), MCMCThreads(), 3000, 4)
end

# Fit the post-intervention model with interaction (reuse the counterfactual_model_interaction from PPC)
println("Fitting post-intervention model with S×R interaction...")
if AD_BACKEND !== nothing
    post_intervention_chain_interaction = sample(counterfactual_model_interaction, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
    post_intervention_chain_interaction = sample(counterfactual_model_interaction, NUTS(), MCMCThreads(), 3000, 4)
end

## Generate counterfactuals using posterior predictive approach with interaction (RECOMMENDED METHOD)

"""
    generate_counterfactuals_with_interaction(pre_chain, df_infected; n_samples=1000, use_effect_coding=true)

Generate counterfactual vaccine responses under parasite elimination intervention do(P=0)
with Sex × Reproductive Status interaction effects.

Uses posterior predictive sampling to generate both factual (with parasites) and
counterfactual (without parasites) vaccine responses for each individual, accounting
for interaction effects between sex and reproductive status.

# Arguments
- `pre_chain`: MCMC chain from fitted factual model with interaction
- `df_infected`: DataFrame with infected mouse data
- `n_samples`: Number of posterior samples to draw
- `use_effect_coding`: Whether to use effect coding for interaction terms

# Returns
- `E_factual`: Matrix of factual vaccine responses (n_obs × n_samples)
- `E_counterfactual`: Matrix of counterfactual vaccine responses (n_obs × n_samples)
"""
function generate_counterfactuals_with_interaction(pre_chain, df_infected; n_samples::Int=1000, use_effect_coding::Bool=true)
    n_obs = nrow(df_infected)
    chain_samples = DataFrame(pre_chain)
    n_chain_samples = nrow(chain_samples)

    # Pre-allocate with specific types for better performance
    E_factual = Matrix{Float64}(undef, n_obs, n_samples)
    E_counterfactual = Matrix{Float64}(undef, n_obs, n_samples)

    # Sample from posterior
    sample_indices = rand(1:n_chain_samples, n_samples)

    # Extract data vectors once for efficiency
    V_vec = df_infected.V
    D_vec = df_infected.D
    H_vec = df_infected.H
    M_vec = df_infected.M
    P_vec = df_infected.P
    R_vec = df_infected.R
    S_vec = df_infected.S
    IDidx_vec = df_infected.IDidx_infected
    Ḟ_vec = df_infected.Ḟ

    for (s, idx) in enumerate(sample_indices)
        # Extract parameters (type stable) INCLUDING INTERACTION
        α::Float64 = chain_samples.α[idx]
        βV::Float64 = chain_samples.βV[idx]
        βD::Float64 = chain_samples.βD[idx]
        βḞ::Float64 = chain_samples.βḞ[idx]
        βH::Float64 = chain_samples.βH[idx]
        βM::Float64 = chain_samples.βM[idx]
        βP::Float64 = chain_samples.βP[idx]
        βR::Float64 = chain_samples.βR[idx]
        βS::Float64 = chain_samples.βS[idx]
        βSR::Float64 = chain_samples.βSR[idx]  # NEW: Extract interaction coefficient
        σ::Float64 = chain_samples.σ[idx]

        # Extract random effects
        n_unique_infected = length(unique(df_infected.ID))
        α_ID = Vector{Float64}(undef, n_unique_infected)
        for i in 1:n_unique_infected
            α_ID[i] = chain_samples[idx, Symbol("α_ID[$i]")]
        end

        # Generate predictions
        for i in 1:n_obs
            f_val::Float64 = ismissing(Ḟ_vec[i]) ? 0.0 : Ḟ_vec[i]

            # Compute interaction term
            if use_effect_coding
                s_coded = S_vec[i] == 1 ? -1.0 : 1.0  # Male=-1, Female=1
                r_coded = R_vec[i] == 1 ? -1.0 : 1.0  # Non-repro=-1, Repro=1
                sr_interact = s_coded * r_coded

                base_μ = α + α_ID[IDidx_vec[i]] + βV * V_vec[i] + βD * D_vec[i] +
                         βḞ * f_val + βH * H_vec[i] + βM * M_vec[i] +
                         βR * r_coded + βS * s_coded + βSR * sr_interact
            else
                sr_interact = S_vec[i] * R_vec[i]
                base_μ = α + α_ID[IDidx_vec[i]] + βV * V_vec[i] + βD * D_vec[i] +
                         βḞ * f_val + βH * H_vec[i] + βM * M_vec[i] +
                         βR * R_vec[i] + βS * S_vec[i] + βSR * sr_interact
            end

            # Factual prediction (with parasites)
            μ_factual = base_μ + βP * P_vec[i]
            E_factual[i, s] = μ_factual + σ * randn()

            # Counterfactual prediction (without parasites: P = 0)
            E_counterfactual[i, s] = base_μ + σ * randn()
        end
    end

    return E_factual, E_counterfactual
end

# Generate counterfactuals using posterior predictive approach with interaction
println("Generating counterfactuals with S×R interaction using posterior predictive approach...")
E_factual_int, E_counterfactual_int = generate_counterfactuals_with_interaction(
    pre_intervention_chain_interaction, dag_df_infected; use_effect_coding=USE_EFFECT_CODING)

# Add predictions to dataframe - using this as our main approach
dag_df_infected.E_factual_mean = vec(mean(E_factual_int, dims=2))
dag_df_infected.E_counterfactual_mean = vec(mean(E_counterfactual_int, dims=2))
dag_df_infected.E_diff_counterfactual = dag_df_infected.E_counterfactual_mean .- dag_df_infected.E_factual_mean

# Calculate uncertainty measures
dag_df_infected.E_factual_sd = vec(std(E_factual_int, dims=2))
dag_df_infected.E_counterfactual_sd = vec(std(E_counterfactual_int, dims=2))

# Calculate Cohen's d effect size for standardised assessment
dag_df_infected.E_cohens_d = begin
    # Pooled standard deviation for Cohen's d calculation
    pooled_sd = sqrt.((dag_df_infected.E_factual_sd .^ 2 .+ dag_df_infected.E_counterfactual_sd .^ 2) ./ 2)
    # Avoid division by very small numbers
    safe_pooled_sd = max.(pooled_sd, 0.1)  # Minimum threshold for stability
    dag_df_infected.E_diff_counterfactual ./ safe_pooled_sd
end

println("Counterfactual Results with S×R Interaction:")
println("Mean counterfactual effect: ", round(mean(dag_df_infected.E_diff_counterfactual), digits=3))
println("Mean Cohen's d: ", round(mean(dag_df_infected.E_cohens_d), digits=3))

# Clinical significance analysis
dag_df_infected.E_clinical_significance = begin
    abs_cohens_d = abs.(dag_df_infected.E_cohens_d)
    ifelse.(abs_cohens_d .< 0.2, "Negligible",
        ifelse.(abs_cohens_d .< 0.5, "Small",
            ifelse.(abs_cohens_d .< 0.8, "Moderate", "Large")))
end

dag_df_infected.E_clinical_direction = begin
    cohens_d = dag_df_infected.E_cohens_d
    abs_cohens_d = abs.(cohens_d)
    direction = ifelse.(cohens_d .>= 0, "Beneficial", "Detrimental")
    magnitude = ifelse.(abs_cohens_d .< 0.2, "Negligible",
        ifelse.(abs_cohens_d .< 0.5, "Small",
            ifelse.(abs_cohens_d .< 0.8, "Moderate", "Large")))
    magnitude .* " " .* direction
end

println("Summary of counterfactual effects with interaction:")
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
precis(DataFrame(pre_intervention_chain_interaction)[!, r"α\b|β"])
println("\nPre-intervention model with interaction diagnostics:")
plot_chains_df(pre_intervention_chain_interaction; show_traces=false)

## NEW: Interaction-specific analysis functions

"""
    calculate_group_specific_effects(chain_interaction, use_effect_coding=true)

Calculate group-specific counterfactual effects for each Sex × Reproductive Status combination.

With effect coding, group effects are calculated as:
- Male, Non-reproductive: -βS - βR + βSR
- Male, Reproductive: -βS + βR - βSR
- Female, Non-reproductive: +βS - βR - βSR
- Female, Reproductive: +βS + βR + βSR

# Arguments
- `chain_interaction`: MCMC chain from model with interaction
- `use_effect_coding`: Whether effect coding was used in the model

# Returns
Dictionary with group-specific effect estimates (mean ± SD)
"""
function calculate_group_specific_effects(chain_interaction; use_effect_coding::Bool=true)
    println("=== GROUP-SPECIFIC COUNTERFACTUAL EFFECTS ===")

    # Extract parameter samples
    chain_df = DataFrame(chain_interaction)
    βSR_samples = chain_df.βSR
    βS_samples = chain_df.βS
    βR_samples = chain_df.βR

    if use_effect_coding
        # For effect coding, group effects relative to grand mean
        effects = Dict(
            "Male_NonReproductive" => -βS_samples - βR_samples + βSR_samples,
            "Male_Reproductive" => -βS_samples + βR_samples - βSR_samples,
            "Female_NonReproductive" => βS_samples - βR_samples - βSR_samples,
            "Female_Reproductive" => βS_samples + βR_samples + βSR_samples
        )

        println("Effect coding interpretation:")
        println("Values represent deviations from overall mean effect")
    else
        # For raw coding, need to consider the specific contrasts
        println("Raw coding interpretation:")
        println("Values represent effects for specific group combinations")

        # This would need more complex calculation depending on reference levels
        effects = Dict(
            "Male_NonReproductive" => βS_samples .* 1 + βR_samples .* 1 + βSR_samples .* 1,
            "Male_Reproductive" => βS_samples .* 1 + βR_samples .* 2 + βSR_samples .* 2,
            "Female_NonReproductive" => βS_samples .* 2 + βR_samples .* 1 + βSR_samples .* 2,
            "Female_Reproductive" => βS_samples .* 2 + βR_samples .* 2 + βSR_samples .* 4
        )
    end

    # Calculate summary statistics for each group
    group_summaries = Dict()
    for (group, samples) in effects
        mean_effect = mean(samples)
        sd_effect = std(samples)
        ci_lower, ci_upper = quantile(samples, [0.025, 0.975])

        group_summaries[group] = (
            mean=mean_effect,
            sd=sd_effect,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )

        println("$group: $(round(mean_effect, digits=3)) ± $(round(sd_effect, digits=3))")
        println("  95% CI: [$(round(ci_lower, digits=3)), $(round(ci_upper, digits=3))]")
    end

    return group_summaries
end

"""
    test_interaction_significance(chain_interaction)

Test the significance of the Sex × Reproductive Status interaction effect.

# Arguments
- `chain_interaction`: MCMC chain from model with interaction

# Returns
Summary statistics and probability assessments for interaction effect
"""
function test_interaction_significance(chain_interaction)
    println("\n=== INTERACTION SIGNIFICANCE TESTING ===")

    chain_df = DataFrame(chain_interaction)
    βSR_samples = chain_df.βSR

    # Summary statistics
    mean_βSR = mean(βSR_samples)
    sd_βSR = std(βSR_samples)
    ci_lower, ci_upper = quantile(βSR_samples, [0.025, 0.975])

    println("Sex × Reproductive Status Interaction (βSR):")
    println("Mean: $(round(mean_βSR, digits=4))")
    println("SD: $(round(sd_βSR, digits=4))")
    println("95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")

    # Probability assessments
    prob_positive = mean(βSR_samples .> 0)
    prob_negative = mean(βSR_samples .< 0)
    prob_substantial = mean(abs.(βSR_samples) .> 0.1)  # |βSR| > 0.1 for standardised outcome
    prob_large = mean(abs.(βSR_samples) .> 0.2)       # |βSR| > 0.2 for standardised outcome

    println("\nProbability assessments:")
    println("P(βSR > 0): $(round(prob_positive, digits=3))")
    println("P(βSR < 0): $(round(prob_negative, digits=3))")
    println("P(|βSR| > 0.1): $(round(prob_substantial, digits=3)) [substantial interaction]")
    println("P(|βSR| > 0.2): $(round(prob_large, digits=3)) [large interaction]")

    # Effect size interpretation
    if abs(mean_βSR) < 0.1
        interpretation = "negligible"
    elseif abs(mean_βSR) < 0.2
        interpretation = "small"
    elseif abs(mean_βSR) < 0.3
        interpretation = "moderate"
    else
        interpretation = "large"
    end

    direction = mean_βSR > 0 ? "Female reproductive advantage" : "Female reproductive disadvantage"

    println("\nInterpretation: $(interpretation) interaction effect")
    println("Direction: $(direction) (relative to other groups)")

    return (
        mean=mean_βSR,
        sd=sd_βSR,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        prob_positive=prob_positive,
        prob_substantial=prob_substantial,
        interpretation=interpretation
    )
end

# Run interaction-specific analyses
group_effects = calculate_group_specific_effects(pre_intervention_chain_interaction; use_effect_coding=USE_EFFECT_CODING)
interaction_test = test_interaction_significance(pre_intervention_chain_interaction)

# Define the plotting function before using it
"""
    plot_bayesian_S_R_interaction(df_infected, chain_interaction; saveplot=false)

Plot Sex × Reproductive Status interaction effects using Bayesian posterior estimates.
"""
function plot_bayesian_S_R_interaction(df_infected, chain_interaction; saveplot::Bool=false)
    # Extract group effects
    group_effects_local = calculate_group_specific_effects(chain_interaction; use_effect_coding=USE_EFFECT_CODING)

    # Prepare data for plotting
    groups = ["Male_NonReproductive", "Male_Reproductive", "Female_NonReproductive", "Female_Reproductive"]
    means = [group_effects_local[g].mean for g in groups]
    cis_lower = [group_effects_local[g].ci_lower for g in groups]
    cis_upper = [group_effects_local[g].ci_upper for g in groups]

    # Create plot
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        title="Bayesian Sex × Reproductive Status Interaction Effects\n(Counterfactual - Factual)",
        xlabel="Group",
        ylabel="Effect Size (Standardised Units)",
        xticklabelrotation=π / 4
    )

    # Plot points and error bars
    x_pos = 1:4
    scatter!(ax, x_pos, means, markersize=15, color=:blue)
    errorbars!(ax, x_pos, means, means .- cis_lower, cis_upper .- means, color=:blue, linewidth=2)

    # Add horizontal line at zero
    hlines!(ax, [0], color=:red, linestyle=:dash, alpha=0.7)

    # Customize x-axis
    ax.xticks = (x_pos, ["Male\nNon-repro", "Male\nRepro", "Female\nNon-repro", "Female\nRepro"])

    # Add effect size annotations
    for (i, (mean_val, group)) in enumerate(zip(means, groups))
        if abs(mean_val) > 0.1
            effect_size = abs(mean_val) < 0.2 ? "Small" : abs(mean_val) < 0.3 ? "Moderate" : "Large"
            text!(ax, i, mean_val + sign(mean_val) * 0.05, text=effect_size, align=(:center, :bottom), fontsize=10)
        end
    end

    if saveplot
        save("../manuscript/Figures/plots/bayesian_SR_interaction_effects.svg", fig)
    end

    return fig
end

# Now that group_effects is calculated, create the Bayesian interaction plots
with_theme(theme_minimal()) do
    plot_bayesian_S_R_interaction(dag_df_infected, pre_intervention_chain_interaction, saveplot=SAVE_PLOTS)
end

## Analysis helper functions (updated for interaction)

"""
    calculate_counterfactual_percentage_change(df_infected; method="relative_to_baseline")

Calculate percentage change in vaccine response under counterfactual parasite elimination.
Enhanced version that accounts for interaction effects.

# Arguments
- `df_infected`: DataFrame with counterfactual predictions
- `method`: Calculation method. Options:
  - `"relative_to_baseline"`: Change relative to individual baseline values
  - `"relative_to_population"`: Change relative to population mean
  - `"relative_to_original_scale"`: Change in back-transformed OD scale
  - `"cohen_d_to_percent"`: Convert Cohen's d to percentage using pooled SD
  - `"by_group"`: Calculate separately for each Sex × Reproductive Status group

# Returns
Vector of percentage changes for each observation (also prints summary statistics).

Different methods handle the standardised nature of the vaccine response differently.
Enhanced to show group-specific effects when using "by_group" method.
"""
function calculate_counterfactual_percentage_change(df_infected; method::String="relative_to_baseline")

    if method == "relative_to_baseline"
        # Method 1: Direct percentage change in standardised units
        # Use absolute values to avoid issues with negative standardised scores
        baseline_E = abs.(df_infected.E_factual_mean)
        change_E = df_infected.E_counterfactual_mean .- df_infected.E_factual_mean

        # Avoid division by very small numbers
        safe_baseline = max.(baseline_E, 0.1)
        percent_change = (change_E ./ safe_baseline) .* 100

        println("Method 1 - Relative to baseline (standardised units):")

    elseif method == "relative_to_population"
        # Method 2: Change relative to population mean
        pop_mean_E = mean(df_infected.E_factual_mean)
        change_E = df_infected.E_counterfactual_mean .- df_infected.E_factual_mean
        percent_change = (change_E ./ abs(pop_mean_E)) .* 100

        println("Method 2 - Relative to population mean:")

    elseif method == "relative_to_original_scale"
        # Method 3: Back-transform to original OD scale
        # Need original OD data to reverse the standardisation
        if !hasproperty(df_infected, :OD)
            error("Original OD data not available for back-transformation")
        end

        # Approximate back-transformation (this is rough since we used log-transformation)
        # Better would be to store the original transformation parameters
        original_logOD_mean = mean(log10.(1 .+ df_infected.OD))
        original_logOD_std = std(log10.(1 .+ df_infected.OD))

        # Back-transform standardised E to log OD scale
        factual_logOD = df_infected.E_factual_mean .* original_logOD_std .+ original_logOD_mean
        counterfactual_logOD = df_infected.E_counterfactual_mean .* original_logOD_std .+ original_logOD_mean

        # Convert to OD scale
        factual_OD = 10 .^ factual_logOD .- 1
        counterfactual_OD = 10 .^ counterfactual_logOD .- 1

        # Calculate percentage change in original OD units
        safe_factual_OD = max.(factual_OD, 0.01)  # Avoid division by zero
        percent_change = ((counterfactual_OD .- factual_OD) ./ safe_factual_OD) .* 100

        println("Method 3 - Back-transformed to original OD scale:")

    elseif method == "cohen_d_to_percent"
        # Method 4: Convert Cohen's d to percentage using effect size interpretation
        # Cohen's d represents standardised effect size
        # Convert to percentage improvement using pooled standard deviation

        pooled_sd = sqrt.((df_infected.E_factual_sd .^ 2 .+ df_infected.E_counterfactual_sd .^ 2) ./ 2)
        safe_pooled_sd = max.(pooled_sd, 0.1)

        # Percentage change = (Cohen's d × pooled SD) / |baseline| × 100
        baseline_E = abs.(df_infected.E_factual_mean)
        safe_baseline = max.(baseline_E, 0.1)

        percent_change = (df_infected.E_cohens_d .* safe_pooled_sd ./ safe_baseline) .* 100

        println("Method 4 - Cohen's d converted to percentage:")

    elseif method == "by_group"
        # Method 5: Calculate percentage changes by Sex × Reproductive Status group
        println("Method 5 - By Sex × Reproductive Status groups:")

        # Create group labels
        group_labels = [
            (s == 1 ? "Male" : "Female") * "_" * (r == 1 ? "NonReproductive" : "Reproductive")
            for (s, r) in zip(df_infected.S, df_infected.R)
        ]

        # Calculate baseline method for each group
        baseline_E = abs.(df_infected.E_factual_mean)
        change_E = df_infected.E_counterfactual_mean .- df_infected.E_factual_mean
        safe_baseline = max.(baseline_E, 0.1)
        percent_change = (change_E ./ safe_baseline) .* 100

        # Add group information to results
        df_infected.group_label = group_labels

        # Summary by group
        println("\nGroup-specific results:")
        for group in unique(group_labels)
            group_mask = group_labels .== group
            group_changes = percent_change[group_mask]

            if length(group_changes) > 0
                println("$group (n=$(length(group_changes))):")
                println("  Mean: $(round(mean(group_changes), digits=2))%")
                println("  SD: $(round(std(group_changes), digits=2))%")
                println("  Range: $(round(minimum(group_changes), digits=1))% to $(round(maximum(group_changes), digits=1))%")
            end
        end

        return percent_change

    else
        error("Unknown method: $method. Choose from: relative_to_baseline, relative_to_population, relative_to_original_scale, cohen_d_to_percent, by_group")
    end

    # Summary statistics (for non-group methods)
    println("Mean percentage change: $(round(mean(percent_change), digits=2))%")
    println("Median percentage change: $(round(median(percent_change), digits=2))%")
    println("Standard deviation: $(round(std(percent_change), digits=2))%")
    println("Range: $(round(minimum(percent_change), digits=2))% to $(round(maximum(percent_change), digits=2))%")

    # Count beneficial vs detrimental changes
    beneficial = sum(percent_change .> 0)
    detrimental = sum(percent_change .< 0)
    println("Beneficial changes: $beneficial mice ($(round(100*beneficial/length(percent_change), digits=1))%)")
    println("Detrimental changes: $detrimental mice ($(round(100*detrimental/length(percent_change), digits=1))%)")

    return percent_change
end

"""
    calculate_vaccination_effect_percentage(df_infected)

Calculate percentage improvement in vaccination effect under parasite elimination.
Enhanced version that accounts for interaction effects.

Compares vaccination coefficients between factual and counterfactual scenarios
to determine how much better vaccines work when parasites are eliminated,
with separate analysis for each Sex × Reproductive Status group.

# Arguments
- `df_infected`: DataFrame with counterfactual predictions

# Returns
Named tuple with vaccination effect coefficients and improvement statistics.
"""
function calculate_vaccination_effect_percentage(df_infected)
    println("=== VACCINATION EFFECT PERCENTAGE IMPROVEMENT WITH INTERACTION ===")

    # Overall vaccination effects
    factual_model = fit(MixedModel, @formula(E_factual_mean ~ V + (1 | ID)), df_infected)
    counterfactual_model = fit(MixedModel, @formula(E_counterfactual_mean ~ V + (1 | ID)), df_infected)

    # Extract vaccination coefficients
    β_factual = coef(factual_model)[2]  # V coefficient
    β_counterfactual = coef(counterfactual_model)[2]  # V coefficient

    # Calculate percentage improvement in vaccination effect
    vaccination_improvement = ((β_counterfactual - β_factual) / abs(β_factual)) * 100

    println("Overall Effects:")
    println("Factual vaccination effect (β_V): $(round(β_factual, digits=3))")
    println("Counterfactual vaccination effect (β_V): $(round(β_counterfactual, digits=3))")
    println("Vaccination effect improvement: $(round(vaccination_improvement, digits=1))%")

    # Also calculate absolute improvement
    absolute_improvement = β_counterfactual - β_factual
    println("Absolute improvement: $(round(absolute_improvement, digits=3)) standardised units")

    # Group-specific analysis
    println("\nGroup-specific vaccination effects:")

    group_results = Dict()

    for (s, sex_label) in enumerate(["Male", "Female"])
        for (r, repro_label) in enumerate(["Non-reproductive", "Reproductive"])
            # Filter to specific group
            group_mask = (df_infected.S .== s) .& (df_infected.R .== r)
            group_data = df_infected[group_mask, :]

            if nrow(group_data) > 5  # Ensure sufficient data
                # Fit group-specific models
                try
                    factual_group = fit(MixedModel, @formula(E_factual_mean ~ V + (1 | ID)), group_data)
                    counterfactual_group = fit(MixedModel, @formula(E_counterfactual_mean ~ V + (1 | ID)), group_data)

                    β_factual_group = coef(factual_group)[2]
                    β_counterfactual_group = coef(counterfactual_group)[2]
                    improvement_group = ((β_counterfactual_group - β_factual_group) / abs(β_factual_group)) * 100

                    group_key = "$(sex_label)_$(repro_label)"
                    group_results[group_key] = (
                        factual=β_factual_group,
                        counterfactual=β_counterfactual_group,
                        improvement=improvement_group,
                        n=nrow(group_data)
                    )

                    println("$sex_label, $repro_label (n=$(nrow(group_data))):")
                    println("  Factual: $(round(β_factual_group, digits=3))")
                    println("  Counterfactual: $(round(β_counterfactual_group, digits=3))")
                    println("  Improvement: $(round(improvement_group, digits=1))%")

                catch e
                    println("$sex_label, $repro_label: Insufficient data for analysis (n=$(nrow(group_data)))")
                end
            else
                println("$sex_label, $repro_label: Insufficient data (n=$(nrow(group_data)))")
            end
        end
    end

    return (
        overall_factual_effect=β_factual,
        overall_counterfactual_effect=β_counterfactual,
        overall_percentage_improvement=vaccination_improvement,
        overall_absolute_improvement=absolute_improvement,
        group_results=group_results
    )
end

## Causal effect analysis using generative model predictions with interaction

# Check adjustment sets for vaccination effect
adjustmentSets(dag, "V", "E", effect="total") # { }
adjustmentSets(dag_m, "V", "E", effect="total") # { }

# Total effect of V on E (factual vs counterfactual) with interaction
if USE_EFFECT_CODING
    glmm_V_E_total_factual = fit(MixedModel, @formula(E_factual_mean ~ V + S_coded + R_coded + S_coded & R_coded + (1 | ID)), dag_df_infected)
    glmm_V_E_total_counterfactual = fit(MixedModel, @formula(E_counterfactual_mean ~ V + S_coded + R_coded + S_coded & R_coded + (1 | ID)), dag_df_infected)
else
    glmm_V_E_total_factual = fit(MixedModel, @formula(E_factual_mean ~ V + S + R + S & R + (1 | ID)), dag_df_infected)
    glmm_V_E_total_counterfactual = fit(MixedModel, @formula(E_counterfactual_mean ~ V + S + R + S & R + (1 | ID)), dag_df_infected)
end

# Diagnostics for vaccination effect model with interaction
qqnorm(glmm_V_E_total_counterfactual; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_total_counterfactual)
coefplot(boot)

# Check adjustment sets for direct effects
adjustmentSets(dag, "V", "E", effect="direct") # { D, F, H, M, P, R, S }
adjustmentSets(dag_m, "V", "E", effect="direct") # { D, F, H, M, P, R, S }

# Direct effect of V on E (factual vs counterfactual) with interaction
if USE_EFFECT_CODING
    glmm_V_E_direct_factual = fit(MixedModel, @formula(E_factual_mean ~ V + D + Ḟ + H + M + P + S_coded + R_coded + S_coded & R_coded + (1 | ID)), dag_df_infected)
    glmm_V_E_direct_counterfactual = fit(MixedModel, @formula(E_counterfactual_mean ~ V + D + Ḟ + H + M + S_coded + R_coded + S_coded & R_coded + (1 | ID)), dag_df_infected)
else
    glmm_V_E_direct_factual = fit(MixedModel, @formula(E_factual_mean ~ V + D + Ḟ + H + M + P + S + R + S & R + (1 | ID)), dag_df_infected)
    glmm_V_E_direct_counterfactual = fit(MixedModel, @formula(E_counterfactual_mean ~ V + D + Ḟ + H + M + S + R + S & R + (1 | ID)), dag_df_infected)
end

# Diagnostics for direct vaccination effect model with interaction
qqnorm(glmm_V_E_direct_counterfactual; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_direct_counterfactual)
coefplot(boot)

## Generate enhanced plots with clinical significance and interaction effects

with_theme(theme_minimal()) do
    plot_E_factual_counterfactual(dag_df_infected, saveplot=SAVE_PLOTS)
end

with_theme(theme_minimal()) do
    plot_counterfactual_effects_with_significance(dag_df_infected, saveplot=SAVE_PLOTS)
end

with_theme(theme_minimal()) do
    plot_clinical_significance_summary(dag_df_infected, saveplot=SAVE_PLOTS)
end

## NEW: Enhanced interaction analysis with Cohen's d and Bayesian estimates

# Fit model using Cohen's d as outcome with interaction
if USE_EFFECT_CODING
    glmm_cohens_d_interaction = fit(MixedModel, @formula(E_cohens_d ~ -1 + D + S_coded + R_coded + V + M + Ḟ + S_coded & R_coded + (1 | ID)), dag_df_infected)
else
    glmm_cohens_d_interaction = fit(MixedModel, @formula(E_cohens_d ~ -1 + D + S + R + V + M + Ḟ + S & R + (1 | ID)), dag_df_infected)
end

# Enhanced interaction plots using Bayesian estimates
# Note: group_effects is calculated below, so moved this plot there
# with_theme(theme_minimal()) do
#     plot_bayesian_S_R_interaction(dag_df_infected, pre_intervention_chain_interaction, saveplot=SAVE_PLOTS)
# end

with_theme(theme_minimal()) do
    plot_S_R_interaction_cohens_d(dag_df_infected, saveplot=SAVE_PLOTS)
end

with_theme(theme_minimal()) do
    plot_S_R_interaction_factual_counterfactual(dag_df_infected, saveplot=SAVE_PLOTS)
end

# Define the detailed plotting function before using it
"""
    plot_sex_reproductive_interaction_detailed_bayesian(df_infected, group_effects; saveplot=false)

Create detailed interaction plot with Bayesian estimates and individual data points.
"""
function plot_sex_reproductive_interaction_detailed_bayesian(df_infected, group_effects; saveplot::Bool=false)

    fig = Figure(size=(1000, 700))

    # Main interaction plot
    ax1 = Axis(fig[1, 1],
        title="Sex × Reproductive Status Interaction\nCounterfactual Effects by Group",
        xlabel="Reproductive Status",
        ylabel="Counterfactual Effect (Standardised Units)"
    )

    # Prepare group data
    male_nonrepro = df_infected[(df_infected.S.==1).&(df_infected.R.==1), :E_diff_counterfactual]
    male_repro = df_infected[(df_infected.S.==1).&(df_infected.R.==2), :E_diff_counterfactual]
    female_nonrepro = df_infected[(df_infected.S.==2).&(df_infected.R.==1), :E_diff_counterfactual]
    female_repro = df_infected[(df_infected.S.==2).&(df_infected.R.==2), :E_diff_counterfactual]

    # Plot individual points with jitter
    x_jitter = 0.1
    scatter!(ax1, fill(1, length(male_nonrepro)) .+ (rand(length(male_nonrepro)) .- 0.5) .* x_jitter,
        male_nonrepro, color=(:blue, 0.3), markersize=8)
    scatter!(ax1, fill(2, length(male_repro)) .+ (rand(length(male_repro)) .- 0.5) .* x_jitter,
        male_repro, color=(:blue, 0.3), markersize=8)
    scatter!(ax1, fill(1, length(female_nonrepro)) .+ (rand(length(female_nonrepro)) .- 0.5) .* x_jitter,
        female_nonrepro, color=(:red, 0.3), markersize=8)
    scatter!(ax1, fill(2, length(female_repro)) .+ (rand(length(female_repro)) .- 0.5) .* x_jitter,
        female_repro, color=(:red, 0.3), markersize=8)

    # Plot group means from Bayesian analysis
    male_means = [group_effects["Male_NonReproductive"].mean, group_effects["Male_Reproductive"].mean]
    female_means = [group_effects["Female_NonReproductive"].mean, group_effects["Female_Reproductive"].mean]

    lines!(ax1, [1, 2], male_means, color=:blue, linewidth=3, label="Male")
    lines!(ax1, [1, 2], female_means, color=:red, linewidth=3, label="Female")
    scatter!(ax1, [1, 2], male_means, color=:blue, markersize=15)
    scatter!(ax1, [1, 2], female_means, color=:red, markersize=15)

    # Add confidence bands
    male_cis_lower = [group_effects["Male_NonReproductive"].ci_lower, group_effects["Male_Reproductive"].ci_lower]
    male_cis_upper = [group_effects["Male_NonReproductive"].ci_upper, group_effects["Male_Reproductive"].ci_upper]
    female_cis_lower = [group_effects["Female_NonReproductive"].ci_lower, group_effects["Female_Reproductive"].ci_lower]
    female_cis_upper = [group_effects["Female_NonReproductive"].ci_upper, group_effects["Female_Reproductive"].ci_upper]

    band!(ax1, [1, 2], male_cis_lower, male_cis_upper, color=(:blue, 0.2))
    band!(ax1, [1, 2], female_cis_lower, female_cis_upper, color=(:red, 0.2))

    # Customize axes
    ax1.xticks = ([1, 2], ["Non-reproductive", "Reproductive"])
    hlines!(ax1, [0], color=:black, linestyle=:dash, alpha=0.5)
    axislegend(ax1, position=(:right, :top))

    # Side panel: Effect sizes
    ax2 = Axis(fig[1, 2],
        title="Effect Sizes\n(Cohen's d equivalent)",
        xlabel="Group",
        ylabel="Cohen's d"
    )

    groups = ["Male\nNon-repro", "Male\nRepro", "Female\nNon-repro", "Female\nRepro"]
    cohen_d_means = [
        group_effects["Male_NonReproductive"].mean,
        group_effects["Male_Reproductive"].mean,
        group_effects["Female_NonReproductive"].mean,
        group_effects["Female_Reproductive"].mean
    ]

    colors = [:blue, :blue, :red, :red]
    barplot!(ax2, 1:4, cohen_d_means, color=colors, alpha=0.7)
    ax2.xticks = (1:4, groups)
    hlines!(ax2, [0], color=:black, linestyle=:dash)

    # Add effect size thresholds
    hlines!(ax2, [0.2, -0.2], color=:gray, linestyle=:dot, alpha=0.5)
    hlines!(ax2, [0.5, -0.5], color=:gray, linestyle=:dot, alpha=0.5)
    text!(ax2, 4.5, 0.2, text="Small", align=(:left, :center), fontsize=10, color=:gray)
    text!(ax2, 4.5, 0.5, text="Medium", align=(:left, :center), fontsize=10, color=:gray)

    if saveplot
        save("../manuscript/Figures/plots/detailed_bayesian_SR_interaction.svg", fig)
    end

    return fig
end

# Generate detailed Sex × Reproductive status interaction plots with Bayesian uncertainty
with_theme(theme_minimal()) do
    plot_sex_reproductive_interaction_detailed_bayesian(dag_df_infected, group_effects, saveplot=SAVE_PLOTS)
end

# Note: plot_sex_reproductive_heatmap_bayesian function needs to be defined
# Commenting out until function is implemented
# with_theme(theme_minimal()) do
#     plot_sex_reproductive_heatmap_bayesian(dag_df_infected, group_effects, saveplot=SAVE_PLOTS)
# end

# Diagnostics for Cohen's d interaction model
boot_cohens_d = parametricbootstrap(MersenneTwister(1234), 10_000, glmm_cohens_d_interaction)
coefplot(boot_cohens_d)

## NEW: Enhanced plotting functions for interaction analysis
# (Functions moved up in the script to fix ordering issues)

## Calculate percentage improvements using different methods with interaction analysis

println("\n" * "="^60)
println("PERCENTAGE IMPROVEMENT CALCULATIONS WITH INTERACTION")
println("="^60)

# Method 1: Vaccination effect improvement (recommended for policy/clinical interpretation)
vaccination_results = calculate_vaccination_effect_percentage(dag_df_infected)

println("\n" * "-"^50)

# Method 2: Individual-level percentage changes (multiple approaches)
# Try different methods to see which gives most interpretable results

println("Individual-level percentage changes:")
println()

# Method 2a: Relative to baseline (most direct)
percent_changes_baseline = calculate_counterfactual_percentage_change(dag_df_infected; method="relative_to_baseline")
dag_df_infected.percent_change_baseline = percent_changes_baseline

println("\n" * "-"^30)

# Method 2b: Relative to population mean
percent_changes_population = calculate_counterfactual_percentage_change(dag_df_infected; method="relative_to_population")
dag_df_infected.percent_change_population = percent_changes_population

println("\n" * "-"^30)

# Method 2c: Cohen's d to percentage
percent_changes_cohens = calculate_counterfactual_percentage_change(dag_df_infected; method="cohen_d_to_percent")
dag_df_infected.percent_change_cohens = percent_changes_cohens

println("\n" * "-"^30)

# Method 2d: NEW - By group analysis
percent_changes_by_group = calculate_counterfactual_percentage_change(dag_df_infected; method="by_group")
dag_df_infected.percent_change_by_group = percent_changes_by_group

# Debug: Check if Cohen's d values vary with interaction
println("DEBUG: Cohen's d values with interaction (first 10): ", dag_df_infected.E_cohens_d[1:min(10, nrow(dag_df_infected))])
println("DEBUG: Are all Cohen's d values equal? ", all(dag_df_infected.E_cohens_d .≈ dag_df_infected.E_cohens_d[1]))
println("DEBUG: Cohen's d mean: ", mean(dag_df_infected.E_cohens_d))
println("DEBUG: Cohen's d std: ", std(dag_df_infected.E_cohens_d))

# Group-specific Cohen's d analysis
println("\nCohen's d by group:")
if hasproperty(dag_df_infected, :group_label)
    for group in unique(dag_df_infected.group_label)
        group_mask = dag_df_infected.group_label .== group
        group_cohens = dag_df_infected.E_cohens_d[group_mask]
        println("$group: mean = $(round(mean(group_cohens), digits=3)), sd = $(round(std(group_cohens), digits=3))")
    end
else
    # Create group labels manually if not already created
    for (s, sex_label) in enumerate(["Male", "Female"])
        for (r, repro_label) in enumerate(["NonReproductive", "Reproductive"])
            group_mask = (dag_df_infected.S .== s) .& (dag_df_infected.R .== r)
            group_cohens = dag_df_infected.E_cohens_d[group_mask]
            if length(group_cohens) > 0
                println("$(sex_label)_$(repro_label): mean = $(round(mean(group_cohens), digits=3)), sd = $(round(std(group_cohens), digits=3))")
            end
        end
    end
end

println("\n" * "-"^30)

# Method 2e: Back-transformed to original scale (if OD data available)
if hasproperty(dag_df_infected, :OD)
    println("Attempting back-transformation to original OD scale...")
    try
        percent_changes_original = calculate_counterfactual_percentage_change(dag_df_infected; method="relative_to_original_scale")
        dag_df_infected.percent_change_original = percent_changes_original
    catch e
        println("Back-transformation failed: $e")
    end
else
    println("Original OD data not available for back-transformation")
end

println("\n" * "="^60)
println("SUMMARY RECOMMENDATIONS WITH INTERACTION")
println("="^60)

println("For policy/clinical interpretation, use vaccination effect improvement:")
println("- Overall vaccine efficacy improves by $(round(vaccination_results.overall_percentage_improvement, digits=1))% when parasites are eliminated")

if !isempty(vaccination_results.group_results)
    println("\nGroup-specific vaccination improvements:")
    for (group, results) in vaccination_results.group_results
        println("- $group: $(round(results.improvement, digits=1))% improvement")
    end
end

println()
println("For individual-level analysis with interaction effects:")
println("- Mean individual improvement: $(round(mean(percent_changes_cohens), digits=1))%")
println("- Interaction effect significance: $(interaction_test.interpretation)")

# Create summary table of individual-level improvement methods including group analysis
println("\n" * "="^60)
println("INDIVIDUAL-LEVEL PERCENTAGE IMPROVEMENT METHODS WITH INTERACTION")
println("="^60)

# Collect individual-level data for cleaner processing
individual_data = [percent_changes_baseline, percent_changes_population, percent_changes_by_group]
methods = ["Relative to own baseline", "Relative to population mean", "By Sex×Reproductive group"]

# Calculate means and standard deviations for individual-level methods
means = mean.(individual_data)
stds = std.(individual_data)

summary_table = DataFrame(
    Method=methods,
    Mean_SD=["$(round(means[i], digits=1)) ± $(round(stds[i], digits=1))" for i in 1:length(means)],
    Median=round.(median.(individual_data), digits=1),
    Range=["$(round(minimum(data), digits=1)) to $(round(maximum(data), digits=1))" for data in individual_data]
);

println(summary_table)

println("\n" * "="^60)
println("INTERACTION ANALYSIS SUMMARY")
println("="^60)

println("Sex × Reproductive Status Interaction:")
println("- Interaction coefficient (βSR): $(round(interaction_test.mean, digits=4)) ± $(round(interaction_test.sd, digits=4))")
println("- 95% CI: [$(round(interaction_test.ci_lower, digits=4)), $(round(interaction_test.ci_upper, digits=4))]")
println("- Interpretation: $(interaction_test.interpretation)")
println("- Probability of substantial effect (|βSR| > 0.1): $(round(interaction_test.prob_substantial, digits=3))")

println("\nGroup-specific counterfactual effects:")
for (group, effects) in group_effects
    println("- $group: $(round(effects.mean, digits=3)) [$(round(effects.ci_lower, digits=3)), $(round(effects.ci_upper, digits=3))]")
end

# Model summary statistics
chain_summary = DataFrame(pre_intervention_chain_interaction)
param_summary = describe(chain_summary[!, r"^β"])
println("Parameter estimates summary:")
println(param_summary)

# Convergence diagnostics
println("\nConvergence diagnostics:")
println("Rhat values (should be < 1.01):")
chain_summary_diag = summarize(pre_intervention_chain_interaction)
rhat_vals = chain_summary_diag[:, :rhat]
param_names = chain_summary_diag[:, :parameters]
for (param, rhat) in zip(param_names, rhat_vals)
    if startswith(string(param), "β")
        println("  $param: $(round(rhat, digits=4))")
    end
end
