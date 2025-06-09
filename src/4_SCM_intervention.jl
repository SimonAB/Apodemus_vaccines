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

## Configuration constants

# Set default plotting behaviour - change to false to disable plot saving
const SAVE_PLOTS = false

# Performance configuration following Turing.jl best practices
# For models with many parameters (>20), consider AutoZygote() or AutoReverseDiff()
# For models with few parameters (<20), AutoForwardDiff() is typically fastest
const AD_BACKEND = nothing  # Use default (AutoForwardDiff) for our model size

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

Bayesian structural causal model for vaccine response enabling counterfactual inference under parasite elimination.

This model implements the structural causal framework by:
1. Pre-intervention: Models E as function of all relevant causes including P
2. Post-intervention: Models E under do(P=0) by removing P→E pathway
3. Counterfactual queries: Enables comparison between factual and counterfactual worlds

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

# Model Structure
- Factual: E ~ f(V, D, Ḟ, H, M, P, R, S, α_ID) + ε
- Counterfactual: E ~ f(V, D, Ḟ, H, M, R, S, α_ID) + ε [P pathway severed]

# Assumptions
- Modularity: Each structural equation can be modified independently
- No unmeasured confounding given DAG structure
- Individual-level confounding handled via random effects
"""
@model function Counterfactual_E_Model(IDidx, E, V, D, Ḟ, H, M, P, R, S, nP, post_P, post_nP;
    n_id=length(unique(IDidx)), intervention=false)

    # Pre-compute statistics for type stability and performance
    N::Int = length(E)
    E_mean::Float64 = mean(E)
    E_std::Float64 = std(E)

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

        # Compute base effects for all individuals
        base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M + βR * R + βS * S

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
        base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M + βR * R + βS * S

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
    Twin_World_E_Model(IDidx, E_factual, E_counterfactual, V, D, Ḟ, H, M, P, R, S,
                       post_P, post_nP; n_id=length(unique(IDidx)))

Joint model for factual and counterfactual vaccine responses using the "twin world" approach.

This model simultaneously samples both the observed world (with parasites) and
counterfactual world (without parasites), sharing individual-level random effects
to maintain individual identity across worlds.

# Arguments
- `E_factual`: Observed vaccine response
- `E_counterfactual`: Counterfactual vaccine response (under do(P=0))
- Other arguments as in `Counterfactual_E_Model`

# Returns
Joint posterior enabling direct estimation of individual treatment effects.
"""
@model function Twin_World_E_Model(IDidx, E_factual, E_counterfactual, V, D, Ḟ, H, M, P, R, S,
    post_P, post_nP; n_id=length(unique(IDidx)))

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
    βR ~ Normal(0, 0.5)
    βS ~ Normal(0, 0.5)

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

        # Compute base effects for all individuals
        base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M + βR * R + βS * S

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
        base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M + βR * R + βS * S

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

# Generate prior predictive checks to assess model priors
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
        title_suffix="Factual Model", saveplot=SAVE_PLOTS)

    plot_prior_predictive_check(dag_df_infected.E, counterfactual_prior_E;
        title_suffix="Counterfactual Model", saveplot=SAVE_PLOTS)
end

# Assess prior adequacy
assess_prior_adequacy(dag_df_infected.E, factual_prior_E; model_name="Factual (With Parasites)")
assess_prior_adequacy(dag_df_infected.E, counterfactual_prior_E; model_name="Counterfactual (Without Parasites)")

## Model fitting

# Fit the pre-intervention model (reuse the factual_model from PPC)
println("\nFitting pre-intervention model...")
if AD_BACKEND !== nothing
    pre_intervention_chain = sample(factual_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
    pre_intervention_chain = sample(factual_model, NUTS(), MCMCThreads(), 3000, 4)
end

# Fit the post-intervention model (reuse the counterfactual_model from PPC)
println("Fitting post-intervention model...")
if AD_BACKEND !== nothing
    post_intervention_chain = sample(counterfactual_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
    post_intervention_chain = sample(counterfactual_model, NUTS(), MCMCThreads(), 3000, 4)
end

## Generate counterfactuals using posterior predictive approach (RECOMMENDED METHOD)

"""
    generate_counterfactuals(pre_chain, df_infected; n_samples=1000)

Generate counterfactual vaccine responses under parasite elimination intervention do(P=0).

Uses posterior predictive sampling to generate both factual (with parasites) and
counterfactual (without parasites) vaccine responses for each individual.

# Arguments
- `pre_chain`: MCMC chain from fitted factual model
- `df_infected`: DataFrame with infected mouse data
- `n_samples`: Number of posterior samples to draw

# Returns
- `E_factual`: Matrix of factual vaccine responses (n_obs × n_samples)
- `E_counterfactual`: Matrix of counterfactual vaccine responses (n_obs × n_samples)
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

        # Extract random effects
        n_unique_infected = length(unique(df_infected.ID))
        α_ID = Vector{Float64}(undef, n_unique_infected)
        for i in 1:n_unique_infected
            α_ID[i] = chain_samples[idx, Symbol("α_ID[$i]")]
        end

        # Generate predictions
        for i in 1:n_obs
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

# Generate counterfactuals using posterior predictive approach
println("Generating counterfactuals using posterior predictive approach...")
E_factual, E_counterfactual = generate_counterfactuals(pre_intervention_chain, dag_df_infected)

# Add predictions to dataframe - using this as our main approach
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

println("Counterfactual Results:")
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

## Analysis helper functions

"""
    calculate_counterfactual_percentage_change(df_infected; method="relative_to_baseline")

Calculate percentage change in vaccine response under counterfactual parasite elimination.

# Arguments
- `df_infected`: DataFrame with counterfactual predictions
- `method`: Calculation method. Options:
  - `"relative_to_baseline"`: Change relative to individual baseline values
  - `"relative_to_population"`: Change relative to population mean
  - `"relative_to_original_scale"`: Change in back-transformed OD scale
  - `"cohen_d_to_percent"`: Convert Cohen's d to percentage using pooled SD

# Returns
Vector of percentage changes for each observation (also prints summary statistics).

Different methods handle the standardised nature of the vaccine response differently.
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

    else
        error("Unknown method: $method. Choose from: relative_to_baseline, relative_to_population, relative_to_original_scale, cohen_d_to_percent")
    end

    # Summary statistics
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

Compares vaccination coefficients between factual and counterfactual scenarios
to determine how much better vaccines work when parasites are eliminated.

# Arguments
- `df_infected`: DataFrame with counterfactual predictions

# Returns
Named tuple with vaccination effect coefficients and improvement statistics.
"""
function calculate_vaccination_effect_percentage(df_infected)
    println("=== VACCINATION EFFECT PERCENTAGE IMPROVEMENT ===")

    # Fit models to get vaccination coefficients
    factual_model = fit(MixedModel, @formula(E_factual_mean ~ V + (1 | ID)), df_infected)
    counterfactual_model = fit(MixedModel, @formula(E_counterfactual_mean ~ V + (1 | ID)), df_infected)

    # Extract vaccination coefficients
    β_factual = coef(factual_model)[2]  # V coefficient
    β_counterfactual = coef(counterfactual_model)[2]  # V coefficient

    # Calculate percentage improvement in vaccination effect
    vaccination_improvement = ((β_counterfactual - β_factual) / abs(β_factual)) * 100

    println("Factual vaccination effect (β_V): $(round(β_factual, digits=3))")
    println("Counterfactual vaccination effect (β_V): $(round(β_counterfactual, digits=3))")
    println("Vaccination effect improvement: $(round(vaccination_improvement, digits=1))%")

    # Also calculate absolute improvement
    absolute_improvement = β_counterfactual - β_factual
    println("Absolute improvement: $(round(absolute_improvement, digits=3)) standardised units")

    return (
        factual_effect=β_factual,
        counterfactual_effect=β_counterfactual,
        percentage_improvement=vaccination_improvement,
        absolute_improvement=absolute_improvement
    )
end

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
    plot_E_factual_counterfactual(dag_df_infected, saveplot=SAVE_PLOTS)
end

with_theme(theme_minimal()) do
    plot_counterfactual_effects_with_significance(dag_df_infected, saveplot=SAVE_PLOTS)
end

with_theme(theme_minimal()) do
    plot_clinical_significance_summary(dag_df_infected, saveplot=SAVE_PLOTS)
end

## Interaction analysis with Cohen's d

# Fit model using Cohen's d as outcome
glmm_cohens_d_interaction = fit(MixedModel, @formula(E_cohens_d ~ -1 + D + R + S + V + M + Ḟ + S & R + (1 | ID)), dag_df_infected)

# Generate Cohen's d interaction plots
with_theme(theme_minimal()) do
    plot_S_R_interaction_cohens_d(dag_df_infected, saveplot=SAVE_PLOTS)
end

with_theme(theme_minimal()) do
    plot_S_R_interaction_factual_counterfactual(dag_df_infected, saveplot=SAVE_PLOTS)
end

# Generate detailed Sex × Reproductive status interaction plots
with_theme(theme_minimal()) do
    plot_sex_reproductive_interaction_detailed(dag_df_infected, saveplot=SAVE_PLOTS)
end

with_theme(theme_minimal()) do
    plot_sex_reproductive_heatmap(dag_df_infected, saveplot=SAVE_PLOTS)
end

# Diagnostics for Cohen's d interaction model
boot_cohens_d = parametricbootstrap(MersenneTwister(1234), 10_000, glmm_cohens_d_interaction)
coefplot(boot_cohens_d)



## Calculate percentage improvements using different methods

println("\n" * "="^60)
println("PERCENTAGE IMPROVEMENT CALCULATIONS")
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

# Debug: Check if Cohen's d values are all the same
println("DEBUG: Cohen's d values (first 10): ", dag_df_infected.E_cohens_d[1:min(10, nrow(dag_df_infected))])
println("DEBUG: Are all Cohen's d values equal? ", all(dag_df_infected.E_cohens_d .≈ dag_df_infected.E_cohens_d[1]))
println("DEBUG: Cohen's d mean: ", mean(dag_df_infected.E_cohens_d))
println("DEBUG: Cohen's d std: ", std(dag_df_infected.E_cohens_d))

println("\n" * "-"^30)

# Method 2d: Back-transformed to original scale (if OD data available)
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
println("SUMMARY RECOMMENDATIONS")
println("="^60)

println("For policy/clinical interpretation, use vaccination effect improvement:")
println("- Vaccine efficacy improves by $(round(vaccination_results.percentage_improvement, digits=1))% when parasites are eliminated")
println()
println("For individual-level analysis, Cohen's d method is most standardised:")
println("- Mean individual improvement: $(round(mean(percent_changes_cohens), digits=1))%")
println("- This represents the practical significance of parasite elimination for each mouse")

# Create summary table of individual-level improvement methods
println("\n" * "="^60)
println("INDIVIDUAL-LEVEL PERCENTAGE IMPROVEMENT METHODS")
println("="^60)

# Collect individual-level data for cleaner processing
individual_data = [percent_changes_baseline, percent_changes_population]
methods = ["Relative to own baseline", "Relative to population mean"]

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
