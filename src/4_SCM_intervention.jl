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
cd("./src/")
include("TuringUtils.jl")
include("TuringPlots.jl")

# Import data
include("DataWrangler.jl")

## Data preparation

# All cases - use more efficient filtering
df = encode_df(df) # Choose between df and df_unique (the latter has no repeated measures)
df = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df) # More efficient than Query.jl for simple filters
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

# Post-intervention DAG - P→E pathway removed
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
ridgeplot(boot)

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

    # Population-level priors (weakly informative)
    α ~ Normal(E_mean, 2.5 * E_std)  # Overall intercept

    # Structural coefficients for all variables
    βV ~ Normal(0, 0.5)   # Vaccination effect
    βD ~ Normal(0, 0.5)   # Diet effect
    βḞ ~ Normal(0, 0.5)   # Fat effect
    βH ~ Normal(0, 0.5)   # Habitat effect
    βM ~ Normal(0, 0.5)   # Mass effect
    βR ~ Normal(0, 0.5)   # Reproductive status effect
    βS ~ Normal(0, 0.5)   # Sex effect

    # Parasite effect - key parameter that changes under intervention
    if intervention
        # Post-intervention: P has no causal effect on E (pathway severed)
        βP ~ Normal(0, 0.001)  # Essentially constrained to 0
    else
        # Pre-intervention: P can causally influence E
        βP ~ Normal(0, 0.5)
    end

    # Residual variance
    σ ~ Exponential(E_std)

    # Random effects for individual-level variation
    τ ~ truncated(Cauchy(0, 2); lower=0)  # Individual-level SD
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

    # Shared population-level parameters
    α ~ Normal(E_mean, 2.5 * E_std)

    # Structural coefficients (shared across worlds)
    βV ~ Normal(0, 0.5)
    βD ~ Normal(0, 0.5)
    βḞ ~ Normal(0, 0.5)
    βH ~ Normal(0, 0.5)
    βM ~ Normal(0, 0.5)
    βP ~ Normal(0, 0.5)  # Parasite effect in factual world
    βR ~ Normal(0, 0.5)
    βS ~ Normal(0, 0.5)

    # Shared residual variance
    σ ~ Exponential(E_std)

    # Shared random effects (key for maintaining individual identity)
    τ ~ truncated(Cauchy(0, 2); lower=0)
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

## Model fitting

# Fit the pre-intervention model
println("Fitting pre-intervention model...")
pre_intervention_model = Counterfactual_E_Model(
    dag_df_infected.IDidx_infected, dag_df_infected.E, dag_df_infected.V,
    dag_df_infected.D, dag_df_infected.Ḟ, dag_df_infected.H,
    dag_df_infected.M, dag_df_infected.P, dag_df_infected.R,
    dag_df_infected.S, dag_df_infected.nP, dag_df_infected.post_P,
    dag_df_infected.post_nP; n_id=n_infected_ids, intervention=false
)

pre_intervention_chain = sample(pre_intervention_model, NUTS(), MCMCThreads(), 3000, 4)

# Fit the post-intervention model
println("Fitting post-intervention model...")
post_intervention_model = Counterfactual_E_Model(
    dag_df_infected.IDidx_infected, dag_df_infected.E, dag_df_infected.V,
    dag_df_infected.D, dag_df_infected.Ḟ, dag_df_infected.H,
    dag_df_infected.M, dag_df_infected.post_P, dag_df_infected.R,
    dag_df_infected.S, dag_df_infected.post_nP, dag_df_infected.post_P,
    dag_df_infected.post_nP; n_id=n_infected_ids, intervention=true
)

post_intervention_chain = sample(post_intervention_model, NUTS(), MCMCThreads(), 3000, 4)

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
    sample_indices = rand(1:n_chain_samples, n_samples)  # More efficient than sample()

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
            E_factual[i, s] = μ_factual + σ * randn()  # More efficient than rand(Normal())

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

println("Summary of counterfactual effects:")
println("Mean counterfactual effect (E_counterfactual - E_factual): ",
    mean(dag_df_infected.E_diff_counterfactual))
println("SD of counterfactual effects: ",
    std(dag_df_infected.E_diff_counterfactual))

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
ridgeplot(boot)

## Visualisation functions

"""
    plot_E_factual_counterfactual(df; saveplot=false)

Create a visualisation comparing factual and counterfactual vaccine response distributions.

# Arguments
- `df::DataFrame`: DataFrame containing the data with columns:
    - E_factual_mean: Factual vaccine response (with parasites)
    - E_counterfactual_mean: Counterfactual vaccine response (without parasites)
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Details
- Plots histograms of factual and counterfactual vaccine responses from generative models
- Factual data shown in light blue-grey (with parasites)
- Counterfactual data shown in orange (without parasites)
- Includes a legend in the top-left corner
- Axis labels are bold and sized at 16pt

# Returns
- `Figure`: A CairoMakie figure containing the distribution comparison plot
"""
function plot_E_factual_counterfactual(df; saveplot::Bool=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # Define custom colour (#c0c7db) - more efficient color definition
    light_blue_grey = RGB(0.7529f0, 0.7804f0, 0.8588f0)  # Use Float32 for efficiency

    # Plot histograms of factual and counterfactual responses
    hist!(ax, df.E_factual_mean, color=light_blue_grey, label="Factual (with parasites)")
    hist!(ax, df.E_counterfactual_mean, color=:orange, label="Counterfactual (without parasites)")
    axislegend(ax, position=:lt)

    # Configure axis labels and styling
    ax.xlabel = "Vaccine response"
    ax.ylabel = "Population count"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    # Save plot if requested
    if saveplot
        safe_plot_save("E_factual_counterfactual.pdf", fig)
    end
    fig
end

"""
    plot_counterfactual_effects(df; saveplot=false)

Create a visualisation of the counterfactual effect of parasite elimination on vaccine response for each mouse.
Optimized version with better performance for large datasets.

# Arguments
- `df::DataFrame`: DataFrame containing the data with columns:
    - IDidx: Mouse identifier
    - nP: Observed parasite count
    - E_factual_mean: Factual vaccine response (with parasites)
    - E_counterfactual_mean: Counterfactual vaccine response (without parasites)
    - S: Sex (1=male, 2=female)
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Details
- Plots factual and counterfactual vaccine responses from generative models for each mouse
- Uses vertical lines to connect factual and counterfactual points
- Orange lines indicate increased vaccine response under parasite elimination
- Black lines indicate decreased or unchanged vaccine response
- Factual points shown as circles (with parasites)
- Counterfactual points shown as triangles (without parasites)
- Sex-specific colours for point outlines

# Returns
- `Figure`: A CairoMakie figure containing the counterfactual effect plot
"""
function plot_counterfactual_effects(df; saveplot::Bool=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # Define colours - use symbols for efficiency
    male_color = :steelblue
    female_color = :crimson
    light_blue_grey = RGB(0.7529f0, 0.7804f0, 0.8588f0)  # Float32 for efficiency

    # Get unique mice once for efficiency
    unique_mice = unique(df.IDidx)

    # Pre-allocate vectors for batch plotting
    factual_x = Float64[]
    factual_y = Float64[]
    counterfactual_x = Float64[]
    counterfactual_y = Float64[]
    line_segments = Pair{Point2f,Point2f}[]

    for mouse in unique_mice
        mouse_mask = df.IDidx .== mouse
        mouse_data = view(df, mouse_mask, :)  # Use view for efficiency

        # Determine sex colour for the outline - assumes sex is constant for a mouse
        sex_outline_color = mouse_data.S[1] == 1 ? male_color : female_color

        for i in 1:length(mouse_data.nP)
            x_val = mouse_data.nP[i]
            factual_val = mouse_data.E_factual_mean[i]
            counterfactual_val = mouse_data.E_counterfactual_mean[i]

            # Store data for batch plotting
            push!(factual_x, x_val)
            push!(factual_y, factual_val)
            push!(counterfactual_x, x_val)
            push!(counterfactual_y, counterfactual_val)

            # Draw vertical lines connecting factual and counterfactual points
            line_color = factual_val < counterfactual_val ? :orange : :black
            lines!(ax, [x_val, x_val], [factual_val, counterfactual_val],
                color=line_color, linewidth=3, alpha=0.5)
        end

        # Plot factual and counterfactual points with sex-based outlines
        mouse_indices = findall(mouse_mask)
        scatter!(ax, mouse_data.nP, mouse_data.E_factual_mean,
            color=light_blue_grey, strokecolor=sex_outline_color, strokewidth=2,
            label="Factual (with parasites)", markersize=16, alpha=0.7)
        scatter!(ax, mouse_data.nP, mouse_data.E_counterfactual_mean,
            color=:orange, strokecolor=sex_outline_color, strokewidth=2,
            label="Counterfactual (without parasites)", marker=:utriangle,
            markersize=16, alpha=0.7)
    end

    # Configure axis labels and styling
    ax.xlabel = "Observed Parasite Count"
    ax.ylabel = "Vaccine response"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    # Add legend with sex-specific elements
    pre_male = MarkerElement(color=light_blue_grey, marker=:circle, markersize=16,
        strokecolor=male_color, strokewidth=2)
    pre_female = MarkerElement(color=light_blue_grey, marker=:circle, markersize=16,
        strokecolor=female_color, strokewidth=2)
    post_male = MarkerElement(color=:orange, marker=:utriangle, markersize=16,
        strokecolor=male_color, strokewidth=2)
    post_female = MarkerElement(color=:orange, marker=:utriangle, markersize=16,
        strokecolor=female_color, strokewidth=2)

    axislegend(ax,
        [pre_male, pre_female, post_male, post_female],
        ["Factual (Male)", "Factual (Female)", "Counterfactual (Male)", "Counterfactual (Female)"],
        position=:rt)

    # Save plot if requested
    if saveplot
        safe_plot_save("counterfactual_effects_on_E.pdf", fig)
    end

    fig
end

## Generate plots

with_theme(theme_minimal()) do
    plot_E_factual_counterfactual(dag_df_infected, saveplot=true)
end

with_theme(theme_minimal()) do
    plot_counterfactual_effects(dag_df_infected, saveplot=true)
end

## Interaction analysis: Sex × Reproductive status

# Fit model with interaction term
glmm_E_diff_interaction = fit(MixedModel, @formula(E_diff_counterfactual ~ -1 + D + R + S + V + M + Ḟ + S & R + (1 | ID)), dag_df_infected)

"""
    plot_S_R_interaction(df; saveplot=false)

Create an interaction plot visualising the effect of Sex (S) and Reproductive status (R)
on the counterfactual effect of parasite elimination.
Optimized version with better performance.

# Arguments
- `df::DataFrame`: DataFrame containing the data
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Returns
- `Figure`: A CairoMakie figure containing the interaction plot
"""
function plot_S_R_interaction(df; saveplot::Bool=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # Get unique values of Sex (S) and Reproductive status (R)
    S_values = sort(unique(df.S))  # Sort for consistency
    R_values = sort(unique(df.R))

    # Pre-allocate arrays with specific types for better performance
    n_S, n_R = length(S_values), length(R_values)
    means = Matrix{Float64}(undef, n_S, n_R)
    sems = Matrix{Float64}(undef, n_S, n_R)

    # Calculate means and standard errors for each combination
    for (i, s) in enumerate(S_values), (j, r) in enumerate(R_values)
        mask = (df.S .== s) .& (df.R .== r)
        values = view(df.E_diff_counterfactual, mask)  # Use view for efficiency
        n_vals = length(values)

        if n_vals > 0
            means[i, j] = mean(values)
            sems[i, j] = std(values) / sqrt(n_vals)
        else
            means[i, j] = NaN
            sems[i, j] = NaN
        end
    end

    # Define colours
    male_color = :steelblue
    female_color = :crimson

    # Plot lines and error bars for each Sex
    for (i, s) in enumerate(S_values)
        sex_color = s == 1 ? male_color : female_color
        sex_label = s == 1 ? "Male" : "Female"

        # Filter out NaN values
        valid_indices = .!isnan.(means[i, :])
        if any(valid_indices)
            valid_R = R_values[valid_indices]
            valid_means = means[i, valid_indices]
            valid_sems = sems[i, valid_indices]

            lines!(ax, valid_R, valid_means, label=sex_label, linewidth=2, color=sex_color)
            errorbars!(ax, valid_R, valid_means, valid_sems, color=sex_color, linewidth=1)
            scatter!(ax, valid_R, valid_means, marker=:circle, markersize=14, color=sex_color)
        end
    end

    # Configure axes
    ax.xticks = (R_values, ["Non-reproductive", "Reproductive"])
    ax.xreversed = true
    ax.xlabel = "Reproductive status"
    ax.ylabel = "Counterfactual effect of parasite elimination"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    axislegend(ax, position=:rt)

    if saveplot
        safe_plot_save("S_R_interaction.pdf", fig)
    end
    fig
end

"""
    plot_S_R_interaction_factual_counterfactual(df; saveplot=false)

Create an interaction plot showing factual vs counterfactual vaccine responses
by Sex and Reproductive status. Optimized version with better performance.

# Arguments
- `df::DataFrame`: DataFrame containing the data
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Returns
- `Figure`: A CairoMakie figure containing the interaction plot
"""
function plot_S_R_interaction_factual_counterfactual(df; saveplot::Bool=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    S_values = sort(unique(df.S))
    R_values = sort(unique(df.R))

    # Pre-allocate arrays with specific types
    n_S, n_R = length(S_values), length(R_values)
    means_factual = Matrix{Float64}(undef, n_S, n_R)
    means_counterfactual = Matrix{Float64}(undef, n_S, n_R)
    sems_factual = Matrix{Float64}(undef, n_S, n_R)
    sems_counterfactual = Matrix{Float64}(undef, n_S, n_R)

    # Calculate means and standard errors
    for (i, s) in enumerate(S_values), (j, r) in enumerate(R_values)
        mask = (df.S .== s) .& (df.R .== r)

        # Factual
        values_factual = view(df.E_factual_mean, mask)
        n_vals = length(values_factual)
        if n_vals > 0
            means_factual[i, j] = mean(values_factual)
            sems_factual[i, j] = std(values_factual) / sqrt(n_vals)
        else
            means_factual[i, j] = NaN
            sems_factual[i, j] = NaN
        end

        # Counterfactual
        values_counterfactual = view(df.E_counterfactual_mean, mask)
        if n_vals > 0
            means_counterfactual[i, j] = mean(values_counterfactual)
            sems_counterfactual[i, j] = std(values_counterfactual) / sqrt(n_vals)
        else
            means_counterfactual[i, j] = NaN
            sems_counterfactual[i, j] = NaN
        end
    end

    # Define colours
    male_color = :steelblue
    female_color = :crimson

    # Plot for each Sex and condition
    for (i, s) in enumerate(S_values)
        sex_color = s == 1 ? male_color : female_color
        sex_label = s == 1 ? "Male" : "Female"

        # Filter out NaN values
        valid_indices = .!isnan.(means_factual[i, :])
        if any(valid_indices)
            valid_R = R_values[valid_indices]

            # Factual (solid lines)
            valid_means_f = means_factual[i, valid_indices]
            valid_sems_f = sems_factual[i, valid_indices]
            lines!(ax, valid_R, valid_means_f, label="$(sex_label) (Factual)",
                linewidth=2, color=sex_color, linestyle=:solid)
            errorbars!(ax, valid_R, valid_means_f, valid_sems_f,
                color=sex_color, linewidth=1)
            scatter!(ax, valid_R, valid_means_f, marker=:circle, markersize=14, color=sex_color)

            # Counterfactual (dashed lines)
            valid_means_c = means_counterfactual[i, valid_indices]
            valid_sems_c = sems_counterfactual[i, valid_indices]
            lines!(ax, valid_R, valid_means_c, label="$(sex_label) (Counterfactual)",
                linewidth=2, color=sex_color, linestyle=:dash)
            errorbars!(ax, valid_R, valid_means_c, valid_sems_c,
                color=sex_color, linewidth=1)
            scatter!(ax, valid_R, valid_means_c, marker=:utriangle, markersize=14, color=sex_color)
        end
    end

    # Configure axes
    ax.xticks = (R_values, ["Non-reproductive", "Reproductive"])
    ax.xreversed = true
    ax.xlabel = "Reproductive status"
    ax.ylabel = "Standardised vaccine response (E)"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    axislegend(ax, position=:rt)

    if saveplot
        safe_plot_save("S_R_interaction_factual_counterfactual.pdf", fig)
    end
    fig
end

## Generate interaction plots

with_theme(theme_minimal()) do
    plot_S_R_interaction(dag_df_infected, saveplot=true)
end

with_theme(theme_minimal()) do
    plot_S_R_interaction_factual_counterfactual(dag_df_infected, saveplot=true)
end

# Ridge plot for interaction model comparison
boot = parametricbootstrap(MersenneTwister(1234), 10_000, glmm_E_diff_interaction)
ridgeplot(boot)
