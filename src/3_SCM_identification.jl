#=
SCM Identification
- Julia version: 1.11
- Author: Simon A Babayan
=#

#=
This script implements statistical identification of the Structural Causal Model (SCM)
for vaccine efficacy analysis in wood mice, using adjustment sets derived from the
causal DAG to estimate direct and total causal effects.

Key Features:
- Bayesian models with weakly informative priors for standardised outcomes
- Mixed-effects models for comparison and validation
- Comprehensive causal effect identification following do-calculus
- Proper adjustment for confounding and mediation
- Performance optimizations following Turing.jl and Julia best practices
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

using RCall
@rlibrary dagitty

# Plotting & diagnostics
using CairoMakie
CairoMakie.activate!(type="svg")
using MixedModelsMakie

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

# Automatic differentiation backend configuration
# Set to AutoZygote() or AutoReverseDiff() for large models, or leave as nothing for default
const AD_BACKEND = nothing

# Default plotting behaviour - set to true to save plots to disk
const SAVE_PLOTS = false

## Helper functions for causal effect analysis

"""
    fit_and_diagnose_mixedmodel(formula, data; n_bootstrap=3000, save_plots=false, plot_prefix="")

Fit a mixed-effects model with comprehensive diagnostics and bootstrap inference.

# Arguments
- `formula`: Model formula
- `data`: DataFrame with model data
- `n_bootstrap`: Number of bootstrap samples for inference
- `save_plots`: Whether to save diagnostic plots
- `plot_prefix`: Prefix for saved plot filenames

# Returns
- Fitted MixedModel object
- Bootstrap results for inference
"""
function fit_and_diagnose_mixedmodel(formula, data; n_bootstrap=3000, save_plots=false, plot_prefix="")
  # Fit model
  model = fit(MixedModel, formula, data)

  # Diagnostics
  qqnorm(model; qqline=:fitrobust)
  if save_plots && !isempty(plot_prefix)
    safe_plot_save("$(plot_prefix)_qqnorm.pdf", current_figure())
  end

  # Bootstrap inference
  println("Running bootstrap inference...")
  boot = parametricbootstrap(MersenneTwister(1234), n_bootstrap, model)

  # Generate plots
  coef_plot = coefplot(boot)
  ridge_plot = ridgeplot(boot)

  if save_plots && !isempty(plot_prefix)
    safe_plot_save("$(plot_prefix)_coefplot.pdf", coef_plot)
    safe_plot_save("$(plot_prefix)_ridgeplot.pdf", ridge_plot)
  end

  return model, boot
end

"""
    analyse_causal_effect(effect_name, adjustment_formula, data; n_bootstrap=3000)

Standardised analysis for a causal effect using mixed-effects modelling.

# Arguments
- `effect_name`: Descriptive name for the causal effect
- `adjustment_formula`: Model formula with proper adjustment set
- `data`: DataFrame with model data
- `n_bootstrap`: Number of bootstrap samples

# Returns
- Fitted model and bootstrap results
"""
function analyse_causal_effect(effect_name, adjustment_formula, data; n_bootstrap=3000)
  println("=== $effect_name ===")

  model, boot = fit_and_diagnose_mixedmodel(
    adjustment_formula,
    data;
    n_bootstrap=n_bootstrap,
    save_plots=false
  )

  return model, boot
end

## Data preparation

# Filter and encode data
df = encode_df(df)
df = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df)
df.IDidx = get_idx(:ID, df)[1]

# Restrict to unique cases (no repeated measures)
df_unique = encode_df(df_unique)
df_unique = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df_unique)

# Build DAG DataFrame with required variables
dag_df = select(df, :E, :H, :V, :D, :R, :S, :M, :Ḟ, :P, :nP, :ID, :IDidx, :vax_history, :Vidx)
dag_df.lognP = log10.(1 .+ dag_df.nP)

dag_df_unique = select(df_unique, :E, :H, :V, :D, :R, :S, :M, :Ḟ, :P, :nP, :Vidx, :vax_history, :ID)
filtered_df = dag_df[dag_df.P.==1, :]
filtered_unique_df = dag_df_unique[dag_df_unique.P.==1, :]

# Extract infected mice subset
infected_mask = dag_df.P .== 2
dag_df_infected = dag_df[infected_mask, :]

# Re-index mouse IDs for infected subset
unique_ids_infected = unique(dag_df_infected.ID)
n_infected_ids = length(unique_ids_infected)
id_mapping = Dict{eltype(unique_ids_infected),Int}()
for (i, id) in enumerate(unique_ids_infected)
  id_mapping[id] = i
end
dag_df_infected.IDidx_infected = [id_mapping[id] for id in dag_df_infected.ID]

## DAG specification

# Graphical causal hypothesis
dag = dagitty("dag{ D -> E; D -> F; D -> M; D -> P; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> P; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; R -> P; S -> E; S -> F; S -> M; S -> P; S -> R; V -> E; V -> F; V -> M; V -> P; V -> R; }")

## Average causal effect of V on E

adjustmentSets(dag, "V", "E", effect="total") # {} -> V is assumed to be a RCT

"""
    V_E_Model(IDidx, E, V; n_id=length(unique(IDidx)))

Bayesian model for the total causal effect of vaccination on vaccine response.

Since vaccination is randomised (RCT), no adjustment variables are needed
to identify the total causal effect.

# Arguments
- `IDidx`: Mouse identifier indices
- `E`: Standardised vaccine response (outcome)
- `V`: Vaccination status (1=adjuvant, 2=vaccine)
- `n_id`: Number of unique individuals

# Returns
Posterior samples for vaccination effect (βVE) and structural parameters.

# Model Structure
E ~ α + α_ID[IDidx] + βVE * V + ε
"""
@model function V_E_Model(IDidx, E, V; n_id=length(unique(IDidx)))
  # Population-level priors (weakly informative for standardised outcome)
  α ~ Normal(0, 1)
  βVE ~ Normal(0, 0.5)
  σ ~ Exponential(1)
  ν ~ LogNormal(2, 1)

  # Random intercepts for individual mice
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Likelihood for vaccine response
  Ê = @. α + α_ID[IDidx] + βVE * V
  return E ~ MvNormal(Ê, σ^2 * I)
end

# Prior predictive checks for V_E_Model
println("=== PRIOR PREDICTIVE CHECK: V→E Model ===")

# Generate prior predictions manually
n_prior_samples = 1000
n_obs = length(dag_df.E)
V_E_prior_pred_values = zeros(n_prior_samples * n_obs)

idx = 1
for i in 1:n_prior_samples
  # Sample from priors
  α = rand(Normal(0, 1))
  βVE = rand(Normal(0, 0.5))
  τ = rand(Exponential(1))
  α_ID = rand(Normal(0, τ), maximum(dag_df.IDidx))

  # Generate predictions
  Ê = α .+ α_ID[dag_df.IDidx] .+ βVE * dag_df.V
  V_E_prior_pred_values[idx:idx+n_obs-1] = Ê
  idx += n_obs
end

with_theme(theme_minimal()) do
  plot_prior_predictive_check(dag_df.E, V_E_prior_pred_values;
    title_suffix="V→E Model", saveplot=SAVE_PLOTS)
end
assess_prior_adequacy(dag_df.E, V_E_prior_pred_values; model_name="V→E Model")

# Sample from posterior
V_E_model = V_E_Model(dag_df.IDidx, dag_df.E, dag_df.V)
if AD_BACKEND !== nothing
  V_E_chn = sample(V_E_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  V_E_chn = sample(V_E_model, NUTS(), MCMCThreads(), 3000, 4)
end

V_E_chn_df = DataFrame(V_E_chn)[!, r"α\b|β"]
precis(V_E_chn_df)

plot_chains_df(V_E_chn)

## Direct effect of V on E

adjustmentSets(dag, "V", "E", effect="direct") # { D, F, H, M, P, R, S }

"""
    V_E_NDE_Model(IDidx, E, V, D, Ḟ, H, M, P, R, S; n_id=length(unique(IDidx)))

Bayesian model for the natural direct effect (NDE) of vaccination on vaccine response.

Adjusts for all mediating variables to isolate the direct vaccination pathway.

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
- `n_id`: Number of unique individuals

# Returns
Posterior samples for direct vaccination effect and adjustment coefficients.

# Model Structure
Adjusts for {D, F, H, M, P, R, S} to block all indirect pathways.
Handles missing fat scores via Bayesian imputation.
"""
@model function V_E_NDE_Model(IDidx, E, V, D, Ḟ, H, M, P, R, S; n_id=length(unique(IDidx)))
  # Population-level priors (weakly informative for standardised outcome)
  α ~ Normal(0, 1)
  βVE ~ Normal(0, 0.5)
  βDE ~ Normal(0, 0.5)
  βFE ~ Normal(0, 0.5)
  βHE ~ Normal(0, 1)    # Allow larger habitat effect
  βME ~ Normal(0, 0.5)
  βPE ~ Normal(0, 0.75) # Allow larger parasite effect
  βRE ~ Normal(0, 0.5)
  βSE ~ Normal(0, 0.5)
  σ ~ Exponential(1)
  ν ~ LogNormal(2, 1)

  # Random intercepts for individual mice
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Handle missing fat score data
  missing_mask = ismissing.(Ḟ)
  N_missing::Int = sum(missing_mask)

  if N_missing > 0
    missing_indices = findall(missing_mask)
    F_impute ~ filldist(Normal(0, 1), N_missing)
    ν_F ~ Normal(0, 0.5)
    σ_F ~ Exponential()

    # Create fat values vector with imputed missing values
    f_vals = map(i -> missing_mask[i] ? (i in missing_indices ? ν_F + σ_F * F_impute[findfirst(==(i), missing_indices)] : zero(eltype(α))) : Ḟ[i], 1:length(E))

    # Likelihood for observed fat values
    for i in 1:length(E)
      if !missing_mask[i]
        Ḟ[i] ~ Normal(ν_F, σ_F)
      end
    end

    # Likelihood for vaccine response
    for i in 1:length(E)
      Ê = α + α_ID[IDidx[i]] + βVE * V[i] + βDE * D[i] + βFE * f_vals[i] +
          βHE * H[i] + βME * M[i] + βPE * P[i] + βRE * R[i] + βSE * S[i]
      E[i] ~ Normal(Ê, σ)
    end
  else
    # Handle case with no missing fat data
    f_vals = [ismissing(f) ? zero(eltype(α)) : f for f in Ḟ]
    for i in 1:length(E)
      Ê = α + α_ID[IDidx[i]] + βVE * V[i] + βDE * D[i] + βFE * f_vals[i] +
          βHE * H[i] + βME * M[i] + βPE * P[i] + βRE * R[i] + βSE * S[i]
      E[i] ~ Normal(Ê, σ)
    end
  end
end

V_E_DE_model = V_E_NDE_Model(dag_df.IDidx, dag_df.E, dag_df.V, dag_df.D, dag_df.Ḟ,
  dag_df.H, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S)

if AD_BACKEND !== nothing
  V_E_DE_chn = sample(V_E_DE_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  V_E_DE_chn = sample(V_E_DE_model, NUTS(), MCMCThreads(), 3000, 4)
end

V_E_DE_chn_df = DataFrame(V_E_DE_chn)[!, r"α\b|β"]
precis(V_E_DE_chn_df)

## Mixed model comparison for V→E direct effect
glmm_V_E_NDE = fit(MixedModel, @formula(E ~ 1 + V + D + Ḟ + H + M + P + R + S + (1 | ID)), dag_df)

qqnorm(glmm_V_E_NDE; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_NDE)
coefplot(boot)
ridgeplot(boot)

## Total effect of nP on E

adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V}

# Plot correlation
layers = linear() + visual(Scatter)
plt = data(dag_df_infected) * mapping(:lognP, :E)
p = draw(layers * plt, axis=(xlabel="H. polygyrus worm count (log10(1 + nP))", ylabel="α-DT IgG1 (standardised)"))
safe_plot_save("P_E_cor.pdf", p)

# Naive model (without proper adjustment) - for comparison
naive_glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + (1 | ID)), dag_df_infected; progress=false)

qqnorm(naive_glmm_P_E; qqline=:fitrobust)

boot = parametricbootstrap(MersenneTwister(42), 3000, naive_glmm_P_E)
cp = coefplot(boot; conf_level=0.95)
safe_plot_save("P_E_coefplot.pdf", cp)
ridgeplot(boot; conf_level=0.95)
safe_plot_save("P_E_ridgeplot.pdf", cp)

"""
    Naive_P_E_Model(IDidx, E, nP; n_id=length(unique(IDidx)))

Naive Bayesian model for parasite effect on vaccine response without proper causal adjustment.

Used for comparison with properly adjusted model to demonstrate confounding bias.

# Arguments
- `IDidx`: Mouse identifier indices
- `E`: Standardised vaccine response (outcome)
- `nP`: Log-transformed parasite count
- `n_id`: Number of unique individuals

# Returns
Posterior samples for parasite effect (βPE) - likely confounded.

# Model Structure
E ~ α + α_ID[IDidx] + βPE * nP + ε
No adjustment for confounders - violates backdoor criterion.
"""
@model function Naive_P_E_Model(IDidx, E, nP; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  βPE ~ Normal(0, 0.75)
  σ ~ Exponential(1)

  # Random intercepts for individual mice
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Likelihood for vaccine response
  Ê = @. α + α_ID[IDidx] + βPE * nP
  return E ~ MvNormal(Ê, σ^2 * I)
end

# Prior predictive checks
println("=== PRIOR PREDICTIVE CHECK: Naive P→E Model ===")

# Generate prior predictions manually for infected mice
n_prior_samples = 1000
n_obs_infected = length(dag_df_infected.E)
naive_P_E_prior_pred_values = zeros(n_prior_samples * n_obs_infected)

idx = 1
for i in 1:n_prior_samples
  # Sample from priors
  α = rand(Normal(0, 1))
  βPE = rand(Normal(0, 0.75))
  τ = rand(Exponential(1))
  α_ID = rand(Normal(0, τ), maximum(dag_df_infected.IDidx))

  # Generate predictions
  Ê = α .+ α_ID[dag_df_infected.IDidx] .+ βPE * dag_df_infected.lognP
  naive_P_E_prior_pred_values[idx:idx+n_obs_infected-1] = Ê
  idx += n_obs_infected
end

with_theme(theme_minimal()) do
  plot_prior_predictive_check(dag_df_infected.E, naive_P_E_prior_pred_values;
    title_suffix="Naive P→E Model", saveplot=SAVE_PLOTS)
end
assess_prior_adequacy(dag_df_infected.E, naive_P_E_prior_pred_values; model_name="Naive P→E Model")

# Sample from posterior
naive_P_E_model = Naive_P_E_Model(dag_df_infected.IDidx, dag_df_infected.E, dag_df_infected.lognP; n_id=maximum(dag_df_infected.IDidx))
if AD_BACKEND !== nothing
  naive_P_E_chn = sample(naive_P_E_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  naive_P_E_chn = sample(naive_P_E_model, NUTS(), MCMCThreads(), 3000, 4)
end

naive_P_E_chn_df = DataFrame(naive_P_E_chn)[!, r"α\b|β"]
precis(naive_P_E_chn_df)

p = plot_chains_df(naive_P_E_chn; show_intercept=true, show_traces=false)
safe_plot_save("naive_P_E_chn_df.pdf", p)

## Properly-adjusted model: total effect of nP on E

adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V }

# Mixed models - comparison across all mice and infected only
glmm_P_E_all = fit(MixedModel, @formula(E ~ 1 + lognP + D + H + R + S + V + (1 | ID)), dag_df)
glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + H + R + S + V + (1 | ID)), dag_df_infected)

# Diagnostics for infected-only model
qqnorm(glmm_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E)
coefplot(boot)
ridgeplot(boot)
ridge2d(boot)

# Direct effect of P on E among infected
adjustmentSets(dag, "P", "E", effect="direct") # { D, F, H, M, R, S, V}

glmm_DE_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + Ḟ + M + R + S + V + (1 | ID)), dag_df_infected)

qqnorm(glmm_DE_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_DE_P_E)
coefplot(boot)
ridgeplot(boot)

"""
    P_E_Model(IDidx, E, P, D, H, R, S, V; n_id=length(unique(IDidx)))

Bayesian model for the total causal effect of parasite burden on vaccine response.

Properly adjusted for confounders using minimal sufficient adjustment set.

# Arguments
- `IDidx`: Mouse identifier indices
- `E`: Standardised vaccine response (outcome)
- `P`: Log-transformed parasite count
- `D`: Diet supplementation (confounding adjustment)
- `H`: Habitat (confounding adjustment)
- `R`: Reproductive status (confounding adjustment)
- `S`: Sex (confounding adjustment)
- `V`: Vaccination status (confounding adjustment)
- `n_id`: Number of unique individuals

# Returns
Posterior samples for causal parasite effect and adjustment coefficients.

# Model Structure
Adjusts for {D, H, R, S, V} to satisfy backdoor criterion.
Estimates total causal effect P → E.
"""
@model function P_E_Model(IDidx, E, P, D, H, R, S, V; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  βPE ~ Normal(0, 0.75)
  βDE ~ Normal(0, 0.5)
  βHE ~ Normal(0, 1)
  βRE ~ Normal(0, 0.5)
  βSE ~ Normal(0, 0.5)
  βVE ~ Normal(0, 0.5)
  σ ~ Exponential(1)
  ν ~ LogNormal(2, 1)

  # Random intercepts for individual mice
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Likelihood for vaccine response
  Ê = α .+ α_ID[IDidx] .+ βPE * P .+ βDE * D .+ βHE * H .+ βRE * R .+ βSE * S .+ βVE * V
  return E ~ MvNormal(Ê, σ^2 * I)
end

# Prior predictive checks
println("=== PRIOR PREDICTIVE CHECK: P→E Model (Properly Adjusted) ===")

# Generate prior predictions manually for properly adjusted model
n_prior_samples = 1000
n_obs_infected = length(dag_df_infected.E)
P_E_prior_pred_values = zeros(n_prior_samples * n_obs_infected)

idx = 1
for i in 1:n_prior_samples
  # Sample from priors
  α = rand(Normal(0, 1))
  βPE = rand(Normal(0, 0.75))
  βDE = rand(Normal(0, 0.5))
  βHE = rand(Normal(0, 1))
  βRE = rand(Normal(0, 0.5))
  βSE = rand(Normal(0, 0.5))
  βVE = rand(Normal(0, 0.5))
  τ = rand(Exponential(1))
  α_ID = rand(Normal(0, τ), maximum(dag_df_infected.IDidx))

  # Generate predictions
  Ê = α .+ α_ID[dag_df_infected.IDidx] .+ βPE * log10.(1 .+ dag_df_infected.nP) .+
      βDE * dag_df_infected.D .+ βHE * dag_df_infected.H .+ βRE * dag_df_infected.R .+
      βSE * dag_df_infected.S .+ βVE * dag_df_infected.V
  P_E_prior_pred_values[idx:idx+n_obs_infected-1] = Ê
  idx += n_obs_infected
end

with_theme(theme_minimal()) do
  plot_prior_predictive_check(dag_df_infected.E, P_E_prior_pred_values;
    title_suffix="P→E Model (Adjusted)", saveplot=SAVE_PLOTS)
end
assess_prior_adequacy(dag_df_infected.E, P_E_prior_pred_values; model_name="P→E Model (Properly Adjusted)")

# Extended prior sampling
P_E_model = P_E_Model(dag_df_infected.IDidx, dag_df_infected.E, log10.(1 .+ dag_df_infected.nP),
  dag_df_infected.D, dag_df_infected.H, dag_df_infected.R,
  dag_df_infected.S, dag_df_infected.V; n_id=maximum(dag_df_infected.IDidx))
P_E_priors = sample(P_E_model, Prior(), MCMCThreads(), 3000, 4)
summarize(P_E_priors)

P_E_priors_df = DataFrame(P_E_priors)[!, r"α\b|β"]
precis(P_E_priors_df)

p = plot_chains_df(P_E_priors; show_intercept=true)
safe_plot_save("P_E_priors.pdf", p)
p

# Posterior sampling
if AD_BACKEND !== nothing
  P_E_chn = sample(P_E_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  P_E_chn = sample(P_E_model, NUTS(), MCMCThreads(), 3000, 4)
end

P_E_chn_df = DataFrame(P_E_chn)[!, r"α\b|β"]
precis(P_E_chn_df)

p = plot_chains_df(P_E_chn)
safe_plot_save("P_E_chn_df.pdf", p)

## Direct effect of P on E

adjustmentSets(dag, "P", "E", effect="direct") # { D, F, H, M, R, S, V }

# Mixed-effects model for comparison
glmm_DE_P_E = fit(MixedModel, @formula(E ~ 1 + P + D + Ḟ + H + M + R + S + V + (1 | ID)), dag_df)

qqnorm(glmm_DE_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_DE_P_E)
coefplot(boot)
ridgeplot(boot)

"""
    DE_P_E_Model(IDidx, E, P, D, Ḟ, H, M, R, S, V; n_id=length(unique(IDidx)))

Bayesian model for the natural direct effect of parasite infection on vaccine response.

Blocks all indirect pathways through mediating variables.

# Arguments
- `IDidx`: Mouse identifier indices
- `E`: Standardised vaccine response (outcome)
- `P`: Parasite infection status (1=uninfected, 2=infected)
- `D`: Diet supplementation (mediator adjustment)
- `Ḟ`: Standardised fat scores (mediator adjustment)
- `H`: Habitat (confounder adjustment)
- `M`: Standardised mass (mediator adjustment)
- `R`: Reproductive status (mediator adjustment)
- `S`: Sex (confounder adjustment)
- `V`: Vaccination status (mediator adjustment)
- `n_id`: Number of unique individuals

# Returns
Posterior samples for direct parasite effect, blocking mediation pathways.

# Model Structure
Adjusts for {D, F, H, M, R, S, V} to block indirect effects.
Estimates direct effect P → E (not mediated through other variables).
Handles missing fat scores via Bayesian imputation.
"""
@model function DE_P_E_Model(IDidx, E, P, D, Ḟ, H, M, R, S, V; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  βPE ~ Normal(0, 0.75)
  βDE ~ Normal(0, 0.5)
  βFE ~ Normal(0, 0.5)
  βHE ~ Normal(0, 1)
  βME ~ Normal(0, 0.5)
  βRE ~ Normal(0, 0.5)
  βSE ~ Normal(0, 0.5)
  βVE ~ Normal(0, 0.5)
  σ ~ Exponential(1)

  # Random intercepts for individual mice
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Handle missing fat score data
  missing_mask = ismissing.(Ḟ)
  N_missing::Int = sum(missing_mask)

  if N_missing > 0
    missing_indices = findall(missing_mask)
    F_impute ~ filldist(Normal(0, 1), N_missing)
    ν_F ~ Normal(0, 0.5)
    σ_F ~ Exponential()

    # Create fat values vector with imputed missing values
    f_vals = map(i -> missing_mask[i] ? (i in missing_indices ? ν_F + σ_F * F_impute[findfirst(==(i), missing_indices)] : zero(eltype(α))) : Ḟ[i], 1:length(E))

    # Likelihood for observed fat values
    for i in 1:length(E)
      if !missing_mask[i]
        Ḟ[i] ~ Normal(ν_F, σ_F)
      end
    end

    # Likelihood for vaccine response
    for i in 1:length(E)
      Ê = α + α_ID[IDidx[i]] + βPE * P[i] + βDE * D[i] + βFE * f_vals[i] +
          βHE * H[i] + βME * M[i] + βRE * R[i] + βSE * S[i] + βVE * V[i]
      E[i] ~ Normal(Ê, σ)
    end
  else
    # Handle case with no missing fat data
    f_vals = [ismissing(f) ? zero(eltype(α)) : f for f in Ḟ]
    for i in 1:length(E)
      Ê = α + α_ID[IDidx[i]] + βPE * P[i] + βDE * D[i] + βFE * f_vals[i] +
          βHE * H[i] + βME * M[i] + βRE * R[i] + βSE * S[i] + βVE * V[i]
      E[i] ~ Normal(Ê, σ)
    end
  end
end

DE_P_E_model = DE_P_E_Model(dag_df.IDidx, dag_df.E, dag_df.P, dag_df.D, dag_df.Ḟ,
  dag_df.H, dag_df.M, dag_df.R, dag_df.S, dag_df.V)

Turing.setadbackend(:forwarddiff)
if AD_BACKEND !== nothing
  DE_P_E_chn = sample(DE_P_E_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  DE_P_E_chn = sample(DE_P_E_model, NUTS(), MCMCThreads(), 3000, 4)
end
Turing.setadbackend(:reversediff)

DE_P_E_chn_df = DataFrame(DE_P_E_chn)[!, r"α\b|β"]
precis(DE_P_E_chn_df)

p1 = plot_chains_df(DE_P_E_chn; show_traces=true)
safe_plot_save("MultiLevel_DE_P_E_chn_traces.pdf", p1)

p2 = plot_chains_df(DE_P_E_chn; show_traces=false, xlab_dist="Direct Causal Effect")
safe_plot_save("MultiLevel_DE_P_E_chn.pdf", p2)

## Direct effect of reproductive status R on parasite burden P

adjustmentSets(dag, "R", "P", effect="direct") # D, H, S, V

# Mixed-effects model
glmm_R_nP = fit(MixedModel, @formula(lognP ~ R + D + H + S + V + (1 | ID)), dag_df)

"""
    R_nP_Model(nP, R, D, H, S, V, IDidx; n_id=length(unique(IDidx)))

Bayesian model for the direct causal effect of reproductive status on parasite burden.

# Arguments
- `nP`: Log-transformed parasite count (outcome)
- `R`: Reproductive status (1=non-reproductive, 2=reproductive)
- `D`: Diet supplementation (confounder adjustment)
- `H`: Habitat (confounder adjustment)
- `S`: Sex (confounder adjustment)
- `V`: Vaccination status (confounder adjustment)
- `IDidx`: Mouse identifier indices
- `n_id`: Number of unique individuals

# Returns
Posterior samples for direct reproductive effect on parasite burden.

# Model Structure
Adjusts for {D, H, S, V} to satisfy backdoor criterion.
Estimates direct effect R → P.
"""
@model function R_nP_Model(nP, R, D, H, S, V, IDidx; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 2)
  βR ~ Normal(0, 0.75)
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 1)
  βS ~ Normal(0, 0.5)
  βV ~ Normal(0, 0.5)
  σ ~ Exponential(1.5)

  # Random intercepts for individual mice
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Likelihood for parasite burden
  nP̂ = @. α + α_ID[IDidx] + βR * R + βD * D + βH * H + βS * S + βV * V
  nP ~ MvNormal(nP̂, σ^2 * I)
end

R_nP_model = R_nP_Model(log10.(1 .+ dag_df.nP), dag_df.R, dag_df.D, dag_df.H, dag_df.S, dag_df.V, dag_df.IDidx)

if AD_BACKEND !== nothing
  R_nP_chn = sample(R_nP_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3_000, 4)
else
  R_nP_chn = sample(R_nP_model, NUTS(), MCMCThreads(), 3_000, 4)
end

R_nP_chn_df = DataFrame(R_nP_chn)[!, r"α\b|β"]
precis(R_nP_chn_df)

p = plot_chains_df(R_nP_chn; show_intercept=true)
safe_plot_save("R_nP_chn.pdf", p)

## Total effect of D on nP

adjustmentSets(dag, "D", "P", effect="total") # {H}

"""
    D_nP_Total_Model(nP, D, H, IDidx; n_id=length(unique(IDidx)))

Bayesian model for the total causal effect of diet supplementation on parasite burden.

# Arguments
- `nP`: Log-transformed parasite count (outcome)
- `D`: Diet supplementation (1=low, 2=high)
- `H`: Habitat (confounder adjustment)
- `IDidx`: Mouse identifier indices
- `n_id`: Number of unique individuals

# Returns
Posterior samples for total diet effect on parasite burden.

# Model Structure
Adjusts for {H} to satisfy backdoor criterion.
Estimates total effect D → P.
"""
@model function D_nP_Total_Model(nP, D, H, IDidx; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 2)
  βD ~ Normal(0, 0.75)
  βH ~ Normal(0, 1)
  σ ~ Exponential(1.5)

  # Random intercepts
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Likelihood
  nP̂ = @. α + α_ID[IDidx] + βD * D + βH * H
  nP ~ MvNormal(nP̂, σ^2 * I)
end

D_nP_model = D_nP_Total_Model(log10.(1 .+ dag_df.nP), dag_df.D, dag_df.H, dag_df.IDidx)

if AD_BACKEND !== nothing
  D_nP_chn = sample(D_nP_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3_000, 4)
else
  D_nP_chn = sample(D_nP_model, NUTS(), MCMCThreads(), 3_000, 4)
end
D_nP_chn_df = DataFrame(D_nP_chn)[!, r"α\b|β"]
precis(D_nP_chn_df)

# Mixed-effects model for total effect among infected
glmm_D_nP = fit(MixedModel, @formula(lognP ~ D + (1 | ID)), dag_df_infected)

boot = parametricbootstrap(MersenneTwister(1234), 10000, glmm_D_nP)
coefplot(boot)
ridgeplot(boot)

# Plot diet effect on parasite burden
df_plot = (; x=Bool.(dag_df.D[dag_df.P.==2] .- 1), y=dag_df.nP[dag_df.P.==2])
layers = visual(BoxPlot)
plt = data(df_plot) * mapping(:x, :y, color=:x)
p = draw(layers * plt, axis=(xlabel="Diet supplemented", ylabel="H. polygyrus (count)"))

## Direct effect of D on nP

adjustmentSets(dag, "D", "P", effect="direct") # { H, R, S, V}

"""
    D_nP_Direct_Model(nP, D, H, R, S, V, IDidx; n_id=length(unique(IDidx)))

Bayesian model for the direct causal effect of diet supplementation on parasite burden.

Blocks indirect pathways through mediating variables.

# Arguments
- `nP`: Log-transformed parasite count (outcome)
- `D`: Diet supplementation (1=low, 2=high)
- `H`: Habitat (confounder adjustment)
- `R`: Reproductive status (mediator adjustment)
- `S`: Sex (mediator adjustment)
- `V`: Vaccination status (mediator adjustment)
- `IDidx`: Mouse identifier indices
- `n_id`: Number of unique individuals

# Returns
Posterior samples for direct diet effect on parasite burden.

# Model Structure
Adjusts for {H, R, S, V} to block indirect pathways.
Estimates direct effect D → P (not mediated).
"""
@model function D_nP_Direct_Model(nP, D, H, R, S, V, IDidx; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 2.5)
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 0.5)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)
  βV ~ Normal(0, 0.5)
  σ ~ Exponential(std(nP))

  # Random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Likelihood
  nP̂ = @. α + α_ID[IDidx] + βD * D + βH * H + βR * R + βS * S + βV * V
  nP ~ MvNormal(nP̂, σ^2 * I)
end

D_nP_direct_model = D_nP_Direct_Model(log10.(1 .+ dag_df.nP), dag_df.D, dag_df.H, dag_df.R, dag_df.S, dag_df.V, dag_df.IDidx)

if AD_BACKEND !== nothing
  D_nP_direct_chn = sample(D_nP_direct_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3_000, 4)
else
  D_nP_direct_chn = sample(D_nP_direct_model, NUTS(), MCMCThreads(), 3_000, 4)
end

D_nP_direct_chn_df = DataFrame(D_nP_direct_chn)[!, r"α\b|β"]
precis(D_nP_direct_chn_df)

p = plot_chains_df(D_nP_direct_chn; show_intercept=true, show_traces=false)
safe_plot_save("D_nP_chn.pdf", p)

## Total effect of S on E

adjustmentSets(dag, "S", "E", effect="total") # { }

glmm_S_E = fit(MixedModel, @formula(E ~ 1 + S + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_E)
coefplot(boot)
ridgeplot(boot)

"""
    S_E_Total_Model(IDidx, E, S; n_id=length(unique(IDidx)))

Bayesian model for the total causal effect of sex on vaccine response.

# Arguments
- `IDidx`: Mouse identifier indices
- `E`: Standardised vaccine response (outcome)
- `S`: Sex (1=male, 2=female)
- `n_id`: Number of unique individuals

# Returns
Posterior samples for sex effect on vaccine response.

# Model Structure
E ~ α + α_ID[IDidx] + βS * S + ε
No adjustment needed for total effect (no confounders).
"""
@model function S_E_Total_Model(IDidx, E, S; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  βS ~ Normal(0, 0.5)
  σ ~ Exponential(1)

  # Random intercepts for individual mice
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Likelihood for vaccine response
  Ê = @. α + α_ID[IDidx] + βS * S
  return E ~ MvNormal(Ê, σ^2 * I)
end

S_E_total_model = S_E_Total_Model(dag_df.IDidx, dag_df.E, dag_df.S)

if AD_BACKEND !== nothing
  S_E_total_chn = sample(S_E_total_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  S_E_total_chn = sample(S_E_total_model, NUTS(), MCMCThreads(), 3000, 4)
end

S_E_total_chn_df = DataFrame(S_E_total_chn)[!, r"α\b|β"]
precis(S_E_total_chn_df)

## Direct effect of S on E

adjustmentSets(dag, "S", "E", effect="direct") # { D, F, H, M, P, R, V }

glmm_S_E_direct = fit(MixedModel, @formula(E ~ 1 + S + D + Ḟ + H + M + nP + R + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_E_direct)
coefplot(boot)
ridgeplot(boot)

"""
    S_E_Direct_Model(E, S, D, Ḟ, H, M, P, R, V, IDidx; n_id=length(unique(IDidx)))

Bayesian model for the direct causal effect of sex on vaccine response.

Blocks all indirect pathways through mediating variables.

# Arguments
- `E`: Standardised vaccine response (outcome)
- `S`: Sex (1=male, 2=female)
- `D`: Diet supplementation (mediator adjustment)
- `Ḟ`: Standardised fat scores (mediator adjustment)
- `H`: Habitat (confounder adjustment)
- `M`: Standardised mass (mediator adjustment)
- `P`: Parasite burden (mediator adjustment)
- `R`: Reproductive status (mediator adjustment)
- `V`: Vaccination status (mediator adjustment)
- `IDidx`: Mouse identifier indices
- `n_id`: Number of unique individuals

# Returns
Posterior samples for direct sex effect on vaccine response.

# Model Structure
Adjusts for {D, F, H, M, P, R, V} to block indirect effects.
Handles missing fat scores via Bayesian imputation.
"""
@model function S_E_Direct_Model(E, S, D, Ḟ, H, M, P, R, V, IDidx; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  σ ~ Exponential(1)

  βS ~ Normal(0, 0.5)
  βD ~ Normal(0, 0.5)
  βF ~ Normal(0, 0.5)
  βH ~ Normal(0, 1)
  βM ~ Normal(0, 0.5)
  βP ~ Normal(0, 0.75)
  βR ~ Normal(0, 0.5)
  βV ~ Normal(0, 0.5)

  # Random intercepts for individual mice
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Handle missing fat score data
  missing_mask = ismissing.(Ḟ)
  N_missing::Int = sum(missing_mask)

  if N_missing > 0
    missing_indices = findall(missing_mask)
    F_impute ~ filldist(Normal(0, 1), N_missing)
    ν_F ~ Normal(0, 0.5)
    σ_F ~ Exponential()

    # Create fat values vector with imputed missing values
    f_vals = map(i -> missing_mask[i] ? (i in missing_indices ? ν_F + σ_F * F_impute[findfirst(==(i), missing_indices)] : zero(eltype(α))) : Ḟ[i], 1:length(E))

    # Likelihood for observed fat values
    for i in 1:length(E)
      if !missing_mask[i]
        Ḟ[i] ~ Normal(ν_F, σ_F)
      end
    end

    # Likelihood for vaccine response
    for i in 1:length(E)
      μ = α + α_ID[IDidx[i]] + βS * S[i] + βD * D[i] + βF * f_vals[i] +
          βH * H[i] + βM * M[i] + βP * P[i] + βR * R[i] + βV * V[i]
      E[i] ~ Normal(μ, σ)
    end
  else
    # Handle case with no missing fat data
    f_vals = [ismissing(f) ? zero(eltype(α)) : f for f in Ḟ]
    for i in 1:length(E)
      μ = α + α_ID[IDidx[i]] + βS * S[i] + βD * D[i] + βF * f_vals[i] +
          βH * H[i] + βM * M[i] + βP * P[i] + βR * R[i] + βV * V[i]
      E[i] ~ Normal(μ, σ)
    end
  end
end

S_E_direct_model = S_E_Direct_Model(dag_df.E, dag_df.S, dag_df.D, dag_df.Ḟ, dag_df.H,
  dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.V, dag_df.IDidx)

Turing.setadbackend(:forwarddiff)
S_E_direct_chn = sample(S_E_direct_model, NUTS(), MCMCThreads(), 3000, 4)
Turing.setadbackend(:reversediff)

S_E_direct_chn_df = DataFrame(S_E_direct_chn)[!, r"α\b|β"]
precis(S_E_direct_chn_df)

## Direct effect of S on nP

adjustmentSets(dag, "S", "P", effect="direct") # {D, H, R, V}

"""
    S_nP_Model(nP, S, D, H, R, V, IDidx; n_id=length(unique(IDidx)))

Bayesian model for the direct causal effect of sex on parasite burden.

# Arguments
- `nP`: Log-transformed parasite count (outcome)
- `S`: Sex (1=male, 2=female)
- `D`: Diet supplementation (confounder adjustment)
- `H`: Habitat (confounder adjustment)
- `R`: Reproductive status (mediator adjustment)
- `V`: Vaccination status (confounder adjustment)
- `IDidx`: Mouse identifier indices
- `n_id`: Number of unique individuals

# Returns
Posterior samples for direct sex effect on parasite burden.

# Model Structure
Adjusts for {D, H, R, V} to block confounding pathways.
Estimates direct effect S → P.
"""
@model function S_nP_Model(nP, S, D, H, R, V, IDidx; n_id=length(unique(IDidx)))
  # Population-level priors
  α ~ Normal(0, 2)
  βS ~ Normal(0, 0.75)
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 1)
  βR ~ Normal(0, 0.5)
  βV ~ Normal(0, 0.5)
  σ ~ Exponential(1.5)

  # Random intercepts
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)

  # Likelihood
  nP̂ = @. α + α_ID[IDidx] + βS * S + βD * D + βH * H + βR * R + βV * V
  nP ~ MvNormal(nP̂, σ^2 * I)
end

S_nP_model = S_nP_Model(log10.(1 .+ dag_df.nP), dag_df.S, dag_df.D, dag_df.H, dag_df.R, dag_df.V, dag_df.IDidx)

if AD_BACKEND !== nothing
  S_nP_chn = sample(S_nP_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3_000, 4)
else
  S_nP_chn = sample(S_nP_model, NUTS(), MCMCThreads(), 3_000, 4)
end

S_nP_chn_df = DataFrame(S_nP_chn)[!, r"α\b|β"]
precis(S_nP_chn_df)

p = plot_chains_df(S_nP_chn; show_intercept=true, show_traces=false)
safe_plot_save("S_nP_chn.pdf", p)

## Total effect of H on E

adjustmentSets(dag, "H", "E") # { }

"""
    H_E_Total_Model(IDidx, Vidx, E, H; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))

Bayesian model for the total causal effect of habitat on vaccine response.

# Arguments
- `IDidx`: Mouse identifier indices
- `Vidx`: Vaccination history indices
- `E`: Standardised vaccine response (outcome)
- `H`: Habitat (1=lab, 2=wild)
- `n_id`: Number of unique individuals
- `n_vax`: Number of unique vaccination histories

# Returns
Posterior samples for habitat effect on vaccine response.

# Model Structure
E ~ α + α_ID[IDidx] + α_vax[Vidx] + βH * H + ε
No adjustment needed for total effect (no confounders).
"""
@model function H_E_Total_Model(IDidx, Vidx, E, H; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  βH ~ Normal(0, 1)  # Allow larger habitat effect
  σ ~ Exponential(1)
  ν ~ LogNormal(2, 1)

  # Random intercepts
  τ ~ Exponential(1)
  τᵦ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)
  α_vax ~ filldist(Normal(0, τᵦ), n_vax)

  # Likelihood
  Ê = @. α + α_ID[IDidx] + α_vax[Vidx] + βH * H
  return E ~ MvNormal(Ê, σ^2 * I)
end

H_E_total_model = H_E_Total_Model(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.H)

if AD_BACKEND !== nothing
  H_E_total_chn = sample(H_E_total_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  H_E_total_chn = sample(H_E_total_model, NUTS(), MCMCThreads(), 3000, 4)
end

H_E_total_chn_df = DataFrame(H_E_total_chn)[!, r"α\b|β"]
precis(H_E_total_chn_df)

p = plot_chains_df(H_E_total_chn; show_intercept=true)
safe_plot_save("H_E_total_chn.pdf", p)

## Direct effect of H on E

adjustmentSets(dag, "H", "E", effect="direct") # { D, F, M, P, R, S, V }

"""
    H_E_Direct_Model(IDidx, Vidx, E, H, D, Ḟ, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))

Bayesian model for the direct causal effect of habitat on vaccine response.

Blocks all indirect pathways through mediating variables.

# Arguments
- `IDidx`: Mouse identifier indices
- `Vidx`: Vaccination history indices
- `E`: Standardised vaccine response (outcome)
- `H`: Habitat (1=lab, 2=wild)
- `D`: Diet supplementation (mediator adjustment)
- `Ḟ`: Standardised fat scores (mediator adjustment)
- `M`: Standardised mass (mediator adjustment)
- `P`: Parasite burden (mediator adjustment)
- `R`: Reproductive status (mediator adjustment)
- `S`: Sex (confounder adjustment)
- `n_id`: Number of unique individuals
- `n_vax`: Number of unique vaccination histories

# Returns
Posterior samples for direct habitat effect on vaccine response.

# Model Structure
Adjusts for {D, F, M, P, R, S, V} to block indirect effects.
Handles missing fat scores via Bayesian imputation.
"""
@model function H_E_Direct_Model(IDidx, Vidx, E, H, D, Ḟ, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  σ ~ Exponential(1)

  βH ~ Normal(0, 1)    # Allow larger habitat effect
  βD ~ Normal(0, 0.5)
  βF ~ Normal(0, 0.5)
  βM ~ Normal(0, 0.5)
  βP ~ Normal(0, 0.75)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)

  # Random intercepts
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)
  α_vax ~ filldist(Normal(0, τ), n_vax)

  # Missing fat value imputation
  N_missing = sum(ismissing.(Ḟ))
  if N_missing > 0
    F_impute ~ filldist(Normal(0, 1), N_missing)
    ν_F ~ Normal(0, 0.5)
    σ_F ~ Exponential()
  end

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      if N_missing > 0
        f_imputed = ν_F + σ_F * F_impute[i_missing]
        i_missing += 1
      else
        f_imputed = 0.0
      end
    else
      if N_missing > 0
        Ḟ[i] ~ Normal(ν_F, σ_F)
      end
      f_imputed = Ḟ[i]
    end

    # Likelihood
    Ê = α + α_ID[IDidx[i]] + α_vax[Vidx[i]] + βH * H[i] + βD * D[i] +
        βF * f_imputed + βM * M[i] + βP * P[i] + βR * R[i] + βS * S[i]
    E[i] ~ Normal(Ê, σ)
  end
end

H_E_direct_model = H_E_Direct_Model(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.H, dag_df.D,
  dag_df.Ḟ, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S)

if AD_BACKEND !== nothing
  H_E_direct_chn = sample(H_E_direct_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  H_E_direct_chn = sample(H_E_direct_model, NUTS(), MCMCThreads(), 3000, 4)
end

H_E_direct_chn_df = DataFrame(H_E_direct_chn)[!, r"α\b|β"]
precis(H_E_direct_chn_df)

p1 = plot_chains_df(H_E_direct_chn; show_traces=true)
safe_plot_save("H_E_direct_chn_traces.pdf", p1)

p2 = plot_chains_df(H_E_direct_chn; show_traces=false)
safe_plot_save("H_E_direct_chn.pdf", p2)

# Mixed model for comparison
glmm_H_E_direct = fit(MixedModel, @formula(E ~ 1 + H + D + Ḟ + M + nP + R + S + (1 | V) + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_E_direct)
glmm_H_E_coefplot = coefplot(boot)
safe_plot_save("glmm_H_E_direct_coefplot.pdf", glmm_H_E_coefplot)
glmm_H_E_ridgeplot = ridgeplot(boot)
safe_plot_save("glmm_H_E_direct_ridgeplot.pdf", glmm_H_E_ridgeplot)

## Direct effect of H on P

adjustmentSets(dag, "H", "P", effect="direct") # { D, R, S, V }

glmm_H_P = fit(MixedModel, @formula(lognP ~ 1 + H + D + R + S + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_P)
coefplot(boot)
ridgeplot(boot)

"""
    H_nP_Model(nP, H, D, R, S, IDidx, Vidx; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))

Bayesian model for the direct causal effect of habitat on parasite burden.

# Arguments
- `nP`: Log-transformed parasite count (outcome)
- `H`: Habitat (1=lab, 2=wild)
- `D`: Diet supplementation (mediator adjustment)
- `R`: Reproductive status (mediator adjustment)
- `S`: Sex (confounder adjustment)
- `IDidx`: Mouse identifier indices
- `Vidx`: Vaccination history indices
- `n_id`: Number of unique individuals
- `n_vax`: Number of unique vaccination histories

# Returns
Posterior samples for direct habitat effect on parasite burden.

# Model Structure
Adjusts for {D, R, S, V} to block confounding and mediating pathways.
Estimates direct effect H → P.
"""
@model function H_nP_Model(nP, H, D, R, S, IDidx, Vidx; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # Population-level priors
  α ~ Normal(0, 2)
  βH ~ Normal(0, 1)
  βD ~ Normal(0, 0.5)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)
  σ ~ Exponential(1.5)

  # Random intercepts
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)
  α_vax ~ filldist(Normal(0, τ), n_vax)

  # Likelihood
  nP̂ = @. α + α_ID[IDidx] + α_vax[Vidx] + βH * H + βD * D + βR * R + βS * S
  return nP ~ MvNormal(nP̂, σ^2 * I)
end

H_nP_model = H_nP_Model(log10.(1 .+ dag_df.nP), dag_df.H, dag_df.D, dag_df.R, dag_df.S, dag_df.IDidx, dag_df.Vidx)

if AD_BACKEND !== nothing
  H_nP_chn = sample(H_nP_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  H_nP_chn = sample(H_nP_model, NUTS(), MCMCThreads(), 3000, 4)
end

H_nP_chn_df = DataFrame(H_nP_chn)[!, r"α\b|β"]
precis(H_nP_chn_df)

p = plot_chains_df(H_nP_chn; show_intercept=true)
safe_plot_save("H_nP_chn.pdf", p)

## Total effect of D on E

adjustmentSets(dag, "D", "E", effect="total") # {H}

"""
    D_E_Total_Model(IDidx, Vidx, E, D, H; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))

Bayesian model for the total causal effect of diet supplementation on vaccine response.

# Arguments
- `IDidx`: Mouse identifier indices
- `Vidx`: Vaccination history indices
- `E`: Standardised vaccine response (outcome)
- `D`: Diet supplementation (1=low, 2=high)
- `H`: Habitat (confounder adjustment)
- `n_id`: Number of unique individuals
- `n_vax`: Number of unique vaccination histories

# Returns
Posterior samples for total diet effect on vaccine response.

# Model Structure
Adjusts for {H} to satisfy backdoor criterion.
Estimates total effect D → E.
"""
@model function D_E_Total_Model(IDidx, Vidx, E, D, H; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  βD ~ Normal(0, 0.75)
  βH ~ Normal(0, 1)
  σ ~ Exponential(1)
  ν ~ LogNormal(2, 1)

  # Random intercepts
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)
  α_vax ~ filldist(Normal(0, τ), n_vax)

  # Likelihood
  Ê = @. α + α_ID[IDidx] + α_vax[Vidx] + βD * D + βH * H
  return E ~ MvNormal(Ê, σ^2 * I)
end

D_E_total_model = D_E_Total_Model(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.D, dag_df.H)

if AD_BACKEND !== nothing
  D_E_total_chn = sample(D_E_total_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  D_E_total_chn = sample(D_E_total_model, NUTS(), MCMCThreads(), 3000, 4)
end

D_E_total_chn_df = DataFrame(D_E_total_chn)[!, r"β"]
precis(D_E_total_chn_df)

p = plot_chains_df(D_E_total_chn)
safe_plot_save("D_E_total_chn.pdf", p)

## Direct effect of D on E

adjustmentSets(dag, "D", "E", effect="direct") # {F, H, M, P, R, S, V}

"""
    D_E_Direct_Model(IDidx, Vidx, E, D, Ḟ, H, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))

Bayesian model for the direct causal effect of diet supplementation on vaccine response.

Blocks all indirect pathways through mediating variables.

# Arguments
- `IDidx`: Mouse identifier indices
- `Vidx`: Vaccination history indices
- `E`: Standardised vaccine response (outcome)
- `D`: Diet supplementation (1=low, 2=high)
- `Ḟ`: Standardised fat scores (mediator adjustment)
- `H`: Habitat (confounder adjustment)
- `M`: Standardised mass (mediator adjustment)
- `P`: Parasite burden (mediator adjustment)
- `R`: Reproductive status (mediator adjustment)
- `S`: Sex (confounder adjustment)
- `n_id`: Number of unique individuals
- `n_vax`: Number of unique vaccination histories

# Returns
Posterior samples for direct diet effect on vaccine response.

# Model Structure
Adjusts for {F, H, M, P, R, S, V} to block indirect effects.
Handles missing fat scores via Bayesian imputation.
"""
@model function D_E_Direct_Model(IDidx, Vidx, E, D, Ḟ, H, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  σ ~ Exponential(1)

  βD ~ Normal(0, 0.75)
  βF ~ Normal(0, 0.5)
  βH ~ Normal(0, 1)
  βM ~ Normal(0, 0.5)
  βP ~ Normal(0, 0.75)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)

  # Random intercepts
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)
  α_vax ~ filldist(Normal(0, τ), n_vax)

  # Missing fat value imputation
  N_missing = sum(ismissing.(Ḟ))
  if N_missing > 0
    F_impute ~ filldist(Normal(0, 1), N_missing)
    ν_F ~ Normal(0, 0.5)
    σ_F ~ Exponential()
  end

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      if N_missing > 0
        f_imputed = ν_F + σ_F * F_impute[i_missing]
        i_missing += 1
      else
        f_imputed = 0.0
      end
    else
      if N_missing > 0
        Ḟ[i] ~ Normal(ν_F, σ_F)
      end
      f_imputed = Ḟ[i]
    end

    # Likelihood
    Ê = α + α_ID[IDidx[i]] + α_vax[Vidx[i]] + βD * D[i] + βF * f_imputed +
        βH * H[i] + βM * M[i] + βP * P[i] + βR * R[i] + βS * S[i]
    E[i] ~ Normal(Ê, σ)
  end
end

D_E_direct_model = D_E_Direct_Model(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.D, dag_df.Ḟ,
  dag_df.H, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S)

Turing.setadbackend(:reversediff)
if AD_BACKEND !== nothing
  D_E_direct_chn = sample(D_E_direct_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  D_E_direct_chn = sample(D_E_direct_model, NUTS(), MCMCThreads(), 3000, 4)
end

D_E_direct_chn_df = DataFrame(D_E_direct_chn)[!, r"α\b|β"]
precis(D_E_direct_chn_df)

p1 = plot_chains_df(D_E_direct_chn; show_traces=true)
safe_plot_save("D_E_direct_chn_traces.pdf", p1)

p2 = plot_chains_df(D_E_direct_chn; res=(12, 10), show_traces=false)
safe_plot_save("D_E_direct_chn.pdf", p2)

## Direct effect of F on E

adjustmentSets(dag, "F", "E", effect="direct") # { D, H, M, P, R, S, V }

"""
    F_E_Direct_Model(IDidx, Vidx, E, Ḟ, D, H, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))

Bayesian model for the direct causal effect of fat scores on vaccine response.

# Arguments
- `IDidx`: Mouse identifier indices
- `Vidx`: Vaccination history indices
- `E`: Standardised vaccine response (outcome)
- `Ḟ`: Standardised fat scores (treatment)
- `D`: Diet supplementation (confounder adjustment)
- `H`: Habitat (confounder adjustment)
- `M`: Standardised mass (mediator adjustment)
- `P`: Parasite burden (confounder adjustment)
- `R`: Reproductive status (confounder adjustment)
- `S`: Sex (confounder adjustment)
- `n_id`: Number of unique individuals
- `n_vax`: Number of unique vaccination histories

# Returns
Posterior samples for direct fat effect on vaccine response.

# Model Structure
Adjusts for {D, H, M, P, R, S, V} to isolate direct Fat → E pathway.
Handles missing fat scores via Bayesian imputation.
"""
@model function F_E_Direct_Model(IDidx, Vidx, E, Ḟ, D, H, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # Population-level priors
  α ~ Normal(0, 1)
  σ ~ Exponential(1)

  βF ~ Normal(0, 0.5)
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 1)
  βM ~ Normal(0, 0.5)
  βP ~ Normal(0, 0.75)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)

  # Random intercepts
  τ ~ Exponential(1)
  α_ID ~ filldist(Normal(0, τ), n_id)
  α_vax ~ filldist(Normal(0, τ), n_vax)

  # Missing fat value imputation
  N_missing = sum(ismissing.(Ḟ))
  if N_missing > 0
    F_impute ~ filldist(Normal(0, 1), N_missing)
    ν_F ~ Normal(0, 0.5)
    σ_F ~ Exponential()
  end

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      if N_missing > 0
        f_imputed = ν_F + σ_F * F_impute[i_missing]
        i_missing += 1
      else
        f_imputed = 0.0
      end
    else
      if N_missing > 0
        Ḟ[i] ~ Normal(ν_F, σ_F)
      end
      f_imputed = Ḟ[i]
    end

    # Likelihood
    Ê = α + α_ID[IDidx[i]] + α_vax[Vidx[i]] + βF * f_imputed + βD * D[i] +
        βH * H[i] + βM * M[i] + βP * P[i] + βR * R[i] + βS * S[i]
    E[i] ~ Normal(Ê, σ)
  end
end

F_E_direct_model = F_E_Direct_Model(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.Ḟ, dag_df.D,
  dag_df.H, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S)

Turing.setadbackend(:reversediff)
if AD_BACKEND !== nothing
  F_E_direct_chn = sample(F_E_direct_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  F_E_direct_chn = sample(F_E_direct_model, NUTS(), MCMCThreads(), 3000, 4)
end

F_E_direct_chn_df = DataFrame(F_E_direct_chn)[!, r"α\b|β"]
precis(F_E_direct_chn_df)

## Direct effect of M on E

adjustmentSets(dag, "M", "E", effect="direct") # { D, F, H, P, R, S}

"""
    M_E_Model(E, M, D, Ḟ, P, R, S, H)

Bayesian model for the direct causal effect of mass on vaccine response.

# Arguments
- `E`: Standardised vaccine response (outcome)
- `M`: Standardised mass (treatment)
- `D`: Diet supplementation (confounder adjustment)
- `Ḟ`: Standardised fat scores (confounder adjustment)
- `P`: Parasite burden (confounder adjustment)
- `R`: Reproductive status (confounder adjustment)
- `S`: Sex (confounder adjustment)
- `H`: Habitat (confounder adjustment)

# Returns
Posterior samples for direct mass effect on vaccine response.

# Model Structure
Adjusts for {D, F, H, P, R, S} to isolate direct M → E pathway.
Handles missing fat scores via Bayesian imputation.
"""
@model function M_E_Model(E, M, D, Ḟ, P, R, S, H)
  # Population-level priors
  α ~ Normal(0, 1)
  σ ~ Exponential(1)

  βM ~ Normal(0, 0.5)
  βD ~ Normal(0, 0.5)
  βF ~ Normal(0, 0.5)
  βH ~ Normal(0, 1)
  βP ~ Normal(0, 0.75)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)

  # Handle missing fat score data
  missing_mask = ismissing.(Ḟ)
  N_missing::Int = sum(missing_mask)

  if N_missing > 0
    missing_indices = findall(missing_mask)
    F_impute ~ filldist(Normal(0, 1), N_missing)
    ν_F ~ Normal(0, 0.5)
    σ_F ~ Exponential()

    # Create fat values vector with imputed missing values
    f_vals = map(i -> missing_mask[i] ? (i in missing_indices ? ν_F + σ_F * F_impute[findfirst(==(i), missing_indices)] : zero(eltype(α))) : Ḟ[i], 1:length(E))

    # Likelihood for observed fat values
    for i in 1:length(E)
      if !missing_mask[i]
        Ḟ[i] ~ Normal(ν_F, σ_F)
      end
    end

    # Likelihood for vaccine response
    for i in 1:length(E)
      μ = α + βM * M[i] + βD * D[i] + βF * f_vals[i] + βH * H[i] +
          βP * P[i] + βR * R[i] + βS * S[i]
      E[i] ~ Normal(μ, σ)
    end
  else
    # Handle case with no missing fat data
    f_vals = [ismissing(f) ? zero(eltype(α)) : f for f in Ḟ]
    for i in 1:length(E)
      μ = α + βM * M[i] + βD * D[i] + βF * f_vals[i] + βH * H[i] +
          βP * P[i] + βR * R[i] + βS * S[i]
      E[i] ~ Normal(μ, σ)
    end
  end
end

M_E_model = M_E_Model(dag_df.E, dag_df.M, dag_df.D, dag_df.Ḟ, log10.(1 .+ dag_df.nP),
  dag_df.R, dag_df.S, dag_df.H)

if AD_BACKEND !== nothing
  M_E_chn = sample(M_E_model, NUTS(adtype=AD_BACKEND), MCMCThreads(), 3000, 4)
else
  M_E_chn = sample(M_E_model, NUTS(), MCMCThreads(), 3000, 4)
end

M_E_chn_df = DataFrame(M_E_chn)[!, r"α\b|β"]
precis(M_E_chn_df)

p = plot_chains_df(M_E_chn; show_intercept=true)
safe_plot_save("M_E_chn.pdf", p)

## Comprehensive mixed-effects analyses for remaining causal effects

# Use helper function for remaining analyses to maintain clean structure
analyse_causal_effect("Direct Effect: Diet → Fat", @formula(Ḟ ~ 1 + D + H + P + R + S + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Diet → Mass", @formula(M ~ 1 + D + Ḟ + H + P + R + S + V + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Diet → Reproductive Status", @formula(R ~ 1 + D + H + (1 | ID)), dag_df)

analyse_causal_effect("Direct Effect: Fat → Mass", @formula(M ~ 1 + Ḟ + D + H + P + R + S + V + (1 | ID)), dag_df)

analyse_causal_effect("Direct Effect: Parasites → Fat", @formula(Ḟ ~ 1 + P + D + H + R + S + V + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Parasites → Mass", @formula(M ~ 1 + P + D + Ḟ + H + R + S + V + (1 | ID)), dag_df)

analyse_causal_effect("Direct Effect: Reproductive Status → Fat", @formula(Ḟ ~ 1 + R + D + H + P + S + V + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Reproductive Status → Mass", @formula(M ~ 1 + R + D + Ḟ + H + P + S + V + (1 | ID)), dag_df)

analyse_causal_effect("Direct Effect: Sex → Fat", @formula(Ḟ ~ 1 + S + D + H + P + R + V + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Sex → Mass", @formula(M ~ 1 + S + D + Ḟ + H + P + R + V + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Sex → Reproductive Status", @formula(R ~ 1 + S + (1 | ID)), dag_df)

analyse_causal_effect("Direct Effect: Habitat → Diet", @formula(D ~ 1 + H + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Habitat → Fat", @formula(Ḟ ~ 1 + H + D + P + R + S + V + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Habitat → Mass", @formula(M ~ 1 + H + D + Ḟ + P + R + S + V + (1 | ID)), dag_df)
analyse_causal_effect("Direct Effect: Habitat → Reproductive Status", @formula(R ~ 1 + H + D + (1 | ID)), dag_df)
