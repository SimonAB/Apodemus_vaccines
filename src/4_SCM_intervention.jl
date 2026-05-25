#=
SCM Intervention
- Julia version: 1.12
- Author: Simon A Babayan
=#

#=
Interventional simulation (do(P=0)) and paired counterfactual prediction for parasite
elimination, using structural causal models. Terminology follows Pearl/Bareinboim:
observational branch (manuscript ``E_{factual}'', with parasites) vs interventional
posterior predictive ``E_{do(P=0)}'' under do(P=0), then unit-level counterfactual contrasts
``ΔE = E_{do(P=0)} - E_{factual}'' (stored as `delta_E` in DataFrames).

REPL: Run `# %%` cells in order (e.g. VS Code / Cursor: “Julia: Execute Cell”).
=#

# %%
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

using Printf
using PrettyTables

# Include modules
if isdir("./src/")
    cd("./src/")
end
include("TuringUtils.jl")

# Import data
include("DataWrangler.jl")

# %%
## Configuration constants


# Performance configuration following Turing.jl best practices
# For models with many parameters (>20), consider AutoZygote() or AutoReverseDiff()
# For models with few parameters (<20), AutoForwardDiff() is typically fastest
const AD_BACKEND = nothing  # Use default (AutoForwardDiff) for our model size

# NUTS: first positional argument is δ (dual-averaging target acceptance; default 0.65).
# Raising δ makes the step size more conservative and usually removes divergent transitions.
const NUTS_TARGET_ACCEPT = 0.95

# %%
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

# Select only infected mice for anthelmintic / paired-contrast (ΔE) analysis
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

# %%
## DAG specification

# Original DAG - includes P→E pathway
dag = dagitty("dag{ D -> E; D -> F; D -> M; D -> P; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> P; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; R -> P; S -> E; S -> F; S -> M; S -> P; S -> R; V -> E; V -> F; V -> M; V -> P; V -> R; }")

# Post-intervention DAG - {Pa} -> P -> E pathway removed
dag_m = dagitty("dag{ D -> E; D -> F; D -> M; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; S -> E; S -> F; S -> M; S -> R; V -> E; V -> F; V -> M; V -> R; }")

# %%
## Preliminary analysis: Effect of P on E

# Check adjustment sets
adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V}
adjustmentSets(dag_m, "P", "E", effect="total") # { }

# Mixed models to verify P→E relationship exists and is eliminated post-intervention
glmm_P_E_factual = fit(MixedModel, @formula(E ~ 1 + lognP + D + R + S + V + (1 | ID)), dag_df_infected)
glmm_P_E_do_P0 = fit(MixedModel, @formula(E ~ 1 + post_nP + (1 | ID)), dag_df_infected)

# Diagnostics for P→E models
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E_factual)

# %%
## Bayesian generative models (observational vs interventional)

"""
    EndpointE_BranchModel(IDidx, E, V, D, Ḟ, H, M, P, R, S, nP, post_P, post_nP;
                          n_id=length(unique(IDidx)), intervention=false)

Bayesian structural causal model for vaccine response: observational (factual) fit when
`intervention=false`, or interventional (post-intervention, do(P=0)) fit when `intervention=true`.

This model implements the structural causal framework by:
1. Observational (factual): models E as a function of all relevant causes including P
2. Interventional (post-intervention): models E under do(P=0) by removing the P→E pathway
3. Paired counterfactual contrasts: combine posterior draws from (1) with post-interventional
   predictions from (2) (e.g. twin-world or posterior predictive pairing in this script)

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
- Observational (factual): E ~ f(V, D, Ḟ, H, M, P, R, S, α_ID) + ε
- Post-interventional (do(P=0)): E ~ f(V, D, Ḟ, H, M, R, S, α_ID) + ε [P→E pathway severed]

# Assumptions
- Modularity: Each structural equation can be modified independently
- No unmeasured confounding given DAG structure
- Individual-level confounding handled via random effects
"""
@model function EndpointE_BranchModel(IDidx, E, V, D, Ḟ, H, M, P, R, S, nP, post_P, post_nP;
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

            # Add parasite effect only in observational (factual) scenario
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

            # Add parasite effect only in observational (factual) scenario
            if !intervention
                μ_E += βP * P[i]
            end

            E[i] ~ Normal(μ_E, σ)
        end
    end
end


"""
    Twin_World_E_Model(IDidx, E_factual, E_do_P0, V, D, Ḟ, H, M, P, R, S,
                       post_P, post_nP; n_id=length(unique(IDidx)))

Joint model for observational (factual) and post-interventional vaccine responses using the "twin world" approach.

This model simultaneously samples both the observational (factual) world (with parasites) and
the post-interventional world under do(P=0) (parasite effect on E removed), sharing individual-level random effects
to maintain identity across worlds for paired counterfactual contrasts.

# Arguments
- `E_factual`: Observed vaccine response (observational layer; manuscript ``E_{factual}'')
- `E_do_P0`: Posterior predictive E under do(P=0) (interventional layer; manuscript ``E_{do(P=0)}'')
- Other arguments as in `EndpointE_BranchModel`

# Returns
Joint posterior enabling direct estimation of individual treatment effects.
"""
@model function Twin_World_E_Model(IDidx, E_factual, E_do_P0, V, D, Ḟ, H, M, P, R, S,
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
    βP ~ Normal(0, 0.75)  # Allow moderate to large parasite effect in observational (factual) world
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

        # Likelihood for observational (factual) and post-interventional paired observations
        for i in 1:N
            μ_base = base_effects[i] + α_ID[IDidx[i]]

            # Observational (factual) world (with parasites)
            μ_E_factual = μ_base + βP * P[i]
            E_factual[i] ~ Normal(μ_E_factual, σ)

            # Post-interventional world (do(P=0) on E)
            E_do_P0[i] ~ Normal(μ_base, σ)
        end
    else
        # No missing data case
        f_vals = [ismissing(f) ? zero(eltype(α)) : f for f in Ḟ]
        base_effects = @. α + βV * V + βD * D + βḞ * f_vals + βH * H + βM * M + βR * R + βS * S

        for i in 1:N
            μ_base = base_effects[i] + α_ID[IDidx[i]]

            # Observational (factual) world (with parasites)
            μ_E_factual = μ_base + βP * P[i]
            E_factual[i] ~ Normal(μ_E_factual, σ)

            # Post-interventional world (do(P=0) on E)
            E_do_P0[i] ~ Normal(μ_base, σ)
        end
    end
end


# %%
## Prior predictive checks

# Generate prior predictive checks to assess model priors
println("=== PRIOR PREDICTIVE CHECKS FOR INTERVENTION MODELS ===")

# Observational (factual) model (with parasites)
println("Checking priors for observational (factual) model...")
factual_model = EndpointE_BranchModel(
    dag_df_infected.IDidx_infected, dag_df_infected.E, dag_df_infected.V,
    dag_df_infected.D, dag_df_infected.Ḟ, dag_df_infected.H,
    dag_df_infected.M, dag_df_infected.P, dag_df_infected.R,
    dag_df_infected.S, dag_df_infected.nP, dag_df_infected.post_P,
    dag_df_infected.post_nP; n_id=n_infected_ids, intervention=false
)

factual_prior_samples = sample(factual_model, Prior(), 1000)
factual_prior_E = vec(Array(factual_prior_samples)[:, end])  # Extract E predictions

# Post-interventional model (do(P=0) on E)
println("Checking priors for post-interventional (do(P=0)) model...")
interventional_model = EndpointE_BranchModel(
    dag_df_infected.IDidx_infected, dag_df_infected.E, dag_df_infected.V,
    dag_df_infected.D, dag_df_infected.Ḟ, dag_df_infected.H,
    dag_df_infected.M, dag_df_infected.post_P, dag_df_infected.R,
    dag_df_infected.S, dag_df_infected.post_nP, dag_df_infected.post_P,
    dag_df_infected.post_nP; n_id=n_infected_ids, intervention=true
)

interventional_prior_samples = sample(interventional_model, Prior(), 1000)
interventional_prior_E = vec(Array(interventional_prior_samples)[:, end])  # Extract E predictions

# Assess prior adequacy
assess_prior_adequacy(dag_df_infected.E, factual_prior_E; model_name="Observational (factual, with parasites)")
assess_prior_adequacy(dag_df_infected.E, interventional_prior_E; model_name="Post-interventional (do(P=0))")

# %%
## Model fitting

# Fit the pre-intervention model (reuse the factual_model from PPC)
println("\nFitting pre-intervention model...")
if AD_BACKEND !== nothing
    pre_intervention_chain = sample(
        factual_model,
        NUTS(NUTS_TARGET_ACCEPT; adtype=AD_BACKEND),
        MCMCThreads(),
        3000,
        4,
    )
else
    pre_intervention_chain = sample(
        factual_model,
        NUTS(NUTS_TARGET_ACCEPT),
        MCMCThreads(),
        3000,
        4,
    )
end

# Fit the post-intervention model (reuse the interventional_model from PPC)
println("Fitting post-intervention model...")
if AD_BACKEND !== nothing
    post_intervention_chain = sample(
        interventional_model,
        NUTS(NUTS_TARGET_ACCEPT; adtype=AD_BACKEND),
        MCMCThreads(),
        3000,
        4,
    )
else
    post_intervention_chain = sample(
        interventional_model,
        NUTS(NUTS_TARGET_ACCEPT),
        MCMCThreads(),
        3000,
        4,
    )
end

"""
    _format_mcmc_numeric_for_table(x::Real) -> String

Format a scalar for supplementary MCMC tables: at most three digits after the decimal
point, with trailing zeros (and a trailing decimal point) removed when not needed.
"""
function _format_mcmc_numeric_for_table(x::Real)
    xf = Float64(x)
    !isfinite(xf) && return string(xf)
    y = round(xf; digits=3)
    s = @sprintf("%.3f", y)
    s = rstrip(rstrip(s, '0'), '.')
    return isempty(s) ? "0" : s
end

"""
    write_mcmc_diagnostics_table(chains_by_model; out_dir, filename_stem, parameter_regex)

Write MCMC diagnostics (Rhat, bulk/tail ESS, MCSE) for one or more fitted chains.
Numeric columns are written with at most three decimal places; trailing zeros are dropped
where they do not add precision. The returned `DataFrame` retains rounded `Float64`
values for programmatic use.

# Arguments
- `chains_by_model`: Dict mapping model name => MCMCChains.Chains

# Keywords
- `out_dir`: output directory (created if missing)
- `filename_stem`: base filename for outputs (writes `.csv` and `.tex`)
- `parameter_regex`: only keep parameters matching this regex (default keeps β and σ)
"""
function write_mcmc_diagnostics_table(
    chains_by_model::AbstractDict;
    out_dir::AbstractString,
    filename_stem::AbstractString="SuppTable_MCMC_diagnostics_SCM",
    parameter_regex::Regex=r"^β|^σ",
)
    mkpath(out_dir)

    diagnostics = DataFrame()
    for (model_name, chn) in pairs(chains_by_model)
        s = summarize(chn)
        df = DataFrame(s)
        df.model .= model_name
        df = df[occursin.(parameter_regex, String.(df.parameters)), :]
        append!(diagnostics, df; promote=true)
    end

    select!(diagnostics,
        :model,
        :parameters,
        :mean,
        :std,
        :mcse,
        :ess_bulk,
        :ess_tail,
        :rhat,
        :ess_per_sec,
    )

    numeric_cols = (:mean, :std, :mcse, :ess_bulk, :ess_tail, :rhat, :ess_per_sec)
    for c in numeric_cols
        diagnostics[!, c] = round.(Float64.(diagnostics[!, c]); digits=3)
    end

    export_df = copy(diagnostics)
    for c in numeric_cols
        export_df[!, c] = _format_mcmc_numeric_for_table.(diagnostics[!, c])
    end

    CSV.write(joinpath(out_dir, filename_stem * ".csv"), export_df)

    open(joinpath(out_dir, filename_stem * ".tex"), "w") do io
        # One header row only (default PrettyTables adds a second row of column eltypes).
        pretty_table(
            io,
            export_df;
            backend=:latex,
            column_labels=[string.(names(export_df))],
        )
    end

    return diagnostics
end

# Save a supplementary table of MCMC diagnostics (SCM intervention models).
# Short model labels in the table body; the caption gives full observational vs do(P=0) wording.
diagnostics_out_dir = normpath(joinpath(@__DIR__, "..", "results", "tables"))
write_mcmc_diagnostics_table(
    Dict(
        "Observational" => pre_intervention_chain,
        "Interventional do(P=0)" => post_intervention_chain,
    );
    out_dir=diagnostics_out_dir,
    filename_stem="SuppTable_MCMC_diagnostics_SCM_intervention",
)

# %%
## Generate paired E predictions and counterfactual contrasts ΔE (posterior predictive)


"""
    sample_paired_E_predictions(pre_chain, df_infected; n_samples=1000)

Generate paired draws of E under the observational (factual) generative model and under
post-interventional do(P=0), for per-mouse counterfactual contrasts (ΔE).

Uses posterior predictive sampling from the observational (factual) fit, then applies the
do(P=0) structural substitution for the paired post-interventional predictions.

# Arguments
- `pre_chain`: MCMC chain from the fitted observational (factual) model
- `df_infected`: DataFrame with infected mouse data
- `n_samples`: Number of posterior samples to draw

# Returns
- `E_factual`: Matrix of observational (factual) vaccine responses (n_obs × n_samples)
- `E_do_P0`: Matrix of interventional (post-interventional) paired responses (n_obs × n_samples)
"""
function sample_paired_E_predictions(pre_chain, df_infected; n_samples::Int=1000)
    n_obs = nrow(df_infected)
    chain_samples = DataFrame(pre_chain)
    n_chain_samples = nrow(chain_samples)

    # Pre-allocate with specific types for better performance
    E_factual = Matrix{Float64}(undef, n_obs, n_samples)
    E_do_P0 = Matrix{Float64}(undef, n_obs, n_samples)

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

            # Observational (factual) prediction (with parasites)
            μ_factual = base_μ + βP * P_vec[i]
            E_factual[i, s] = μ_factual + σ * randn()

            # Post-interventional paired draw (do(P=0) on E)
            E_do_P0[i, s] = base_μ + σ * randn()
        end
    end

    return E_factual, E_do_P0
end

# Generate paired post-interventional predictions (posterior predictive)
println("Generating paired observational vs post-interventional draws...")
E_factual, E_do_P0 = sample_paired_E_predictions(pre_intervention_chain, dag_df_infected)

# Add predictions to dataframe - using this as our main approach
dag_df_infected.E_factual_mean = vec(mean(E_factual, dims=2))
dag_df_infected.E_do_P0_mean = vec(mean(E_do_P0, dims=2))
dag_df_infected.delta_E = dag_df_infected.E_do_P0_mean .- dag_df_infected.E_factual_mean

# Calculate uncertainty measures
dag_df_infected.E_factual_sd = vec(std(E_factual, dims=2))
dag_df_infected.E_do_P0_sd = vec(std(E_do_P0, dims=2))

# Calculate Cohen's d effect size for standardised assessment
dag_df_infected.E_cohens_d = begin
    # Pooled standard deviation for Cohen's d calculation
    pooled_sd = sqrt.((dag_df_infected.E_factual_sd .^ 2 .+ dag_df_infected.E_do_P0_sd .^ 2) ./ 2)
    # Avoid division by very small numbers
    safe_pooled_sd = max.(pooled_sd, 0.1)  # Minimum threshold for stability
    dag_df_infected.delta_E ./ safe_pooled_sd
end

println("Paired-contrast summary (ΔE = E_do(P=0) − E_factual):")
println("Mean paired contrast (E_post − E_obs): ", round(mean(dag_df_infected.delta_E), digits=3))
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

println("Summary of paired contrasts:")
println("Mean paired contrast (E_post − E_obs): ",
    round(mean(dag_df_infected.delta_E), digits=3))
println("SD of paired contrasts: ",
    round(std(dag_df_infected.delta_E), digits=3))
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


# %%
## Main effects interpretation for Sex and Reproductive Status

"""
    interpret_main_effects(chain, df_infected)

Interpret the main effects of Sex (S) and Reproductive Status (R) from the Bayesian model.

This function extracts posterior samples for βS and βR, calculates summary statistics,
provides interpretations based on effect sizes.

# Arguments
- `chain`: MCMC chain from the fitted Bayesian model
- `df_infected`: DataFrame with infected mouse data for group-specific analysis

# Returns
Named tuple with effect estimates and interpretations for both S and R.
"""
function interpret_main_effects(chain, df_infected)
println("=== MAIN EFFECTS INTERPRETATION: SEX AND REPRODUCTIVE STATUS ===")

    # Prepare to store Cohen's d estimates
    factors = [:D, :R, :S, :V, :M, :Ḟ]
    mean_d_values = []
    sem_d_values = []
    labels = []

    # Compute Cohen's d estimates for each factor
    for factor in factors
        mask = dag_df_infected[:, factor] .== 1
        E_cohens_d_masked = dag_df_infected[mask, :E_cohens_d]

        # Compute mean and SEM
        mean_d = mean(E_cohens_d_masked)
        sem_d = std(E_cohens_d_masked) / sqrt(length(E_cohens_d_masked))

        # Store results
        push!(mean_d_values, mean_d)
        push!(sem_d_values, sem_d)
        push!(labels, string(factor))
    end

    println("Computed Cohen's d estimates for factors: ", labels)

    # Extract parameter samples
    chain_df = DataFrame(chain)
    βS_samples = chain_df.βS
    βR_samples = chain_df.βR

    # Sex effect analysis
    println("\n--- SEX EFFECT (βS) ---")
    println("Coding: 1=Male, 2=Female")

    βS_mean = mean(βS_samples)
    βS_sd = std(βS_samples)
    βS_ci_lower, βS_ci_upper = quantile(βS_samples, [0.025, 0.975])

    println("Sex effect (βS):")
    println("Mean: $(round(βS_mean, digits=4))")
    println("SD: $(round(βS_sd, digits=4))")
    println("95% CI: [$(round(βS_ci_lower, digits=4)), $(round(βS_ci_upper, digits=4))]")

    # Probability assessments for sex
    prob_female_advantage = mean(βS_samples .> 0)
    prob_male_advantage = mean(βS_samples .< 0)
    prob_substantial_sex = mean(abs.(βS_samples) .> 0.1)
    prob_large_sex = mean(abs.(βS_samples) .> 0.2)

    println("\nProbability assessments (Sex):")
    println("P(βS > 0) [Female advantage]: $(round(prob_female_advantage, digits=3))")
    println("P(βS < 0) [Male advantage]: $(round(prob_male_advantage, digits=3))")
    println("P(|βS| > 0.1) [Substantial effect]: $(round(prob_substantial_sex, digits=3))")
    println("P(|βS| > 0.2) [Large effect]: $(round(prob_large_sex, digits=3))")

    # Effect size interpretation for sex
    if abs(βS_mean) < 0.1
        sex_interpretation = "negligible"
    elseif abs(βS_mean) < 0.2
        sex_interpretation = "small"
    elseif abs(βS_mean) < 0.3
        sex_interpretation = "moderate"
    else
        sex_interpretation = "large"
    end

    sex_direction = βS_mean > 0 ? "Female advantage" : "Male advantage"
    println("\nSex effect interpretation: $(sex_interpretation) effect")
    println("Direction: $(sex_direction)")

    # Reproductive Status effect analysis
    println("\n--- REPRODUCTIVE STATUS EFFECT (βR) ---")
    println("Coding: 1=Non-reproductive, 2=Reproductive")

    βR_mean = mean(βR_samples)
    βR_sd = std(βR_samples)
    βR_ci_lower, βR_ci_upper = quantile(βR_samples, [0.025, 0.975])

    println("Reproductive Status effect (βR):")
    println("Mean: $(round(βR_mean, digits=4))")
    println("SD: $(round(βR_sd, digits=4))")
    println("95% CI: [$(round(βR_ci_lower, digits=4)), $(round(βR_ci_upper, digits=4))]")

    # Probability assessments for reproductive status
    prob_repro_advantage = mean(βR_samples .> 0)
    prob_nonrepro_advantage = mean(βR_samples .< 0)
    prob_substantial_repro = mean(abs.(βR_samples) .> 0.1)
    prob_large_repro = mean(abs.(βR_samples) .> 0.2)

    println("\nProbability assessments (Reproductive Status):")
    println("P(βR > 0) [Reproductive advantage]: $(round(prob_repro_advantage, digits=3))")
    println("P(βR < 0) [Non-reproductive advantage]: $(round(prob_nonrepro_advantage, digits=3))")
    println("P(|βR| > 0.1) [Substantial effect]: $(round(prob_substantial_repro, digits=3))")
    println("P(|βR| > 0.2) [Large effect]: $(round(prob_large_repro, digits=3))")

    # Effect size interpretation for reproductive status
    if abs(βR_mean) < 0.1
        repro_interpretation = "negligible"
    elseif abs(βR_mean) < 0.2
        repro_interpretation = "small"
    elseif abs(βR_mean) < 0.3
        repro_interpretation = "moderate"
    else
        repro_interpretation = "large"
    end

    repro_direction = βR_mean > 0 ? "Reproductive advantage" : "Non-reproductive advantage"
    println("\nReproductive Status effect interpretation: $(repro_interpretation) effect")
    println("Direction: $(repro_direction)")

    # Group-specific analysis (based on observed data)
    println("\n--- GROUP-SPECIFIC ΔE (COUNTERFACTUAL CONTRASTS) ---")

    # Create group summaries
    group_summary = combine(
        groupby(df_infected, [:S, :R]),
        :delta_E => mean => :mean_effect,
        :delta_E => std => :sd_effect,
        :E_cohens_d => mean => :mean_cohens_d,
        nrow => :count
    )

    for row in eachrow(group_summary)
        sex_label = row.S == 1 ? "Male" : "Female"
        repro_label = row.R == 1 ? "Non-reproductive" : "Reproductive"

        println("$sex_label, $repro_label (n=$(row.count)):")
        println("  Mean paired contrast: $(round(row.mean_effect, digits=3)) ± $(round(row.sd_effect, digits=3))")
        println("  Mean Cohen's d: $(round(row.mean_cohens_d, digits=3))")
    end


    return (
        sex_effect=(
            mean=βS_mean,
            sd=βS_sd,
            ci_lower=βS_ci_lower,
            ci_upper=βS_ci_upper,
            prob_female_advantage=prob_female_advantage,
            prob_substantial=prob_substantial_sex,
            interpretation=sex_interpretation,
            direction=sex_direction
        ),
        repro_effect=(
            mean=βR_mean,
            sd=βR_sd,
            ci_lower=βR_ci_lower,
            ci_upper=βR_ci_upper,
            prob_repro_advantage=prob_repro_advantage,
            prob_substantial=prob_substantial_repro,
            interpretation=repro_interpretation,
            direction=repro_direction
        ),
        group_summary=group_summary
    )
end


# %%
## Compute Cohen's d estimates for each main effect
include("compute_cohens_d_main_effects.jl")
cohens_d_main_effects = compute_cohens_d_main_effects(dag_df_infected)

# %%
## Interpret main effects of Sex and Reproductive Status
main_effects_results = interpret_main_effects(pre_intervention_chain, dag_df_infected)

# %%
## Analysis helper functions

"""
    paired_contrast_percentage_change(df_infected; method="relative_to_baseline")

Calculate percentage change in vaccine response under do(P=0) (paired observational vs post-interventional predictions).

# Arguments
- `df_infected`: DataFrame with `E_factual_mean` and `E_do_P0_mean` columns
- `method`: Calculation method. Options:
  - `"relative_to_baseline"`: Change relative to individual baseline values
  - `"relative_to_population"`: Change relative to population mean
  - `"relative_to_original_scale"`: Change in back-transformed OD scale
  - `"cohen_d_to_percent"`: Convert Cohen's d to percentage using pooled SD

# Returns
Vector of percentage changes for each observation (also prints summary statistics).

Different methods handle the standardised nature of the vaccine response differently.
"""
function paired_contrast_percentage_change(df_infected; method::String="relative_to_baseline")

    if method == "relative_to_baseline"
        # Method 1: Direct percentage change in standardised units
        # Use absolute values to avoid issues with negative standardised scores
        baseline_E = abs.(df_infected.E_factual_mean)
        change_E = df_infected.E_do_P0_mean .- df_infected.E_factual_mean

        # Avoid division by very small numbers
        safe_baseline = max.(baseline_E, 0.1)
        percent_change = (change_E ./ safe_baseline) .* 100

        println("Method 1 - Relative to baseline (standardised units):")

    elseif method == "relative_to_population"
        # Method 2: Change relative to population mean
        pop_mean_E = mean(df_infected.E_factual_mean)
        change_E = df_infected.E_do_P0_mean .- df_infected.E_factual_mean
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
        logOD_do_P0 = df_infected.E_do_P0_mean .* original_logOD_std .+ original_logOD_mean

        # Convert to OD scale
        factual_OD = 10 .^ factual_logOD .- 1
        OD_do_P0 = 10 .^ logOD_do_P0 .- 1

        # Calculate percentage change in original OD units
        safe_factual_OD = max.(factual_OD, 0.01)  # Avoid division by zero
        percent_change = ((OD_do_P0 .- factual_OD) ./ safe_factual_OD) .* 100

        println("Method 3 - Back-transformed to original OD scale:")

    elseif method == "cohen_d_to_percent"
        # Method 4: Convert Cohen's d to percentage using effect size interpretation
        # Cohen's d represents standardised effect size
        # Convert to percentage improvement using pooled standard deviation

        pooled_sd = sqrt.((df_infected.E_factual_sd .^ 2 .+ df_infected.E_do_P0_sd .^ 2) ./ 2)
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

Compares vaccination coefficients between observational (factual) and post-interventional scenarios
to determine how much better vaccines work when parasites are eliminated.

# Arguments
- `df_infected`: DataFrame with `E_factual_mean` and `E_do_P0_mean` columns

# Returns
Named tuple with vaccination effect coefficients and improvement statistics.
"""
function calculate_vaccination_effect_percentage(df_infected)
    println("=== VACCINATION EFFECT PERCENTAGE IMPROVEMENT ===")

    # Fit models to get vaccination coefficients
    factual_model = fit(MixedModel, @formula(E_factual_mean ~ V + (1 | ID)), df_infected)
    interventional_model = fit(MixedModel, @formula(E_do_P0_mean ~ V + (1 | ID)), df_infected)

    # Extract vaccination coefficients
    β_factual = coef(factual_model)[2]  # V coefficient
    βV_do_P0 = coef(interventional_model)[2]  # V coefficient

    # Calculate percentage improvement in vaccination effect
    vaccination_improvement = ((βV_do_P0 - β_factual) / abs(β_factual)) * 100

    println("Observational (factual) vaccination effect (β_V): $(round(β_factual, digits=3))")
    println("Post-interventional vaccination effect (β_V): $(round(βV_do_P0, digits=3))")
    println("Vaccination effect improvement: $(round(vaccination_improvement, digits=1))%")

    # Also calculate absolute improvement
    absolute_improvement = βV_do_P0 - β_factual
    println("Absolute improvement: $(round(absolute_improvement, digits=3)) standardised units")

    return (
        factual_effect=β_factual,
        interventional_effect=βV_do_P0,
        percentage_improvement=vaccination_improvement,
        absolute_improvement=absolute_improvement
    )
end


# %%
## Causal effect analysis using generative model predictions

# Check adjustment sets for vaccination effect
adjustmentSets(dag, "V", "E", effect="total") # { }
adjustmentSets(dag_m, "V", "E", effect="total") # { }

# Total effect of V on E (observational vs post-interventional predictions)
glmm_V_E_total_factual = fit(MixedModel, @formula(E_factual_mean ~ V + (1 | ID)), dag_df_infected)
glmm_V_E_total_do_P0 = fit(MixedModel, @formula(E_do_P0_mean ~ V + (1 | ID)), dag_df_infected)

# Diagnostics for vaccination effect model
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_total_do_P0)

# Check adjustment sets for direct effects
adjustmentSets(dag, "V", "E", effect="direct") # { D, F, H, M, P, R, S }
adjustmentSets(dag_m, "V", "E", effect="direct") # { D, F, H, M, P, R, S }

# Direct effect of V on E (observational vs post-interventional predictions)
glmm_V_E_direct_factual = fit(MixedModel, @formula(E_factual_mean ~ V + D + Ḟ + H + M + P + R + S + (1 | ID)), dag_df_infected)
glmm_V_E_direct_do_P0 = fit(MixedModel, @formula(E_do_P0_mean ~ V + D + Ḟ + H + M + R + S + (1 | ID)), dag_df_infected)

# Diagnostics for direct vaccination effect model
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_direct_do_P0)

# %%
# %%
## Interaction analysis with Cohen's d

# Fit model using Cohen's d as outcome
glmm_cohens_d_no_interaction = fit(MixedModel, @formula(E_cohens_d ~ -1 + D + R + S + V + M + Ḟ + (1 | ID)), dag_df_infected)
glmm_cohens_d_interaction = fit(MixedModel, @formula(E_cohens_d ~ -1 + D + R + S + V + M + Ḟ + S & R + (1 | ID)), dag_df_infected)

# Diagnostics for Cohen's d interaction model
boot_cohens_d = parametricbootstrap(MersenneTwister(1234), 10_000, glmm_cohens_d_interaction)

# %%
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
percent_changes_baseline = paired_contrast_percentage_change(dag_df_infected; method="relative_to_baseline")
dag_df_infected.percent_change_baseline = percent_changes_baseline

println("\n" * "-"^30)

# Method 2b: Relative to population mean
percent_changes_population = paired_contrast_percentage_change(dag_df_infected; method="relative_to_population")
dag_df_infected.percent_change_population = percent_changes_population

println("\n" * "-"^30)

# Method 2c: Cohen's d to percentage
percent_changes_cohens = paired_contrast_percentage_change(dag_df_infected; method="cohen_d_to_percent")
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
        percent_changes_original = paired_contrast_percentage_change(dag_df_infected; method="relative_to_original_scale")
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
    Mean_SD=["$(round(means[i], digits=1)) ± $(round(stds[i], digits=1))" for i in eachindex(means)],
    Median=round.(median.(individual_data), digits=1),
    Range=["$(round(minimum(data), digits=1)) to $(round(maximum(data), digits=1))" for data in individual_data]
);

println(summary_table)
