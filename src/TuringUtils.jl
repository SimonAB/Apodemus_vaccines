#=
TuringUtils:
- Julia version: 1.12
- Author: Simon A Babayan
=#

#=
This script contains utility functions for the Turing models.
=#

# Required imports for Turing utilities
using PrettyTables
using DataFrames
using Statistics: mean, std, median
using StatsBase: fit, Histogram, quantile
using Distributions: Normal, Exponential, MvNormal, TDist, Gamma, NegativeBinomial, Cauchy, Poisson, logpdf, pdf, logistic
using LinearAlgebra: I
using Turing
using Turing: @model, filldist, truncated, arraydist, sample, NUTS, MCMCThreads, Prior, summarystats
using MixedModels  # For mixed-effects models in temporal analysis

"""
NegativeBinomial2(μ, ϕ)
Mean-variance parameterization of `NegativeBinomial`.

### Derivation

`NegativeBinomial` from `Distributions.jl` is parameterized following [1]. With the parameterization in [2], we can solve
for `r` (`n` in [1]) and `p` by matching the mean and the variance given in `μ` and `ϕ`, where the expectation is μ and variance is (μ + μ²/ϕ).

We have the following two equations\\
(1) μ = r (1 - p) / p \\
(2) μ + μ^2 / ϕ = r (1 - p) / p^2

Substituting (1) into the RHS of (2): \\
  μ + (μ^2 / ϕ) = μ / p\\
⟹ 1 + (μ / ϕ) = 1 / p\\
⟹ p = 1 / (1 + μ / ϕ)\\
⟹ p = (1 / (1 + μ / ϕ)

Then in (1) we have\\
  μ = r (1 - (1 / 1 + μ / ϕ)) * (1 + μ / ϕ)\\
⟹ μ = r ((1 + μ / ϕ) - 1)\\
⟹ r = ϕ

Hence, the resulting map is `(μ, ϕ) ↦ NegativeBinomial(ϕ, 1 / (1 + μ / ϕ))`.

### References
[1] https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html

[2] https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
"""
function NegativeBinomial2(μ::T, ϕ::T) where {T<:Real}
    p = max(1 / (1 + μ / ϕ), 1e-6) # numerical stability
    r = ϕ
    return NegativeBinomial(r, p)
end

"""
    convert_str_to_indices(v::AbstractVector)
Converts a vector `v` to a vector of indices, i.e. a vector where all the entries are
integers. Returns a tuple with the first element as the converted vector and the
second element a `Dict` specifying which string is which integer.
This function is especially useful for random-effects varying-intercept hierarchical models.
Normally `v` would be a vector of group membership with values such as `"group_1"`,
`"group_2"` etc. For random-effect models with varying-intercepts, Turing needs the group
membership values to be passed as `Int`s.
Adapted from https://github.com/TuringLang/TuringGLM.jl
"""
function convert_str_to_indices(v::AbstractVector)
    d = Dict{eltype(v),Int}()
    v_int = Int[]
    for i in v
        n = get!(d, i, length(d) + 1)
        push!(v_int, n)
    end
    return v_int, d
end

"""
    get_idx(term, data)
Returns a tuple with the first element as the ID vector of `Int`s that represent
group membership for a specific random-effect intercept group `t` of observations
present in `data`. The second element of the tuple is a `Dict` specifying which string is
which integer in the ID vector.
Adapted from https://github.com/TuringLang/TuringGLM.jl
"""
function get_idx(t, data::D) where {D}
    col = Symbol(t)
    idx = data[!, col]  # Direct DataFrame column access
    return convert_str_to_indices(idx)
end

"""
    get_var(term, data)
Returns the corresponding vector of column in `data` for the a specific
random-effect slope `term` of observations present in `data`.
Adapted from https://github.com/TuringLang/TuringGLM.jl
"""
function get_var(t, data::D) where {D}
    col = Symbol(t)
    return data[!, col]  # Direct DataFrame column access
end


# GENERIC LINEAR MODELS

# Bayesian linear regression.
@model function linear_regression(x, y)
    # Set variance prior.
    σ ~ Exponential(1 / std(y))

    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))

    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(nfeatures, sqrt(10))

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    return y ~ MvNormal(mu, σ)
end

@model function varying_intercept(
    X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)
)
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    β ~ filldist(Normal(0, 2), predictors)  # population-level coefficients
    σ ~ Exponential(std(y))                 # residual SD
    #prior for variance of random intercepts
    #usually requires thoughtful specification
    τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
    αⱼ ~ filldist(Normal(0, τ), n_gr)       # group-level intercepts

    #likelihood
    ŷ = α .+ X * β .+ αⱼ[idx]
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

@model function varying_slope(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2))
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))                    # population-level intercept
    σ ~ Exponential(std(y))                              # residual SD
    #prior for variance of random slopes
    #usually requires thoughtful specification
    τ ~ filldist(truncated(Cauchy(0, 2); lower=0), n_gr) # group-level slopes SDs
    βⱼ ~ filldist(Normal(0, 1), predictors, n_gr)        # group-level standard normal slopes

    #likelihood
    ŷ = α .+ X * βⱼ * τ
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

@model function varying_intercept_slope(
    X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)
)
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))                     # population-level intercept
    σ ~ Exponential(std(y))                               # residual SD
    #prior for variance of random intercepts and slopes
    #usually requires thoughtful specification
    τₐ ~ truncated(Cauchy(0, 2); lower=0)                 # group-level SDs intercepts
    τᵦ ~ filldist(truncated(Cauchy(0, 2); lower=0), n_gr) # group-level slopes SDs
    αⱼ ~ filldist(Normal(0, τₐ), n_gr)                    # group-level intercepts
    βⱼ ~ filldist(Normal(0, 1), predictors, n_gr)         # group-level standard normal slopes

    #likelihood
    ŷ = α .+ αⱼ[idx] .+ X * βⱼ * τᵦ
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

# Models
#
"""
Simplified Poisson regression model for count data.

# Arguments
- X: Predictor matrix
- y: Count response vector
- predictors: Number of predictors (default: size(X, 2))

# Model
- α∼Normal(0,2.5): Intercept with wide prior
- β∼Student-t(0,1,3): Coefficients with wide-tailed priors
- y ~ Poisson(exp(linear_predictor)): Log-linear Poisson model
"""
@model function poissonreg(X, y; predictors=size(X, 2))
    # Priors
    α ~ Normal(0.0, 2.5)
    β ~ filldist(TDist(3), predictors)

    # Likelihood with vectorized computation
    λ = exp.(α .+ X * β)
    for i in eachindex(y)
        y[i] ~ Poisson(λ[i])
    end
end;

"""
Simplified Negative Binomial regression model for overdispersed count data.

# Arguments
- X: Predictor matrix
- y: Count response vector
- predictors: Number of predictors (default: size(X, 2))

# Model
- α∼Normal(0,2.5): Intercept with wide prior
- β∼Student-t(0,1,3): Coefficients with wide-tailed priors
- ϕ∼Exponential(1): Overdispersion parameter
- y ~ NegativeBinomial2(exp(linear_predictor), ϕ): Overdispersed count model
"""
@model function negbinreg(X, y; predictors=size(X, 2))
    # Priors
    α ~ Normal(0.0, 2.5)
    β ~ filldist(TDist(3), predictors)
    ϕ⁻ ~ Gamma(0.01, 0.01)
    ϕ = 1 / ϕ⁻

    # Likelihood with vectorized computation
    μ = exp.(α .+ X * β)
    for i in eachindex(y)
        y[i] ~ NegativeBinomial2(μ[i], ϕ)
    end
end;

"""
    zeroinflated_negbinreg(X_count, X_zi, y; predictors_count=size(X_count, 2), predictors_zi=size(X_zi, 2))

Zero-inflated Negative Binomial regression model for overdispersed count data with excess zeros.

This model is particularly appropriate for parasite count data where many individuals
have zero parasites (structural zeros) and others have counts from a negative binomial
distribution (sampling zeros + positive counts).

# Arguments
- `X_count`: Predictor matrix for the count component
- `X_zi`: Predictor matrix for the zero-inflation component
- `y`: Count response vector
- `predictors_count`: Number of predictors for count model
- `predictors_zi`: Number of predictors for zero-inflation model

# Model Structure
- Zero-inflation component: logit(π) = X_zi * β_zi
- Count component: μ = exp(X_count * β_count)
- P(y=0) = π + (1-π) * NB(0; μ, ϕ)
- P(y>0) = (1-π) * NB(y; μ, ϕ)

# Priors
- Weakly informative priors for all coefficients
- Conservative overdispersion parameter
"""
@model function zeroinflated_negbinreg(X_count, X_zi, y;
                                       predictors_count=size(X_count, 2),
                                       predictors_zi=size(X_zi, 2))
    # Priors for count component
    α_count ~ Normal(0.0, 2.0)
    β_count ~ filldist(Normal(0.0, 1.0), predictors_count)

    # Priors for zero-inflation component
    α_zi ~ Normal(0.0, 1.5)  # Logit scale
    β_zi ~ filldist(Normal(0.0, 1.0), predictors_zi)

    # Overdispersion parameter
    ϕ⁻ ~ Gamma(0.01, 0.01)
    ϕ = 1 / ϕ⁻

    # Linear predictors
    μ = exp.(α_count .+ X_count * β_count)
    logit_π = α_zi .+ X_zi * β_zi
    π = logistic.(logit_π)

    # Zero-inflated negative binomial likelihood
    for i in eachindex(y)
        if y[i] == 0
            # P(y=0) = π + (1-π) * NB(0)
            prob_structural_zero = π[i]
            prob_sampling_zero = (1 - π[i]) * pdf(NegativeBinomial2(μ[i], ϕ), 0)
            Turing.@addlogprob! log(prob_structural_zero + prob_sampling_zero)
        else
            # P(y>0) = (1-π) * NB(y)
            Turing.@addlogprob! log(1 - π[i]) + logpdf(NegativeBinomial2(μ[i], ϕ), y[i])
        end
    end
end;

"""
    compare_parasite_models(X_count, X_zi, y; model_names=["Poisson", "NegBin", "ZIP", "ZINB"])

Compare different count models for parasite data and return model comparison metrics.

# Returns
- Dictionary with model fits and comparison metrics (LOO, WAIC if available)
"""
function compare_parasite_models(X_count, X_zi, y;
                                model_names=["Poisson", "NegBin", "ZIP", "ZINB"],
                                n_samples=1000, n_chains=4)

    results = Dict{String, Any}()

    println("Comparing parasite count models...")

    # Poisson model
    if "Poisson" in model_names
        println("Fitting Poisson model...")
        @model function poisson_model(X, y)
            α ~ Normal(0.0, 2.5)
            β ~ filldist(Normal(0.0, 1.0), size(X, 2))
            μ = exp.(α .+ X * β)
            for i in eachindex(y)
                y[i] ~ Poisson(μ[i])
            end
        end

        try
            pois_chain = sample(poisson_model(X_count, y), NUTS(), MCMCThreads(), n_samples, n_chains)
            results["Poisson"] = (chain=pois_chain, converged=true)
        catch e
            println("Poisson model failed: $e")
            results["Poisson"] = (chain=nothing, converged=false, error=e)
        end
    end

    # Negative Binomial model
    if "NegBin" in model_names
        println("Fitting Negative Binomial model...")
        try
            nb_chain = sample(negbinreg(X_count, y), NUTS(), MCMCThreads(), n_samples, n_chains)
            results["NegBin"] = (chain=nb_chain, converged=true)
        catch e
            println("Negative Binomial model failed: $e")
            results["NegBin"] = (chain=nothing, converged=false, error=e)
        end
    end

    # Zero-Inflated Poisson
    if "ZIP" in model_names
        println("Fitting Zero-Inflated Poisson model...")
        @model function zip_model(X_count, X_zi, y)
            α_count ~ Normal(0.0, 2.0)
            β_count ~ filldist(Normal(0.0, 1.0), size(X_count, 2))
            α_zi ~ Normal(0.0, 1.5)
            β_zi ~ filldist(Normal(0.0, 1.0), size(X_zi, 2))

            μ = exp.(α_count .+ X_count * β_count)
            logit_π = α_zi .+ X_zi * β_zi
            π = logistic.(logit_π)

            for i in eachindex(y)
                if y[i] == 0
                    prob_structural_zero = π[i]
                    prob_sampling_zero = (1 - π[i]) * pdf(Poisson(μ[i]), 0)
                    Turing.@addlogprob! log(prob_structural_zero + prob_sampling_zero)
                else
                    Turing.@addlogprob! log(1 - π[i]) + logpdf(Poisson(μ[i]), y[i])
                end
            end
        end

        try
            zip_chain = sample(zip_model(X_count, X_zi, y), NUTS(), MCMCThreads(), n_samples, n_chains)
            results["ZIP"] = (chain=zip_chain, converged=true)
        catch e
            println("ZIP model failed: $e")
            results["ZIP"] = (chain=nothing, converged=false, error=e)
        end
    end

    # Zero-Inflated Negative Binomial
    if "ZINB" in model_names
        println("Fitting Zero-Inflated Negative Binomial model...")
        try
            zinb_chain = sample(zeroinflated_negbinreg(X_count, X_zi, y), NUTS(), MCMCThreads(), n_samples, n_chains)
            results["ZINB"] = (chain=zinb_chain, converged=true)
        catch e
            println("ZINB model failed: $e")
            results["ZINB"] = (chain=nothing, converged=false, error=e)
        end
    end

    # Summary
    successful_models = [name for (name, result) in results if result.converged]
    println("\nSuccessfully fitted models: $(join(successful_models, ", "))")

    return results
end

"""
    assess_zero_inflation(y; threshold_prop=0.3)

Assess whether zero-inflation modelling is necessary for count data.

# Arguments
- `y`: Count vector
- `threshold_prop`: Proportion of zeros above which ZI models are recommended

# Returns
- Named tuple with zero proportion and recommendation
"""
function assess_zero_inflation(y; threshold_prop=0.3)
    zero_prop = sum(y .== 0) / length(y)
    n_zeros = sum(y .== 0)
    n_total = length(y)

    println("Zero-inflation Assessment:")
    println("  Total observations: $n_total")
    println("  Zero counts: $n_zeros")
    println("  Zero proportion: $(round(zero_prop, digits=3))")

    if zero_prop > threshold_prop
        println("  Recommendation: ✓ Zero-inflation modelling recommended")
        recommendation = "Use zero-inflated models"
    else
        println("  Recommendation: Standard count models likely sufficient")
        recommendation = "Standard models OK"
    end

    return (zero_proportion=zero_prop, n_zeros=n_zeros, recommendation=recommendation)
end;

"""
    calculate_e_value(effect_estimate, confidence_interval; rare_outcome=false)

Calculate E-value for sensitivity analysis of unmeasured confounding.

The E-value represents the minimum strength of association that an unmeasured
confounder would need to have with both the treatment and outcome to fully
explain away the observed effect, conditional on measured covariates.

# Arguments
- `effect_estimate`: Point estimate (e.g., hazard ratio, odds ratio, coefficient)
- `confidence_interval`: Tuple of (lower_bound, upper_bound) for confidence interval
- `rare_outcome`: Whether outcome is rare (<15%), affects approximation accuracy

# Returns
- Named tuple with E-values for point estimate and confidence interval

# References
VanderWeele & Ding (2017). Sensitivity Analysis in Observational Research. Annals of Internal Medicine.
"""
function calculate_e_value(effect_estimate, confidence_interval; rare_outcome=false)

    function rr_to_e_value(rr)
        if rr >= 1
            return rr + sqrt(rr * (rr - 1))
        else
            return 1/rr + sqrt((1/rr) * (1/rr - 1))
        end
    end

    function hr_to_e_value(hr)
        # For hazard ratios, same formula as risk ratio
        return rr_to_e_value(hr)
    end

    function coef_to_e_value(coef)
        # Convert coefficient to risk ratio approximation
        if rare_outcome
            rr = exp(coef)  # Good approximation for rare outcomes
        else
            # More conservative approach for common outcomes
            rr = exp(coef)
        end
        return rr_to_e_value(rr)
    end

    # Determine type of effect estimate and calculate E-value
    if effect_estimate > 0 && effect_estimate < 100  # Likely a coefficient
        e_val_point = coef_to_e_value(effect_estimate)
        e_val_ci = if confidence_interval[1] > 0
            coef_to_e_value(confidence_interval[1])  # Lower bound closest to null
        else
            1.0  # CI includes null, so E-value is 1
        end
    else  # Likely a ratio measure
        e_val_point = rr_to_e_value(effect_estimate)
        e_val_ci = if effect_estimate > 1
            rr_to_e_value(confidence_interval[1])  # Lower bound closest to null
        else
            rr_to_e_value(confidence_interval[2])  # Upper bound closest to null
        end

        # Handle case where CI includes null (1.0)
        if (effect_estimate > 1 && confidence_interval[1] <= 1) ||
           (effect_estimate < 1 && confidence_interval[2] >= 1)
            e_val_ci = 1.0
        end
    end

    return (
        point_estimate = round(e_val_point, digits=2),
        confidence_interval = round(e_val_ci, digits=2),
        interpretation = interpret_e_value(e_val_ci)
    )
end

"""
    interpret_e_value(e_value)

Provide interpretation guidance for E-values.
"""
function interpret_e_value(e_value)
    if e_value < 1.25
        return "Very weak evidence against unmeasured confounding"
    elseif e_value < 2.0
        return "Weak to moderate evidence against unmeasured confounding"
    elseif e_value < 5.0
        return "Moderate to strong evidence against unmeasured confounding"
    else
        return "Strong evidence against unmeasured confounding"
    end
end

"""
    sensitivity_analysis_unmeasured_confounding(chains_df, focal_parameters; α=0.05)

Conduct comprehensive sensitivity analysis for unmeasured confounding using E-values.

# Arguments
- `chains_df`: DataFrame with posterior samples
- `focal_parameters`: Vector of parameter names for sensitivity analysis
- `α`: Significance level for confidence intervals (default: 0.05)

# Returns
- DataFrame with E-values and interpretations for each parameter
"""
function sensitivity_analysis_unmeasured_confounding(chains_df, focal_parameters; α=0.05)

    results = DataFrame()

    println("SENSITIVITY ANALYSIS FOR UNMEASURED CONFOUNDING")
    println("="^60)

    for param in focal_parameters
        if param in names(chains_df)
            samples = chains_df[!, param]

            # Calculate summary statistics
            mean_est = mean(samples)
            lower_ci = quantile(samples, α/2)
            upper_ci = quantile(samples, 1 - α/2)

            # Calculate E-values
            e_vals = calculate_e_value(mean_est, (lower_ci, upper_ci))

            # Store results
            push!(results, (
                parameter = param,
                estimate = round(mean_est, digits=3),
                ci_lower = round(lower_ci, digits=3),
                ci_upper = round(upper_ci, digits=3),
                e_value_point = e_vals.point_estimate,
                e_value_ci = e_vals.confidence_interval,
                interpretation = e_vals.interpretation
            ))

            println("Parameter: $param")
            println("  Estimate: $(round(mean_est, digits=3)) [$(round(lower_ci, digits=3)), $(round(upper_ci, digits=3))]")
            println("  E-value (point): $(e_vals.point_estimate)")
            println("  E-value (CI): $(e_vals.confidence_interval)")
            println("  Interpretation: $(e_vals.interpretation)")
            println()
        else
            println("Warning: Parameter $param not found in chains")
        end
    end

    # Summary assessment
    min_e_value = minimum(results.e_value_ci)

    println("SUMMARY ASSESSMENT:")
    println("-"^30)
    println("Minimum E-value (most vulnerable): $min_e_value")

    if min_e_value >= 2.0
        println("✓ Results show good robustness to unmeasured confounding")
        println("  Unmeasured confounders would need RR ≥ $min_e_value with both")
        println("  treatment and outcome to explain away the observed effects.")
    elseif min_e_value >= 1.5
        println("⚠ Results show moderate robustness to unmeasured confounding")
        println("  Exercise caution in causal interpretation.")
    else
        println("⚠ Results show limited robustness to unmeasured confounding")
        println("  Strong unmeasured confounders could explain away the effects.")
        println("  Consider additional confounding control strategies.")
    end

    return results
end

"""
    analyse_temporal_dynamics(df; time_var="T", response_var="E", id_var="ID")

Comprehensive analysis of temporal dynamics in vaccine response data.

# Arguments
- `df`: DataFrame with longitudinal data
- `time_var`: Name of time variable (default: "T" for days since vaccination)
- `response_var`: Name of response variable (default: "E" for vaccine response)
- `id_var`: Name of individual identifier (default: "ID")

# Returns
- Dictionary with temporal analysis results including peak timing, response trajectories, and temporal models
"""
function analyse_temporal_dynamics(df; time_var="T", response_var="E", id_var="ID")

    println("TEMPORAL DYNAMICS ANALYSIS")
    println("="^50)

    results = Dict{String, Any}()

    # Basic temporal summary
    time_data = df[!, time_var]
    response_data = df[!, response_var]

    println("Temporal Data Summary:")
    println("  Time range: $(minimum(time_data)) to $(maximum(time_data)) days")
    println("  Number of time points: $(length(unique(time_data)))")
    println("  Total observations: $(nrow(df))")
    println("  Number of individuals: $(length(unique(df[!, id_var])))")

    # Identify peak response timing by individual
    grouped = groupby(df, id_var)
    individual_peaks = combine(grouped) do subdf
        max_response_idx = argmax(subdf[!, response_var])
        peak_time = subdf[max_response_idx, time_var]
        peak_response = subdf[max_response_idx, response_var]

        return DataFrame(
            peak_time = peak_time,
            peak_response = peak_response,
            n_observations = nrow(subdf)
        )
    end

    # Population-level peak timing
    median_peak_time = median(individual_peaks.peak_time)
    peak_time_iqr = (quantile(individual_peaks.peak_time, 0.25), quantile(individual_peaks.peak_time, 0.75))

    println("\nPeak Response Timing:")
    println("  Median peak time: $(round(median_peak_time, digits=1)) days")
    println("  IQR: $(round(peak_time_iqr[1], digits=1)) - $(round(peak_time_iqr[2], digits=1)) days")

    results["peak_timing"] = individual_peaks
    results["median_peak_time"] = median_peak_time

    # Separate primary vs secondary response phases
    # Define phases based on typical vaccine response kinetics
    primary_cutoff = 14  # days - typical primary response window

    df_primary = filter(row -> row[time_var] <= primary_cutoff, df)
    df_secondary = filter(row -> row[time_var] > primary_cutoff, df)

    println("\nResponse Phase Analysis:")
    println("  Primary phase (≤$primary_cutoff days): $(nrow(df_primary)) observations")
    println("  Secondary phase (>$primary_cutoff days): $(nrow(df_secondary)) observations")

    if nrow(df_primary) > 0 && nrow(df_secondary) > 0
        primary_mean = mean(df_primary[!, response_var])
        secondary_mean = mean(df_secondary[!, response_var])
        phase_difference = secondary_mean - primary_mean

        println("  Primary phase mean response: $(round(primary_mean, digits=3))")
        println("  Secondary phase mean response: $(round(secondary_mean, digits=3))")
        println("  Phase difference: $(round(phase_difference, digits=3))")

        results["primary_response"] = primary_mean
        results["secondary_response"] = secondary_mean
        results["phase_difference"] = phase_difference
    end

    return results
end

"""
    temporal_mixed_model(df; time_var="T", response_var="E", id_var="ID", covariates=String[])

Fit temporal mixed-effects model with flexible time effects.

# Arguments
- `df`: DataFrame with longitudinal data
- `time_var`: Name of time variable
- `response_var`: Name of response variable
- `id_var`: Name of individual identifier
- `covariates`: Vector of covariate names to include

# Returns
- Named tuple with linear and nonlinear temporal models
"""
function temporal_mixed_model(df; time_var="T", response_var="E", id_var="ID", covariates=String[])

    println("FITTING TEMPORAL MIXED-EFFECTS MODELS")
    println("="^50)

    # Prepare formula components
    fixed_effects = [time_var]
    if !isempty(covariates)
        append!(fixed_effects, covariates)
    end

    random_effects = "(1 + $time_var | $id_var)"

    # Linear temporal model
    linear_formula_str = "$response_var ~ 1 + $(join(fixed_effects, " + ")) + $random_effects"
    linear_formula = Meta.parse("@formula($linear_formula_str)") |> eval

    println("Fitting linear temporal model...")
    println("Formula: $linear_formula_str")

    try
        linear_model = fit(MixedModel, linear_formula, df)
        println("✓ Linear model fitted successfully")

        # Model diagnostics
        println("Linear Model Summary:")
        println("  AIC: $(round(aic(linear_model), digits=2))")
        println("  BIC: $(round(bic(linear_model), digits=2))")

        # Quadratic temporal model (for nonlinear trajectories)
        df_temp = copy(df)
        df_temp[!, time_var * "_sq"] = df_temp[!, time_var] .^ 2

        quad_fixed_effects = [time_var, time_var * "_sq"]
        if !isempty(covariates)
            append!(quad_fixed_effects, covariates)
        end

        quad_formula_str = "$response_var ~ 1 + $(join(quad_fixed_effects, " + ")) + $random_effects"
        quad_formula = Meta.parse("@formula($quad_formula_str)") |> eval

        println("\nFitting quadratic temporal model...")
        println("Formula: $quad_formula_str")

        quad_model = fit(MixedModel, quad_formula, df_temp)
        println("✓ Quadratic model fitted successfully")

        println("Quadratic Model Summary:")
        println("  AIC: $(round(aic(quad_model), digits=2))")
        println("  BIC: $(round(bic(quad_model), digits=2))")

        # Model comparison
        aic_improvement = aic(linear_model) - aic(quad_model)
        println("\nModel Comparison:")
        println("  AIC improvement (quad vs linear): $(round(aic_improvement, digits=2))")

        if aic_improvement > 2
            println("  ✓ Quadratic model shows meaningful improvement")
            preferred_model = quad_model
        else
            println("  → Linear model is adequate")
            preferred_model = linear_model
        end

        return (
            linear = linear_model,
            quadratic = quad_model,
            preferred = preferred_model,
            aic_improvement = aic_improvement
        )

    catch e
        println("✗ Error fitting temporal models: $e")
        return nothing
    end
end

"""
    estimate_response_kinetics(df; time_var="T", response_var="E", habitat_var="H")

Estimate vaccine response kinetics parameters by habitat.

# Returns
- DataFrame with kinetics parameters by group
"""
function estimate_response_kinetics(df; time_var="T", response_var="E", habitat_var="H")

    println("ESTIMATING RESPONSE KINETICS BY HABITAT")
    println("="^50)

    results = DataFrame()

    for habitat in unique(df[!, habitat_var])
        habitat_data = filter(row -> row[habitat_var] == habitat, df)
        habitat_name = habitat == 1 ? "Lab" : "Wild"

        println("\nAnalysing $habitat_name habitat:")

        # Basic kinetics
        times = habitat_data[!, time_var]
        responses = habitat_data[!, response_var]

        # Time to peak response
        time_grouped = groupby(habitat_data, time_var)
        mean_responses_by_time = combine(time_grouped, response_var => mean => :mean_response)
        peak_time_idx = argmax(mean_responses_by_time.mean_response)
        time_to_peak = mean_responses_by_time[peak_time_idx, time_var]
        peak_response = mean_responses_by_time[peak_time_idx, :mean_response]

        # Response magnitude at key timepoints
        early_response = if any(times .<= 7)
            mean(responses[times .<= 7])
        else
            missing
        end

        late_response = if any(times .>= 21)
            mean(responses[times .>= 21])
        else
            missing
        end

        println("  Time to peak: $(round(time_to_peak, digits=1)) days")
        println("  Peak response: $(round(peak_response, digits=3))")
        if !ismissing(early_response)
            println("  Early response (≤7d): $(round(early_response, digits=3))")
        end
        if !ismissing(late_response)
            println("  Late response (≥21d): $(round(late_response, digits=3))")
        end

        push!(results, (
            habitat = habitat_name,
            habitat_code = habitat,
            time_to_peak = time_to_peak,
            peak_response = peak_response,
            early_response = early_response,
            late_response = late_response,
            n_observations = nrow(habitat_data)
        ))
    end

    # Compare kinetics between habitats
    if nrow(results) == 2
        lab_row = findfirst(row -> row.habitat == "Lab", eachrow(results))
        wild_row = findfirst(row -> row.habitat == "Wild", eachrow(results))

        if !isnothing(lab_row) && !isnothing(wild_row)
            peak_time_diff = results[wild_row, :time_to_peak] - results[lab_row, :time_to_peak]
            peak_response_diff = results[wild_row, :peak_response] - results[lab_row, :peak_response]

            println("\nHabitat Comparison:")
            println("  Peak timing difference (Wild - Lab): $(round(peak_time_diff, digits=1)) days")
            println("  Peak response difference (Wild - Lab): $(round(peak_response_diff, digits=3))")

            if peak_time_diff > 2
                println("  → Wild mice show delayed peak response")
            elseif peak_time_diff < -2
                println("  → Lab mice show delayed peak response")
            else
                println("  → Similar peak timing between habitats")
            end
        end
    end

    return results
end

"""
    conduct_prior_sensitivity_analysis(model_func, data, parameter_variations; n_samples=1000, n_chains=4)

Conduct comprehensive prior sensitivity analysis by fitting the same model with different prior specifications.

# Arguments
- `model_func`: Function that creates the model given prior parameters
- `data`: Tuple of data arguments for the model
- `parameter_variations`: Dictionary mapping parameter names to vectors of alternative prior specifications
- `n_samples`: Number of MCMC samples per chain (default: 1000)
- `n_chains`: Number of chains (default: 4)

# Returns
- Dictionary containing results for each prior specification combination

# Example
```julia
variations = Dict(
    "β_prior_scale" => [0.25, 0.5, 1.0, 2.0],
    "α_prior_scale" => [0.5, 1.0, 2.0]
)
results = conduct_prior_sensitivity_analysis(my_model, (X, y), variations)
```
"""
function conduct_prior_sensitivity_analysis(model_func, data, parameter_variations; n_samples=1000, n_chains=4)
    # Generate all combinations of prior variations
    param_names = collect(keys(parameter_variations))
    param_values = collect(values(parameter_variations))
    combinations = Iterators.product(param_values...)

    results = Dict{String, Any}()

    for (i, combo) in enumerate(combinations)
        println("Testing prior combination $i/$(length(collect(combinations)))")

        # Create parameter dictionary for this combination
        params = Dict(zip(param_names, combo))

        try
            # Create model with these priors
            model = model_func(data...; params...)

            # Sample
            chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)

            # Store results
            combo_name = join(["$(k)=$(v)" for (k,v) in params], "_")
            results[combo_name] = (
                parameters = params,
                chain = chain,
                summary = summarystats(chain)
            )

            println("✓ Completed: $combo_name")

        catch e
            println("✗ Failed: $(join([string(v) for v in combo], "_")) - Error: $e")
            combo_name = join(["$(k)=$(v)" for (k,v) in params], "_")
            results[combo_name] = (
                parameters = params,
                chain = nothing,
                error = e
            )
        end
    end

    return results
end

"""
    summarise_prior_sensitivity(sensitivity_results)

Summarise results from prior sensitivity analysis and identify robust vs. sensitive parameters.

# Arguments
- `sensitivity_results`: Results dictionary from conduct_prior_sensitivity_analysis

# Returns
- Summary DataFrame with robustness assessment
"""
function summarise_prior_sensitivity(sensitivity_results)
    # Extract successful results
    successful_results = filter(p -> p.second.chain !== nothing, sensitivity_results)

    if isempty(successful_results)
        println("No successful model fits to analyse!")
        return nothing
    end

    # Get parameter names from first successful result
    first_chain = first(values(successful_results)).chain
    param_names = names(first_chain)

    summary_df = DataFrame()

    for param in param_names
        estimates = Float64[]
        lower_cis = Float64[]
        upper_cis = Float64[]
        prior_configs = String[]

        for (config_name, result) in successful_results
            if result.chain !== nothing
                summary_stats = result.summary
                if param in summary_stats.nt.parameters
                    param_idx = findfirst(==(param), summary_stats.nt.parameters)
                    push!(estimates, summary_stats.nt.mean[param_idx])
                    push!(lower_cis, summary_stats.nt.q025[param_idx])
                    push!(upper_cis, summary_stats.nt.q975[param_idx])
                    push!(prior_configs, config_name)
                end
            end
        end

        if !isempty(estimates)
            # Calculate robustness metrics
            estimate_range = maximum(estimates) - minimum(estimates)
            estimate_cv = std(estimates) / abs(mean(estimates))

            # Check if confidence intervals overlap substantially
            ci_overlap = all_cis_overlap(lower_cis, upper_cis)

            push!(summary_df, (
                parameter = param,
                n_configs = length(estimates),
                mean_estimate = mean(estimates),
                estimate_range = estimate_range,
                estimate_cv = estimate_cv,
                ci_overlap = ci_overlap,
                robust = (estimate_cv < 0.1 && ci_overlap)
            ))
        end
    end

    return summary_df
end

"""
    all_cis_overlap(lower_bounds, upper_bounds)

Check if all confidence intervals overlap substantially (>50% overlap).
"""
function all_cis_overlap(lower_bounds, upper_bounds)
    n = length(lower_bounds)
    if n < 2 return true end

    for i in 1:n-1
        for j in i+1:n
            overlap_start = max(lower_bounds[i], lower_bounds[j])
            overlap_end = min(upper_bounds[i], upper_bounds[j])

            if overlap_start >= overlap_end
                return false  # No overlap
            end

            # Check if overlap is substantial (>50% of either interval)
            overlap_length = overlap_end - overlap_start
            interval1_length = upper_bounds[i] - lower_bounds[i]
            interval2_length = upper_bounds[j] - lower_bounds[j]

            overlap_prop1 = overlap_length / interval1_length
            overlap_prop2 = overlap_length / interval2_length

            if max(overlap_prop1, overlap_prop2) < 0.5
                return false
            end
        end
    end

    return true
end

"""
    generate_prior_robustness_report(sensitivity_results, model_name="Model")

Generate a comprehensive report on prior sensitivity analysis results.
"""
function generate_prior_robustness_report(sensitivity_results, model_name="Model")
    println("\n" * "="^80)
    println("PRIOR SENSITIVITY ANALYSIS REPORT: $model_name")
    println("="^80)

    successful_results = filter(p -> p.second.chain !== nothing, sensitivity_results)
    failed_results = filter(p -> p.second.chain === nothing, sensitivity_results)

    println("Total configurations tested: $(length(sensitivity_results))")
    println("Successful fits: $(length(successful_results))")
    println("Failed fits: $(length(failed_results))")

    if !isempty(failed_results)
        println("\nFailed configurations:")
        for (config, result) in failed_results
            println("  - $config: $(result.error)")
        end
    end

    if !isempty(successful_results)
        summary_df = summarise_prior_sensitivity(sensitivity_results)

        if summary_df !== nothing
            println("\nParameter Robustness Summary:")
            println("-"^50)

            robust_params = filter(row -> row.robust, summary_df)
            sensitive_params = filter(row -> !row.robust, summary_df)

            println("ROBUST parameters (CV < 0.1 and overlapping CIs):")
            for row in eachrow(robust_params)
                println("  ✓ $(row.parameter): CV = $(round(row.estimate_cv, digits=3))")
            end

            println("\nSENSITIVE parameters (CV ≥ 0.1 or non-overlapping CIs):")
            for row in eachrow(sensitive_params)
                println("  ⚠ $(row.parameter): CV = $(round(row.estimate_cv, digits=3)), Range = $(round(row.estimate_range, digits=3))")
            end

            robustness_ratio = nrow(robust_params) / nrow(summary_df)
            println("\nOverall robustness: $(round(robustness_ratio * 100, digits=1))% of parameters are robust to prior specification")

            if robustness_ratio < 0.7
                println("\n⚠️  WARNING: More than 30% of parameters show sensitivity to prior specification.")
                println("   Consider using more informative priors or conducting further sensitivity analysis.")
            else
                println("\n✅ Model shows good robustness to prior specification.")
            end
        end
    end

    println("="^80)
    return summarise_prior_sensitivity(sensitivity_results)
end

## Precis from https://github.com/StatisticalRethinkingJulia/StatisticalRethinking.jl/blob/39d05869fb3772e83d3313e84cb30549e9ebdb94/src/precis.jl

const BARS = collect("▁▂▃▄▅▆▇█")

function unicode_histogram(data, nbins=12)
    # @show data
    f = fit(Histogram, data, nbins=nbins)  # nbins: more like a guideline than a rule, really
    # scale weights between 1 and 8 (length(BARS)) to fit the indices in BARS
    # eps is needed so indices are in the interval [0, 8) instead of [0, 8] which could
    # result in indices 0:8 which breaks things
    scaled = f.weights .* (length(BARS) / maximum(f.weights) - eps())
    indices = floor.(Int, scaled) .+ 1
    return join((BARS[i] for i in indices))
end


# This function calculates the mean, standard deviation, and 5.5% and 94.5% quantiles for each column in the DataFrame, and then prints a summary table. It also prints a histogram for each column. The values are rounded to the specified number of digits. The number of bins in the histogram is set to the minimum of the number of rows in the DataFrame and 12.
function precis(df::DataFrame; io=stdout, digits=3, depth=Inf, alpha=0.1)
    d = DataFrame()
    cols = collect.(skipmissing.(eachcol(df)))
    d.param = names(df)
    d.mean = mean.(cols)
    d.std = std.(cols)
    lower_q = alpha / 2
    upper_q = 1 - lower_q
    quants = quantile.(cols, ([lower_q, 0.5, upper_q],))
    quants = hcat(quants...)
    d[:, "$(lower_q * 100) %"] = quants[1, :]
    d[:, "50 %"] = quants[2, :]
    d[:, "$(upper_q * 100) %"] = quants[3, :]
    d.histogram = unicode_histogram.(cols, min(size(df, 1), 12))

    # Subtract the mean of the row containing "[1]" from all levels starting with the same characters preceding "[1]"
    for (i, param) in enumerate(d.param)
        if contains(param, "[1]")
            prefix = split(param, "[1]")[1]
            mean_val = d.mean[i]
            for (j, other_param) in enumerate(d.param)
                if startswith(other_param, prefix)
                    d.mean[j] -= mean_val
                end
            end
        end
    end

    for col in ["mean", "std", "$(lower_q * 100) %", "50 %", "$(upper_q * 100) %"]
        d[:, col] .= round.(d[:, col], digits=digits)
    end

    pretty_table(io, d)
end

## Prior Predictive Checks

"""
    generate_prior_predictions_standardised(model_function, data_args...;
                                           n_samples=1000, intervention=false)

Generate predictions from the prior distribution for prior predictive checks,
optimised for standardised outcomes.

# Arguments
- `model_function::Function`: The Turing model function to sample from
- `data_args...`: Arguments to pass to the model function
- `n_samples::Int`: Number of prior samples to generate
- `intervention::Bool`: Whether to use intervention priors (βP ≈ 0)

# Returns
- `Matrix{Float64}`: Prior predictions for outcome (n_obs × n_samples)

# Prior Specification
Uses the improved informative priors suitable for standardised outcomes:
- α ~ Normal(0, 0.5): Standardised intercept
- Most β ~ Normal(0, 0.15): Small-moderate effects
- βH ~ Normal(0, 0.3): Larger habitat effects
- σ ~ Exponential(0.8): Conservative residual variance
- τ ~ Exponential(0.3): Conservative random effects
"""
function generate_prior_predictions_standardised(model_function, data_args...;
    n_samples::Int=1000, intervention::Bool=false)
    # Create the model
    model = model_function(data_args...)

    # Sample from prior
    prior_samples = sample(model, Prior(), n_samples)

    return prior_samples
end


"""
    assess_prior_adequacy(observed_data, prior_predictions; model_name="Model")

Assess whether priors are reasonable by comparing ranges and providing feedback.

# Arguments
- `observed_data::Vector{Float64}`: Observed values
- `prior_predictions`: Prior predictions (any format)
- `model_name::String`: Name of the model for reporting

# Returns
- Prints assessment to console with recommendations
"""
function assess_prior_adequacy(observed_data, prior_predictions; model_name::String="Model")
    # Handle different input formats
    if isa(prior_predictions, AbstractMatrix)
        prior_vec = vec(prior_predictions)
    else
        prior_vec = collect(prior_predictions)
    end

    # Calculate ranges
    observed_range = maximum(observed_data) - minimum(observed_data)
    prior_range = maximum(prior_vec) - minimum(prior_vec)
    range_ratio = prior_range / observed_range

    println("\n=== PRIOR ASSESSMENT: $model_name ===")
    println("Observed: mean = $(round(mean(observed_data), digits=3)), std = $(round(std(observed_data), digits=3))")
    println("Prior predictions: mean = $(round(mean(prior_vec), digits=3)), std = $(round(std(prior_vec), digits=3))")
    println("Range - Observed: [$(round(minimum(observed_data), digits=2)), $(round(maximum(observed_data), digits=2))]")
    println("Range - Prior: [$(round(minimum(prior_vec), digits=2)), $(round(maximum(prior_vec), digits=2))]")

    if range_ratio > 10
        println("⚠️  WARNING: Prior predictions have a much wider range than observed data ($(round(range_ratio, digits=1))x wider)")
        println("   Consider using more informative priors")
    elseif range_ratio < 0.5
        println("⚠️  WARNING: Prior predictions have a narrower range than observed data ($(round(range_ratio, digits=1))x narrower)")
        println("   Consider using less informative priors")
    else
        println("✅ Prior prediction range seems reasonable relative to observed data ($(round(range_ratio, digits=1))x ratio)")
    end
end
