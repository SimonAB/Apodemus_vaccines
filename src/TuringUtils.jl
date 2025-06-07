#=
TuringUtils:
- Julia version: 1.11
- Author: Simon A Babayan
=#

#=
This script contains utility functions for the Turing models.
=#

# Required imports for Turing utilities
using PrettyTables
using DataFrames
using Statistics: mean, std
using StatsBase: fit, Histogram, quantile
using Distributions: Normal, Exponential, MvNormal, TDist, Gamma, NegativeBinomial, Cauchy, Poisson
using LinearAlgebra: I
using Turing: @model, filldist, truncated, arraydist

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

    pretty_table(io, d, vlines=[0, 1, 7])
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
