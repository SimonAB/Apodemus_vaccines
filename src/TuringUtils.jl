#=
TuringUtils:
- Julia version: 1.9
- Author: Simon A Babayan
- Date: 2022-03-25
=#

using PrettyTables

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
    idx = Tables.getcolumn(data, col)
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
    return Tables.getcolumn(data, col)
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
- α∼Normal(0,2.5) – This means a normal distribution centered on 0 with a standard deviation of 2.5. That prior should with ease cover all possible values of α. Remember that the normal distribution has support over all the real number line ∈(−∞,+∞).
- β∼Student-t(0,1,3) – The predictors all have a prior distribution of a Student-t distribution centered on 0 with variance 1 and degrees of freedom ν=3. That wide-tailed t distribution will cover all possible values for our coefficients. Remember the Student-t also has support over all the real number line ∈(−∞,+∞). Also the filldist() is a nice Turing's function which takes any univariate or multivariate distribution and returns another distribution that repeats the input distribution.

Turing's arraydist() function wraps an array of distributions returning a new distribution sampling from the individual distributions. And the LazyArrays' LazyArray() constructor wrap a lazy object that wraps a computation producing an array to an array. Last, but not least, the macro @~ creates a broadcast and is a nice short hand for the familiar dot . broadcasting operator in Julia. This is an efficient way to tell Turing that our y vector is distributed lazily as a NegativeBinomial2 broadcasted to α added to the product of the data matrix X and β coefficient vector. Note that NegativeBinomial2 does not apply exponentiation so we had to include the exp.() broadcasted function to all the linear predictors.

See: https://storopoli.github.io/Bayesian-Julia/pages/09_count_reg/#using_poisson_likelihood
"""
@model function poissonreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)

    #likelihood
    y ~ arraydist(LazyArray(@~ LogPoisson.(α .+ X * β)))
end;

"""
- α∼Normal(0,2.5) – This means a normal distribution centered on 0 with a standard deviation of 2.5. That prior should with ease cover all possible values of α. Remember that the normal distribution has support over all the real number line ∈(−∞,+∞).
- β∼Student-t(0,1,3) – The predictors all have a prior distribution of a Student-t distribution centered on 0 with variance 1 and degrees of freedom ν=3. That wide-tailed t distribution will cover all possible values for our coefficients. Remember the Student-t also has support over all the real number line ∈(−∞,+∞). Also the filldist() is a nice Turing's function which takes any univariate or multivariate distribution and returns another distribution that repeats the input distribution.
- ϕ∼Exponential(1) – overdispersion parameter of the negative binomial distribution.

Turing's arraydist() function wraps an array of distributions returning a new distribution sampling from the individual distributions. And the LazyArrays' LazyArray() constructor wrap a lazy object that wraps a computation producing an array to an array. Last, but not least, the macro @~ creates a broadcast and is a nice short hand for the familiar dot . broadcasting operator in Julia. This is an efficient way to tell Turing that our y vector is distributed lazily as a NegativeBinomial2 broadcasted to α added to the product of the data matrix X and β coefficient vector. Note that NegativeBinomial2 does not apply exponentiation so we had to include the exp.() broadcasted function to all the linear predictors.
See: https://storopoli.github.io/Bayesian-Julia/pages/09_count_reg/#using_poisson_likelihood
"""
@model function negbinreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)
    ϕ⁻ ~ Gamma(0.01, 0.01)
    ϕ = 1 / ϕ⁻

    #likelihood
    y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(α .+ X * β), ϕ)))
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
function precis(df::DataFrame; io=stdout, digits=4, depth=Inf, alpha=0.11)
    d = DataFrame()
    cols = collect.(skipmissing.(eachcol(df)))
    d.param = names(df)
    d.mean = mean.(cols)
    d.std = std.(cols)
    quants = quantile.(cols, ([alpha / 2, 0.5, 1 - alpha / 2],))
    quants = hcat(quants...)
    d[:, "5.5%"] = quants[1, :]
    d[:, "50%"] = quants[2, :]
    d[:, "94.5%"] = quants[3, :]
    d.histogram = unicode_histogram.(cols, min(size(df, 1), 12))

    for col in ["mean", "std", "5.5%", "50%", "94.5%"]
        d[:, col] .= round.(d[:, col], digits=digits)
    end

    pretty_table(io, d, nosubheader=true, vlines=[0, 1, 7])
end
