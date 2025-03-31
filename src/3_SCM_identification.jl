#=
SCM Identification
- Julia version: 1.11
- Author: Simon A Babayan
=#

#=
This script will statistically identify the SCM using the data from the manuscript.
=#

## Import packages

print("Running on ", Threads.nthreads(), " threads.")
# data handling
using CSV, DataFrames, Query
# stats
using Random
using Distributions
using HypothesisTests
using MixedModels
# modelling
using LazyArrays
using LinearAlgebra: I
using MCMCChains
using Turing
using ReverseDiff
# Turing.setbackend(:reversediff)
# Turing.setrdcache(true)

using RCall
@rlibrary dagitty # we use the original dagitty from R until julia native version improves

# plotting & diagnostics
using CairoMakie
CairoMakie.activate!(type="svg")
using MixedModelsMakie
# using Formatting

# include modules
cd("./src/")
include("TuringUtils.jl")
include("TuringPlots.jl")

# import data
include("DataWrangler.jl")

# all cases
df = encode_df(df) # choose between df and df_unique (the latter has no repeated measures)
df =
  df |>
  # @filter(_.vax_history != "DD") |> # keep only mice with a single vaccination or adjuvant
  @filter(_.days_since_1st_D_or_A ≥ 4) |> # remove entries which were measured less than a week after vaccination
  DataFrame
df.IDidx = get_idx(:ID, df)[1]


# restrict to unique cases (no repeated measures):
df_unique = encode_df(df_unique)
df_unique =
  df_unique |>
  @filter(_.days_since_1st_D_or_A ≥ 4) |> # remove entries which were measured less than a week after vaccination
  DataFrame

# Build DAG dataFrame
dag_df = df[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :P, :nP, :ID, :IDidx, :vax_history, :Vidx]]
dag_df.lognP = log10.(1 .+ dag_df.nP);
# describe(dag_df)

dag_df_unique = df_unique[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :P, :nP, :Vidx, :vax_history, :ID]];
filtered_df = dag_df[dag_df.P.==1, :];
filtered_unique_df = dag_df_unique[dag_df_unique.P.==1, :];

# select only infected mice
dag_df_infected = dag_df[dag_df.P.==2, :] # select rows of dag_df for which dag_df.P.==2

# DAG specification - this is our graphical causal hypothesis

dag = dagitty("dag{ D -> E; D -> F; D -> M; D -> P; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> P; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; R -> P; S -> E; S -> F; S -> M; S -> P; S -> R; V -> E; V -> F; V -> M; V -> P; V -> R; }")

## Average causal effect of V on E
adjustmentSets(dag, "V", "E", effect="total") # {} -> V is assumed to be a RCT

@model function V_E(IDidx, E, V; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))
  βVE ~ Normal(0, 0.5)
  σ ~ Exponential(std(E))
  ν ~ LogNormal(2, 1)

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts

  # likelihood
  Ê = @. α + α_ID[IDidx] + βVE * V
  return E ~ MvNormal(Ê, σ^2 * I)
end

V_E_model = V_E(dag_df.IDidx, dag_df.E, dag_df.V); # note log10(1+X)-transformed parasite counts.

V_E_chn = sample(V_E_model, NUTS(), MCMCThreads(), 3000, 4)

V_E_chn_df = DataFrame(V_E_chn)[!, r"α\b|β"];
precis(V_E_chn_df)

plot_chains_df(V_E_chn)

## Direct effect of V on E

adjustmentSets(dag, "V", "E", effect="direct") # { D, F, H, M, P, R, S }

@model function V_E_NDE(IDidx, E, V, D, Ḟ, H, M, P, R, S; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))  # overall intercept
  βVE ~ Normal(0, 0.5)  # slope of V on E
  βDE ~ Normal(0, 0.5)  # slope of D on E
  βFE ~ Normal(0, 0.5)  # slope of F on E
  βHE ~ Normal(0, 0.5)  # slope of H on E
  βME ~ Normal(0, 0.5)  # slope of M on E
  βPE ~ Normal(0, 0.5)  # slope of P on E
  βRE ~ Normal(0, 0.5)  # slope of R on E
  βSE ~ Normal(0, 0.5)  # slope of S on E
  σ ~ Exponential(std(E))  # residual SD
  ν ~ LogNormal(2, 1)  # residual degrees of freedom

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)     # group-level intercepts

  # missing F values
  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(Normal(0, 1), N_missing)
  ν ~ Normal(0, 0.5) # imputed mean
  σ_F ~ Exponential() # imputed SD

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      F_impute[i_missing] ~ Normal(ν, σ_F)
      f_imputed = F_impute[i_missing]
      i_missing += 1
    else
      Ḟ[i] ~ Normal(ν, σ_F)
      f_imputed = Ḟ[i]
    end
    # likelihood
    Ê = @. α + α_ID[IDidx][i] + βVE * V[i] + βDE * D[i] + βFE * f_imputed + βHE * H[i] + βME * M[i] + βPE * P[i] + βRE * R[i] + βSE * S[i]
    E[i] ~ Normal(Ê, σ)
  end
end

V_E_DE_model = V_E_NDE(dag_df.IDidx, dag_df.E, dag_df.V, dag_df.D, dag_df.Ḟ, dag_df.H, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S); # note log10(1+X)-transformed parasite counts.

# Turing.setadbackend(:forwarddiff)
V_E_DE_chn = sample(V_E_DE_model, NUTS(), MCMCThreads(), 3000, 4)

V_E_DE_chn_df = DataFrame(V_E_DE_chn)[!, r"α\b|β"];
precis(V_E_DE_chn_df)

# Mixed model
glmm_V_E_NDE = fit(MixedModel, @formula(E ~ 1 + V + D + Ḟ + H + M + P + R + S + (1 | ID)), dag_df)

qqnorm(glmm_V_E_NDE; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_NDE);
coefplot(boot)
ridgeplot(boot)


## Total effect of `nP` on `E`
adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V}

# Plot
layers = linear() + visual(Scatter)
plt = data(dag_df_infected) * mapping(:lognP, :E)
p = draw(layers * plt, axis=(xlabel="H. polygyrus worm count (log10(1 + nP))", ylabel="α-DT IgG1 (standardised)"))
save("../manuscript/Figures/plots/P_E_cor.pdf", p)

# Naive model (does not use adjustment set)
# point estimate sanity check
naive_glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + (1 | ID)), dag_df_infected; progress=false)

qqnorm(naive_glmm_P_E; qqline=:fitrobust)

boot = parametricbootstrap(MersenneTwister(42), 3000, naive_glmm_P_E);
cp = coefplot(boot; conf_level=0.95)
save("../manuscript/Figures/plots/P_E_coefplot.pdf", cp)
ridgeplot(boot; conf_level=0.95)
save("../manuscript/Figures/plots/P_E_ridgeplot.pdf", cp)

# Bayesian model
@model function naive_P_E(IDidx, E, nP; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))
  βPE ~ Normal(0, 0.5)
  σ ~ Exponential(std(E))
  # ν ~ LogNormal(2, 1)

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts

  # likelihood
  Ê = @. α + α_ID[IDidx] + βPE * nP
  return E ~ MvNormal(Ê, σ^2 * I)
end

naive_P_E_model = naive_P_E(dag_df_infected.IDidx, dag_df_infected.E, dag_df_infected.lognP); # note log10(1+X)-transformed parasite counts.
# naive_P_E_model = naive_P_E(dag_df.IDidx, dag_df.E, dag_df.P); # note Probability of being infected

# naive_P_E_model = varying_intercept(dag_df.nP, dag_df.IDidx, dag_df.E); # note log10(1+X)-transformed parasite counts.

naive_P_E_chn = sample(naive_P_E_model, NUTS(), MCMCThreads(), 3000, 4);

naive_P_E_chn_df = DataFrame(naive_P_E_chn)[!, r"α\b|β"];
precis(naive_P_E_chn_df)

p = plot_chains_df(naive_P_E_chn; show_intercept=true, show_traces=false)
save("../manuscript/Figures/plots/naive_P_E_chn_df.pdf", p)

## Properly-adjusted model: total effect of `nP` on `E`
adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V }

# Mixed Model - excluding H since length(unique(dag_df_infected.H))=1
glmm_P_E_all = fit(MixedModel, @formula(E ~ 1 + lognP + D + H + R + S + V + (1 | ID)), dag_df)

qqnorm(glmm_P_E_all; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E_all);
coefplot(glmm_P_E_all)
coefplot(boot)
ridgeplot(boot)

# Infected only
glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + H + R + S + V + (1 | ID)), dag_df_infected)

qqnorm(glmm_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E);
coefplot(boot)
ridgeplot(boot)
ridge2d(boot)

# Direct effect of P on E among the infected
adjustmentSets(dag, "P", "E", effect="direct") # { D, F, H, M, R, S, V}

# Mixed model

glmm_DE_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + Ḟ + M + R + S + V + (1 | ID)), dag_df_infected)

qqnorm(glmm_DE_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_DE_P_E);
coefplot(boot)
ridgeplot(boot)

# Population-level model for the log of the expected number
@model function P_E(IDidx, E, P, D, H, R, S, V; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))
  βPE ~ Normal(0, 0.5)
  βDE ~ Normal(0, 0.5)
  βHE ~ Normal(0, 0.5)
  βRE ~ Normal(0, 0.5)
  βSE ~ Normal(0, 0.5)
  βVE ~ Normal(0, 0.5)
  σ ~ Exponential(std(E))
  ν ~ LogNormal(2, 1)

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts

  # likelihood
  Ê = α .+ α_ID[IDidx] .+ βPE * P .+ βDE * D .+ βHE * H .+ βRE * R .+ βSE * S .+ βVE * V
  return E ~ MvNormal(Ê, σ^2 * I)
end

P_E_model = P_E(dag_df_infected.IDidx, dag_df_infected.E, log10.(1 .+ dag_df_infected.nP), dag_df_infected.D, dag_df_infected.H, dag_df_infected.R, dag_df_infected.S, dag_df_infected.V); # note log10(1+X)-transformed parasite counts.


P_E_priors = sample(P_E_model, Prior(), MCMCThreads(), 3000, 4);
summarize(P_E_priors)

P_E_priors_df = DataFrame(P_E_priors)[!, r"α\b|β"];
precis(P_E_priors_df)

p = plot_chains_df(P_E_priors; show_intercept=true)
save("../manuscript/Figures/plots/P_E_priors.pdf", p)
p

# updating with data
P_E_chn = sample(P_E_model, NUTS(), MCMCThreads(), 3000, 4);

P_E_chn_df = DataFrame(P_E_chn)[!, r"α\b|β"];
precis(P_E_chn_df)

p = plot_chains_df(P_E_chn)
save("../manuscript/Figures/plots/P_E_chn_df.pdf", p)

## Direct effect of `P` on `E`

# Minimal sufficient adjustment set for estimating the direct effect of P on E:
adjustmentSets(dag, "P", "E", effect="direct") # { D, F, H, M, R, S, V }

# GLMM

glmm_DE_P_E = fit(MixedModel, @formula(E ~ 1 + P + D + Ḟ + H + M + R + S + V + (1 | ID)), dag_df)

qqnorm(glmm_DE_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_DE_P_E);
coefplot(boot)
ridgeplot(boot)

# Bayesian model

@model function DE_P_E(IDidx, E, P, D, Ḟ, H, M, R, S, V; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))

  βPE ~ Normal(0, 0.5)
  βDE ~ Normal(0, 0.5)
  βFE ~ Normal(0, 0.5)
  βHE ~ Normal(0, 0.5)
  βME ~ Normal(0, 0.5)
  βRE ~ Normal(0, 0.5)
  βSE ~ Normal(0, 0.5)
  βVE ~ Normal(0, 0.5)

  σ ~ Exponential(std(E))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts

  # missing F values
  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(Normal(0, 1), N_missing)
  F_impute = convert(Array, F_impute)
  ν ~ Normal(0, 0.5) # imputed mean
  σ_F ~ Exponential() # imputed SD

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      F_impute[i_missing] ~ Normal(ν, σ_F)
      f_imputed = F_impute[i_missing]
      i_missing += 1
    else
      Ḟ[i] ~ Normal(ν, σ_F)
      f_imputed = Ḟ[i]
    end
    # likelihood
    Ê = @. α + α_ID[IDidx][i] + βPE * P[i] + βDE * D[i] + βFE * f_imputed + βHE * H[i] + βME * M[i] + βRE * R[i] + βSE * S[i] + βVE * V[i]
    E[i] ~ Normal(Ê, σ)
  end
end

DE_P_E_model = DE_P_E(dag_df.IDidx, dag_df.E, dag_df.P, dag_df.D, dag_df.Ḟ, dag_df.H, dag_df.M, dag_df.R, dag_df.S, dag_df.V);

Turing.setadbackend(:forwarddiff)
# Turing.setrdcache(false)
DE_P_E_chn = sample(DE_P_E_model, NUTS(), MCMCThreads(), 3000, 4);
Turing.setadbackend(:reversediff)
# Turing.setrdcache(true)

DE_P_E_chn_df = DataFrame(DE_P_E_chn)[!, r"α\b|β"];
precis(DE_P_E_chn_df)

p1 = plot_chains_df(DE_P_E_chn; show_traces=true)
save("../manuscript/Figures/plots/MultiLevel_DE_P_E_chn_traces.pdf", p1)

p2 = plot_chains_df(DE_P_E_chn; show_traces=false, xlab_dist="Direct Casual Effect")
save("../manuscript/Figures/plots/MultiLevel_DE_P_E_chn.pdf", p2)

## Direct effect of reproductive status R on parasite burden P

adjustmentSets(dag, "R", "P", effect="direct") # D, H, S, V

# GLMM

glmm_R_nP = fit(MixedModel, @formula(lognP ~ R + D + H + S + V + (1 | ID)), dag_df)

# Bayesian model
@model function R_nP(nP, R, D, H, S, V, IDidx; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(0, 2.5)
  βR ~ Normal(0, 0.5)
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)
  βV ~ Normal(0, 0.5)
  σ ~ Exponential(std(nP))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)

  # likelihood
  nP̂ = @. α + α_ID[IDidx] + βR * R + βD * D + βH * H + βS * S + βV * V
  nP ~ MvNormal(nP̂, σ^2 * I)
end


R_nP_model = R_nP(log10.(1 .+ dag_df.nP), dag_df.R, dag_df.D, dag_df.H, dag_df.S, dag_df.V, dag_df.IDidx)

R_nP_chn = sample(R_nP_model, NUTS(), MCMCThreads(), 3_000, 4);

R_nP_chn_df = DataFrame(R_nP_chn)[!, r"α\b|β"];
precis(R_nP_chn_df)

p = plot_chains_df(R_nP_chn; show_intercept=true)
save("../manuscript/Figures/plots/R_nP_chn.pdf", p)

## Total effect of D on nP [among the infected]

adjustmentSets(dag, "D", "P", effect="total") # {H}

@model function D_nP(nP, D, H, IDidx; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(0, 2.5)
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 0.5)
  σ ~ Exponential(std(nP))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)

  # likelihood
  nP̂ = @. α + α_ID[IDidx] + βD * D + βH * H
  nP ~ MvNormal(nP̂, σ^2)
end

D_nP_model = D_nP(log10.(1 .+ dag_df.nP), dag_df.D, dag_df.H, dag_df.IDidx) # among all mice
# D_nP_model = D_nP(dag_df_infected.nP, dag_df_infected.D, dag_df_infected.H, dag_df_infected.IDidx) # among the infected; this doesn't converge...

D_nP_chn = sample(D_nP_model, NUTS(), MCMCThreads(), 3_000, 4);
D_nP_chn_df = DataFrame(D_nP_chn)[!, r"α\b|β"];
precis(D_nP_chn_df)

# GLMM for the total effect of D on nP among the infected

glmm_D_nP = fit(MixedModel, @formula(lognP ~ D + (1 | ID)), dag_df_infected)

boot = parametricbootstrap(MersenneTwister(1234), 10000, glmm_D_nP);
coefplot(boot)
ridgeplot(boot)

# Plot
df = (; x=Bool.(dag_df.D[dag_df.P.==2] .- 1), y=dag_df.nP[dag_df.P.==2])
layers = visual(BoxPlot)
plt = data(df) * mapping(:x, :y, color=:x)
p = draw(layers * plt, axis=(xlabel="Diet supplemented", ylabel="H. polygyrus (count)"))


## Direct effet of D on nP
adjustmentSets(dag, "D", "P", effect="direct") # { H, R, S, V}

@model function D_nP(nP, D, H, R, S, V, IDidx; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(0, 2.5)
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 0.5)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)
  βV ~ Normal(0, 0.5)

  σ ~ Exponential(std(nP))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)

  # likelihood
  nP̂ = @. α + α_ID[IDidx] + βD * D + βH * H + βR * R + βS * S + βV * V
  nP ~ MvNormal(nP̂, σ^2 * I)
end

D_nP_model = D_nP(log10.(1 .+ dag_df.nP), dag_df.D, dag_df.H, dag_df.R, dag_df.S, dag_df.V, dag_df.IDidx)

D_nP_chn = sample(D_nP_model, NUTS(), MCMCThreads(), 3_000, 4);

D_nP_chn_df = DataFrame(D_nP_chn)[!, r"α\b|β"];
precis(D_nP_chn_df)

p = plot_chains_df(D_nP_chn; show_intercept=true, show_traces=false)
save("../manuscript/Figures/plots/D_nP_chn.pdf", p)

## Total effect of S on E

adjustmentSets(dag, "S", "E", effect="total") # { }

glmm_S_E = fit(MixedModel, @formula(E ~ 1 + S + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_E);
coefplot(boot)
ridgeplot(boot)

# Bayesian model

@model function S_E(IDidx, E, S; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))  # overall intercept
  βS_ ~ Normal(0, 0.5)  # slope of S on E
  σ ~ Exponential(std(E))  # residual SD

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)     # group-level intercepts

  # likelihood
  Ê = @. α + α_ID[IDidx] + βS_ * S
  return E ~ MvNormal(Ê, σ^2 * I)
end

S_E_model = S_E(dag_df.IDidx, dag_df.E, dag_df.S); # note log10(1+X)-transformed parasite counts.

S_E_chn = sample(S_E_model, NUTS(), MCMCThreads(), 3000, 4)

S_E_chn_df = DataFrame(S_E_chn)[!, r"α\b|β"];
precis(S_E_chn_df)

## Direct effect of S on E

adjustmentSets(dag, "S", "E", effect="direct") # { D, F, H, M, P, R, V }

glmm_D_S_E = fit(MixedModel, @formula(E ~ 1 + S + D + Ḟ + H + M + nP + R + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_S_E);
coefplot(boot)
ridgeplot(boot)

# Bayesian model
@model function S_E(E, S, D, Ḟ, H, M, P, R, V, IDidx; n_id=length(unique(IDidx)))

  α ~ Normal(mean(E), 2.5 * std(E))
  σ ~ Exponential(std(E))

  βS ~ Normal(0, 1)
  βD ~ Normal(0, 1)
  βF ~ Normal(0, 1)
  βH ~ Normal(0, 1)
  βM ~ Normal(0, 1)
  βP ~ Normal(0, 1)
  βR ~ Normal(0, 1)
  βV ~ Normal(0, 1)

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts

  # missing F values
  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(Normal(), N_missing)
  ν ~ Normal(0.5, 1)
  σ_F ~ Exponential()

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      F_impute[i_missing] ~ Normal(ν, σ_F)
      f_imputed = F_impute[i_missing]
      i_missing += 1
    else
      Ḟ[i] ~ Normal(ν, σ_F)
      f_imputed = Ḟ[i]
    end

    # likelihood
    µ = @. α + α_ID[IDidx][i] + βS * S[i] + βD * D[i] + βF * f_imputed + βH * H[i] + βM * M[i] + βP * P[i] + βR * R[i] + βV * V[i]
    E[i] ~ Normal(µ, σ)
  end
end

S_E_model = S_E(dag_df.E, dag_df.S, dag_df.D, dag_df.Ḟ, dag_df.H, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.T, dag_df.IDidx)

Turing.setadbackend(:forwarddiff)
S_E_ch = sample(S_E_model, NUTS(), MCMCThreads(), 3000, 4)
Turing.setadbackend(:reverse_diff)

S_E_df = DataFrame(S_E_ch)[!, r"α\b|β"];
precis(S_E_df)


## Direct effect of S on nP

adjustmentSets(dag, "S", "P", effect="direct") # {D, H, R, V}

@model function S_nP(nP, S, D, H, R, V, IDidx; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(0, 2.5)
  βS ~ Normal(0, 0.5)
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 0.5)
  βR ~ Normal(0, 0.5)
  βV ~ Normal(0, 0.5)

  σ ~ Exponential(std(nP))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)

  # likelihood
  nP̂ = @. α + α_ID[IDidx] + βS * S + βD * D + βH * H + βR * R + βV * V
  nP ~ MvNormal(nP̂, σ^2 * I)
end

S_nP_model = S_nP(log10.(1 .+ dag_df.nP), dag_df.S, dag_df.D, dag_df.H, dag_df.R, dag_df.V, dag_df.IDidx)

S_nP_chn = sample(S_nP_model, NUTS(), MCMCThreads(), 3_000, 4);

S_nP_chn_df = DataFrame(S_nP_chn)[!, r"α\b|β"];
precis(S_nP_chn_df)

p = plot_chains_df(S_nP_chn; show_intercept=true, show_traces=false)
save("../manuscript/Figures/plots/S_nP_chn.pdf", p)


## Total effect of H on E
adjustmentSets(dag, "H", "E") # { }

@model function H_E(IDidx, Vidx, E, H; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))
  βH_ ~ Normal(0, 0.5)
  σ ~ Exponential(std(E))
  ν ~ LogNormal(2, 1)

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)   # group-level SDs intercepts
  τᵦ ~ truncated(Cauchy(0, 2); lower=0)   # group-level SDs intercepts

  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # likelihood
  Ê = @. α + α_ID[IDidx] + α_vax[Vidx] + βH_ * H * τᵦ
  return E ~ MvNormal(Ê, σ^2 * I)
end

H_E_model = H_E(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.H);

H_E_chn = sample(H_E_model, NUTS(), MCMCThreads(), 3000, 4)

H_E_chn_df = DataFrame(H_E_chn)[!, r"α\b|β"];
precis(H_E_chn_df)

# include("./TuringPlots.jl")
p = plot_chains_df(H_E_chn; show_intercept=true)
save("../manuscript/Figures/plots/H_E_chn.pdf", p)
p

## Direct effect of H on E
adjustmentSets(dag, "H", "E", effect="direct") # { D, F, M, P, R, S, V } - V is treated as random effect

@model function DE_H_E(IDidx, Vidx, E, H, D, Ḟ, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))

  βHab ~ Normal(0, 0.5)
  βDiet ~ Normal(0, 0.5)
  βFat ~ Normal(0, 0.5)
  βMass ~ Normal(0, 0.5)
  βPara ~ Normal(0, 0.5)
  βRep ~ Normal(0, 0.5)
  βSex ~ Normal(0, 0.5)

  σ ~ Exponential(std(E))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # missing F values
  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(TDist(3), N_missing)
  F_impute = convert(Array, F_impute)
  ν ~ Normal(0.5, 1) # imputed mean
  σ_F ~ Exponential() # imputed SD

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      F_impute[i_missing] ~ Normal(ν, σ_F)
      f_imputed = F_impute[i_missing]
      i_missing += 1
    else
      Ḟ[i] ~ Normal(ν, σ_F)
      f_imputed = Ḟ[i]
    end
    # likelihood
    Ê = @. α + α_ID[IDidx][i] + α_vax[Vidx][i] + βHab * H[i] + βDiet * D[i] + βFat * f_imputed + βMass * M[i] + βPara * P[i] + βRep * R[i] + βSex * S[i]
    E[i] ~ Normal(Ê, σ)
  end
end

DE_H_E_model = DE_H_E(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.H, dag_df.D, dag_df.Ḟ, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S);

DE_H_E_chn = sample(DE_H_E_model, NUTS(), MCMCThreads(), 3000, 4);
summarize(DE_H_E_chn)

DE_H_E_chn_df = DataFrame(DE_H_E_chn)[!, r"α\b|β"];
precis(DE_H_E_chn_df)

p1 = plot_chains_df(DE_H_E_chn; show_traces=true)
save("../manuscript/Figures/plots/MultiLevel_DE_H_E_chn_traces.pdf", p1)
p1

p2 = plot_chains_df(DE_H_E_chn; show_traces=false)
save("../manuscript/Figures/plots/MultiLevel_DE_H_E_chn.pdf", p2)
p2

# Mixed model
glmm_DE_H_E = fit(MixedModel, @formula(E ~ 1 + H + D + Ḟ + M + nP + R + S + (1 | V) + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_DE_H_E);
glmm_DE_H_E_coefplot = coefplot(boot)
save("../manuscript/Figures/plots/MultiLevel_DE_H_E_coefplot.pdf", glmm_DE_H_E_coefplot)
glmm_DE_H_E_ridgeplot = ridgeplot(boot)
save("../manuscript/Figures/plots/MultiLevel_DE_H_E_ridgeplot.pdf", glmm_DE_H_E_ridgeplot)

## Total effect of Diet on E
adjustmentSets(dag, "D", "E", effect="total") # {H}

@model function D_E(IDidx, Vidx, E, D, H; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))
  βD ~ Normal(0, 0.5)
  βH ~ Normal(0, 0.5)
  σ ~ Exponential(std(E))
  ν ~ LogNormal(2, 1)


  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # likelihood
  Ê = @. α + α_ID[IDidx] + α_vax[Vidx] + βD * D + βH * H
  return E ~ MvNormal(Ê, σ^2 * I)
end

D_E_model = D_E(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.D, dag_df.H);

D_E_chn = sample(D_E_model, NUTS(), MCMCThreads(), 3000, 3)

D_E_chn_df = DataFrame(D_E_chn)[!, r"β"];
precis(D_E_chn_df)

p = plot_chains_df(D_E_chn)
save("../manuscript/Figures/plots/D_E_chn.pdf", p)
p

## Direct effect of diet D on E
adjustmentSets(dag, "D", "E", effect="direct") # {F, H, M, P, R, S, V} - V is treated as random effect

@model function DE_D_E(IDidx, Vidx, E, D, Ḟ, H, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))

  βDE ~ Normal(0, 0.5)
  βFE ~ Normal(0, 0.5)
  βHE ~ Normal(0, 0.5)
  βME ~ Normal(0, 0.5)
  βPE ~ Normal(0, 0.5)
  βRE ~ Normal(0, 0.5)
  βSE ~ Normal(0, 0.5)

  σ ~ Exponential(std(E))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)      # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # missing F values
  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(Normal(), N_missing)
  F_impute = convert(Array, F_impute)
  ν ~ Normal(0.5, 1) # imputed mean
  σ_F ~ Exponential() # imputed SD

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      F_impute[i_missing] ~ Normal(ν, σ_F)
      f_imputed = F_impute[i_missing]
      i_missing += 1
    else
      Ḟ[i] ~ Normal(ν, σ_F)
      f_imputed = Ḟ[i]
    end
    # likelihood
    Ê = @. α + α_ID[IDidx][i] + α_vax[Vidx][i] + βDE * D[i] + βFE * f_imputed + βHE * H[i] + βME * M[i] + βPE * P[i] + βRE * R[i] + βSE * S[i]
    E[i] ~ Normal(Ê, σ)
  end
end

DE_D_E_model = DE_D_E(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.D, dag_df.Ḟ, dag_df.H, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S);

Turing.setadbackend(:reversediff)
DE_D_E_chn = sample(DE_D_E_model, NUTS(), MCMCThreads(), 3000, 3);

DE_D_E_chn_df = DataFrame(DE_D_E_chn)[!, r"α\b|β"];
precis(DE_D_E_chn_df)

p1 = plot_chains_df(DE_D_E_chn; show_traces=true)
save("../manuscript/Figures/plots/MultiLevel_DE_D_E_chn_traces.pdf", p1)
p1
p2 = plot_chains_df(DE_D_E_chn; res=(12, 10), show_traces=false)
save("../manuscript/Figures/plots/MultiLevel_DE_D_E_chn.pdf", p2)
p2

## Direct effet of F on E

adjustmentSets(dag, "F", "E", effect="direct") # { D, H, M, P, R, S, V }

@model function DE_F_E(IDidx, Vidx, E, Ḟ, D, H, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))

  βDF ~ Normal(0, 0.5)
  βFF ~ Normal(0, 0.5)
  βHF ~ Normal(0, 0.5)
  βMF ~ Normal(0, 0.5)
  βPF ~ Normal(0, 0.5)
  βRF ~ Normal(0, 0.5)
  βSF ~ Normal(0, 0.5)

  σ ~ Exponential(std(E))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)      # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # missing F values
  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(Normal(), N_missing)
  F_impute = convert(Array, F_impute)
  ν ~ Normal(0.5, 1) # imputed mean
  σ_F ~ Exponential() # imputed SD

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      F_impute[i_missing] ~ Normal(ν, σ_F)
      f_imputed = F_impute[i_missing]
      i_missing += 1
    else
      Ḟ[i] ~ Normal(ν, σ_F)
      f_imputed = Ḟ[i]
    end
    # likelihood
    Ê = @. α + α_ID[IDidx][i] + α_vax[Vidx][i] + βFE * f_imputed + βDF * D[i] + βHF * H[i] + βMF * M[i] + βPF * P[i] + βRF * R[i] + βSF * S[i]
    E[i] ~ Normal(Ê, σ)
  end
end

DE_F_E_model = DE_F_E(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.Ḟ, dag_df.D, dag_df.H, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S);

Turing.setadbackend(:reversediff)
DE_F_E_chn = sample(DE_F_E_model, NUTS(), MCMCThreads(), 3000, 3);

DE_F_E_chn_df = DataFrame(DE_F_E_chn)[!, r"α\b|β"];
precis(DE_F_E_chn_df)

## Direct effect of M on E
adjustmentSets(dag, "M", "E", effect="direct") # { D, F, H, P, R, S}

@model function M_E(E, M, D, Ḟ, P, R, S, H)

  α ~ Normal(mean(E), 2.5 * std(E))
  σ ~ Exponential(std(E))

  σ_F ~ Exponential()
  ν ~ Normal(0.5, 1)

  βM ~ Normal(0, 1)
  βD ~ Normal(0, 1)
  βF ~ Normal(0, 1)
  βH ~ Normal(0, 1)
  βP ~ Normal(0, 1)
  βR ~ Normal(0, 1)
  βS ~ Normal(0, 1)

  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(Normal(), N_missing)

  i_missing = 1
  for i in eachindex(Ḟ)
    if ismissing(Ḟ[i])
      F_impute[i_missing] ~ Normal(ν, σ_F)
      f_imputed = F_impute[i_missing]
      i_missing += 1
    else
      Ḟ[i] ~ Normal(ν, σ_F)
      f_imputed = Ḟ[i]
    end
    µ = @. α + βM * M[i] + βD * D[i] + βF * f_imputed + βH * H[i] + βP * P[i] + βR * R[i] + βS * S[i]
    E[i] ~ Normal(µ, σ^2 * I)
  end
end

M_E_model = M_E(dag_df.E, dag_df.M, dag_df.D, dag_df.Ḟ, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S, dag_df.H)

M_E_ch = sample(M_E_model, NUTS(), MCMCThreads(), 3000, 4)

M_E_df = DataFrame(M_E_ch)[!, r"α\b|β"];
precis(M_E_df)

coeftab_plot(M_E_df, pars=[:βW, :βD, :βM, :βF, :βP, :βR, :βS])

p = plot_chains_df(M_E_ch; show_intercept=true)
save("../manuscript/Figures/plots/M_E_chn.pdf", p)
p

## Direct effet of H on P

adjustmentSets(dag, "H", "P", effect="direct")# { D, R, S, V }

glmm_H_P = fit(MixedModel, @formula(lognP ~ 1 + H + D + R + S + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_P);
coefplot(boot)
ridgeplot(boot)

@model function H_nP_unique(nP, H, D, R, S, T)

  α ~ Normal(mean(nP), 2.5 * std(nP))
  σ ~ Exponential(std(nP))

  βH ~ Normal(0, 3)
  βD ~ Normal(0, 3)
  βR ~ Normal(0, 3)
  βS ~ Normal(0, 3)
  βT ~ Normal(0, 3)

  # likelihood
  nP̂ = α .+ βH .* H .+ βD .* D .+ βR .* R .+ βS .* S .+ βT .* T
  return nP ~ MvNormal(nP̂, σ^2 * I)
end

H_nP_model_unique = H_nP_unique(log10.(1 .+ filtered_unique_df.nP), filtered_unique_df.H, filtered_unique_df.D, filtered_unique_df.R, filtered_unique_df.S, filtered_unique_df.T)

H_nP_ch_unique = sample(H_nP_model_unique, NUTS(), MCMCThreads(), 3000, 4)

H_nP_df_unique = DataFrame(H_nP_ch_unique)[!, r"α\b|β"];

precis(H_nP_df_unique)

p = plot_chains_df(H_nP_ch_unique; show_intercept=true)
save("../manuscript/Figures/plots/H_nP_unique_chn.pdf", p)
p

## Total effect of D on P

adjustmentSets(dag, "D", "P", effect="total")   # { H }

glmm_D_P = fit(MixedModel, @formula(lognP ~ 1 + D + H + (1 | ID)), dag_df)

qqnorm(glmm_D_P; qqline=:fitrobust)
hist(residuals(glmm_D_P))

## Direct effet of D on P

adjustmentSets(dag, "D", "P", effect="direct")# { H, R, S, V }

glmm_D_P = fit(MixedModel, @formula(lognP ~ 1 + H + D + R + S + V + (1 | ID)), dag_df)

qqnorm(glmm_H_P; qqline=:fitrobust)
hist(residuals(glmm_H_P))

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_P);
coefplot(boot)
ridgeplot(boot)

@model function H_nP(nP, H, D, R, S, IDidx, Vidx; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # population-level priors
  α ~ Normal(mean(nP), 2.5 * std(nP))
  βH ~ Normal(0, 0.5)
  βD ~ Normal(0, 0.5)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)

  σ ~ Exponential(std(nP))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # likelihood
  nP̂ = @. α + α_ID[IDidx] + α_vax[Vidx] + βH * H + βD * D + βR * R + βS * S
  return nP ~ MvNormal(nP̂, σ^2 * I)
end

H_nP_model = H_nP(log10.(1 .+ dag_df.nP), dag_df.H, dag_df.D, dag_df.R, dag_df.S, dag_df.IDidx, dag_df.Vidx);

H_nP_ch = sample(H_nP_model, NUTS(), MCMCThreads(), 3000, 4);

H_nP_df = DataFrame(H_nP_ch)[!, r"α\b|β"];

precis(H_nP_df)

p = plot_chains_df(H_nP_ch; show_intercept=true)
save("../manuscript/Figures/plots/H_nP_chn.pdf", p)
p

## Direct effect of D on F

adjustmentSets(dag, "D", "F", effect="direct") # { H, P, R, S}

glmm_D_F = fit(MixedModel, @formula(Ḟ ~ 1 + D + H + P + R + S + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_F);
coefplot(boot)
ridgeplot(boot)

## Direct effect of D on M

adjustmentSets(dag, "D", "M", effect="direct") # { F, H, P, R, S, V }

glmm_D_M = fit(MixedModel, @formula(M ~ 1 + D + Ḟ + H + P + R + S + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_M);
coefplot(boot)
ridgeplot(boot)

## Direct effect of D on R

adjustmentSets(dag, "D", "R", effect="direct") # {H}

glmm_D_R = fit(MixedModel, @formula(R ~ 1 + D + H + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_R);
coefplot(boot)
ridgeplot(boot)

## Direct effect of F on M

adjustmentSets(dag, "F", "M", effect="direct") # { D, H, P, R, S, V }

glmm_F_M = fit(MixedModel, @formula(M ~ 1 + Ḟ + D + H + P + R + S + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_F_M);
coefplot(boot)
ridgeplot(boot)



## Direct effect of P on F

adjustmentSets(dag, "P", "F", effect="direct") # { D, H, R, S, V }

glmm_P_F = fit(MixedModel, @formula(Ḟ ~ 1 + P + D + H + R + S + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_P_F);
coefplot(boot)
ridgeplot(boot)



## Direct effect of P on M

adjustmentSets(dag, "P", "M", effect="direct") # { D, F, H, R, S, V }

glmm_P_M = fit(MixedModel, @formula(M ~ 1 + P + D + Ḟ + H + R + S + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_P_M);
coefplot(boot)
ridgeplot(boot)

## Direct effect of R on F

adjustmentSets(dag, "R", "F", effect="direct") # { D, H, P, S, V }

glmm_R_F = fit(MixedModel, @formula(Ḟ ~ 1 + R + D + H + P + S + V + (1 | ID)), dag_df)


boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_R_F);
coefplot(boot)

## Direct effect of R on M

adjustmentSets(dag, "R", "M", effect="direct") # { D, F, H, P, S, V }

glmm_R_M = fit(MixedModel, @formula(M ~ 1 + R + D + Ḟ + H + P + S + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_R_M);
coefplot(boot)
ridgeplot(boot)

## Direct effect of S on F

adjustmentSets(dag, "S", "F", effect="direct") # { D, H, P, R, T }

glmm_S_F = fit(MixedModel, @formula(Ḟ ~ 1 + S + D + H + P + R + T + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_F);
coefplot(boot)
ridgeplot(boot)

## Direct effect of S on M

adjustmentSets(dag, "S", "M", effect="direct") # { D, F, H, P, R, V }

glmm_S_M = fit(MixedModel, @formula(M ~ 1 + S + D + Ḟ + H + P + R + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_M);
coefplot(boot)
ridgeplot(boot)

## Direct effect of S on R

adjustmentSets(dag, "S", "R", effect="direct") # { }

glmm_S_R = fit(MixedModel, @formula(R ~ 1 + S + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_R);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on D

adjustmentSets(dag, "H", "D", effect="direct") # { }

glmm_H_D = fit(MixedModel, @formula(D ~ 1 + H + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_D);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on F

adjustmentSets(dag, "H", "F", effect="direct") # { D, P, R, S, V }

glmm_H_F = fit(MixedModel, @formula(Ḟ ~ 1 + H + D + P + R + S + V + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_F);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on M

adjustmentSets(dag, "H", "M", effect="direct") # { D, F, P, R, S, V }

glmm_H_M = fit(MixedModel, @formula(M ~ 1 + H + D + Ḟ + P + R + S + V + (1 | ID)), dag_df)


boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_M);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on R

adjustmentSets(dag, "H", "R", effect="direct") # { D }

glmm_H_R = fit(MixedModel, @formula(R ~ 1 + H + D + (1 | ID)), dag_df)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_R);
coefplot(boot)
ridgeplot(boot)
