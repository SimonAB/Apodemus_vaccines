#=
SCM Specification
- Julia version: 1.9
- Author: Simon A Babayan
- Date: 2022-08-01
=#

## Start REPL

"""
# If you run this in terminal (ctrl-I, then ctrl-\)
julia  --threads auto --project=.
"""

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
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

using RCall
@rlibrary dagitty # we use the original dagitty from R until julia native version improves

# plotting & diagnostics
using CairoMakie
CairoMakie.activate!(type="svg")
using MixedModelsMakie
using Formatting

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
dag_df = df[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :T, :P, :nP, :ID, :IDidx, :vax_history, :Vidx]]
dag_df.lognP = log10.(1 .+ dag_df.nP);
# describe(dag_df)

dag_df_unique = df_unique[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :T, :P, :nP, :Vidx, :vax_history, :ID]];
filtered_df = dag_df[dag_df.P.==1, :];
filtered_unique_df = dag_df_unique[dag_df_unique.P.==1, :];


# DAG specification - this is our graphical causal hypothesis

dag = dagitty("dag{
D -> E;
D -> F;
D -> M;
D -> P;
D -> R;
F -> E;
F -> M;
H -> E;
H -> F;
H -> M;
H -> P;
H -> R;
M -> E;
P -> E;
P -> F;
P -> M;
R -> E;
R -> F;
R -> M;
R -> P;
S -> E;
S -> F;
S -> M;
S -> P;
S -> R;
V -> E;
V -> F;
V -> M;
V -> P;
V -> R;
}")

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

"""
┌───────┬──────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%     histogram │
├───────┼──────────────────────────────────────────────────────────┤
│     α │ -2.7198  0.2365  -3.0992  -2.7175  -2.3477    ▁▁▁▃▇█▆▂▁▁ │
│   βVE │  1.5018  0.1268   1.3014   1.5004   1.7072  ▁▁▁▂▅██▅▂▁▁▁ │
└───────┴──────────────────────────────────────────────────────────┘
"""

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

Turing.setadbackend(:forwarddiff)
V_E_DE_chn = sample(V_E_DE_model, NUTS(), MCMCThreads(), 3000, 4)

V_E_DE_chn_df = DataFrame(V_E_DE_chn)[!, r"α\b|β"];
precis(V_E_DE_chn_df)

# Mixed model
glmm_V_E_NDE = fit(MixedModel, @formula(E ~ 1 + V + D + Ḟ + H + M + P + R + S + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + V + D + Ḟ + H + M + P + R + S + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -208.3876   416.7752   438.7752   440.0323   476.2046

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.046424 0.215463
Residual              0.340310 0.583361
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  -2.07923     0.361006   -5.76    <1e-08
V             1.49444     0.111327   13.42    <1e-40
D            -0.263531    0.0964268  -2.73    0.0063
Ḟ             0.0136384   0.0731522   0.19    0.8521
H            -0.627028    0.18903    -3.32    0.0009
M            -0.111196    0.0670038  -1.66    0.0970
P             0.180073    0.146183    1.23    0.2180
R             0.0257785   0.170083    0.15    0.8795
S             0.163506    0.109328    1.50    0.1348
────────────────────────────────────────────────────


"""
qqnorm(glmm_V_E_NDE; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_NDE);
coefplot(boot)
ridgeplot(boot)


## Total effect of `nP` on `E`
adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V}

# selected only infected mice
dag_df_infected = dag_df[dag_df.P.==2, :] # select rows of dag_df for which dag_df.P.==2

# Plot
layers = linear() + visual(Scatter)
plt = data(dag_df_infected) * mapping(:lognP, :E)
p = draw(layers * plt, axis=(xlabel="H. polygyrus worm count (log10(1 + nP))", ylabel="α-DT IgG1 (standardised)"))
save("../manuscript/Figures/plots/P_E_cor.pdf", p)

# Naive model (does not use adjustment set)
# point estimate sanity check
naive_glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + (1 | ID)), dag_df_infected; progress=false)
"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + lognP + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
   -60.6290   121.2581   129.2581   130.1276   136.9854

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.096246 0.310236
Residual              0.549422 0.741230
 Number of obs: 51; levels of grouping factors: 19

  Fixed-effects parameters:
───────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)
───────────────────────────────────────────────────
(Intercept)   0.559782    0.384286   1.46    0.1452
lognP        -0.660584    0.285738  -2.31    0.0208
───────────────────────────────────────────────────

"""

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
"""
┌───────┬───────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%      histogram │
├───────┼───────────────────────────────────────────────────────────┤
│     α │  0.0796  0.0823  -0.0517   0.0799   0.2103  ▁▁▁▂▄▇██▅▂▁▁▁ │
│   βPE │ -0.2949  0.1425  -0.5233  -0.2964  -0.0649   ▁▁▂▅██▆▂▁▁▁▁ │
└───────┴───────────────────────────────────────────────────────────┘
"""

p = plot_chains_df(naive_P_E_chn; show_intercept=true, show_traces=false)
save("../manuscript/Figures/plots/naive_P_E_chn_df.pdf", p)

## Properly-adjusted model: total effect of `nP` on `E`
adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V }

# Mixed Model - excluding H since length(unique(dag_df_infected.H))=1
glmm_P_E_all = fit(MixedModel, @formula(E ~ 1 + lognP + D + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + lognP + D + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -300.1878   600.3755   616.3755   616.8898   645.7070

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.061167 0.247321
Residual              0.413615 0.643129
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
───────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)
───────────────────────────────────────────────────
(Intercept)  -2.31676    0.327291   -7.08    <1e-11
lognP        -0.13274    0.113216   -1.17    0.2410
D            -0.284799   0.0898423  -3.17    0.0015
R            -0.508572   0.13659    -3.72    0.0002
S             0.320681   0.0921007   3.48    0.0005
V             1.61709    0.116514   13.88    <1e-43
───────────────────────────────────────────────────

"""

qqnorm(glmm_P_E_all; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E_all);
coefplot(glmm_P_E_all)
coefplot(boot)
ridgeplot(boot)

# Infected only
glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + R + S + V + (1 | ID)), dag_df_infected)
"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + lognP + D + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
   -54.1721   108.3441   124.3441   127.7727   139.7987

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.019575 0.139909
Residual              0.471247 0.686474
 Number of obs: 51; levels of grouping factors: 19

  Fixed-effects parameters:
───────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)
───────────────────────────────────────────────────
(Intercept)  -1.61109     0.960908  -1.68    0.0936
lognP        -0.52165     0.315781  -1.65    0.0985
D            -0.151094    0.24211   -0.62    0.5326
R            -0.116407    0.324117  -0.36    0.7195
S             0.157128    0.240046   0.65    0.5127
V             1.16835     0.338118   3.46    0.0005
───────────────────────────────────────────────────


"""

qqnorm(glmm_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E);
coefplot(boot)
ridgeplot(boot)
ridge2d(boot)


# Direct effect of P on E among the infected
adjustmentSets(dag, "P", "E", effect="direct") # { D, F, H, M, R, S, V}

# Mixed model

glmm_DE_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + Ḟ + M + R + S + V + (1 | ID)), dag_df_infected)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + lognP + D + Ḟ + M + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
   -53.9266   107.8533   127.8533   133.3533   147.1715

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.016659 0.129068
Residual              0.469236 0.685008
 Number of obs: 51; levels of grouping factors: 19

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  -1.49         1.07116   -1.39    0.1642
lognP        -0.584714     0.35534   -1.65    0.0999
D            -0.126434     0.249144  -0.51    0.6118
Ḟ             0.0990188    0.154197   0.64    0.5208
M             0.0377835    0.226099   0.17    0.8673
R            -0.135389     0.386997  -0.35    0.7265
S             0.177533     0.249498   0.71    0.4767
V             1.17781      0.335573   3.51    0.0004
────────────────────────────────────────────────────
"""

qqnorm(glmm_DE_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_DE_P_E);
coefplot(boot)
ridgeplot(boot)


# Population-level model for the log of the expected number
@model function P_E(IDidx, E, P, D, R, S, T; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))
  βPE ~ Normal(0, 0.5)
  βDE ~ Normal(0, 0.5)
  βRE ~ Normal(0, 0.5)
  βSE ~ Normal(0, 0.5)
  βTE ~ Normal(0, 0.5)
  σ ~ Exponential(std(E))
  ν ~ LogNormal(2, 1)

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts

  # likelihood
  Ê = α .+ α_ID[IDidx] .+ βPE * P .+ βDE * D .+ βRE * R .+ βSE * S .+ βTE * T
  return E ~ MvNormal(Ê, σ^2 * I)
end

P_E_model = P_E(dag_df_infected.IDidx, dag_df_infected.E, log10.(1 .+ dag_df_infected.nP), dag_df_infected.D, dag_df_infected.R, dag_df_infected.S, dag_df_infected.T); # note log10(1+X)-transformed parasite counts.

P_E_priors = sample(P_E_model, Prior(), MCMCThreads(), 3000, 4);
summarize(P_E_priors)

P_E_priors_df = DataFrame(P_E_priors)[!, r"α\b|β"];
precis(P_E_priors_df)
"""
┌───────┬───────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%   94.5%   histogram │
├───────┼───────────────────────────────────────────────────────┤
│     α │ -0.0083  2.4806  -3.9559  -0.0138  3.9856  ▁▁▂▅██▅▂▁▁ │
│   βDE │ -0.0053  0.4986  -0.7983  -0.0052   0.784  ▁▁▁▄██▄▁▁▁ │
│   βPE │  0.0038  0.5024  -0.7917   0.0064  0.8134   ▁▁▄██▄▁▁▁ │
│   βRE │  0.0029  0.5008  -0.8006   0.0001  0.8195   ▁▁▁▄██▄▁▁ │
│   βSE │ -0.0154  0.4994  -0.8142  -0.0166  0.7811  ▁▁▄██▃▁▁▁▁ │
│   βTE │ -0.0011  0.4985    -0.79   0.0014  0.7969    ▁▁▄██▄▁▁ │
└───────┴───────────────────────────────────────────────────────┘

"""
p = plot_chains_df(P_E_priors; show_intercept=true)
save("../manuscript/Figures/plots/P_E_priors.pdf", p)
p

# updating with data
P_E_chn = sample(P_E_model, NUTS(), MCMCThreads(), 3000, 4);

P_E_chn_df = DataFrame(P_E_chn)[!, r"α\b|β"];
precis(P_E_chn_df)

"""
┌───────┬────────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%       histogram │
├───────┼────────────────────────────────────────────────────────────┤
│     α │  0.3864  0.3244  -0.1341   0.3885   0.9046  ▁▁▁▁▃▆██▆▃▁▁▁▁ │
│   βDE │ -0.2805  0.1133    -0.46  -0.2804  -0.0987     ▁▁▁▃██▅▂▁▁▁ │
│   βHE │ -0.5419  0.1713  -0.8159  -0.5425  -0.2652        ▁▁▂▆█▄▁▁ │
│   βPE │ -0.0102  0.1453   -0.243  -0.0108   0.2248    ▁▁▁▃▆██▅▂▁▁▁ │
│   βRE │  -0.269  0.1788   -0.553  -0.2698   0.0197        ▁▁▁▄█▆▂▁ │
│   βSE │  0.1879  0.1175  -0.0016   0.1899   0.3749     ▁▁▁▂▅██▄▁▁▁ │
│   βTE │    0.04  0.0042   0.0333     0.04   0.0467        ▁▁▃██▃▁▁ │
└───────┴────────────────────────────────────────────────────────────┘

"""

# PE_coeftab = coeftab_plot(P_E_chn_df; legend=false, xgrid=false, size=(600, 200))
# savefig(PE_coeftab, "../plots/PE_coeftab.pdf")

p = plot_chains_df(P_E_chn)
save("../manuscript/Figures/plots/P_E_chn_df.pdf", p)

## Direct effect of `P` on `E`

# Minimal sufficient adjustment set for estimating the direct effect of P on E:
adjustmentSets(dag, "P", "E", effect="direct") # { D, F, H, M, R, S, V }

# GLMM

glmm_DE_P_E = fit(MixedModel, @formula(E ~ 1 + P + D + Ḟ + H + M + R + S + V + (1 | ID)), dag_df)
"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + nP + D + Ḟ + H + M + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -208.8643   417.7286   439.7286   440.9857   477.1580

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.045342 0.212936
Residual              0.342847 0.585531
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)  -2.0121      0.354653    -5.67    <1e-07
nP           -0.00144412  0.00192667  -0.75    0.4535
D            -0.272597    0.0960341   -2.84    0.0045
Ḟ             0.00860208  0.0733332    0.12    0.9066
H            -0.544374    0.178197    -3.05    0.0023
M            -0.0992379   0.066968    -1.48    0.1384
R             0.0707625   0.173147     0.41    0.6828
S             0.148807    0.109957     1.35    0.1760
V             1.50833     0.111397    13.54    <1e-41
─────────────────────────────────────────────────────

"""

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

"""
┌───────┬───────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%      histogram │
├───────┼───────────────────────────────────────────────────────────┤
│     α │  0.5287   0.355  -0.0405   0.5303   1.0902         ▁▂██▂▁ │
│   βPE │ -0.0048  0.1452  -0.2404  -0.0051   0.2252  ▁▁▁▁▃▆██▅▂▁▁▁ │
│   βDE │ -0.2756  0.1192  -0.4672  -0.2756  -0.0864    ▁▁▁▃▇█▅▂▁▁▁ │
│   βFE │ -0.0601  0.0684  -0.1687  -0.0619   0.0522     ▁▁▃▆█▇▄▂▁▁ │
│   βHE │ -0.6275  0.2001  -0.9478   -0.625  -0.3083     ▁▁▁▄██▃▁▁▁ │
│   βME │  -0.008  0.0759  -0.1278  -0.0086   0.1151  ▁▁▁▃▆██▅▂▁▁▁▁ │
│   βRE │ -0.2614  0.1856  -0.5579  -0.2613   0.0372       ▁▁▄█▆▂▁▁ │
│   βSE │  0.1681  0.1371  -0.0479   0.1666   0.3865  ▁▁▁▁▃▆█▇▄▂▁▁▁ │
│   βVE │  0.0389  0.0046   0.0315   0.0389   0.0461       ▁▁▄█▇▂▁▁ │
└───────┴───────────────────────────────────────────────────────────┘

"""

p1 = plot_chains_df(DE_P_E_chn; show_traces=true)
save("../manuscript/Figures/plots/MultiLevel_DE_P_E_chn_traces.pdf", p1)

p2 = plot_chains_df(DE_P_E_chn; show_traces=false, xlab_dist="Direct Casual Effect")
save("../manuscript/Figures/plots/MultiLevel_DE_P_E_chn.pdf", p2)

## Direct effect of reproductive status R on parasite burden P

adjustmentSets(dag, "R", "P", effect="direct") # D, H, S, V

# GLMM

glmm_R_nP = fit(MixedModel, @formula(lognP ~ R + D + H + S + V + (1 | ID)), dag_df)
"""
Linear mixed model fit by maximum likelihood
 lognP ~ 1 + R + D + H + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
    75.7468  -151.4936  -135.4936  -134.9793  -106.1621

Variance components:
            Column    Variance  Std.Dev.
ID       (Intercept)  0.1634105 0.4042407
Residual              0.0077654 0.0881214
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)  -0.492284     0.19141    -2.57    0.0101
R             0.0445632    0.039308    1.13    0.2569
D            -0.00415668   0.0433778  -0.10    0.9237
H             0.539661     0.0851757   6.34    <1e-09
S            -0.0713095    0.0786907  -0.91    0.3648
V             0.0113116    0.023732    0.48    0.6336
─────────────────────────────────────────────────────

"""

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

"""
┌───────┬──────────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %      histogram │
├───────┼──────────────────────────────────────────────────────┤
│     α │  -0.48  0.195  -0.796  -0.482  -0.154      ▁▁▁▅█▆▂▁▁ │
│    βR │  0.046   0.04  -0.019   0.045   0.113        ▁▁▃█▇▂▁ │
│    βD │ -0.005  0.045  -0.079  -0.005   0.068      ▁▁▁▃██▂▁▁ │
│    βH │  0.525  0.087   0.377   0.525   0.666       ▁▁▂▆█▄▁▁ │
│    βS │ -0.067  0.077  -0.192  -0.067   0.058  ▁▁▁▂▃▇█▇▅▂▁▁▁ │
│    βV │  0.011  0.024  -0.029   0.011   0.051    ▁▁▁▃▆█▆▃▁▁▁ │
└───────┴──────────────────────────────────────────────────────┘


"""

p = plot_chains_df(R_nP_chn; show_intercept=true)
save("../manuscript/Figures/plots/R_nP_chn.pdf", p)

## Total effect of D on nP

adjustmentSets(dag, "D", "P", effect="total") # {}

@model function D_nP(nP, D, IDidx; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(0, 2.5)
  βD ~ Normal(0, 0.5)
  σ ~ Exponential(std(nP))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)

  # likelihood
  nP̂ = @. α + α_ID[IDidx] + βD * D
  nP ~ MvNormal(nP̂, σ^2 * I)
end

D_nP_model = D_nP(log10.(1 .+ dag_df.nP), dag_df.D, dag_df.IDidx)
D_nP_chn = sample(D_nP_model, NUTS(), MCMCThreads(), 3_000, 4);
D_nP_chn_df = DataFrame(D_nP_chn)[!, r"α\b|β"];
precis(D_nP_chn_df)
"""
┌───────┬────────────────────────────────────────────────────────┐
│ param │   mean     std     5.5%     50%   94.5%      histogram │
├───────┼────────────────────────────────────────────────────────┤
│     α │ 0.2076  0.0841   0.0741  0.2073  0.3425  ▁▁▁▃▆██▆▄▂▁▁▁ │
│    βD │ 0.0026   0.045  -0.0688   0.002  0.0741       ▁▁▃██▄▁▁ │
└───────┴────────────────────────────────────────────────────────┘

"""

# Plot
df = (; x=Bool.(dag_df.D[dag_df.P.==2] .- 1), y=dag_df.nP[dag_df.P.==2])
layers = visual(BoxPlot)
plt = data(df) * mapping(:x, :y, color=:x)
p = draw(layers * plt, axis=(xlabel="Diet supplemented", ylabel="H. polygyrus (count)"))
save("../manuscript/Figures/plots/D_P_cor.pdf", p)


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

"""
┌───────┬──────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%   94.5%      histogram │
├───────┼──────────────────────────────────────────────────────────┤
│     α │ -0.4628   0.182  -0.7462  -0.4668   -0.17       ▁▁▄█▆▂▁▁ │
│    βD │ -0.0025  0.0435  -0.0713  -0.0023  0.0658       ▁▁▃██▃▁▁ │
│    βH │  0.5258  0.0869   0.3896   0.5243  0.6673        ▁▂▆█▄▁▁ │
│    βR │   0.043  0.0403  -0.0209   0.0426  0.1074       ▁▁▃█▇▂▁▁ │
│    βS │ -0.0753  0.0785  -0.2015  -0.0734  0.0477  ▁▁▁▂▄▇█▇▄▂▁▁▁ │
│    βT │  0.0008  0.0007  -0.0003   0.0008  0.0018    ▁▁▁▃▆█▇▃▁▁▁ │
└───────┴──────────────────────────────────────────────────────────┘

"""

p = plot_chains_df(D_nP_chn; show_intercept=true, show_traces=false)
save("../manuscript/Figures/plots/D_nP_chn.pdf", p)

## Total effect of S on E

adjustmentSets(dag, "S", "E", effect="total") # { }

glmm_S_E = fit(MixedModel, @formula(E ~ 1 + S + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + S + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -376.2616   752.5232   760.5232   760.6640   775.1889

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.387887 0.622806
Residual              0.535476 0.731762
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
───────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)
───────────────────────────────────────────────────
(Intercept)  -0.414166    0.23313   -1.78    0.0756
S             0.292378    0.150432   1.94    0.0519
───────────────────────────────────────────────────

"""

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

"""
┌───────┬──────────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %      histogram │
├───────┼──────────────────────────────────────────────────────┤
│     α │ -0.377  0.227  -0.751  -0.377  -0.005     ▁▁▁▄██▅▁▁▁ │
│   βS_ │  0.267  0.145   0.029   0.267   0.503  ▁▁▁▁▃▆█▇▄▂▁▁▁ │
└───────┴──────────────────────────────────────────────────────┘

"""

## Direct effect of S on E

adjustmentSets(dag, "S", "E", effect="direct") # { D, F, H, M, P, R, V }

glmm_D_S_E = fit(MixedModel, @formula(E ~ 1 + S + D + Ḟ + H + M + nP + R + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + S + D + Ḟ + H + M + nP + R + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -208.8643   417.7286   439.7286   440.9857   477.1580

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.045342 0.212936
Residual              0.342847 0.585531
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)  -2.0121      0.354653    -5.67    <1e-07
S             0.148807    0.109957     1.35    0.1760
D            -0.272597    0.0960341   -2.84    0.0045
Ḟ             0.00860208  0.0733332    0.12    0.9066
H            -0.544374    0.178197    -3.05    0.0023
M            -0.0992379   0.066968    -1.48    0.1384
nP           -0.00144412  0.00192667  -0.75    0.4535
R             0.0707625   0.173147     0.41    0.6828
V             1.50833     0.111397    13.54    <1e-41
─────────────────────────────────────────────────────

"""

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

"""
┌───────┬───────────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %       histogram │
├───────┼───────────────────────────────────────────────────────┤
│     α │  -0.47  0.203   -0.81   -0.47  -0.141       ▁▁▂▅█▆▂▁▁ │
│    βS │ -0.072   0.08  -0.209  -0.071   0.057   ▁▁▁▂▄▇██▄▂▁▁▁ │
│    βD │ -0.003  0.044  -0.077  -0.003   0.067         ▁▁▃██▃▁ │
│    βH │  0.522  0.087   0.378   0.523   0.665  ▁▁▁▁▃▅▇█▇▅▂▁▁▁ │
│    βR │  0.046   0.04  -0.019   0.045   0.112        ▁▁▃██▂▁▁ │
│    βV │  0.011  0.024  -0.028   0.011   0.052      ▁▁▃▆█▆▃▁▁▁ │
└───────┴───────────────────────────────────────────────────────┘

"""

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
  # E ~ MvNormal(Ê, σ^2 * I)
  return E ~ MvNormal(Ê, σ^2 * I)
end

H_E_model = H_E(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.H);

H_E_chn = sample(H_E_model, NUTS(), MCMCThreads(), 3000, 4)

H_E_chn_df = DataFrame(H_E_chn)[!, r"α\b|β"];
precis(H_E_chn_df)

"""
┌───────┬────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%   histogram │
├───────┼────────────────────────────────────────────────────────┤
│     α │  0.9162  0.2276   0.5508   0.9156   1.2718  ▁▁▂▆█▇▃▁▁▁ │
│   βH_ │ -0.4629  0.2431  -0.9265  -0.4067  -0.1809   ▁▁▁▁▂▃▆█▂ │
└───────┴────────────────────────────────────────────────────────┘

"""
# include("./TuringPlots.jl")
p = plot_chains_df(H_E_chn; show_intercept=true)
save("../manuscript/Figures/plots/H_E_chn.pdf", p)
p

## Direct effect of H on E
adjustmentSets(dag, "H", "E", effect="direct") # { D, F, M, P, R, S, V } - V is treated as random effect

@model function DE_H_E(IDidx, Vidx, E, H, D, Ḟ, M, P, R, S; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))

  βHE ~ Normal(0, 0.5)
  βDE ~ Normal(0, 0.5)
  βFE ~ Normal(0, 0.5)
  βME ~ Normal(0, 0.5)
  βPE ~ Normal(0, 0.5)
  βRE ~ Normal(0, 0.5)
  βSE ~ Normal(0, 0.5)

  σ ~ Exponential(std(E))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # missing F values
  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(TDist(3), N_missing)
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
    Ê = @. α + α_ID[IDidx][i] + α_vax[Vidx][i] + βHE * H[i] + βDE * D[i] + βFE * f_imputed + βME * M[i] + βPE * P[i] + βRE * R[i] + βSE * S[i]
    E[i] ~ Normal(Ê, σ)
  end
end

DE_H_E_model = DE_H_E(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.H, dag_df.D, dag_df.Ḟ, dag_df.M, dag_df.nP, dag_df.R, dag_df.S);

Turing.setadbackend(:forwarddiff)
DE_H_E_chn = sample(DE_H_E_model, NUTS(), MCMCThreads(), 3000, 4);
Turing.setadbackend(:reverse_diff)
summarize(DE_H_E_chn)

DE_H_E_chn_df = DataFrame(DE_H_E_chn)[!, r"α\b|β"];
precis(DE_H_E_chn_df)

"""
┌───────┬────────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%       histogram │
├───────┼────────────────────────────────────────────────────────────┤
│     α │  0.7413  0.3251   0.2239   0.7385    1.271   ▁▁▁▂▄▇█▇▅▂▁▁▁ │
│   βDE │ -0.2173  0.0889  -0.3617  -0.2176  -0.0762  ▁▁▁▁▂▄▇█▇▅▃▁▁▁ │
│   βFE │ -0.0315   0.051  -0.1132  -0.0316   0.0501       ▁▁▂▆█▅▂▁▁ │
│   βHE │ -0.6442  0.1656  -0.9096  -0.6437  -0.3809   ▁▁▁▂▄▇█▇▅▂▁▁▁ │
│   βME │ -0.0831  0.0564  -0.1737  -0.0825   0.0066       ▁▁▃▇█▅▂▁▁ │
│   βPE │ -0.0475  0.0494  -0.1259  -0.0475    0.032        ▁▁▃██▄▁▁ │
│   βRE │  -0.081  0.1505  -0.3252    -0.08   0.1602    ▁▁▁▂▅██▆▃▁▁▁ │
│   βSE │  0.1462  0.1043  -0.0203   0.1465   0.3145       ▁▁▂▆█▆▂▁▁ │
│   βTE │   0.017  0.0046   0.0093   0.0171   0.0242        ▁▂▆█▅▁▁▁ │
└───────┴────────────────────────────────────────────────────────────┘

"""
p1 = plot_chains_df(DE_P_E_chn; show_traces=true)
save("../manuscript/Figures/plots/MultiLevel_DE_P_E_chn_traces.pdf", p1)
p1

p2 = plot_chains_df(DE_P_E_chn; show_traces=false)
save("../manuscript/Figures/plots/MultiLevel_DE_P_E_chn.pdf", p2)
p2

# Mixed model
glmm_H_E = fit(MixedModel, @formula(E ~ 1 + H + D + Ḟ + M + nP + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + H + D + Ḟ + M + nP + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -208.8643   417.7286   439.7286   440.9857   477.1580

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.045342 0.212936
Residual              0.342847 0.585531
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)  -2.0121      0.354653    -5.67    <1e-07
H            -0.544374    0.178197    -3.05    0.0023
D            -0.272597    0.0960341   -2.84    0.0045
Ḟ             0.00860208  0.0733332    0.12    0.9066
M            -0.0992379   0.066968    -1.48    0.1384
nP           -0.00144412  0.00192667  -0.75    0.4535
R             0.0707625   0.173147     0.41    0.6828
S             0.148807    0.109957     1.35    0.1760
V             1.50833     0.111397    13.54    <1e-41
─────────────────────────────────────────────────────


"""

# Total effect of Diet on E
adjustmentSets(dag, "D", "E") # {}

@model function D_E(IDidx, Vidx, E, D; n_id=length(unique(IDidx)), n_vax=length(unique(Vidx)))
  # population-level priors
  α ~ Normal(mean(E), 2.5 * std(E))
  βD_ ~ Normal(0, 0.5)
  σ ~ Exponential(std(E))
  ν ~ LogNormal(2, 1)


  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # likelihood
  Ê = @. α + α_ID[IDidx] + α_vax[Vidx] + βD_ * D
  return E ~ MvNormal(Ê, σ^2 * I)
end

D_E_model = D_E(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.D);

D_E_chn = sample(D_E_model, NUTS(), MCMCThreads(), 3000, 3)

D_E_chn_df = DataFrame(D_E_chn)[!, r"β"];
precis(D_E_chn_df)

"""
┌───────┬───────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%  histogram │
├───────┼───────────────────────────────────────────────────────┤
│   βD_ │ -0.2573  0.1011  -0.4188  -0.2569  -0.0971  ▁▁▂▆█▅▂▁▁ │
└───────┴───────────────────────────────────────────────────────┘

"""

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
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
  α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

  # missing F values
  N_missing = sum(ismissing.(Ḟ))
  F_impute ~ filldist(Normal(), N_missing)
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


"""
without Vidx (this is the correct adjustment set):
┌───────┬───────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%      histogram │
├───────┼───────────────────────────────────────────────────────────┤
│     α │  0.6127  0.3782    0.016    0.611   1.2197        ▁▁▆█▃▁▁ │
│   βDE │ -0.2841  0.1228   -0.482  -0.2831  -0.0905    ▁▁▁▄██▅▂▁▁▁ │
│   βFE │ -0.0728  0.0689  -0.1822  -0.0735   0.0393  ▁▁▁▃▇█▆▃▁▁▁▁▁ │
│   βHE │ -0.6875  0.2195  -1.0278  -0.6904  -0.3299      ▁▁▂▆█▆▂▁▁ │
│   βME │ -0.0099  0.0791  -0.1372  -0.0104    0.118   ▁▁▂▃▆██▅▂▁▁▁ │
│   βPE │   0.007  0.0665  -0.1005    0.008   0.1132    ▁▁▂▄▇█▅▂▁▁▁ │
│   βRE │ -0.2553  0.2023  -0.5835   -0.254   0.0654      ▁▁▁▄█▇▂▁▁ │
│   βSE │   0.169  0.1456  -0.0602   0.1668   0.4037  ▁▁▁▁▃▆█▇▄▂▁▁▁ │
│   βTE │  0.0386  0.0047   0.0312   0.0385   0.0461      ▁▁▁▄█▆▂▁▁ │
└───────┴───────────────────────────────────────────────────────────┘
with Vidx:
┌───────┬────────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%       histogram │
├───────┼────────────────────────────────────────────────────────────┤
│   βDE │ -0.2164  0.0893    -0.36  -0.2169  -0.0748  ▁▁▁▂▄▇██▅▃▁▁▁▁ │
│   βFE │ -0.0317  0.0511  -0.1127  -0.0319     0.05       ▁▁▂▆█▅▂▁▁ │
│   βHE │ -0.6443  0.1643  -0.9053  -0.6447  -0.3826        ▁▁▃█▇▂▁▁ │
│   βME │ -0.0836  0.0569  -0.1747  -0.0836   0.0066      ▁▁▁▃▇█▅▂▁▁ │
│   βPE │ -0.0472  0.0482  -0.1246  -0.0467    0.029       ▁▁▃██▄▁▁▁ │
│   βRE │ -0.0802  0.1502  -0.3171  -0.0817   0.1587    ▁▁▂▅██▇▃▁▁▁▁ │
│   βSE │  0.1462  0.1059  -0.0216   0.1454   0.3146       ▁▁▂▆█▆▂▁▁ │
│   βTE │   0.017  0.0046   0.0096    0.017   0.0244       ▁▁▂▆█▅▁▁▁ │
└───────┴────────────────────────────────────────────────────────────┘

"""
p1 = plot_chains_df(DE_D_E_chn; show_traces=true)
save("../manuscript/Figures/plots/MultiLevel_DE_D_E_chn_traces.pdf", p1)
p1
p2 = plot_chains_df(DE_D_E_chn; res=(12, 10), show_traces=false)
save("../manuscript/Figures/plots/MultiLevel_DE_D_E_chn.pdf", p2)
p2

## Direct effet of F on E

adjustmentSets(dag, "F", "E", effect="direct") # { D, H, M, P, R, S, T }


@model function F_E(E, Ḟ, D, P, R, S, T, H, M)

  α ~ Normal(mean(E), 2.5 * std(E))
  σ ~ Exponential(std(E))

  σ_F ~ Exponential()
  ν ~ Normal(0.5, 1)

  βF ~ Normal(0, 1)
  βD ~ Normal(0, 1)
  βP ~ Normal(0, 1)
  βR ~ Normal(0, 1) # good to include for precision even if blocked by collider F->M<-R
  βS ~ Normal(0, 1)
  βT ~ Normal(0, 1)
  βH ~ Normal(0, 1)
  βM ~ Normal(0, 1)

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
    µ = @. α + βF * f_imputed + βD * D[i] + βP * P[i] + βR * R[i] + βS * S[i] + βT * T[i] + βH * H[i] + βM * M[i]
    E[i] ~ Normal(µ, σ^2 * I)
  end
end

# F_E_ch = sample(F_E(complete_df.E, complete_df.F, complete_df.D, complete_df.R, complete_df.S, complete_df.T, complete_df.H), NUTS(), MCMCThreads(), 1000, 4) # try without missing values
F_E_ch = sample(F_E(dag_df.E, dag_df.Ḟ, dag_df.D, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S, dag_df.T, dag_df.H, dag_df.M), NUTS(), MCMCThreads(), 3000, 4)


F_E_df = DataFrame(F_E_ch)[!, r"α\b|β"];
precis(F_E_df)

"""
┌───────┬───────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%      histogram │
├───────┼───────────────────────────────────────────────────────────┤
│     α │  0.4576  0.3105  -0.0319    0.456   0.9593  ▁▁▁▂▅██▆▄▂▁▁▁ │
│    βD │ -0.2549  0.0972  -0.4103  -0.2553   -0.099      ▁▁▂▆█▅▂▁▁ │
│    βF │ -0.0102  0.0686  -0.1187  -0.0102   0.1001   ▁▁▁▃▆█▇▄▂▁▁▁ │
│    βH │ -0.6785  0.1911   -0.983  -0.6779  -0.3715      ▁▁▁▅█▆▂▁▁ │
│    βM │ -0.0901  0.0651  -0.1936  -0.0904   0.0141    ▁▁▂▄██▅▂▁▁▁ │
│    βP │ -0.0295  0.0526  -0.1136  -0.0293   0.0548      ▁▁▂▆█▅▂▁▁ │
│    βR │ -0.1288  0.2013  -0.4531  -0.1298   0.1935       ▁▁▂▆█▅▂▁ │
│    βS │  0.0646  0.1127  -0.1144   0.0648   0.2433      ▁▁▂▅█▇▃▁▁ │
│    βT │  0.0459  0.0049   0.0381   0.0459   0.0537      ▁▁▃▇█▄▁▁▁ │
└───────┴───────────────────────────────────────────────────────────┘
"""

# coeftab_plot(F_E_df, pars=[:βF, :βD, :βP, :βR, :βS, :βT, :βW])

p = plot_chains_df(F_E_ch; show_intercept=true)
save("../manuscript/Figures/plots/F_E_chn.pdf", p)
p

## Direct effect of M on E
adjustmentSets(dag, "M", "E", effect="direct") # { D, F, H, P, R, S, T }

@model function M_E(E, M, D, Ḟ, P, R, S, T, H)

  α ~ Normal(mean(E), 2.5 * std(E))
  σ ~ Exponential(std(E))

  σ_F ~ Exponential()
  ν ~ Normal(0.5, 1)

  βM ~ Normal(0, 1)
  βD ~ Normal(0, 1)
  βF ~ Normal(0, 1)
  βH ~ Normal(0, 1)
  βP ~ Normal(0, 1)
  βR ~ Normal(0, 1) # good to include for precision even if blocked by collider F->M<-R
  βS ~ Normal(0, 1)
  βT ~ Normal(0, 1)

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
    µ = @. α + βM * M[i] + βD * D[i] + βF * f_imputed + βH * H[i] + βP * P[i] + βR * R[i] + βS * S[i] + βT * T[i]
    E[i] ~ Normal(µ, σ^2 * I)
  end
end

M_E_model = M_E(dag_df.E, dag_df.M, dag_df.D, dag_df.Ḟ, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S, dag_df.T, dag_df.H)

M_E_ch = sample(M_E_model, NUTS(), MCMCThreads(), 3000, 4)

M_E_df = DataFrame(M_E_ch)[!, r"α\b|β"];
precis(M_E_df)

"""
┌───────┬───────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%      histogram │
├───────┼───────────────────────────────────────────────────────────┤
│     α │  0.4553  0.3072  -0.0304   0.4536    0.951  ▁▁▁▂▅██▆▃▁▁▁▁ │
│    βD │ -0.2529  0.0974  -0.4086  -0.2518  -0.0977      ▁▁▂▆█▅▂▁▁ │
│    βF │ -0.0101  0.0683  -0.1176  -0.0105   0.0992   ▁▁▁▃▆██▄▂▁▁▁ │
│    βH │ -0.6815  0.1909  -0.9884  -0.6824   -0.378      ▁▁▁▅█▆▂▁▁ │
│    βM │  -0.092  0.0652  -0.1954  -0.0923   0.0123   ▁▁▁▂▄██▆▂▁▁▁ │
│    βP │ -0.0297  0.0527  -0.1146  -0.0296   0.0536      ▁▁▂▆█▅▂▁▁ │
│    βR │ -0.1242  0.1992   -0.443  -0.1255   0.1922      ▁▁▂▆█▅▁▁▁ │
│    βS │  0.0623  0.1124  -0.1174   0.0625    0.241     ▁▁▂▆█▇▃▁▁▁ │
│    βT │  0.0459  0.0049   0.0383   0.0459   0.0538      ▁▁▃▇█▄▁▁▁ │
└───────┴───────────────────────────────────────────────────────────┘
"""

coeftab_plot(M_E_df, pars=[:βW, :βD, :βM, :βF, :βP, :βR, :βS])

p = plot_chains_df(M_E_ch; show_intercept=true)
save("../manuscript/Figures/plots/M_E_chn.pdf", p)
p

## Direct effet of H on P

adjustmentSets(dag, "H", "P", effect="direct")# { D, R, S, T }

glmm_H_P = fit(MixedModel, @formula(lognP ~ 1 + H + D + R + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 lognP ~ 1 + H + D + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
    76.3168  -152.6337  -136.6337  -136.1194  -107.3022

Variance components:
            Column    Variance  Std.Dev.
ID       (Intercept)  0.1635051 0.4043576
Residual              0.0077141 0.0878299
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                    Coef.  Std. Error      z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)  -0.480939     0.184299    -2.61    0.0091
H             0.54097      0.0851168    6.36    <1e-09
D            -0.00496528   0.0431236   -0.12    0.9083
R             0.0402971    0.0388508    1.04    0.2996
S            -0.072663     0.0787063   -0.92    0.3559
T             0.000762771  0.00065167   1.17    0.2418
──────────────────────────────────────────────────────

"""

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

## Direct effet of D on P

adjustmentSets(dag, "D", "P", effect="direct")# { H, R, S, T }

glmm_D_P = fit(MixedModel, @formula(lognP ~ 1 + H + D + R + S + T + (1 | ID)), dag_df)

qqnorm(glmm_H_P; qqline=:fitrobust)
hist(residuals(glmm_H_P))
"""
Linear mixed model fit by maximum likelihood
 lognP ~ 1 + H + D + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -164.7185   329.4371   345.4371   345.9514   374.7685

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.866887 0.931068
Residual              0.040899 0.202236
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)  -1.1074      0.424364    -2.61    0.0091
H             1.24563     0.195989     6.36    <1e-09
D            -0.011433    0.0992959   -0.12    0.9083
R             0.0927875   0.0894574    1.04    0.2996
S            -0.167313    0.181228    -0.92    0.3559
T             0.00175634  0.00150053   1.17    0.2418
─────────────────────────────────────────────────────
"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_P);
coefplot(boot)
ridgeplot(boot)

@model function H_nP(nP, H, D, R, S, T, IDidx; n_id=length(unique(IDidx)))
  # population-level priors
  α ~ Normal(mean(nP), 2.5 * std(nP))
  βH ~ Normal(0, 0.5)
  βD ~ Normal(0, 0.5)
  βR ~ Normal(0, 0.5)
  βS ~ Normal(0, 0.5)
  βT ~ Normal(0, 0.5)

  σ ~ Exponential(std(nP))

  # priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts

  # likelihood
  nP̂ = @. α + α_ID[IDidx] + βH * H + βD * D + βR * R + βS * S + βT * T
  return nP ~ MvNormal(nP̂, σ^2 * I)
end

H_nP_model = H_nP(log10.(1 .+ dag_df.nP), dag_df.H, dag_df.D, dag_df.R, dag_df.S, dag_df.T, dag_df.IDidx);

H_nP_ch = sample(H_nP_model, NUTS(), MCMCThreads(), 3000, 4);

H_nP_df = DataFrame(H_nP_ch)[!, r"α\b|β"];

precis(H_nP_df)

"""
┌───────┬────────────────────────────────────────────────────────────┐
│ param │    mean     std     5.5%      50%    94.5%       histogram │
├───────┼────────────────────────────────────────────────────────────┤
│     α │  -0.454  0.1884  -0.7511  -0.4571  -0.1445        ▁▁▄█▆▂▁▁ │
│    βD │ -0.0058  0.0433  -0.0741  -0.0064   0.0631        ▁▁▃█▇▂▁▁ │
│    βH │  0.5197  0.0874   0.3806   0.5192   0.6602        ▁▁▂▇█▄▁▁ │
│    βR │  0.0416  0.0403   -0.022   0.0418   0.1065        ▁▁▃█▇▂▁▁ │
│    βS │ -0.0743  0.0791  -0.2044  -0.0728   0.0489  ▁▁▁▁▂▄▇█▇▄▂▁▁▁ │
│    βT │  0.0008  0.0007  -0.0003   0.0008   0.0018     ▁▁▁▃▆█▇▃▁▁▁ │
└───────┴────────────────────────────────────────────────────────────┘
"""

p = plot_chains_df(H_nP_ch; show_intercept=true)
save("../manuscript/Figures/plots/H_nP_chn.pdf", p)
p

## Direct effect of D on F

adjustmentSets(dag, "D", "F", effect="direct") # { H, P, R, S, T }

glmm_D_F = fit(MixedModel, @formula(Ḟ ~ 1 + D + H + P + R + S + T + (1 | ID)), dag_df)

"""
Minimizing 13    Time: 0:00:00 (10.93 ms/it)
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + D + H + P + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.9478   405.8957   423.8957   424.7447   454.5198

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.147341 0.383850
Residual              0.253006 0.502997
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.72242     0.295979      9.20    <1e-19
D             0.00816337  0.0999789     0.08    0.9349
H            -1.6543      0.16051     -10.31    <1e-24
P             0.0113786   0.167606      0.07    0.9459
R             0.089091    0.15239       0.58    0.5588
S            -0.329883    0.105401     -3.13    0.0017
T            -0.00333938  0.00443263   -0.75    0.4512
──────────────────────────────────────────────────────


"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_F);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of D on M

adjustmentSets(dag, "D", "M", effect="direct") # { F, H, P, R, S, T }

glmm_D_M = fit(MixedModel, @formula(M ~ 1 + D + Ḟ + H + P + R + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + D + Ḟ + H + P + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -150.7892   301.5784   321.5784   322.6211   355.6052

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.504269 0.710119
Residual              0.059222 0.243357
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.640662    0.400984     1.60    0.1101
D             0.437883    0.102503     4.27    <1e-04
Ḟ             0.0645476   0.0429016    1.50    0.1324
H            -0.527277    0.198484    -2.66    0.0079
P             0.13476     0.195148     0.69    0.4898
R             0.272096    0.104188     2.61    0.0090
S            -0.895595    0.145849    -6.14    <1e-09
T             0.00703544  0.00249179   2.82    0.0048
─────────────────────────────────────────────────────
"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of D on R

adjustmentSets(dag, "D", "R", effect="direct") # {}

glmm_D_R = fit(MixedModel, @formula(R ~ 1 + D + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 R ~ 1 + D + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
   -35.1084    70.2168    78.2168    78.3577    92.8826

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.153391 0.391652
Residual              0.026892 0.163988
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)   1.27536     0.100695   12.67    <1e-36
D            -0.0241109   0.0607939  -0.40    0.6917
────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_R);
coefplot(boot)
ridgeplot(boot)

## Direct effect of F on M

adjustmentSets(dag, "F", "M", effect="direct") # { D, H, P, R, S, T }

glmm_F_M = fit(MixedModel, @formula(M ~ 1 + Ḟ + D + H + P + R + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + Ḟ + D + H + P + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -150.7892   301.5784   321.5784   322.6211   355.6052

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.504269 0.710119
Residual              0.059222 0.243357
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.640662    0.400984     1.60    0.1101
Ḟ             0.0645476   0.0429016    1.50    0.1324
D             0.437883    0.102503     4.27    <1e-04
H            -0.527277    0.198484    -2.66    0.0079
P             0.13476     0.195148     0.69    0.4898
R             0.272096    0.104188     2.61    0.0090
S            -0.895595    0.145849    -6.14    <1e-09
T             0.00703544  0.00249179   2.82    0.0048
─────────────────────────────────────────────────────
"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_F_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of P on F

adjustmentSets(dag, "P", "F", effect="direct") # { D, H, R, S, T }

glmm_P_F = fit(MixedModel, @formula(Ḟ ~ 1 + P + D + H + R + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + P + D + H + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.9478   405.8957   423.8957   424.7447   454.5198

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.147341 0.383850
Residual              0.253006 0.502997
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.72242     0.295979      9.20    <1e-19
P             0.0113786   0.167606      0.07    0.9459
D             0.00816337  0.0999789     0.08    0.9349
H            -1.6543      0.16051     -10.31    <1e-24
R             0.089091    0.15239       0.58    0.5588
S            -0.329883    0.105401     -3.13    0.0017
T            -0.00333938  0.00443263   -0.75    0.4512
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_P_F);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of P on M

adjustmentSets(dag, "P", "M", effect="direct") # { D, F, H, R, S, T }

glmm_P_M = fit(MixedModel, @formula(M ~ 1 + P + D + Ḟ + H + R + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + P + D + Ḟ + H + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -150.7892   301.5784   321.5784   322.6211   355.6052

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.504269 0.710119
Residual              0.059222 0.243357
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.640662    0.400984     1.60    0.1101
P             0.13476     0.195148     0.69    0.4898
D             0.437883    0.102503     4.27    <1e-04
Ḟ             0.0645476   0.0429016    1.50    0.1324
H            -0.527277    0.198484    -2.66    0.0079
R             0.272096    0.104188     2.61    0.0090
S            -0.895595    0.145849    -6.14    <1e-09
T             0.00703544  0.00249179   2.82    0.0048
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_P_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of R on F

adjustmentSets(dag, "R", "F", effect="direct") # { D, H, P, S, T }

glmm_R_F = fit(MixedModel, @formula(Ḟ ~ 1 + R + D + H + P + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + R + D + H + P + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.9478   405.8957   423.8957   424.7447   454.5198

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.147341 0.383850
Residual              0.253006 0.502997
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.72242     0.295979      9.20    <1e-19
R             0.089091    0.15239       0.58    0.5588
D             0.00816337  0.0999789     0.08    0.9349
H            -1.6543      0.16051     -10.31    <1e-24
P             0.0113786   0.167606      0.07    0.9459
S            -0.329883    0.105401     -3.13    0.0017
T            -0.00333938  0.00443263   -0.75    0.4512
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_R_F);
coefplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of R on M

adjustmentSets(dag, "R", "M", effect="direct") # { D, F, H, P, S, T }

glmm_R_M = fit(MixedModel, @formula(M ~ 1 + R + D + Ḟ + H + P + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + R + D + Ḟ + H + P + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -150.7892   301.5784   321.5784   322.6211   355.6052

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.504269 0.710119
Residual              0.059222 0.243357
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.640662    0.400984     1.60    0.1101
R             0.272096    0.104188     2.61    0.0090
D             0.437883    0.102503     4.27    <1e-04
Ḟ             0.0645476   0.0429016    1.50    0.1324
H            -0.527277    0.198484    -2.66    0.0079
P             0.13476     0.195148     0.69    0.4898
S            -0.895595    0.145849    -6.14    <1e-09
T             0.00703544  0.00249179   2.82    0.0048
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_R_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of S on F

adjustmentSets(dag, "S", "F", effect="direct") # { D, H, P, R, T }

glmm_S_F = fit(MixedModel, @formula(Ḟ ~ 1 + S + D + H + P + R + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + S + D + H + P + R + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.9478   405.8957   423.8957   424.7447   454.5198

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.147341 0.383850
Residual              0.253006 0.502997
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.72242     0.295979      9.20    <1e-19
S            -0.329883    0.105401     -3.13    0.0017
D             0.00816337  0.0999789     0.08    0.9349
H            -1.6543      0.16051     -10.31    <1e-24
P             0.0113786   0.167606      0.07    0.9459
R             0.089091    0.15239       0.58    0.5588
T            -0.00333938  0.00443263   -0.75    0.4512
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_F);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of S on M

adjustmentSets(dag, "S", "M", effect="direct") # { D, F, H, P, R, T }

glmm_S_M = fit(MixedModel, @formula(M ~ 1 + S + D + Ḟ + H + P + R + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + S + D + Ḟ + H + P + R + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -150.7892   301.5784   321.5784   322.6211   355.6052

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.504269 0.710119
Residual              0.059222 0.243357
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.640662    0.400984     1.60    0.1101
S            -0.895595    0.145849    -6.14    <1e-09
D             0.437883    0.102503     4.27    <1e-04
Ḟ             0.0645476   0.0429016    1.50    0.1324
H            -0.527277    0.198484    -2.66    0.0079
P             0.13476     0.195148     0.69    0.4898
R             0.272096    0.104188     2.61    0.0090
T             0.00703544  0.00249179   2.82    0.0048
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of S on R

adjustmentSets(dag, "S", "R", effect="direct") # { }

glmm_S_R = fit(MixedModel, @formula(R ~ 1 + S + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 R ~ 1 + S + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
   -35.1855    70.3710    78.3710    78.5118    93.0367

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.152593 0.390631
Residual              0.026990 0.164287
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   1.24182      0.119847   10.36    <1e-24
S            -0.00228394   0.0778031  -0.03    0.9766
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_R);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of T on F

adjustmentSets(dag, "T", "F", effect="direct") # { D, H, P, R, S }

glmm_T_F = fit(MixedModel, @formula(Ḟ ~ 1 + T + D + H + P + R + S + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + T + D + H + P + R + S + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.9478   405.8957   423.8957   424.7447   454.5198

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.147341 0.383850
Residual              0.253006 0.502997
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.72242     0.295979      9.20    <1e-19
T            -0.00333938  0.00443263   -0.75    0.4512
D             0.00816337  0.0999789     0.08    0.9349
H            -1.6543      0.16051     -10.31    <1e-24
P             0.0113786   0.167606      0.07    0.9459
R             0.089091    0.15239       0.58    0.5588
S            -0.329883    0.105401     -3.13    0.0017
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_T_F);
coefplot(boot)
ridgeplot(boot)

## Direct effect of T on M

adjustmentSets(dag, "T", "M", effect="direct") # { D, F, H, P, R, S }

glmm_T_M = fit(MixedModel, @formula(M ~ 1 + T + D + Ḟ + H + P + R + S + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + T + D + Ḟ + H + P + R + S + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -150.7892   301.5784   321.5784   322.6211   355.6052

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.504269 0.710119
Residual              0.059222 0.243357
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.640662    0.400984     1.60    0.1101
T             0.00703544  0.00249179   2.82    0.0048
D             0.437883    0.102503     4.27    <1e-04
Ḟ             0.0645476   0.0429016    1.50    0.1324
H            -0.527277    0.198484    -2.66    0.0079
P             0.13476     0.195148     0.69    0.4898
R             0.272096    0.104188     2.61    0.0090
S            -0.895595    0.145849    -6.14    <1e-09
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_T_M);
coefplot(boot)
ridgeplot(boot)

## Direct effect of T on R

adjustmentSets(dag, "T", "R", effect="direct") # { H }

glmm_T_R = fit(MixedModel, @formula(R ~ 1 + T + H + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 R ~ 1 + T + H + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
    16.7620   -33.5239   -23.5239   -23.3119    -5.1918

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.047543 0.218044
Residual              0.028003 0.167341
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  0.31996     0.0736581    4.34    <1e-04
T            0.00122903  0.00116347   1.06    0.2908
H            0.65482     0.0489267   13.38    <1e-40
────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_T_R);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on D

adjustmentSets(dag, "H", "D", effect="direct") # { }

glmm_H_D = fit(MixedModel, @formula(D ~ 1 + H + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 D ~ 1 + H + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
   -11.9807    23.9614    31.9614    32.1022    46.6271

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.234320 0.484066
Residual              0.016476 0.128358
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
──────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)
──────────────────────────────────────────────────
(Intercept)  1.44816    0.141159   10.26    <1e-23
H            0.059346   0.0971981   0.61    0.5415
──────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_D);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on F

adjustmentSets(dag, "H", "F", effect="direct") # { D, P, R, S, T }

glmm_H_F = fit(MixedModel, @formula(Ḟ ~ 1 + H + D + P + R + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + H + D + P + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.9478   405.8957   423.8957   424.7447   454.5198

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.147341 0.383850
Residual              0.253006 0.502997
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.72242     0.295979      9.20    <1e-19
H            -1.6543      0.16051     -10.31    <1e-24
D             0.00816337  0.0999789     0.08    0.9349
P             0.0113786   0.167606      0.07    0.9459
R             0.089091    0.15239       0.58    0.5588
S            -0.329883    0.105401     -3.13    0.0017
T            -0.00333938  0.00443263   -0.75    0.4512
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_F);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on M

adjustmentSets(dag, "H", "M", effect="direct") # { D, F, P, R, S, T }

glmm_H_M = fit(MixedModel, @formula(M ~ 1 + H + D + Ḟ + P + R + S + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + H + D + Ḟ + P + R + S + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -150.7892   301.5784   321.5784   322.6211   355.6052

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.504269 0.710119
Residual              0.059222 0.243357
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.640662    0.400984     1.60    0.1101
H            -0.527277    0.198484    -2.66    0.0079
D             0.437883    0.102503     4.27    <1e-04
Ḟ             0.0645476   0.0429016    1.50    0.1324
P             0.13476     0.195148     0.69    0.4898
R             0.272096    0.104188     2.61    0.0090
S            -0.895595    0.145849    -6.14    <1e-09
T             0.00703544  0.00249179   2.82    0.0048
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_M);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on R

adjustmentSets(dag, "H", "R", effect="direct") # { T }

glmm_H_R = fit(MixedModel, @formula(R ~ 1 + H + T + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 R ~ 1 + H + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
    16.7620   -33.5239   -23.5239   -23.3119    -5.1918

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.047543 0.218044
Residual              0.028003 0.167341
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  0.31996     0.0736581    4.34    <1e-04
H            0.65482     0.0489267   13.38    <1e-40
T            0.00122903  0.00116347   1.06    0.2908
────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_R);
coefplot(boot)
ridgeplot(boot)
