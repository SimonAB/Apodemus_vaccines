#=
SCM Specification
- Julia version: 1.10
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
dag_df = df[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :T, :P, :nP, :ID, :IDidx, :vax_history, :Vidx]]
dag_df.lognP = log10.(1 .+ dag_df.nP);
# describe(dag_df)

dag_df_unique = df_unique[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :T, :P, :nP, :Vidx, :vax_history, :ID]];
filtered_df = dag_df[dag_df.P.==1, :];
filtered_unique_df = dag_df_unique[dag_df_unique.P.==1, :];

# select only infected mice
dag_df_infected = dag_df[dag_df.P.==2, :] # select rows of dag_df for which dag_df.P.==2

# DAG specification - this is our graphical causal hypothesis

dag = dagitty("dag{
D -> E;
D -> F;
D -> M;
D -> P;
D -> R;
F -> E;
F -> M;
H -> D;
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

# Turing.setadbackend(:forwarddiff)
V_E_DE_chn = sample(V_E_DE_model, NUTS(), MCMCThreads(), 3000, 4)

V_E_DE_chn_df = DataFrame(V_E_DE_chn)[!, r"α\b|β"];
precis(V_E_DE_chn_df)

# Mixed model
glmm_V_E_NDE = fit(MixedModel, @formula(E ~ 1 + V + D + Ḟ + H + M + P + R + S + (1 | ID)), dag_df)

"""
┌────────┬─────────────────────────────────────────────────────────────┐
│  param │    mean      std    5.0 %     50 %   95.0 %       histogram │
│ String │ Float64  Float64  Float64  Float64  Float64          String │
├────────┼─────────────────────────────────────────────────────────────┤
│      α │   -1.43    0.351   -2.005   -1.432   -0.845          ▁▁▇█▂▁ │
│    βVE │   1.481    0.118    1.284    1.481    1.674     ▁▁▂▅██▄▁▁▁▁ │
│    βDE │  -0.306    0.093   -0.458   -0.307   -0.155        ▁▁▃██▃▁▁ │
│    βFE │  -0.178     0.06   -0.275    -0.18   -0.077     ▁▁▁▃▇█▆▂▁▁▁ │
│    βHE │  -0.744    0.169   -1.021   -0.745   -0.464        ▁▁▂▆█▄▁▁ │
│    βME │   0.047    0.061   -0.051    0.046    0.148   ▁▁▂▅██▄▂▁▁▁▁▁ │
│    βPE │  -0.019    0.114   -0.209    -0.02    0.167      ▁▁▂▅█▇▃▁▁▁ │
│    βRE │  -0.169    0.166   -0.445    -0.17    0.102  ▁▁▁▁▂▅██▇▄▂▁▁▁ │
│    βSE │   0.274    0.107    0.102    0.272    0.453     ▁▁▁▅█▇▃▁▁▁▁ │
└────────┴─────────────────────────────────────────────────────────────┘


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
glmm_P_E_all = fit(MixedModel, @formula(E ~ 1 + lognP + D + H + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + lognP + D + H + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -291.8487   583.6974   601.6974   602.3426   634.6953

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.047808 0.218650
Residual              0.398274 0.631089
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)  -1.97382      0.325576   -6.06    <1e-08
lognP        -0.00896315   0.112239   -0.08    0.9364
D            -0.290815     0.0858287  -3.39    0.0007
H            -0.608419     0.146348   -4.16    <1e-04
R            -0.0965752    0.165227   -0.58    0.5589
S             0.287458     0.0881604   3.26    0.0011
V             1.61397      0.112689   14.32    <1e-45
─────────────────────────────────────────────────────

"""

qqnorm(glmm_P_E_all; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E_all);
coefplot(glmm_P_E_all)
coefplot(boot)
ridgeplot(boot)

# Infected only
glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + H + R + S + V + (1 | ID)), dag_df_infected)
"""
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/wR4rk/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 E ~ 1 + lognP + D + H + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
   -54.1721   108.3441   124.3441   127.7727   139.7987

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.019575 0.139909
Residual              0.471247 0.686474
 Number of obs: 51; levels of grouping factors: 19

  Fixed-effects parameters:
────────────────────────────────────────────────────
                 Coef.  Std. Error       z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  -1.61109     0.960908   -1.68    0.0936
lognP        -0.52165     0.315781   -1.65    0.0985
D            -0.151094    0.24211    -0.62    0.5326
H            -0.0       NaN         NaN       NaN
R            -0.116407    0.324117   -0.36    0.7195
S             0.157128    0.240046    0.65    0.5127
V             1.16835     0.338118    3.46    0.0005
────────────────────────────────────────────────────

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
"""
┌───────┬─────────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %     histogram │
├───────┼─────────────────────────────────────────────────────┤
│     α │ -0.537  0.141  -0.777  -0.535  -0.311  ▁▁▁▁▃▆█▇▄▂▁▁ │
│    βD │ -0.009  0.043  -0.079  -0.009   0.063      ▁▁▃█▇▂▁▁ │
│    βH │  0.558  0.091   0.409   0.557   0.707      ▁▁▅█▆▂▁▁ │
└───────┴─────────────────────────────────────────────────────┘

"""

# GLMM for the total effect of D on nP among the infected

glmm_D_nP = fit(MixedModel, @formula(lognP ~ D + (1 | ID)), dag_df_infected)

"""
Linear mixed model fit by maximum likelihood
 lognP ~ 1 + D + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
    13.1019   -26.2039   -18.2039   -17.3343   -10.4766

Variance components:
            Column    Variance  Std.Dev.
ID       (Intercept)  0.1648723 0.4060446
Residual              0.0083749 0.0915143
 Number of obs: 51; levels of grouping factors: 19

  Fixed-effects parameters:
──────────────────────────────────────────────────
                 Coef.  Std. Error     z  Pr(>|z|)
──────────────────────────────────────────────────
(Intercept)  1.20796     0.131198   9.21    <1e-19
D            0.0449608   0.0622307  0.72    0.4700
──────────────────────────────────────────────────

"""

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

"""
┌───────────┬───────────────────────────────────────────────────────┐
│     param │   mean    std   5.0 %    50 %  95.0 %       histogram │
├───────────┼───────────────────────────────────────────────────────┤
│         α │  1.065  0.315   0.546   1.068   1.577         ▁▁▆█▂▁▁ │
│  βHabitat │ -0.631  0.157  -0.888  -0.631  -0.371  ▁▁▁▂▄▆██▅▂▁▁▁▁ │
│     βDiet │ -0.225  0.092  -0.375  -0.225  -0.073       ▁▁▁▄█▇▂▁▁ │
│      βFat │ -0.075   0.05  -0.155  -0.075   0.008       ▁▁▂▆█▅▂▁▁ │
│     βMass │ -0.048  0.057   -0.14  -0.049   0.045     ▁▁▁▄██▅▁▁▁▁ │
│ βParasite │  -0.11  0.112  -0.297  -0.111   0.073     ▁▁▂▅██▄▁▁▁▁ │
│    βRepro │ -0.091  0.147  -0.332  -0.091   0.154    ▁▁▁▂▅██▆▃▁▁▁ │
│      βSex │  0.176  0.104   0.005   0.175   0.349        ▁▁▅█▇▃▁▁ │
└───────────┴───────────────────────────────────────────────────────┘


"""

p1 = plot_chains_df(DE_H_E_chn; show_traces=true)
save("../manuscript/Figures/plots/MultiLevel_DE_H_E_chn_traces.pdf", p1)
p1

p2 = plot_chains_df(DE_H_E_chn; show_traces=false)
save("../manuscript/Figures/plots/MultiLevel_DE_H_E_chn.pdf", p2)
p2

# Mixed model
glmm_DE_H_E = fit(MixedModel, @formula(E ~ 1 + H + D + Ḟ + M + nP + R + S + (1 | V) + (1 | ID)), dag_df)

"""
Minimizing 30    Time: 0:00:00 (11.40 ms/it)
Linear mixed model fit by maximum likelihood
 E ~ 1 + H + D + Ḟ + M + nP + R + S + (1 | V) + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -214.5942   429.1884   451.1884   452.4455   488.6178

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.048500 0.220227
V        (Intercept)  0.561252 0.749168
Residual              0.343692 0.586253
 Number of obs: 222; levels of grouping factors: 105, 2

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.251257    0.612351     0.41    0.6816
H            -0.54066     0.179478    -3.01    0.0026
D            -0.273063    0.0968134   -2.82    0.0048
Ḟ             0.00994091  0.0736591    0.13    0.8926
M            -0.0988904   0.0674378   -1.47    0.1425
nP           -0.00142628  0.00195037  -0.73    0.4646
R             0.0698526   0.174033     0.40    0.6881
S             0.149096    0.110913     1.34    0.1789
─────────────────────────────────────────────────────

"""

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

"""
┌───────┬──────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %  histogram │
├───────┼──────────────────────────────────────────────────┤
│    βD │ -0.248  0.087   -0.39  -0.247  -0.106  ▁▁▁▅█▅▁▁▁ │
│    βH │ -0.654  0.098  -0.815  -0.655  -0.492  ▁▁▂▅█▅▂▁▁ │
└───────┴──────────────────────────────────────────────────┘

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


"""
┌───────┬───────────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %       histogram │
├───────┼───────────────────────────────────────────────────────┤
│     α │  1.061  0.307   0.559   1.062   1.568   ▁▁▁▂▅██▇▃▂▁▁▁ │
│   βDE │ -0.226  0.089  -0.374  -0.225  -0.078        ▁▁▄█▇▂▁▁ │
│   βFE │ -0.074   0.05  -0.155  -0.074   0.006        ▁▂▆█▆▂▁▁ │
│   βHE │ -0.628  0.157  -0.883  -0.629  -0.366  ▁▁▁▂▄▇██▅▂▁▁▁▁ │
│   βME │ -0.047  0.055  -0.138  -0.047   0.044       ▁▁▄██▄▁▁▁ │
│   βPE │ -0.109  0.112  -0.296  -0.108   0.076       ▁▂▄██▄▁▁▁ │
│   βRE │ -0.094  0.146  -0.335  -0.094   0.145   ▁▁▁▂▆██▆▃▁▁▁▁ │
│   βSE │  0.178  0.103   0.009   0.178   0.345        ▁▁▅█▇▃▁▁ │
└───────┴───────────────────────────────────────────────────────┘

"""

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

"""
┌───────┬───────────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %       histogram │
├───────┼───────────────────────────────────────────────────────┤
│     α │  1.069  0.314   0.555   1.069   1.583    ▁▁▁▂▅▇█▆▄▂▁▁ │
│   βFE │ -0.077  0.049  -0.156  -0.077   0.005       ▁▁▂▆█▅▂▁▁ │
│   βDF │ -0.225   0.09  -0.371  -0.227  -0.076        ▁▁▄█▆▂▁▁ │
│   βHF │ -0.633   0.16  -0.895  -0.632  -0.369  ▁▁▁▁▂▄▇██▄▂▁▁▁ │
│   βMF │ -0.048  0.055  -0.139  -0.047   0.044      ▁▁▁▄██▄▁▁▁ │
│   βPF │  -0.11  0.113  -0.296   -0.11   0.077      ▁▁▁▅██▄▁▁▁ │
│   βRF │ -0.091  0.147  -0.334  -0.091   0.149    ▁▁▁▂▅██▆▃▁▁▁ │
│   βSF │  0.175  0.103   0.006   0.176   0.345      ▁▁▁▅█▇▃▁▁▁ │
└───────┴───────────────────────────────────────────────────────┘

"""




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
└───────┴───────────────────────────────────────────────────────────┘
"""

coeftab_plot(M_E_df, pars=[:βW, :βD, :βM, :βF, :βP, :βR, :βS])

p = plot_chains_df(M_E_ch; show_intercept=true)
save("../manuscript/Figures/plots/M_E_chn.pdf", p)
p

## Direct effet of H on P

adjustmentSets(dag, "H", "P", effect="direct")# { D, R, S, V }

glmm_H_P = fit(MixedModel, @formula(lognP ~ 1 + H + D + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 lognP ~ 1 + H + D + R + S + V + (1 | ID)
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
H             0.539661     0.0851757   6.34    <1e-09
D            -0.00415668   0.0433778  -0.10    0.9237
R             0.0445632    0.039308    1.13    0.2569
S            -0.0713095    0.0786907  -0.91    0.3648
V             0.0113116    0.023732    0.48    0.6336
─────────────────────────────────────────────────────

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

adjustmentSets(dag, "D", "P", effect="direct")# { H, R, S, V }

glmm_D_P = fit(MixedModel, @formula(lognP ~ 1 + H + D + R + S + V + (1 | ID)), dag_df)

qqnorm(glmm_H_P; qqline=:fitrobust)
hist(residuals(glmm_H_P))
"""
Linear mixed model fit by maximum likelihood
 lognP ~ 1 + H + D + R + S + V + (1 | ID)
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
H             0.539661     0.0851757   6.34    <1e-09
D            -0.00415668   0.0433778  -0.10    0.9237
R             0.0445632    0.039308    1.13    0.2569
S            -0.0713095    0.0786907  -0.91    0.3648
V             0.0113116    0.023732    0.48    0.6336
─────────────────────────────────────────────────────

"""

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

"""
┌───────┬──────────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %      histogram │
├───────┼──────────────────────────────────────────────────────┤
│     α │ -0.423  0.263  -0.851  -0.427   0.015     ▁▁▂▅██▄▂▁▁ │
│    βH │  0.517  0.086   0.379   0.516   0.658       ▁▂▇█▃▁▁▁ │
│    βD │ -0.002  0.043  -0.075  -0.003   0.069       ▁▁▃██▃▁▁ │
│    βR │   0.05  0.041  -0.016   0.049   0.117        ▁▁▃██▃▁ │
│    βS │ -0.076  0.081  -0.211  -0.077   0.059  ▁▁▁▂▄██▇▄▂▁▁▁ │
└───────┴──────────────────────────────────────────────────────┘


"""

p = plot_chains_df(H_nP_ch; show_intercept=true)
save("../manuscript/Figures/plots/H_nP_chn.pdf", p)
p

## Direct effect of D on F

adjustmentSets(dag, "D", "F", effect="direct") # { H, P, R, S}

glmm_D_F = fit(MixedModel, @formula(Ḟ ~ 1 + D + H + P + R + S + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + D + H + P + R + S + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -203.2277   406.4553   422.4553   423.1314   449.6768

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.144320 0.379895
Residual              0.255575 0.505544
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.70376      0.294176     9.19    <1e-19
D             0.00729174   0.0997214    0.07    0.9417
H            -1.6568       0.160156   -10.34    <1e-24
P            -0.00274903   0.165899    -0.02    0.9868
R             0.0767133    0.151855     0.51    0.6134
S            -0.33479      0.104839    -3.19    0.0014
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_F);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of D on M

adjustmentSets(dag, "D", "M", effect="direct") # { F, H, P, R, S, V }

glmm_D_M = fit(MixedModel, @formula(M ~ 1 + D + Ḟ + H + P + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + D + Ḟ + H + P + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.5818   309.1637   329.1637   330.2063   363.1904

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.507168 0.712157
Residual              0.062677 0.250354
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)   0.595669    0.462329    1.29    0.1976
D             0.441376    0.104883    4.21    <1e-04
Ḟ             0.0546147   0.0437905   1.25    0.2123
H            -0.55507     0.201918   -2.75    0.0060
P             0.174696    0.197429    0.88    0.3762
R             0.310284    0.109216    2.84    0.0045
S            -0.886441    0.146589   -6.05    <1e-08
V             0.0583054   0.124591    0.47    0.6398
────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of D on R

adjustmentSets(dag, "D", "R", effect="direct") # {H}

glmm_D_R = fit(MixedModel, @formula(R ~ 1 + D + H + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 R ~ 1 + D + H + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
    16.2329   -32.4658   -22.4658   -22.2538    -4.1337

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.048602 0.220458
Residual              0.027884 0.166985
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.35744      0.0947436   3.77    0.0002
D            -0.00953244   0.0430006  -0.22    0.8246
H             0.656937     0.0493367  13.32    <1e-39
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_R);
coefplot(boot)
ridgeplot(boot)

## Direct effect of F on M

adjustmentSets(dag, "F", "M", effect="direct") # { D, H, P, R, S, V }

glmm_F_M = fit(MixedModel, @formula(M ~ 1 + Ḟ + D + H + P + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + Ḟ + D + H + P + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.5818   309.1637   329.1637   330.2063   363.1904

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.507168 0.712157
Residual              0.062677 0.250354
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)   0.595669    0.462329    1.29    0.1976
Ḟ             0.0546147   0.0437905   1.25    0.2123
D             0.441376    0.104883    4.21    <1e-04
H            -0.55507     0.201918   -2.75    0.0060
P             0.174696    0.197429    0.88    0.3762
R             0.310284    0.109216    2.84    0.0045
S            -0.886441    0.146589   -6.05    <1e-08
V             0.0583054   0.124591    0.47    0.6398
────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_F_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of P on F

adjustmentSets(dag, "P", "F", effect="direct") # { D, H, R, S, V }

glmm_P_F = fit(MixedModel, @formula(Ḟ ~ 1 + P + D + H + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + P + D + H + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.5831   405.1662   423.1662   424.0153   453.7903

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.139557 0.373573
Residual              0.256358 0.506319
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                  Coef.  Std. Error       z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   2.47549     0.352539     7.02    <1e-11
P            -0.01649     0.165081    -0.10    0.9204
D             0.0089668   0.0990151    0.09    0.9278
H            -1.6637      0.15933    -10.44    <1e-24
R             0.0790238   0.151328     0.52    0.6015
S            -0.329262    0.104082    -3.16    0.0016
V             0.135703    0.118894     1.14    0.2537
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_P_F);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of P on M

adjustmentSets(dag, "P", "M", effect="direct") # { D, F, H, R, S, V }

glmm_P_M = fit(MixedModel, @formula(M ~ 1 + P + D + Ḟ + H + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + P + D + Ḟ + H + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.5818   309.1637   329.1637   330.2063   363.1904

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.507168 0.712157
Residual              0.062677 0.250354
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)   0.595669    0.462329    1.29    0.1976
P             0.174696    0.197429    0.88    0.3762
D             0.441376    0.104883    4.21    <1e-04
Ḟ             0.0546147   0.0437905   1.25    0.2123
H            -0.55507     0.201918   -2.75    0.0060
R             0.310284    0.109216    2.84    0.0045
S            -0.886441    0.146589   -6.05    <1e-08
V             0.0583054   0.124591    0.47    0.6398
────────────────────────────────────────────────────


"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_P_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of R on F

adjustmentSets(dag, "R", "F", effect="direct") # { D, H, P, S, V }

glmm_R_F = fit(MixedModel, @formula(Ḟ ~ 1 + R + D + H + P + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + R + D + H + P + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.5831   405.1662   423.1662   424.0153   453.7903

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.139557 0.373573
Residual              0.256358 0.506319
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.47549      0.352539     7.02    <1e-11
R             0.0790238    0.151328     0.52    0.6015
D             0.00896681   0.0990151    0.09    0.9278
H            -1.6637       0.15933    -10.44    <1e-24
P            -0.01649      0.165081    -0.10    0.9204
S            -0.329262     0.104082    -3.16    0.0016
V             0.135703     0.118894     1.14    0.2537
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_R_F);
coefplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of R on M

adjustmentSets(dag, "R", "M", effect="direct") # { D, F, H, P, S, V }

glmm_R_M = fit(MixedModel, @formula(M ~ 1 + R + D + Ḟ + H + P + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + R + D + Ḟ + H + P + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.5818   309.1637   329.1637   330.2063   363.1904

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.507168 0.712157
Residual              0.062677 0.250354
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)   0.595669    0.462329    1.29    0.1976
R             0.310284    0.109216    2.84    0.0045
D             0.441376    0.104883    4.21    <1e-04
Ḟ             0.0546147   0.0437905   1.25    0.2123
H            -0.55507     0.201918   -2.75    0.0060
P             0.174696    0.197429    0.88    0.3762
S            -0.886441    0.146589   -6.05    <1e-08
V             0.0583054   0.124591    0.47    0.6398
────────────────────────────────────────────────────


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

adjustmentSets(dag, "S", "M", effect="direct") # { D, F, H, P, R, V }

glmm_S_M = fit(MixedModel, @formula(M ~ 1 + S + D + Ḟ + H + P + R + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + S + D + Ḟ + H + P + R + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.5818   309.1637   329.1637   330.2063   363.1904

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.507168 0.712157
Residual              0.062677 0.250354
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)   0.595669    0.462329    1.29    0.1976
S            -0.886441    0.146589   -6.05    <1e-08
D             0.441376    0.104883    4.21    <1e-04
Ḟ             0.0546147   0.0437905   1.25    0.2123
H            -0.55507     0.201918   -2.75    0.0060
P             0.174696    0.197429    0.88    0.3762
R             0.310284    0.109216    2.84    0.0045
V             0.0583054   0.124591    0.47    0.6398
────────────────────────────────────────────────────

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

adjustmentSets(dag, "H", "F", effect="direct") # { D, P, R, S, V }

glmm_H_F = fit(MixedModel, @formula(Ḟ ~ 1 + H + D + P + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + H + D + P + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.5831   405.1662   423.1662   424.0153   453.7903

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.139557 0.373573
Residual              0.256358 0.506319
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.47549      0.352539     7.02    <1e-11
H            -1.6637       0.15933    -10.44    <1e-24
D             0.00896681   0.0990151    0.09    0.9278
P            -0.01649      0.165081    -0.10    0.9204
R             0.0790238    0.151328     0.52    0.6015
S            -0.329262     0.104082    -3.16    0.0016
V             0.135703     0.118894     1.14    0.2537
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_F);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on M

adjustmentSets(dag, "H", "M", effect="direct") # { D, F, P, R, S, V }

glmm_H_M = fit(MixedModel, @formula(M ~ 1 + H + D + Ḟ + P + R + S + V + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 M ~ 1 + H + D + Ḟ + P + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.5818   309.1637   329.1637   330.2063   363.1904

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.507168 0.712157
Residual              0.062677 0.250354
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)   0.595669    0.462329    1.29    0.1976
H            -0.55507     0.201918   -2.75    0.0060
D             0.441376    0.104883    4.21    <1e-04
Ḟ             0.0546147   0.0437905   1.25    0.2123
P             0.174696    0.197429    0.88    0.3762
R             0.310284    0.109216    2.84    0.0045
S            -0.886441    0.146589   -6.05    <1e-08
V             0.0583054   0.124591    0.47    0.6398
────────────────────────────────────────────────────



"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_M);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on R

adjustmentSets(dag, "H", "R", effect="direct") # { D }

glmm_H_R = fit(MixedModel, @formula(R ~ 1 + H + D + (1 | ID)), dag_df)

"""
Linear mixed model fit by maximum likelihood
 R ~ 1 + H + D + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
    16.2329   -32.4658   -22.4658   -22.2538    -4.1337

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.048602 0.220458
Residual              0.027884 0.166985
 Number of obs: 289; levels of grouping factors: 110

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.35744      0.0947436   3.77    0.0002
H             0.656937     0.0493367  13.32    <1e-39
D            -0.00953244   0.0430006  -0.22    0.8246
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_R);
coefplot(boot)
ridgeplot(boot)
