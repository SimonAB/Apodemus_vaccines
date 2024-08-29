#=
SCM Specification post intervention
- Julia version: 1.10
- Author: Simon A Babayan
- Date: 2024-07-24
=#

#=
Here, we simulate the effect of an intervention on the SCM. Specifically, we want to see the effects of removing Parasite infection on the system and on vaccine efficacy.
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

# Set up intervention: P = 0
df.P .= 0
df.nP .= 0

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

## Average causal effect of V on E (total effect)
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
V_E_DE_chn = sample(V_E_DE_model, NUTS(), MCMCThreads(), 1000, 4)

V_E_DE_chn_df = DataFrame(V_E_DE_chn)[!, r"α\b|β"];
precis(V_E_DE_chn_df)

# Mixed model
glmm_V_E_NDE = fit(MixedModel, @formula(E ~ 1 + V + D + Ḟ + H + M + P + R + S + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 E ~ 1 + V + D + Ḟ + H + M + P + R + S + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -209.1437   418.2874   438.2874   439.3300   472.3142

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.046775 0.216276
Residual              0.342608 0.585327
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                  Coef.   Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)  -1.97789      0.35272     -5.61    <1e-07
V             1.50277      0.111504    13.48    <1e-40
D            -0.279366     0.0958964   -2.91    0.0036
Ḟ             0.0109418    0.0733687    0.15    0.8814
H            -0.548415     0.178548    -3.07    0.0021
M            -0.103157     0.066915    -1.54    0.1232
P            -0.0        NaN          NaN       NaN
R             0.0447345    0.169964     0.26    0.7924
S             0.158273     0.109624     1.44    0.1488
──────────────────────────────────────────────────────

"""
qqnorm(glmm_V_E_NDE; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_V_E_NDE);
coefplot(boot)
ridgeplot(boot)


### Prediction of the post-intervention values of V:

# Make a prediction given an input vector.
function prediction(chain, x)
  p = get_params(chain[200:end, :, :])
  targets = p.intercept' .+ x * reduce(hcat, p.coefficients)'
  return vec(mean(targets; dims=2))
end

# Calculate the predictions for the training and testing sets and unstandardize them.
post_vacc_matrix = Matrix(select(dag_df, Not(:E)))
post_prediction_bayes = prediction(V_E_DE_chn, post_vacc_matrix)
StatsBase.reconstruct!(dag_df.E, post_prediction_bayes)

# Show the predictions on the test data set.
DataFrame(; E=dag_df.E, Bayes=post_prediction_bayes)


## Direct effect of S on E

adjustmentSets(dag, "S", "E", effect="direct") # { D, F, H, M, P, R, V }

glmm_D_S_E = fit(MixedModel, @formula(E ~ 1 + S + D + Ḟ + H + M + nP + R + V + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 E ~ 1 + S + D + Ḟ + H + M + nP + R + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -209.1437   418.2874   438.2874   439.3300   472.3142

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.046775 0.216276
Residual              0.342608 0.585327
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                  Coef.   Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)  -1.97789      0.35272     -5.61    <1e-07
S             0.158273     0.109624     1.44    0.1488
D            -0.279366     0.0958964   -2.91    0.0036
Ḟ             0.0109418    0.0733687    0.15    0.8814
H            -0.548415     0.178548    -3.07    0.0021
M            -0.103157     0.066915    -1.54    0.1232
nP           -0.0        NaN          NaN       NaN
R             0.0447345    0.169964     0.26    0.7924
V             1.50277      0.111504    13.48    <1e-40
──────────────────────────────────────────────────────

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
Before intervention:
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
Before intervention:

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

After intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Minimizing 32    Time: 0:00:00 (11.12 ms/it)
Linear mixed model fit by maximum likelihood
 E ~ 1 + H + D + Ḟ + M + nP + R + S + (1 | V) + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -214.8601   429.7203   449.7203   450.7629   483.7470

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.049939 0.223469
V        (Intercept)  0.557017 0.746336
Residual              0.343470 0.586063
 Number of obs: 222; levels of grouping factors: 105, 2

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                  Coef.   Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.276702     0.609987     0.45    0.6501
H            -0.544708     0.17982     -3.03    0.0025
D            -0.279671     0.0966775   -2.89    0.0038
Ḟ             0.0122398    0.0736939    0.17    0.8681
M            -0.102744     0.0673809   -1.52    0.1273
nP           -0.0        NaN          NaN       NaN
R             0.0443664    0.170847     0.26    0.7951
S             0.158348     0.110584     1.43    0.1522
──────────────────────────────────────────────────────


"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_DE_H_E);
glmm_DE_H_E_coefplot = coefplot(boot)
save("../manuscript/Figures/plots/MultiLevel_DE_H_E_coefplot.pdf", glmm_DE_H_E_coefplot)
glmm_DE_H_E_ridgeplot = ridgeplot(boot)
save("../manuscript/Figures/plots/MultiLevel_DE_H_E_ridgeplot.pdf", glmm_DE_H_E_ridgeplot)


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
Before intervention:
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
Before intervention:
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
Before intervention:
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

## Direct effect of D on M

adjustmentSets(dag, "D", "M", effect="direct") # { F, H, P, R, S, V }

glmm_D_M = fit(MixedModel, @formula(M ~ 1 + D + Ḟ + H + P + R + S + V + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 M ~ 1 + D + Ḟ + H + P + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.9670   309.9340   327.9340   328.7831   358.5581

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.515804 0.718195
Residual              0.062231 0.249462
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                  Coef.   Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.709957     0.449016     1.58    0.1138
D             0.433761     0.104878     4.14    <1e-04
Ḟ             0.0528504    0.0436848    1.21    0.2264
H            -0.481975     0.185886    -2.59    0.0095
P            -0.0        NaN          NaN       NaN
R             0.314197     0.10863      2.89    0.0038
S            -0.895573     0.147342    -6.08    <1e-08
V             0.0640979    0.124644     0.51    0.6071
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_D_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model


## Direct effect of F on M

adjustmentSets(dag, "F", "M", effect="direct") # { D, H, P, R, S, V }

glmm_F_M = fit(MixedModel, @formula(M ~ 1 + Ḟ + D + H + P + R + S + V + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 M ~ 1 + Ḟ + D + H + P + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.9670   309.9340   327.9340   328.7831   358.5581

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.515804 0.718195
Residual              0.062231 0.249462
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                  Coef.   Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.709957     0.449016     1.58    0.1138
Ḟ             0.0528504    0.0436848    1.21    0.2264
D             0.433761     0.104878     4.14    <1e-04
H            -0.481975     0.185886    -2.59    0.0095
P            -0.0        NaN          NaN       NaN
R             0.314197     0.10863      2.89    0.0038
S            -0.895573     0.147342    -6.08    <1e-08
V             0.0640979    0.124644     0.51    0.6071
──────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_F_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of R on F

adjustmentSets(dag, "R", "F", effect="direct") # { D, H, P, S, V }

glmm_R_F = fit(MixedModel, @formula(Ḟ ~ 1 + R + D + H + P + S + V + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + R + D + H + P + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.5881   405.1762   421.1762   421.8522   448.3976

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.139322 0.373259
Residual              0.256510 0.506468
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                 Coef.   Std. Error       z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   2.46613     0.340223     7.25    <1e-12
R             0.076455    0.149381     0.51    0.6088
D             0.010099    0.0983997    0.10    0.9183
H            -1.66999     0.145672   -11.46    <1e-29
P            -0.0       NaN          NaN       NaN
S            -0.328225    0.103541    -3.17    0.0015
V             0.134809    0.118504     1.14    0.2553
─────────────────────────────────────────────────────


"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_R_F);
coefplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of R on M

adjustmentSets(dag, "R", "M", effect="direct") # { D, F, H, P, S, V }

glmm_R_M = fit(MixedModel, @formula(M ~ 1 + R + D + Ḟ + H + P + S + V + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 M ~ 1 + R + D + Ḟ + H + P + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.9670   309.9340   327.9340   328.7831   358.5581

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.515804 0.718195
Residual              0.062231 0.249462
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                  Coef.   Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.709957     0.449016     1.58    0.1138
R             0.314197     0.10863      2.89    0.0038
D             0.433761     0.104878     4.14    <1e-04
Ḟ             0.0528504    0.0436848    1.21    0.2264
H            -0.481975     0.185886    -2.59    0.0095
P            -0.0        NaN          NaN       NaN
S            -0.895573     0.147342    -6.08    <1e-08
V             0.0640979    0.124644     0.51    0.6071
──────────────────────────────────────────────────────



"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_R_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of S on F

adjustmentSets(dag, "S", "F", effect="direct") # { D, H, P, R, T }

glmm_S_F = fit(MixedModel, @formula(Ḟ ~ 1 + S + D + H + P + R + T + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + S + D + H + P + R + T + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.9501   405.9003   421.9003   422.5763   449.1217

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.147497 0.384053
Residual              0.252925 0.502917
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
────────────────────────────────────────────────────────
                   Coef.    Std. Error       z  Pr(>|z|)
────────────────────────────────────────────────────────
(Intercept)   2.7297        0.276414      9.88    <1e-22
S            -0.330664      0.104798     -3.16    0.0016
D             0.00736893    0.0993668     0.07    0.9409
H            -1.64998       0.146852    -11.24    <1e-28
P            -0.0         NaN           NaN       NaN
R             0.0906866     0.150783      0.60    0.5475
T            -0.00330533    0.00440085   -0.75    0.4526
────────────────────────────────────────────────────────


"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_F);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of S on M

adjustmentSets(dag, "S", "M", effect="direct") # { D, F, H, P, R, V }

glmm_S_M = fit(MixedModel, @formula(M ~ 1 + S + D + Ḟ + H + P + R + V + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 M ~ 1 + S + D + Ḟ + H + P + R + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.9670   309.9340   327.9340   328.7831   358.5581

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.515804 0.718195
Residual              0.062231 0.249462
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                  Coef.   Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.709957     0.449016     1.58    0.1138
S            -0.895573     0.147342    -6.08    <1e-08
D             0.433761     0.104878     4.14    <1e-04
Ḟ             0.0528504    0.0436848    1.21    0.2264
H            -0.481975     0.185886    -2.59    0.0095
P            -0.0        NaN          NaN       NaN
R             0.314197     0.10863      2.89    0.0038
V             0.0640979    0.124644     0.51    0.6071
──────────────────────────────────────────────────────


"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_S_M);
coefplot(boot)
ridgeplot(boot)

# TODO: code Bayesian version of this model

## Direct effect of H on F

adjustmentSets(dag, "H", "F", effect="direct") # { D, P, R, S, V }

glmm_H_F = fit(MixedModel, @formula(Ḟ ~ 1 + H + D + P + R + S + V + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 Ḟ ~ 1 + H + D + P + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -202.5881   405.1762   421.1762   421.8522   448.3976

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.139322 0.373259
Residual              0.256510 0.506468
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
─────────────────────────────────────────────────────
                 Coef.   Std. Error       z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   2.46613     0.340223     7.25    <1e-12
H            -1.66999     0.145672   -11.46    <1e-29
D             0.010099    0.0983997    0.10    0.9183
P            -0.0       NaN          NaN       NaN
R             0.076455    0.149381     0.51    0.6088
S            -0.328225    0.103541    -3.17    0.0015
V             0.134809    0.118504     1.14    0.2553
─────────────────────────────────────────────────────

"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_F);
coefplot(boot)
ridgeplot(boot)

## Direct effect of H on M

adjustmentSets(dag, "H", "M", effect="direct") # { D, F, P, R, S, V }

glmm_H_M = fit(MixedModel, @formula(M ~ 1 + H + D + Ḟ + P + R + S + V + (1 | ID)), dag_df)

"""
Before intervention:
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

Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 M ~ 1 + H + D + Ḟ + P + R + S + V + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -154.9670   309.9340   327.9340   328.7831   358.5581

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.515804 0.718195
Residual              0.062231 0.249462
 Number of obs: 222; levels of grouping factors: 105

  Fixed-effects parameters:
──────────────────────────────────────────────────────
                  Coef.   Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   0.709957     0.449016     1.58    0.1138
H            -0.481975     0.185886    -2.59    0.0095
D             0.433761     0.104878     4.14    <1e-04
Ḟ             0.0528504    0.0436848    1.21    0.2264
P            -0.0        NaN          NaN       NaN
R             0.314197     0.10863      2.89    0.0038
S            -0.895573     0.147342    -6.08    <1e-08
V             0.0640979    0.124644     0.51    0.6071
──────────────────────────────────────────────────────


"""

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_H_M);
coefplot(boot)
ridgeplot(boot)
