#=
MultiLevel Models
- Julia version: 1.9
- Author: Simon A Babayan
- Date: 2022-08-01
=#

## Import packages

print("Running on ", Threads.nthreads(), " threads.")
using CategoricalArrays, LazyArrays
using DataFrames, CSV, Query
using Random
using Statistics, Distributions
using StatsBase, HypothesisTests
using MLDataUtils: shuffleobs, splitobs, rescale! # Functionality for splitting and normalizing the data.
using LinearAlgebra
using MixedModels
using Turing
using AlgebraOfGraphics, CairoMakie
using MCMCChains
using Colors

# GLMakie.activate!()
CairoMakie.activate!(; type="svg")
using MixedModelsMakie

cd("./src/")
include("TuringUtils.jl")
include("TuringPlots.jl")

## import data
include("DataWrangler.jl")
df = encode_df(df) # df includes repeated measures
df =
  df |>
  @filter(_.days_since_1st_D_or_A ≥ 1) |> # remove most recent post vaccination samples
  DataFrame


df_unique = encode_df(df_unique) # df_unique no repeated measures
df_unique =
  df_unique |>
  @filter(_.days_since_1st_D_or_A ≥ 4) |> # remove entries which were measured less than a week after vaccination
  DataFrame

levels!(df.vax_history, ["A", "D", "AD", "DA", "DD"]);
levels(df[!, :vax_history])

# levels!(CategoricalArray(convert(Vector{String}, df.vax_history)), ["A", "D", "AD", "DA", "DD"])
df = sort(df, :vax_history)

df.IDidx = get_idx(:ID, df)[1];

countmap(df.vax_history)

## Check distribution of transformed E
seroconv = standardize(ZScoreTransform, df.E[df[!, :logOD].>0], dims=1);

f = Figure()
hist(f[1, 1], seroconv, bins=10, normalization=:pdf)
save("../manuscript/Figures/plots/E_hist.pdf", f)
f

ExactOneSampleKSTest(seroconv, Normal())
"""
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.0669681

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.2448

Details:
    number of observations:   229

"""

## Difference between boosted, non-boosted, and control mice in lab vs wild

# Wild Boosted vs Lab Boosted
Δm = mean(df.E[(df.vax_history.=="DD").&(df.islab.==1).&(df.isvax.==1)]) - mean(df.E[(df.vax_history.=="DD").&(df.islab.==0).&(df.isvax.==1)]) |> x -> round(x; digits=2);
Δstd_error = sqrt((std(df.E[(df.vax_history.=="DD").&(df.islab.==1).&(df.isvax.==1)])^2 / length(df.E[(df.vax_history.=="DD").&(df.islab.==1).&(df.isvax.==1)])) + (std(df.E[(df.vax_history.=="DD").&(df.islab.==0).&(df.isvax.==1)])^2 / length(df.E[(df.vax_history.=="DD").&(df.islab.==0).&(df.isvax.==1)]))) |> x -> round(x; digits=2);
println("Wild boosted vs. Lab boosted OD: $Δm ± $Δstd_error")

# Wild Non-Boosted vs Lab Non-Boosted
Δm = mean(df.E[(df.vax_history.!="DD").&(df.islab.==1).&(df.isvax.==1)]) - mean(df.E[(df.vax_history.!="DD").&(df.islab.==0).&(df.isvax.==1)]) |> x -> round(x; digits=2);
Δstd_error = sqrt((std(df.E[(df.vax_history.!="DD").&(df.islab.==1).&(df.isvax.==1)])^2 / length(df.E[(df.vax_history.!="DD").&(df.islab.==1).&(df.isvax.==1)])) + (std(df.E[(df.vax_history.!="DD").&(df.islab.==0).&(df.isvax.==1)])^2 / length(df.E[(df.vax_history.!="DD").&(df.islab.==0).&(df.isvax.==1)]))) |> x -> round(x; digits=2);
println("Wild non-boosted vs. Lab non-boosted OD: $Δm ± $Δstd_error")

# Wild vaccinated vs Lab vaccinated
Δm = mean(df.E[(df.islab.==1).&(df.isvax.==1)]) - mean(df.E[(df.islab.==0).&(df.isvax.==1)]) |> x -> round(x; digits=2);
Δstd_error = sqrt((std(df.E[(df.islab.==1).&(df.isvax.==1)])^2 / length(df.E[(df.islab.==1).&(df.isvax.==1)])) + (std(df.E[(df.islab.==0).&(df.isvax.==1)])^2 / length(df.E[(df.islab.==0).&(df.isvax.==1)]))) |> x -> round(x; digits=2);
println("Wild vaccinated vs. Lab vaccinated OD: $Δm ± $Δstd_error")

# Lab Non-Boosted vs Lab Boosted
Δm = mean(df.E[(df.vax_history.=="DD").&(df.islab.==1).&(df.isvax.==1)]) - mean(df.E[(df.vax_history.=="DA").&(df.islab.==1).&(df.isvax.==1)]) |> x -> round(x; digits=2);
Δstd_error = sqrt((std(df.E[(df.vax_history.=="DD").&(df.islab.==1).&(df.isvax.==1)])^2 / length(df.E[(df.vax_history.=="DD").&(df.islab.==1).&(df.isvax.==1)])) + (std(df.E[(df.vax_history.=="DA").&(df.islab.==1).&(df.isvax.==1)])^2 / length(df.E[(df.vax_history.=="DA").&(df.islab.==1).&(df.isvax.==1)]))) |> x -> round(x; digits=2);
println("Lab boosted vs. Lab non-boosted OD: $Δm ± $Δstd_error")

# Wild Non-Boosted vs Wild Boosted
Δm = mean(df.E[(df.vax_history.=="DD").&(df.islab.==0).&(df.isvax.==1)]) - mean(df.E[(df.vax_history.=="DA").&(df.islab.==0).&(df.isvax.==1)]) |> x -> round(x; digits=2);
Δstd_error = sqrt((std(df.E[(df.vax_history.=="DD").&(df.islab.==0).&(df.isvax.==1)])^2 / length(df.E[(df.vax_history.=="DD").&(df.islab.==0).&(df.isvax.==1)])) + (std(df.E[(df.vax_history.=="DA").&(df.islab.==0).&(df.isvax.==1)])^2 / length(df.E[(df.vax_history.=="DA").&(df.islab.==0).&(df.isvax.==1)]))) |> x -> round(x; digits=2);
println("Wild boosted vs. Wild non-boosted OD: $Δm ± $Δstd_error")


# Percentage drop in OD in wild vs lab (original value - new value) / original value * 100%
Δpc = (mean(df.OD[(df.islab.==1).&(df.isvax.==1)]) - mean(df.OD[(df.islab.==0).&(df.isvax.==1)])) / mean(df.OD[(df.islab.==1).&(df.isvax.==1)]) * 100 |> x -> round(x; digits=1); # 47%
Δpc_std_error = sqrt((std(df.OD[(df.islab.==1).&(df.isvax.==1)])^2 / length(df.OD[(df.islab.==1).&(df.isvax.==1)])) + (std(df.OD[(df.islab.==0).&(df.isvax.==1)])^2 / length(df.OD[(df.islab.==0).&(df.isvax.==1)]))) |> x -> round(x; digits=1);
println("Percentage drop in OD in wild vs lab: $Δpc ± $Δpc_std_error")

## Is there a significant interaction between vaccine responses and habitat? (GLMM specification)

vi_form_1 = @formula(E ~ (1 | ID) + vax_history + H + D + vax_history & H + H & D)
vi_form_2 = @formula(E ~ (1 | ID) + vax_history + H + D + vax_history & H)
vi_form_3 = @formula(E ~ (1 | ID) + vax_history + H + D + H & D)
vi_form_4 = @formula(E ~ (1 | ID) + vax_history + H + D)

# Linear mixed model
mm1 = fit(MixedModel, vi_form_1, df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + vax_history + H + D + vax_history & H + H & D + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -256.1625   512.3250   540.3250   541.7683   592.4552

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.023588 0.153583
Residual              0.290597 0.539070
 Number of obs: 306; levels of grouping factors: 111

  Fixed-effects parameters:
────────────────────────────────────────────────────────────
                          Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────────────
(Intercept)          -0.996679     0.408709  -2.44    0.0147
vax_history: D        2.10621      0.311163   6.77    <1e-10
vax_history: AD       1.2898       0.350575   3.68    0.0002
vax_history: DA       3.7257       0.350636  10.63    <1e-25
vax_history: DD       3.06113      0.350444   8.74    <1e-17
H                     0.0515372    0.290924   0.18    0.8594
D                    -0.192551     0.206688  -0.93    0.3515
vax_history: D & H   -0.623578     0.221786  -2.81    0.0049
vax_history: AD & H  -0.395607     0.263034  -1.50    0.1326
vax_history: DA & H  -1.39953      0.252658  -5.54    <1e-07
vax_history: DD & H  -0.569947     0.250667  -2.27    0.0230
H & D                -0.0463013    0.147164  -0.31    0.7530
────────────────────────────────────────────────────────────

"""

mm2 = fit(MixedModel, vi_form_2, df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + vax_history + H + D + vax_history & H + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -256.2118   512.4236   538.4236   539.6702   586.8302

Variance components:
            Column   VarianceStd.Dev.
ID       (Intercept)  0.02395 0.15475
Residual              0.29039 0.53888
 Number of obs: 306; levels of grouping factors: 111

  Fixed-effects parameters:
────────────────────────────────────────────────────────────
                          Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────────────
(Intercept)          -0.903765    0.282284   -3.20    0.0014
vax_history: D        2.1078      0.311322    6.77    <1e-10
vax_history: AD       1.28526     0.350196    3.67    0.0002
vax_history: DA       3.7263      0.35085    10.62    <1e-25
vax_history: DD       3.05852     0.350563    8.72    <1e-17
H                    -0.0183844   0.187321   -0.10    0.9218
D                    -0.253916    0.0689616  -3.68    0.0002
vax_history: D & H   -0.625372    0.221869   -2.82    0.0048
vax_history: AD & H  -0.390852    0.26256    -1.49    0.1366
vax_history: DA & H  -1.40037     0.252815   -5.54    <1e-07
vax_history: DD & H  -0.568815    0.250797   -2.27    0.0233
────────────────────────────────────────────────────────────

"""
# LRT: mm1 vs mm2
MixedModels.likelihoodratiotest(mm1, mm2)

"""
Model Formulae
1: E ~ 1 + vax_history + H + D + vax_history & H + (1 | ID)
2: E ~ 1 + vax_history + H + D + vax_history & H + H & D + (1 | ID)
─────────────────────────────────────────────────
     model-dof  -2 logLik      χ²  χ²-dof  P(>χ²)
─────────────────────────────────────────────────
[1]         13   512.4236
[2]         14   512.3250  0.0986       1  0.7535
─────────────────────────────────────────────────

"""

mm3 = fit(MixedModel, vi_form_3, df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + vax_history + H + D + H & D + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -271.4715   542.9431   562.9431   563.6888   600.1789

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.042669 0.206564
Residual              0.307718 0.554724
 Number of obs: 306; levels of grouping factors: 111

  Fixed-effects parameters:
────────────────────────────────────────────────────────
                      Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────────
(Intercept)      -0.176857     0.370188  -0.48    0.6328
vax_history: D    1.27974      0.111026  11.53    <1e-30
vax_history: AD   0.761144     0.119261   6.38    <1e-09
vax_history: DA   1.90328      0.124943  15.23    <1e-51
vax_history: DD   2.30338      0.124939  18.44    <1e-75
H                -0.560726     0.254118  -2.21    0.0273
D                -0.185484     0.225546  -0.82    0.4109
H & D            -0.0597382    0.159233  -0.38    0.7075
────────────────────────────────────────────────────────
"""


# LRT: mm1 vs mm3
MixedModels.likelihoodratiotest(mm1, mm3)

"""
Model Formulae
1: E ~ 1 + vax_history + H + D + H & D + (1 | ID)
2: E ~ 1 + vax_history + H + D + vax_history & H + H & D + (1 | ID)
──────────────────────────────────────────────────
     model-dof  -2 logLik       χ²  χ²-dof  P(>χ²)
──────────────────────────────────────────────────
[1]         10   542.9431
[2]         14   512.3250  30.6180       4  <1e-05
──────────────────────────────────────────────────

"""


mm4 = fit(MixedModel, vi_form_4, df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + vax_history + H + D + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -271.5416   543.0831   561.0831   561.6913   594.5954

Variance components:
            Column   Variance Std.Dev.
ID       (Intercept)  0.043238 0.207937
Residual              0.307435 0.554468
 Number of obs: 306; levels of grouping factors: 111

  Fixed-effects parameters:
────────────────────────────────────────────────────────
                      Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────────
(Intercept)      -0.0559073   0.182375   -0.31    0.7592
vax_history: D    1.27861     0.111086   11.51    <1e-29
vax_history: AD   0.762977    0.119162    6.40    <1e-09
vax_history: DA   1.90266     0.125024   15.22    <1e-51
vax_history: DD   2.3019      0.124996   18.42    <1e-75
H                -0.651016    0.0817969  -7.96    <1e-14
D                -0.265308    0.0752443  -3.53    0.0004
────────────────────────────────────────────────────────

"""

# LRT: mm2 vs mm4
MixedModels.likelihoodratiotest(mm2, mm4)

"""
Model Formulae
1: E ~ 1 + vax_history + H + D + (1 | ID)
2: E ~ 1 + vax_history + H + D + vax_history & H + (1 | ID)
──────────────────────────────────────────────────
     model-dof  -2 logLik       χ²  χ²-dof  P(>χ²)
──────────────────────────────────────────────────
[1]          9   543.0831
[2]         13   512.4236  30.6595       4  <1e-05
──────────────────────────────────────────────────

"""

# Final model (mm2)
mm = fit(MixedModel, vi_form_2, df)

"""
Linear mixed model fit by maximum likelihood
 E ~ 1 + vax_history + H + D + vax_history & H + (1 | ID)
   logLik   -2 logLik     AIC       AICc        BIC
  -256.2118   512.4236   538.4236   539.6702   586.8302

Variance components:
            Column   VarianceStd.Dev.
ID       (Intercept)  0.02395 0.15475
Residual              0.29039 0.53888
 Number of obs: 306; levels of grouping factors: 111

  Fixed-effects parameters:
────────────────────────────────────────────────────────────
                          Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────────────
(Intercept)          -0.903765    0.282284   -3.20    0.0014
vax_history: D        2.1078      0.311322    6.77    <1e-10
vax_history: AD       1.28526     0.350196    3.67    0.0002
vax_history: DA       3.7263      0.35085    10.62    <1e-25
vax_history: DD       3.05852     0.350563    8.72    <1e-17
H                    -0.0183844   0.187321   -0.10    0.9218
D                    -0.253916    0.0689616  -3.68    0.0002
vax_history: D & H   -0.625372    0.221869   -2.82    0.0048
vax_history: AD & H  -0.390852    0.26256    -1.49    0.1366
vax_history: DA & H  -1.40037     0.252815   -5.54    <1e-07
vax_history: DD & H  -0.568815    0.250797   -2.27    0.0233
────────────────────────────────────────────────────────────

"""


qqnorm(mm; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, mm);
coefplot(mm)
coefplot(boot)
ridgeplot(boot)

## Bayesian model
@model function varying_intercept(IDidx, Vidx, H, D, E)

  n_id = length(unique(IDidx))
  n_vax = length(unique(Vidx))
  Ē = mean(E)

  #priors for fixed effects
  α ~ Normal(Ē, 2.5 * std(E))           # population-level intercept
  βv ~ filldist(Normal(Ē, 2), n_vax)    # population-level slopes relative to adjuvant control
  βh ~ Normal(0, 2)                     # population-level slopes
  βd ~ Normal(0, 2)                     # population-level slopes
  βvh ~ filldist(Normal(Ē, 2), n_vax)   # interaction term between Vidx and H
  σ ~ Exponential(std(E))               # residual SD

  #priors for variance of random intercepts
  τ ~ truncated(Cauchy(0, 2); lower=0)  # group-level SDs intercepts
  α_ID ~ filldist(Normal(0, τ), n_id)   # group-level intercepts

  #likelihood
  Ê = @. α + α_ID[IDidx] + βv[Vidx] + βh * H + βd * D + βvh[Vidx] .* H
  E ~ MvNormal(Ê, σ^2 * I)

end

vi_model = varying_intercept(df.IDidx, df.Vidx, df.H, df.D, df.E);

vi_chn = sample(vi_model, NUTS(), MCMCThreads(), 3000, 4);

p = plot_chains_df(vi_chn)
save("../manuscript/Figures/plots/IgG1_varint.pdf", p)
p

vi_chn_df = DataFrame(vi_chn)[!, r"α\b|β"];
precis(vi_chn_df)

"""
┌────────┬─────────────────────────────────────────────────────────────┐
│  param │    mean      std    5.0 %     50 %   95.0 %       histogram │
│ String │ Float64  Float64  Float64  Float64  Float64          String │
├────────┼─────────────────────────────────────────────────────────────┤
│      α │  -0.165    0.836   -1.532   -0.165     1.21        ▁▁▃█▇▂▁▁ │
│  βv[1] │     0.0    0.853   -2.108     -0.7    0.695  ▁▁▁▁▂▄▇█▇▅▂▁▁▁ │
│  βv[2] │   2.063    0.836   -0.012    1.369    2.722  ▁▁▁▂▄▆█▇▅▃▁▁▁▁ │
│  βv[3] │   1.264    0.854   -0.852    0.569    1.939        ▁▁▁▄█▅▁▁ │
│  βv[4] │   3.652    0.851    1.546    2.961    4.331         ▁▁▃██▃▁ │
│  βv[5] │   2.999    0.847    0.903    2.302    3.687   ▁▁▁▂▄▇█▇▅▂▁▁▁ │
│     βh │  -1.603    0.843   -2.971   -1.611   -0.196        ▁▁▁▆█▄▁▁ │
│     βd │  -0.253    0.071   -0.371   -0.253   -0.135   ▁▁▁▂▅██▅▂▁▁▁▁ │
│ βvh[1] │     0.0    0.858    0.123    1.569    2.955        ▁▁▅█▅▁▁▁ │
│ βvh[2] │  -0.594    0.846   -0.448    0.974    2.331        ▁▁▃██▃▁▁ │
│ βvh[3] │  -0.377    0.853   -0.245    1.189     2.58        ▁▁▂▇█▃▁▁ │
│ βvh[4] │  -1.349    0.853   -1.224    0.217    1.594         ▁▂▇█▄▁▁ │
│ βvh[5] │  -0.527    0.854   -0.411     1.03    2.416        ▁▁▃██▃▁▁ │
└────────┴─────────────────────────────────────────────────────────────┘

"""

## Draw figures

set_aog_theme!()

size_inches = (8, 10)
size_pt = 72 .* size_inches
fig = Figure(resolution=size_pt, fontsize=12)

# Fig A
ax1 = fig[1:3, 1] # = GridLayout() # this is actually a subfigure that will house the axes drawn by AoG
bp = data(df) * visual(BoxPlot; show_outliers=false) * mapping(:vax_history => "Immunisation protocol",
       :OD => "DTV-specific IgG1 (OD)",
       col=:Env,
       color=:Diet,
       dodge=:Diet)
bps = draw!(ax1, bp)
legend!(ax1[1, 3], bps, valign=:top, patchsize=(10, 10))
#
# Fig B
# ax2 = Axis(fig[2,1])
# CairoMakie.density!(ax2, randn(200))
# params = Symbol.(names(vi_chn_df))[2:end]
params = [:βh, :βd, Symbol("βv[1]"), Symbol("βv[2]"), Symbol("βv[3]"), Symbol("βv[4]"), Symbol("βv[5]"), Symbol("βvh[1]"), Symbol("βvh[2]"), Symbol("βvh[3]"), Symbol("βvh[4]"), Symbol("βvh[5]")]
chns_df = DataFrame(vi_chn)[!, [:α, params...]]
n_chains = length(chains(vi_chn))
n_samples = length(vi_chn)

# Set the first level of each categorical factor as reference for the others
ref_values = Dict()
prefixes = unique([split(param, "[")[1] for param in names(chns_df) if contains(param, "[")])
for prefix in prefixes
  ref_param = prefix * "[1]"
  if ref_param in names(chns_df)
    mean_val = mean(chns_df[:, ref_param])
    ref_values[prefix] = mean_val
  end
end
for (j, param) in enumerate(names(chns_df))
  prefix = split(param, "[")[1]
  if haskey(ref_values, prefix)
    chns_df[:, param] .-= ref_values[prefix]
  end
end

# Calculate the global mean of the intercept α
global_mean_alpha = mean(chns_df[:, :α])

# densities
for (i, param) in enumerate(params)
  gl = fig[3+i, 1]
  ax = Axis(gl; ylabel=[
    "Hab",
    "Diet",
    "A",
    "D",
    "AD",
    "DA",
    "DD",
    "A:H",
    "D:H",
    "AD:H",
    "DA:H",
    "DD:H"][i])
  for chain in 1:n_chains
    values = vi_chn[:, param, chain]
    CairoMakie.density!(ax, values,
      color=(:slategray2, 0.7),
      strokewidth=0.1,
      strokecolor=:grey40)
  end

  # Set x-axis limits to align density plots on value 0
  min_value = minimum(minimum(eachcol(chns_df)))
  max_value = maximum(maximum(eachcol(chns_df)))
  prefix = split(string(param), "[")[1]
  ref_value = get(ref_values, prefix, global_mean_alpha)
  xlims!(ax, (min_value, max_value))

  hideydecorations!(ax; label=false, grid=false)
  if i < length(params)
    hidexdecorations!(ax; grid=false)
  else
    hidexdecorations!(ax; grid=false, label=false)
    ax.xlabel = "Posterior distribution of effect on DTV-specific IgG1"
  end
  vlines!(ax, ref_value, color=:grey40, linestyle=:dot)
  text!(ax, ref_value - 0.3, 0; text="–", color=:grey10, weight=:bold)
  text!(ax, ref_value + 0.1, 0; text="+", color=:grey10, weight=:bold)
  xlims!(-6, 6)
end

# Panels
for (label, layout) in zip(["A", "B"], [ax1, fig[4, 1]])
  Label(layout[1, 1, TopLeft()], label,
    fontsize=12,
    font="TeX Gyre Heros Bold",
    padding=(0, 5, 5, 0),
    halign=:right)
end

rowgap!(ax1.layout, 10)
rowsize!(ax1.layout, 2, Relative(0.4))

save("../manuscript/Figures/plots/IgG1.pdf", fig, pt_per_unit=1)
fig
