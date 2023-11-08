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
# using StatisticalRethinking
# using StatisticalRethinkingPlots
using MCMCChains
using Colors
# using KittyTerminalImages # show images in Kitty

cd("./src/")
include("TuringUtils.jl")
include("TuringPlots.jl")

# GLMakie.activate!()
CairoMakie.activate!(; type="svg")

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

vi_form_1 = @formula(E ~ (1 | ID) + vax_history + H + D + vax_history & H)
vi_form_2 = @formula(E ~ (1 | ID) + vax_history + H + D)
# vi_form = @formula(E ~ (1 | ID) + H + S + D + vax_history)
# vi_form = @formula(E ~ (1 | ID) + vax_history + Env + Diet)
mm1 = fit(MixedModel, vi_form_1, df)

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

mm2 = fit(MixedModel, vi_form_2, df)

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

# LRT
MixedModels.likelihoodratiotest(mm1, mm2)

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

## Bayesian model
Ē_ctrl = mean(df[df.vax_history.=="A", :E])

@model function varying_intercept(IDidx, Vidx, H, D, E; n_id=length(unique(IDidx)), Ē_ref=mean(df[df.vax_history.=="A", :E]))
    vax_μ = zeros(length(unique(Vidx)))

    #priors
    α ~ Normal(mean(E), 2.5 * std(E))         # population-level intercept
    βv ~ MvNormal(vax_μ .- Ē_ref, 2)          # population-level slopes relative to adjuvant control
    βh ~ Normal(0, 2)                         # population-level slopes
    βd ~ Normal(0, 2)                         # population-level slopes
    σ ~ Exponential(std(E))                   # residual SD

    #priors for variance of random intercepts
    τ ~ truncated(Cauchy(0, 2); lower=0)      # group-level SDs intercepts
    α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts

    #likelihood
    Ê = @. α + α_ID[IDidx] + βv[Vidx] + βh * H + βd * D
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
┌───────┬───────────────────────────────────────────────────────┐
│ param │   mean    std   5.0 %    50 %  95.0 %       histogram │
├───────┼───────────────────────────────────────────────────────┤
│     α │ -0.116  0.857  -1.523  -0.114   1.291        ▁▁▃██▂▁▁ │
│ βv[1] │  0.062  0.849  -1.343    0.06   1.471  ▁▁▁▁▃▆██▆▄▂▁▁▁ │
│ βv[2] │  1.335  0.847  -0.066    1.34   2.737   ▁▁▁▂▄▇██▅▃▁▁▁ │
│ βv[3] │  0.823  0.849  -0.578   0.823   2.226  ▁▁▁▁▂▄▇██▅▃▁▁▁ │
│ βv[4] │  1.961  0.848   0.567   1.962   3.356  ▁▁▁▂▃▆██▆▃▁▁▁▁ │
│ βv[5] │  2.357   0.85   0.949    2.36   3.763   ▁▁▁▂▄▇██▅▃▁▁▁ │
│    βh │  -0.65  0.083  -0.788   -0.65  -0.513        ▁▁▅█▅▁▁▁ │
│    βd │ -0.265  0.079  -0.394  -0.264  -0.137        ▁▁▁▅█▄▁▁ │
└───────┴───────────────────────────────────────────────────────┘

"""

## Draw figures

set_aog_theme!()

size_inches = (8, 8)
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
params = [:βh, :βd, Symbol("βv[1]"), Symbol("βv[2]"), Symbol("βv[3]"), Symbol("βv[4]"), Symbol("βv[5]")]
n_chains = length(chains(vi_chn))
n_samples = length(vi_chn)

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
        "DD"][i])
    for chain in 1:n_chains
        values = vi_chn[:, param, chain]
        CairoMakie.density!(ax, values,
            color=(:slategray2, 0.7),
            strokewidth=0.1,
            strokecolor=:grey40)
    end
    hideydecorations!(ax; label=false, grid=false)
    if i < length(params)
        hidexdecorations!(ax; grid=false)
    else
        ax.xlabel = "Posterior distribution of effect on DTV-specific IgG1"
    end
    vlines!(ax, 0, color=:grey40, linestyle=:dot)
    xlims!(-4, 6)
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
