#=
MultiLevel Models
- Julia version: 1.11
- Author: Simon A Babayan
=#

#=
This script will fit a multilevel model to the data and generate the figures for the manuscript.
=#

## Import packages

print("Running on ", Threads.nthreads(), " threads.")
# installkernel("Julia", "--project=@. --threads=auto")
using CategoricalArrays, LazyArrays
using DataFrames, CSV
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

# cd("./src/")
include("TuringUtils.jl")
include("TuringPlots.jl")

## Import data
include("DataWrangler.jl")
# All cases - use more efficient filtering with missing value handling
df = encode_df(df) # df includes repeated measures
df = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 1, df)

# Unique cases - use efficient filtering with missing value handling
df_unique = encode_df(df_unique) # df_unique no repeated measures
df_unique = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df_unique)

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
safe_plot_save("E_hist.pdf", f)
f

ExactOneSampleKSTest(seroconv, Normal())


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
mm2 = fit(MixedModel, vi_form_2, df)


# LRT: mm1 vs mm2
MixedModels.likelihoodratiotest(mm1, mm2)
mm3 = fit(MixedModel, vi_form_3, df)

# LRT: mm1 vs mm3
MixedModels.likelihoodratiotest(mm1, mm3)
mm4 = fit(MixedModel, vi_form_4, df)

# LRT: mm2 vs mm4
MixedModels.likelihoodratiotest(mm2, mm4)

# Final model (mm2)
mm = fit(MixedModel, vi_form_2, df)


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
    Ê = @. α + α_ID[IDidx] + βv[Vidx] + βh * H + βd * D + βvh[Vidx] * H
    E ~ MvNormal(Ê, σ^2 * I)

end

vi_model = varying_intercept(df.IDidx, df.Vidx, df.H, df.D, df.E);

vi_chn = sample(vi_model, NUTS(), MCMCThreads(), 3000, 4);

p = plot_chains_df(vi_chn)
safe_plot_save("IgG1_varint.pdf", p)
p

vi_chn_df = DataFrame(vi_chn)[!, r"α\b|β"];
precis(vi_chn_df)

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
    text!(ax, ref_value - 0.3, 0; text="–", color=:grey10, font=:bold)
    text!(ax, ref_value + 0.1, 0; text="+", color=:grey10, font=:bold)
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

safe_plot_save("IgG1.pdf", fig, pt_per_unit=1)
fig
