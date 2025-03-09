#=
SCM Specification post intervention
- Julia version: 1.11
with - Author: Simon A Babayan
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
@rlibrary dagitty

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

# Set up intervention: post-intervention P, Pn = 0 (no infection)
df.post_P .= 0
df.post_nP .= 0

# Set up dietary intervention: post-intervention D
df.post_D0 .= 0
df.post_D1 .= 1


# restrict to unique cases (no repeated measures):
df_unique = encode_df(df_unique)
df_unique =
    df_unique |>
    @filter(_.days_since_1st_D_or_A ≥ 4) |> # remove entries which were measured less than a week after vaccination
    DataFrame

# Build DAG dataFrame
dag_df = df[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :T, :P, :nP, :ID, :IDidx, :vax_history, :Vidx, :post_P, :post_nP, :post_D0, :post_D1]]
dag_df.lognP = log10.(1 .+ dag_df.nP);
# describe(dag_df)

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

dag_m = dagitty("dag{
D -> E;
D -> F;
D -> M;
D -> R;
F -> E;
F -> M;
H -> E;
H -> F;
H -> M;
H -> R;
M -> E;
P -> E;
P -> F;
P -> M;
R -> E;
R -> F;
R -> M;
S -> E;
S -> F;
S -> M;
S -> R;
V -> E;
V -> F;
V -> M;
V -> R;
}")

## Total effect of P on E among the infected
adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V}
adjustmentSets(dag_m, "P", "E", effect="total") # { }

# Mixed model

glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + R + S + V + (1 | ID)), dag_df_infected)
glmm_P_E_post = fit(MixedModel, @formula(E ~ 1 + post_nP + D + R + S + V + (1 | ID)), dag_df_infected)

"""
Pre-intervention:
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



Post-intervention:
┌ Warning: Fixed-effects matrix is rank deficient
└ @ MixedModels ~/.julia/packages/MixedModels/hs2Ke/src/Xymat.jl:41
Linear mixed model fit by maximum likelihood
 E ~ 1 + post_nP + D + R + S + V + (1 | ID)
	 logLik   -2 logLik     AIC       AICc        BIC
	 -55.4672   110.9344   124.9344   127.5391   138.4572

Variance components:
						Column   Variance Std.Dev.
ID       (Intercept)  0.036547 0.191174
Residual              0.481738 0.694073
 Number of obs: 51; levels of grouping factors: 19

	Fixed-effects parameters:
────────────────────────────────────────────────────
								 Coef.  Std. Error       z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  -1.35493     0.992592   -1.37    0.1722
post_nP      -0.0       NaN         NaN       NaN
D            -0.286635    0.236475   -1.21    0.2255
R            -0.437918    0.271235   -1.61    0.1064
S             0.228331    0.24924     0.92    0.3596
V             1.03056     0.344176    2.99    0.0028
────────────────────────────────────────────────────


"""

# Plot the distribution of the effects of P on E
qqnorm(glmm_P_E; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E);
coefplot(boot)
ridgeplot(boot)

# Post-intervention
qqnorm(glmm_P_E_post; qqline=:fitrobust)
boot = parametricbootstrap(MersenneTwister(42), 3000, glmm_P_E_post);
coefplot(boot)
ridgeplot(boot)

# Difference in E between pre- and post-intervention
dag_df_infected.E_pre = predict(glmm_P_E)
dag_df_infected.E_post = predict(glmm_P_E_post)
dag_df_infected.E_diff = dag_df_infected.E_pre - dag_df_infected.E_post

# Plot E_pre and E_post
function plot_E_pre_post(df; saveplot=false)
    fig = Figure()
    ax = Axis(fig[1, 1])
    hist!(ax, df.E_pre, label="Pre-intervention (observed)")
    hist!(ax, df.E_post, color=:orange, label="Post-intervention (simulated)")
    axislegend(ax, position=:lt)

    # Add labels and title
    ax.xlabel = "Vaccine response"
    ax.ylabel = "Population count"
    # ax.title = "Effect of removing Parasite infection on vaccine response"
    if saveplot
        save("../manuscript/Figures/plots/E_pre_post.pdf", fig)
    end
    fig
end

with_theme(theme_minimal()) do
    plot_E_pre_post(dag_df_infected, saveplot=true)
end


## Total effect of V on E among the infected
adjustmentSets(dag, "V", "E", effect="total") # {}
adjustmentSets(dag_m, "V", "E", effect="total") # { }

# Mixed model

glmm_V_E = fit(MixedModel, @formula(E ~ V + (1 | ID)), dag_df_infected)
glmm_V_E_post = fit(MixedModel, @formula(E ~ V + (1 | ID)), dag_df_infected)


## Plot each mouse before and after intervention as a point, with points of pre and pst intervention linked by an arrow using https://aog.makie.org/stable/ @aog

function plot_post_anthelminthic(df; saveplot=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # Create a scatter plot for each mouse's pre- and post-intervention E values
    for mouse in unique(df.IDidx)
        mouse_data = df[df.IDidx.==mouse, :]
        for i in 1:size(mouse_data, 1)
            if mouse_data.E_pre[i] < mouse_data.E_post[i]
                lines!(ax, [mouse_data.nP[i], mouse_data.nP[i]], [mouse_data.E_pre[i], mouse_data.E_post[i]], color=:orange, linewidth=3, alpha=0.7)
            else
                lines!(ax, [mouse_data.nP[i], mouse_data.nP[i]], [mouse_data.E_pre[i], mouse_data.E_post[i]], color=:black, alpha=0.7)
            end
        end
        scatter!(ax, mouse_data.nP, mouse_data.E_pre, color=:blue, label="Pre-intervention", markersize=14)
        scatter!(ax, mouse_data.nP, mouse_data.E_post, color=:orange, label="Post-intervention", marker=:utriangle, markersize=14)
    end

    # Add labels and title
    ax.xlabel = "Observed Parasite Count"
    ax.ylabel = "Vaccine response"
    # ax.title = "Effect of anthelmintic intervention on vaccine response"
    # Add text box with orange and blue color labels
    text!(ax, "● Pre-intervention (observed)", position=(115, 0.3), color=:blue, fontsize=13, font=:bold)
    text!(ax, "▲ Post-intervention (simulated)", position=(115, 0.2), color=:orange, fontsize=13, font=:bold)
    if saveplot
        save("../manuscript/Figures/plots/post_anthelminthic_effect_on_E.pdf", fig)
    end
    fig
end

with_theme(theme_minimal()) do
    plot_post_anthelminthic(dag_df_infected, saveplot=true)
end

## Association between E_diff and sex, diet, etc.

glmm_E_diff = fit(MixedModel, @formula(E_diff ~ 1 + D + R + S + V + M + Ḟ + (1 | ID)), dag_df_infected)

boot = parametricbootstrap(MersenneTwister(1234), 3000, glmm_E_diff);
coefplot(boot) |> save("../manuscript/Figures/plots/E_diff_association.pdf")
ridgeplot(boot)

## Total effect of D on E, and effect of D=0

adjustmentSets(dag, "D", "E", effect="total") # {H}

## Predict how supplemented individuals would respond to supplementation if they had not been supplemented

glmm_D_E = fit(MixedModel, @formula(E ~ 1 + D + H + (1 | ID)), dag_df)
glmm_D0_E = fit(MixedModel, @formula(E ~ 1 + post_D0 + H + (1 | ID)), dag_df)
glmm_D1_E = fit(MixedModel, @formula(E ~ 1 + post_D1 + H + (1 | ID)), dag_df)


dag_df.E_pre = predict(glmm_D_E)
dag_df.D0_E = predict(glmm_D0_E)
dag_df.D1_E = predict(glmm_D1_E)

# Plot D0_E and D1_E
fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, dag_df.E_pre, color=:blue, label="E_pre")
hist!(ax, dag_df.D0_E, color=:orange, label="D0_E")
hist!(ax, dag_df.D1_E, color=:green, label="D1_E")
axislegend(ax, position=:lt)
fig

# Plot of effect of withdrawing supplementation on E (D=0)

function plot_post_dietary(df; saveplot=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # Create a scatter plot for each mouse's pre- and post-intervention E values for D=0
    for mouse in unique(df.IDidx)
        mouse_data = df[df.IDidx.==mouse, :]
        for i in 1:size(mouse_data, 1)
            lines!(ax, [mouse_data.D0_E[i], mouse_data.E_pre[i]], [mouse_data.D0_E[i], mouse_data.D0_E[i]], color=:blue, linewidth=3)
        end
        scatter!(ax, mouse_data.nP, mouse_data.E_pre, color=:blue, markersize=14, alpha=0.7)
        scatter!(ax, mouse_data.nP, mouse_data.D0_E, color=:green, marker=:utriangle, markersize=14)
    end

    # Add labels and title
    ax.xlabel = "Dietary supplementation"
    ax.ylabel = "Vaccine response"
    ax.title = "Effect of withdrawing dietary supplementation on vaccine response"
    # Add text box with orange and blue color labels
    text!(ax, "• No supplementation", position=(145, -1.5), color=:blue, fontsize=13, font=:bold)
    text!(ax, "• Supplementation", position=(145, -1.6), color=:green, fontsize=13, font=:bold)
    if saveplot
        save("../manuscript/Figures/plots/no_supplementation_on_E.pdf", fig)
    end
    fig
end

with_theme(theme_minimal()) do
    plot_post_dietary(dag_df, saveplot=false)
end
