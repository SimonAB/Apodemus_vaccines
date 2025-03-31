#=
SCM Intervention
- Julia version: 1.11
with - Author: Simon A Babayan
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
dag_df = df[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :P, :nP, :ID, :IDidx, :vax_history, :Vidx, :post_P, :post_nP, :post_D0, :post_D1]]
dag_df.lognP = log10.(1 .+ dag_df.nP);
# describe(dag_df)

# select only infected mice
dag_df_infected = dag_df[dag_df.P.==2, :] # select rows of dag_df for which dag_df.P.==2

# DAG specification - this is our graphical causal hypothesis

dag = dagitty("dag{ D -> E; D -> F; D -> M; D -> P; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> P; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; R -> P; S -> E; S -> F; S -> M; S -> P; S -> R; V -> E; V -> F; V -> M; V -> P; V -> R; }")

dag_m = dagitty("dag{ D -> E; D -> F; D -> M; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; S -> E; S -> F; S -> M; S -> R; V -> E; V -> F; V -> M; V -> R; }")

## Total effect of P on E among the infected
adjustmentSets(dag, "P", "E", effect="total") # { D, H, R, S, V}
adjustmentSets(dag_m, "P", "E", effect="total") # { }

# Mixed model

glmm_P_E = fit(MixedModel, @formula(E ~ 1 + lognP + D + R + S + V + (1 | ID)), dag_df_infected)
glmm_P_E_post = fit(MixedModel, @formula(E ~ 1 + post_nP + D + R + S + V + (1 | ID)), dag_df_infected)


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
dag_df_infected.E_diff = dag_df_infected.E_post - dag_df_infected.E_pre

# Plot E_pre and E_post
"""
    plot_E_pre_post(df; saveplot=false)

Create a visualisation comparing pre- and post-intervention vaccine response distributions.

# Arguments
- `df::DataFrame`: DataFrame containing the data with columns:
    - E_pre: Pre-intervention vaccine response
    - E_post: Post-intervention vaccine response
- `saveplot::Bool=false`: If true, saves the plot as a PDF in "../manuscript/Figures/plots/E_pre_post.pdf"

# Details
- Plots histograms of pre- and post-intervention vaccine responses
- Pre-intervention data shown in default colour
- Post-intervention data shown in orange
- Includes a legend in the top-left corner
- Axis labels are bold and sized at 16pt

# Returns
- `Figure`: A CairoMakie figure containing the distribution comparison plot
"""
function plot_E_pre_post(df; saveplot=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # Plot histograms of pre- and post-intervention responses
    hist!(ax, df.E_pre, label="Pre-intervention (observed)")
    hist!(ax, df.E_post, color=:orange, label="Post-intervention (simulated)")
    axislegend(ax, position=:lt)

    # Configure axis labels and styling
    ax.xlabel = "Vaccine response"
    ax.ylabel = "Population count"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    # Save plot if requested
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

"""
    plot_post_anthelminthic(df; saveplot=false)

Create a visualisation of the effect of anthelmintic intervention on vaccine response for each mouse.

# Arguments
- `df::DataFrame`: DataFrame containing the data with columns:
    - IDidx: Mouse identifier
    - nP: Observed parasite count
    - E_pre: Pre-intervention vaccine response
    - E_post: Post-intervention vaccine response
- `saveplot::Bool=false`: If true, saves the plot as a PDF in "../manuscript/Figures/plots/post_anthelminthic_effect_on_E.pdf"

# Details
- Plots pre- and post-intervention vaccine responses for each mouse
- Uses vertical lines to connect pre- and post-intervention points
- Orange lines indicate increased vaccine response after intervention
- Black lines indicate decreased or unchanged vaccine response
- Pre-intervention points shown as blue circles
- Post-intervention points shown as orange triangles
- Includes a legend in the top-right corner

# Returns
- `Figure`: A CairoMakie figure containing the intervention effect plot
"""
function plot_post_anthelminthic(df; saveplot=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # Plot data for each mouse
    for mouse in unique(df.IDidx)
        mouse_data = df[df.IDidx.==mouse, :]
        for i in 1:size(mouse_data, 1)
            # Draw vertical lines connecting pre- and post-intervention points
            if mouse_data.E_pre[i] < mouse_data.E_post[i]
                lines!(ax, [mouse_data.nP[i], mouse_data.nP[i]],
                    [mouse_data.E_pre[i], mouse_data.E_post[i]],
                    color=:orange, linewidth=3, alpha=0.7)
            else
                lines!(ax, [mouse_data.nP[i], mouse_data.nP[i]],
                    [mouse_data.E_pre[i], mouse_data.E_post[i]],
                    color=:black, alpha=0.7)
            end
        end
        # Plot pre- and post-intervention points
        scatter!(ax, mouse_data.nP, mouse_data.E_pre,
            color=:steelblue, label="Pre-intervention",
            markersize=14)
        scatter!(ax, mouse_data.nP, mouse_data.E_post,
            color=:orange, label="Post-intervention",
            marker=:utriangle, markersize=14)
    end

    # Configure axis labels and styling
    ax.xlabel = "Observed Parasite Count"
    ax.ylabel = "Vaccine response"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    # Add legend text
    text!(ax, "● Pre-intervention (observed)",
        position=(115, 0.3), color=:steelblue,
        fontsize=13, font=:bold)
    text!(ax, "▲ Post-intervention (simulated)",
        position=(115, 0.2), color=:orange,
        fontsize=13, font=:bold)

    # Save plot if requested
    if saveplot
        save("../manuscript/Figures/plots/post_anthelminthic_effect_on_E.pdf", fig)
    end

    fig
end

with_theme(theme_minimal()) do
    plot_post_anthelminthic(dag_df_infected, saveplot=true)
end

## Association between E_diff and sex, diet, etc.

# First fit the model with interaction term
glmm_E_diff = fit(MixedModel, @formula(E_diff ~ -1 + D + R + S + V + M + Ḟ + S & R + (1 | ID)), dag_df_infected)

"""
    plot_S_R_interaction(df; saveplot=false)

Create an interaction plot visualising the effect of Sex (S) and Reproductive status (R) on vaccine response change (E_diff).

# Arguments
- `df::DataFrame`: DataFrame containing the data with columns S (Sex), R (Reproductive status), and E_diff (change in vaccine response)
- `saveplot::Bool=false`: If true, saves the plot as a PDF in "../manuscript/Figures/plots/S_R_interaction.pdf"

# Details
- Calculates mean and standard error of the mean (SEM) for each combination of Sex and Reproductive status
- Plots lines connecting means for each Sex with error bars representing SEM
- Uses distinct colours for males (steelblue) and females (crimson)
- X-axis shows Reproductive status (Non-reproductive on left, Reproductive on right)
- Y-axis shows change in vaccine response (E_diff = E_post - E_pre)
- Includes a legend in the top-left corner

# Returns
- `Figure`: A CairoMakie figure containing the interaction plot
"""
function plot_S_R_interaction(df; saveplot=false)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # Get unique values of Sex (S) and Reproductive status (R)
    S_values = unique(df.S)
    R_values = unique(df.R)

    # Calculate means and standard errors for each combination of Sex and Reproductive status
    means = zeros(length(S_values), length(R_values))
    sems = zeros(length(S_values), length(R_values))
    for (i, s) in enumerate(S_values)
        for (j, r) in enumerate(R_values)
            values = df.E_diff[df.S.==s.&&df.R.==r]
            means[i, j] = mean(values)
            sems[i, j] = std(values) / sqrt(length(values))  # Standard error of the mean
        end
    end

    # Define distinct colours for males and females
    male_color = :steelblue
    female_color = :crimson

    # Plot lines and error bars for each Sex
    for (i, s) in enumerate(S_values)
        sex_color = s == 1 ? male_color : female_color
        sex_label = s == 1 ? "Male" : "Female"

        # Plot lines connecting means
        lines!(ax, R_values, means[i, :],
            label=sex_label,
            linewidth=2,
            color=sex_color)

        # Plot points with error bars representing standard error of the mean
        errorbars!(ax, R_values, means[i, :], sems[i, :],
            color=sex_color,
            linewidth=1)
        scatter!(ax, R_values, means[i, :],
            marker=:circle,
            markersize=14,
            color=sex_color)
    end

    # Add axis labels and title
    ax.xlabel = "Reproductive status"
    ax.ylabel = "Change in vaccine response (E_diff)"
    # ax.title = "Interaction between Sex and Reproductive status on vaccine response change"

    # Set x-axis ticks and labels (Non-reproductive on left, Reproductive on right)
    ax.xticks = (R_values, ["Non-reproductive", "Reproductive"])
    ax.xreversed = true  # Invert x-axis to show Non-reproductive on left

    # Make axis labels bigger and bold
    ax.xlabel = "Reproductive status"
    ax.ylabel = "Change in vaccine response (E_diff)"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    # Add legend in top-left corner
    axislegend(ax, position=:lt)

    # Save plot if requested
    if saveplot
        save("../manuscript/Figures/plots/S_R_interaction.pdf", fig)
    end
    fig
end

with_theme(theme_minimal()) do
    plot_S_R_interaction(dag_df_infected, saveplot=true)
end

# Also create the original ridge plot for comparison
boot = parametricbootstrap(MersenneTwister(1234), 10_000, glmm_E_diff)
ridgeplot(boot) |> save("../manuscript/Figures/plots/E_diff_association_ridgeplot.pdf")
