using DataFrames, Query
using CategoricalArrays
using CSV
using StatsBase
using Gadfly
using Cairo

include("1_data_import_cleanup.jl")

# boxplot of OD by environment
OD_Env_plot = plot(
    data,
    x = :Env,
    y = :logOD,
    color = :Env,
    Geom.boxplot,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("Environment")
)

OD_Env_plot |> PNG("plots/OD_Env_plot.png")

# boxplot of OD by sex
OD_Sex_plot = plot(
    data,
    x = :Sex,
    y = :logOD,
    color = :Sex,
    Geom.boxplot,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("Sex")
)

OD_Sex_plot |> PNG("plots/OD_Sex_plot.png")

# boxplot of OD by diet
OD_Diet_plot = plot(
    data,
    x = :Diet,
    y = :logOD,
    color = :Diet,
    Geom.boxplot,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("Diet")
)

OD_Diet_plot |> PNG("plots/OD_Diet_plot.png")

# dot plot of OD by body mass
OD_body_mass_plot = plot(
    data,
    x = :Weight,
    y = :logOD,
    Geom.point,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("Body mass (g)"),
    Geom.line,
    Stat.smooth(method = :lm)
)

OD_body_mass_plot |> PNG("plots/OD_body_mass_plot.png")

# dot plot of OD by body mass
OD_time_plot = plot(
    data,
    x = :days_since_1st_D_inj,
    y = :logOD,
    Geom.point,
    color = :Env,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("time since vaccination (days)"),
)

OD_time_plot |> PNG("plots/OD_time_plot.png")


# refutation plots

include("3_refutation.jl")

# placebo treatment for diet
placebo_4_plot = plot(placebo4, y = :Effect_size, x = :Edge, Geom.bar, color = :Edge)
placebo_4_plot |> PNG("plots/placebo_4_plot.png")

placebo_8_plot = plot(placebo8, y = :Effect_size, x = :Edge, Geom.bar, color = :Edge)
placebo_8_plot |> PNG("plots/placebo_8_plot.png")