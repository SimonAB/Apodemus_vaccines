using MixedModels
using DataFrames, Query
using CategoricalArrays
using CSV
using StatsBase
using Gadfly
using Cairo

include("1_data_import_cleanup.jl")

## Diet plots 
# boxplot of weight by environment
Weight_Env_plot = plot(
    data,
    x=:Diet,
    y=:Weight,
    color=:Diet,
    Geom.boxplot,
    Guide.ylabel("Body mass (g)"),
    Guide.xlabel("Environment")
)

Weight_Env_plot |> PNG("plots/Weight_Env_plot.png")

# boxplot of fat scores by environment
dorsal_fat_Env_plot = plot(
    data,
    x=:Diet,
    y=:Fat_Scores_Dorsal,
    color=:Diet,
    Geom.boxplot,
    Guide.ylabel("Dorsal fat score"),
    Guide.xlabel("Environment")
)

pelvic_fat_Env_plot = plot(
    data,
    x=:Diet,
    y=:Fat_Scores_Pelvic,
    color=:Diet,
    Geom.boxplot,
    Guide.ylabel("Pelvic fat score"),
    Guide.xlabel("Environment")
)

# Fat_Scores_Env_plot = Compose.hstack(dorsal_fat_Env_plot, pelvic_fat_Env_plot)
# Fat_Scores_Env_plot |> PNG("plots/Fat_Scores_Env_plot.png")

# boxplot of OD by environment
OD_Env_plot = plot(
    data,
    x=:Diet,
    y=:logOD,
    color=:Diet,
    Geom.boxplot,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("Environment")
)

OD_Env_plot |> PNG("plots/OD_Env_plot.png")

## Diet plots
Weight_Env_plot = plot(
    data,
    x=:Diet,
    y=:Weight,
    color=:Diet,
    Geom.boxplot,
    Guide.ylabel("Body mass (g)"),
    Guide.xlabel("Diet")
)

Weight_Env_plot |> PNG("plots/Weight_Env_plot.png")

# boxplot of fat scores by environment
dorsal_fat_Env_plot = plot(
    data,
    x=:Diet,
    y=:Fat_Scores_Dorsal,
    color=:Diet,
    Geom.boxplot,
    Guide.ylabel("Dorsal fat score"),
    Guide.xlabel("Diet")
)

pelvic_fat_Env_plot = plot(
    data,
    x=:Diet,
    y=:Fat_Scores_Pelvic,
    color=:Diet,
    Geom.boxplot,
    Guide.ylabel("Pelvic fat score"),
    Guide.xlabel("Diet")
)

Fat_Scores_Diet_plot = hstack(dorsal_fat_Diet_plot, pelvic_fat_Diet_plot)
Fat_Scores_Diet_plot |> PNG("plots/Fat_Scores_Diet_plot.png")

# boxplot of OD by diet
OD_Diet_plot = plot(
    data,
    x=:Diet,
    y=:logOD,
    color=:Diet,
    Geom.boxplot,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("Diet")
)

OD_Diet_plot |> PNG("plots/OD_Diet_plot.png")

# boxplot of OD by sex
OD_Sex_plot = plot(
    data,
    x=:Sex,
    y=:logOD,
    color=:Sex,
    Geom.boxplot,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("Sex")
)

OD_Sex_plot |> PNG("plots/OD_Sex_plot.png")

# dot plot of OD by body mass
OD_body_mass_plot = plot(
    data,
    x=:Weight,
    y=:logOD,
    Geom.point,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("Body mass (g)"),
    Geom.line,
    Stat.smooth(method=:lm)
)

OD_body_mass_plot |> PNG("plots/OD_body_mass_plot.png")

# dot plot of OD by body mass
OD_time_plot = plot(
    data,
    x=:days_since_1st_D_inj,
    y=:logOD,
    Geom.point,
    color=:Diet,
    Guide.ylabel("log(10) IgG1 OD"),
    Guide.xlabel("time since vaccination (days)"),
)

OD_time_plot |> PNG("plots/OD_time_plot.png")

# refutation plots

include("3_refutation.jl")

# placebo treatment for diet
placebo_4_plot = plot(
    placebo4,
    y=:Effect_size,
    x=:Edge,
    Geom.bar,
    color=:Edge,
    Guide.ylabel("Effect Size"),
    Guide.xlabel(nothing))

placebo_4_plot |> PNG("plots/placebo_4_plot.png")

placebo_8_plot = plot(
    placebo8,
    y=:Effect_size,
    x=:Edge,
    Geom.bar,
    color=:Edge,
    Guide.ylabel("Effect Size"),
    Guide.xlabel(nothing))

placebo_8_plot |> PNG("plots/placebo_8_plot.png")

# dummy outcome treatment
dummy_outcome_plot = plot(
    dummyfits,
    x=:Edge,
    y=:D,
    Geom.bar,
    Guide.ylabel("Effect size after Dummy Outcome treatment", orientation=:vertical),
)

dummy_outcome_plot |> PNG("plots/dummy_outcome_plot.png")

# common cause treatment
common_cause_plot = plot(
    commoncausefits,
    x=:Edge,
    y=:PcDiff,
    Geom.bar,
    Scale.x_discrete,
    Guide.ylabel("% Difference in Effect Size"),
    Guide.xlabel(nothing)
)

common_cause_plot |> PNG("plots/common_cause_plots.png")

# predictions plots

include("5_predictions.jl")

# plot training set for observed intercept
OD_train_intercept_plot = plot(
    train,
    Geom.point,
    x=:OD_predict,
    y=:logOD,
    Guide.xlabel("predicted log(10) IgG1 OD"),
    Guide.ylabel("observed log(10) IgG1 OD"),
    Geom.smooth(method=:lm),
)

OD_train_intercept_plot |> PNG("plots/OD_train_intercept_plot.png")

# plot predictions
OD_predict_plot = plot(
    test,
    Geom.point,
    x=:OD_predict,
    y=:logOD,
    Guide.xlabel("predicted log(10) IgG1 OD"),
    Guide.ylabel("observed log(10) IgG1 OD"),
    Geom.abline,
)

OD_predict_plot |> PNG("plots/OD_predict_plot.png")

# quantile-quantile plot to examine fit
qq = plot(
    y=GLM.residuals(fitallmodel),
    x=Normal(),
    Stat.qq,
    Geom.point,
    Guide.xlabel("theoretical normal quantiles"),
    Guide.ylabel("sample residuals"),
)

qq |> PNG("plots/qq.png")

# plot of predictions with confidence bands
bandsplot =
    plot(Guide.xlabel("predicted log(10) IgG1 OD"), Guide.ylabel("observed log(10) IgG1 OD"))

push!(
    bandsplot,
    layer(
        test,
        Geom.point,
        x=:OD_predict,
        y=:logOD,
    ),
)

push!(
    bandsplot,
    layer(
        test,
        x=:OD_predict,
        y=:logOD,
        Geom.line,
        Stat.smooth(method=:lm, levels=[0.95]),
    )
)

push!(
    bandsplot,
    layer(
        test,
        x=:OD_predict,
        y=:logOD,
        Geom.ribbon,
        Stat.smooth(method=:lm, levels=[0.95]),
    ),
)

bandsplot |> PNG("plots/bandsplot.png")

# plot of residuals
residualsplot = plot(
    x=GLM.residuals(fitallmodel),
    Geom.histogram,
    Guide.xlabel("residual size"),
    Guide.ylabel("frequency"),
)

residualsplot |> PNG("plots/residualsplot.png")

# quantile-quantile plot to examine fit
qq = plot(
    y=GLM.residuals(fitallmodel),
    x=Normal(),
    Stat.qq,
    Geom.point,
    Guide.xlabel("theoretical normal quantiles"),
    Guide.ylabel("sample residuals"),
)

qq |> PNG("plots/qq.png")