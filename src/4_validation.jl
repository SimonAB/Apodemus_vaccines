using DataFrames, Query
using CategoricalArrays
using CSV
using StatsBase
using GLM, MixedModels
using Gadfly
using Cairo
using RCall

include("1_data_import_cleanup.jl")
include("2_independence_checks_DAG_weights.jl")

# plot predicted OD against observed OD against a line of x=y
OD_predict_train_plot = plot(
    train,
    Geom.point,
    x = :OD_predict_intercept,
    y = :logOD,
    Guide.xlabel("predicted log OD"),
    Guide.ylabel("observed log OD"),
    Geom.abline,
)

# compute a model to examine fit
fit_train_model = fit(MixedModel, @formula(logOD ~ OD_predict_intercept + (1 | ID)), train)

# plot q-q plot to examine residual distributiom 
qq_train = plot(
    y = GLM.residuals(fit_train_model),
    x = Normal(),
    Stat.qq,
    Geom.point,
    Guide.xlabel("theoretical normal quantiles"),
    Guide.ylabel("sample residuals"),
)

# plot fit with confidence bands
bands_train_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))
push!(
    bands_train_plot,
    layer(
        train,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
    ),
)
push!(
    bands_train_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.line,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)
push!(
    bands_train_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

# histogram of residuals
residuals_train_plot = plot(
    x = GLM.residuals(fit_train_model),
    Geom.histogram,
    Guide.xlabel("residual size"),
    Guide.ylabel("frequency"),
)

# colour by sex
bands_sex_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))
push!(
    bands_sex_plot,
    layer(
        train,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        color = :Sex,
    ),
)
push!(
    bands_sex_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.line,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)
push!(
    bands_sex_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

#colour by diet
bands_diet_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))
push!(
    bands_diet_plot,
    layer(
        train,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        color = :Diet,
    ),
)
push!(
    bands_diet_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.line,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)
push!(
    bands_diet_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

#colour by weight
bands_weight_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))
push!(
    bands_weight_plot,
    layer(
        train,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        color = :Weight,
    ),
)
push!(
    bands_weight_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.line,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)
push!(
    bands_weight_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

# plot by environment
bands_env_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))
push!(
    bands_env_plot,
    layer(
        train,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        color = :Env,
    ),
)
push!(
    bands_env_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.line,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)
push!(
    bands_env_plot,
    layer(
        train,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

# plot confidence elipses for environment
elipseplot = plot(
    train,
    Guide.ylabel("observed log OD"),
    Guide.xlabel("predicted log OD"),
    x = :OD_predict_intercept,
    y = :logOD,
    color = :Env,
    Geom.point,
    Geom.ellipse(fill = true),
    layer(Geom.ellipse(levels = [0.99]), style(line_style = [:dot])),
)

# plot predicted OD values by predictors for ease of visualisation
#predicted OD by weight
weightODplot = plot(
    train,
    Geom.point,
    x = :Weight,
    y = :OD_predict_intercept,
    color = :Env,
    Guide.ylabel("predicted log OD"),
    layer(
        Geom.line,
        x = :Weight,
        y = :OD_predict_intercept,
        Stat.smooth(method = :lm, levels = [0.95]),
        color = :Env,
    ),
    layer(
        Geom.ribbon,
        x = :Weight,
        y = :OD_predict_intercept,
        Stat.smooth(method = :lm),
        group = :Env,
        color = :Env,
    ),
)

# predicted OD by Sex
SexODplot = plot(
    train,
    Geom.boxplot,
    x = :Sex,
    y = :OD_predict_intercept,
    color = :Env,
    Guide.ylabel("predicted log OD"),
)

# predicted OD by diet
DietODplot = plot(
    train,
    Geom.boxplot,
    x = :Diet,
    y = :OD_predict_intercept,
    color = :Env,
    Guide.ylabel("predicted log OD"),
)

# predicted OD by ID
OD_predict_ID_plot = plot(
    train,
    Geom.point,
    x = :OD_predict_intercept,
    y = :logOD,
    Guide.xlabel("predicted log OD"),
    Guide.ylabel("observed log OD", orientation = :vertical),
    color = :ID,
    Geom.abline,
    Guide.colorkey(title = nothing, labels = nothing, pos = nothing),
)