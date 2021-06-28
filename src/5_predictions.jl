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

# predict OD for the test set
# predict OD without intercept
test.OD_predict = (
    (w8.β[2] * test.islow) +
    (w9.β[2] * test.Weight) +
    (w10.β[2] * test.ismale) +
    (w11.β[2] * test.iswild) +
    (w12.β[2] * test.days_since_1st_D_inj) .+ mean(test.logOD)
)

# find observed intercept
interceptmodel = fit(MixedModel, @formula(logOD ~ OD_predict + (1 | ID)), test)

# predict OD with intercept
test.OD_predict_intercept = (test.OD_predict .+ interceptmodel.β[1])

# plot predictions
OD_predict_plot = plot(
    test,
    Geom.point,
    x = :OD_predict_intercept,
    y = :logOD,
    Guide.xlabel("predicted log OD"),
    Guide.ylabel("observed log OD"),
    Geom.abline,
)

fitmodel = fit(MixedModel, @formula(logOD ~ OD_predict_intercept + (1 | ID)), test)

qq = plot(
    y = GLM.residuals(fitmodel),
    x = Normal(),
    Stat.qq,
    Geom.point,
    Guide.xlabel("theoretical normal quantiles"),
    Guide.ylabel("sample residuals"),
)

bandsplot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))

push!(
    bandsplot,
    layer(
        test,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.abline,
        slope = [0.946],
        intercept = [0.037],
    ),
)


push!(
    bandsplot,
    layer(
        test,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

residualsplot = plot(
    x = GLM.residuals(fitmodel),
    Geom.histogram,
    Guide.xlabel("residual size"),
    Guide.ylabel("frequency"),
)
