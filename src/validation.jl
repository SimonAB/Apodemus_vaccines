# Load packages
using DataFrames, Query
using CategoricalArrays
using CSV
using StatsBase
using GLM, MixedModels
using Gadfly
using Cairo
using RCall

include("wild_vacc_data_cleanup.jl")
include("wild_vacc_data_analysis.jl")

# import dataset with fits adjusted for removed nodes
markov = CSV.read(
    "./data/markov predicts.csv.csv";
    missingstrings = ["NA"],
    pool = true,
    copycols = true,
)

# categorical blocking vector
categorical!(markov, :ID)

# fit full markov blanket prediction to observed OD
fitmodel = fit(MixedModel, @formula(OD ~ OD_predict + (1|ID)), markov)

#create DataFrame with AICs from all models
markovAICs = DataFrame(
    Node = String[],
    full_AIC = Float64[],
    dropped_AIC = Float64[],
    delta_AIC = Float64[],
)

# compare fit without time node
no_time_plot = plot(
    markov,
    y = :OD,
    x = :no_time,
    Geom.point,
    Guide.xlabel("Prediction without time node"),
    Guide.ylabel("OD"),
    color = :ID,
)

no_time_fitmodel = fit(MixedModel, @formula(OD ~ no_time + (1|ID)), markov)
push!(
    markovAICs,
    (
        "time",
        aic(fitmodel),
        aic(no_time_fitmodel),
        aic(fitmodel) - aic(no_time_fitmodel),
    ),
)

# compare fit without Env node
no_Env_plot = plot(
    markov,
    y = :OD,
    x = :no_Env,
    Geom.point,
    Guide.xlabel("Prediction without Env node"),
    Guide.ylabel("OD"),
    color = :ID,
)

no_Env_fitmodel = fit(MixedModel, @formula(OD ~ no_Env + (1|ID)), markov)
push!(
    markovAICs,
    (
        "Env",
        aic(fitmodel),
        aic(no_Env_fitmodel),
        aic(fitmodel) - aic(no_Env_fitmodel),
    ),
)

# compare fit without Weight node
no_Weight_plot = plot(
    markov,
    y = :OD,
    x = :no_Weight,
    Geom.point,
    Guide.xlabel("Prediction without Weight node"),
    Guide.ylabel("OD"),
    color = :ID,
)

no_Weight_fitmodel = fit(MixedModel, @formula(OD ~ no_Weight + (1|ID)), markov)
push!(
    markovAICs,
    (
        "Weight",
        aic(fitmodel),
        aic(no_Weight_fitmodel),
        aic(fitmodel) - aic(no_Weight_fitmodel),
    ),
)

# compare fit without Diet node
no_Diet_plot = plot(
    markov,
    y = :OD,
    x = :no_Diet,
    Geom.point,
    Guide.xlabel("Prediction without Diet node"),
    Guide.ylabel("OD"),
    color = :ID,
)

no_Diet_fitmodel = fit(MixedModel, @formula(OD ~ no_Diet + (1|ID)), markov)
push!(
    markovAICs,
    (
        "Diet",
        aic(fitmodel),
        aic(no_Diet_fitmodel),
        aic(fitmodel) - aic(no_Diet_fitmodel),
    ),
)

# compare fit without Sex node
no_Sex_plot = plot(
    markov,
    y = :OD,
    x = :no_Sex,
    Geom.point,
    Guide.xlabel("Prediction without Sex node"),
    Guide.ylabel("OD"),
    color = :ID,
)

no_Sex_fitmodel = fit(MixedModel, @formula(OD ~ no_Sex + (1|ID)), markov)
push!(
    markovAICs,
    (
        "Sex",
        aic(fitmodel),
        aic(no_Sex_fitmodel),
        aic(fitmodel) - aic(no_Sex_fitmodel),
    ),
)

# print AICs
printstyled(markovAICs)

# create iswild
data.iswild = data[:,:islab] .-1
data.iswild = data[:,:iswild] .^2

# create islow
data.islow = data[:,:ishigh] .-1
data.islow = data[:,:islow] .^2

# predict fat scores
data.fat_predict = (
    (w5.β[2] * data.iswild) +
    (w6.β[2] * data.ismale) +
    (w7.β[2] * data.days_since_1st_D_inj) .+
    mean(skipmissing(data.Fat_Scores_Sum))
)

# predict weight
data.Weight_predict = (
    (w1.β[2] * data.iswild) +
    (w2.β[2] * data.Fat_Scores_Sum) +
    (w3.β[2] * data.ismale) +
    (w4.β[2] * data.islow) .+
    mean(data.Weight)
)

# predict OD without intercept
data.OD_predict = (
    (w8.β[2] * data.islow) +
    (w9.β[2] * data.Weight) +
    (w10.β[2] * data.ismale) +
    (w11.β[2] * data.iswild) +
    (w12.β[2] * data.days_since_1st_D_inj) .+ mean(both.logOD)
)

#filter for weights set
train, test = partition(unique(data.ID), 0.9, shuffle = true, rng = 793426)
both = filter(:ID => in(Set(train)), data)

# categorical blocking vector
categorical!(both, :ID)

# find observed intercept 
interceptmodel = fit(MixedModel, @formula(logOD ~ OD_predict + (1 | ID)), both)

# predict OD with intercept
data.OD_predict_intercept = (data.OD_predict .+ interceptmodel.β[1])

# plot predictions
fat_predict_plot = plot(
    valid,
    Geom.point,
    x = :fat_predict,
    y = :Fat_Scores_Sum,
    Guide.xlabel("predicted fat scores"),
    Guide.ylabel("observed fat scores"),
)

weight_predict_plot = plot(
    valid,
    Geom.point,
    x = :Weight_predict,
    y = :Weight,
    Guide.xlabel("predicted weight"),
    Guide.ylabel("observed weight"),
)

OD_predict_plot = plot(
    valid,
    Geom.point,
    x = :OD_predict_intercept,
    y = :logOD,
    Guide.xlabel("predicted log OD"),
    Guide.ylabel("observed log OD"),
    Geom.abline,
)

fitmodel = fit(MixedModel, @formula(logOD ~ OD_predict_intercept + (1 | ID)), valid)

qq = plot(
    y = GLM.residuals(fitmodel),
    x = Normal(),
    Stat.qq,
    Geom.point,
    Guide.xlabel("theoretical normal quantiles"),
    Guide.ylabel("sample residuals"),
)

bandsplot = plot()

push!(
    bandsplot,
    layer(
        valid,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.abline,
slope = coef(fitmodel)[2]
intercept = coef(fitmodel)[1]
    ),
)


push!(
    bandsplot,
    layer(
        valid,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

