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
    "./data/markov predicts.csv", DataFrame;
    missingstrings = ["NA"],
    pool = true,
    copycols = true,
)

# categorical blocking vector
markov.ID = categorical(markov.ID)

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

bandsplot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))

push!(
    bandsplot,
    layer(
        valid,
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
        valid,
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

# small test set, so consider these graphs for training set
# filter for train set
both = filter(:ID => in(Set(train)), data)

# categorical blocking vector
categorical!(both, :ID)

OD_predict_train_plot = plot(
    both,
    Geom.point,
    x = :OD_predict_intercept,
    y = :logOD,
    Guide.xlabel("predicted log OD"),
    Guide.ylabel("observed log OD"),
    Geom.abline,
)

fit_train_model = fit(MixedModel, @formula(logOD ~ OD_predict_intercept + (1 | ID)), both)

qq_train = plot(
    y = GLM.residuals(fit_train_model),
    x = Normal(),
    Stat.qq,
    Geom.point,
    Guide.xlabel("theoretical normal quantiles"),
    Guide.ylabel("sample residuals"),
)

bands_train_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))

push!(
    bands_train_plot,
    layer(
        both,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.abline,
        slope = [0.844],
        intercept = [0.021],
    ),
)


push!(
    bands_train_plot,
    layer(
        both,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

residuals_train_plot = plot(
    x = GLM.residuals(fit_train_model),
    Geom.histogram,
    Guide.xlabel("residual size"),
    Guide.ylabel("frequency"),
)

bands_sex_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))

push!(
    bands_sex_plot,
    layer(
        both,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.abline,
        slope = [0.844],
        intercept = [0.021],
        color = :Sex,
    ),
)

push!(
    bands_sex_plot,
    layer(
        both,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

bands_diet_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))

push!(
    bands_diet_plot,
    layer(
        both,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.abline,
        slope = [0.844],
        intercept = [0.021],
        color = :Diet,
    ),
)


push!(
    bands_diet_plot,
    layer(
        both,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

bands_weight_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))

push!(
    bands_weight_plot,
    layer(
        both,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.abline,
        slope = [0.844],
        intercept = [0.021],
        color = :Weight,
    ),
)


push!(
    bands_weight_plot,
    layer(
        both,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

bands_env_plot =
    plot(Guide.xlabel("predicted log OD"), Guide.ylabel("observed log OD"))

push!(
    bands_env_plot,
    layer(
        both,
        Geom.point,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.abline,
        slope = [0.844],
        intercept = [0.021],
        color = :Env,
    ),
)


push!(
    bands_env_plot,
    layer(
        both,
        x = :OD_predict_intercept,
        y = :logOD,
        Geom.ribbon,
        Stat.smooth(method = :lm, levels = [0.95]),
    ),
)

elipseplot = plot(
    both,
    Guide.ylabel("observed log OD"),
    Guide.xlabel("predicted log OD"),
    x = :OD_predict_intercept,
    y = :logOD,
    color = :Env,
    Geom.point,
    Geom.ellipse(fill = true),
    layer(Geom.ellipse(levels = [0.99]), style(line_style = [:dot])),
)

weightODplot = plot(
    both,
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

SexODplot = plot(
    both,
    Geom.boxplot,
    x = :Sex,
    y = :OD_predict_intercept,
    color = :Env,
    Guide.ylabel("predicted log OD"),
)

DietODplot = plot(
    both,
    Geom.boxplot,
    x = :Diet,
    y = :OD_predict_intercept,
    color = :Env,
    Guide.ylabel("predicted log OD"),
)

OD_predict_ID_plot = plot(
    both,
    Geom.point,
    x = :OD_predict_intercept,
    y = :logOD,
    Guide.xlabel("predicted log OD"),
    Guide.ylabel("observed log OD", orientation = :vertical),
    color = :ID,
    Geom.abline,
    Guide.colorkey(title = nothing, labels = nothing, pos = nothing),
)

