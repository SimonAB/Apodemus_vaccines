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

# check the fit for validation set
valid = CSV.File(
  "./data/validation.csv";
  missingstrings = ["NA"],
  pool = true,
#   copycols = true,
) |> DataFrame
valid[!, :ID] = categorical(valid[!, :ID])

lines = CSV.File(
  "./data/lines.csv";
  missingstrings = ["NA"],
  pool = true,
#   copycols = true,
) |> DataFrame


p1 = plot(
    valid,
    y = :OD_predict,
    x = :OD,
    Geom.point,
    Geom.line,
    Guide.xlabel("Observed OD"),
    Guide.ylabel("Predicted OD"),
    color = :ID,
    Geom.abline(style = :dash, color = "red"))

validmodel = fit(MixedModel, @formula(OD ~ OD_predict + (1|ID)), valid)


# filter for just test set
valid = filter(:ID => in(Set(test)), data)

# plot predictions
weight_predict_plot = plot(
    valid,
    Geom.point,
    x = :Weight_predict,
    y = :Weight,
    Guide.xlabel("predicted weight"),
    Guide.ylabel("observed weight"),
)

fat_predict_plot = plot(
    valid,
    Geom.point,
    x = :fat_predict,
    y = :Fat_Scores_Sum,
    Guide.xlabel("predicted fat scores"),
    Guide.ylabel("observed fat scores"),
)

OD_predict_plot = plot(
    valid,
    Geom.point,
    x = :OD_predict,
    y = :OD,
    Guide.xlabel("predicted OD"),
    Guide.ylabel("observed OD"),
)
p2 = plot(
    valid,
    y = :OD_predict,
    x = :OD,
    Geom.point,
    Guide.xlabel("Observed OD"),
    Guide.ylabel("Predicted OD"),
    color = :ID,
    slope = [validmodel.beta[2]],
    intercept = [validmodel.beta[1]],
    Geom.abline(style = :dash, color = "blue"),
)


predictlayer = layer(
    valid,
    x = :days_since_1st_D_inj,
    y = :OD_predict,
    Geom.point,
    Theme(default_color = "red"))
observedlayer = layer(
    valid,
    x = :days_since_1st_D_inj,
    y = :OD,
    Geom.point)
lineslayer = layer(
    lines,
    y = :lines,
    x = :days_since_1st_D_inj,
    Geom.line,
    Theme(default_color = "black"),
    group = :fit)
p3 = plot(valid, predictlayer, observedlayer, lineslayer)

p4 = plot(valid, x = :diff, Geom.hair(orientation=:horizontal), color = :ID)

@rput valid
R"cor.test(valid$abs_diff, valid$OD)"
p5 = plot(valid, y = :abs_diff, x = :OD)

R"cor.test(valid$diff, valid$OD)"
diffODmodel = fit(MixedModel, @formula(diff ~ OD + (1|ID)), valid)
p6 = plot(
    valid,
    y = :diff,
    x = :OD,
    Guide.xlabel("Observed OD"),
    Guide.ylabel("Observed - Predicted OD"),
    Geom.point,
    slope = [diffODmodel.beta[2]],
    intercept = [diffODmodel.beta[1]],
    Geom.abline(style = :dash, color = "blue"),
)

R"cor.test(valid$abs_diff, valid$days_since_1st_D_inj)"
p7 = plot(valid, y = :abs_diff, x = :days_since_1st_D_inj)

R"cor.test(valid$diff, valid$days_since_1st_D_inj)"
p8 = plot(valid, y = :diff, x = :days_since_1st_D_inj)

R"t.test(valid$diff ~ valid$Sex)"
R"t.test(valid$abs_diff ~ valid$Sex)"

R"t.test(valid$diff ~ valid$Diet)"
R"t.test(valid$abs_diff ~ valid$Diet)"

R"t.test(valid$diff ~ valid$Env)"
R"t.test(valid$abs_diff ~ valid$Env)"

R"cor.test(valid$abs_diff, valid$Weight)"
absdiffWeightmodel = fit(MixedModel, @formula(abs_diff ~ Weight + (1|ID)), valid)
p9 = plot(
    valid,
    y = :abs_diff,
    x = :Weight,
    Geom.point,
    slope = [absdiffWeightmodel.beta[2]],
    intercept = [absdiffWeightmodel.beta[1]],
    Geom.abline(style = :dash, color = "blue"),
)
R"cor.test(valid$diff, valid$Weight)"
diffWeightmodel = fit(MixedModel, @formula(diff ~ Weight + (1|ID)), valid)
p10 = plot(
    valid,
    y = :diff,
    x = :Weight,
    Geom.point,
    slope = [diffWeightmodel.beta[2]],
    intercept = [diffWeightmodel.beta[1]],
    Geom.abline(style = :dash, color = "blue"),
)

# import dataset with fits adjusted for removed nodes
markov = CSV.read(
    "./data/markov predicts.csv.csv";
    missingstrings = ["NA"],
    pool = true,
    copycols = true,
)

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

# predict OD
data.OD_predict = (
    (w8.β[2] * data.islow) +
    (w9.β[2] * data.Weight) +
    (w10.β[2] * data.ismale) +
    (w11.β[2] * data.iswild) +
    (w12.β[2] * data.days_since_1st_D_inj) .+
    mean(data.logOD)
)

data.OD_predict_Weight = (
    (w8.β[2] * data.islow) +
    (w9.β[2] * data.Weight_predict) +
    (w10.β[2] * data.ismale) +
    (w11.β[2] * data.iswild) +
    (w12.β[2] * data.days_since_1st_D_inj) .+
    mean(data.OD)
)


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
    x = :OD_predict,
    y = :OD,
    Guide.xlabel("predicted OD"),
    Guide.ylabel("observed OD"),
)

weight_predict_fat_plot = plot(
    valid,
    Geom.point,
    x = :Weight_predict_fat,
    y = :Weight,
    Guide.xlabel("predicted weight"),
    Guide.ylabel("observed weight"),
)

OD_predict_fat_plot = plot(
    valid,
    Geom.point,
    x = :OD_predict_fat,
    y = :OD,
    Guide.xlabel("predicted OD"),
    Guide.ylabel("observed OD"),
)
