using DataFrames, Query
using CSV
using StatsBase
using GLM, MixedModels
using Gadfly
using Cairo
using RCall

# check the fit for validation set
valid = CSV.read(
  "./data/validation.csv";
  missingstrings = ["NA"],
  pool = true,
  copycols = true,
)

lines = CSV.read(
  "./data/lines.csv";
  missingstrings = ["NA"],
  pool = true,
  copycols = true,
)

categorical!(valid, :ID)
p1 = plot(
    valid,
    y = :OD_predict,
    x = :OD,
    Geom.point,
    Guide.xlabel("Observed OD"),
    Guide.ylabel("Predicted OD"),
    color = :ID,
    Geom.abline(style = :dash, color = "red"))

validmodel = fit(MixedModel, @formula(OD ~ OD_predict + (1|ID)), valid)
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
    "/Users/ewanwsmith/Downloads/markov_blanket.csv";
    missingstrings = ["NA"],
    pool = true,
    copycols = true,
)
categorical!(markov, :ID)

# fit full markov blanket prediction to observed OD
fitmodel = fit(MixedModel, @formula(OD ~ OD_predict + (1|ID)), markov)

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
