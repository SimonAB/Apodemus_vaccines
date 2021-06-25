using trainFrames, Query
using CSV
using StatsBase
using GLM, MixedModels
using Gadfly
using Cairo
using RCall

include(1_train_import_cleanup.jl)

# independence checks


# DAG weights
w1 = fit(MixedModel, @formula(Weight ~ Env + (1 | ID)), train)
w2 = fit(MixedModel, @formula(Weight ~ Fat_Scores_Sum + Env + (1 | ID)), train)
w3 = fit(MixedModel, @formula(Weight ~ Sex + Fat_Scores_Sum + (1 | ID)), train)
w4 = fit(MixedModel, @formula(Weight ~ Diet + Env + (1 | ID)), train)

w5 = fit(MixedModel, @formula(Fat_Scores_Sum ~ Env + (1 | ID)), train)
w6 = fit(MixedModel, @formula(Fat_Scores_Sum ~ Sex + Env + (1 | ID)), train)
w7 = fit(
    MixedModel,
    @formula(Fat_Scores_Sum ~ days_since_1st_trt + Env + (1 | ID)),
    train,
)

w8 = fit(
    MixedModel,
    @formula(logOD ~ Diet + Weight + days_since_1st_D_inj + Env + (1 | ID)),
    train,
)
w9 = fit(
    MixedModel,
    @formula(logOD ~ Weight + Diet + Sex + Env + days_since_1st_D_inj + (1 | ID)),
    train,
)
w10 = fit(
    MixedModel,
    @formula(logOD ~ Sex + Weight + days_since_1st_D_inj + (1 | ID)),
    train,
)
w11 = fit(
    MixedModel,
    @formula(logOD ~ Env + Fat_Scores_Sum + days_since_1st_D_inj + (1 | ID)),
    train,
)
w12 =
    fit(MixedModel, @formula(logOD ~ days_since_1st_D_inj + Env + (1 | ID)), train)


# predict fat scores
train.fat_predict = (
    (w5.β[2] * train.iswild) +
    (w6.β[2] * train.ismale) +
    (w7.β[2] * train.days_since_1st_D_inj) .+
    mean(skipmissing(train.Fat_Scores_Sum))
)

# predict weight
train.Weight_predict = (
    (w1.β[2] * train.iswild) +
    (w2.β[2] * train.Fat_Scores_Sum) +
    (w3.β[2] * train.ismale) +
    (w4.β[2] * train.islow) .+
    mean(train.Weight)
)

# predict OD without intercept
train.OD_predict = (
    (w8.β[2] * train.islow) +
    (w9.β[2] * train.Weight) +
    (w10.β[2] * train.ismale) +
    (w11.β[2] * train.iswild) +
    (w12.β[2] * train.days_since_1st_D_inj) .+ mean(train.logOD)
)

# find observed intercept
intercepttrainmodel = fit(MixedModel, @formula(logOD ~ OD_predict + (1 | ID)), train)

# predict OD with intercept
train.OD_predict_intercept = (train.OD_predict .+ intercepttrainmodel.β[1])
