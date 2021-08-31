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
test.OD_predict = (
    (w8.β[2] * test.islow) +
    (w9.β[2] * test.Weight) +
    (w10.β[2] * test.ismale) +
    (w11.β[2] * test.iswild) +
    (w12.β[2] * test.days_since_1st_D_inj)
    .+ intercepttrainmodel.β[1]
)

fitmodel = fit(MixedModel, @formula(logOD ~ OD_predict + (1 | ID)), test)

# create a function for conditional R2
function condR2(m::MixedModel)
    var_fixed = MixedModels.varest(m)
    var_random = sum(sum(MixedModels.condVar(m)))
    var_error = StatsBase.var(residuals(m))
    conditional_R2 = (var_fixed + var_random) / (var_fixed + var_random + var_error)

    return conditional_R2
end



# predict OD for the entire set
data.OD_predict = (
    (w8.β[2] * data.islow) +
    (w9.β[2] * data.Weight) +
    (w10.β[2] * data.ismale) +
    (w11.β[2] * data.iswild) +
    (w12.β[2] * data.days_since_1st_D_inj)
    .+ intercepttrainmodel.β[1]
)

fitallmodel = fit(MixedModel, @formula(logOD ~ OD_predict + (1 | ID)), data)

condR2(fitallmodel)

condR2(fitmodel)