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

function condR2(x::MixedModel)
    numerator = ( varest(x) + condVar(x) )
    denominator = ( varest(x)  + condVar(x) + var(residuals(x)) )
    conditional_R2 = numerator / denominator
    return conditional_R2
end

condR2(fitmodel)