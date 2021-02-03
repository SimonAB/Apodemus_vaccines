using DataFrames, Query
using CSV
using StatsBase
using GLM, MixedModels
using Gadfly
using Cairo
# using RCall

# Import & filter data
include("wild_vacc_data_cleanup.jl")

# Randomly subset data
labset = rand(lab.ID, 20)
unique(labset)
sort!(labset)
# chose mice 5,6, 9, 12, 15, 18, 27, 29, 31, 34, 35, and 37

# lab DAG independence tests

@rput lab
R"t.test(lab$age_lab ~ lab$Diet)"
lm(@formula(age_lab ~ OD + Fat_Scores_Sum + days_since_1st_D_inj), vax)
lm(@formula(age_lab ~ OD + Sex + days_since_1st_D_inj), vax)
lm(@formula(age_lab ~ Fat_Scores_Sum + Sex), lab)
lm(@formula(age_lab ~ Weight + Sex), lab)
R"chisq.test(lab$Diet, lab$Sex)"
R"t.test(lab$days_since_1st_trt ~ lab$Diet)"
R"t.test(lab$Fat_Scores_Sum ~ lab$Diet)"
lm(@formula(ismale ~ OD + Fat_Scores_Sum), vax)
R"t.test(lab$days_since_1st_trt ~ lab$Sex)"
lm(@formula(OD ~ Weight + Diet + Fat_Scores_Sum), vax)
R"cor.test(lab$days_since_1st_trt, lab$Fat_Scores_Sum)"
R"cor.test(lab$days_since_1st_trt, lab$Weight)"


# combined DAG independence checks

@rput vax
R"chisq.test(vax$Env, vax$Sex)"
R"t.test(vax$days_since_1st_trt ~ vax$Env)"
lm(@formula(ishigh ~ Fat_Scores_Sum + Env), vax)
R"chisq.test(vax$Diet, vax$Sex)"
R"t.test(vax$days_since_1st_trt ~ vax$Diet)"
R"t.test(vax$Fat_Scores_Sum ~ vax$Sex)"
R"t.test(vax$days_since_1st_D_inj ~ vax$Sex)"
R"t.test(both$Fat_Scores_Sum ~ both$Sex)"
R"t.test(both$days_since_1st_trt ~ both$Sex)"
R"chisq.test(both$Sex, both$Env)"
R"t.test(both$days_since_1st_trt ~ both$Env)"


# repro combined DAG independence checks

@rput both vax
R"chisq.test(both$Env, both$Sex)"
R"t.test(both$days_since_1st_trt ~ both$Env)"
lm(@formula(OD ~ isrepro + Env + Weight + Diet + Fat_Scores_Sum + Sex + days_since_1st_D_inj), vax)
lm(@formula(ishigh ~ Fat_Scores_Sum + Env), both)
R"chisq.test(both$Diet, both$Sex)"
R"t.test(both$days_since_1st_trt ~ both$Diet)"
lm(@formula(ishigh ~ isrepro + Env), both)
R"t.test(both$Fat_Scores_Sum ~ both$Sex)"
lm(@formula(Fat_Scores_Sum ~ isrepro + Env), both)
R"t.test(both$days_since_1st_trt ~ both$Sex)"
R"chisq.test(both$Sex, both$isrepro)"
R"t.test(both$days_since_1st_trt ~ both$isrepro)"


# vax combined DAG independence checks

fit(MixedModel, @formula(islab ~ Weight + Fat_Scores_Sum + isrepro + Sex + (1|ID)), both)
R"chisq.test(both$Env, both$Diet)"
R"chisq.test(both$Env, both$Sex)"
R"t.test(both$days_since_1st_trt ~ both$Env)"
R"chisq.test(both$Env, both$isvax)"
fit(MixedModel, @formula(OD ~ Fat_Scores_Sum + Env + Weight + Diet + Sex + days_since_1st_D_inj + (1|ID)), both)
fit(MixedModel, @formula(OD ~ isrepro + Env + Weight + Diet + Sex + (1|ID)), both)
fit(MixedModel, @formula(OD ~ isrepro + Env + Weight + Diet + Sex + days_since_1st_D_inj + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ days_since_1st_trt + Env + Fat_Scores_Sum + Sex + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ days_since_1st_trt + Fat_Scores_Sum + isrepro + Sex + (1|ID)), both)
R"t.test(both$Weight ~ both$isvax)"
R"t.test(both$Fat_Scores_Sum ~ both$Diet)"
R"t.test(both$days_since_1st_trt ~ both$Diet)"
R"chisq.test(both$Diet, both$isrepro)"
R"chisq.test(both$Diet, both$isvax)"
fit(MixedModel, @formula(Fat_Scores_Sum ~ isrepro + Env + (1|ID)), both)
R"t.test(both$Fat_Scores_Sum ~ both$isvax)"
R"t.test(both$days_since_1st_trt ~ both$Sex)"
R"chisq.test(both$Sex, both$isrepro)"
R"chisq.test(both$Sex, both$isvax)"
R"t.test(both$days_since_1st_D_inj ~ both$isrepro)"
R"t.test(both$days_since_1st_D_inj ~ both$isvax)"
R"chisq.test(both$isrepro, both$isvax)"
