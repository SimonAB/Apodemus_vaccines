using DataFrames, Query
using CSV
using StatsBase
using GLM, MixedModels
using Gadfly
using Cairo
Pkg.add("RCall")
using RCall

#import data & filter for just lab mice
data = CSV.read(
  "/Users/ewanwsmith/github/Apodemus_vaccines/data/OD_data.csv";
  missingstrings = ["NA"],
  pool = true,
  copycols = true,
)

lab =
  data |>
  @filter(_.:Env == "Lab") |>
  DataFrame

#test independences of current DAG
@rput lab 
t1 = R"cor.test(lab$Weight, lab$days_since_1st_D_inj)"
t2 = R"cor.test(lab$days_since_1st_D_inj, lab$Fat_Scores_Sum)"
t3 = R"t.test(lab$days_since_1st_D_inj ~ lab$Diet)"
t4 = R"t.test(lab$days_since_1st_D_inj ~ lab$Sex)"
t5 = R"t.test(lab$Fat_Scores_Sum ~ lab$Diet)"
t6 = R"chisq.test(lab$Diet, lab$Sex)"

#get model coefficients
w1 = lm(@formula(Weight ~ Sex + Fat_Scores_Sum + Diet), lab)
w2 = lm(@formula(Weight ~ Sex + Diet), lab)
w3 = lm(@formula(Fat_Scores_Sum ~ Diet), lab)
w4 = lm(@formula(Fat_Scores_Sum ~ Sex), lab)
w5 = lm(@formula(OD ~ days_since_1st_D_inj + Sex + Diet), lab)
w6 = lm(@formula(OD ~ Weight + Diet + days_since_1st_D_inj + Sex + Fat_Scores_Sum), lab)
w7  = lm(@formula(OD ~ Fat_Scores_Sum + Diet + Sex + days_since_1st_D_inj), lab)
