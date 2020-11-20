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
m1 = lm(@formula(Weight ~ days_since_1st_D_inj), lab)
m2 = lm(@formula(days_since_1st_D_inj ~ Fat_Scores_Sum), lab)
m3 = R"t.test(lab$days_since_1st_D_inj ~ lab$Diet)"
m4 = R"t.test(lab$days_since_1st_D_inj ~ lab$Sex)"
m5 = R"chisq.test(lab$Diet, lab$Sex)"
