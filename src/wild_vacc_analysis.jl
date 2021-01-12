using DataFrames, Query
using CSV
using StatsBase
using GLM, MixedModels
using Gadfly
using Cairo
Pkg.add("RCall")
using RCall
Pkg.add("LsqFit")
using LsqFit

#import data & filter for just lab mice
raw_data = CSV.read(
  "/Users/ewanwsmith/Documents/joint_dataset_4analysis.csv";
  missingstrings = ["NA"],
  pool = true,
  copycols = true,
)

lab = raw_data |> @filter(_.Env == "Lab") |> @dropna(:ID) |> DataFrame
vax = lab |> @filter(_.boost == 0) |> @dropna(:days_since_1st_D_inj) |> DataFrame


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


# lab DAG weights
lm(@formula(age_lab ~ Sex), lab)
lm(@formula(Weight ~ Sex), lab)
lm(@formula(Fat_Scores_Sum ~ Sex), lab)
lm(@formula(Weight ~ Fat_Scores_Sum + Sex), lab)
lm(@formula(Weight ~ Diet), lab)
lm(@formula(OD ~ Diet), vax)
lm(@formula(OD ~ Fat_Scores_Sum), vax)
lm(@formula(OD ~ days_since_1st_D_inj), vax)
lm(@formula(age_lab ~ days_since_1st_trt), lab)
