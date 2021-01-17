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


both = raw_data |> @dropna(:ID) |> DataFrame
categorical(both.Env)
p = plot(
    both,
    y = :OD,
    x = :days_since_1st_D_inj,
    Geom.point,
    Guide.xlabel("days since vaccination"),
    Guide.ylabel("antibody OD"), color = :Env,
)
both = raw_data |> @dropna(:ID) |> @filter(_.days_since_1st_D_inj < 40) |> DataFrame
vax = both |> @filter(_.boost == 0) |> @dropna(:days_since_1st_D_inj) |> DataFrame

# combined DAG independence checks
lm(@formula(OD ~ Sex + Env + Weight + Diet + Fat_Scores_Sum + days_since_1st_trt), vax)
lm(@formula(ishigh ~ Fat_Scores_Sum + Env), both)
@rput both
R"chisq.test(both$Diet, both$Sex)"
R"t.test(both$days_since_1st_trt ~ both$Diet)"
R"t.test(both$Fat_Scores_Sum ~ both$Sex)"
R"t.test(both$days_since_1st_trt ~ both$Sex)"
R"chisq.test(both$Sex, both$Env)"
R"t.test(both$days_since_1st_trt ~ both$Env)"


# combined DAG weights
lm(@formula(OD ~ Env), both)
lm(@formula(OD ~ Diet + Env), vax)
lm(@formula(OD ~ Weight + Env + Fat_Scores_Sum), vax)
lm(@formula(OD ~ Fat_Scores_Sum + Env + days_since_1st_D_inj), vax)
lm(@formula(OD ~ days_since_1st_D_inj), vax)
lm(@formula(Weight ~ Fat_Scores_Sum + Sex + Env + days_since_1st_trt), both)
lm(@formula(Weight ~ Diet + Env), both)
lm(@formula(Weight ~ Env), both)
lm(@formula(Weight ~ days_since_1st_trt), both)
lm(@formula(Weight ~ Sex), both)
lm(@formula(Fat_Scores_Sum ~ Sex + Env), both)
lm(@formula(Fat_Scores_Sum ~ days_since_1st_trt), both)
lm(@formula(Fat_Scores_Sum ~ Env), both)
