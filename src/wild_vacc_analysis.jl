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
categorical!(lab, :ID)
categorical!(vax, :ID)

fit(MixedModel, @formula(Fat_Scores_Sum ~ Sex + (1|ID)), lab)
fit(MixedModel, @formula(Weight ~ Sex + (1|ID)), lab)
fit(MixedModel, @formula(Weight ~ Fat_Scores_Sum + Sex + (1|ID)), lab)
fit(MixedModel, @formula(Weight ~ Diet + (1|ID)), lab)
fit(MixedModel, @formula(OD ~ Diet + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ days_since_1st_D_inj + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Fat_Scores_Sum + (1|ID)), vax)

# combined DAG weights
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

# combined DAG weights
fit(MixedModel, @formula(Weight ~ days_since_1st_trt + (1|ID)), vax)
fit(MixedModel, @formula(Weight ~ Fat_Scores_Sum + days_since_1st_trt + Env + (1|ID)), vax)
fit(MixedModel, @formula(Weight ~ Sex + (1|ID)), vax)
fit(MixedModel, @formula(Weight ~ Diet + Env + (1|ID)), vax)
fit(MixedModel, @formula(Weight ~ Env + (1|ID)), vax)

fit(MixedModel, @formula(Fat_Scores_Sum ~ Env + (1|ID)), vax)
fit(MixedModel, @formula(Fat_Scores_Sum ~ days_since_1st_trt + (1|ID)), vax)

fit(MixedModel, @formula(OD ~ days_since_1st_D_inj + Weight + Fat_Scores_Sum + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Sex + Weight + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Weight + Diet + Fat_Scores_Sum + Env + days_since_1st_D_inj + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Diet + Weight + Env + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Fat_Scores_Sum + Env + days_since_1st_D_inj + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Env + Fat_Scores_Sum + Weight + (1|ID)), vax)

# repro combined DAG weights
categorical!(both, :ID)
categorical!(vax, :ID)
fit(MixedModel, @formula(Weight ~ days_since_1st_trt + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ Fat_Scores_Sum + days_since_1st_trt + Env + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ Sex + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ Diet + Env + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ Env + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ isrepro + Env + (1|ID)), both)


fit(MixedModel, @formula(isrepro ~ Env + (1|ID)), both)

fit(MixedModel, @formula(Fat_Scores_Sum ~ Env + (1|ID)), both)
fit(MixedModel, @formula(Fat_Scores_Sum ~ days_since_1st_trt + (1|ID)), both)

fit(MixedModel, @formula(OD ~ days_since_1st_D_inj + Weight + Fat_Scores_Sum + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Sex + Weight + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Weight + Diet + Fat_Scores_Sum + Env + days_since_1st_D_inj + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Diet + Weight + Env + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Fat_Scores_Sum + Env + days_since_1st_D_inj + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Env + Fat_Scores_Sum + Weight + (1|ID)), vax)



