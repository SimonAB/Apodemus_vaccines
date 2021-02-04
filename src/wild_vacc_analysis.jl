using GLM, MixedModels
using Gadfly
using Cairo
# Pkg.add("RCall")
# using RCall
using LsqFit

# Import & filter data
include("wild_vacc_data_cleanup.jl")

# DAG weights
fit(MixedModel, @formula(Weight ~ Sex + (1|ID)), lab)
fit(MixedModel, @formula(Weight ~ Fat_Scores_Sum + Sex + (1|ID)), lab)
fit(MixedModel, @formula(Weight ~ Diet + (1|ID)), lab)
fit(MixedModel, @formula(OD ~ Diet + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ days_since_1st_D_inj + (1|ID)), vax)
fit(MixedModel, @formula(OD ~ Fat_Scores_Sum + (1|ID)), vax)

# combined DAG weights
p = plot(
    data,
    y = :OD,
    x = :days_since_1st_D_inj,
    Geom.point,
    Guide.xlabel("days since vaccination"),
    Guide.ylabel("antibody OD"), color = :Env,
)


# combined DAG weights

fit(MixedModel, @formula(Weight ~ isrepro + Env + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ Fat_Scores_Sum + Env + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ Sex + Fat_Scores_Sum + (1|ID)), both)
fit(MixedModel, @formula(Weight ~ Diet + Env + (1|ID)), both)

fit(MixedModel, @formula(isrepro ~ Env + (1|ID)), both)

fit(MixedModel, @formula(Fat_Scores_Sum ~ Env + (1|ID)), both)
fit(MixedModel, @formula(Fat_Scores_Sum ~ Sex + Env + (1|ID)), both)
fit(MixedModel, @formula(Fat_Scores_Sum ~ days_since_1st_trt + Env + (1|ID)), both)

fit(MixedModel, @formula(OD ~ Diet + Weight + days_since_1st_D_inj + Env + (1|ID)), both)
fit(MixedModel, @formula(OD ~ Weight + Diet + Sex + Env + days_since_1st_D_inj + (1|ID)), both)
fit(MixedModel, @formula(OD ~ Sex + Weight + days_since_1st_D_inj + (1|ID)), both)
fit(MixedModel, @formula(OD ~ Env + Fat_Scores_Sum + isrepro + days_since_1st_D_inj + (1|ID)), both)
fit(MixedModel, @formula(OD ~ days_since_1st_D_inj + Fat_Scores_Sum + (1|ID)), both)
fit(MixedModel, @formula(OD ~ isvax + (1 | ID)), vax)
