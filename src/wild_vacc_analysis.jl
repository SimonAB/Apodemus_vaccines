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
w1 = fit(MixedModel, @formula(Weight ~ Env + (1 | ID)), both)
w2 = fit(MixedModel, @formula(Weight ~ Fat_Scores_Sum + Env + (1 | ID)), both)
w3 = fit(MixedModel, @formula(Weight ~ Sex + Fat_Scores_Sum + (1 | ID)), both)
w4 = fit(MixedModel, @formula(Weight ~ Diet + Env + (1 | ID)), both)

w5 = fit(MixedModel, @formula(Fat_Scores_Sum ~ Env + (1 | ID)), both)
w6 = fit(MixedModel, @formula(Fat_Scores_Sum ~ Sex + Env + (1 | ID)), both)
w7 = fit(
    MixedModel,
    @formula(Fat_Scores_Sum ~ days_since_1st_trt + Env + (1 | ID)),
    both,
)

w8 = fit(
    MixedModel,
    @formula(logOD ~ Diet + Weight + days_since_1st_D_inj + Env + (1 | ID)),
    both,
)
w9 = fit(
    MixedModel,
    @formula(logOD ~ Weight + Diet + Sex + Env + days_since_1st_D_inj + (1 | ID)),
    both,
)
w10 = fit(
    MixedModel,
    @formula(logOD ~ Sex + Weight + days_since_1st_D_inj + (1 | ID)),
    both,
)
w11 = fit(
    MixedModel,
    @formula(logOD ~ Env + Fat_Scores_Sum + days_since_1st_D_inj + (1 | ID)),
    both,
)
w12 =
    fit(MixedModel, @formula(logOD ~ days_since_1st_D_inj + Env + (1 | ID)), both)


