using GLM, MixedModels
using Gadfly
using Cairo
using Random
Pkg.add("Missings")
import Missings

# Import & filter data
include("wild_vacc_data_cleanup.jl")

# categorical blocking vector
categorical!(both, :ID)

# generate random normal response variable (dummy response) 
rng = MersenneTwister(820480);
randnormalOD = DataFrame(randn(rng, Float64, (nrow(both), 1)))
rename!(randnormalOD,:x1 => :randOD)
randnormalOD.Row = (both.Row)

# merge dummy response into new dataframe
refute = copy(both; copycols=true)
refute.randOD = (randnormalOD.randOD)

# generate all-zero variable (placebo treatment) 
placebo = DataFrame(Array{Union{Missing, Int}}(missing, nrow(both), 1))
rename!(placebo,:x1 => :placebo)
for col in names(placebo)
   placebo[col] = Missings.coalesce.(placebo[col], 0)
end

# merge placebo treatment into refute dataframe
refute.placebo = placebo.placebo

# generate random normal variable (common cause) and merge into refute dataframe
commoncause = DataFrame(randn(rng, Float64, (nrow(both), 1)))
rename!(commoncause,:x1 => :commoncause)

# merge common cause variable into refute dataframe
refute.commoncause = (commoncause.commoncause)

# random normal variable for OD
fit(
    MixedModel,
    @formula(randOD ~ Diet + Weight + days_since_1st_D_inj + Env + (1 | ID)),
    refute,)
fit(
    MixedModel,
    @formula(
        randOD ~ Weight + Diet + Sex + Env + days_since_1st_D_inj + (1 | ID)),
    refute,)
fit(
    MixedModel,
    @formula(randOD ~ Sex + Weight + days_since_1st_D_inj + (1 | ID)),
    refute,)
fit(
    MixedModel,
    @formula(
        randOD ~
            Env + Fat_Scores_Sum + isrepro + days_since_1st_D_inj + (1 | ID)),
    refute,)
fit(
    MixedModel,
    @formula(randOD ~ days_since_1st_D_inj + Fat_Scores_Sum + (1 | ID)),
    refute,)

# placebo treatment for diet
categorical!(refute, :ID)
fit(MixedModel, @formula(Weight ~ placebo + Env + (1 | ID)), refute)
fit(
    MixedModel,
    @formula(OD ~ placebo + Weight + days_since_1st_D_inj + Env + (1 | ID)),
    refute,)

# placebo treatment for Sex 
fit(MixedModel, @formula(Weight ~ placebo + Fat_Scores_Sum + (1 | ID)), refute)
fit(MixedModel, @formula(Fat_Scores_Sum ~ placebo + Env + (1 | ID)), refute)
fit(
    MixedModel,
    @formula(OD ~ placebo + Weight + days_since_1st_D_inj + (1 | ID)),
    refute,)

# placebo treatment for environment 
fit(MixedModel, @formula(isrepro ~ placebo + (1 | ID)), refute)
fit(MixedModel, @formula(Fat_Scores_Sum ~ placebo + (1 | ID)), refute)
fit(
    MixedModel,
    @formula(
        OD ~
            placebo +
            Fat_Scores_Sum +
            isrepro +
            days_since_1st_D_inj +
            (1 | ID)),
    refute,)
