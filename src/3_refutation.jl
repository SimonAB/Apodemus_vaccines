using GLM, MixedModels
using Gadfly
using Cairo
using Random
import Missings
using Missings

include("1_data_import_cleanup.jl")
include("2_independence_checks_DAG_weights.jl")

# generate random normal  variable
rng = MersenneTwister(141687);
randnormal = DataFrame(randn(rng, Float64, (nrow(train), 1)), :auto)
train.randnormal = randnormal.x1

# generate all-zero variable
placebo = DataFrame(x1=Int64[])
push!(placebo, [0])
repeat!(placebo, nrows(train))
train.placebo = placebo.x1

# common cause in all adjustment sets 
# fit weights models with random normal common cause variable (c)
c1 = fit(MixedModel, @formula(Weight ~ Env + randnormal + (1 | ID)), train)

c2 = fit(
    MixedModel,
    @formula(Weight ~ Fat_Scores_Sum + Env + randnormal + (1 | ID)),
    train,
)

c3 = fit(
    MixedModel,
    @formula(Weight ~ Sex + Fat_Scores_Sum + randnormal + (1 | ID)),
    train,
)

c4 = fit(
    MixedModel,
    @formula(Weight ~ Diet + Env + randnormal + (1 | ID)),
    train,
)

c5 = fit(
    MixedModel,
    @formula(Fat_Scores_Sum ~ Env + randnormal + (1 | ID)),
    train,
)

c6 = fit(
    MixedModel,
    @formula(Fat_Scores_Sum ~ Sex + Env + randnormal + (1 | ID)),
    train,
)

c7 = fit(
    MixedModel,
    @formula(
        Fat_Scores_Sum ~ days_since_1st_trt + Env + randnormal + (1 | ID)
    ),
    train,
)

c8 = fit(
    MixedModel,
    @formula(
        logOD ~
            Diet + Weight + days_since_1st_D_inj + Env + randnormal + (1 | ID)
    ),
    train,
)

c9 = fit(
    MixedModel,
    @formula(
        logOD ~
            Weight +
            Diet +
            Sex +
            Env +
            days_since_1st_D_inj +
            randnormal +
            (1 | ID)
    ),
    train,
)

c10 = fit(
    MixedModel,
    @formula(logOD ~ Sex + Weight + days_since_1st_D_inj + randnormal + (1 | ID)),
    train,
)

c11 = fit(
    MixedModel,
    @formula(
        logOD ~
            Env +
            Fat_Scores_Sum +
            days_since_1st_D_inj +
            randnormal +
            (1 | ID)
    ),
    train,
)

c12 = fit(
    MixedModel,
    @formula(logOD ~ days_since_1st_D_inj + Env + randnormal + (1 | ID)),
    train,
)

# create fits dataframe with percentage difference in weights after common cause added to adjustment set
commoncausefits = DataFrame(
    W=Float64[],
    C=Float64[],
    Diff=Float64[],
    PcDiff=Float64[],
    Edge=String[],
)


push!(
    commoncausefits,
    (
        abs(w1.beta[2]),
        abs(c1.beta[2]),
        abs(w1.beta[2]) - abs(c1.beta[2]),
        (((abs(w1.beta[2]) - abs(c1.beta[2])) / abs(w1.beta[2])) * 100),
        "env -> weight",
    ),
)
push!(
    commoncausefits,
    (
        abs(w2.beta[2]),
        abs(c2.beta[2]),
        abs(w2.beta[2]) - abs(c2.beta[2]),
        (((abs(w2.beta[2]) - abs(c2.beta[2])) / abs(w2.beta[2])) * 100),
        "fat -> weight",
        ),
)
push!(
    commoncausefits,
    (
        abs(w3.beta[2]),
        abs(c3.beta[2]),
        abs(w3.beta[2]) - abs(c3.beta[2]),
        (((abs(w3.beta[2]) - abs(c3.beta[2])) / abs(w3.beta[2])) * 100),
        "sex -> weight",
    ),
)
push!(
    commoncausefits,
    (
        abs(w4.beta[2]),
        abs(c4.beta[2]),
        abs(w4.beta[2]) - abs(c4.beta[2]),
        (((abs(w4.beta[2]) - abs(c4.beta[2])) / abs(w4.beta[2])) * 100),
        "ciet -> weight",
    ),
)
push!(
    commoncausefits,
    (
        abs(w5.beta[2]),
        abs(c5.beta[2]),
        abs(w5.beta[2]) - abs(c5.beta[2]),
        (((abs(w5.beta[2]) - abs(c5.beta[2])) / abs(w5.beta[2])) * 100),
        "emv -> fat",
    ),
)
push!(
    commoncausefits,
    (
        abs(w6.beta[2]),
        abs(c6.beta[2]),
        abs(w6.beta[2]) - abs(c6.beta[2]),
        (((abs(w6.beta[2]) - abs(c6.beta[2])) / abs(w6.beta[2])) * 100),
        "sex -> fat"
    ),
)
push!(
    commoncausefits,
    (
        abs(w7.beta[2]),
        abs(c7.beta[2]),
        abs(w7.beta[2]) - abs(c7.beta[2]),
        (((abs(w7.beta[2]) - abs(c7.beta[2])) / abs(w7.beta[2])) * 100),
        "sex -> fat",
    ),
)
push!(
    commoncausefits,
    (
        abs(w8.beta[2]),
        abs(c8.beta[2]),
        abs(w8.beta[2]) - abs(c8.beta[2]),
        (((abs(w8.beta[2]) - abs(c8.beta[2])) / abs(w8.beta[2])) * 100),
        "ciet -> logOc",
    ),
)
push!(
    commoncausefits,
    (
        abs(w9.beta[2]),
        abs(c9.beta[2]),
        abs(w9.beta[2]) - abs(c9.beta[2]),
        (((abs(w9.beta[2]) - abs(c9.beta[2])) / abs(w9.beta[2])) * 100),
        "weight -> logOc"
    ),
)
push!(
    commoncausefits,
    (
        abs(w10.beta[2]),
        abs(c10.beta[2]),
        abs(w10.beta[2]) - abs(c10.beta[2]),
        (((abs(w10.beta[2]) - abs(c10.beta[2])) / abs(w10.beta[2])) * 100),
        "sex -> logOc",
    ),
)
push!(
    commoncausefits,
    (
        abs(w11.beta[2]),
        abs(c11.beta[2]),
        abs(w11.beta[2]) - abs(c11.beta[2]),
        (((abs(w11.beta[2]) - abs(c11.beta[2])) / abs(w11.beta[2])) * 100),
        "sex -> logOc",
    ),
)
push!(
    commoncausefits,
    (
        abs(w12.beta[2]),
        abs(c12.beta[2]),
        abs(w12.beta[2]) - abs(c12.beta[2]),
        (((abs(w12.beta[2]) - abs(c12.beta[2])) / abs(w12.beta[2])) * 100),
        "time -> logOc",
    ),
)

common_cause_fits_plot = plot(
    commoncausefits,
    x=:Edge,
    y=:PcDiff,
    Geom.bar,
    Scale.x_discrete,
    Guide.ylabel("% Difference in Effect Size"),
)

# compare % change in effect size with p-value of original model
commoncausefits.p_value = [w1.pvalues[2],w2.pvalues[2],w3.pvalues[2],w4.pvalues[2],w9.pvalues[2],w6.pvalues[2],w7.pvalues[2],w8.pvalues[2],w9.pvalues[2],w10.pvalues[2],w11.pvalues[2],w12.pvalues[2],]

@rput commoncausefits
R"cor.test(commoncausefits$p_value, commoncausefits$PcDiff)"
commoncauseps = lm(@formula(PcDiff ~ p_value), commoncausefits)

common_cause_p_plot = plot(
    commoncausefits,
    x=:p_value,
    y=:PcDiff,
    Geom.point,
    slope=[commoncauseps.model.pp.beta0[2]],
    intercept=[commoncauseps.model.pp.beta0[1]],
    Geom.abline(style=:dash, color="red"),
)

# fit dummy outcome variables
# random normal variable for weight
d1 = fit(
    MixedModel,
    @formula(randnormal ~ Env + (1 | ID)),
    train,
)
d2 = fit(
    MixedModel,
    @formula(randnormal ~ Fat_Scores_Sum + Env + (1 | ID)),
    train,
)
d3 = fit(MixedModel, @formula(randnormal ~ Sex + Fat_Scores_Sum + (1 | ID)), train)

d4 = fit(MixedModel, @formula(randnormal ~ Diet + Env + (1 | ID)), train)

# random normal variable for fat scores
d5 = fit(MixedModel, @formula(randnormal ~ Env + (1 | ID)), train)
d6 = fit(MixedModel, @formula(randnormal ~ Sex + Env + (1 | ID)), train)
d7 = fit(
    MixedModel,
    @formula(randnormal ~ days_since_1st_trt + Env + (1 | ID)),
    train,
)

# random normal variable for logOD
d8 = fit(
    MixedModel,
    @formula(randnormal ~ Diet + Weight + days_since_1st_D_inj + Env + (1 | ID)),
    train,
)
d9 = fit(
    MixedModel,
    @formula(
        randnormal ~
            Weight + Diet + Sex + Env + days_since_1st_D_inj + (1 | ID)
    ),
    train,
)
d10 = fit(
    MixedModel,
    @formula(randnormal ~ Sex + Weight + days_since_1st_D_inj + (1 | ID)),
    train,
)
d11 = fit(
    MixedModel,
    @formula(
        randnormal ~
            Env + Fat_Scores_Sum + isrepro + days_since_1st_D_inj + (1 | ID)
    ),
    train,
)
d12 = fit(
    MixedModel,
    @formula(randnormal ~ days_since_1st_D_inj + Fat_Scores_Sum + (1 | ID)),
    train,
)

# create fits dataframe with percentage difference in weights after dummy response variable
dummyfits = DataFrame(
    W=Float64[],
    D=Float64[],
    Diff=Float64[],
    PcDiff=Float64[],
    Edge=String[],
)


push!(
    dummyfits,
    (
        abs(w1.beta[2]),
        abs(d1.beta[2]),
        abs(w1.beta[2]) - abs(d1.beta[2]),
        (((abs(w1.beta[2]) - abs(d1.beta[2])) / abs(w1.beta[2])) * 100),
        "env -> weight",
    ),
)
push!(
    dummyfits,
    (
        abs(w2.beta[2]),
        abs(d2.beta[2]),
        abs(w2.beta[2]) - abs(d2.beta[2]),
        (((abs(w2.beta[2]) - abs(d2.beta[2])) / abs(w2.beta[2])) * 100),
        "fat -> weight",
    ),
)
push!(
    dummyfits,
    (
        abs(w3.beta[2]),
        abs(d3.beta[2]),
        abs(w3.beta[2]) - abs(d3.beta[2]),
        (((abs(w3.beta[2]) - abs(d3.beta[2])) / abs(w3.beta[2])) * 100),
        "sex -> weight",
    ),
)
push!(
    dummyfits,
    (
        abs(w4.beta[2]),
        abs(d4.beta[2]),
        abs(w4.beta[2]) - abs(d4.beta[2]),
        (((abs(w4.beta[2]) - abs(d4.beta[2])) / abs(w4.beta[2])) * 100),
        "diet -> weight",
    ),
)
push!(
    dummyfits,
    (
        abs(w5.beta[2]),
        abs(d5.beta[2]),
        abs(w5.beta[2]) - abs(d5.beta[2]),
        (((abs(w5.beta[2]) - abs(d5.beta[2])) / abs(w5.beta[2])) * 100),
        "emv -> fat",
    ),
)
push!(
    dummyfits,
    (
        abs(w6.beta[2]),
        abs(d6.beta[2]),
        abs(w6.beta[2]) - abs(d6.beta[2]),
        (((abs(w6.beta[2]) - abs(d6.beta[2])) / abs(w6.beta[2])) * 100),
        "sex -> fat"
    ),
)
push!(
    dummyfits,
    (
        abs(w7.beta[2]),
        abs(d7.beta[2]),
        abs(w7.beta[2]) - abs(d7.beta[2]),
        (((abs(w7.beta[2]) - abs(d7.beta[2])) / abs(w7.beta[2])) * 100),
        "sex -> fat",
    ),
)
push!(
    dummyfits,
    (
        abs(w8.beta[2]),
        abs(d8.beta[2]),
        abs(w8.beta[2]) - abs(d8.beta[2]),
        (((abs(w8.beta[2]) - abs(d8.beta[2])) / abs(w8.beta[2])) * 100),
        "diet -> logOD",
    ),
)
push!(
    dummyfits,
    (
        abs(w9.beta[2]),
        abs(d9.beta[2]),
        abs(w9.beta[2]) - abs(d9.beta[2]),
        (((abs(w9.beta[2]) - abs(d9.beta[2])) / abs(w9.beta[2])) * 100),
        "weight -> logOD"
    ),
)
push!(
    dummyfits,
    (
        abs(w10.beta[2]),
        abs(d10.beta[2]),
        abs(w10.beta[2]) - abs(d10.beta[2]),
        (((abs(w10.beta[2]) - abs(d10.beta[2])) / abs(w10.beta[2])) * 100),
        "sex -> logOD",
    ),
)
push!(
    dummyfits,
    (
        abs(w11.beta[2]),
        abs(d11.beta[2]),
        abs(w11.beta[2]) - abs(d11.beta[2]),
        (((abs(w11.beta[2]) - abs(d11.beta[2])) / abs(w11.beta[2])) * 100),
        "sex -> logOD",
    ),
)
push!(
    dummyfits,
    (
        abs(w12.beta[2]),
        abs(d12.beta[2]),
        abs(w12.beta[2]) - abs(d12.beta[2]),
        (((abs(w12.beta[2]) - abs(d12.beta[2])) / abs(w12.beta[2])) * 100),
        "time -> logOD",
    ),
)

dummy_fits_effect_plot = plot(
    dummyfits,
    x=:Edge,
    y=:PcDiff,
    Geom.bar,
    Scale.x_discrete,
    Guide.ylabel("% Change in Effect Size"),
)

# not enough change in effect size on time -> OD edge
# check p values
w12.pvalues[2] 
d12.pvalues[2]
# almost no change in effect size but massive loss in significance


# placebo treatment for diet supplementation
# w4 = diet -> body mass
p4 = fit(MixedModel, @formula(Weight ~ placebo + Env + (1 | ID)), train)
placebo4 = DataFrame(
    Effect_size=Float64[],
    Edge=String[],
)

push!(placebo4,(w4.beta[2], "diet -> body mass"))
push!(placebo4,(p4.beta[2], "placebo -> body mass"))

placebo_4_plot = plot(placebo4, y=:Effect_size, x=:Edge, Geom.bar, color=:Edge)



# w8 = diet -> OD
p8 = fit(
    MixedModel,
    @formula(logOD ~ placebo + Weight + days_since_1st_D_inj + Env + (1 | ID)),
    train,
)
placebo8 = DataFrame(
    Effect_size=Float64[],
    Edge=String[],
)

push!(placebo8,(w8.beta[2], "diet -> body mass"))
push!(placebo8,(p8.beta[2], "placebo -> body mass"))

placebo_8_plot = plot(placebo8, y=:Effect_size, x=:Edge, Geom.bar, color=:Edge)
