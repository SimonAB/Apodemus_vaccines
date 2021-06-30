using GLM, MixedModels
using Gadfly
using Cairo
using Random
import Missings
using Missings

include("1_data_import_cleanup.jl")
include("2_independence_checks_DAG_weights.jl")

#generate random normal  variable
randnormal = DataFrame(randn(rng, Float64, (nrow(train), 1)), :auto)
rng = MersenneTwister(365743);
train.randnormal = randnormal.x1

# generate all-zero variable
placebo = DataFrame(x1 = Int64[])
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
        OD ~
            Diet + Weight + days_since_1st_D_inj + Env + randnormal + (1 | ID)
    ),
    train,
)

c9 = fit(
    MixedModel,
    @formula(
        OD ~
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
    @formula(OD ~ Sex + Weight + days_since_1st_D_inj + randnormal + (1 | ID)),
    train,
)

c11 = fit(
    MixedModel,
    @formula(
        OD ~
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
    @formula(OD ~ days_since_1st_D_inj + Env + randnormal + (1 | ID)),
    train,
)

# create fits dataframe with percentage difference in weights after common cause added to adjustment set
commoncausefits = DataFrame(
    W = Float64[],
    C = Float64[],
    Diff = Float64[],
    PcDiff = Float64[],
    Edge = String[],
)


push!(
    commoncausefits,
    (
        w1.β[2],
        c1.β[2],
        w1.β[2] - c1.β[2],
        abs(((w1.β[2] - c1.β[2]) / w1.β[2]) * 100),
        "env -> weight",
    ),
)
push!(
    commoncausefits,
    (
        w2.β[2],
        c2.β[2],
        w2.β[2] - c2.β[2],
        abs(((w2.β[2] - c2.β[2]) / w2.β[2]) * 100),
        "fat -> weight",
    ),
)
push!(
    commoncausefits,
    (
        w3.β[2],
        c3.β[2],
        w3.β[2] - c3.β[2],
        abs(((w3.β[2] - c3.β[2]) / w3.β[2]) * 100),
        "sex -> weight",
    ),
)
push!(
    commoncausefits,
    (
        w4.β[2],
        c4.β[2],
        w4.β[2] - c4.β[2],
        abs(((w4.β[2] - c4.β[2]) / w4.β[2]) * 100),
        "diet -> weight",
    ),
)
push!(
    commoncausefits,
    (
        w5.β[2],
        c5.β[2],
        w5.β[2] - c5.β[2],
        abs(((w5.β[2] - c5.β[2]) / w5.β[2]) * 100),
        "env -> fat",
    ),
)
push!(
    commoncausefits,
    (
        w6.β[2],
        c6.β[2],
        w6.β[2] - c6.β[2],
        abs(((w6.β[2] - c6.β[2]) / w6.β[2]) * 100),
        "sex -> fat",
    ),
)
push!(
    commoncausefits,
    (
        w7.β[2],
        c7.β[2],
        w7.β[2] - c7.β[2],
        abs(((w7.β[2] - c7.β[2]) / w7.β[2]) * 100),
        "time -> fat",
    ),
)
push!(
    commoncausefits,
    (
        w8.β[2],
        c8.β[2],
        w8.β[2] - c8.β[2],
        abs(((w8.β[2] - c8.β[2]) / w8.β[2]) * 100),
        "diet -> OD",
    ),
)
push!(
    commoncausefits,
    (
        w9.β[2],
        c9.β[2],
        w9.β[2] - c9.β[2],
        abs(((w9.β[2] - c9.β[2]) / w9.β[2]) * 100),
        "weight -> OD",
    ),
)
push!(
    commoncausefits,
    (
        w10.β[2],
        c10.β[2],
        w10.β[2] - c10.β[2],
        abs(((w10.β[2] - c10.β[2]) / w10.β[2]) * 100),
        "sex -> OD",
    ),
)
push!(
    commoncausefits,
    (
        w11.β[2],
        c11.β[2],
        w11.β[2] - c11.β[2],
        abs(((w11.β[2] - c11.β[2]) / w11.β[2]) * 100),
        "env -> OD",
    ),
)
push!(
    commoncausefits,
    (
        w12.β[2],
        c12.β[2],
        w12.β[2] - c12.β[2],
        abs(((w12.β[2] - c12.β[2]) / w12.β[2]) * 100),
        "time -> OD",
    ),
)

common_cause_fits_plot = plot(
    commoncausefits,
    x = :Edge,
    y = :PcDiff,
    Geom.bar,
    Scale.x_discrete,
    Guide.ylabel("% Difference in Effect Size"),
)

# compare % change in effect size with p-value of original model
commoncausefits.p_value = [w1.pvalues[2],w2.pvalues[2],w3.pvalues[2],w4.pvalues[2],w5.pvalues[2],w6.pvalues[2],w7.pvalues[2],w8.pvalues[2],w9.pvalues[2],w10.pvalues[2],w11.pvalues[2],w12.pvalues[2],]

@rput commoncausefits
R"cor.test(commoncausefits$p_value, commoncausefits$PcDiff)"
commoncauseps = lm(@formula(PcDiff ~ p_value), commoncausefits)

common_cause_p_plot = plot(
    commoncausefits,
    x = :p_value,
    y = :PcDiff,
    Geom.point,
    slope = [commoncauseps.model.pp.beta0[2]],
    intercept = [commoncauseps.model.pp.beta0[1]],
    Geom.abline(style = :dash, color = "red"),
)




