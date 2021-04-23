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

#generate random normal OD response variable and merge into new dataframe
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

# generate Weight dummy response variable and merge into refute dataframe
rng = MersenneTwister(729302);
randnormalWeight = DataFrame(randn(rng, Float64, (nrow(both), 1)))
rename!(randnormalWeight, :x1 => :randWeight)
refute.randWeight = (randnormalWeight.randWeight)
lm(@formula(randOD ~ randWeight), refute) #checking the random variables are different
refute.commoncause = (commoncause.commoncause)

# generate repro dummy response variable and merge into refute dataframe
rng = MersenneTwister(194740);
randnormalRepro = DataFrame(randn(rng, Float64, (nrow(both), 1)))
rename!(randnormalRepro, :x1 => :randRepro)
refute.randRepro = (randnormalRepro.randRepro)
lm(@formula(randWeight ~ randRepro), refute) #checking the random variables are different

# generate Fat dummy response variable and merge into refute dataframe
rng = MersenneTwister(103840);
randnormalFat = DataFrame(randn(rng, Float64, (nrow(both), 1)))
rename!(randnormalFat, :x1 => :randFat)
refute.randFat = (randnormalFat.randFat)
lm(@formula(randOD ~ randFat), refute) #checking the random variables are different


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

# common cause in all adjustment sets 
# fit weights models (w) and weights models with random normal common cause variable (c)
categorical(both.ID)
w1 = fit(MixedModel, @formula(Weight ~ isrepro + Env + (1 | ID)), both)
c1 = fit(
    MixedModel,
    @formula(Weight ~ isrepro + Env + commoncause + (1 | ID)),
    refute,
)

w2 = fit(MixedModel, @formula(Weight ~ Fat_Scores_Sum + Env + (1 | ID)), both)
c2 = fit(
    MixedModel,
    @formula(Weight ~ Fat_Scores_Sum + Env + commoncause + (1 | ID)),
    refute,
)

w3 = fit(MixedModel, @formula(Weight ~ Sex + Fat_Scores_Sum + (1 | ID)), both)
c3 = fit(
    MixedModel,
    @formula(Weight ~ Sex + Fat_Scores_Sum + commoncause + (1 | ID)),
    refute,
)

w4 = fit(MixedModel, @formula(Weight ~ Diet + Env + (1 | ID)), both)
c4 = fit(
    MixedModel,
    @formula(Weight ~ Diet + Env + commoncause + (1 | ID)),
    refute,
)

w5 = fit(MixedModel, @formula(isrepro ~ Env + (1 | ID)), both)
c5 = fit(MixedModel, @formula(isrepro ~ Env + commoncause + (1 | ID)), refute)

w6 = fit(MixedModel, @formula(Fat_Scores_Sum ~ Env + (1 | ID)), both)
c6 = fit(
    MixedModel,
    @formula(Fat_Scores_Sum ~ Env + commoncause + (1 | ID)),
    refute,
)

w7 = fit(MixedModel, @formula(Fat_Scores_Sum ~ Sex + Env + (1 | ID)), both)
c7 = fit(
    MixedModel,
    @formula(Fat_Scores_Sum ~ Sex + Env + commoncause + (1 | ID)),
    refute,
)

w8 = fit(
    MixedModel,
    @formula(Fat_Scores_Sum ~ days_since_1st_trt + Env + (1 | ID)),
    both,
)
c8 = fit(
    MixedModel,
    @formula(
        Fat_Scores_Sum ~ days_since_1st_trt + Env + commoncause + (1 | ID)
    ),
    refute,
)

w9 = fit(
    MixedModel,
    @formula(OD ~ Diet + Weight + days_since_1st_D_inj + Env + (1 | ID)),
    both,
)
c9 = fit(
    MixedModel,
    @formula(
        OD ~
            Diet + Weight + days_since_1st_D_inj + Env + commoncause + (1 | ID)
    ),
    refute,
)

w10 = fit(
    MixedModel,
    @formula(OD ~ Weight + Diet + Sex + Env + days_since_1st_D_inj + (1 | ID)),
    both,
)
c10 = fit(
    MixedModel,
    @formula(
        OD ~
            Weight +
            Diet +
            Sex +
            Env +
            days_since_1st_D_inj +
            commoncause +
            (1 | ID)
    ),
    refute,
)

w11 = fit(
    MixedModel,
    @formula(OD ~ Sex + Weight + days_since_1st_D_inj + (1 | ID)),
    both,
)
c11 = fit(
    MixedModel,
    @formula(OD ~ Sex + Weight + days_since_1st_D_inj + commoncause + (1 | ID)),
    refute,
)

w12 = fit(
    MixedModel,
    @formula(
        OD ~
            Env + Fat_Scores_Sum + isrepro + days_since_1st_D_inj + (1 | ID)
    ),
    both,
)
c12 = fit(
    MixedModel,
    @formula(
        OD ~
            Env +
            Fat_Scores_Sum +
            isrepro +
            days_since_1st_D_inj +
            commoncause +
            (1 | ID)
    ),
    refute,
)

w13 = fit(
    MixedModel,
    @formula(OD ~ days_since_1st_D_inj + Fat_Scores_Sum + (1 | ID)),
    both,
)
c13 = fit(
    MixedModel,
    @formula(
        OD ~ days_since_1st_D_inj + Fat_Scores_Sum + commoncause + (1 | ID)
    ),
    refute,
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
        "repro -> weight",
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
        "env -> repro",
    ),
)
push!(
    commoncausefits,
    (
        w6.β[2],
        c6.β[2],
        w6.β[2] - c6.β[2],
        abs(((w6.β[2] - c6.β[2]) / w6.β[2]) * 100),
        "env -> fat",
    ),
)
push!(
    commoncausefits,
    (
        w7.β[2],
        c7.β[2],
        w7.β[2] - c7.β[2],
        abs(((w7.β[2] - c7.β[2]) / w7.β[2]) * 100),
        "sex -> fat",
    ),
)
push!(
    commoncausefits,
    (
        w8.β[2],
        c8.β[2],
        w8.β[2] - c8.β[2],
        abs(((w8.β[2] - c8.β[2]) / w8.β[2]) * 100),
        "time -> fat",
    ),
)
push!(
    commoncausefits,
    (
        w9.β[2],
        c9.β[2],
        w9.β[2] - c9.β[2],
        abs(((w9.β[2] - c9.β[2]) / w9.β[2]) * 100),
        "diet -> OD",
    ),
)
push!(
    commoncausefits,
    (
        w10.β[2],
        c10.β[2],
        w10.β[2] - c10.β[2],
        abs(((w10.β[2] - c10.β[2]) / w10.β[2]) * 100),
        "weight -> OD",
    ),
)
push!(
    commoncausefits,
    (
        w11.β[2],
        c11.β[2],
        w11.β[2] - c11.β[2],
        abs(((w11.β[2] - c11.β[2]) / w11.β[2]) * 100),
        "sex -> OD",
    ),
)
push!(
    commoncausefits,
    (
        w12.β[2],
        c12.β[2],
        w12.β[2] - c12.β[2],
        abs(((w12.β[2] - c12.β[2]) / w12.β[2]) * 100),
        "env -> OD",
    ),
)
push!(
    commoncausefits,
    (
        w13.β[2],
        c13.β[2],
        w13.β[2] - c13.β[2],
        abs(((w13.β[2] - c13.β[2]) / w13.β[2]) * 100),
        "time -> OD",
    ),
)

p = plot(
    commoncausefits,
    x = :Edge,
    y = :PcDiff,
    Geom.bar,
    Scale.x_discrete,
    Guide.ylabel("% Difference in Effect Size"),
)

# plot effect size changes
p |> PNG("/Users/ewanwsmith/Downloads/commoncauseplot.png", 6inch, 4inch)

# compare % change in effect size with p-value of original model
commoncausefits.p_value = [w1.pvalues[2],w2.pvalues[2],w3.pvalues[2],w4.pvalues[2],w5.pvalues[2],w6.pvalues[2],w7.pvalues[2],w8.pvalues[2],w9.pvalues[2],w10.pvalues[2],w11.pvalues[2],w12.pvalues[2],w13.pvalues[2],]

@rput commoncausefits
R"cor.test(commoncausefits$p_value, commoncausefits$PcDiff)"
commoncauseps = lm(@formula(PcDiff ~ p_value), commoncausefits)


p2 = plot(
    commoncausefits,
    x = :p_value,
    y = :PcDiff,
    Geom.point,
    slope = [commoncauseps.model.pp.beta0[2]],
    intercept = [commoncauseps.model.pp.beta0[1]],
    Geom.abline(style = :dash, color = "red"),
)


# fit dummy outcome variables
# random normal variable for weight
d1 = fit(
    MixedModel,
    @formula(randWeight ~ isrepro + Env + (1 | ID)),
    refute,
)
d2 = fit(MixedModel, @formula(randWeight ~ Fat_Scores_Sum + Env + (1 | ID)), refute)
d3 = fit(MixedModel, @formula(randWeight ~ Sex + Fat_Scores_Sum + (1 | ID)), refute)
d4 = fit(MixedModel, @formula(randWeight ~ Diet + Env + (1 | ID)), refute)

# random normal variable for repro
d5 = fit(MixedModel, @formula(randRepro ~ Env + (1 | ID)), refute)

# random normal variable for fat scores
d6 = fit(MixedModel, @formula(randFat ~ Env + (1 | ID)), refute)
d7= fit(MixedModel, @formula(randFat ~ Sex + Env + (1 | ID)), refute)
d8 = fit(
    MixedModel,
    @formula(randFat ~ days_since_1st_trt + Env + (1 | ID)),
    refute,
)

# random normal variable for OD
d9 = fit(
    MixedModel,
    @formula(randOD ~ Diet + Weight + days_since_1st_D_inj + Env + (1 | ID)),
    refute,
)
d10 = fit(
    MixedModel,
    @formula(
        randOD ~ Weight + Diet + Sex + Env + days_since_1st_D_inj + (1 | ID)
    ),
    refute,
)
d11 = fit(
    MixedModel,
    @formula(randOD ~ Sex + Weight + days_since_1st_D_inj + (1 | ID)),
    refute,
)
d12 = fit(
    MixedModel,
    @formula(
        randOD ~
            Env + Fat_Scores_Sum + isrepro + days_since_1st_D_inj + (1 | ID)
    ),
    refute,
)
d13 = fit(
    MixedModel,
    @formula(randOD ~ days_since_1st_D_inj + Fat_Scores_Sum + (1 | ID)),
    refute,
)
