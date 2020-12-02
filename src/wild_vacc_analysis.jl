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

#get model coefficients
w1 = lm(@formula(Weight ~ Sex + Fat_Scores_Sum + Diet), lab)
w2 = lm(@formula(Weight ~ Sex + Diet), lab)
w3 = lm(@formula(Fat_Scores_Sum ~ Diet), lab)
w4 = lm(@formula(Fat_Scores_Sum ~ Sex), lab)
w5 = lm(@formula(OD ~ days_since_1st_D_inj + Sex + Diet), lab)
w6 = lm(@formula(OD ~ Weight + Diet + days_since_1st_D_inj + Sex + Fat_Scores_Sum), lab)
w7  = lm(@formula(OD ~ Fat_Scores_Sum + Diet + Sex + days_since_1st_D_inj), lab)

#wild mice OD over time curves
wild =
  data |>
  @filter(_.:Env == "Wild") |>
  @dropna(:OD) |>
  @dropna(:days_since_1st_D_inj) |>
  DataFrame

@. linear(x, p) = @. (p[1]*x + p[2])
p0 = [0.5, 0.5]
fit = curve_fit(linear, wild.days_since_1st_D_inj, wild.OD, p0)
fit.param
standard_errors(fit)

@. binomial(x, p) = @. (p[1]*x^2 + p[2]*x + p[3])
p0 = [0.5, 0.5, 0.5]
fit = curve_fit(binomial, wild.days_since_1st_D_inj, wild.OD, p0)
fit.param
standard_errors(fit)

@. trinomial(x, p) = @. (p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4])
p0 = [0.5, 0.5, 0.5, 0.5]
fit = curve_fit(trinomial, wild.days_since_1st_D_inj, wild.OD, p0)
fit.param
standard_errors(fit)

#lab mice OD over time curves
lab =
  data |>
  @filter(_.:Env == "Lab") |>
  @dropna(:OD) |>
  @dropna(:days_since_1st_D_inj) |>
  DataFrame

@. linear(x, p) = @. (p[1]*x + p[2])
p0 = [0.5, 0.5]
fit = curve_fit(linear, lab.days_since_1st_D_inj, lab.OD, p0)
fit.param
standard_errors(fit)

@. binomial(x, p) = @. (p[1]*x^2 + p[2]*x + p[3])
p0 = [0.5, 0.5, 0.5]
fit = curve_fit(binomial, lab.days_since_1st_D_inj, lab.OD, p0)
fit.param
standard_errors(fit)

@. trinomial(x, p) = @. (p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4])
p0 = [0.5, 0.5, 0.5, 0.5]
fit = curve_fit(trinomial, lab.days_since_1st_D_inj, lab.OD, p0)
fit.param
standard_errors(fit)

