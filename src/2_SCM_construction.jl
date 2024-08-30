#=
SCM construction
- Julia version: 1.10
- Author: Simon A Babayan
- Date: 2022-08-01
=#

## Import packages
print("Running on ", Threads.nthreads(), " threads.")
using CSV, DataFrames
using HypothesisTests
using GLM, MixedModels
using RCall
@rlibrary dagitty # we use the original dagitty from R
# using Dagitty
# plotting & diagnostics
# using StatsPlots

## import data
# cd("./src/")
include("DataWrangler.jl")
df = encode_df(df) # choose between df and df_unique (the latter has no repeated measures)
df_sel =
  df |>
  # @filter(_.vax_history != "DD") |> # keep only mice with a single vaccination or adjuvant
  @filter(_.days_since_1st_D_or_A ≥ 4) |> # remove entries which were measured less than a week after vaccination
  @dropna(:Fat_Scores_Sum) |> # drop missing fat scores (will be imputed in 4_Fat_Scores.jl)
  DataFrame

dag_df = df_sel[!, [:E, :H, :V, :D, :R, :S, :M, :F, :T, :P, :nP, :Vidx, :vax_history, :ID]]
# Convert 1 / 0 to true / false
# dag_df[!, :V] = convert(Vector{Bool}, dag_df[!, :V])
# dag_df[!, :P] = convert(Vector{Bool}, dag_df[!, :P])

# standardize F
dag_df.F = standardize(ZScoreTransform, convert(Vector{Float64}, df_sel.F), dims=1)

describe(dag_df)

# DAG specification
#
dag = dagitty("dag{D -> E;
D -> F;
D -> M;
D -> P;
D -> R;
F -> E;
F -> M;
M -> E;
P -> E;
P -> F;
P -> M;
R -> P;
R -> E;
R -> F;
R -> M;
S -> E;
S -> F;
S -> M;
S -> P;
S -> R;
T -> E;
T -> F;
T -> M;
T -> R;
T -> P;
V -> E;
V -> T;
H -> D;
H -> E;
H -> F;
H -> M;
H -> P;
H -> R;
H -> T}")

impliedCovarianceMatrix(dag)

impliedConditionalIndependencies(dag)
"""
# D _||_ S
# D _||_ T | H
# D _||_ V
# F _||_ V | H, T
# H _||_ S
# H _||_ V
# M _||_ V | H, T
# P _||_ V | H, T
# R _||_ V | H, T
# S _||_ T
# S _||_ V
"""

# rough test of conditional independencies in light of observed data
localTests(dag, data=dag_df[!, [:E, :H, :V, :D, :R, :S, :M, :F, :T, :P, :nP]])

"""
                     estimate   p.value        2.5%      97.5%
D _||_ S        -0.0172266751 0.7987565 -0.14856766 0.11470795
D _||_ T | H     0.0197460565 0.7706040 -0.11251961 0.15132830
D _||_ V        -0.0136431919 0.8399845 -0.14506002 0.11824387
F _||_ V | H, T  0.1096978953 0.1047006 -0.02290590 0.23856108
H _||_ S        -0.0439666120 0.5150038 -0.17463969 0.08821756
H _||_ V         0.1015131793 0.1316953 -0.03056852 0.23015431
M _||_ V | H, T  0.0493261739 0.4670973 -0.08349025 0.18043335
P _||_ V | H, T -0.0383759289 0.5716722 -0.16979502 0.09437505
R _||_ V | H, T -0.0001032651 0.9987863 -0.13237561 0.13217267
S _||_ T         0.0850799776 0.2069056 -0.04712103 0.21438240
S _||_ V        -0.0676627762 0.3159333 -0.19759595 0.06458582
"""


# more sophisticated test of conditional independencies in light of observed data using mixed models
# D _||_ S
form = @formula(D ~ (1 | ID) + (1 | vax_history) + S)
fit(MixedModel, form, dag_df)
"""
  Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)   1.5753      0.148334   10.62    <1e-25
S            -0.0217966   0.0962488  -0.23    0.8208
────────────────────────────────────────────────────

"""

# D _||_ T | H
# lm(@formula(D~T + H), dag_df)
form = @formula(D ~ (1 | ID) + (1 | vax_history) + T + H)
fit(MixedModel, form, dag_df)
"""
──────────────────────────────────────────────────────
                    Coef.  Std. Error      z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   1.49871      0.146024    10.26    <1e-23
T            -0.000446439  0.00161071  -0.28    0.7817
H             0.0382489    0.0986797    0.39    0.6983
──────────────────────────────────────────────────────
"""

# D _||_ V
# lm(@formula(D~V), dag_df)
form = @formula(D ~ (1 | ID) + (1 | vax_history) + V)
fit(MixedModel, form, dag_df)
"""
───────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)
───────────────────────────────────────────────────
(Intercept)   1.77241    0.144087   12.30    <1e-34
V            -0.130845   0.0776232  -1.69    0.0919
───────────────────────────────────────────────────
"""

# F _||_ V | H, T
# lm(@formula(F ~ V + T + H), dag_df)
form = @formula(F ~ (1 | ID) + (1 | vax_history) + V + T + H)
fit(MixedModel, form, dag_df)
"""
──────────────────────────────────────────────────────
                   Coef.  Std. Error       z  Pr(>|z|)
──────────────────────────────────────────────────────
(Intercept)   2.01607     0.257508      7.83    <1e-14
V             0.157994    0.121869      1.30    0.1948
T            -0.00434388  0.00439817   -0.99    0.3233
H            -1.55994     0.111566    -13.98    <1e-43
──────────────────────────────────────────────────────

"""


# H _||_ S
dag_df.Sbinom = dag_df.S .- 1;
# glm(@formula(S~H), dag_df, Binomial())
form = @formula(Sbinom ~ (1 | ID) + (1 | vax_history) + H)
fit(MixedModel, form, dag_df, Bernoulli())

"""
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)  -0.00780427     3.31393  -0.00    0.9981
H            -2.17449        2.47199  -0.88    0.3790
─────────────────────────────────────────────────────

"""
# H _||_ V
dag_df.Vbinom = dag_df.V .- 1;
# glm(@formula(V~H), dag_df, Binomial())
form = @formula(Vbinom ~ (1 | ID) + (1 | vax_history) + H)
fit(MixedModel, form, dag_df, Bernoulli())

"""
────────────────────────────────────────────────
               Coef.  Std. Error     z  Pr(>|z|)
────────────────────────────────────────────────
(Intercept)  2.40954     26.7591  0.09    0.9283
H            4.59369     21.4596  0.21    0.8305
────────────────────────────────────────────────

"""

# M _||_ V | H, T
# lm(@formula(M ~ V + T + H), dag_df)
form = @formula(M ~ (1 | ID) + (1 | vax_history) + V + T + H)
fit(MixedModel, form, dag_df)
"""
─────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────────
(Intercept)   0.366603    0.383416     0.96    0.3390
V            -0.0742001   0.152415    -0.49    0.6264
T             0.00288989  0.00281343   1.03    0.3043
H            -0.259693    0.191162    -1.36    0.1743
─────────────────────────────────────────────────────
"""
#
#
# P _||_ V | H, T
# form = @formula(nP ~ (1|ID) + V + T + H)
# fit(MixedModel, form, dag_df, NegativeBinomial())
glm(@formula(nP ~ V + T + H), dag_df, NegativeBinomial())
"""
──────────────────────────────────────────────────────────────────────────────────────
                     Coef.     Std. Error      z  Pr(>|z|)     Lower 95%     Upper 95%
──────────────────────────────────────────────────────────────────────────────────────
(Intercept)  -29.4711       215.847        -0.14    0.8914  -452.523      393.581
V              0.0677435      0.0410931     1.65    0.0992    -0.0127975    0.148285
T              0.000475571    0.000422646   1.13    0.2605    -0.0003528    0.00130394
H             14.6386       107.923         0.14    0.8921  -196.887      226.164
──────────────────────────────────────────────────────────────────────────────────────
"""


# R _||_ V | H, T
dag_df.Rbinom = dag_df.R .- 1;
form = @formula(Rbinom ~ (1 | ID) + V + T + H)
fit(MixedModel, form, dag_df, Bernoulli())

"""
───────────────────────────────────────────────────────
                   Coef.    Std. Error      z  Pr(>|z|)
───────────────────────────────────────────────────────
(Intercept)  -43.5752     7706.83       -0.01    0.9955
V             -0.884962      1.24859    -0.71    0.4785
T              0.0392297     0.0376262   1.04    0.2971
H             23.0322     3853.41        0.01    0.9952
───────────────────────────────────────────────────────
"""

# S _||_ T
dag_df.Sbinom = dag_df.S .- 1;
# glm(@formula(Sbinom ~ T), dag_df, Bernoulli())
form = @formula(Sbinom ~ (1 | ID) + (1 | vax_history) + T)
fit(MixedModel, form, dag_df, Bernoulli())

"""
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  -4.53965      4.18828   -1.08    0.2784
T             0.0114615    0.186602   0.06    0.9510
────────────────────────────────────────────────────
"""

# S _||_ V
dag_df.Sbinom = dag_df.S .- 1;
# glm(@formula(Sbinom~V), dag_df, Binomial())
form = @formula(Sbinom ~ (1 | ID) + (1 | vax_history) + V)
fit(MixedModel, form, dag_df, Bernoulli())

"""
──────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)
──────────────────────────────────────────────────
(Intercept)  -1.98452     5.08381  -0.39    0.6963
V            -1.61016     3.01747  -0.53    0.5936
──────────────────────────────────────────────────
"""
