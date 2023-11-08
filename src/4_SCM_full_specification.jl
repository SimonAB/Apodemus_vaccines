#=
Full SCM specification
- Julia version: 1.9
- Author: Simon A Babayan
- Date: 2022-08-01
=#

## Start REPL

"""
# Run this in terminal (ctrl-I, then ctrl-\)
julia  --threads auto --project=.
"""
## Import packages

# Data handling
using CSV, DataFrames, Query
# Stats
using HypothesisTests
using Distributions
# modelling
using LazyArrays
using LinearAlgebra: I
using ReverseDiff
using MCMCChains
using StatisticalRethinking
using Turing
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

using RCall
@rlibrary dagitty # we use the original dagitty from R
# using Dagitty
# using StructuralCausalModels

# plotting & diagnostics
using GLMakie
GLMakie.activate!()
# using Plots, StatsPlots, PlotThemes, StatisticalRethinkingPlots

# include modules
cd("./src/")
include("TuringUtils.jl")
include("TuringPlots.jl")

## DAG specification
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

## import data
include("DataWrangler.jl")

# all cases
df = encode_df(df) # choose between df and df_unique (the latter has no repeated measures)
df =
    df |>
    # @filter(_.vax_history != "DD") |> # keep only mice with a single vaccination or adjuvant
    @filter(_.days_since_1st_D_or_A ≥ 4) |> # remove entries which were measured less than a week after vaccination
    DataFrame
df.IDidx = get_idx(:ID, df)[1]

# restrict to unique cases (no repeated measures):
df_unique = encode_df(df_unique)
df_unique =
    df_unique |>
    @filter(_.days_since_1st_D_or_A ≥ 4) |> # remove entries which were measured less than a week after vaccination
    DataFrame


## Build DAG dataFrame
dag_df = df[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :T, :P, :nP, :ID, :IDidx, :vax_history, :Vidx]]
describe(dag_df)
dag_df_unique = df_unique[!, [:E, :H, :V, :D, :R, :S, :M, :Ḟ, :T, :P, :nP, :Vidx, :vax_history, :ID]]


## Full model
#
@model function Full_SCM(Vidx, H, S, D, T, R, P, Ḟ, M, E, IDidx; n_vax=length(unique(Vidx)), n_id=length(unique(IDidx)))
    # population-level priors
    α ~ Normal(mean(E), 2.5 * std(E))

    V ~ Dirilchet(4, 2)

    βHE ~ Normal(0, 0.5)
    βDE ~ Normal(0, 0.5)
    βFE ~ Normal(0, 0.5)
    βME ~ Normal(0, 0.5)
    βPE ~ Normal(0, 0.5)
    βRE ~ Normal(0, 0.5)
    βSE ~ Normal(0, 0.5)
    βTE ~ Normal(0, 0.5)

    σ ~ Exponential(std(E))

    # priors for variance of random intercepts
    τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
    α_ID ~ filldist(Normal(0, τ), n_id)       # group-level intercepts
    α_vax ~ filldist(Normal(0, τ), n_vax)     # group-level intercepts

    # missing F values
    N_missing = sum(ismissing.(Ḟ))
    F_impute ~ filldist(Normal(), N_missing)
    ν ~ Normal(0.5, 1) # imputed mean
    σ_F ~ Exponential() # imputed SD

    i_missing = 1
    for i in eachindex(Ḟ)
        if ismissing(Ḟ[i])
            F_impute[i_missing] ~ Normal(ν, σ_F)
            f_imputed = F_impute[i_missing]
            i_missing += 1
        else
            Ḟ[i] ~ Normal(ν, σ_F)
            f_imputed = Ḟ[i]
        end
        # likelihood
        Ê = @. α + α_ID[IDidx][i] + α_vax[Vidx][i] + βHE * H[i] + βDE * D[i] + βFE * f_imputed + βME * M[i] + βPE * P[i] + βRE * R[i] + βSE * S[i] + βTE * T[i]
        E[i] ~ Normal(Ê, σ)
        # E[i] ~ arraydist(LocationScale.(Ê, σ, TDist.(ν)))
    end
end

Full_SCM_model = Full_SCM(dag_df.IDidx, dag_df.Vidx, dag_df.E, dag_df.H, dag_df.D, dag_df.Ḟ, dag_df.M, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S, dag_df.T);

Full_SCM_chn = sample(Full_SCM_model, NUTS(), MCMCThreads(), 3000, 4);
summarize(Full_SCM_chn)

Full_SCM_chn_df = DataFrame(Full_SCM_chn)[!, r"α\b|β"];
precis(Full_SCM_chn_df)



# describe(dag_df.F)

# t = Vector{Union{Missing,Float64}}(missing, nrow(dag_df))
# present_mask_F = completecases(dag_df, :F)
# t[present_mask_F] .= standardize(ZScoreTransform, Vector{Float64}(dag_df.F[present_mask_F]), dims=1)

# # create Ḟ, which is the standardised F with missing values
# dag_df.Ḟ = t

# @model function M_E(E, H, S, D, T, R, M, F, P, Vidx, IDidx)

#     σ ~ Exponential()
#     σ_F ~ Exponential()
#     α ~ Normal(0, 2)
#     ν ~ Normal(0.5, 1)
#     βM ~ Normal(0, 1)
#     βD ~ Normal(0, 1)
#     βF ~ Normal(0, 1)
#     βP ~ Normal(0, 1)
#     βR ~ Normal(0, 1) # good to include for precision even if blocked by collider F->M<-R
#     βS ~ Normal(0, 1)
#     βT ~ Normal(0, 1)
#     βW ~ Normal(0, 1)

#     N_missing = sum(ismissing.(Ḟ))
#     F_impute ~ filldist(Normal(), N_missing)

#     i_missing = 1
#     for i in eachindex(Ḟ)
#         if ismissing(Ḟ[i])
#             F_impute[i_missing] ~ Normal(ν, σ_F)
#             f = F_impute[i_missing]
#             i_missing += 1
#         else
#             Ḟ[i] ~ Normal(ν, σ_F)
#             f = Ḟ[i]
#         end
#         µ = @. α + βM * M[i] + βD * D[i] + βF * f + βP * P[i] + βR * R[i] + βS*S[i] + βT*T[i] + βW*H[i]
#         E[i] ~ Normal(µ, σ^2 * I)
#     end
# end

# M_E_ch = sample(M_E(dag_df.E, dag_df.M, dag_df.D, dag_df.Ḟ, log10.(1 .+ dag_df.nP), dag_df.R, dag_df.S, dag_df.T, dag_df.H), NUTS(), MCMCThreads(), 1500, 4)

# M_E_df = DataFrame(M_E_ch)
# precis(M_E_df)

# coeftab_plot(M_E_df, pars=[:βW, :βD, :βM, :βF, :βP, :βR, :βS])

# # plot(M_E_ch)
