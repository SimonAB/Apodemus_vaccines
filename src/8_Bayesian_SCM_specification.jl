### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ ff543d39-daa0-4d9c-aa97-5141a23a90d2
begin
	# allow more interactivity
	using PlutoUI, Images

	# data manipulation
	using CSV, DataFrames, Query
    using LazyArrays
    using ScientificTypes

	# splitting an normalising data
	using MLDataUtils: shuffleobs, splitobs, stratifiedobs, rescale!

	# modelling
	using Distributions
	using Turing
    using StatisticalRethinking, StructuralCausalModels

	# plotting & diagnostics
	using MCMCChains, StatsPlots, StatisticalRethinkingPlots
end

# ╔═╡ 1521f146-1473-11ec-0dde-d172a83b0632
md"# Bayesian Structural Causal Model: model specification"

# ╔═╡ 8587d46e-f55f-400c-9fc1-559fde09a8eb
PlutoUI.TableOfContents(aside=true)

# ╔═╡ 0ce9a69e-6547-40d6-a61f-6d9e60e24511
md"## Update & load packages"

# ╔═╡ 55da67e4-2f85-42c9-b254-98ada53cd06e
# Check how many threads are available
Threads.nthreads()

# ╔═╡ daa270cf-13a4-4968-bcda-7312ae9238a6
md"## Check & fix data"

# ╔═╡ 4a2ef9d7-e52b-495e-b801-9fc8c4bda829
md"### Import dataset"

# ╔═╡ 929da46e-51fb-46e0-bd9e-4dc8cad7eeda
dag_df = CSV.File("../data/dag_df.csv";
	missingstring="NA", pool=true) |>
    DataFrame

# ╔═╡ ffe93a75-57ad-4425-8b17-0748d68fe5a5
describe(dag_df)

# ╔═╡ e024da52-2530-49ca-94d7-b9f2059f35e4
# Standardise the features that needed for Turing
begin
	scaled_M = standardize(ZScoreTransform, dag_df[!, :M])
	scaled_F = standardize(ZScoreTransform, dag_df[!, :F])
	scaled_E = standardize(ZScoreTransform, dag_df[!, :E])
	scaled_W = standardize(ZScoreTransform, convert(Vector{Float64},dag_df[!, :W]))
end

# ╔═╡ 8800062f-9941-4965-9b34-2fd83cfbf679
md"## Causal hypothesis DAG"

# ╔═╡ b9b7505b-73ed-4a8a-be03-282298cbe2d0
load("../Apodemus-vacc-DAGs/DAG.jpg")

# ╔═╡ 61f53b16-605c-4208-8ab1-19111dceb090
md"""
**Main read-out:**
- ``E``: Vaccine Efficacy measured as DT-specific ELISA optical density mean-centered and scaled to unit variance. Continuous.

**Experimental interventions:**
- ``V``: Randomised controlled vaccination with a single dose of DT-alum vaccine. Boolean (DT-alum=1; alum=0).
- ``W``: Wild or lab mouse. Boolean (Wild=1; Lab=0)
- ``D``: Dietary supplementation with TransBreed® or basic chow. Boolean (TransBreed® = 1; Chow=0). *Note: in the wild, mice also forage from their environment.*

**Covariates:**
- ``T``: Number of days since vaccination. Discrete count.
- ``F``: Fat scores. Discrete, sum of 2 scores out of 3.
- ``M``: Body Mass. Continuous (grams).
- ``S``: Sex. Boolean (Male=1; Female=0).
- ``R``: Reproductive status. Boolean (Reproductive=1; Non-reproductive=0).
- ``P``: Presence of *Heligmosomoides polygyrus* in gut. Boolean (Infected=1).
"""

# ╔═╡ 370c8b64-489e-4b3e-a7af-3cb14aabcd89
md"Daggitty.net version:"
#=
**From [Dagitty.net](http://dagitty.net/dags.html#):**
dag {
bb="-2.633,-3.002,3.168,4.373"
D [exposure,pos="-1.075,-2.081"]
E [outcome,pos="1.054,0.603"]
F [pos="-0.219,-0.159"]
M [pos="0.403,-1.000"]
P [pos="0.483,-2.667"]
R [pos="0.337,-2.257"]
S [pos="1.815,-0.693"]
T [pos="-1.426,0.228"]
V [exposure,pos="-2.282,0.563"]
W [pos="-2.150,-0.585"]
D -> E
D -> F
D -> M
D -> P
D -> R
F -> E
F -> M
M -> E
P -> E
R -> E
R -> F
R -> M
S -> E
S -> F
S -> M
S -> P
S -> R
T -> E
T -> F
T -> M
T -> R
V -> E
V -> T
W -> E
W -> F
W -> M
W -> P
W -> R
W -> T
}

=#

# ╔═╡ 06e8df45-f82a-49f8-b5b2-2f40a9cc513a
dag = DAG("dag", "dag {
D -> E;
D -> F;
D -> M;
D -> P;
D -> R;
F -> E;
F -> M;
M -> E;
P -> E;
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
V -> E;
V -> T;
W -> E;
W -> F;
W -> M;
W -> P;
W -> R;
W -> T
}")

# ╔═╡ 1c6982b8-2993-478e-a9da-21dd93ca56a8
#=
       From worker 5:	  :M => [:W, :T, :S, :R, :F, :D]
      From worker 5:	  :F => [:W, :T, :S, :R, :D]
      From worker 5:	  :E => [:W, :V, :T, :S, :R, :M, :F, :D]
      From worker 5:	  :T => :V
      From worker 5:	BasisSet[
      From worker 5:	  :D ∐ :V
      From worker 5:	  :F ∐ :V | [:T]
      From worker 5:	  :F ∐ :V | [:W, :T]
      From worker 5:	  :F ∐ :V | [:T, :S]
      From worker 5:	  :F ∐ :V | [:T, :D]
      From worker 5:	  :F ∐ :V | [:T, :R]
      From worker 5:	  :F ∐ :V | [:W, :T, :S]
      From worker 5:	  :F ∐ :V | [:W, :T, :D]
      From worker 5:	  :F ∐ :V | [:W, :T, :R]
      From worker 5:	  :F ∐ :V | [:T, :S, :D]
      From worker 5:	  :F ∐ :V | [:T, :S, :R]
      From worker 5:	  :F ∐ :V | [:T, :D, :R]
      From worker 5:	  :F ∐ :V | [:W, :T, :S, :D]
      From worker 5:	  :F ∐ :V | [:W, :T, :S, :R]
      From worker 5:	  :F ∐ :V | [:W, :T, :D, :R]
      From worker 5:	  :F ∐ :V | [:T, :S, :D, :R]
      From worker 5:	  :F ∐ :V | [:W, :T, :S, :D, :R]
      From worker 5:	  :M ∐ :V | [:T]
      From worker 5:	  :M ∐ :V | [:W, :T]
      From worker 5:	  :M ∐ :V | [:T, :S]
      From worker 5:	  :M ∐ :V | [:T, :D]
      From worker 5:	  :M ∐ :V | [:T, :R]
      From worker 5:	  :M ∐ :V | [:T, :F]
      From worker 5:	  :M ∐ :V | [:W, :T, :S]
      From worker 5:	  :M ∐ :V | [:W, :T, :D]
      From worker 5:	  :M ∐ :V | [:W, :T, :R]
      From worker 5:	  :M ∐ :V | [:W, :T, :F]
      From worker 5:	  :M ∐ :V | [:T, :S, :D]
      From worker 5:	  :M ∐ :V | [:T, :S, :R]
      From worker 5:	  :M ∐ :V | [:T, :S, :F]
      From worker 5:	  :M ∐ :V | [:T, :D, :R]
      From worker 5:	  :M ∐ :V | [:T, :D, :F]
      From worker 5:	  :M ∐ :V | [:T, :R, :F]
      From worker 5:	  :M ∐ :V | [:W, :T, :S, :D]
      From worker 5:	  :M ∐ :V | [:W, :T, :S, :R]
      From worker 5:	  :M ∐ :V | [:W, :T, :S, :F]
      From worker 5:	  :M ∐ :V | [:W, :T, :D, :R]
      From worker 5:	  :M ∐ :V | [:W, :T, :D, :F]
      From worker 5:	  :M ∐ :V | [:W, :T, :R, :F]
      From worker 5:	  :M ∐ :V | [:T, :S, :D, :R]
      From worker 5:	  :M ∐ :V | [:T, :S, :D, :F]
      From worker 5:	  :M ∐ :V | [:T, :S, :R, :F]
      From worker 5:	  :M ∐ :V | [:T, :D, :R, :F]
      From worker 5:	  :M ∐ :V | [:W, :T, :S, :D, :R]
      From worker 5:	  :M ∐ :V | [:W, :T, :S, :D, :F]
      From worker 5:	  :M ∐ :V | [:W, :T, :S, :R, :F]
      From worker 5:	  :M ∐ :V | [:W, :T, :D, :R, :F]
      From worker 5:	  :M ∐ :V | [:T, :S, :D, :R, :F]
      From worker 5:	  :M ∐ :V | [:W, :T, :S, :D, :R, :F]
      From worker 5:	  :R ∐ :V | [:T]
      From worker 5:	  :R ∐ :V | [:W, :T]
      From worker 5:	  :R ∐ :V | [:T, :S]
      From worker 5:	  :R ∐ :V | [:T, :D]
      From worker 5:	  :R ∐ :V | [:W, :T, :S]
      From worker 5:	  :R ∐ :V | [:W, :T, :D]
      From worker 5:	  :R ∐ :V | [:T, :S, :D]
      From worker 5:	  :R ∐ :V | [:W, :T, :S, :D]
      From worker 5:	  :S ∐ :D
      From worker 5:	  :S ∐ :V
      From worker 5:	  :T ∐ :S
      From worker 5:	  :T ∐ :S | [:V]
      From worker 5:	  :T ∐ :D
      From worker 5:	  :T ∐ :D | [:V]
      From worker 5:	  :W ∐ :T
      From worker 5:	  :W ∐ :T | [:V]
      From worker 5:	  :W ∐ :S
      From worker 5:	  :W ∐ :D
      From worker 5:	  :W ∐ :V
      From worker 5:	]
=#

# ╔═╡ f1b5a984-83a3-4eff-a0ac-47208e8fd8e6
basis_set(dag)

# ╔═╡ 4fb746bc-9305-483a-b1fa-62705ceeec92
md"""
## Causal model [functions]

### Functional relationships derived from the DAG

``T ≔ f(V, W, σ)``\
``P ≔ f(D, W, S, σ)``\
``R ≔ f(T, W, D, S, σ)``\
``F ≔ f(T, W, D, R, S, σ)``\
``M ≔ f(F, T, W, D, R, S, σ)``\
``E ≔ f(V, T, F, W, D, M, R, S, P, σ)``\
"""

# ╔═╡ 74c0f4ed-5f0a-42e8-8faa-77d445270fdc
md"### *T ≔ f(V, W, σ)*"

# ╔═╡ 0ddca3c8-ff5b-4c35-903f-106b88c65cae
adjustment_sets(dag, :T, :V)

# ╔═╡ e59857ba-b22a-447a-96bb-83508b6c53aa
histogram(dag_df.T)

# ╔═╡ c6c73202-1f10-41e8-be55-b679d1494b8b
md"``T`` appears to be negative binomially ditributed. Julia's implimentation in the `Distributions` package isn'
t very useful for count data. We will use [Storoli](https://storopoli.io/Bayesian-Julia/pages/8_count_reg/)'s alternative implementaion:"

# ╔═╡ b0cf5427-ec23-4d5b-a215-7a80a84219fd
function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ

    return NegativeBinomial(r, p)
end

# ╔═╡ 2561bb90-bd8a-4403-b9fd-f16be8f2aeac
@model function X_T(T, V, W)
	# priors
	α ~ Normal(0, sqrt(7))
	βV ~ Normal(0, sqrt(10))
	βW ~ Normal(0, sqrt(10))
	ϕ⁻ ~ Gamma(0.01, 0.01)
    ϕ = 1 / ϕ⁻

	# likelihood
    T̂ = @. exp(α + βV*V + βW*W)
	T ~ arraydist(LazyArray(@~ NegativeBinomial2.(T̂, ϕ)))

	# estimate missing values (not working yet; see:https://turing.ml/dev/docs/using-turing/guide)
	# for i in eachindex(V)
 #        V[i] ~ NegativeBinomial2.(T, ϕ)
 #    end
end

# ╔═╡ bf26ce54-86bc-4f61-8e3f-72aabfea7151
X_T_model = X_T(dag_df.T, dag_df.V, dag_df.W)

# ╔═╡ e1e2b9d7-4359-4a0f-a55e-38120277a48c
X_T_chn = sample(X_T_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ b2f2d255-4d5c-49af-994d-134226ad99d4
plot(X_T_chn)

# ╔═╡ 41e488d2-9850-49fb-83ef-8faacc47fd87
begin
	X_T_df = DataFrame(X_T_chn)
	PRECIS(X_T_df)
end

# ╔═╡ ae2eff2c-7117-4691-95c2-03d3700d8e09
coeftab_plot(X_T_df; legend=false)

# ╔═╡ 94d43861-fe21-419b-8f50-c7918d06e0a0
X_T_chn[:βV].data

# ╔═╡ 9b864114-1fb8-4bdb-8897-7cb3aa4cac92
md"""### *nP ≔ f(D, W, S, σ)*

Is worm burden affected by diet, and sex?

"""

# ╔═╡ e16a49ac-a31e-4397-b457-7a7ce544431d
histogram(dag_df.nP)

# ╔═╡ 6f916627-cf94-4100-ae19-868c3f3575d8
@model function X_nP(nP, D, W, S)
	# priors
	α ~ Normal(0, sqrt(7))
	βD ~ Normal(0, sqrt(10))
	βW ~ Normal(0, sqrt(10))
	βS ~ Normal(0, sqrt(10))
	ϕ⁻ ~ Gamma(0.01, 0.01)
    ϕ = 1 / ϕ⁻

	# likelihood
    nP̂ = @. exp(α + βD*D + βW*W + βS*S)
	nP ~ arraydist(LazyArray(@~ NegativeBinomial2.(nP̂, ϕ)))
end

# ╔═╡ f2ff081c-055e-4f49-9b14-9da1a0d96a13
X_nP_model = X_nP(dag_df.nP, dag_df.D, dag_df.W, dag_df.S)

# ╔═╡ 064c54fc-1379-430b-afa2-6c9e959d2e0d
X_nP_chn = sample(X_nP_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ d5769dc2-587a-40cc-89c6-e2ecbfbb80ab
plot(X_nP_chn)

# ╔═╡ 486ab1ae-41a9-4217-9059-013602b18833
begin
	X_nP_chn_df = DataFrame(X_nP_chn);
	PRECIS(X_nP_chn_df)
end

# ╔═╡ fd5b52d0-cfef-46ed-add6-74d2685bb4c8
coeftab_plot(X_nP_chn_df; legend=false)

# ╔═╡ b138caee-7d16-4888-88d6-252ab2fd1721
md"""### *P ≔ f(D, W, S, σ)*
Is the probability of being infected affected by diet and sex?
"""

# ╔═╡ 8406604f-d304-46e0-8e06-88c71e5a5466
histogram(dag_df.P)

# ╔═╡ a38da3a3-b501-4bc2-8f2a-f41c8370e27e
@model function X_P(P, D, W, S)
	# priors
	α ~ Normal(0, sqrt(7))
	βD ~ Normal(0, sqrt(10))
	βW ~ Normal(0, sqrt(10))
	βS ~ Normal(0, sqrt(10))

	# likelihood
	P̂ = @. α .+ βD*D + βW*W + βS*S
	P ~ arraydist(LazyArray(@~ BernoulliLogit.(P̂)))
end

# ╔═╡ f4bf39f1-b08a-4367-b933-059bc0b24ecd
X_P_model = X_P(dag_df.P, dag_df.D, dag_df.W, dag_df.S)

# ╔═╡ 5f740ce7-7df4-4e19-9de9-c01d0d72529c
X_P_chn = sample(X_P_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ 6ae1366c-4abc-4c95-a971-a2ef84dbb48e
plot(X_P_chn)

# ╔═╡ 65d77803-cdfc-4f85-93fc-447db0e9fd0f
begin
	X_P_chn_df = DataFrame(X_P_chn);
	PRECIS(X_P_chn_df)
end

# ╔═╡ 2f40e971-72d0-4011-af26-f505aaaa7fb2
coeftab_plot(X_P_chn_df)

# ╔═╡ 39be5929-3150-4045-a71d-855602e9230e
md"### *R ≔ f(T, W, D, S, σ)*"

# ╔═╡ bdbbf960-5cdd-4e28-b4d5-df6056d1d1f4
@model function X_R(R, T, W, D, S)
	# priors
	α ~ Normal(0, sqrt(7))
	βT ~ Normal(0, sqrt(10))
	βW ~ Normal(0, sqrt(10))
	βD ~ Normal(0, sqrt(10))
	βS ~ Normal(0, sqrt(10))

	# likelihood
	R̂ = @. α .+ βT.*T + βW*W + βD*D + βS*S
	R ~ arraydist(LazyArray(@~ BernoulliLogit.(R̂)))
end

# ╔═╡ c5f14355-340f-4f0e-901a-72e7238b3865
X_R_model = X_R(dag_df.R, dag_df.T, dag_df.W, dag_df.D, dag_df.S)

# ╔═╡ 4f47e1ff-dc76-46d7-9a71-cc5b5b500753
X_R_chn = sample(X_R_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ fbb76a4c-40d1-4ef4-9993-ddcaa3b92a2d
plot(X_R_chn)

# ╔═╡ 75701150-68b2-44a1-85ab-6b0c87c259d2
X_R_chn_df = DataFrame(X_R_chn);

# ╔═╡ c081aeeb-c4d8-4bac-b655-2516bff24d23
PRECIS(X_R_chn_df)

# ╔═╡ 1a7933da-44f1-4ab6-962f-077d774989ab
coeftab_plot(X_R_chn_df; legend=false)

# ╔═╡ 67aef37a-22af-4cb9-9c5f-1ae6a34fa2e1
md"### *F ≔ f(T, W, D, R, S, σ)*"

# ╔═╡ ccef428e-0867-4da0-b08b-009c2cb8f929
adjustment_sets(dag, :F, :T)

# ╔═╡ 4a58afa8-8159-4445-be5d-9e78ff726001
adjustment_sets(dag, :F, :W)

# ╔═╡ 6aa86cd0-c43a-4c59-8a57-ef049417125b
md"""
Regarding S, R, and D:
> In Model 8 (``X → Y ← Z``), ``Z`` is not a confounder nor does it block any back-door paths. Likewise, controlling for ``Z`` does not open any back-door paths from ``X`` to ``Y``. Thus, in terms of asymptotic bias, ``Z`` is a “neutral control.” Analysis shows, however, that controlling for ``Z`` reduces the variation of the outcome variable ``Y``, and helps to improve the precision of the ACE estimate in ﬁnite samples (Hahn, 2004; White and Lu, 2011; Henckel et al., 2019; Rotnitzky and Smucler, 2019).\
> *Cinelli2021*
"""

# ╔═╡ 91321b0d-96b9-41be-964a-67a3fd93b100
histogram(dag_df.F; legend=false)

# ╔═╡ d7c820cd-54bf-4ae6-81a7-2dc07a2777e5
histogram(scaled_F; legend=false, bins=4)

# ╔═╡ aea2d21e-7f90-4e1f-8f25-38a68ed87afa
md" ⇒ best use `scaled_f`"

# ╔═╡ 2880c185-ce8c-4eb7-9e62-cca63d60ed68
@model function X_F(F, T, W, S, D)
	# priors
	α ~ Normal(0, sqrt(7))
	βT ~ Normal(0, sqrt(10))
	βW ~ Normal(0, sqrt(10))
	βS ~ Normal(0, sqrt(10))
	βD ~ Normal(0, sqrt(10))
	σ² ~ TruncatedNormal(0, 100, 0, Inf)

	# likelihood
	F̂ = @. α + βT*T + βW*W + βS*S + βD*D
	F ~ MvNormal(F̂, sqrt(σ²))
end

# ╔═╡ 5e8ddf3a-30be-4b4f-bdb8-5e3c891ef114
X_F_model = X_F(scaled_F, dag_df.T, dag_df.W, dag_df.S, dag_df.D)

# ╔═╡ b857361c-ea06-4703-9f89-214238c14f10
X_F_chn = sample(X_F_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ e8703f11-aea1-4c8d-847c-272e04f6295c
plot(X_F_chn)

# ╔═╡ dfefabc0-ff41-4b5b-b3f9-0e2e415c472d
X_F_df = X_F_chn |> DataFrame;

# ╔═╡ 433cd2ae-dae7-4d28-968a-753fc75128e5
PRECIS(X_F_df)

# ╔═╡ 2ff3b3a4-c618-485a-adef-a498ddff6b44
coeftab_plot(X_F_df; legend=false)

# ╔═╡ ebca4ad8-2cf2-4890-8fd6-49ecf5b95076
md"### *M ≔ f(F, T, W, D, R, S, σ)*"

# ╔═╡ 38f8c509-00c1-466d-8f0b-7e65179095a3
adjustment_sets(dag, :F, :M)

# ╔═╡ a78e967a-96f9-4e40-8b84-57b9324b95f1
histogram(scaled_M)

# ╔═╡ 824b0622-b6d3-40c6-99da-998e0d0590f5
histogram(dag_df.M)

# ╔═╡ 5841350a-8287-4d67-ab5c-f5f1107240f1
@model function X_M(M, F, T, R, D, W, S)
	# priors
	α ~ Normal(0, sqrt(7))
	βF ~ Normal(0, sqrt(10))
	βT ~ Normal(0, sqrt(10))
	βR ~ Normal(0, sqrt(10))
	βD ~ Normal(0, sqrt(10))
	βW ~ Normal(0, sqrt(10))
	βS ~ Normal(0, sqrt(10))
	σ² ~ TruncatedNormal(0, 100, 0, Inf)

	# likelihood
	M̂ = @. α + βF*F + βT*T + βW*W + βS*S + βR*R + βD*D
	M ~ MvNormal(M̂, sqrt(σ²))
end

# ╔═╡ 6caa0ffa-303b-441b-ade3-84dba5f1b5f9
X_M_model = X_M(dag_df.M, dag_df.F, dag_df.T, dag_df.R, dag_df.D, dag_df.W, dag_df.S)

# ╔═╡ 40cd5588-ecb5-4228-9007-3bbd0c293949
X_M_chn = sample(X_M_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ a1a2dc92-6ad4-472f-93f2-3b6b4c17495d
plot(X_M_chn)

# ╔═╡ 991d6686-550e-4bbb-86bd-f4f3238376ff
X_M_chn_df = DataFrame(X_M_chn);

# ╔═╡ a0729aa7-87be-4530-be6d-d83fc55631b8
PRECIS(X_M_chn_df)

# ╔═╡ 351da487-bad5-4f3b-82b1-802e958d2af0
coeftab_plot(X_M_chn_df)

# ╔═╡ 6531aae8-008f-4d20-81fe-58810b3b6492
md"### *E ≔ f(V, T, F, W, D, M, R, S, P, σ)*"

# ╔═╡ 7784e2d4-ba1c-43a6-bd38-5826e7e84e7c
histogram(scaled_E)

# ╔═╡ 21126b87-74cf-49bb-b7ec-d169dbe1adc6
@model function X_E(E, V, T, F, W, D, M, R, S, P)
	# priors
	α ~ Normal(0, sqrt(7))
	βV ~ Normal(0, sqrt(10))
	βT ~ Normal(0, sqrt(10))
	βF ~ Normal(0, sqrt(10))
	βW ~ Normal(0, sqrt(10))
	βD ~ Normal(0, sqrt(10))
	βM ~ Normal(0, sqrt(10))
	βR ~ Normal(0, sqrt(10))
	βS ~ Normal(0, sqrt(10))
	βP ~ Normal(0, sqrt(10))
	σ² ~ TruncatedNormal(0, 100, 0, Inf)

	# likelihood
	Ê = @. α + βV*V + βT*T + βF*F + βW*W + βD*D + βM*M + βR*R + βS*S + βP*P
	E ~ MvNormal(Ê, sqrt(σ²))
end

# ╔═╡ f9e11607-49be-41f9-a138-7358cb8412c7
X_E_model = X_E(scaled_E, dag_df.V, dag_df.T, dag_df.F, dag_df.W, dag_df.D, dag_df.M, dag_df.R, dag_df.S, dag_df.P) # note we are using P here, not nP

# ╔═╡ fac115e6-655e-4179-9ca9-bbffe74257c1
X_E_chn = sample(X_E_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ 33388b4f-334d-4a00-86b5-187724764a8f
plot(X_E_chn)

# ╔═╡ 3adfcaf1-61ed-453a-a347-6c4909e351f9
begin
	X_E_chn_df = DataFrame(X_E_chn)
	PRECIS(X_E_chn_df)
end

# ╔═╡ dc7a3f1e-c594-4323-ae1c-ba5edd3bde12
coeftab_plot(X_E_chn_df; legend=false)

# ╔═╡ f1ec6c25-f23d-46d8-a364-55012d9f1f0e
md"## Final Structural Causal Model"

# ╔═╡ feeb9c4d-b008-49f3-8ce8-d22d7ee46e1a
load("../Apodemus-vacc-DAGs/DAG_param.jpg")

# ╔═╡ 82b3776d-5226-4935-96c2-1028a6f0b84c
md" ## Total effects of treatments on *E*"

# ╔═╡ 22d3f36d-1a0f-4873-bf95-99017d2faa1b
md"### Effect of *D* on *E*"

# ╔═╡ 89204946-0032-41d2-b8a9-5098f759855c
adjustment_sets(dag, :D, :E)

# ╔═╡ f629cf94-b767-4c98-91ae-83ddad7ad319
@model function D_E(E, D)
	# priors
	α ~ Normal(0, sqrt(7))
	βD ~ Normal(0, sqrt(10))
	σ² ~ TruncatedNormal(0, 100, 0, Inf)

	# likelihood
	Ê = @. α + βD*D
	E ~ MvNormal(Ê, sqrt(σ²))
end

# ╔═╡ be293fa3-8160-4109-a95a-49199067235c
D_E_model = D_E(scaled_E, dag_df.D)

# ╔═╡ a466e611-5780-4e31-9bc7-4d89053a5b45
D_E_chn = sample(D_E_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ 2ca40136-efdc-45b4-8f6b-c7f8a73616bf
plot(D_E_chn)

# ╔═╡ f092f12b-8026-4eaa-8c78-dc7a700a66e8
begin
	D_E_chn_df = DataFrame(D_E_chn)
	PRECIS(D_E_chn_df)
end

# ╔═╡ 2d3edb03-023e-4d2e-9354-40c4bc543e06
coeftab_plot(D_E_chn_df)

# ╔═╡ da284c10-19b1-4a79-b6d3-6cb9e8f8d751
md"### Effect of *W* on *E*"

# ╔═╡ bcdde5c4-d69b-409c-a7a9-f3f3b571a439
adjustment_sets(dag, :W, :E)

# ╔═╡ 7ffb6905-e417-44ff-8b5a-0c8adf242d2e
@model function W_E(E, W)
	# priors
	α ~ Normal(0, sqrt(7))
	βW ~ Normal(0, sqrt(10))
	σ² ~ TruncatedNormal(0, 100, 0, Inf)

	# likelihood
	Ê = @. α + βW*W
	E ~ MvNormal(Ê, sqrt(σ²))
end

# ╔═╡ dec533a8-e808-4042-96da-0ee1a0ec9e4b
W_E_model = W_E(scaled_E, dag_df.W)

# ╔═╡ b92ff6ad-199e-44e4-a974-1ad804f8034a
W_E_chn = sample(W_E_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ 4b74f82d-994d-4cfa-8dea-80077bd44da5
plot(W_E_chn)

# ╔═╡ 654d18cb-9f2a-4e59-97aa-fe74814ed151
begin
	W_E_chn_df = DataFrame(W_E_chn)
	PRECIS(W_E_chn_df)
end

# ╔═╡ 21a6fd05-7001-4572-b4e3-4fcd31a52be1
coeftab_plot(W_E_chn_df)

# ╔═╡ f5c9086e-be9a-41e1-869b-3a879e9b44bf
md"### Effect of *V* on *E*"

# ╔═╡ 6cc684c7-22f7-4873-a2cb-63cf01189818
adjustment_sets(dag, :V, :E)

# ╔═╡ 11f2147b-3385-4ec0-808a-94d63b99aa9a
@model function V_E(E, V)
	# priors
	α ~ Normal(0, sqrt(7))
	βV ~ Normal(0, sqrt(10))
	σ² ~ TruncatedNormal(0, 100, 0, Inf)

	# likelihood
	Ê = @. α + βV*V
	E ~ MvNormal(Ê, sqrt(σ²))
end

# ╔═╡ 5c22520d-50d6-4d46-a2c6-8a75028cf26c
V_E_model = V_E(scaled_E, dag_df.V)

# ╔═╡ c8104c12-0459-4cf7-b830-faa316cedcbb
V_E_chn = sample(V_E_model, NUTS(0.65), MCMCThreads(), 1500, 4);

# ╔═╡ 60aad7ed-3d15-4536-aa43-457e53aac678
plot(V_E_chn)

# ╔═╡ 025a9214-5c8b-4927-87c6-a5d14c0186e8
begin
	V_E_chn_df = DataFrame(V_E_chn)
	PRECIS(V_E_chn_df)
end

# ╔═╡ af3582e2-2d4c-4ff0-b0e9-0d143b8f7579
coeftab_plot(V_E_chn_df)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
MLDataUtils = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Query = "1a8c2f83-1ff3-5112-b086-8aa67b057ba1"
ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"
StatisticalRethinking = "2d09df54-9d0f-5258-8220-54c2a3d4fbee"
StatisticalRethinkingPlots = "e1a513d0-d9d9-49ff-a6dd-9d2e9db473da"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
StructuralCausalModels = "a41e6734-49ce-4065-8b83-aff084c01dfd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.8.5"
DataFrames = "~1.3.0"
Distributions = "~0.24.18"
Images = "~0.24.1"
LazyArrays = "~0.21.20"
MCMCChains = "~4.14.1"
MLDataUtils = "~0.5.4"
PlutoUI = "~0.7.1"
Query = "~1.0.0"
ScientificTypes = "~2.0.1"
StatisticalRethinking = "~3.4.3"
StatisticalRethinkingPlots = "~0.3.0"
StatsPlots = "~0.14.29"
StructuralCausalModels = "~1.0.8"
Turing = "~0.15.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "7fcd8ce8931c56ba62827c87a291ea72ee07ce31"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "2.5.0"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "ba9984ea1829e16b3a02ee49497c84c9795efa25"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[deps.AdvancedHMC]]
deps = ["ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "Parameters", "ProgressMeter", "Random", "Requires", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7e85ed4917716873423f8d47da8c1275f739e0b7"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.2.27"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "57bda8215ba78990ce600972b533e2f6516287e8"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.5.9"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "130d6b17a3a9d420d9a6b37412cae03ffd6a64ff"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[deps.Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[deps.ArrayInterface]]
deps = ["LinearAlgebra", "Requires", "SparseArrays"]
git-tree-sha1 = "a2a1884863704e0414f6f164a1f6f4a2a62faf4e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "2.14.17"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "623a32b87ef0b85d26320a8cc7e57ded707aef64"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.7.5"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[deps.BSplines]]
deps = ["LinearAlgebra", "OffsetArrays", "RecipesBase"]
git-tree-sha1 = "54308218333f6b8967311c76f2cf89ab495542d6"
uuid = "488c2830-172b-11e9-1591-253b8a7df96d"
version = "0.3.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "0ad226aa72d8671f20d0316e03028f0ba1624307"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.32"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "068fda9b756e41e6c75da7b771e6f89fa8a43d15"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "0.7.0"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "Compat", "Distributions", "Functors", "LinearAlgebra", "MappedArrays", "NNlib", "NonlinearSolve", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "88a1303ee10c24b6df86eeafb98e502115c7be58"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.8.16"

[[deps.BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[deps.BrowseTables]]
deps = ["ArgCheck", "DefaultApplication", "DocStringExtensions", "Parameters", "Tables"]
git-tree-sha1 = "2df4c05941860fd6149c349422d584174044718a"
uuid = "5f4fecfd-7eb0-5078-b7f6-ad1f2563c22a"
version = "0.3.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[deps.CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "c308f209870fdbd84cb20332b6dfaf14bf3387f8"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.2"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Reexport", "Requires", "Statistics"]
git-tree-sha1 = "422db294d817de46668a3bf119175080ab093b23"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.7.70"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4b28f88cecf5d9a07c85b9ce5209a361ecaff34a"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.9.45"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "2e993336a3f68216be91eb8ee4625ebbaba19147"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefaultApplication]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "fc2b7122761b22c87fec8bf2ea4dc4563d9f8c24"
uuid = "3f0dd361-4fe0-5fc6-8523-80b14ec94d85"
version = "1.0.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "77b4ca280084423b728662fe040e5ff8819347c5"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.1"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "d8f468c5cd4d94e86816603f7d18ece910b4aaf1"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.5.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a837fdf80f333415b69684ba8e8ae6ba76de6aaa"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.24.18"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "f773f784beca655b28ec1b235dbb9f5a6e5e151f"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.29"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DrWatson]]
deps = ["Dates", "FileIO", "LibGit2", "MacroTools", "Pkg", "Random", "Requires", "UnPack"]
git-tree-sha1 = "d6aa02ad618cf31af9bbbf87f87baad632538211"
uuid = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
version = "2.5.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "84f04fe68a3176a583b864e492578b9466d87f1e"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.6"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "Bijectors", "ChainRulesCore", "Distributions", "MacroTools", "NaturalSort", "Random", "ZygoteRules"]
git-tree-sha1 = "c32726683fc17742ece85ac63e8368b033cffa44"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.10.20"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EllipsisNotation]]
git-tree-sha1 = "18ee049accec8763be17a933737c1dd0fdf8673a"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.0.0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "4471d36a75e9168f80708155df33e1601b11c13c"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "0.3.1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2db648b6712831ecb333eae76dbfd1c156ca13bb"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.2"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2b72a5624e289ee18256111657663721d59c143e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.24"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
deps = ["MacroTools"]
git-tree-sha1 = "f40adc6422f548176bb4351ebd29e4abf773040a"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.1.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "74ef6288d071f58033d54fd6708d4bc23a8b8972"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "ffdcd4272a7cc36442007bca41aa07ca3cc5fda4"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "1.3.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3395d4d4aeb3c9d31f5929d32760d8baeee88aaf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.5.0+0"

[[deps.IdentityRanges]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be8fcd695c4da16a1d6d0cd213cb88090a150e3b"
uuid = "bbac6d45-d8f3-5730-bfe4-7a449cd117ca"
version = "0.3.1"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "794ad1d922c432082bc1aaa9fa8ffbd1fe74e621"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.9"

[[deps.ImageContrastAdjustment]]
deps = ["ColorVectorSpace", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "2e6084db6cccab11fe0bc3e4130bd3d117092ed9"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.7"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[deps.ImageDistances]]
deps = ["ColorVectorSpace", "Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "6378c34a3c3a216235210d19b9f495ecfff2f85f"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.13"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ColorVectorSpace", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageCore", "LinearAlgebra", "OffsetArrays", "Requires", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "bf96839133212d3eff4a1c3a80c57abc7cfbf0ce"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.6.21"

[[deps.ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "a2951c93684551467265e0e32b577914f69532be"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ea2b6fd947cdfc43c6b8c15cff982533ec1f72cd"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+0"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ColorVectorSpace", "ImageAxes", "ImageCore", "IndirectArrays"]
git-tree-sha1 = "ae76038347dc4edcdb06b541595268fca65b6a42"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.5"

[[deps.ImageMorphology]]
deps = ["ColorVectorSpace", "ImageCore", "LinearAlgebra", "TiledIteration"]
git-tree-sha1 = "68e7cbcd7dfaa3c2f74b0a8ab3066f5de8f2b71d"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.2.11"

[[deps.ImageQualityIndexes]]
deps = ["ColorVectorSpace", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1198f85fa2481a3bb94bf937495ba1916f12b533"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.2.2"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageCore", "OffsetArrays", "Requires", "StackViews"]
git-tree-sha1 = "832abfd709fa436a562db47fd8e81377f72b01f9"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.1"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "IdentityRanges", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e4cc551e4295a5c96545bb3083058c24b78d4cf0"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.8.13"

[[deps.Images]]
deps = ["AxisArrays", "Base64", "ColorVectorSpace", "FileIO", "Graphics", "ImageAxes", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageShow", "ImageTransformations", "IndirectArrays", "OffsetArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "8b714d5e11c91a0d945717430ec20f9251af4bd2"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.24.1"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InitialValues]]
git-tree-sha1 = "7f6a4508b4a6f46db5ccd9799a3fc71ef5cad6e6"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.11"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "1e0e51692a3a77f1eeb51bf741bdd0439ed210e7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.2"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterableTables]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Requires", "TableTraits", "TableTraitsUtils"]
git-tree-sha1 = "70300b876b2cebde43ebc0df42bc8c94a144e1b4"
uuid = "1c8ee90f-4401-5389-894e-7a04a3dc0f4d"
version = "1.0.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "1f93019153b4e9dab37e561b61f92b431f2ecedb"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "0.21.20"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LearnBase]]
deps = ["LinearAlgebra", "StatsBase"]
git-tree-sha1 = "47e6f4623c1db88570c7a7fa66c6528b92ba4725"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.3.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtask]]
deps = ["BinaryProvider", "Libdl", "Pkg"]
git-tree-sha1 = "83e082fccb4e37d93df6440cdbd41dcbe5e46cb6"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.4.2"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "BenchmarkTools", "DiffResults", "DocStringExtensions", "Random", "Requires", "TransformVariables", "UnPack"]
git-tree-sha1 = "b8a3c29fdd8c512a7e80c4ec27d609e594a89860"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "0.10.6"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "dfeda1c1130990428720de0024d4516b1902ce98"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.7"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "DocStringExtensions", "IfElse", "LinearAlgebra", "OffsetArrays", "Requires", "SLEEFPirates", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "5f275de503982d59bd82eb1e4fbc273f55a72dee"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.10.0"

[[deps.MCMCChains]]
deps = ["AbstractFFTs", "AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MLJModelInterface", "NaturalSort", "PrettyTables", "Random", "RecipesBase", "Serialization", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "2cc8ea543a3dbb951e8b3310a4ab7605790aa9f1"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "4.14.1"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[deps.MLDataPattern]]
deps = ["LearnBase", "MLLabelUtils", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e99514e96e8b8129bb333c69e063a56ab6402b5b"
uuid = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
version = "0.5.4"

[[deps.MLDataUtils]]
deps = ["DataFrames", "DelimitedFiles", "LearnBase", "MLDataPattern", "MLLabelUtils", "Statistics", "StatsBase"]
git-tree-sha1 = "ee54803aea12b9c8ee972e78ece11ac6023715e6"
uuid = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
version = "0.5.4"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "0174e9d180b0cae1f8fe7976350ad52f0e70e0d8"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.3.3"

[[deps.MLLabelUtils]]
deps = ["LearnBase", "MappedArrays", "StatsBase"]
git-tree-sha1 = "3211c1fdd1efaefa692c8cf60e021fb007b76a08"
uuid = "66a33bbf-0c2b-5fc8-a008-9da813334f0a"
version = "0.5.6"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "1a0358d0283b84c3ccf9537843e3583c3b896c59"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "0.8.5"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "Setfield"]
git-tree-sha1 = "4f65bdbbe93475f6ff9ea6969b21532f88d359be"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MonteCarloMeasurements]]
deps = ["Distributed", "Distributions", "LinearAlgebra", "MacroTools", "Random", "RecipesBase", "Requires", "SLEEFPirates", "StaticArrays", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "d9dc441d5a0393ee80aea49d0fb13281d7cc7c91"
uuid = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
version = "1.0.3"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "2eb305b13eaed91d7da14269bf17ce6664bfee3d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.31"

[[deps.NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[deps.NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[deps.Netpbm]]
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "8dc3be3e9edf976a3e79363b3bd2ad776a627c31"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.12"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "35d435b512fbab1d1a29138b5229279925eba369"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.5.0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "6d105d40e30b635cfed9d52ec29cf456e27d38f8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.12"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[deps.PersistenceDiagramsBase]]
deps = ["Compat", "Tables"]
git-tree-sha1 = "ec6eecbfae1c740621b5d903a69ec10e30f3f4bc"
uuid = "b1ad91c1-539c-4ace-90bd-ea06abc420fa"
version = "0.1.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "e4fe0b50af3130ddd25e793b471cb43d5279e3e6"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "bc70e31a04f22780b57ad399ff94f9b78a1e7b39"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.5"

[[deps.PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.Quaternions]]
deps = ["DualNumbers", "LinearAlgebra"]
git-tree-sha1 = "adf644ef95a5e26c8774890a509a55b7791a139f"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.4.2"

[[deps.Query]]
deps = ["DataValues", "IterableTables", "MacroTools", "QueryOperators", "Statistics"]
git-tree-sha1 = "a66aa7ca6f5c29f0e303ccef5c8bd55067df9bbe"
uuid = "1a8c2f83-1ff3-5112-b086-8aa67b057ba1"
version = "1.0.0"

[[deps.QueryOperators]]
deps = ["DataStructures", "DataValues", "IteratorInterfaceExtensions", "TableShowUtils"]
git-tree-sha1 = "911c64c204e7ecabfd1872eb93c49b4e7c701f02"
uuid = "2aef5ad7-51ca-5a8f-8e88-e75cf067b44b"
version = "0.9.3"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[deps.RecursiveArrayTools]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "b3f4e34548b3d3d00e5571fd7bc0a33980f01571"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.11.4"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization"]
git-tree-sha1 = "2e1a88c083ebe8ba69bc0b0084d4b4ba4aa35ae0"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.1.13"

[[deps.Reexport]]
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "8f82019e525f4d5c669692772a6f4b0a58b06a6a"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.2.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "dbf5f991130238f10abbf4f2d255fb2837943c43"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.1.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Libdl", "VectorizationBase"]
git-tree-sha1 = "3d44bb7517298fd262915924fdc1645c61a6ef17"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.8"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "0b5b04995a3208f82ae69a17389c60b762000793"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.13.6"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "PersistenceDiagramsBase", "PrettyTables", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "d21a6b62f64cee2f2def2f55b70cb91ebc0dea7b"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "2.0.1"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "185e373beaf6b381c1e7151ce2c2a722351d6637"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "2.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "244586bc07462d22aed0113af9c731f2a518c93e"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "0afd9e6c623e379f593da01f20590bacc26d1d14"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[deps.StatisticalRethinking]]
deps = ["AxisArrays", "BSplines", "BrowseTables", "CSV", "DataFrames", "Distributions", "DocStringExtensions", "Formatting", "GLM", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MCMCChains", "MonteCarloMeasurements", "NamedArrays", "NamedTupleTools", "Optim", "OrderedCollections", "Parameters", "PrettyTables", "Random", "Reexport", "Requires", "Statistics", "StatsBase", "StatsFuns", "StatsModelComparisons", "StatsPlots", "StructuralCausalModels", "Tables", "Test", "Unicode"]
git-tree-sha1 = "ebbe2734084e08d7be5e52dda303e4d19b826ab8"
uuid = "2d09df54-9d0f-5258-8220-54c2a3d4fbee"
version = "3.4.3"

[[deps.StatisticalRethinkingPlots]]
deps = ["DataFrames", "DocStringExtensions", "GR", "LaTeXStrings", "Parameters", "Plots", "Requires", "StatsBase", "StatsPlots"]
git-tree-sha1 = "df0b9242b6099f79a3112beaabcc6af9cde3d654"
uuid = "e1a513d0-d9d9-49ff-a6dd-9d2e9db473da"
version = "0.3.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "730732cae4d3135e2f2182bd47f8d8b795ea4439"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "2.1.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[deps.StatsFuns]]
deps = ["Rmath", "SpecialFunctions"]
git-tree-sha1 = "ced55fd4bae008a8ea12508314e725df61f0ba45"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.7"

[[deps.StatsModelComparisons]]
deps = ["DrWatson", "RecipesBase", "Statistics", "StatsFuns"]
git-tree-sha1 = "8f939fa88acc1bb85f247d7fe81d83fa3456b286"
uuid = "854dedd9-9477-4a25-907d-7fd989bfdd01"
version = "1.1.0"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "677488c295051568b0b79a77a8c44aa86e78b359"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.28"

[[deps.StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "d6956cefe3766a8eb5caae9226118bb0ac61c8ac"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.29"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[deps.StructuralCausalModels]]
deps = ["CSV", "Combinatorics", "DataFrames", "DataStructures", "Distributions", "DocStringExtensions", "LinearAlgebra", "NamedArrays", "Reexport", "Statistics"]
git-tree-sha1 = "6db3ef54a262bda2c8c1b3b7c0e1f4c35d9686b7"
uuid = "a41e6734-49ce-4065-8b83-aff084c01dfd"
version = "1.0.8"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableShowUtils]]
deps = ["DataValues", "Dates", "JSON", "Markdown", "Test"]
git-tree-sha1 = "14c54e1e96431fb87f0d2f5983f090f1b9d06457"
uuid = "5e66a065-1f0a-5976-b372-e0b8c017ca10"
version = "0.2.5"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["VectorizationBase"]
git-tree-sha1 = "64a2be7c73951d7c402eb40a16055e5e6fdda468"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.2.3"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "c342ae2abf4902d65a0b0bf59b28506a6e17078a"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.2"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "bccb153150744d476a6a8d4facf5299325d5a442"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.67"

[[deps.TransformVariables]]
deps = ["ArgCheck", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "Pkg", "Random", "UnPack"]
git-tree-sha1 = "9433efc8545a53a9a34de0cdb9316f9982a9f290"
uuid = "84d833dd-6860-57f9-a1a7-6da5db126cff"
version = "0.4.1"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedVI", "BangBang", "Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "2b57320e0f4c59e264dd34d5f0fb3bcc67ac65ae"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.15.1"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "Hwloc", "IfElse", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "9f27ddc74f04319574749bb0a5b3f231a5e7db16"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.16.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─1521f146-1473-11ec-0dde-d172a83b0632
# ╟─8587d46e-f55f-400c-9fc1-559fde09a8eb
# ╟─0ce9a69e-6547-40d6-a61f-6d9e60e24511
# ╠═ff543d39-daa0-4d9c-aa97-5141a23a90d2
# ╠═55da67e4-2f85-42c9-b254-98ada53cd06e
# ╟─daa270cf-13a4-4968-bcda-7312ae9238a6
# ╟─4a2ef9d7-e52b-495e-b801-9fc8c4bda829
# ╠═929da46e-51fb-46e0-bd9e-4dc8cad7eeda
# ╠═ffe93a75-57ad-4425-8b17-0748d68fe5a5
# ╠═e024da52-2530-49ca-94d7-b9f2059f35e4
# ╟─8800062f-9941-4965-9b34-2fd83cfbf679
# ╠═b9b7505b-73ed-4a8a-be03-282298cbe2d0
# ╟─61f53b16-605c-4208-8ab1-19111dceb090
# ╠═370c8b64-489e-4b3e-a7af-3cb14aabcd89
# ╠═06e8df45-f82a-49f8-b5b2-2f40a9cc513a
# ╟─1c6982b8-2993-478e-a9da-21dd93ca56a8
# ╠═f1b5a984-83a3-4eff-a0ac-47208e8fd8e6
# ╟─4fb746bc-9305-483a-b1fa-62705ceeec92
# ╟─74c0f4ed-5f0a-42e8-8faa-77d445270fdc
# ╠═0ddca3c8-ff5b-4c35-903f-106b88c65cae
# ╠═e59857ba-b22a-447a-96bb-83508b6c53aa
# ╟─c6c73202-1f10-41e8-be55-b679d1494b8b
# ╠═b0cf5427-ec23-4d5b-a215-7a80a84219fd
# ╠═2561bb90-bd8a-4403-b9fd-f16be8f2aeac
# ╠═bf26ce54-86bc-4f61-8e3f-72aabfea7151
# ╠═e1e2b9d7-4359-4a0f-a55e-38120277a48c
# ╠═b2f2d255-4d5c-49af-994d-134226ad99d4
# ╠═41e488d2-9850-49fb-83ef-8faacc47fd87
# ╠═ae2eff2c-7117-4691-95c2-03d3700d8e09
# ╠═94d43861-fe21-419b-8f50-c7918d06e0a0
# ╟─9b864114-1fb8-4bdb-8897-7cb3aa4cac92
# ╠═e16a49ac-a31e-4397-b457-7a7ce544431d
# ╠═6f916627-cf94-4100-ae19-868c3f3575d8
# ╠═f2ff081c-055e-4f49-9b14-9da1a0d96a13
# ╠═064c54fc-1379-430b-afa2-6c9e959d2e0d
# ╠═d5769dc2-587a-40cc-89c6-e2ecbfbb80ab
# ╠═486ab1ae-41a9-4217-9059-013602b18833
# ╠═fd5b52d0-cfef-46ed-add6-74d2685bb4c8
# ╟─b138caee-7d16-4888-88d6-252ab2fd1721
# ╠═8406604f-d304-46e0-8e06-88c71e5a5466
# ╠═a38da3a3-b501-4bc2-8f2a-f41c8370e27e
# ╠═f4bf39f1-b08a-4367-b933-059bc0b24ecd
# ╠═5f740ce7-7df4-4e19-9de9-c01d0d72529c
# ╠═6ae1366c-4abc-4c95-a971-a2ef84dbb48e
# ╠═65d77803-cdfc-4f85-93fc-447db0e9fd0f
# ╠═2f40e971-72d0-4011-af26-f505aaaa7fb2
# ╟─39be5929-3150-4045-a71d-855602e9230e
# ╠═bdbbf960-5cdd-4e28-b4d5-df6056d1d1f4
# ╠═c5f14355-340f-4f0e-901a-72e7238b3865
# ╠═4f47e1ff-dc76-46d7-9a71-cc5b5b500753
# ╠═fbb76a4c-40d1-4ef4-9993-ddcaa3b92a2d
# ╠═75701150-68b2-44a1-85ab-6b0c87c259d2
# ╠═c081aeeb-c4d8-4bac-b655-2516bff24d23
# ╠═1a7933da-44f1-4ab6-962f-077d774989ab
# ╟─67aef37a-22af-4cb9-9c5f-1ae6a34fa2e1
# ╠═ccef428e-0867-4da0-b08b-009c2cb8f929
# ╠═4a58afa8-8159-4445-be5d-9e78ff726001
# ╟─6aa86cd0-c43a-4c59-8a57-ef049417125b
# ╠═91321b0d-96b9-41be-964a-67a3fd93b100
# ╠═d7c820cd-54bf-4ae6-81a7-2dc07a2777e5
# ╟─aea2d21e-7f90-4e1f-8f25-38a68ed87afa
# ╠═2880c185-ce8c-4eb7-9e62-cca63d60ed68
# ╠═5e8ddf3a-30be-4b4f-bdb8-5e3c891ef114
# ╠═b857361c-ea06-4703-9f89-214238c14f10
# ╠═e8703f11-aea1-4c8d-847c-272e04f6295c
# ╠═dfefabc0-ff41-4b5b-b3f9-0e2e415c472d
# ╠═433cd2ae-dae7-4d28-968a-753fc75128e5
# ╠═2ff3b3a4-c618-485a-adef-a498ddff6b44
# ╟─ebca4ad8-2cf2-4890-8fd6-49ecf5b95076
# ╠═38f8c509-00c1-466d-8f0b-7e65179095a3
# ╠═a78e967a-96f9-4e40-8b84-57b9324b95f1
# ╠═824b0622-b6d3-40c6-99da-998e0d0590f5
# ╠═5841350a-8287-4d67-ab5c-f5f1107240f1
# ╠═6caa0ffa-303b-441b-ade3-84dba5f1b5f9
# ╠═40cd5588-ecb5-4228-9007-3bbd0c293949
# ╠═a1a2dc92-6ad4-472f-93f2-3b6b4c17495d
# ╠═991d6686-550e-4bbb-86bd-f4f3238376ff
# ╠═a0729aa7-87be-4530-be6d-d83fc55631b8
# ╠═351da487-bad5-4f3b-82b1-802e958d2af0
# ╟─6531aae8-008f-4d20-81fe-58810b3b6492
# ╠═7784e2d4-ba1c-43a6-bd38-5826e7e84e7c
# ╠═21126b87-74cf-49bb-b7ec-d169dbe1adc6
# ╠═f9e11607-49be-41f9-a138-7358cb8412c7
# ╠═fac115e6-655e-4179-9ca9-bbffe74257c1
# ╠═33388b4f-334d-4a00-86b5-187724764a8f
# ╠═3adfcaf1-61ed-453a-a347-6c4909e351f9
# ╠═dc7a3f1e-c594-4323-ae1c-ba5edd3bde12
# ╟─f1ec6c25-f23d-46d8-a364-55012d9f1f0e
# ╠═feeb9c4d-b008-49f3-8ce8-d22d7ee46e1a
# ╟─82b3776d-5226-4935-96c2-1028a6f0b84c
# ╟─22d3f36d-1a0f-4873-bf95-99017d2faa1b
# ╠═89204946-0032-41d2-b8a9-5098f759855c
# ╠═f629cf94-b767-4c98-91ae-83ddad7ad319
# ╠═be293fa3-8160-4109-a95a-49199067235c
# ╠═a466e611-5780-4e31-9bc7-4d89053a5b45
# ╠═2ca40136-efdc-45b4-8f6b-c7f8a73616bf
# ╠═f092f12b-8026-4eaa-8c78-dc7a700a66e8
# ╠═2d3edb03-023e-4d2e-9354-40c4bc543e06
# ╟─da284c10-19b1-4a79-b6d3-6cb9e8f8d751
# ╠═bcdde5c4-d69b-409c-a7a9-f3f3b571a439
# ╠═7ffb6905-e417-44ff-8b5a-0c8adf242d2e
# ╠═dec533a8-e808-4042-96da-0ee1a0ec9e4b
# ╠═b92ff6ad-199e-44e4-a974-1ad804f8034a
# ╠═4b74f82d-994d-4cfa-8dea-80077bd44da5
# ╠═654d18cb-9f2a-4e59-97aa-fe74814ed151
# ╠═21a6fd05-7001-4572-b4e3-4fcd31a52be1
# ╟─f5c9086e-be9a-41e1-869b-3a879e9b44bf
# ╠═6cc684c7-22f7-4873-a2cb-63cf01189818
# ╠═11f2147b-3385-4ec0-808a-94d63b99aa9a
# ╠═5c22520d-50d6-4d46-a2c6-8a75028cf26c
# ╠═c8104c12-0459-4cf7-b830-faa316cedcbb
# ╠═60aad7ed-3d15-4536-aa43-457e53aac678
# ╠═025a9214-5c8b-4927-87c6-a5d14c0186e8
# ╠═af3582e2-2d4c-4ff0-b0e9-0d143b8f7579
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
