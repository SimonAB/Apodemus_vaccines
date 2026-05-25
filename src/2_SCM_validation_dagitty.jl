#=
DAG checks via R dagitty (R + CRAN dagitty; Julia RCall built for that R).
Requires parent to define Main.SCM_VALIDATION_DAG_STRING before `include`.

REPL: Run `# %%` cells in order if editing this file in isolation.
=#
# %%
using RCall
reval("library(dagitty)")
reval("dag <- dagitty(\"" * Main.SCM_VALIDATION_DAG_STRING * "\")")

println("dagitty: implied covariance matrix (linear-Gaussian parametrisation)")
reval("print(round(impliedCovarianceMatrix(dag), 4))")

println("dagitty: implied marginal/conditional independencies (six pairwise exogenous relations; GLMMs 1–6)")
reval("print(impliedConditionalIndependencies(dag))")

# %%
println("dagitty: local tests (Gaussian exploratory; project GLMMs are authoritative)")
rd = dag_df[!, [:E, :H, :V, :D, :R, :S, :M, :F, :P, :nP]]
@rput rd
reval("print(localTests(dag, data = rd))")
