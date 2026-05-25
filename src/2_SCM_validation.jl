#=
SCM validation (Julia 1.12) — Simon A. Babayan

Eleven GLMM checks match the manuscript / SI: six pairwise marginal balance relations among
exogenous D, H, V, S, plus five auxiliary V–mediator screens. R dagitty's
impliedConditionalIndependencies() lists exactly those six pairwise relations; checks 7–11 are
extra screens for this study.

Optional: APODEMUS_SKIP_DAGITTY=1 skips RCall/dagitty (GLMM block still runs). After brew upgrade r,
point RCall at the active R: ENV["R_HOME"] from `R RHOME`, then Pkg.build("RCall"). CRAN dagitty
needs curl and V8; brew install curl helps, and Sys.setenv(DOWNLOAD_STATIC_V8=TRUE) before
install.packages("V8") avoids a local libv8 compile.

Run: julia --threads=auto --project=.

REPL: Run `# %%` cells in order (e.g. VS Code / Cursor: “Julia: Execute Cell”).
=#

# %%
print("Running on ", Threads.nthreads(), " threads.")
using DataFrames
using GLM, MixedModels
using StatsBase

const _SKIP_DAGITTY = get(ENV, "APODEMUS_SKIP_DAGITTY", "0") == "1"
if _SKIP_DAGITTY
    @info "APODEMUS_SKIP_DAGITTY=1: skipping RCall/dagitty (GLMM validation still runs)."
end

# %%

# --- helpers ---

_last_β(m) = coef(m)[end]
_last_p(m) = coeftable(m).cols[4][end]

function _println_glmm_result(title::AbstractString, m; p_value_only::Bool=false)
    p = round(_last_p(m), digits=3)
    if p_value_only
        println(title, ": p = ", p)
    else
        println(title, ": β = ", round(_last_β(m), digits=3), ", p = ", p)
    end
end

"""
    generate_independence_report(models, test_descriptions; pass_threshold=0.05, borderline_threshold=0.01)

Summarise conditional-independence screens; last fixed-effect coefficient in each model is the
tested contrast (predictor of interest last in each formula).
"""
function generate_independence_report(models, test_descriptions; pass_threshold=0.05, borderline_threshold=0.01)
    coefficients = [round(_last_β(m), digits=3) for m in models]
    p_values = [round(_last_p(m), digits=4) for m in models]

    test_results = DataFrame(
        Independence_Test=test_descriptions.independence_tests,
        Description=test_descriptions.descriptions,
        Coefficient=coefficients,
        P_Value=p_values,
    )

    test_results.Assessment = map(test_results.P_Value) do p
        if p >= pass_threshold
            "PASS"
        elseif p >= borderline_threshold
            "BORDERLINE"
        else
            "FAIL"
        end
    end

    println("\n", "="^80)
    println("CONDITIONAL INDEPENDENCE VALIDATION REPORT")
    println("="^80)
    println(test_results)

    n_pass = count(==("PASS"), test_results.Assessment)
    n_borderline = count(==("BORDERLINE"), test_results.Assessment)
    n_fail = count(==("FAIL"), test_results.Assessment)
    total_tests = nrow(test_results)

    println("\n", "-"^80)
    println("VALIDATION SUMMARY:")
    println("-"^80)
    println("Total tests: $total_tests")
    println("PASS (p ≥ $pass_threshold): $n_pass/$total_tests ($(round(100 * n_pass / total_tests, digits=1))%)")
    println("BORDERLINE ($borderline_threshold ≤ p < $pass_threshold): $n_borderline/$total_tests ($(round(100 * n_borderline / total_tests, digits=1))%)")
    println("FAIL (p < $borderline_threshold): $n_fail/$total_tests ($(round(100 * n_fail / total_tests, digits=1))%)")

    println("\nInterpretation:")
    println("• PASS: supports conditional independence at the chosen threshold")
    println("• BORDERLINE: weak evidence; consider follow-up")
    println("• FAIL: evidence against independence; revisit DAG or confounding")

    if n_fail > 0
        failing = test_results[test_results.Assessment.=="FAIL", :Independence_Test]
        println("\nWARNING: possible violations:")
        for t in failing
            println("   - ", t)
        end
        println("Consider revising the DAG or measured covariates.")
    elseif n_borderline > 0
        border = test_results[test_results.Assessment.=="BORDERLINE", :Independence_Test]
        println("\nNOTE: borderline tests:")
        for t in border
            println("   - ", t)
        end
    else
        println("\nAll conditional independence screens passed at the stated thresholds.")
    end

    return test_results
end

# %%

# --- data ---

if isdir("./src/")
    cd("./src/")
end
include("DataWrangler.jl")

df = encode_df(df)
df_sel = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df)
df_sel = filter(row -> !ismissing(row.Fat_Scores_Sum), df_sel)

dag_df = df_sel[!, [:E, :H, :V, :D, :R, :S, :M, :F, :T, :P, :nP, :Vidx, :vax_history, :ID]]
dag_df.F = standardize(ZScoreTransform, StatsBase.convert(Vector{Float64}, df_sel.F), dims=1)

# Single copy of the DAG string (matches manuscript / dagitty script; GLMM block does not parse it).
const SCM_VALIDATION_DAG_STRING =
    "dag{ D -> E; D -> F; D -> M; D -> P; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> P; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; R -> P; S -> E; S -> F; S -> M; S -> P; S -> R; V -> E; V -> F; V -> M; V -> P; V -> R; }"

if !_SKIP_DAGITTY
    include(joinpath(@__DIR__, "2_SCM_validation_dagitty.jl"))
end

# Bernoulli nodes: 0/1 coding for MixedModels Bernoulli()
dag_df.Vbinom = dag_df.V .- 1
dag_df.Sbinom = dag_df.S .- 1
dag_df.Rbinom = dag_df.R .- 1
dag_df.Pbinom = dag_df.P .- 1

# %%

println("\nTesting conditional independencies (mixed models; RE = ID + vax_history):")
println("="^51)

println("\n[1/11] D ⊥ S")
model_D_S = fit(MixedModel, @formula(D ~ (1 | ID) + (1 | vax_history) + S), dag_df)
_println_glmm_result("  Diet ~ Sex", model_D_S)

println("\n[2/11] D ⊥ H")
model_D_H = fit(MixedModel, @formula(D ~ (1 | ID) + (1 | vax_history) + H), dag_df)
_println_glmm_result("  Diet ~ Habitat", model_D_H)

println("\n[3/11] D ⊥ V")
model_D_V = fit(MixedModel, @formula(D ~ (1 | ID) + (1 | vax_history) + V), dag_df)
_println_glmm_result("  Diet ~ Vaccination (V)", model_D_V)

println("\n[4/11] H ⊥ S")
model_H_S = fit(MixedModel, @formula(H ~ (1 | ID) + (1 | vax_history) + S), dag_df)
_println_glmm_result("  Habitat ~ Sex", model_H_S)

println("\n[5/11] H ⊥ V  (Vbinom ~ H)")
model_H_V = fit(MixedModel, @formula(Vbinom ~ (1 | ID) + (1 | vax_history) + H), dag_df, Bernoulli())
_println_glmm_result("  Vaccination ~ Habitat", model_H_V)

println("\n[6/11] S ⊥ V  (Sbinom ~ V)")
model_S_V = fit(MixedModel, @formula(Sbinom ~ (1 | ID) + (1 | vax_history) + V), dag_df, Bernoulli())
_println_glmm_result("  Sex (binary) ~ Vaccination", model_S_V)

println("\n--- Auxiliary V–mediator screens (V last; p is for antigen vs adjuvant-only) ---")

println("\n[7/11] F ⊥ V | D, H")
model_F_V = fit(MixedModel, @formula(F ~ (1 | ID) + (1 | vax_history) + D + H + V), dag_df)
_println_glmm_result("  F | D,H,V (V coef)", model_F_V; p_value_only=true)

println("\n[8/11] M ⊥ V | D, H")
model_M_V = fit(MixedModel, @formula(M ~ (1 | ID) + (1 | vax_history) + D + H + V), dag_df)
_println_glmm_result("  M | D,H,V (V coef)", model_M_V; p_value_only=true)

println("\n[9/11] R ⊥ V | D, H")
model_R_V = fit(MixedModel, @formula(Rbinom ~ (1 | ID) + (1 | vax_history) + D + H + V), dag_df, Bernoulli())
_println_glmm_result("  R | D,H,V (V coef)", model_R_V; p_value_only=true)

println("\n[10/11] P ⊥ V | D, H")
model_P_V_DH = fit(MixedModel, @formula(Pbinom ~ (1 | ID) + (1 | vax_history) + D + H + V), dag_df, Bernoulli())
_println_glmm_result("  P | D,H,V (V coef)", model_P_V_DH; p_value_only=true)

println("\n[11/11] P ⊥ V | D, R, S, H")
model_P_V_full = fit(MixedModel, @formula(Pbinom ~ (1 | ID) + (1 | vax_history) + D + R + S + H + V), dag_df, Bernoulli())
_println_glmm_result("  P | D,R,S,H,V (V coef)", model_P_V_full; p_value_only=true)

test_info = (
    independence_tests=[
        "D ⊥ S | {ID, vax_history}",
        "D ⊥ H | {ID, vax_history}",
        "D ⊥ V | {ID, vax_history}",
        "H ⊥ S | {ID, vax_history}",
        "H ⊥ V | {ID, vax_history}",
        "S ⊥ V | {ID, vax_history}",
        "F ⊥ V | D, H, {ID, vax_history}",
        "M ⊥ V | D, H, {ID, vax_history}",
        "R ⊥ V | D, H, {ID, vax_history}",
        "P ⊥ V | D, H, {ID, vax_history}",
        "P ⊥ V | D, R, S, H, {ID, vax_history}",
    ],
    descriptions=[
        "Diet independent of Sex",
        "Diet independent of Habitat",
        "Diet independent of Vaccination",
        "Habitat independent of Sex",
        "Habitat independent of Vaccination",
        "Sex independent of Vaccination",
        "Fat score independent of V given D, H",
        "Body mass independent of V given D, H",
        "Reproductive status independent of V given D, H",
        "Parasite infection independent of V given D, H",
        "Parasite infection independent of V given D, R, S, H",
    ],
)

validation_results = generate_independence_report(
    [
        model_D_S,
        model_D_H,
        model_D_V,
        model_H_S,
        model_H_V,
        model_S_V,
        model_F_V,
        model_M_V,
        model_R_V,
        model_P_V_DH,
        model_P_V_full,
    ],
    test_info,
)