#=
SCM Validation
- Julia version: 1.11
- Author: Simon A Babayan
=#

#=
This script validates the Structural Causal Model (SCM) using conditional independence tests.
It tests whether the proposed causal structure is consistent with the observed data patterns
by examining implied conditional independencies from the DAG specification.
=#

## Import packages
print("Running on ", Threads.nthreads(), " threads.")
using CSV, DataFrames
using HypothesisTests
using GLM, MixedModels
using RCall
@rlibrary dagitty

## Import data
if isdir("./src/")
    cd("./src/")
end
include("DataWrangler.jl")

# Data preparation for validation
df = encode_df(df)
df_sel = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 4, df)
df_sel = filter(row -> !ismissing(row.Fat_Scores_Sum), df_sel)

# Select variables for DAG validation
dag_df = df_sel[!, [:E, :H, :V, :D, :R, :S, :M, :F, :T, :P, :nP, :Vidx, :vax_history, :ID]]

# Standardise fat scores for consistent scaling
dag_df.F = standardize(ZScoreTransform, StatsBase.convert(Vector{Float64}, df_sel.F), dims=1)

describe(dag_df)

## DAG specification and theoretical validation

# Define the proposed causal structure
dag = dagitty("dag{ D -> E; D -> F; D -> M; D -> P; D -> R; F -> E; F -> M; H -> E; H -> F; H -> M; H -> P; H -> R; M -> E; P -> E; P -> F; P -> M; R -> E; R -> F; R -> M; R -> P; S -> E; S -> F; S -> M; S -> P; S -> R; V -> E; V -> F; V -> M; V -> P; V -> R; }")

# Examine implied statistical relationships
impliedCovarianceMatrix(dag)
impliedConditionalIndependencies(dag)

# Quick conditional independence screening using local tests
localTests(dag, data=dag_df[!, [:E, :H, :V, :D, :R, :S, :M, :F, :P, :nP]])

## Rigorous conditional independence testing using mixed models

# The DAG implies specific conditional independencies that should hold if the causal structure is correct.
# We test these using mixed-effects models with appropriate random effects for repeated measures.

println("Testing conditional independencies implied by the DAG:")
println("=" * "="^50)

# Test 1: D ⊥ S | {ID, vax_history}
println("Testing D ⊥ S (Diet independent of Sex)")
form = @formula(D ~ (1 | ID) + (1 | vax_history) + S)
model_D_S = fit(MixedModel, form, dag_df)
println("Diet ~ Sex: ", round(coef(model_D_S)[end], digits=3), " (p = ", round(coeftable(model_D_S).cols[4][end], digits=3), ")")

# Test 2: D ⊥ H | {ID, vax_history}
println("\nTesting D ⊥ H (Diet independent of Habitat)")
form = @formula(D ~ (1 | ID) + (1 | vax_history) + H)
model_D_H = fit(MixedModel, form, dag_df)
println("Diet ~ Habitat: ", round(coef(model_D_H)[end], digits=3), " (p = ", round(coeftable(model_D_H).cols[4][end], digits=3), ")")

# Test 3: D ⊥ V | {ID, vax_history}
println("\nTesting D ⊥ V (Diet independent of Vaccination)")
form = @formula(D ~ (1 | ID) + (1 | vax_history) + V)
model_D_V = fit(MixedModel, form, dag_df)
println("Diet ~ Vaccination: ", round(coef(model_D_V)[end], digits=3), " (p = ", round(coeftable(model_D_V).cols[4][end], digits=3), ")")

# Test 4: H ⊥ S | {ID, vax_history}
println("\nTesting H ⊥ S (Habitat independent of Sex)")
form = @formula(H ~ (1 | ID) + (1 | vax_history) + S)
model_H_S = fit(MixedModel, form, dag_df)
println("Habitat ~ Sex: ", round(coef(model_H_S)[end], digits=3), " (p = ", round(coeftable(model_H_S).cols[4][end], digits=3), ")")

# Test 5: H ⊥ V | {ID, vax_history} (binary outcome)
println("\nTesting H ⊥ V (Habitat independent of Vaccination)")
dag_df.Vbinom = dag_df.V .- 1
form = @formula(Vbinom ~ (1 | ID) + (1 | vax_history) + H)
model_H_V = fit(MixedModel, form, dag_df, Bernoulli())
println("Vaccination ~ Habitat: ", round(coef(model_H_V)[end], digits=3), " (p = ", round(coeftable(model_H_V).cols[4][end], digits=3), ")")

# Test 6: S ⊥ V | {ID, vax_history} (binary outcome)
println("\nTesting S ⊥ V (Sex independent of Vaccination)")
dag_df.Sbinom = dag_df.S .- 1
form = @formula(Sbinom ~ (1 | ID) + (1 | vax_history) + V)
model_S_V = fit(MixedModel, form, dag_df, Bernoulli())
println("Vaccination ~ Sex: ", round(coef(model_S_V)[end], digits=3), " (p = ", round(coeftable(model_S_V).cols[4][end], digits=3), ")")

## Summary report of all conditional independence tests
"""
    generate_independence_report(models, test_descriptions; pass_threshold=0.05, borderline_threshold=0.01)

Generate a comprehensive validation report for conditional independence tests from fitted models.

# Arguments
- `models`: Vector of fitted MixedModel objects
- `test_descriptions`: Named tuple with `independence_tests` and `descriptions` vectors
- `pass_threshold`: P-value threshold for PASS assessment (default: 0.05)
- `borderline_threshold`: P-value threshold for BORDERLINE assessment (default: 0.01)

# Returns
- DataFrame with test results and assessments
- Prints formatted validation report to console

# Assessment Criteria
- PASS (p ≥ pass_threshold): Strong evidence supporting conditional independence
- BORDERLINE (borderline_threshold ≤ p < pass_threshold): Weak evidence, warrants investigation
- FAIL (p < borderline_threshold): Evidence against independence, suggests DAG misspecification
"""
function generate_independence_report(models, test_descriptions; pass_threshold=0.05, borderline_threshold=0.01)
    # Extract results from fitted models
    coefficients = [round(coef(model)[end], digits=3) for model in models]
    p_values = [round(coeftable(model).cols[4][end], digits=4) for model in models]

    # Create results DataFrame
    test_results = DataFrame(
        Independence_Test=test_descriptions.independence_tests,
        Description=test_descriptions.descriptions,
        Coefficient=coefficients,
        P_Value=p_values
    )

    # Categorise results based on p-values
    test_results.Assessment = map(test_results.P_Value) do p
        if p >= pass_threshold
            "PASS"
        elseif p >= borderline_threshold
            "BORDERLINE"
        else
            "FAIL"
        end
    end

    # Print formatted report
    println("\n" * "="^80)
    println("CONDITIONAL INDEPENDENCE VALIDATION REPORT")
    println("="^80)
    println(test_results)

    # Summary statistics
    n_pass = sum(test_results.Assessment .== "PASS")
    n_borderline = sum(test_results.Assessment .== "BORDERLINE")
    n_fail = sum(test_results.Assessment .== "FAIL")
    total_tests = nrow(test_results)

    println("\n" * "-"^80)
    println("VALIDATION SUMMARY:")
    println("-"^80)
    println("Total tests: $total_tests")
    println("PASS (p ≥ $pass_threshold): $n_pass/$total_tests ($(round(100*n_pass/total_tests, digits=1))%)")
    println("BORDERLINE ($borderline_threshold ≤ p < $pass_threshold): $n_borderline/$total_tests ($(round(100*n_borderline/total_tests, digits=1))%)")
    println("FAIL (p < $borderline_threshold): $n_fail/$total_tests ($(round(100*n_fail/total_tests, digits=1))%)")

    println("\nInterpretation:")
    println("• PASS: Strong evidence supporting conditional independence assumption")
    println("• BORDERLINE: Weak evidence, may warrant further investigation")
    println("• FAIL: Strong evidence against conditional independence, suggests DAG misspecification")

    # Actionable feedback based on results
    if n_fail > 0
        failing_tests = test_results[test_results.Assessment.=="FAIL", :Independence_Test]
        println("\n⚠️  WARNING: The following conditional independence assumptions are violated:")
        for test in failing_tests
            println("   - $test")
        end
        println("Consider revising the DAG structure or including additional confounders.")
    elseif n_borderline > 0
        borderline_tests = test_results[test_results.Assessment.=="BORDERLINE", :Independence_Test]
        println("\n⚡ NOTE: The following assumptions show borderline evidence:")
        for test in borderline_tests
            println("   - $test")
        end
        println("Monitor these relationships in subsequent analyses.")
    else
        println("\n✅ All conditional independence assumptions are well-supported by the data.")
    end

    return test_results
end

## Generate validation report

# Define test descriptions
test_info = (
    independence_tests=[
        "D ⊥ S | {ID, vax_history}",
        "D ⊥ H | {ID, vax_history}",
        "D ⊥ V | {ID, vax_history}",
        "H ⊥ S | {ID, vax_history}",
        "H ⊥ V | {ID, vax_history}",
        "S ⊥ V | {ID, vax_history}"
    ],
    descriptions=[
        "Diet independent of Sex",
        "Diet independent of Habitat",
        "Diet independent of Vaccination",
        "Habitat independent of Sex",
        "Habitat independent of Vaccination",
        "Sex independent of Vaccination"
    ]
)

# Generate comprehensive validation report
validation_results = generate_independence_report(
    [model_D_S, model_D_H, model_D_V, model_H_S, model_H_V, model_S_V],
    test_info
)
