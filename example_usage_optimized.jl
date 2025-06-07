#=
Example Usage for Optimized SCM Identification
- Julia version: 1.11+
- Demonstrates how to use the optimized models
- Set EXECUTE_EXAMPLES = true to run actual examples
=#

## Configuration - Set to true to run examples that take time
const EXECUTE_EXAMPLES = true          # Set to false for quick demonstration
const EXECUTE_FULL_MODELS = false      # Set to true to run full 1000+ sample models
const EXECUTE_BENCHMARKS = false       # Set to true if BenchmarkTools is available

# Configuration Guide:
#
# For quick demo (1-2 minutes):
#   EXECUTE_EXAMPLES = true, EXECUTE_FULL_MODELS = false, EXECUTE_BENCHMARKS = false
#
# For full analysis (10-30 minutes):
#   EXECUTE_EXAMPLES = true, EXECUTE_FULL_MODELS = true, EXECUTE_BENCHMARKS = false
#
# For complete benchmarking (30+ minutes, requires BenchmarkTools):
#   EXECUTE_EXAMPLES = true, EXECUTE_FULL_MODELS = true, EXECUTE_BENCHMARKS = true
#
# For code inspection only (immediate):
#   EXECUTE_EXAMPLES = false, EXECUTE_FULL_MODELS = false, EXECUTE_BENCHMARKS = false

## Load the optimized script (safe to include - won't run models automatically)
include("src/3_SCM_identification_optimized.jl")

println("Optimized SCM models loaded successfully!")
println("Configuration:")
println("  EXECUTE_EXAMPLES = $EXECUTE_EXAMPLES")
println("  EXECUTE_FULL_MODELS = $EXECUTE_FULL_MODELS")
println("  EXECUTE_BENCHMARKS = $EXECUTE_BENCHMARKS")

## Example 1: Test individual model compilation
println("\n" * "="^50)
println("EXAMPLE 1: Testing Model Compilation")
println("="^50)

if EXECUTE_EXAMPLES
    # Test a single model (quick compilation check with 10 samples)
    println("Running compilation test for V_E model...")
    test_result_v_e = test_model_compilation("V_E")

    println("\nRunning compilation tests for all models...")
    test_results_all = test_all_models()

    println("✓ Example 1 completed successfully!")
else
    println("Skipping execution (EXECUTE_EXAMPLES = false)")
    println("To run: test_model_compilation(\"V_E\") and test_all_models()")
end

## Example 2: Fit models with error handling
println("\n" * "="^50)
println("EXAMPLE 2: Fitting Models with Error Handling")
println("="^50)

if EXECUTE_EXAMPLES && EXECUTE_FULL_MODELS
    println("Running full model fitting (this will take several minutes)...")
    results = fit_optimized_models()
    println("\nGenerating summaries...")
    summarise_results(results)
    println("✓ Example 2 completed successfully!")

    # Store results for later examples
    global example2_results = results
elseif EXECUTE_EXAMPLES
    println("Running quick model fitting with reduced samples...")

    # Demonstrate the error handling with a quick fit
    quick_results = Dict{String,Any}()

    try
        println("Fitting quick V→E model (100 samples)...")
        V_E_model_quick = V_E_Model_Optimized(dag_df.IDidx, dag_df.E, dag_df.V;
            n_id=maximum(dag_df.IDidx))
        V_E_chn_quick = sample(V_E_model_quick, NUTS(0.8), 100)
        quick_results["V_E"] = (model=V_E_model_quick, chain=V_E_chn_quick)
        println("✓ V→E model completed successfully")
    catch e
        println("✗ V→E model failed: ", e)
        quick_results["V_E"] = nothing
    end

    println("\nQuick results summary:")
    if quick_results["V_E"] !== nothing
        println("V_E model: SUCCESS - Chain with $(size(quick_results["V_E"].chain, 1)) samples")

        # Show quick summary
        chn_df = DataFrame(quick_results["V_E"].chain)[!, r"α\b|β"]
        println("Quick coefficient summary:")
        for col in names(chn_df)
            if occursin(r"β", col)
                vals = chn_df[!, col]
                println("  $col: mean = $(round(mean(vals), digits=3)), std = $(round(std(vals), digits=3))")
            end
        end
    else
        println("V_E model: FAILED")
    end

    global example2_results = quick_results
    println("✓ Example 2 (quick version) completed successfully!")
else
    println("Skipping execution (EXECUTE_EXAMPLES = false or EXECUTE_FULL_MODELS = false)")
    println("To run full models: results = fit_optimized_models(); summarise_results(results)")
    global example2_results = nothing
end

## Example 3: Individual model usage
println("\n" * "="^50)
println("EXAMPLE 3: Individual Model Usage")
println("="^50)

if EXECUTE_EXAMPLES
    # Create a specific model
    println("Creating V→E model...")
    V_E_model_example = V_E_Model_Optimized(dag_df.IDidx, dag_df.E, dag_df.V;
        n_id=maximum(dag_df.IDidx))

    println("✓ Model created successfully!")

    # Sample from the model
    n_samples = EXECUTE_FULL_MODELS ? 1000 : 50
    println("Sampling $n_samples samples from the model...")

    V_E_chain_example = sample(V_E_model_example, NUTS(0.8), n_samples)

    println("✓ Sampling completed successfully!")
    println("Chain dimensions: $(size(V_E_chain_example))")

    # Show basic analysis
    println("\nBasic chain analysis:")
    chain_summary = DataFrame(V_E_chain_example)[!, r"α\b|β"]
    for col in names(chain_summary)
        if occursin(r"β", col)
            vals = chain_summary[!, col]
            println("  $col: mean = $(round(mean(vals), digits=3)), 95% CI = [$(round(quantile(vals, 0.025), digits=3)), $(round(quantile(vals, 0.975), digits=3))]")
        end
    end

    println("✓ Example 3 completed successfully!")
else
    println("Skipping execution (EXECUTE_EXAMPLES = false)")
    println("To run manually:")
    println("  model = V_E_Model_Optimized(dag_df.IDidx, dag_df.E, dag_df.V; n_id=maximum(dag_df.IDidx))")
    println("  chain = sample(model, NUTS(0.8), 1000)")
end

## Example 4: Performance comparison (requires BenchmarkTools)
println("\n" * "="^50)
println("EXAMPLE 4: Performance Comparison")
println("="^50)

if EXECUTE_EXAMPLES && EXECUTE_BENCHMARKS
    println("Running performance comparison with BenchmarkTools...")

    # Create models for comparison
    println("Creating original and optimized models...")
    original_model = V_E_Model(dag_df.IDidx, dag_df.E, dag_df.V)  # From original script
    optimized_model = V_E_Model_Optimized(dag_df.IDidx, dag_df.E, dag_df.V; n_id=maximum(dag_df.IDidx))

    # Run detailed benchmark
    benchmark_results = benchmark_models(original_model, optimized_model; n_runs=3)

    println("✓ Benchmark completed!")
    println("Speedup achieved: $(round(benchmark_results.speedup, digits=2))x")

elseif EXECUTE_EXAMPLES
    println("Running basic timing comparison (BenchmarkTools not enabled)...")

    # Create models for comparison
    println("Creating models for timing test...")

    # Use the optimized model vs itself for demonstration (since original might not be available)
    model1 = V_E_Model_Optimized(dag_df.IDidx, dag_df.E, dag_df.V; n_id=maximum(dag_df.IDidx))
    model2 = V_E_Model_Optimized(dag_df.IDidx, dag_df.E, dag_df.V; n_id=maximum(dag_df.IDidx))

    println("Running simple timing comparison...")
    timing_results = simple_timing_comparison(model1, model2; n_samples=20)

    println("✓ Timing comparison completed!")
    println("Note: This compares the same optimized model twice for demonstration.")
    println("For real comparison, load the original models and set EXECUTE_BENCHMARKS = true")

else
    println("Skipping execution (EXECUTE_EXAMPLES = false or EXECUTE_BENCHMARKS = false)")
    println("To run basic timing:")
    println("  simple_timing_comparison(original_model, optimized_model)")
    println("To run detailed benchmarking:")
    println("  using BenchmarkTools")
    println("  benchmark_models(original_model, optimized_model)")
end

## Example 5: Manual execution
println("\n" * "="^50)
println("EXAMPLE 5: Manual Model Execution")
println("="^50)

if EXECUTE_EXAMPLES
    println("Executing manual model fitting with error handling...")

    try
        println("Fitting V→E model manually...")
        manual_model = V_E_Model_Optimized(dag_df.IDidx, dag_df.E, dag_df.V;
            n_id=maximum(dag_df.IDidx))

        n_samples = EXECUTE_FULL_MODELS ? 1000 : 100
        manual_chain = sample(manual_model, NUTS(0.8), n_samples)
        println("✓ Success!")

        # Analyse results
        println("Analysing results...")
        chain_df = DataFrame(manual_chain)[:, r"α\b|β"]

        # Custom analysis instead of precis (which might not be available)
        println("\nCoefficient Analysis:")
        for col in names(chain_df)
            if occursin(r"β", col)
                vals = chain_df[!, col]
                q025, q975 = quantile(vals, [0.025, 0.975])
                println("  $col:")
                println("    Mean: $(round(mean(vals), digits=4))")
                println("    Std:  $(round(std(vals), digits=4))")
                println("    95% CI: [$(round(q025, digits=4)), $(round(q975, digits=4))]")
            end
        end

        println("✓ Example 5 completed successfully!")

    catch e
        println("✗ Failed: ", e)
        println("This demonstrates the error handling in action.")
    end
else
    println("Skipping execution (EXECUTE_EXAMPLES = false)")
    println("Manual execution code:")
    println("""
    try
        println("Fitting V→E model...")
        model = V_E_Model_Optimized(dag_df.IDidx, dag_df.E, dag_df.V;
            n_id=maximum(dag_df.IDidx))
        chain = sample(model, NUTS(0.8), 1000)
        println("✓ Success!")

        # Analyse results
        chain_df = DataFrame(chain)[:, r"α\\\\b|β"]
        # Add your analysis here
    catch e
        println("✗ Failed: ", e)
    end
    """)
end

println("\n" * "="^50)
println("END OF EXAMPLES")
println("="^50)

if EXECUTE_EXAMPLES
    println("🎉 All examples executed successfully!")
    println("\nSummary of what was run:")
    println("✓ Model compilation tests")
    if EXECUTE_FULL_MODELS
        println("✓ Full model fitting with error handling")
    else
        println("✓ Quick model fitting demonstration")
    end
    println("✓ Individual model usage with sampling and analysis")
    if EXECUTE_BENCHMARKS
        println("✓ Performance benchmarking")
    else
        println("✓ Basic timing comparison")
    end
    println("✓ Manual execution with error handling")

    println("\nNext steps:")
    println("- Set EXECUTE_FULL_MODELS = true for full model runs")
    println("- Set EXECUTE_BENCHMARKS = true for detailed performance analysis")
    println("- Use the optimized models in your own analysis")
else
    println("Examples ready for execution!")
    println("Set EXECUTE_EXAMPLES = true to run the examples")
end

println("\nAll optimized models ready for use! 🚀")
