#=
Compute Cohen's d estimates for each main effect
Julia version: 1.12
Author: Simon A Babayan
=#

#=
This script computes Cohen's d estimates for each factor term in [:D, :R, :S, :V, :M, :Ḟ].
For each factor term `term`:
- Filter `df` where `term` is 1 (or the appropriate level)
- Compute `mean_d = mean(df.E_cohens_d[mask])`
- Compute `sem_d = std(df.E_cohens_d[mask]) / sqrt(count(mask))`
Collect into arrays `values`, `sems`, and `labels`.

REPL: Run `# %%` cells in order (e.g. VS Code / Cursor: “Julia: Execute Cell”).
=#

# %%
using DataFrames, Statistics

# %%
function compute_cohens_d_main_effects(df_infected)
    println("=== COMPUTING COHEN'S D ESTIMATES FOR MAIN EFFECTS ===")

    # Define the factor terms to analyze
    factors = [:D, :R, :S, :V, :M, :Ḟ]

    # Initialize arrays to store results
    values = Float64[]
    sems = Float64[]
    labels = String[]

    # Compute Cohen's d estimates for each factor
    for factor in factors
        println("\nProcessing factor: $factor")

        # Create mask for factor level 1
        mask = df_infected[:, factor] .== 1
        n_masked = sum(mask)

        if n_masked > 0
            # Extract Cohen's d values for this subset
            E_cohens_d_masked = df_infected[mask, :E_cohens_d]

            # Compute mean and SEM
            mean_d = mean(E_cohens_d_masked)
            sem_d = std(E_cohens_d_masked) / sqrt(n_masked)

            # Store results
            push!(values, mean_d)
            push!(sems, sem_d)
            push!(labels, string(factor))

            println("  n = $n_masked")
            println("  Mean Cohen's d = $(round(mean_d, digits=4))")
            println("  SEM = $(round(sem_d, digits=4))")
        else
            println("  Warning: No observations with $factor == 1")
            # Store NaN values for missing data
            push!(values, NaN)
            push!(sems, NaN)
            push!(labels, string(factor))
        end
    end

    # Print summary table
    println("\n" * "="^60)
    println("SUMMARY: COHEN'S D ESTIMATES FOR MAIN EFFECTS")
    println("="^60)
    println("Factor\t\tMean Cohen's d\t\tSEM")
    println("-"^50)
    for i in eachindex(labels)
        if !isnan(values[i])
            println("$(labels[i])\t\t$(round(values[i], digits=4))\t\t\t$(round(sems[i], digits=4))")
        else
            println("$(labels[i])\t\tNo data\t\t\tNo data")
        end
    end

    return (values=values, sems=sems, labels=labels)
end

# %%
# If this script is being run directly (not included), it will need the data
if abspath(PROGRAM_FILE) == @__FILE__
    println("This script needs to be run after loading the data from 4_SCM_intervention.jl")
    println("Include this file or copy the function to use it with your dag_df_infected DataFrame")
end
