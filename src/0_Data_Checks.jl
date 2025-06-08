## Data exploration

# Load libraries

using DataFrames
using CSV
using StatsPlots
using LaTeXStrings
println("JULIA_NUM_THREADS: ", Threads.nthreads())

# Load data

datadir = "data"
datafile = "joint_dataset_4analysis_checked.csv"
filepath = isdir(datadir) ? joinpath("./", datadir, datafile) : joinpath("../", datadir, datafile)

df = CSV.File(filepath; select=[:Diet, :H_poly, :Pinworms, :Cestode, :Fat_Scores_Sum, :OD_avg], pool=true, missingstring="NA") |> DataFrame

"""
    data_check(df::DataFrame, variable_choice::String)

Perform exploratory data analysis for a specified variable, creating visualisations
showing the relationship between the variable and diet supplementation.

This function filters the data to remove zero values and missing observations,
then creates box plots to visualise the distribution of the variable by diet group.
For parasite count variables, it applies a log10(1+x) transformation.

# Arguments
- `df::DataFrame`: The DataFrame containing the data to be analysed
- `variable_choice::String`: The variable name to analyse. Should be one of:
  - "H_poly": H. polygyrus parasite count (will be log-transformed)
  - "Pinworms": Pinworm parasite count
  - "Cestode": Cestode parasite count
  - "Fat_Scores_Sum": Sum of fat scores
  - "OD_avg": Average optical density (vaccine response)

# Returns
- Nothing (creates plot as side effect)

# Details
- Filters out zero values and missing observations
- Applies log10(1+x) transformation to H_poly for better visualisation
- Creates box plots showing variable distribution by diet group
- Uses LaTeX formatting for axis labels where appropriate
"""
function data_check(df::DataFrame, variable_choice::String)

    df = filter(row -> !ismissing(row[variable_choice]) && row[variable_choice] != 0, df)

    ## Some quick plots

    # Plot
    if variable_choice == "H_poly"
        df[!, variable_choice*"_log1p"] = log10.(1 .+ df[!, variable_choice])
        Plots.boxplot(df.Diet, df[!, variable_choice*"_log1p"], seriestype=:scatter, legend=false, xlabel="Diet", ylabel=L"log_{10}" * "(1+" * variable_choice * ")", title="Diet vs " * L"log_{10}" * "(1+" * variable_choice * ")")
    else
        Plots.boxplot(df.Diet, df[!, variable_choice], seriestype=:scatter, legend=false, xlabel="Diet", ylabel=variable_choice, title="Diet vs " * variable_choice)
    end
end

data_check(df, "H_poly")

data_check(df, "Pinworms")

data_check(df, "Cestode")
