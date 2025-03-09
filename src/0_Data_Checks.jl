# Data exploration

# %% Load libraries

using DataFrames
using CSV
using StatsPlots
using LaTeXStrings

# %% Load data

datadir = "data"
datafile = "joint_dataset_4analysis_checked.csv"
filepath = isdir(datadir) ? joinpath("./", datadir, datafile) : joinpath("../", datadir, datafile)

df = CSV.File(filepath; select=[:Diet, :H_poly, :Pinworms, :Cestode, :Fat_Scores_Sum, :OD_avg], pool=true, missingstring="NA") |> DataFrame

"""
    data_check(df, variable_choice)

This function takes a DataFrame `df` and a string `variable_choice` as input.
It filters out rows where the value of the column specified by `variable_choice` is 0,
and then drops rows with missing values.
It then creates a scatter plot of the variable against the 'Diet' column.
If the variable is 'H_poly', it applies a log transformation before plotting.

# Arguments
- `df::DataFrame`: The DataFrame to be checked.
- `variable_choice::String`: The variable to be checked. This should be one of "Pinworms", "Cestode", "Fat_Scores_Sum", or "OD_avg".

# Returns
Nothing. The function creates a plot as a side effect.

"""
function data_check(df::DataFrame, variable_choice::String)
    df = df[df[!, variable_choice].!=0, :]
    dropmissing!(df)

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
