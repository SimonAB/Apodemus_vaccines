#=
This script will wrangle the data from the raw data file into a clean data file.
=#

# PACKAGES
using CSV, DataFrames, Query
using CategoricalArrays
using StatsBase

# IMPORT raw df (full joint table) OR fall back to pre-filtered `clean_data.csv`

datadir = isdir("data") ? "data" : "../data"
joint_file = "joint_dataset_4analysis_checked.csv"
clean_file = "clean_data.csv"
joint_path = joinpath(datadir, joint_file)
clean_path = joinpath(datadir, clean_file)

if isfile(joint_path)
    rawdata = CSV.read(joint_path, DataFrame; missingstring="NA")
    _load_clean_only = false
elseif isfile(clean_path)
    rawdata = CSV.read(clean_path, DataFrame; missingstring="NA")
    _load_clean_only = true
else
    error(
        "No data file found. Place either `data/$(joint_file)` or `data/$(clean_file)` under the project root (see data/README.md).",
    )
end

# After redaction / Excel export some numeric columns may arrive as strings; coerce for modelling.
function _parse_numeric_column!(df::DataFrame, name::Symbol)
    hasproperty(df, name) || return
    col = df[!, name]
    if nonmissingtype(eltype(col)) <: AbstractString
        df[!, name] = map(col) do x
            ismissing(x) && return missing
            v = strip(string(x))
            isempty(v) && return missing
            return parse(Float64, v)
        end
    end
    return
end

for sym in [
    :Block, :Weight, :Fat_Scores_Dorsal, :Fat_Scores_Pelvic,
    :days_since_1st_trt, :days_since_2nd_trt, :days_since_1st_D_or_A, :days_since_1st_D_inj,
    :blank_avg, :blank_sd, :cutoff_new, :age_lab,
    :ticks_total, :fleas, :mites, :Testes, :H_poly, :Pinworms, :Cestode,
    :Col_5_bcen, :Col_6_bcen, :Col_7_bcen, :Col_5_avg, :Col_6_avg, :Col_7_avg,
    :OD_avg, :OD, :Fat_Scores_Sum,
    :ishigh, :ismale, :isrepro, :isadult, :islab, :isvax, :isHp, :nHp,
]
    _parse_numeric_column!(rawdata, sym)
end

# Clean data — skipped when loading `clean_data.csv` (already filtered & deduplicated upstream)
df = rawdata
if !_load_clean_only
    df = filter(row -> !ismissing(row.ID), df)
    df = filter(row -> !ismissing(row.Weight), df)
    df = filter(row -> !ismissing(row.Diet), df)
    df = filter(row -> !ismissing(row.repro_bin), df)
    df = filter(row -> !ismissing(row.days_since_1st_D_or_A) && row.days_since_1st_D_or_A ≥ 0, df)
end
# Optional filters (commented out)
# df = filter(row -> row.vax_history != "DD", df) # Keep only mice with single vaccination or adjuvant
# df = filter(row -> row.OD > 0, df) # Remove individuals who didn't seroconvert

# Select only the last complete entry for each mouse - more efficient approach
# Find maximum days_since_1st_D_or_A for each ID
grouped_df = groupby(df, :ID)
dataunique = combine(grouped_df, :days_since_1st_D_or_A => maximum => :days_since_1st_D_or_A)

# Join and ensure no repeated measures - using efficient DataFrame operations
df_unique = innerjoin(df, dataunique, on=[:ID, :days_since_1st_D_or_A])
df_unique = unique(df_unique, :ID) # Ensure no repeated measures
"Total number of individual wood mice: $(length(unique(df.ID)))"

# ENCODE & STANDARDISE

"""
    encode_df(df::DataFrame)

Encode and standardise variables for structural causal model analysis.

This function transforms the raw data into the variables used in the causal DAG,
including standardisation of continuous variables and creation of appropriate
node encodings for causal inference.

# Arguments
- `df::DataFrame`: Raw data frame containing vaccination experiment data

# Returns
- `DataFrame`: Processed data frame with encoded variables:
  - `E`: Standardised vaccine response (log-transformed OD)
  - `H`: Habitat (1=lab, 2=wild)
  - `V`: Vaccination status (1=adjuvant, 2=vaccine)
  - `D`: Diet supplementation (1=low, 2=high)
  - `R`: Reproductive status (1=non-reproductive, 2=reproductive)
  - `S`: Sex (1=male, 2=female)
  - `P`: Parasite infection status (1=uninfected, 2=infected)
  - `M`: Standardised body mass
  - `F`: Fat scores (raw)
  - `Ḟ`: Standardised fat scores with missingness preserved
  - `nP`: Parasite count (H. polygyrus + Cestodes)
  - `Vidx`: Vaccination history index (1=A, 2=D, 3=AD, 4=DA, 5=DD)

# Details
- Uses ZScore standardisation for continuous variables
- Preserves missing values in fat scores for Bayesian imputation
- Creates appropriate binary/categorical encodings for causal variables
"""
function encode_df(df)
    # create log OD response variable
    df.logOD = log.(10, 1 .+ df.OD)

    # encoding
    df.ismale = df.Sex .∈ Ref(["M"])
    df.issupplemented = df.Diet .∈ Ref(["High"])
    df.isreproductive = df.repro_bin .∈ Ref(["Reproductive"])
    df.iswild = df.Env .∈ Ref(["Wild"])

    # Ensure proper datatypes (no missing)
    df.Weight = Base.convert(Vector{Float64}, df.Weight)
    df.vax_history = CategoricalArray(Base.convert(Vector{String}, df.vax_history))
    df.ID = CategoricalArray(df.ID)

    # populate nodes
    df.E = standardize(ZScoreTransform, df.logOD, dims=1)
    df.M = standardize(ZScoreTransform, df.Weight, dims=1)
    df.F = df.Fat_Scores_Sum
    df.T = Base.convert(Vector{Float64}, df.days_since_1st_D_or_A)
    # df.T = standardize(ZScoreTransform, df.T, dims=1)
    df.nP = df.nHp .+ df.Cestode #.+ df.Pinworms;

    # index variable
    df.H = @. ifelse(df.Env == "Lab", 1, 2)
    df.D = @. ifelse(df.Diet == "Low", 1, 2)
    df.R = @. ifelse(df.repro_bin == "Reproductive", 2, 1)
    df.S = @. ifelse(df.Sex == "M", 1, 2)
    df.P = @. ifelse(df.isHp == 0, 1, 2)
    df.V = @. ifelse(df.isvax == 1, 2, 1)
    df.Vidx = @. ifelse(df.vax_history == "A", 1, ifelse(df.vax_history == "D", 2, ifelse(df.vax_history == "AD", 3, ifelse(df.vax_history == "DA", 4, 5))))

    # missingness
    t = Vector{Union{Missing,Float64}}(missing, nrow(df))
    present_mask_F = completecases(df, :F)
    t[present_mask_F] .= standardize(ZScoreTransform, Vector{Float64}(df.F[present_mask_F]), dims=1)
    df.Ḟ = t # create Ḟ, which is the standardised F with missing values

    return df
end

# Ensure the processed data is clean before saving
df_clean = copy(df)  # Create a clean copy

# save to disk with error handling
try
    # Use absolute path to avoid any relative path issues
    output_path = joinpath(dirname(@__DIR__), "data", "clean_data.csv")
    CSV.write(output_path, df_clean)
    println("Successfully saved clean data to: $output_path")
catch e
    println("Error saving data: ", e)
    println("Attempting alternative save location...")
    # Fallback: try saving in current directory
    CSV.write("clean_data.csv", df_clean)
    println("Saved clean data to current directory: clean_data.csv")
end

# print outcome
describe(df_clean)
