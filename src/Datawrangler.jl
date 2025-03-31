#=
This script will wrangle the data from the raw data file into a clean data file.
=#

# PACKAGES
using CSV, DataFrames, Query
using CategoricalArrays
using StatsBase

# IMPORT raw df

datadir = "data"
datafile = "joint_dataset_4analysis_checked.csv"
filepath = isdir(datadir) ? joinpath("./", datadir, datafile) : joinpath("../", datadir, datafile)
rawdata = CSV.File(filepath;
    missingstring="NA", pool=true) |>
          DataFrame

# Clean things up
df =
    rawdata |>
    @dropna(:ID) |> # remove entries which lack lab ID or PIT tag IDs (cannot be imputed)
    @dropna(:Weight) |> # remove entries which lack a body mass measurement
    @dropna(:Diet) |> # remove entries which lack a Diet value
    @dropna(:repro_bin) |> # remove entries which lack a repro value
    @filter(_.days_since_1st_D_or_A ≥ 0) |> # remove entries which were measured less than a week after vaccination
    # @filter(_.vax_history != "DD") |> # keep only mice with a single vaccination or adjuvant
    # @filter(_.OD > 0) |> # remove individuals who didn't seroconvert
    DataFrame

# select only the last complete entry for each mouse
dataunique = df |>
             @groupby(_.ID) |>
             @map({ID = key(_), days_since_1st_D_or_A = maximum(_.days_since_1st_D_or_A)}) |>
             DataFrame;

# join the rest of the columns
df_unique = df |>
            @join(dataunique, _.ID, _.ID, {_..., __...}) |>
            @unique(_.ID) |> # ensure no repeated measures
            DataFrame;
"Total number of individual wood mice: $(length(unique(df.ID)))"

# ENCODE & STANDARDISE

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

# save to disk
CSV.write("../data/clean_data.csv", df)

# print outcome
describe(df)
