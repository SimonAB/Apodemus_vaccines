using DataFrames, Query
using CSV
using StatsBase
using MLJ

# Import & filter data

#import data & filter for just lab mice
raw_data = DataFrame!(CSV.File("./data/joint_dataset_4analysis.csv"; pool = true))

data = raw_data |>
    @dropna(:ID) |>
    @filter(_.days_since_1st_D_inj < 40) |>
    @filter(_.boost == 0) |>
    DataFrame

lab = data |>
      @filter(_.Env == "Lab") |>
      @dropna(:ID) |>
      DataFrame

vax = raw_data |> @dropna(:ID) |> @filter(_.boost == 0) |> DataFrame
categorical!(vax, :ID)

#   Specify how to correctly treat columns
categorical!(data, [:ID, :Sex, :Diet, :Treatment])
categorical!(lab, [:ID, :Sex, :Diet, :Treatment])
categorical!(vax, [:ID, :Sex, :Diet, :Treatment])

# Partition dataset into train (fit) and test rows
#filter for vaccination
data =
    raw_data |>
    @dropna(:ID) |>
    @dropna(:Weight) |>
    @filter(_.days_since_1st_D_inj > 7) |>
    @filter(_.boost == 0) |>
    DataFrame

train, test = partition(unique(data.ID), 0.9, shuffle = true, rng = 793426)

both = filter(:ID => in(Set(train)), data)

validate = filter(:ID => in(Set(test)), data)
validate = select(validate, ([:ID, :days_since_1st_D_inj, :Env, :Weight, :Diet, :Sex, :isvax]))
validate = sort(validate, [:ID, :days_since_1st_D_inj])
