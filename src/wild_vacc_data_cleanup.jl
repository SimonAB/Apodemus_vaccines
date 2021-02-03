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

vax = lab |>
      @filter(_.boost == 0) |>
      @dropna(:days_since_1st_D_inj) |>
      DataFrame

#   Specify how to correctly treat columns
categorical!(data, [:ID, :Sex, :Diet, :Treatment])
categorical!(lab, [:ID, :Sex, :Diet, :Treatment])
categorical!(vax, [:ID, :Sex, :Diet, :Treatment])

# Partition dataset into train (fit) and test rows
train, test = partition(unique(data.ID),
                      0.9,
                      shuffle = true,
                      rng = 551234,)

both = filter(:ID => in(Set(train)), data)