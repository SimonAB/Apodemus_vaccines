using DataFrames, Query
using CSV
using StatsBase

# Import & filter data

#import data & filter for just lab mice
data = DataFrame!(CSV.File("./data/joint_dataset_4analysis.csv"; pool = true))

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
