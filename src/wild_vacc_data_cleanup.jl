using DataFrames, Query
using CategoricalArrays
using CSV
using StatsBase
using MLJ

# Import & filter data

raw_data = DataFrame(CSV.File("./data/joint_dataset_4analysis.csv"; pool = true))


# Partition dataset into train (fit) and test rows
#filter for vaccination
data =
    raw_data |>
    @dropna(:ID) |>
    @dropna(:Weight) |>
    @filter(_.days_since_1st_D_inj > 7) |>
    @filter(_.boost == 0) |>
    DataFrame

#   Specify how to correctly treat columns
coerce!(data,:ID => Multiclass,
             :Sex => Multiclass,
             :Diet => Multiclass,
             :Treatment => Multiclass)

train, test = partition(unique(data.ID), 0.9, shuffle = true, rng = 793426)

both = filter(:ID => in(Set(train)), data)

# filter for just test set
valid = filter(:ID => in(Set(test)), data)
