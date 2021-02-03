using DataFrames, Query
using CSV
using StatsBase

# Import & filter data
raw_data = CSV.read("./data/elisa_data_subset_for_analysis.csv"; missingstrings=["NA"], pool=true, copycols=true)
dropmissing!(raw_data, :pittag) # remove blank wells and positive controls

# Check data
describe(raw_data)
countmap(raw_data.sex)
proportionmap(raw_data.sex)

# Remove missing pittag number
raw_data = dropmissing(raw_data, :pittag)


#filter and partition data
data =
    raw_data |>
    @dropna(:ID) |>
    @filter(_.days_since_1st_D_inj < 40) |>
    @filter(_.boost == 0) |>
    DataFrame

categorical(data.Env)

fit, test = partition(
    unique(data.ID),
    0.9,
    shuffle = true,
    rng = 551234,
)

both = data |> @filter(_.ID == fit) |> DataFrame
