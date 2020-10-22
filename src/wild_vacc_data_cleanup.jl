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


# import OD data
data = CSV.read(".data/joint_dataset_4analysis.csv"; missingstrings=["NA"], pool=true, copycols=true)

#remove positive controls
dropmissing!(data, :ID)
categorical!(data, :ID)

#remove lab mice
dropmissing!(data, :grid)
countmap(data.ID)
