using DataFrames, Query
using CSV
using StatsBase

# Import data
raw_data = CSV.read("./data/elisa_data_subset_for_analysis.csv"; missingstrings=["NA"], pool=true, copycols=true)

# Check data
describe(raw_data)
countmap(raw_data.sex)
proportionmap(raw_data.sex)
