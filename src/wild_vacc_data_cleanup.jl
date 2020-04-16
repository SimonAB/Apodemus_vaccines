using DataFrames, Query
using CSV

# Import data
raw_data = CSV.read("./data/elisa_data_subset_for_analysis.csv"; missingstrings=["NA"], pool=true, copycols=true)