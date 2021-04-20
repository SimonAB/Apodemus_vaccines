using GLM, MixedModels
using Gadfly
using Cairo
using Random

# Import & filter data
include("wild_vacc_data_cleanup.jl")

# generate random normal response variable and merge into new dataframe
rng = MersenneTwister(820480);
randnormalOD = DataFrame(randn(rng, Float64, (nrow(both), 1)))
rename!(randnormalOD,:x1 => :randOD)
randnormalOD.Row = (both.Row)
refute = copy(both; copycols=true)
refute.randOD = (randnormalOD.randOD)
