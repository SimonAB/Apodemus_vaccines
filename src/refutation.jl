using GLM, MixedModels
using Gadfly
using Cairo
using Random
Pkg.add("Missings")
import Missings

# Import & filter data
include("wild_vacc_data_cleanup.jl")

# generate random normal response variable (dummy response) 
rng = MersenneTwister(820480);
randnormalOD = DataFrame(randn(rng, Float64, (nrow(both), 1)))
rename!(randnormalOD,:x1 => :randOD)
randnormalOD.Row = (both.Row)

# merge dummy response into new dataframe
refute = copy(both; copycols=true)
refute.randOD = (randnormalOD.randOD)

# generate all-zero variable (placebo treatment) 
placebo = DataFrame(Array{Union{Missing, Int}}(missing, nrow(both), 1))
rename!(placebo,:x1 => :placebodiet)
for col in names(placebo)
   placebo[col] = Missings.coalesce.(placebo[col], 0)
end

# merge placebo treatment into refute dataframe
refute.placebodiet = placebo.placebodiet
