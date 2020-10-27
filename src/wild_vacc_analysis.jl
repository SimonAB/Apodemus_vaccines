using DataFrames, GLM, MixedModels
using Gadfly

#import data 
wild = CSV.read("./data/wild.csv"; missingstrings=["NA"], pool=true, copycols=true)


# linear effect dominated by values <10 days
model1 = lm(@formula(days_since_1st_D_inj ~ OD_avg), wild)
p1 = plot(
    wild,
    y = :OD_avg,
    x = :days_since_1st_D_inj,
    Geom.point,
    color = :Boost,
    Guide.xlabel("days since vaccination"),
    Guide.ylabel("antibody OD"),
    Geom.vline(style = :dash, color = "red"),
    xintercept = [10]
)

wild = wild |>
  @filter(_.:days_since_1st_D_inj > 10) |>
  DataFrame

model2 = lm(@formula(days_since_1st_D_inj ~ OD_avg), wild)
p2 = plot(
    wild,
    y = :OD_avg,
    x = :days_since_1st_D_inj,
    Geom.point,
    color = :Boost,
    Guide.xlabel("days since vaccination"),
    Guide.ylabel("antibody OD"),
)
 
