using DataFrames, Query
using CategoricalArrays
using CSV
using StatsBase
using MLJ

# Import dataset
raw_data = DataFrame(CSV.File("./data/joint_dataset_4analysis.csv"; missingstring="NA", pool=true))

# filter for vaccination & seroconversion
data =
raw_data |>
@dropna(:ID) |> # remove entries which lack lab ID or PIT tag IDs
@dropna(:Weight) |> # remove entries which lack a body mass measurement
@filter(_.days_since_1st_D_inj > 7) |> # remove entries which were measured less than a week after vaccination
@filter(_.boost == 0) |> # remove entries which were vaccinated twice
@filter(_.OD > 0) |> # remove individuals who didn't seroconvert
DataFramePK

# Specify how to correctly treat columns
coerce!(data,:ID => Union{Missing,Multiclass}, # treat these columns as factors which can also handle NAs
             :Sex => Union{Missing,Multiclass},
             :Diet => Union{Missing,Multiclass},
             :Treatment => Union{Missing,Multiclass})

# create log OD response variable
data.logOD = log.(10, 1 .+ data.OD) 

# create iswild
data.iswild = data[:,:islab] .- 1 # dataset only had "islab", so -1 and squaring gives the opposite
data.iswild = data[:,:iswild].^2

# create islow
data.islow = data[:,:ishigh] .- 1 # as above
data.islow = data[:,:islow].^2

# Partition dataset into train (fit) and test rows
trainIDs, testIDs = partition(unique(data.ID), 0.9, shuffle=true, rng=224567)

# filter for just train set
train = filter(:ID => in(Set(trainIDs)), data)

# filter for just test set
test = filter(:ID => in(Set(testIDs)), data)
