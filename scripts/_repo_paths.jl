#=
Shared paths for repository-local scripts (no manuscript checkout required).
=#

"""Repository root (`Apodemus_vaccines_public/`)."""
repo_root() = normpath(joinpath(@__DIR__, ".."))

"""Default redacted analysis table distributed with this repository."""
default_joint_csv() = joinpath(repo_root(), "data", "joint_dataset_4analysis_checked.csv")

"""Optional encoded table produced by `src/DataWrangler.jl`."""
default_clean_csv() = joinpath(repo_root(), "data", "clean_data.csv")

"""Directory for tabular outputs (`results/tables/`)."""
function results_tables_dir()
    d = joinpath(repo_root(), "results", "tables")
    mkpath(d)
    return d
end
