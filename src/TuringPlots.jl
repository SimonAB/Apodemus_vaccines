#=
TuringPlots:
- Julia version: 1.10
- Author: Simon A Babayan
- Date: 2022-09-20
=#

using DataFrames
using AlgebraOfGraphics, CairoMakie
using AlgebraOfGraphics: density

CairoMakie.activate!(; type="svg")

# Plot chains

"""
# Plot MCMCChains with Makie using DataFrames

plot_chains(chns, res=(8, length(chains(chns))); show_intercept=false, show_traces=true) returns a
plot of the traces (optional) and densities of the coefficients and the intercept (optional) from sampled priors or posteriors of a Turing model using CairoMakie.

## Arguments
- `chns`: Sampled priors or posteriors of a Turing.jl model; must be supplied.
- `mask`: regular expression of the paramters to plot (default= r"α\b|β")
- `res`: tuple providing the resolution of the plot in inches (default=(8,6)). Automatically adjusted if only the densities are plotted.
- `show_intercept`: include estimate of the global intercept? (default=true)
- `show_traces`: should the traces be drawn? (default=true)
"""
function plot_chains_df(chns; mask=r"α\b|β", res=(8, 0.8 * length(names(DataFrame(chns)[!, mask]))), show_intercept=false, show_traces=true, xlab_dist="Parameter estimate")
    # Convert chains to DataFrame and filter by mask
    chns_df = DataFrame(chns)[!, mask]
    # Get number of chains and samples
    n_chains = length(chains(chns))
    n_samples = length(chns)
    panel = 1

    # Set the first level of each categorical factor as reference for the others
    ref_values = Dict()
    prefixes = unique([split(param, "[")[1] for param in names(chns_df) if contains(param, "[")])
    for prefix in prefixes
        ref_param = prefix * "[1]"
        if ref_param in names(chns_df)
            mean_val = mean(chns_df[:, ref_param])
            ref_values[prefix] = mean_val
        end
    end
    for (j, param) in enumerate(names(chns_df))
        prefix = split(param, "[")[1]
        if haskey(ref_values, prefix)
            chns_df[:, param] .-= ref_values[prefix]
        end
    end

    # Calculate the global mean of the intercept α
    global_mean_alpha = mean(chns_df[:, :α])

    # Determine parameters to plot
    pms = show_intercept ? Symbol.(names(chns_df)) : Symbol.(names(chns_df))[2:end]
    # Adjust size if intercept is shown
    res = show_intercept ? (res[1], res[2] + 1) : res

    # Set figure size
    size_inches = show_traces ? res : (res[1] ÷ 2, res[2])
    size_pt = 72 .* size_inches
    fig = Figure(; size=size_pt)

    # Plot traces for each parameter
    for (i, param) in enumerate(pms)
        ax = Axis(fig[i, panel]; ylabel=string(param))
        for chain in 1:n_chains
            values = chns[:, param, chain]
            CairoMakie.lines!(ax, 1:n_samples, values,
                color=(cgrad(:RdYlBu_10, n_chains, categorical=true)[chain], 0.7);
                linewidth=1,
                label=string(chain))
        end

        hideydecorations!(ax; label=false)
        if i < length(pms)
            hidexdecorations!(ax; grid=false)
        else
            hidexdecorations!(ax; grid=false, label=false)
            ax.xlabel = "Iteration"
        end
    end

    # Plot densities
    min_value = minimum(minimum(eachcol(chns_df)))
    max_value = maximum(maximum(eachcol(chns_df)))

    for (i, param) in enumerate(pms)
        ax = Axis(fig[i, panel+1]; ylabel=string(param))

        for chain in 1:n_chains
            values = chns[:, param, chain]
            CairoMakie.density!(ax, values,
                color=(cgrad(:RdYlBu_10, n_chains, categorical=true)[chain], 0.7),
                strokewidth=0.3,
                strokecolor=:grey40;
                label=string(chain)
            )
        end

        # Set x-axis limits to align density plots on value 0
        prefix = split(string(param), "[")[1]
        ref_value = get(ref_values, prefix, global_mean_alpha)
        xlims!(ax, (min_value, max_value))

        show_traces ? hideydecorations!(ax) : hideydecorations!(ax; label=false)

        if i < length(pms)
            hidexdecorations!(ax; grid=false)
        else
            hidexdecorations!(ax; grid=false, label=false)
            ax.xlabel = xlab_dist
        end
        vlines!(ax, ref_value, color=:grey10, linestyle=:dot)
        text!(ax, ref_value - 1, 0; text="–", color=:grey10, font=:bold)
        text!(ax, ref_value + 0.5, 0; text="+", color=:grey10, font=:bold)

    end
    # Link x-axes of all density plots
    axes = [only(contents(fig[i, panel+1])) for i in 1:length(pms)]
    linkxaxes!(axes...)
    # Set gaps between plots
    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 10)

    return fig
end

"""
# Plot MCMCChains with Makie

plot_chains(chns, res=(8, length(chains(chns))); show_intercept=false, show_traces=true) returns a
plot of the traces (optional) and densities of the coefficients and the intercept (optional) from sampled priors or posteriors of a Turing model using CairoMakie.

## Arguments
- `chns`: Sampled priors or posteriors of a Turing.jl model; must be supplied.
- `mask`: regular expression of the parameters to plot (default= r"α\b|β")
- `res`: tuple providing the resolution of the plot in inches. Automatically adjusted if only the densities are plotted.
- `show_intercept`: include estimate of the global intercept? (default=true)
- `show_traces`: should the traces be drawn? (default=true)
"""
function plot_chains(chns; mask=r"α\b|β", res=(10, 1.2 * length(filter(contains(mask), String.(names(chns, :parameters))))), show_intercept=true, show_traces=true)

    show_intercept ? res = (res[1], res[2] + 1) : mask = r"β"
    size_inches = res
    size_pt = 72 .* size_inches

    pms = Symbol.(filter(contains(mask), String.(names(chns, :parameters))))
    chain_mapping = mapping(pms .=> "Parameter estimate") * mapping(; color=:chain => nonnumeric, row=dims(1) => renamer(pms))

    if show_traces
        fig = Figure(; size=size_pt)
        plt1 = data(chns) * mapping(:iteration .=> "Iteration") * chain_mapping * visual(Lines)
        plt2 = data(chns) * chain_mapping * density()
        draw!(fig[1, 1], plt1)
        draw!(fig[1, 2], plt2; axis=(ylabel="Density",))
    else
        # plot only the densities
        fig = Figure(; size=size_pt)
        plt = data(chns) * chain_mapping * density()
        draw!(fig[1, 2], plt; axis=(ylabel="Density",))
    end
    return fig
end
