#=
TuringPlots:
- Julia version: 1.9
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
    chns_df = DataFrame(chns)[!, mask]
    n_chains = length(chains(chns))
    n_samples = length(chns)
    panel = 1

    if show_intercept
        pms = Symbol.(names(chns_df))
        res = (res[1], res[2] + 1)
    else
        pms = Symbol.(names(chns_df))[2:end]
    end

    if show_traces
        size_inches = res
        size_pt = 72 .* size_inches
        fig = Figure(; resolution=size_pt)
        # show_traces
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
                ax.xlabel = "Iteration"
            end
        end

        # densities
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

            show_traces ? hideydecorations!(ax) : hideydecorations!(ax; label=false)

            if i < length(pms)
                hidexdecorations!(ax; grid=false)
            else
                ax.xlabel = xlab_dist
            end
            vlines!(ax, 0, color=:grey10, linestyle=:dot)

        end
        axes = [only(contents(fig[i, panel+1])) for i in 1:length(pms)]
        linkxaxes!(axes...)
        rowgap!(fig.layout, 10)
        colgap!(fig.layout, 10)

    else
        # plot only the densities
        size_inches = (res[1] ÷ 2, res[2])
        size_pt = 72 .* size_inches
        fig = Figure(; resolution=size_pt)
        for (i, param) in enumerate(pms)
            ax = Axis(fig[i, 1]; ylabel=string(param))
            for chain in 1:n_chains
                values = chns[:, param, chain]
                CairoMakie.density!(ax, values,
                    # color=(RGB(215 / 255, 230 / 255, 244 / 255), 0.7),
                    color=(cgrad(:RdYlBu_10, n_chains, categorical=true)[chain], 0.7),
                    strokewidth=0.3,
                    strokecolor=:grey40;
                    label=string(chain)
                )
            end

            show_traces ? hideydecorations!(ax) : hideydecorations!(ax; label=false)

            if i < length(pms)
                hidexdecorations!(ax; grid=false)
            else
                ax.xlabel = xlab_dist
            end
            vlines!(ax, 0, color=:grey10, linestyle=:dot)
        end

        axes = [only(contents(fig[i, 1])) for i in 1:length(pms)]
        linkxaxes!(axes...)
    end
    rowgap!(fig.layout, 10)
    # rowsize!(fig.layout, 2, Relative(0.6))
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
        fig = Figure(; resolution=size_pt)
        plt1 = data(chns) * mapping(:iteration .=> "Iteration") * chain_mapping * visual(Lines)
        plt2 = data(chns) * chain_mapping * density()
        draw!(fig[1, 1], plt1)
        draw!(fig[1, 2], plt2; axis=(ylabel="Density",))
    else
        # plot only the densities
        fig = Figure(; resolution=size_pt)
        plt = data(chns) * chain_mapping * density()
        draw!(fig[1, 2], plt; axis=(ylabel="Density",))
    end
    return fig
end
