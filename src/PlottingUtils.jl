#=
PlottingUtils.jl
- Julia version: 1.11
- Author: Simon A Babayan
=#

#=
Domain-specific plotting utilities for vaccine efficacy and counterfactual analysis.
Contains plotting functions for:
- Counterfactual effect visualizations
- Interaction plots (Sex × Reproductive status)
- Clinical significance summaries
- Effect size visualizations
- Cohen's d plotting functions
=#

using DataFrames
using CairoMakie
using Statistics: mean, std
using Colors

# Import safe_plot_save from TuringPlots for convenience
include("TuringPlots.jl")

# Clinical significance thresholds and colors for reference lines
thresholds = [0.2, 0.5, 0.8]
colors = Dict(0.2=>:black, 0.5=>:black, 0.8=>:black)

"""
    plot_E_factual_counterfactual(df; saveplot=false)

Create a visualisation comparing factual and counterfactual vaccine response distributions.

# Arguments
- `df::DataFrame`: DataFrame containing the data with columns:
    - E_factual_mean: Factual vaccine response (with parasites)
    - E_counterfactual_mean: Counterfactual vaccine response (without parasites)
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Details
- Plots histograms of factual and counterfactual vaccine responses from generative models
- Factual data shown in light blue-grey (with parasites)
- Counterfactual data shown in orange (without parasites)
- Includes a legend in the top-left corner
- Axis labels are bold and sized at 16pt

# Returns
- `Figure`: A CairoMakie figure containing the distribution comparison plot
"""
function plot_E_factual_counterfactual(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])

    # Define custom colour (#c0c7db) - more efficient color definition
    light_blue_grey = RGB(0.7529f0, 0.7804f0, 0.8588f0)  # Use Float32 for efficiency

    # Plot histograms of factual and counterfactual responses
    hist!(ax, df.E_factual_mean, color=light_blue_grey, label="Factual (with parasites)")
    hist!(ax, df.E_counterfactual_mean, color=:orange, label="Counterfactual (without parasites)")
    axislegend(ax, position=:lt)

    # Configure axis labels and styling
    ax.xlabel = "Vaccine response"
    ax.ylabel = "Population count"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    # Save plot if requested
    if saveplot
        safe_plot_save("E_factual_counterfactual.pdf", fig)
    end
    fig
end

"""
    plot_counterfactual_effects(df; saveplot=false)

Create a visualisation of the counterfactual effect of parasite elimination on vaccine response for each mouse.
Optimized version with better performance for large datasets.

# Arguments
- `df::DataFrame`: DataFrame containing the data with columns:
    - IDidx: Mouse identifier
    - nP: Observed parasite count
    - E_factual_mean: Factual vaccine response (with parasites)
    - E_counterfactual_mean: Counterfactual vaccine response (without parasites)
    - S: Sex (1=male, 2=female)
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Details
- Plots factual and counterfactual vaccine responses from generative models for each mouse
- Uses vertical lines to connect factual and counterfactual points
- Orange lines indicate increased vaccine response under parasite elimination
- Black lines indicate decreased or unchanged vaccine response
- Factual points shown as circles (with parasites)
- Counterfactual points shown as triangles (without parasites)
- Sex-specific colours for point outlines

# Returns
- `Figure`: A CairoMakie figure containing the counterfactual effect plot
"""
function plot_counterfactual_effects(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])

    # Define colours - use symbols for efficiency
    male_color = :steelblue
    female_color = :crimson
    light_blue_grey = RGB(0.7529f0, 0.7804f0, 0.8588f0)  # Float32 for efficiency

    # Get unique mice once for efficiency
    unique_mice = unique(df.IDidx)

    # Pre-allocate vectors for batch plotting
    factual_x = Float64[]
    factual_y = Float64[]
    counterfactual_x = Float64[]
    counterfactual_y = Float64[]
    line_segments = Pair{Point2f,Point2f}[]

    for mouse in unique_mice
        mouse_mask = df.IDidx .== mouse
        mouse_data = view(df, mouse_mask, :)  # Use view for efficiency

        # Determine sex colour for the outline - assumes sex is constant for a mouse
        sex_outline_color = mouse_data.S[1] == 1 ? male_color : female_color

        for i in 1:length(mouse_data.nP)
            x_val = mouse_data.nP[i]
            factual_val = mouse_data.E_factual_mean[i]
            counterfactual_val = mouse_data.E_counterfactual_mean[i]

            # Store data for batch plotting
            push!(factual_x, x_val)
            push!(factual_y, factual_val)
            push!(counterfactual_x, x_val)
            push!(counterfactual_y, counterfactual_val)

            # Draw vertical lines connecting factual and counterfactual points
            line_color = factual_val < counterfactual_val ? :orange : :black
            lines!(ax, [x_val, x_val], [factual_val, counterfactual_val],
                color=line_color, linewidth=3, alpha=0.5)
        end

        # Plot factual and counterfactual points with sex-based outlines
        mouse_indices = findall(mouse_mask)
        scatter!(ax, mouse_data.nP, mouse_data.E_factual_mean,
            color=light_blue_grey, strokecolor=sex_outline_color, strokewidth=2,
            label="Factual (with parasites)", markersize=16, alpha=0.7)
        scatter!(ax, mouse_data.nP, mouse_data.E_counterfactual_mean,
            color=:orange, strokecolor=sex_outline_color, strokewidth=2,
            label="Counterfactual (without parasites)", marker=:utriangle,
            markersize=16, alpha=0.7)
    end

    # Configure axis labels and styling
    ax.xlabel = "Observed Parasite Count"
    ax.ylabel = "Vaccine response"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    # Add legend with sex-specific elements
    pre_male = MarkerElement(color=light_blue_grey, marker=:circle, markersize=16,
        strokecolor=male_color, strokewidth=2)
    pre_female = MarkerElement(color=light_blue_grey, marker=:circle, markersize=16,
        strokecolor=female_color, strokewidth=2)
    post_male = MarkerElement(color=:orange, marker=:utriangle, markersize=16,
        strokecolor=male_color, strokewidth=2)
    post_female = MarkerElement(color=:orange, marker=:utriangle, markersize=16,
        strokecolor=female_color, strokewidth=2)

    axislegend(ax,
        [pre_male, pre_female, post_male, post_female],
        ["Factual (Male)", "Factual (Female)", "Counterfactual (Male)", "Counterfactual (Female)"],
        position=:rt)

    # Save plot if requested
    if saveplot
        safe_plot_save("counterfactual_effects_on_E.pdf", fig)
    end

    fig
end

"""
    plot_counterfactual_effects_with_significance(df; saveplot=false)

Enhanced version of counterfactual effects plot with clinical significance color coding.
Uses Cohen's d thresholds to indicate practical significance of improvements.

# Arguments
- `df::DataFrame`: DataFrame containing the data with clinical significance columns
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Details
- Line colors indicate clinical significance of improvement:
  - Red: Large effect (|Cohen's d| ≥ 0.8)
  - Orange: Moderate effect (0.5 ≤ |Cohen's d| < 0.8)
  - Yellow: Small effect (0.2 ≤ |Cohen's d| < 0.5)
  - Grey: Negligible effect (|Cohen's d| < 0.2)
- Point shapes and outlines as in original plot
- Includes clinical significance legend

# Returns
- `Figure`: Enhanced counterfactual effect plot with clinical significance
"""
function plot_counterfactual_effects_with_significance(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])

    # Define colours
    male_color = :steelblue
    female_color = :crimson
    light_blue_grey = RGB(0.7529f0, 0.7804f0, 0.8588f0)

    # Clinical significance colors
    significance_colors = Dict(
        "Large" => :red,
        "Moderate" => :orange,
        "Small" => :gold,
        "Negligible" => :lightgrey
    )

    # Get unique mice for efficiency
    unique_mice = unique(df.IDidx)

    for mouse in unique_mice
        mouse_mask = df.IDidx .== mouse
        mouse_data = view(df, mouse_mask, :)

        # Determine sex colour for outlines
        sex_outline_color = mouse_data.S[1] == 1 ? male_color : female_color

        for i in 1:length(mouse_data.nP)
            x_val = mouse_data.nP[i]
            factual_val = mouse_data.E_factual_mean[i]
            counterfactual_val = mouse_data.E_counterfactual_mean[i]

            # Get clinical significance color
            significance = mouse_data.E_clinical_significance[i]
            line_color = significance_colors[significance]

            # Draw lines with clinical significance coloring
            lines!(ax, [x_val, x_val], [factual_val, counterfactual_val],
                color=line_color, linewidth=4, alpha=0.7)
        end

        # Plot factual and counterfactual points
        scatter!(ax, mouse_data.nP, mouse_data.E_factual_mean,
            color=light_blue_grey, strokecolor=sex_outline_color, strokewidth=2,
            markersize=16, alpha=0.8)
        scatter!(ax, mouse_data.nP, mouse_data.E_counterfactual_mean,
            color=:orange, strokecolor=sex_outline_color, strokewidth=2,
            marker=:utriangle, markersize=16, alpha=0.8)
    end

    # Configure axis labels
    ax.xlabel = "Observed Parasite Count"
    ax.ylabel = "Vaccine response"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    # Create comprehensive legend
    # Sex-specific point elements
    pre_male = MarkerElement(color=light_blue_grey, marker=:circle, markersize=16,
        strokecolor=male_color, strokewidth=2)
    pre_female = MarkerElement(color=light_blue_grey, marker=:circle, markersize=16,
        strokecolor=female_color, strokewidth=2)
    post_male = MarkerElement(color=:orange, marker=:utriangle, markersize=16,
        strokecolor=male_color, strokewidth=2)
    post_female = MarkerElement(color=:orange, marker=:utriangle, markersize=16,
        strokecolor=female_color, strokewidth=2)

    # Clinical significance line elements
    large_line = LineElement(color=:red, linewidth=4)
    moderate_line = LineElement(color=:orange, linewidth=4)
    small_line = LineElement(color=:gold, linewidth=4)
    negligible_line = LineElement(color=:lightgrey, linewidth=4)

    # Create legend positioned in top right of main axis
    # Create legend for Mouse & Condition in top right
    axislegend(ax,
        [pre_male, pre_female, post_male, post_female],
        ["Factual (Male)", "Factual (Female)", "Counterfactual (Male)", "Counterfactual (Female)"],
        "Mouse & Condition",
        position=:rt, labelsize=12)

    # Create legend for Effect Size in bottom right
    axislegend(ax,
        [large_line, moderate_line, small_line, negligible_line],
        ["Large effect (d ≥ 0.8)", "Moderate effect (0.5 ≤ d < 0.8)", "Small effect (0.2 ≤ d < 0.5)", "Negligible effect (d < 0.2)"],
        "Effect Size (Cohen's d)",
        position=:rb, labelsize=12)

    if saveplot
        safe_plot_save("counterfactual_effects_clinical_significance.pdf", fig)
    end
    fig
end

"""
    plot_S_R_interaction_cohens_d(df; saveplot=false)

Interaction plot for Sex × Reproductive status using Cohen's d effect sizes.
Provides standardised, interpretable effect size measures.

# Arguments
- `df::DataFrame`: DataFrame containing the data with Cohen's d column
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Returns
- `Figure`: Interaction plot showing Cohen's d effect sizes
"""
function plot_S_R_interaction_cohens_d(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])

    # Get unique values
    S_values = sort(unique(df.S))
    R_values = sort(unique(df.R))

    # Pre-allocate arrays
    n_S, n_R = length(S_values), length(R_values)
    means = Matrix{Float64}(undef, n_S, n_R)
    sems = Matrix{Float64}(undef, n_S, n_R)

    # Calculate means and standard errors for Cohen's d
    for (i, s) in enumerate(S_values), (j, r) in enumerate(R_values)
        mask = (df.S .== s) .& (df.R .== r)
        values = view(df.E_cohens_d, mask)
        n_vals = length(values)

        if n_vals > 0
            means[i, j] = mean(values)
            sems[i, j] = std(values) / sqrt(n_vals)
        else
            means[i, j] = NaN
            sems[i, j] = NaN
        end
    end

    # Define colours
    male_color = :steelblue
    female_color = :crimson

    # Plot lines and error bars
    for (i, s) in enumerate(S_values)
        sex_color = s == 1 ? male_color : female_color
        sex_label = s == 1 ? "Male" : "Female"

        valid_indices = .!isnan.(means[i, :])
        if any(valid_indices)
            valid_R = R_values[valid_indices]
            valid_means = means[i, valid_indices]
            valid_sems = sems[i, valid_indices]

            lines!(ax, valid_R, valid_means, label=sex_label, linewidth=2, color=sex_color)
            errorbars!(ax, valid_R, valid_means, valid_sems, color=sex_color, linewidth=1)
            scatter!(ax, valid_R, valid_means, marker=:circle, markersize=14, color=sex_color)
        end
    end

    # Add clinical significance reference lines
    hlines!(ax, [0.2, 0.5, 0.8], color=:black, linestyle=:dash, alpha=0.3)
    hlines!(ax, [-0.2, -0.5, -0.8], color=:black, linestyle=:dash, alpha=0.3)
    hlines!(ax, [0], color=:black, linestyle=:solid, alpha=0.5)

    # Add reference line labels
    text!(ax, maximum(R_values) + 0.1, 0.2, text="Small effect", fontsize=10, color=:grey)
    text!(ax, maximum(R_values) + 0.1, 0.5, text="Moderate effect", fontsize=10, color=:grey)
    text!(ax, maximum(R_values) + 0.1, 0.8, text="Large effect", fontsize=10, color=:grey)

    # Configure axes
    ax.xticks = (R_values, ["Reproductive", "Non-reproductive"])
    ax.xlabel = "Reproductive status"
    ax.ylabel = "Effect size (Cohen's d)"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    axislegend(ax, position=:rt)

    if saveplot
        safe_plot_save("S_R_interaction_cohens_d.pdf", fig)
    end
    fig
end

"""
    plot_clinical_significance_summary(df; saveplot=false)

Create a summary plot showing the distribution of clinical significance categories.

# Arguments
- `df::DataFrame`: DataFrame with clinical significance data
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Returns
- `Figure`: Bar plot of clinical significance distribution
"""
function plot_clinical_significance_summary(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])

    # Calculate counts and percentages
    categories = ["Negligible", "Small", "Moderate", "Large"]
    counts = [sum(df.E_clinical_significance .== cat) for cat in categories]
    percentages = round.(100 .* counts ./ nrow(df), digits=1)

    # Define colors matching clinical significance
    colors = [:lightgrey, :gold, :orange, :red]

    # Create bar plot
    barplot!(ax, 1:4, counts, color=colors)

    # Add percentage labels on bars
    for i in 1:4
        text!(ax, i, counts[i] + 1, text="$(percentages[i])%",
            align=(:center, :bottom), fontsize=12, color=:black)
    end

    # Configure axes
    ax.xticks = (1:4, categories)
    ax.xlabel = "Clinical Significance Category"
    ax.ylabel = "Number of Mice"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold
    ax.title = "Distribution of Clinical Significance\n(Cohen's d thresholds: 0.2, 0.5, 0.8)"

    if saveplot
        safe_plot_save("clinical_significance_summary.pdf", fig)
    end
    fig
end

"""
    plot_S_R_interaction_factual_counterfactual(df; saveplot=false)

Interaction plot showing factual vs counterfactual vaccine responses
by Sex and Reproductive status.

# Arguments
- `df::DataFrame`: DataFrame containing the data
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Returns
- `Figure`: Interaction plot comparing factual vs counterfactual responses
"""
function plot_S_R_interaction_factual_counterfactual(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])

    S_values = sort(unique(df.S))
    R_values = sort(unique(df.R))

    # Pre-allocate arrays with specific types
    n_S, n_R = length(S_values), length(R_values)
    means_factual = Matrix{Float64}(undef, n_S, n_R)
    means_counterfactual = Matrix{Float64}(undef, n_S, n_R)
    sems_factual = Matrix{Float64}(undef, n_S, n_R)
    sems_counterfactual = Matrix{Float64}(undef, n_S, n_R)

    # Calculate means and standard errors
    for (i, s) in enumerate(S_values), (j, r) in enumerate(R_values)
        mask = (df.S .== s) .& (df.R .== r)

        # Factual
        values_factual = view(df.E_factual_mean, mask)
        n_vals = length(values_factual)
        if n_vals > 0
            means_factual[i, j] = mean(values_factual)
            sems_factual[i, j] = std(values_factual) / sqrt(n_vals)
        else
            means_factual[i, j] = NaN
            sems_factual[i, j] = NaN
        end

        # Counterfactual
        values_counterfactual = view(df.E_counterfactual_mean, mask)
        if n_vals > 0
            means_counterfactual[i, j] = mean(values_counterfactual)
            sems_counterfactual[i, j] = std(values_counterfactual) / sqrt(n_vals)
        else
            means_counterfactual[i, j] = NaN
            sems_counterfactual[i, j] = NaN
        end
    end

    # Define colours
    male_color = :steelblue
    female_color = :crimson

    # Plot for each Sex and condition
    for (i, s) in enumerate(S_values)
        sex_color = s == 1 ? male_color : female_color
        sex_label = s == 1 ? "Male" : "Female"

        # Filter out NaN values
        valid_indices = .!isnan.(means_factual[i, :])
        if any(valid_indices)
            valid_R = R_values[valid_indices]

            # Factual (solid lines)
            valid_means_f = means_factual[i, valid_indices]
            valid_sems_f = sems_factual[i, valid_indices]
            lines!(ax, valid_R, valid_means_f, label="$(sex_label) (Factual)",
                linewidth=2, color=sex_color, linestyle=:solid)
            errorbars!(ax, valid_R, valid_means_f, valid_sems_f,
                color=sex_color, linewidth=1)
            scatter!(ax, valid_R, valid_means_f, marker=:circle, markersize=14, color=sex_color)

            # Counterfactual (dashed lines)
            valid_means_c = means_counterfactual[i, valid_indices]
            valid_sems_c = sems_counterfactual[i, valid_indices]
            lines!(ax, valid_R, valid_means_c, label="$(sex_label) (Counterfactual)",
                linewidth=2, color=sex_color, linestyle=:dash)
            errorbars!(ax, valid_R, valid_means_c, valid_sems_c,
                color=sex_color, linewidth=1)
            scatter!(ax, valid_R, valid_means_c, marker=:utriangle, markersize=14, color=sex_color)
        end
    end

    # Configure axes
    ax.xticks = (R_values, ["Reproductive", "Non-reproductive"])
    ax.xlabel = "Reproductive status"
    ax.ylabel = "Standardised vaccine response (E)"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold

    axislegend(ax, position=:rt)

    if saveplot
        safe_plot_save("S_R_interaction_factual_counterfactual.pdf", fig)
    end
    fig
end

"""
    plot_sex_reproductive_interaction_detailed(df; saveplot=false)

Detailed interaction plot showing all four Sex × Reproductive status combinations
for Cohen's d effect sizes with enhanced visual distinction.

# Arguments
- `df::DataFrame`: DataFrame containing the data
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Returns
- `Figure`: Detailed interaction plot with four distinct groups
"""
function plot_sex_reproductive_interaction_detailed(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])

    # Define the four combinations
    combinations = [
        (S=1, R=1, label="Male Reproductive", color=:steelblue, marker=:circle, linestyle=:solid),
        (S=1, R=2, label="Male Non-reproductive", color=:steelblue, marker=:diamond, linestyle=:dash),
        (S=2, R=1, label="Female Reproductive", color=:crimson, marker=:circle, linestyle=:solid),
        (S=2, R=2, label="Female Non-reproductive", color=:crimson, marker=:diamond, linestyle=:dash)
    ]

    # X-axis positions for the groups (with some spacing for clarity)
    x_positions = [1, 2, 3, 4]
    x_labels = ["Male\nReproductive", "Male\nNon-reproductive", "Female\nReproductive", "Female\nNon-reproductive"]

    means = Float64[]
    sems = Float64[]
    colors = []
    markers = []

    for (i, combo) in enumerate(combinations)
        # Filter data for this specific combination
        mask = (df.S .== combo.S) .& (df.R .== combo.R)
        values = view(df.E_cohens_d, mask)
        n_vals = length(values)

        if n_vals > 0
            mean_val = mean(values)
            sem_val = std(values) / sqrt(n_vals)

            push!(means, mean_val)
            push!(sems, sem_val)
            push!(colors, combo.color)
            push!(markers, combo.marker)

            # Add individual points with jitter for better visibility
            jitter = 0.1 * (rand(n_vals) .- 0.5)
            scatter!(ax, fill(x_positions[i], n_vals) .+ jitter, values,
                color=(combo.color, 0.3), markersize=8, strokewidth=0)
        else
            push!(means, NaN)
            push!(sems, NaN)
            push!(colors, combo.color)
            push!(markers, combo.marker)
        end
    end

    # Plot means with error bars
    for i in 1:4
        if !isnan(means[i])
            scatter!(ax, [x_positions[i]], [means[i]],
                color=colors[i], marker=markers[i], markersize=20,
                strokewidth=2, strokecolor=:white)
            errorbars!(ax, [x_positions[i]], [means[i]], [sems[i]],
                color=colors[i], linewidth=3)
        end
    end

    # Connect points to show interaction pattern
    valid_indices = .!isnan.(means)
    if any(valid_indices)
        lines!(ax, x_positions[valid_indices], means[valid_indices],
            color=:black, linewidth=1, linestyle=:dot, alpha=0.5)
    end

    # Add clinical significance reference lines
    hlines!(ax, [0.2, 0.5, 0.8], color=:green, linestyle=:dash, alpha=0.4)
    hlines!(ax, [-0.2, -0.5, -0.8], color=:red, linestyle=:dash, alpha=0.4)
    hlines!(ax, [0], color=:black, linestyle=:solid, alpha=0.8)

    # Configure axes first to set proper limits
    ax.xticks = (x_positions, x_labels)
    ax.xlabel = "Sex × Reproductive Status"
    ax.ylabel = "Effect size (Cohen's d)"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold
    # ax.title = "Parasite Elimination Effect by Sex and Reproductive Status"

    # Set explicit axis limits to show data properly while allowing space for labels
    xlims!(ax, 0.5, 4.4)

    # Add reference line labels positioned to the right of the data with left alignment
    text!(ax, 4.1, 0.2, text="Small", fontsize=10, color=:green,
        space=:data, align=(:left, :bottom))
    text!(ax, 3.95, 0.5, text="Moderate", fontsize=10, color=:green,
        space=:data, align=(:left, :bottom))
    text!(ax, 4.1, 0.8, text="Large", fontsize=10, color=:green,
        space=:data, align=(:left, :bottom))
    text!(ax, 4.1, -0.2, text="Small", fontsize=10, color=:red,
        space=:data, align=(:left, :bottom))
    text!(ax, 3.95, -0.5, text="Moderate", fontsize=10, color=:red,
        space=:data, align=(:left, :bottom))
    text!(ax, 4.1, -0.8, text="Large", fontsize=10, color=:red,
        space=:data, align=(:left, :bottom))

    # Create custom legend
    legend_elements = []
    legend_labels = []

    for combo in combinations
        push!(legend_elements, MarkerElement(color=combo.color, marker=combo.marker,
            markersize=16, strokewidth=2, strokecolor=:white))
        push!(legend_labels, combo.label)
    end

    # Legend(fig[1, 2], legend_elements, legend_labels, "Groups",
    # framevisible=false, labelsize=12)

    if saveplot
        safe_plot_save("sex_reproductive_interaction_detailed.pdf", fig)
    end
    fig
end

"""
    plot_sex_reproductive_heatmap(df; saveplot=false)

Create a heatmap showing Cohen's d effect sizes for all Sex × Reproductive status combinations.

# Arguments
- `df::DataFrame`: DataFrame containing the data
- `saveplot::Bool=false`: If true, saves the plot as a PDF

# Returns
- `Figure`: Heatmap visualization of the interaction
"""
function plot_sex_reproductive_heatmap(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])

    # Get unique values
    S_values = sort(unique(df.S))
    R_values = sort(unique(df.R))

    # Create matrix for heatmap
    effect_matrix = Matrix{Float64}(undef, length(S_values), length(R_values))
    count_matrix = Matrix{Int}(undef, length(S_values), length(R_values))

    for (i, s) in enumerate(S_values), (j, r) in enumerate(R_values)
        mask = (df.S .== s) .& (df.R .== r)
        values = view(df.E_cohens_d, mask)

        if length(values) > 0
            effect_matrix[i, j] = mean(values)
            count_matrix[i, j] = length(values)
        else
            effect_matrix[i, j] = NaN
            count_matrix[i, j] = 0
        end
    end

    # Create heatmap
    hm = heatmap!(ax, effect_matrix, colormap=:RdBu,
        colorrange=(-1, 1))

    # Add text annotations with values and counts
    for i in 1:length(S_values), j in 1:length(R_values)
        if !isnan(effect_matrix[i, j])
            effect_text = round(effect_matrix[i, j], digits=2)
            count_text = count_matrix[i, j]
            text!(ax, j, i, text="d = $effect_text\nn = $count_text",
                align=(:center, :center), fontsize=12, color=:white)
        end
    end

    # Configure axes
    ax.xticks = (1:length(R_values), ["Reproductive", "Non-reproductive"])
    ax.yticks = (1:length(S_values), ["Male", "Female"])
    ax.xlabel = "Reproductive Status"
    ax.ylabel = "Sex"
    ax.xlabelsize = 16
    ax.ylabelsize = 16
    ax.xlabelfont = :bold
    ax.ylabelfont = :bold
    ax.title = "Effect Size Heatmap: Parasite Elimination by Sex × Reproductive Status"

    # Add colorbar
    Colorbar(fig[1, 2], hm, label="Cohen's d", labelsize=14)

    if saveplot
        safe_plot_save("sex_reproductive_heatmap.pdf", fig)
    end
    fig
end

"""
    plot_cohens_d_no_interaction(df; saveplot=false)

Bar plot of Cohen's d for main effects only (no interactions).

# Arguments
- `df::DataFrame`: must contain columns `D, R, S, V, M, Ḟ, E_cohens_d`
- `saveplot::Bool=false`: save as PDF if true

# Returns
- `Figure`
"""
function plot_cohens_d_no_interaction(df; saveplot::Bool=false)
    fig = Figure(size=(647, 400))
    ax = Axis(fig[1, 1])
    
    # Define the factor terms to analyze
    factors = [:D, :R, :S, :V, :M, :Ḟ]
    
    # Initialize arrays to store results
    values = Float64[]
    sems = Float64[]
    labels = String[]
    
    # Compute Cohen's d estimates for each factor
    for factor in factors
        # Create mask for factor level 1
        mask = df[:, factor] .== 1
        n_masked = sum(mask)
        
        if n_masked > 0
            # Extract Cohen's d values for this subset
            E_cohens_d_masked = df[mask, :E_cohens_d]
            
            # Compute mean and SEM
            mean_d = mean(E_cohens_d_masked)
            sem_d = std(E_cohens_d_masked) / sqrt(n_masked)
            
            # Store results
            push!(values, mean_d)
            push!(sems, sem_d)
            push!(labels, string(factor))
        else
            # Store NaN values for missing data
            push!(values, NaN)
            push!(sems, NaN)
            push!(labels, string(factor))
        end
    end
    
    # Filter out NaN values for plotting
    valid_indices = .!isnan.(values)
    plot_values = values[valid_indices]
    plot_sems = sems[valid_indices]
    plot_labels = labels[valid_indices]
    
    if length(plot_values) > 0
        # Create bar plot
        n_terms = length(plot_values)
        barplot!(ax, 1:n_terms, plot_values, color=:steelblue)
        
        # Add error bars
        for i in 1:n_terms
            errorbars!(ax, [i], [plot_values[i]], [plot_sems[i]], color=:black, linewidth=2)
        end
        
        # Add clinical significance reference lines using defined thresholds
        for threshold in thresholds
            hlines!(ax, [threshold], color=colors[threshold], linestyle=:dash, alpha=0.5)
            hlines!(ax, [-threshold], color=colors[threshold], linestyle=:dash, alpha=0.5)
        end
        hlines!(ax, [0], color=:black, linestyle=:solid, alpha=0.8)
        
        # Configure axes
        ax.xticks = (1:n_terms, plot_labels)
        ax.xlabel = "Main Effects"
        ax.ylabel = "Effect size (Cohen's d)"
        ax.xlabelsize = 16
        ax.ylabelsize = 16
        ax.xlabelfont = :bold
        ax.ylabelfont = :bold
        ax.title = "Main Effects: Cohen's d Estimates"
        
        # Add reference line labels
        y_max = maximum([maximum(plot_values .+ plot_sems), maximum(thresholds)])
        text!(ax, length(plot_values) + 0.3, 0.2, text="Small", fontsize=10, color=:grey)
        text!(ax, length(plot_values) + 0.2, 0.5, text="Moderate", fontsize=10, color=:grey)
        text!(ax, length(plot_values) + 0.3, 0.8, text="Large", fontsize=10, color=:grey)
    else
        # Handle case with no valid data
        text!(ax, 0.5, 0.5, text="No valid data to plot", align=(:center, :center))
    end
    
    if saveplot
        safe_plot_save("cohens_d_no_interaction.pdf", fig)
    end
    fig
end

# Export all plotting functions
export plot_E_factual_counterfactual,
    plot_counterfactual_effects,
    plot_counterfactual_effects_with_significance,
    plot_S_R_interaction_cohens_d,
    plot_clinical_significance_summary,
    plot_S_R_interaction_factual_counterfactual,
    plot_sex_reproductive_interaction_detailed,
    plot_sex_reproductive_heatmap,
    plot_cohens_d_no_interaction
