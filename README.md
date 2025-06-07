# Environmental Drivers of Low Vaccine Responsiveness in a Lab-to-Wild Rodent Model

A comprehensive Julia-based analysis pipeline for investigating vaccine responsiveness in wild vs laboratory populations of wood mice (*Apodemus sylvaticus*).

## Overview

This repository contains the complete data analysis pipeline for the research project examining how environmental factors affect vaccine responsiveness, comparing laboratory-reared and wild wood mice populations. The study investigates the role of habitat, diet, and parasite burden on vaccine-specific antibody responses.

### Key Research Questions

- How do environmental factors affect vaccine responsiveness between wild and laboratory populations?
- What role do diet and nutritional stress play in vaccine efficacy?
- How do helminth infections influence vaccine-specific antibody production?
- Can we develop causally-explicit models to quantify these relationships?

## Repository Structure

### Core Analysis Scripts (`src/`)

The analysis pipeline consists of several sequential Julia scripts:

- **`0_Data_Checks.jl`** - Exploratory data analysis and data validation
- **`1_Multilevel_Models.jl`** - Multilevel Bayesian models for vaccine response analysis
- **`2_SCM_validation.jl`** - Directed Acyclic Graph (DAG) validation for causal inference
- **`3_SCM_identification.jl`** - Structural Causal Model (SCM) identification of causal effects
- **`3_SCM_identification_optimized.jl`** - Performance-optimized version with automatic differentiation compatibility
- **`4_SCM_intervention.jl`** - SCM intervention analysis for causal effect estimation

### Utility Modules

- **`DataWrangler.jl`** - Data processing, cleaning, and encoding functions
- **`TuringUtils.jl`** - Utility functions for Bayesian models using Turing.jl
- **`TuringPlots.jl`** - Plotting functions for model outputs and diagnostics
- **`PlottingUtils.jl`** - Additional plotting utilities for data visualisation

### Data (`data/`)

- **`clean_data.csv`** - Processed dataset ready for analysis
- **`joint_dataset_4analysis_checked.csv`** - Main analysis dataset with validated entries
- **`dag_df.csv`** - Data formatted for causal analysis
- **`readme_column_key.csv`** - Data dictionary explaining variable definitions

### Manuscript (`manuscript/`)

- **`apodemus-vacc-main.tex`** - Main manuscript LaTeX source
- **`apodemus-vacc-main.pdf`** - Compiled manuscript
- **`apodemus-vacc-coverletter.tex`** - Cover letter for journal submission
- **`apodemus-vacc-refs.bib`** - Bibliography database
- **`Figures/`** - Generated figures and plots for the manuscript
- **`plos_latex_template_v3/`** - PLOS journal template files

### Documentation

- **`OPTIMIZATION_SUMMARY.md`** - Comprehensive documentation of Julia 1.11+ optimizations
- **`README_OPTIMIZATION.md`** - Optimization guide and performance improvements
- **`example_usage_optimized.jl`** - Demonstration script showing how to use optimized models

## Setup and Installation

### Prerequisites

- Julia 1.11+ (recommended for optimal performance)
- R (for RCall integration)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/SimonAB/Apodemus_vaccines.git
   cd Apodemus_vaccines
   ```

2. Open the project in [VSCode](https://code.visualstudio.com) or your preferred IDE

3. Launch Julia and activate the project environment:

   ```julia
   # Start Julia REPL (⌘⇧P then "Start Julia REPL" in VSCode)
   ]  # Enter Pkg mode
   activate .
   instantiate
   precompile  # Optional but recommended
   ```

4. Exit Pkg mode and verify installation:

   ```julia
   # Press backspace to exit Pkg mode
   include("example_usage_optimized.jl")
   ```

## Usage

### Quick Start

Run the example usage script to see the analysis pipeline in action:

```julia
include("example_usage_optimized.jl")
```

### Individual Analysis Steps

Execute the analysis pipeline sequentially:

```julia
# Data exploration and validation
include("src/0_Data_Checks.jl")

# Multilevel models
include("src/1_Multilevel_Models.jl")

# Causal inference
include("src/2_SCM_validation.jl")
include("src/3_SCM_identification_optimized.jl")
include("src/4_SCM_intervention.jl")
```

### Custom Analysis

For custom analyses, load the utility modules:

```julia
include("src/DataWrangler.jl")
include("src/TuringUtils.jl")
include("src/PlottingUtils.jl")

# Your custom analysis code here
```

## Key Dependencies

The project utilises several key Julia packages:

- **Turing.jl** - Bayesian modelling and inference
- **DataFrames.jl** - Data manipulation and analysis
- **CairoMakie.jl** - High-quality plotting and visualisation
- **MixedModels.jl** - Mixed-effects modelling
- **AlgebraOfGraphics.jl** - Grammar of graphics implementation
- **GLM.jl** - Generalised linear models
- **MCMCChains.jl** - MCMC chain analysis and diagnostics

## Performance Optimizations

This project has been extensively optimised for Julia 1.11+, featuring:

- **Automatic Differentiation Compatibility** - Full ForwardDiff/ReverseDiff support
- **Type Stability** - Optimised type inference throughout
- **Memory Efficiency** - Reduced allocations and improved memory usage
- **Error Handling** - Robust error handling with graceful degradation
- **Parallel Computing** - Multi-threading support where applicable

See `OPTIMIZATION_SUMMARY.md` for detailed performance improvements and benchmarks.

## Analysis Approach

The project employs a comprehensive causal inference framework:

1. **Multilevel Modelling** - Account for hierarchical data structure (individuals nested within populations)
2. **Directed Acyclic Graphs (DAGs)** - Explicit causal assumptions and confounding relationships
3. **Structural Causal Models** - Quantify causal effects of environmental factors
4. **Bayesian Inference** - Uncertainty quantification and robust statistical inference

## Key Findings

- Vaccine-specific IgG1 antibodies were **47% lower** in wild populations compared to laboratory-reared mice
- **Dietary supplementation** unexpectedly reduced vaccine responsiveness
- **Helminth infection burden** negatively affects vaccine-specific antibody production
- **Environmental factors** play a dramatic role in shaping immune responses to vaccination

## Citation

If you use this code or data, please cite:

> Babayan, S.A., Venkatesan, S., Hall, J.L., Smith, E., Sweeny, A., & Pedersen, A.B. "Environmental drivers of low vaccine responsiveness in a lab-to-wild rodent model." *[Journal]* (in preparation).

## Contributing

This research code is provided for reproducibility and transparency. For questions or collaboration inquiries, please contact:

- Simon A. Babayan: <simon.babayan@glasgow.ac.uk>
- Amy B. Pedersen: <amy.pedersen@ed.ac.uk>

## License

This project is licensed under the terms specified in the `LICENSE` file.

## Acknowledgments

- University of Glasgow, School of Biodiversity, One Health & Veterinary Medicine
- University of Edinburgh, Institute of Ecology and Evolution
- Julia Computing Community for excellent package ecosystem
