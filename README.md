# Environmental drivers of low vaccine responsiveness in a lab-to-wild rodent model

**Short title:** Lab-to-wild vaccine responsiveness

## Authors

- **Simon A. Babayan**<sup>†,*</sup> — School of Biodiversity, One Health & Veterinary Medicine, University of Glasgow, Glasgow, UK  
- **Saudamini Venkatesan**<sup>†</sup> — Institute of Ecology and Evolution, School of Biological Sciences, University of Edinburgh, Edinburgh, UK  
- **Jessica L. Hall**<sup>1,2</sup> — University of Glasgow; University of Edinburgh  
- **Ewan W. Smith** — School of Biodiversity, One Health & Veterinary Medicine, University of Glasgow, Glasgow, UK  
- **Amy Sweeny** — Institute of Ecology and Evolution, School of Biological Sciences, University of Edinburgh, Edinburgh, UK  
- **Amy B. Pedersen** — Institute of Ecology and Evolution, School of Biological Sciences, University of Edinburgh, Edinburgh, UK  

<sup>†</sup> These authors contributed equally to this work.  
<sup>*</sup> Corresponding author: [simon.babayan@glasgow.ac.uk](mailto:simon.babayan@glasgow.ac.uk)

## Abstract

Vaccination is the most effective way to prevent infectious diseases and safeguard public health. Yet, most new vaccines fail in late clinical trials, and even established ones often underperform in populations apart from those in which they were initially tested. This can lead to reduced vaccine responsiveness, breakthrough infections, and prevent or delay herd immunity. While the causes of vaccine hyporesponsiveness remain difficult to identify, quantify, and therefore address, numerous reports indicate a predominant role of environmental factors. This has notably been demonstrated by a reduction in the immunogenicity and efficacy of various vaccines when transitioning from urban to rural human populations. Here, we tested whether and, if so, how the environment can cause vaccine hyporesponsiveness. We hypothesised that if the leading causes of vaccine hyporesponsiveness were environmental, then environmentally driven hyporesponsiveness would be exacerbated when individuals are under nutritional stress; specifically predicting that high-quality diet supplementation would increase vaccine responsiveness. Finally, we predicted that parasitic helminth infections, which are more common in rural populations, would degrade vaccine responsiveness, e.g. due to their ability to modulate host immunity, and that anthelmintic treatment could rescue vaccine responsiveness in infected individuals. To test these hypotheses, we coupled lab and field experiments with structural causal modelling, and quantified diphtheria toxoid-specific IgG1 optical density (OD) in paired conspecific cohorts of laboratory-reared and wild wood mice (*Apodemus sylvaticus*) given a single or two doses of diphtheria toxoid vaccine formulated with alum, with and without diet supplementation. We found that anti-toxoid IgG1 OD was ~47% lower in the wild wood mice compared to the laboratory-reared population. We also demonstrated that, across both habitats (wild and lab), substantial variation in vaccine responsiveness was caused by diet. However, contrary to our predictions, this high-quality dietary supplementation resulted in lower vaccine responsiveness. Further, once the effects of habitat, diet, and sex were adjusted for, increasing helminth infection burdens negatively affected anti-toxoid IgG1 OD. Counterfactual predictions from our structural causal model suggested that targeting anthelmintic treatment at heavily infected individuals could have improved their anti-toxoid IgG1 OD responses by approximately 2 to 4-fold. Our results indicated that the wild environment and access to a high-quality diet played a dramatic role in shaping the immune system's response to immunisation. Further, we showed that laboratory settings, even when using a genetically diverse, non-traditional model, systematically yielded higher IgG1 OD than was observed in free-living conspecifics on the same protocol. We provide a causally explicit modelling approach to quantify how habitat, diet, and parasites jointly shape anti-diphtheria toxoid IgG1 levels in a focal population, and to prioritise adjunct interventions such as deworming where model assumptions hold.

---

This repository provides **Julia code and data** to reproduce the statistical analyses reported in the manuscript. It does not include the manuscript text or figure files.

| Repository | Role |
|------------|------|
| [Apodemus_vaccines](https://github.com/SimonAB/Apodemus_vaccines) | **This repository** — public methods release |
| [Apodemus_vaccines_manuscript](https://github.com/SimonAB/Apodemus_vaccines_manuscript) | Private — manuscript and figures |
| [Apodemus_vaccines_analysis](https://github.com/SimonAB/Apodemus_vaccines_analysis) | Private — full analysis development |

## Requirements

- [Julia](https://julialang.org/) 1.12+ (`Project.toml`)
- Optional: R with CRAN package `dagitty` for extra DAG summaries in `2_SCM_validation.jl`. If R is unavailable:

  ```bash
  export APODEMUS_SKIP_DAGITTY=1
  ```

## Install

```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
```

## Run

```julia
using Pkg
Pkg.activate(@__DIR__)

include("src/DataWrangler.jl")
include("src/1_Multilevel_Models.jl")
include("src/2_SCM_validation.jl")
include("src/3_SCM_identification.jl")
include("src/4_SCM_intervention.jl")
```

Supplementary tables:

```bash
julia --project=. scripts/assay_floor_nonresponder_summary.jl
julia --project=. scripts/render_arm_level_igg1_summary_table.jl
julia --project=. scripts/render_mcmc_diagnostics_table.jl
```

## Licence

See [`LICENSE`](LICENSE).
