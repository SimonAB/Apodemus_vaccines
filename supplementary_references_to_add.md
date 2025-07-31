# Supplementary References to Add to Main Text

Based on the analysis of the manuscript, here are the specific locations where supplementary references should be added:


## 2. Line 279 - SCM Construction Reference  
**Location**: Results section where the SCM construction is discussed
**Current text**: "...we constructed a structural causal model (SCM) of the processes generating variation in vaccine responsiveness $\mathbb{C}_{VE}$ in this system."
**Updated text**: "...we constructed a structural causal model (SCM) of the processes generating variation in vaccine responsiveness $\mathbb{C}_{VE}$ in this system (\text{supdata:SCMconstruction})."

## 3. Line 293 - SCM Construction Reference (alternative location)
**Location**: Where methods section is referenced for construction details
**Current text**: "...see methods section ``\nameref{sec:construction}'')"
**Updated text**: "...see methods section ``\nameref{sec:construction}'', \nameref{supdata:SCMconstruction})"

## 4. Line 427 - Bayesian Implementation Reference
**Location**: Data processing and analyses section
**Current text**: "All other variables were treated as binary. All data processing was performed in Julia..."
**Updated text**: "All other variables were treated as binary (\nameref{supdata:BayesianImplementation}). All data processing was performed in Julia..."

## 5. Line 306 - Model Validation Reference  
**Location**: Where Bayesian model convergence is discussed
**Current text**: "All Bayesian models showed excellent convergence with well-behaved MCMC chains and $\hat{R} < 1.01$ for all parameters"
**Updated text**: "All Bayesian models showed excellent convergence with well-behaved MCMC chains and $\hat{R} < 1.01$ for all parameters (\nameref{supdata:ModelValidation})"

## 6. Line 465 - Effect Size Calculation Reference
**Location**: In the methods section where effect sizes are categorized
**Current text**: "Effect sizes were calculated using Cohen's $d$ with pooled standard deviations, categorised as negligible..."
**Updated text**: "Effect sizes were calculated using Cohen's $d$ with pooled standard deviations, categorised as negligible... (\nameref{supdata:EffectSizeCalculation})"

## Note for Manual Implementation
Due to the complexity of the LaTeX file structure and the way text spans multiple lines, these edits would be best implemented manually by:

1. Opening the manuscript/apodemus-vacc-main.tex file
2. Locating each of the specified lines 
3. Adding the appropriate (\text{supdata:...}) references as indicated above
4. Ensuring proper LaTeX formatting is maintained

The references connect the main text to the detailed methodological descriptions in the supplementary materials, improving the manuscript's organization and accessibility.
