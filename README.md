# Wild Vaccines

## Setup

To set the julia environment up follow the steps below (taken from <https://julialang.github.io/Pkg.jl/v1/environments/>):

1. Clone the current repository (`git clone https://github.com/SimonAB/wild_vaccines.git`)
2. cd to cloned repo (`cd wild_vaccines`)
3. Launch Julia v1.4
4. Enter Pkg mode `]`
5. Activate local environment `activate .`
6. Instantiate Packages `instantiate`
7. [Optional] Precompile local env: `precompile`
8. Exit Pkg mode `<Backspace>`.

## ToDos

- [ ] Clean up data
  - [ ] Remove missing values that cannot be rescued (e.g no pittag number)
  - [ ] Obtain further metadata about vaccinated mice
  - [ ] Merge relevant data into a single csv file
- [ ] Analyse data
  - [ ] Start with simple mixed model conditioning on grid
  - [ ] Try incorporating repeated measures if any, conditioning on pittag