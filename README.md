# Markov Chain Monte Carlo using Hamiltonian Dynamics

## Authors: Erik Wiegers, Michael Wenger

### Topic:
The Metropolis Hastings algorithm (MH) and the Hamiltonian Monte Carlo algorithm (HMC) are implemented and tested in three different problems, namely a two-dimensional and 15-dimensional Gauss distribution and a two-dimensional Rosenbrock distribution.
### Environment
This repository contains an `bayesian.yml` file. It can be used by conda to create an identical environment as used for the analysis. Use the following command to create the environment:

```bash
conda env create -f bayesian.yml
```

The environment will be named `Bayesian`.

Note that in order to reproduce our result the data must not be generated again since it is stored in `HMC_data` and `MH_data`. The generated plots used in the report are stored in `Plots`.

### Scripts
- **Generating the data**: 
  - `HamiltonianMonteCarlo.ipynb`
    - Contains the implementation of the Hamiltonian-Monte-Carlo algorithm and also a script which generates the sample for all three target distributions described above. Three sets of samples are generated with sample size 10'000, 50'000 and 100'000.
  - `MetropolisHasting.ipynb`
    - Exactly the same as `HamiltonianMonteCarlo.ipynb` for the Metropolis-Hastings algorithm.

- **Analysis of the data**:
  The analysis can be found in:
  - `analysis_2D_gaussian.ipynb`
  - `analysis_15D_gaussian.ipynb`
  - `analysis_RB_gaussian.ipynb`
    - All these scripts load in the generated data and print out the acceptance rate for each set of sample. They also calculate the autocorrelation time and some of them are also used for making some plots in order to visualize the data.
  - `Rosenbrocktails.ipynb`
    - Contains the calculation for the ratio of numbers of samples present in the tails of the Rosenbrock distribution compared to the total number of samples.
