# grbpop

Python code used for the modelling of the population of short GRBs in Salafia et al. 2023 (https://arxiv.org/abs/2306.15488). Please cite the paper if you use this code.
The MCMC chains obtained with this code and used to produce the plots in Salafia et al. 2023 are stored at Zenodo: https://zenodo.org/record/8160783

## Dependencies
The `grbpop` module is written in python 3 and depends on the following packages:
- numpy
- scipy
- astropy

In order to run the scripts that use `grbpop` to fit the SGRB population and to reproduce the figures from Salafia et al. 2023, you'll also need:
- emcee
- h5py
- pandas

## Usage

First clone this repository and add its root directory to your PYTHONPATH environment variable. Then you need to run once the `create_pflux_table.py` script in the `grbpop/` subdirectory, in order to create the tables over which the flux as a function of luminosity is interpolated within the code. You then can write `import grbpop` in a python shell or in a script in order to access its module and submodules. The code is composed of the following main scripts:
-   `grbpop.structjet`: this is where the jet structure functions are defined
-   `grbpop.Ppop`: population probabilities and terms in the hierarchical Bayesian inference are computed here
-   `grbpop.globals`: this is where some global variables are defined
-   `grbpop.pdet`: this is where an interpolator is set up to compute the GBM detection efficiency as modelled in Salafia et al. 2023
-   `grbpop.diagnose`: here you find many useful functions to produce diagnostic plots
-   `grbpop.pflux`: here interpolators are set up to use the flux grids
-   `grbpop.spectrum`: here are some convenience functions related to GRB spectra

Most functions in there have docstrings that explain how to use them. 

## Questions 

Please contact om.salafia@inaf.it for any question or if you want to contribute to further development of this code.
