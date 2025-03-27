# Jetutils

An everything package with my jet stream utilities and other related things.
It can do:

**Jet extraction** from lon-lat fields

**Jet categorization** Into subtropical and eddy-driven jets

**Jet tracking** in time

**Jet metrics**: about 20 of them including jet width, afaik a novelty.

**Jet-natural-coordinates plots** what happens around the jet? Using natural coordinates to define normal segments to the jet and interpolate stuff onto those segments.

**Clustering** With various algorithms and sensical pre- and post-processing.

**Clusterplots**: Multi panel plots for easy compositing, with integrated statistical significance testing

**Extreme events regionalization**: Find spatial regions that experience extreme events (heat wave, drought, wind storm) in sync.

**Multi predictor multi lag multi timescale multi region prediction**: From several float series to several binary series. Heavily inspired by [Van Straaten et al. 2022](doi.org/10.1175/MWR-D-21-0201.1)

**Predictor importance studies**: Which jet metrics are important for predicting heat waves in France at a 10 day lag?

**And more**: see [the documentation](https://jet-stream.readthedocs.io/en/latest/index.html). It is about half complete right now. 

## Configuration

A few functions depend on global variables than can be configured. By default, the configuration file simply lets the code guess them, but it's probably not great guesses.
```ini
[PATHS]
DATADIR = guess
FIGURES = guess
RESULTS = guess

[COMPUTE]
N_WORKERS = guess
MEMORY_LIMIT = guess
```
The behaviour can be overriden by creating a config file `~/$HOME/.jetutils.ini` with the same sections and fields. For example, on my laptop, I have:

```ini
[PATHS]
DATADIR = $HOME/Documents/code_local/data
FIGURES = $HOME/Documents/code_local/local_figs
RESULTS = $HOME/Documents/code_local/data/results

[COMPUTE]
N_WORKERS = 8
MEMORY_LIMIT = 8GiB
```

## Citation
If any of this is useful to you please consider citing 

Banderier, H., Tuel, A., Woollings, T., and Martius, O.: Trends and seasonal signals in Atlantic feature-based jet stream characteristics and in weather types, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-3029, 2024. 