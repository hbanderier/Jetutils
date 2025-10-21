# Welcome to Jetutil's documentation!

Jetutils is an everything package with my jet stream utilities and other related things.

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

It's heavily reliant on the [`polars`](https://docs.pola.rs/api/python/stable/reference/index.html) package, and can serve as a proof of concept that you can do a lot of `xarray` stuff with `polars` if you twist it hard enough. Usually for gains, sometimes it's a pure waste of time.

## Citation
If any of this is useful to you, please cite:

Banderier, H., Tuel, A., Woollings, T., and Martius, O.: Seasonal to decadal variability and persistence properties of the Euro-Atlantic jet streams characterized by complementary approaches, Weather Clim. Dynam., 6, 715–739, <https://doi.org/10.5194/wcd-6-715-2025>, 2025. 
