"""
Leave-one-out logistic regression for importance testing
"""

if NODE == 'CLIM':
    streamers = xr.open_dataset('/scratch3/severin/rwb_index/era5/era5_av_streamer_75.nc')['flag_1']
streamers1 = streamers.sel(time=(streamers.time.dt.season=='JJA'), lon=slice(-60, 60), lat=slice(20, 80)) == 1
streamers2 = streamers.sel(time=(streamers.time.dt.season=='JJA'), lon=slice(-60, 60), lat=slice(20, 80)) == 2
timemask_streamer = np.isin(DATERANGEPL_SUMMER, streamers1.time.values)
mu = streamers1.mean(dim='time')
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import r2_score

Y = streamers1.values.reshape(streamers1.shape[0], -1)
mask = np.sum(Y, axis=0) > 4
Y = Y[:, mask]
X = OPP_timeseries_T1[timemask_streamer, :6]
nX = X.shape[1]
Xidx = np.vstack([np.ones(nX, dtype=bool), ~np.eye(nX, dtype=bool)])
from tqdm import tqdm
scores = np.zeros((X.shape[1] + 1, Y.shape[1]))

for i, y in tqdm(enumerate(Y.T), total=np.sum(mask)):
    for j, xidx in enumerate(Xidx):
        log = LogisticRegression(class_weight='balanced')
        log.fit(X[:, xidx], y)
        scores[j, i] = log.score(X[:, xidx], y)
full = scores[0]
loo = scores[1:]
predictor_importance = np.zeros((len(mask), nX))
predictor_importance[mask, :] = (1 - (1 - full) / (1 - loo)).T
predictor_importance[predictor_importance < 0] = 0
predictor_importance = predictor_importance.reshape((*streamers1.shape[1:], 6))
predictor_importance = xr.DataArray(
    predictor_importance, 
    coords={'lat': streamers1.lat.values, 'lon': streamers1.lon.values, 'OPP': np.arange(6)}
)
clu = Clusterplot(2, 3)
clu.add_contourf([predictor_importance[:, :, i] for i in range(6)])