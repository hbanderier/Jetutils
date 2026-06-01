from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from itertools import combinations, combinations_with_replacement

quantif = 3
ds_quantiles = xr.Dataset(coords={'time': props_as_ds.time})
shape = [len(co) for co in ds_quantiles.coords.values()]
for varname in props_as_ds.data_vars:
    if varname == 'is_polar':
        continue
    for j, letter in enumerate(['s', 'p']):
        this_da = props_as_ds[varname][:, j]
        low_thresh = this_da.min().values - 0.1
        for i, quantile in enumerate(np.linspace(1 / quantif, 1, quantif)):
            high_thresh = this_da.quantile(q=quantile).values
            this_predictor = (this_da.values > low_thresh) & (this_da.values <= high_thresh)
            ds_quantiles[f'{varname}_{i + 1}{letter}'] = (list(ds_quantiles.coords.keys()), this_predictor)
            low_thresh = high_thresh

hotspell_binary = get_hotspell_mask(props_as_ds.time).astype(int)[:, :, 0] > 3

predictors = product(props_as_ds.data_vars, ['s', 'p'])
n_predictors = 3
all_combinations = list(combinations(predictors, n_predictors))
quantile_combinations = list(combinations_with_replacement(np.arange(1, quantif + 1), n_predictors))
all_combinations = [[(*el1, el2) for el1, el2 in zip(comb, qcomb)] for comb, qcomb in product(all_combinations, quantile_combinations)]

all_scores = np.zeros((len(REGIONS), len(all_combinations)))
all_coefs = np.zeros((len(REGIONS), len(all_combinations), len(all_combinations[0])))

for i, region in enumerate(REGIONS):
    y = hotspell_binary[:, i].values
    for j, comb in tqdm(enumerate(all_combinations), total=len(all_combinations)):
        X = np.stack([ds_quantiles[f'{varname}_{quantile_idx}{jet}'].values for varname, jet, quantile_idx in comb], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        log = LogisticRegression().fit(X=X_train, y=y_train)
        all_coefs[i, j, :] = log.coef_[0]
        all_scores[i, j] = roc_auc_score(y_test, log.predict(X_test))
    break