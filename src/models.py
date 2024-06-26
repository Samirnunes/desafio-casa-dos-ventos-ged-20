from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

psatjira_models = [
    KNeighborsRegressor(n_neighbors=10),
    RandomForestRegressor(n_estimators=20, max_depth=4, criterion="squared_error", random_state=0)
]

psatjira_models_names = [
    "10nn",
    "random_forest"
]

psatitp_models = [
    XGBRegressor(learning_rate=0.1, max_depth=3, subsample=0.8),
    KNeighborsRegressor(n_neighbors=11, p=1)
]

psatitp_models_names = [
    "xgb_15d",
    "knn_15d"
]

psatimbr_models = [
    KNeighborsRegressor(n_neighbors=24),
    RandomForestRegressor(n_estimators=200, max_depth=3, criterion="squared_error", random_state=0)
]

psatimbr_models_names = [
    "24nn",
    "random_forest"
]