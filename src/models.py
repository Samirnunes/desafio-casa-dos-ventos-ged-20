from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

psatjira_models = [
    KNeighborsRegressor(n_neighbors=10),
    RandomForestRegressor(n_estimators=20, max_depth=4, criterion="squared_error", random_state=0)
]

psatjira_models_names = [
    "10nn",
    "random_forest"
]
