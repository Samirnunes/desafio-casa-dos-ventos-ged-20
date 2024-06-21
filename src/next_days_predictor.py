import pandas as pd
from data_import import *
from create_features import create_new_features_by_df
from sklearn.base import clone

class NextDaysPredictor:
    def __init__(self, model, plant="PSATJIRA"):
        self.__model = clone(model)
        self.__plant = plant
        
    def predict_next_x_days(self, x=15):
        df_train = get_dataset_only_time(self.__plant)
        X_train = df_train.drop(["mean_precipitation"], axis=1)
        y_train = df_train["mean_precipitation"]
        model = self.__model
        model.fit(X_train, y_train)
        df = self.__append_next_x_days(x)
        for i in range(x):
            df_next_x = df.tail(x)
            df_X = df_next_x.drop(["mean_precipitation"], axis=1).copy()
            X = pd.DataFrame(df_X.iloc[i]).T
            y = model.predict(X)
            for j, k in zip(range(i, df_next_x.shape[0]), range(0, df_next_x.shape[1])):
                df_next_x.iloc[j, k] = y
            df.iloc[range(len(df)-x, len(df)), :] = df_next_x
            df['mean_last_30d'] = df['mean_precipitation'].rolling(window=30, closed="left").sum()
            df['mean_last_60d'] = df['mean_precipitation'].rolling(window=60, closed="left").sum()
        return df_next_x["mean_precipitation"]

    def __append_next_x_days(self, x=15):
        plant = "PSATJIRA"
        df = get_dataset_only_time(plant)
        df = df.drop(df.columns[1:], axis=1)
        last_date = df.index[-1]
        next_x_days = pd.date_range(start=last_date, periods=x+1, inclusive="neither")
        new_index = pd.to_datetime(list(next_x_days))
        df_x_days = pd.DataFrame(index=new_index, columns=df.columns)
        df_new = pd.concat([df, df_x_days], axis=0)
        return create_new_features_by_df(df_new)