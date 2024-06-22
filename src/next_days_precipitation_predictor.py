import pandas as pd
from sklearn.base import clone

from data_import import get_dataset_only_time, get_dataset_with_cfs_gefs
from create_features import create_new_features_by_df
from precipitation_preprocessor import PrecipitationPreprocessor


class NextDaysPrecipitationPredictor:
    def __init__(self, model, plant="PSATJIRA"):
        self.__model = clone(model)
        self.__plant = plant

    def predict_next_x_days(self, x=15, use_cfs_gefs=False):
        if use_cfs_gefs:
            df_train = get_dataset_with_cfs_gefs(self.__plant)
        else:
            df_train = get_dataset_only_time(self.__plant)
        X_train = df_train.drop(["mean_precipitation"], axis=1)
        y_train = df_train["mean_precipitation"]
        preprocessor = PrecipitationPreprocessor()
        X_train = preprocessor.fit_transform(X_train)
        model = clone(self.__model)
        model.fit(X_train, y_train)
        if use_cfs_gefs:
            df = self.__append_next_x_days_cfs_gefs(x)
        else:
            df = self.__append_next_x_days(x)
        df = df[df_train.columns]
        for i in range(x):
            df_next_x = df.tail(x)
            df_X = df_next_x.drop(["mean_precipitation"], axis=1).copy()
            X = preprocessor.transform(pd.DataFrame(df_X.iloc[i]).T)
            y = model.predict(X)
            for j, k in zip(range(i, df_next_x.shape[0]), range(0, df_next_x.shape[1])):
                df_next_x.iloc[j, k] = y
            df.iloc[range(len(df)-x, len(df)), :] = df_next_x
            df['mean_last_30d'] = df['mean_precipitation'].rolling(window=30, closed="left").sum()
            df['mean_last_60d'] = df['mean_precipitation'].rolling(window=60, closed="left").sum()
        return df_next_x["mean_precipitation"]

    def __append_next_x_days_cfs_gefs(self, x=15):
        df = get_dataset_with_cfs_gefs(self.__plant)[["mean_precipitation"]]
        last_date = df.index[-1]
        next_x_days = pd.date_range(start=last_date, periods=x+1, inclusive="neither")
        new_index = pd.to_datetime(list(next_x_days)).date
        df_x_days = pd.DataFrame(index=new_index, columns=df.columns)
        df_new = pd.concat([df, df_x_days], axis=0)
        df_new = create_new_features_by_df(df_new)
        c = pd.read_csv(f"../data/ts-{self.__plant}-cfs-model.csv", index_col=[0])
        g = pd.read_csv(f"../data/ts-{self.__plant}-gefs-model.csv", index_col=[0])
        c.index = pd.to_datetime(c.index)
        g.index = pd.to_datetime(g.index)
        df_new.index = pd.to_datetime(df_new.index)
        df_new = pd.concat([df_new, c, g], axis=1, join="inner")
        return df_new

    def __append_next_x_days(self, x=15):
        df = get_dataset_only_time(self.__plant)[["mean_precipitation"]]
        last_date = df.index[-1]
        next_x_days = pd.date_range(start=last_date, periods=x+1, inclusive="neither")
        new_index = pd.to_datetime(list(next_x_days)).date
        df_x_days = pd.DataFrame(index=new_index, columns=df.columns)
        df_new = pd.concat([df, df_x_days], axis=0)
        df_new = create_new_features_by_df(df_new)
        return df_new
