import pandas as pd
from sklearn.model_selection import train_test_split
import math


def separate_plants_ts():
    merge = pd.read_csv("../data/merge.csv")
    for code in merge["ana_code"].unique():
        path = f"../data/ts-{code}.csv"
        ts = merge[merge["ana_code"] == code].sort_values(by="date_ref").reset_index(drop=True)
        ts.index = pd.to_datetime(ts["date_ref"])
        ts = ts.drop(["date_ref", "ana_code"], axis=1)
        ts.to_csv(path)


def get_plants():
    df = pd.read_csv("../data/merge.csv")
    return df["ana_code"].unique()


def import_precipitation_ts():
    df_dict = {}
    for plant in get_plants():
        df_dict[plant] = pd.read_csv(f"../data/ts-{plant}.csv", index_col=0)
    return df_dict


def separate_predictions_plants_ts(name: str):
    df = pd.read_csv(f"../data/{name}.csv")
    df['date_ref'] = pd.to_datetime(df['date_ref'])
    df['date_forecast'] = pd.to_datetime(df['date_forecast'])
    df['date_diff'] = (df['date_forecast'] - df['date_ref']).dt.days

    pivot_df = df.pivot_table(index=['ana_code', 'date_ref'],
                              columns='date_diff',
                              values='mean_precipitation',
                              fill_value=0)
    transformed_df = pivot_df.reset_index()
    transformed_df.columns.name = None  # Remove the column name (date_diff)
    days_columns = [f'{name}_d+{col}' for col in transformed_df.columns[2:]]
    transformed_df.columns = ['ana_code', 'date_ref'] + days_columns

    for code in transformed_df["ana_code"].unique():
        path = f"../data/ts-{code}.csv"
        ts = pd.read_csv(path)
        ts['date_ref'] = pd.to_datetime(ts['date_ref'])
        code_df = transformed_df[transformed_df["ana_code"] == code].drop(columns='ana_code')
        merged = pd.merge(code_df, ts, on='date_ref', how='inner')
        merged.to_csv(path, index=False)


def create_new_features(ts_dict: dict):
    for key, df in ts_dict.items():

        # Calculando as datas anteriores e os valores correspondentes
        for i in range(1, 16):
            df[f'-{i}d'] = df['mean_precipitation'].shift(i)

        ts_dict[key] = df
        df['mean_last_15d'] = df['mean_precipitation'].rolling(window=15).sum()
        df['mean_last_30d'] = df['mean_precipitation'].rolling(window=30).sum()


def split_train_test(ts_dict: dict):
    X_train_dict = {}
    X_test_dict = {}
    y_train_dict = {}
    y_test_dict = {}

    for key, df in ts_dict.items():
        train_size = math.ceil(len(ts_dict[key])*0.98)

        X = df.drop(columns=['mean_precipitation'])
        y = df['mean_precipitation']

        X_train = X[:train_size]
        X_test = X[train_size:]

        y_train = X[:train_size]
        y_test = y[train_size:]

        X_train_dict[key] = X_train
        X_test_dict[key] = X_test
        y_train_dict[key] = y_train
        y_test_dict[key] = y_test

    return X_train_dict, X_test_dict, y_train_dict, y_test_dict