import pandas as pd


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
    df['date_diff'] = (df['date_ref'] - df['date_forecast']).dt.days

    pivot_df = df.pivot_table(index=['ana_code', 'date_forecast'],
                              columns='date_diff',
                              values='mean_precipitation',
                              fill_value=0)
    transformed_df = pivot_df.reset_index()
    transformed_df.columns.name = None  # Remove the column name (date_diff)
    days_columns = [f'{name}_d{col}' for col in transformed_df.columns[2:]]
    transformed_df.columns = ['ana_code', 'date_forecast'] + days_columns

    for code in transformed_df["ana_code"].unique():
        path = f"../data/ts-{code}-{name}-model.csv"
        code_df = transformed_df[transformed_df["ana_code"] == code].drop(columns='ana_code')
        code_df.to_csv(path, index=False)

def create_new_features(ts_dict: dict):
    for key, df in ts_dict.items():
        for i in range(1, 46):
            df[f'lag_{i}'] = df['mean_precipitation'].shift(i)

        ts_dict[key] = df
        df['mean_last_30d'] = df['mean_precipitation'].rolling(window=30).sum()
        df['mean_last_60d'] = df['mean_precipitation'].rolling(window=60).sum()
        df.index = pd.to_datetime(df.index)
        months = df.index.month
        dummies = pd.get_dummies(months, prefix="m", drop_first=True)
        dummies.index = df.index
        df = pd.concat([df, dummies], axis=1)

        path = f"../data/ts-{key}.csv"
        df.to_csv(path)
