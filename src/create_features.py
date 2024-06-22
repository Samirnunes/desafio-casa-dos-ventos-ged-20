import pandas as pd


def create_new_features_by_df(df: pd.DataFrame):
    for i in range(1, 46):
        df[f'lag_{i}'] = df['mean_precipitation'].shift(i)
    df['mean_last_30d'] = df['mean_precipitation'].rolling(window=30, closed="left").sum()
    df['mean_last_60d'] = df['mean_precipitation'].rolling(window=60, closed="left").sum()
    df.index = pd.to_datetime(df.index)
    months = df.index.month
    dummies = pd.get_dummies(months, prefix="m", drop_first=True)
    dummies.index = df.index
    df = pd.concat([df, dummies], axis=1)
    return df


def create_new_features(ts_dict: dict):
    for key, df in ts_dict.items():
        df = create_new_features_by_df(df)
        path = f"../data/ts-{key}.csv"
        df.to_csv(path)
