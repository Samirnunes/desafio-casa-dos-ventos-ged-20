import pandas as pd


def get_plants():
    df = pd.read_csv("../data/merge.csv")
    return df["ana_code"].unique()


def import_precipitation_ts():
    ts_dict = {}
    for plant in get_plants():
        ts_dict[plant] = pd.read_csv(f"../data/ts-{plant}.csv", index_col=0)
    return ts_dict

def get_dataset_only_time(plant="PSATJIRA"):
    ts_dict = import_precipitation_ts()
    return ts_dict[plant].dropna(axis=0)


def get_dataset_with_cfs_gefs(plant="PSATJIRA"):
    df = get_dataset_only_time(plant)
    dfcg = df.copy()
    c = pd.read_csv(f"../data/ts-{plant}-cfs-model.csv", index_col=[0])
    g = pd.read_csv(f"../data/ts-{plant}-gefs-model.csv", index_col=[0])
    dfcg = pd.concat([dfcg, c], axis=1).dropna(axis=0)
    dfcg = pd.concat([dfcg, g], axis=1).dropna(axis=0)
    return dfcg


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