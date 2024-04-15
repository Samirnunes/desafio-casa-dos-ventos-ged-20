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
    return  df["ana_code"].unique()

def import_precipitation_ts():
    df_dict = {}
    for plant in get_plants():
        df_dict[plant] = pd.read_csv(f"../data/ts-{plant}.csv", index_col=0)
    return df_dict