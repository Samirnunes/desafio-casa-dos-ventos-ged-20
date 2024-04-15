import pandas as pd
import matplotlib.pyplot as plt

def plot_plant_ts_daily(ts_dict, plant="PSATCNV"):
    ts_dict[plant].plot()
    plt.show()
    
def plot_plant_ts_accumulated_by_month(ts_dict, plant="PSATCNV"):
    ts = ts_dict[plant]
    plt.figure()
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').sum()
    df.plot(color="darkblue", legend=False)
    plt.xticks(rotation=45)
    plt.title(f"Precipitação média acumulada por mês para {plant}")
    plt.xlabel("Data de referência")
    plt.show()
    
def plot_plant_ts_accumulated_by_year(ts_dict, plant="PSATCNV"):
    ts = ts_dict[plant]
    plt.figure()
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df[df.index < "2024-01-01"]
    df = df.resample('Y').sum()
    df.plot(color="darkblue", legend=False)
    plt.xticks(rotation=45)
    plt.title(f"Precipitação média acumulada por ano para {plant}")
    plt.xlabel("Data de referência")
    plt.grid()
    plt.show()
    
def plot_plant_ts_mean_by_month(ts_dict, plant="PSATCNV"):
    ts = ts_dict[plant]
    plant = "PSATCNV"
    plt.figure()
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').sum()
    df = df.groupby(df.index.month).mean()
    plt.bar(df.index, df["mean_precipitation"], color="darkblue")
    plt.xticks([i+1 for i in range(len(df.index))], list(df.index))
    plt.title(f"Média das precipitações acumuladas por mês para {plant} ao longo dos anos")
    plt.xlabel("Data de referência")
    plt.show()