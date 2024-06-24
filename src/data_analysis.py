import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tabulate import tabulate
from PIL import Image


def plot_plant_ts_daily(ts_dict, plant="PSATCNV", close=True):
    plt.figure(figsize=(8, 6))
    ts_dict[plant]["mean_precipitation"].plot()
    plt.title(f"Precipitação média diária para {plant}")
    folder_path = f"../images/figs-{plant}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = f"{folder_path}daily_mean.png"
    plt.savefig(save_path)
    if close:
        plt.close()
    return save_path


def plot_plant_ts_accumulated_by_month(ts_dict, plant="PSATCNV", close=True):
    plt.figure(figsize=(8, 6))
    ts = ts_dict[plant]["mean_precipitation"]
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').sum()
    df.plot(color="darkblue", legend=False)
    plt.title(f"Precipitação média acumulada por mês para {plant}")
    plt.xlabel("Data de referência")
    folder_path = f"../images/figs-{plant}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = f"{folder_path}accumulated_by_month.png"
    plt.savefig(save_path)
    if close:
        plt.close()
    return save_path


def plot_plant_ts_accumulated_by_year(ts_dict, plant="PSATCNV", close=True):
    plt.figure(figsize=(8, 6))
    ts = ts_dict[plant]["mean_precipitation"]
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df[df.index < "2024-01-01"]
    df = df.resample('Y').sum()
    df.plot(color="darkblue", legend=False)
    plt.title(f"Precipitação média acumulada por ano para {plant}")
    plt.xlabel("Data de referência")
    plt.grid()
    folder_path = f"../images/figs-{plant}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = f"{folder_path}accumulated_by_year.png"
    plt.savefig(save_path)
    if close:
        plt.close()
    return save_path


def plot_plant_ts_mean_by_month(ts_dict, plant="PSATCNV", close=True):
    plt.figure(figsize=(8, 6))
    ts = ts_dict[plant]["mean_precipitation"]
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').sum()
    df = df.groupby(df.index.month).mean()
    plt.bar(df.index, df, color="darkblue")
    plt.xticks([i+1 for i in range(len(df.index))], list(df.index))
    plt.title(f"Média das precipitações acumuladas por mês para {plant} ao longo dos anos")
    plt.xlabel("Data de referência", fontsize=12)
    folder_path = f"../images/figs-{plant}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = f"{folder_path}mean-by-month.png"
    plt.savefig(save_path)
    if close:
        plt.close()
    return save_path


def precipitation_plots(ts_dict, plant="PSATCNV"):
    save_path0 = plot_plant_ts_daily(ts_dict, plant)
    save_path1 = plot_plant_ts_accumulated_by_month(ts_dict, plant)
    save_path2 = plot_plant_ts_accumulated_by_year(ts_dict, plant)
    save_path3 = plot_plant_ts_mean_by_month(ts_dict, plant)
    _, axs = plt.subplots(2, 2, figsize=(20, 18))
    img0 = Image.open(save_path0)
    img1 = Image.open(save_path1)
    img2 = Image.open(save_path2)
    img3 = Image.open(save_path3)
    axs[0, 0].imshow(img0)
    axs[0, 1].imshow(img1)
    axs[1, 0].imshow(img2)
    axs[1, 1].imshow(img3)
    for ax in axs.flatten():
        ax.axis("off")
    folder_path = f"../images/figs-{plant}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = f"{folder_path}all.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.tight_layout()
    plt.show()

def is_stationary(ts_dict, plant="PSATCNV"):
    ts = ts_dict[plant]
    df = ts.copy()

    result = adfuller(df["mean_precipitation"])

    diff = df["mean_precipitation"].diff().dropna()
    result_d = adfuller(diff)

    filtered = df[df["mean_precipitation"] != 0]
    log = np.log(filtered["mean_precipitation"])
    log_diff = log.diff().dropna()
    result_ld = adfuller(log_diff)

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        stat = "\u001b[32mStationary\u001b[0m"
    else:
        stat = "\x1b[31mNon-stationary\x1b[0m"

    if (result_d[1] <= 0.05) & (result_d[4]['5%'] > result_d[0]):
        stat_d = "\u001b[32mStationary\u001b[0m"
    else:
        stat_d = "\x1b[31mNon-stationary\x1b[0m"

    if (result_ld[1] <= 0.05) & (result_ld[4]['5%'] > result_ld[0]):
        stat_ld = "\u001b[32mStationary\u001b[0m"
    else:
        stat_ld = "\x1b[31mNon-stationary\x1b[0m"

    data = [
        ["ADF Statistic", f"{result[0]:0.2f}", f"{result_d[0]:0.2f}", f"{result_ld[0]:0.2f}"],
        ["p-value", f"{result[1]:0.2f}", f"{result_d[1]:0.2f}", f"{result_ld[1]:0.2f}"],
        ["result", stat, stat_d, stat_ld]
    ]
    headers = ["", "Normal", "Diff", "LogDiff"]
    print(tabulate(data, headers=headers, tablefmt="pretty"))


def plot_acf_pacf(ts_dict, plant="PSATCNV"):
    ts = ts_dict[plant]
    df = ts.copy()
    series = df["mean_precipitation"]

    _, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
    plot_acf(series, lags=15, ax=ax[0])
    plot_pacf(series, lags=15, ax=ax[1], method='ols')

    plt.tight_layout()
    plt.show()
    plt.close()
