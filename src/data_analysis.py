import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

def plot_plant_ts_daily(ts_dict, plant="PSATCNV", close=True):
    plt.figure(figsize=(8,6))
    ts_dict[plant].plot()
    plt.title(f"Precipitação média diária para {plant}")
    folder_path = f"figs-{plant}/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    save_path = f"{folder_path}daily_mean.png"
    plt.savefig(save_path)
    if close:
        plt.close()
    return save_path
       
def plot_plant_ts_accumulated_by_month(ts_dict, plant="PSATCNV", close=True):
    plt.figure(figsize=(8, 6))
    ts = ts_dict[plant]
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').sum()
    df.plot(color="darkblue", legend=False)
    plt.title(f"Precipitação média acumulada por mês para {plant}")
    plt.xlabel("Data de referência")
    folder_path = f"figs-{plant}/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    save_path = f"{folder_path}accumulated_by_month.png"
    plt.savefig(save_path)
    if close:
        plt.close()
    return save_path
    
def plot_plant_ts_accumulated_by_year(ts_dict, plant="PSATCNV", close=True):
    plt.figure(figsize=(8,6))
    ts = ts_dict[plant]
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df[df.index < "2024-01-01"]
    df = df.resample('Y').sum()
    df.plot(color="darkblue", legend=False)
    plt.title(f"Precipitação média acumulada por ano para {plant}")
    plt.xlabel("Data de referência")
    plt.grid()
    folder_path = f"figs-{plant}/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    save_path = f"{folder_path}accumulated_by_year.png"
    plt.savefig(save_path)
    if close:
        plt.close()
    return save_path
    
def plot_plant_ts_mean_by_month(ts_dict, plant="PSATCNV", close=True):
    plt.figure(figsize=(8,6))
    ts = ts_dict[plant]
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').sum()
    df = df.groupby(df.index.month).mean()
    plt.bar(df.index, df["mean_precipitation"], color="darkblue")
    plt.xticks([i+1 for i in range(len(df.index))], list(df.index))
    plt.title(f"Média das precipitações acumuladas por mês para {plant} ao longo dos anos")
    plt.xlabel("Data de referência", fontsize=12)
    folder_path = f"figs-{plant}/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
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
    fig, axs= plt.subplots(2, 2, figsize=(20, 18))
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
    folder_path = f"figs-{plant}/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    save_path = f"{folder_path}all.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()