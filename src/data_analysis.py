import pandas as pd
import matplotlib.pyplot as plt

def plot_plant_ts(ts_dict, plant="PSATCNV"):
    ts_dict[plant].plot()
    plt.show()