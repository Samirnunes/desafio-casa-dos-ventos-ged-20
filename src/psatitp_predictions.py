import os

import numpy as np
import matplotlib.pyplot as plt

from next_days_precipitation_predictor import NextDaysPrecipitationPredictor
from models import psatitp_models, psatitp_models_names


def psatitp_predictions_cfs_gefs():
    for model, name in zip(psatitp_models, psatitp_models_names):
        predictor = NextDaysPrecipitationPredictor(model, "PSATITP")
        mask = np.ones(len(predictor.get_columns(True)), dtype=bool)
        mask[15:45] = False  # lag 16 to 45
        mask[58:88] = False  # lag 16 to 45 from cfs
        predictions = predictor.predict_next_x_days(15, True, mask)
        save_path = f"../prediction_results/PSATITP/{name}_cfs_gefs/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.plot(predictions.index, predictions)
        plt.title(f"Predictions for the next 15 days - {name}")
        plt.xlabel("Day")
        plt.ylabel("Prediction")
        plt.xticks(predictions.index, rotation=45)
        plt.savefig(save_path + "plots.png")
        plt.close()

        with open(save_path + "predictions.txt", "w", encoding="utf-8") as f:
            f.write("Predictions for each day (15 days ahead)\n")
            for prediction in zip(predictions.index, predictions):
                f.write(f"{prediction}\n")


def psatitp_predictions_time():
    for model, name in zip(psatitp_models, psatitp_models_names):
        predictor = NextDaysPrecipitationPredictor(model, "PSATITP")
        mask = np.ones(len(predictor.get_columns()), dtype=bool)
        mask[15:45] = False  # lag 16 to 45
        predictions = predictor.predict_next_x_days(15, False, mask)
        save_path = f"../prediction_results/PSATITP/{name}_time/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.plot(predictions.index, predictions)
        plt.title(f"Predictions for the next 15 days - {name}")
        plt.xlabel("Day")
        plt.ylabel("Prediction")
        plt.xticks(predictions.index, rotation=45)
        plt.savefig(save_path + "plots.png")
        plt.close()

        with open(save_path + "predictions.txt", "w", encoding="utf-8") as f:
            f.write("Predictions for each day (15 days ahead)\n")
            for prediction in zip(predictions.index, predictions):
                f.write(f"{prediction}\n")


if __name__ == "__main__":
    psatitp_predictions_time()
    psatitp_predictions_cfs_gefs()
