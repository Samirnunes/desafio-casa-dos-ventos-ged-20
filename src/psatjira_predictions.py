import matplotlib.pyplot as plt
from next_days_predictor import NextDaysPredictor
from models import *
import os

def psatjira_predictions_time():
    for model, name in zip(psatjira_models, psatjira_models_names):
        predictor = NextDaysPredictor(model, "PSATJIRA")
        predictions = predictor.predict_next_x_days(15)
        save_path = f"./prediction_results/{name}_time/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.plot(predictions.index, predictions)
        plt.title(f"Predictions for the next 15 days - {name}")
        plt.xlabel("Day")
        plt.ylabel("Prediction")
        plt.xticks(predictions.index, rotation=45)
        plt.savefig(save_path + "plots.png")
        plt.close()
        
        with open(save_path + "predictions.txt", "w") as f:
            f.write("Predictions for each day (15 days ahead)\n")
            for prediction in zip(predictions.index, predictions):
                f.write(f"{prediction}\n")
    
psatjira_predictions_time()