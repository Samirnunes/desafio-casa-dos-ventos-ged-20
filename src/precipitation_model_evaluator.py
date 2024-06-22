import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from precipitation_preprocessor import PrecipitationPreprocessor
import json
import os

class PrecipitationEvaluator:
    def __init__(self, model):
        self.__model = clone(model)
        
    def cross_validation(self, X, y, save_path="./results/PSATJIRA/"):
        tscv = TimeSeriesSplit(n_splits=10)
        rmse_val_list = []
        rmse_train_list = []
        eval_count = 0
        for train_index, val_index in tscv.split(X):
            model = clone(self.__model)
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            preprocessor = PrecipitationPreprocessor()
            X_train = preprocessor.fit_transform(X_train)
            X_val = preprocessor.transform(X_val)
            model.fit(X_train, y_train)
            y_pred = pd.DataFrame(model.predict(X_val), index=y_val.index)[0]
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_val_list.append(rmse_val)
            rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
            rmse_train_list.append(rmse_train)
            eval_count += 1
            print(f"Evaluation {eval_count} completed.", end="\r")
        mean_rmse_validation = sum(rmse_val_list)/len(rmse_val_list)
        mean_rmse_train = sum(rmse_train_list)/len(rmse_train_list)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        with open(save_path + "rmse_validation.txt", "w") as f:
            f.write("RMSE validation by cross validation step\n")
            for value in rmse_val_list:
                f.write(f"{value}\n")
            f.write("RMSE train by cross validation step\n")
            for value in rmse_train_list:
                f.write(f"{value}\n")
        with open(save_path + "mean_rmse_validation.txt", "w") as f:
            f.write(json.dumps({"validation": mean_rmse_validation, "train": mean_rmse_train}))
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(rmse_val_list)
        axs[0].set_title("RMSE histogram")
        axs[0].set_xlabel("RMSE")
        axs[0].set_ylabel("Count")
        axs[1].plot(y_val.index, y_val)
        axs[1].plot(y_pred.index, y_pred)
        axs[1].set_xticks([])
        axs[1].set_xlabel("Date")
        axs[1].set_title("Last iteration plot")
        axs[1].legend(["y_true", "y_pred"])
        plt.savefig(save_path + "plots.png")
        plt.close()
    
    def test(self, X_train, X_test, y_train, y_test, save_path="./results/PSATJIRA/"):
        model = clone(self.__model)
        X_train_ = X_train.copy()
        X_test_ = X_test.copy()
        preprocessor = PrecipitationPreprocessor()
        X_train_ = preprocessor.fit_transform(X_train_)
        X_test_ = preprocessor.transform(X_test_)
        model.fit(X_train_, y_train)
        y_pred = model.predict(X_test_)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        with open(save_path + "rmse_test.txt", "w") as f:
            f.write(json.dumps({"test": rmse_test}))