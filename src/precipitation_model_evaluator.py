import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from precipitation_preprocessor import PrecipitationPreprocessor


class PrecipitationEvaluator:
    def __init__(self, model):
        self.__model = clone(model)

    def cross_validation(self, X, y, save_path="./results/PSATJIRA/"):
        tscv = TimeSeriesSplit(n_splits=10)
        rmse_val_list = []
        rmse_train_list = []
        mae_val_list = []
        mae_train_list = []
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

            mae_val = mean_absolute_error(y_val, y_pred)
            mae_val_list.append(mae_val)
            mae_train = mean_absolute_error(y_train, model.predict(X_train))
            mae_train_list.append(mae_train)

            eval_count += 1
            print(f"Evaluation {eval_count} completed.", end="\r")

        mean_rmse_validation = sum(rmse_val_list)/len(rmse_val_list)
        mean_rmse_train = sum(rmse_train_list)/len(rmse_train_list)
        mean_mae_validation = sum(mae_val_list)/len(mae_val_list)
        mean_mae_train = sum(mae_train_list)/len(mae_train_list)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            df = pd.DataFrame({
                "RMSE_train": rmse_train_list,
                "RMSE_validation": rmse_val_list,
                "MAE_train": mae_train_list,
                "MAE_validation": mae_val_list
            })
            df.to_csv(f"{save_path}metrics_cs.csv", index_label="Epoch")

        with open(save_path + "mean_metrics_cs.json", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "rmse": {"validation": mean_rmse_validation, "train": mean_rmse_train},
                "mae": {"validation": mean_mae_validation, "train": mean_mae_train}
            }))

        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 3])

        ax_hist1 = fig.add_subplot(gs[0, 0])
        ax_hist2 = fig.add_subplot(gs[0, 1])
        ax_line = fig.add_subplot(gs[1, :])

        ax_hist1.hist(rmse_val_list, color='skyblue', edgecolor='black')
        ax_hist1.set_title("RMSE histogram")
        ax_hist1.set_xlabel("RMSE")
        ax_hist1.set_ylabel("Count")
        ax_hist2.hist(mae_val_list, color='salmon', edgecolor='black')
        ax_hist2.set_title("MAE histogram")
        ax_hist2.set_xlabel("MAE")
        ax_hist2.set_ylabel("Count")

        ax_line.plot(y_val.index, y_val)
        ax_line.plot(y_pred.index, y_pred)
        ax_line.set_xticks([])
        ax_line.set_xlabel("Date")
        ax_line.set_title("Last iteration plot")
        ax_line.legend(["y_true", "y_pred"])

        plt.tight_layout()
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
        mae_test = mean_absolute_error(y_test, y_pred)
        with open(save_path + "metrics_test.json", "w", encoding="utf-8") as f:
            f.write(json.dumps({"rmse": rmse_test, "mae": mae_test}))
