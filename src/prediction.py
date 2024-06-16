import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn import clone


def cross_validation(model, X, y):
    tscv = TimeSeriesSplit(n_splits=10)
    rmse_val_list = []
    rmse_train_list = []
    eval_count = 0
    for train_index, val_index in tscv.split(X):
        print(f"Evaluation {eval_count}:")
        print(f"Train size: {len(train_index)}")
        print(f"Val size: {len(val_index)}")
        model_ = clone(model)
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        X_train, X_val = preprocess(X_train, X_val)
        model_.fit(X_train, y_train)
        y_pred = pd.DataFrame(model_.predict(X_val), index=y_val.index)[0]
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"RMSE val: {rmse_val:0.2f}")
        rmse_val_list.append(rmse_val)
        rmse_train = np.sqrt(mean_squared_error(y_train, model_.predict(X_train)))
        print(f"RMSE train: {rmse_train:0.2f}\n")
        rmse_train_list.append(rmse_train)
        eval_count += 1

    print(f"Mean RMSE on validation: {sum(rmse_val_list)/len(rmse_val_list):0.2f}")
    print(f"Mean RMSE on train: {sum(rmse_train_list)/len(rmse_train_list):0.2f}")

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[0.4, 0.6])
    ax1.hist(rmse_val_list)
    ax1.set_title("RMSE histogram")
    ax1.set_xlabel("RMSE")
    ax1.set_ylabel("Count")
    ax2.plot(y_val.index, y_val)
    ax2.plot(y_pred.index, y_pred)
    ax2.set_xticks([])
    ax2.set_xlabel("Date")
    ax2.set_title("Last iteration plot")
    ax2.legend(["y_true", "y_pred"])

    plt.tight_layout()
    plt.show()
    plt.close()


def test(model, X_train, X_test, y_train, y_test):
    model_ = clone(model)
    X_train_ = X_train.copy()
    X_test_ = X_test.copy()
    X_train_, X_test_ = preprocess(X_train_, X_test_)
    model_.fit(X_train_, y_train)
    y_pred = model_.predict(X_test_)
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")


def preprocess(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train, X_test


def split_train_test_by_size(df: pd.DataFrame, test_size=50):
    train_index = len(df) - test_size

    X = df.drop(columns=['mean_precipitation'])
    y = df['mean_precipitation']

    X_train = X[:train_index]
    X_test = X[train_index:]

    y_train = y[:train_index]
    y_test = y[train_index:]

    return X_train, X_test, y_train, y_test


def split_train_test_by_percentage(df: pd.DataFrame, percent=0.96):
    train_index = math.ceil(len(df)*percent)

    X = df.drop(columns=['mean_precipitation'])
    y = df['mean_precipitation']

    X_train = X[:train_index]
    X_test = X[train_index:]

    y_train = y[:train_index]
    y_test = y[train_index:]

    return X_train, X_test, y_train, y_test
