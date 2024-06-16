import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn import clone
import numpy as np
import matplotlib.pyplot as plt

def cross_validation(model, X, y):
    tscv = TimeSeriesSplit(n_splits=10)
    rmse = []
    eval_count = 0
    for train_index, val_index in tscv.split(X):
        print(f"Evaluation {eval_count}:")
        print(f"Train size: {len(train_index)}")
        print(f"Val size: {len(val_index)}")
        model_ = clone(model)
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model_.fit(X_train, y_train)
        y_pred = pd.DataFrame(model_.predict(X_val), index=y_val.index)[0]
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"RMSE: {rmse_val}\n")
        rmse.append(rmse_val)
        eval_count += 1
    print(f"Mean rmse: {sum(rmse)/len(rmse)}")
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(rmse)
    axs[0].set_title("RMSE histogram")
    axs[0].set_xlabel("RMSE")
    axs[0].set_ylabel("Count")
    axs[1].plot(y_val.index, y_val)
    axs[1].plot(y_pred.index, y_pred)
    axs[1].set_xticks([])
    axs[1].set_xlabel("Date")
    axs[1].set_title("Last iteration plot")
    axs[1].legend(["y_true", "y_pred"])
    plt.show()

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

def preprocess(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train, X_test

def evaluate(model, X_val, y_val):
    y_pred = pd.DataFrame(model.predict(X_val), index=y_val.index)[0]
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_val, y_pred))}")
    y_val.plot()
    y_pred.plot()
    plt.legend(["validation", "prediction on validation"])
    plt.xticks(rotation=45)
    plt.show()
    