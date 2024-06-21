import pandas as pd
import math

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
    