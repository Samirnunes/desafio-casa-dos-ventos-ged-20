import pandas as pd
from sklearn.preprocessing import StandardScaler


class PrecipitationPreprocessor:
    def __init__(self):
        self.__scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_train):
        self.__scaler.fit(X_train)
        self.fitted = True
        return self

    def transform(self, X):
        if self.fitted:
            return pd.DataFrame(self.__scaler.transform(X), columns=X.columns, index=X.index)
        return None

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)
