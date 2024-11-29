import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler

model = joblib.load('nn_model.pkl')

input = pd.read_csv('/files/input.csv')

categorical_cols = ["WindDir3pm", "WindDir9am", "WindGustDir", "RainToday", "Location"]
numerical_median_cols = ["Pressure3pm", "Pressure9am", "Temp3pm", "Temp9am", "MinTemp", "MaxTemp"]
numerical_knn_cols = ["Evaporation", "Rainfall", "Humidity3pm", "WindSpeed3pm", "WindSpeed9am", "Cloud3pm", "Cloud9am", "Humidity9am", "Sunshine", "WindGustSpeed"]


class DateMonthExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column="Date", new_column="Month"):
        self.date_column = date_column
        self.new_column = new_column
    
    def fit(self, X, y=None):
        return self  # No necesita ajuste
    
    def transform(self, X):
        X = X.copy()  # Crear una copia para no modificar los datos originales
        X[self.date_column] = pd.to_datetime(X[self.date_column])  # Convertir a datetime
        X[self.new_column] = X[self.date_column].dt.month  # Extraer el mes
        X.drop(columns=[self.date_column], inplace=True)  # Eliminar la columna original
        return X


class ProbImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # Columnas a imputar
        self.value_probs = {}  # Diccionario para almacenar valores únicos y probabilidades
    
    def fit(self, X, y=None):
        for col in self.columns:
            values = X[col].dropna().value_counts(normalize=True)
            self.value_probs[col] = (values.index.values, values.values)
        return self
    
    def transform(self, X):
        X = X.copy()  # Crear una copia para no modificar el original
        for col in self.columns:
            if col in self.value_probs:
                values, probs = self.value_probs[col]
                X[col] = X[col].apply(
                    lambda x: np.random.choice(values, p=probs) if pd.isnull(x) else x
                )
        return X


# Imputación para columnas numéricas con la mediana
def apply_median_imputation(X, columns):
    imputer_median = SimpleImputer(strategy="median")
    X[columns] = imputer_median.fit_transform(X[columns])
    return X

# Imputación usando KNN para columnas numéricas
def apply_knn_imputation(X, columns):
    knn_imputer = KNNImputer()
    X[columns] = knn_imputer.fit_transform(X[columns])
    return X

# Escalado de características numéricas
def apply_scaling(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

class DummiesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        
    def fit(self, X, y=None):
        return self  # No necesita ajuste
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)  # Convertir a DataFrame si es un ndarray
        X = X.copy()  # Crear una copia para no modificar los datos originales
        
        # Verificar si las columnas están presentes en X
        missing_cols = [col for col in self.categorical_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Las siguientes columnas no están en el DataFrame: {missing_cols}")
        
        # Convertir las columnas categóricas a tipo string
        X.loc[:, self.categorical_cols] = X.loc[:, self.categorical_cols].astype(str)
        
        # Aplicar pd.get_dummies
        X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
        return X

input = apply_median_imputation(input, numerical_median_cols)
input = apply_knn_imputation(input, numerical_knn_cols)

    # Crear transformadores adicionales (imputación probabilística y extracción de fecha)
prob_imputer = ProbImputer(columns=categorical_cols)
date_transformer = DateMonthExtractor(date_column="Date", new_column="Month")
    
    # Transformar los datos de entrenamiento y prueba
input = prob_imputer.fit_transform(input)
input = date_transformer.fit_transform(input)

    
    # Codificar variables categóricas
encoder = DummiesEncoder(categorical_cols=categorical_cols)
input = encoder.fit_transform(input)

# Cargar las columnas esperadas del entrenamiento
train_columns = joblib.load("train_columns.pkl")


# Ajustar las columnas del DataFrame de entrada
input = pd.DataFrame(input, columns=train_columns)  # Asegura el orden correcto
input = input.reindex(columns=train_columns, fill_value=0)  # Rellena columnas faltantes con ceros
input.fillna(0, inplace=True)

# Escalar las características
input = apply_scaling(input)

output = model.predict(input)
output = (output > 0.5).astype(int)

pd.DataFrame(output, columns=['Rain_Prediction']).to_csv('/files/output.csv', index=False)