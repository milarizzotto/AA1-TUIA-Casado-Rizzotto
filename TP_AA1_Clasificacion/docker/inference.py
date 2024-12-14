import joblib
import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.base import BaseEstimator, TransformerMixin

input = pd.read_csv("/files/input.csv")

categorical_cols = ["WindDir3pm", "WindDir9am", "WindGustDir", "RainToday", "Location"]
numerical_median_cols = [
    "Pressure3pm",
    "Pressure9am",
    "Temp3pm",
    "Temp9am",
    "MinTemp",
    "MaxTemp",
]
numerical_knn_cols = [
    "Evaporation",
    "Rainfall",
    "Humidity3pm",
    "WindSpeed3pm",
    "WindSpeed9am",
    "Cloud3pm",
    "Cloud9am",
    "Humidity9am",
    "Sunshine",
    "WindGustSpeed",
]


class ProbImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.value_probs = {}

    def fit(self, X, y=None):
        for col in self.columns:
            values = X[col].dropna().value_counts(normalize=True)
            self.value_probs[col] = (values.index.values, values.values)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in self.value_probs:
                values, probs = self.value_probs[col]
                X[col] = X[col].apply(
                    lambda x: np.random.choice(values, p=probs) if pd.isnull(x) else x
                )
        return X


class DateMonthExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column, new_column):
        self.date_column = date_column
        self.new_column = new_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.new_column] = pd.to_datetime(
            X[self.date_column], errors="coerce"
        ).dt.month
        return X


class DummiesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)


prob_imputer = joblib.load("/files/inference_req/prob_imputer.pkl")
date_transformer = joblib.load("/files/inference_req/date_transformer.pkl")
encoder = joblib.load("/files/inference_req/dummies_encoder.pkl")
median_imputer = joblib.load("/files/inference_req/median_imputer.pkl")
knn_imputer = joblib.load("/files/inference_req/knn_imputer.pkl")
scaler = joblib.load("/files/inference_req/scaler.pkl")
train_columns = joblib.load("/files/inference_req/train_columns.pkl")

# Procesar datos de entrada
input[numerical_median_cols] = median_imputer.transform(input[numerical_median_cols])
input[numerical_knn_cols] = knn_imputer.transform(input[numerical_knn_cols])
input[categorical_cols] = prob_imputer.transform(input[categorical_cols])
input_data = date_transformer.transform(input)
input_data = encoder.transform(input_data)

# Alinear columnas con el conjunto de entrenamiento
for col in train_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[train_columns]

# Escalar
input_data_scaled = scaler.transform(input_data)


# Cargar el modelo ONNX
onnx_model_path = "nn_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

input_array = input_data_scaled.astype(np.float32)
# Realizar la inferencia
input_name = ort_session.get_inputs()[0].name  # Obtener el nombre de la entrada
output_name = ort_session.get_outputs()[0].name  # Obtener el nombre de la salida
predictions = ort_session.run([output_name], {input_name: input_array})[0]

# Procesar el resultado
output = (predictions > 0.5).astype(int)
pd.DataFrame(output, columns=["Rain_Prediction"]).to_csv(
    "/files/output.csv", index=False
)
print("Inferencia completada y resultados guardados en output.csv.")
