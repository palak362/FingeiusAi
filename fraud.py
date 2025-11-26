import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from ..config import FRAUD_AE_PATH, FRAUD_IF_PATH


def train_isolation_forest(X: pd.DataFrame):
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    model.fit(X)
    joblib.dump(model, FRAUD_IF_PATH)
    return model


def build_autoencoder(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)
    ae = models.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    return ae


def train_autoencoder(X: pd.DataFrame, epochs=20, batch_size=64):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ae = build_autoencoder(Xs.shape[1])
    ae.fit(Xs, Xs, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    # save model and scaler
    ae.save(FRAUD_AE_PATH)
    joblib.dump(scaler, str(FRAUD_AE_PATH)+".scaler.pkl")
    return ae
