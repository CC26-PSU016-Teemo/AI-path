from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "teemo_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "hasil_feature.csv"

feature_cols = [
    "Biaya_Rata_Rata",
    "Domain_Akademik",
    "Domain_Bisnis_Karir",
    "Domain_Olahraga_E-Sport",
    "Domain_Seni_Kreatif",
    "Domain_Teknologi",
    "Domain_Umum_Lainnya",
    "Jenjang_Encoded",
    "Is_Online",
    "Is_Offline",
]

OUTPUT_COLS = ["Judul", "Jenjang", "Penyelenggara", "Biaya_Rata_Rata"]
competitions_df = pd.read_csv(DATA_PATH).dropna(subset=feature_cols).reset_index(drop=True)


class CosineSimilarityLayer(tf.keras.layers.Layer):
    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        user_vec, event_vec = inputs
        user_vec = tf.nn.l2_normalize(user_vec, axis=1)
        event_vec = tf.nn.l2_normalize(event_vec, axis=1)
        return tf.reduce_sum(user_vec * event_vec, axis=1, keepdims=True)


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[tf.keras.Model, Any, np.ndarray]:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"CosineSimilarityLayer": CosineSimilarityLayer},
    )
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    event_features = scaler.transform(competitions_df[feature_cols]).astype(np.float32)
    return model, scaler, event_features


def _validate_user_profile(user_profile: dict) -> None:
    missing = [col for col in feature_cols if col not in user_profile]
    if missing:
        raise ValueError(f"Missing user_profile keys: {missing}")


def get_top_n_recommendations(user_profile: dict, n: int = 5) -> list[dict]:
    """
    Args:
        user_profile: dict with keys matching feature_cols (raw/unscaled values)
        n: number of top recommendations to return
    Returns:
        list of dicts with keys: Judul, Jenjang, Penyelenggara, Biaya_Rata_Rata, score
    """
    _validate_user_profile(user_profile)
    model, scaler, event_features = _load_artifacts()

    n = max(1, min(int(n), len(competitions_df)))
    user_raw = pd.DataFrame([{col: user_profile[col] for col in feature_cols}])
    user_features = scaler.transform(user_raw).astype(np.float32)
    user_features = np.repeat(user_features, len(event_features), axis=0)

    scores = model.predict([user_features, event_features], verbose=0).reshape(-1)
    top_indices = np.argsort(scores)[::-1][:n]

    recommendations = []
    for idx in top_indices:
        item = competitions_df.iloc[int(idx)][OUTPUT_COLS].to_dict()
        item["Biaya_Rata_Rata"] = float(item["Biaya_Rata_Rata"])
        item["score"] = float(scores[int(idx)])
        recommendations.append(item)

    return recommendations


if __name__ == "__main__":
    tech_mahasiswa_online = {
        "Biaya_Rata_Rata": 0,
        "Domain_Akademik": 0,
        "Domain_Bisnis_Karir": 1,
        "Domain_Olahraga_E-Sport": 0,
        "Domain_Seni_Kreatif": 0,
        "Domain_Teknologi": 1,
        "Domain_Umum_Lainnya": 0,
        "Jenjang_Encoded": 4,
        "Is_Online": 1,
        "Is_Offline": 0,
    }
    results = get_top_n_recommendations(tech_mahasiswa_online, n=5)
    for result in results:
        print(result)
