"""Минимальный plug&play пример coldscore без coldbench.

Пользователь приносит warm-скоры уже обученного донора и контент warm/cold айтемов.
LinMap учит отображение "контент -> вектор скоров по пользователям" и предсказывает
скоры для новых айтемов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldscore.columns import Columns as C
from coldscore.methods import LinMap
from coldscore.types import ItemFeatures, TransferInputs

warm_features = ItemFeatures(
    item_ids=np.array([10, 11]),
    matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
    feature_names=["genre_action", "genre_drama"],
)
cold_features = ItemFeatures(
    item_ids=np.array([20]),
    matrix=np.array([[1.0, 0.0]]),
    feature_names=["genre_action", "genre_drama"],
)

donor_scores = pd.DataFrame(
    {
        C.User: [1, 1, 2, 2],
        C.Item: [10, 11, 10, 11],
        C.Score: [5.0, 1.0, 1.0, 5.0],
    }
)

inputs = TransferInputs(
    donor_scores=donor_scores,
    warm_features=warm_features,
    cold_features=cold_features,
)

reco = LinMap(alpha=1.0).fit(inputs, seed=42).predict(
    user_ids=np.array([1, 2]),
    cold_item_ids=np.array([20]),
)

if __name__ == "__main__":
    print(reco.to_string(index=False))
