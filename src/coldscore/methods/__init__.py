"""Cold-start методы. Импорт подмодулей регистрирует их в реестре ``methods``."""

from __future__ import annotations

from coldscore.methods.attention_emb import AttentionEmbedding
from coldscore.methods.attention_knn import AttentionKNN
from coldscore.methods.base import (
    ColdStartMethod,
    cross_join_frame,
    methods,
    register_method,
)
from coldscore.methods.baselines import (
    GroupedMostPopular,
    GroupedMostPopularPersonalized,
    MostPopular,
    RandomScorer,
)
from coldscore.methods.debiased_knn import DebiasedKNN
from coldscore.methods.dropoutnet import DropoutNet
from coldscore.methods.embedding_avg import EmbeddingAverage
from coldscore.methods.knn import KNNScoreAggregation
from coldscore.methods.linmap import LinMap
from coldscore.methods.linmap_emb import LinMapEmbedding
from coldscore.methods.logreg_calib import LogRegCalibration
from coldscore.methods.magnitude_scaling import MagnitudeScaling
from coldscore.methods.scale_shift import ScaleShift
from coldscore.methods.stacking import StackingTransfer
from coldscore.methods.stacking_plus import StackingPlus

__all__ = [
    "AttentionEmbedding",
    "AttentionKNN",
    "ColdStartMethod",
    "DebiasedKNN",
    "DropoutNet",
    "EmbeddingAverage",
    "GroupedMostPopular",
    "GroupedMostPopularPersonalized",
    "KNNScoreAggregation",
    "LinMap",
    "LinMapEmbedding",
    "LogRegCalibration",
    "MagnitudeScaling",
    "MostPopular",
    "RandomScorer",
    "ScaleShift",
    "StackingPlus",
    "StackingTransfer",
    "cross_join_frame",
    "methods",
    "register_method",
]
