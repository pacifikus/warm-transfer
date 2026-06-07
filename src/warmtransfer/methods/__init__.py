"""Cold-start методы. Импорт подмодулей регистрирует их в реестре ``methods``."""

from __future__ import annotations

from warmtransfer.methods.attention_emb import AttentionEmbedding
from warmtransfer.methods.attention_knn import AttentionKNN
from warmtransfer.methods.base import (
    ColdStartMethod,
    cross_join_frame,
    methods,
    register_method,
)
from warmtransfer.methods.baselines import (
    GroupedMostPopular,
    GroupedMostPopularPersonalized,
    MostPopular,
    RandomScorer,
)
from warmtransfer.methods.debiased_knn import DebiasedKNN
from warmtransfer.methods.dropoutnet import DropoutNet
from warmtransfer.methods.embedding_avg import EmbeddingAverage
from warmtransfer.methods.knn import KNNScoreAggregation
from warmtransfer.methods.linmap import LinMap
from warmtransfer.methods.linmap_emb import LinMapEmbedding
from warmtransfer.methods.logreg_calib import LogRegCalibration
from warmtransfer.methods.magnitude_scaling import MagnitudeScaling
from warmtransfer.methods.scale_shift import ScaleShift
from warmtransfer.methods.stacking import StackingTransfer
from warmtransfer.methods.stacking_plus import StackingPlus

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
