"""Cold-start methods and baselines implementing the ColdStartMethod interface."""

from warmtransfer.methods.baselines import (
    Random,
    MostPopular,
    GroupedMostPopular,
    ScoreAverage,
)
from warmtransfer.methods.score_agg import ScoreCalibrated
from warmtransfer.methods.embedding_agg import EmbeddingAverage

__all__ = [
    "Random",
    "MostPopular",
    "GroupedMostPopular",
    "ScoreAverage",
    "ScoreCalibrated",
    "EmbeddingAverage",
]
