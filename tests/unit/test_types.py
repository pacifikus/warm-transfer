"""Tests for the dataclass contracts of user-facing types."""

from __future__ import annotations

import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.types import TransferInputs


def test_transfer_inputs_train_interactions_optional_for_score_only_methods() -> None:
    donor_scores = pd.DataFrame({C.User: [1], C.Item: [10], C.Score: [1.0]})

    inputs = TransferInputs(donor_scores=donor_scores)

    assert inputs.train_interactions.empty

