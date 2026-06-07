"""Unified column names for all DataFrames in the library.

The convention is borrowed from MTS RecTools so that the API is idiomatic for its users.
All components (methods, metrics, splitters, adapters) exchange long-format
DataFrames with these column names.
"""

from __future__ import annotations

from typing import Final


class Columns:
    """Namespace with fixed column names.

    Usage::

        from warmtransfer.columns import Columns as C
        df = df.rename(columns={"uid": C.User, "iid": C.Item})
    """

    User: Final = "user_id"
    Item: Final = "item_id"
    Weight: Final = "weight"  # interaction weight/strength (float; 1.0 if no weight)
    Datetime: Final = "datetime"  # interaction timestamp
    Score: Final = "score"  # model score (donor or cold-start method)
    Rank: Final = "rank"  # rank in the output, 1..k

    #: Minimal set of interaction columns.
    Interactions: Final = (User, Item)
    #: End-to-end recommendations/predictions format.
    Recommendations: Final = (User, Item, Score, Rank)
    #: Donor scores format (input of a cold-start method).
    Scores: Final = (User, Item, Score)
