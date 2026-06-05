"""
warmtransfer — post-hoc, model-agnostic item cold-start for recommenders.

Wrap an already-trained recommendation model (treated as a black box) plus
item content features, and predict scores for new items the model never saw —
without retraining the model.
"""

__version__ = "0.1.0"
