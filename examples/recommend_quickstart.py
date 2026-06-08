"""One-call warmtransfer onboarding: which cold-start method fits MY data?

The user brings: interactions (warm history), content features (all items), and donor scores
(their trained model's scores over warm items). ``recommend`` evaluates every feasible method
on an honest holdout and prints a leaderboard + verdict, then we score a new cold item.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import warmtransfer as wt
from warmtransfer.columns import Columns as C
from warmtransfer.types import ItemFeatures

rng = np.random.default_rng(0)
n_users, n_items, n_feat = 60, 40, 6
item_feat = rng.random((n_items, n_feat))
user_pref = rng.random((n_users, n_feat))
affinity = user_pref @ item_feat.T

rows, scores = [], []
for u in range(n_users):
    for it in np.argsort(-affinity[u])[:8]:
        rows.append((u, int(it)))
    for it in range(n_items):
        scores.append((u, it, float(affinity[u, it])))

interactions = pd.DataFrame(rows, columns=[C.User, C.Item])
donor_scores = pd.DataFrame(scores, columns=[C.User, C.Item, C.Score])
content = ItemFeatures(
    item_ids=np.arange(n_items), matrix=item_feat,
    feature_names=[f"f{i}" for i in range(n_feat)],
)

result = wt.recommend(interactions, content, donor_scores, seed=42)

if __name__ == "__main__":
    print(result)  # leaderboard + verdict
    print("\nAbsolute best:", result.best)
    print("Best transfer (used by predict):", result.best_transfer)
    reco = result.predict(user_ids=np.array([0, 1, 2]), cold_item_ids=np.array([0]))
    print(reco.to_string(index=False))
