"""Donor: Two-Tower (neural, torch).

Two independent encoders (towers) over the user and item id-embeddings; the pair score
= dot product of their representations. This family is separate from matrix factorization
(als/bpr), GBDT (catboost) and linear item-item (ease): here the representations are
nonlinear (MLP), but the score stays a dot product → the embeddings are clean and
separable, as the [EMB] methods require.

Training: BCE with negative sampling over implicit interactions. The hyperparameters are
fixed at reasonable values (NOT tuned to beat the baseline). The device is chosen
automatically: cuda if available, otherwise cpu.

embeddings() returns the final tower representations for warm items and users — cold
items are absent from training, their representations do not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer._pdutils import map_codes, unique_sorted
from warmtransfer.bench.adapters.base import ModelAdapter, register_adapter
from warmtransfer.columns import Columns as C
from warmtransfer.schema import validate_interactions
from warmtransfer.seeding import set_global_seed
from warmtransfer.types import Dataset


@register_adapter("two_tower")
class TwoTowerAdapter(ModelAdapter):
    """Two-Tower: dual id-embedding + MLP, score = dot(user_vec, item_vec).

    :param emb_dim: dimensionality of the id-embeddings at the tower input.
    :param out_dim: dimensionality of the output representation (the score latent space).
    :param hidden: hidden layer size of each tower's MLP.
    :param epochs: number of training epochs.
    :param batch_size: batch size of positive examples.
    :param lr: learning rate (Adam).
    :param weight_decay: Adam L2 regularization.
    :param n_negatives: number of random negatives per positive.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        out_dim: int = 64,
        hidden: int = 128,
        epochs: int = 10,
        batch_size: int = 4096,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_negatives: int = 4,
    ) -> None:
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_negatives = n_negatives
        self._user_ids: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None
        self._user_vecs: np.ndarray | None = None
        self._item_vecs: np.ndarray | None = None

    def _build_model(self, n_users: int, n_items: int):
        from torch import nn

        class Tower(nn.Module):
            def __init__(self, n: int, emb_dim: int, hidden: int, out_dim: int) -> None:
                super().__init__()
                self.emb = nn.Embedding(n, emb_dim)
                nn.init.normal_(self.emb.weight, std=0.01)
                self.mlp = nn.Sequential(
                    nn.Linear(emb_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, out_dim),
                )

            def forward(self, idx):
                return self.mlp(self.emb(idx))

        class TwoTower(nn.Module):
            def __init__(self, n_u, n_i, emb_dim, hidden, out_dim) -> None:
                super().__init__()
                self.user_tower = Tower(n_u, emb_dim, hidden, out_dim)
                self.item_tower = Tower(n_i, emb_dim, hidden, out_dim)

        return TwoTower(n_users, n_items, self.emb_dim, self.hidden, self.out_dim)

    def fit(self, dataset: Dataset, seed: int = 0) -> TwoTowerAdapter:
        import torch
        from torch import nn

        set_global_seed(seed)
        torch.manual_seed(seed)
        inter = validate_interactions(dataset.interactions)

        self._user_ids = unique_sorted(inter[C.User])
        self._item_ids = unique_sorted(inter[C.Item])
        self._u_pos = {u: i for i, u in enumerate(self._user_ids)}
        self._i_pos = {it: j for j, it in enumerate(self._item_ids)}
        n_u, n_i = len(self._user_ids), len(self._item_ids)

        u_idx = np.asarray(map_codes(inter[C.User], self._u_pos), dtype=np.int64)
        i_idx = np.asarray(map_codes(inter[C.Item], self._i_pos), dtype=np.int64)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model(n_u, n_i).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()

        users_t = torch.from_numpy(u_idx).to(device)
        items_t = torch.from_numpy(i_idx).to(device)
        n_pos = users_t.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(seed)

        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n_pos, generator=gen)
            for start in range(0, n_pos, self.batch_size):
                b = perm[start : start + self.batch_size]
                bu = users_t[b]
                bi = items_t[b]
                # random negatives [batch, n_neg]
                neg = torch.randint(
                    0, n_i, (bu.shape[0], self.n_negatives), generator=gen
                ).to(device)

                uv = model.user_tower(bu)  # [B, d]
                iv = model.item_tower(bi)  # [B, d]
                nv = model.item_tower(neg.reshape(-1)).reshape(
                    bu.shape[0], self.n_negatives, -1
                )  # [B, n_neg, d]

                pos_logit = (uv * iv).sum(-1)  # [B]
                neg_logit = (uv.unsqueeze(1) * nv).sum(-1)  # [B, n_neg]

                logits = torch.cat([pos_logit, neg_logit.reshape(-1)])
                targets = torch.cat(
                    [torch.ones_like(pos_logit), torch.zeros_like(neg_logit.reshape(-1))]
                )
                loss = bce(logits, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # final tower representations for all warm ids (in batches, no gradient)
        model.eval()
        with torch.no_grad():
            self._user_vecs = self._encode(model.user_tower, n_u, device)
            self._item_vecs = self._encode(model.item_tower, n_i, device)
        return self

    @staticmethod
    def _encode(tower, n: int, device) -> np.ndarray:
        import torch

        out = []
        for start in range(0, n, 8192):
            idx = torch.arange(start, min(start + 8192, n), device=device)
            out.append(tower(idx).cpu().numpy())
        return np.concatenate(out, axis=0)

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        """Scores for the cross product user_ids × item_ids (known warm only)."""
        if self._user_vecs is None or self._item_vecs is None:
            raise RuntimeError("TwoTowerAdapter: call fit() before score()")

        u_known = [u for u in user_ids if u in self._u_pos]
        i_known = [it for it in item_ids if it in self._i_pos]
        u_idx = np.array([self._u_pos[u] for u in u_known], dtype=int)
        i_idx = np.array([self._i_pos[it] for it in i_known], dtype=int)

        uv = self._user_vecs[u_idx]
        iv = self._item_vecs[i_idx]
        scores = uv @ iv.T  # [n_u, n_i]

        return pd.DataFrame(
            {
                C.User: np.repeat(u_known, len(i_known)),
                C.Item: np.tile(i_known, len(u_known)),
                C.Score: scores.reshape(-1),
            }
        )

    def embeddings(self) -> dict[str, np.ndarray] | None:
        if self._user_vecs is None or self._item_vecs is None:
            return None
        return {
            "user": self._user_vecs,
            "item": self._item_vecs,
            "user_ids": self._user_ids,
            "item_ids": self._item_ids,
        }

    def get_params(self) -> dict:
        return {
            "emb_dim": self.emb_dim,
            "out_dim": self.out_dim,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "n_negatives": self.n_negatives,
        }
