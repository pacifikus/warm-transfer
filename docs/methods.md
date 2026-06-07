# Cold-start transfer methods

Access legend: **[MA]** — model-agnostic post-hoc (only warm scores + content needed);
**[EMB]** — access to donor embeddings required; **[TRAIN]** — intervention in training required.

## Main research conclusion

Naive methods (uniform KNN score averaging, LogReg calibration, embedding-avg)
**lose to the Grouped Most Popular baseline**, because they implicitly reproduce
**global neighbor popularity**. Grouped MP estimates popularity within the cold-item's
category more fairly. Therefore the win is achieved not by replacing the baseline, but by:
1. using the baseline as a **feature/anchor** (Stacking, scale&shift);
2. **popularity debiasing** of the content signal;
3. **attention weighting of neighbors by content** instead of uniform averaging (SimCSR).

## Baselines (reference)

| Method | Idea | Access |
|---|---|---|
| `random` | random score | [MA] |
| `most_popular` | global item popularity | [MA] |
| `grouped_most_popular` | global popularity of the cold-item's genre (same for all users) | [MA] |
| `grouped_most_popular_pers` | **personalized**: user's affinity to the cold-item's genres based on their history. **Main target (per-user AUC ≈ 0.72; external baseline ≈ 0.709)** | [MA] |
| `cold_only` | model trained on cold data only | — |

## Naive aggregation methods (reference)

| Method | Idea | Access |
|---|---|---|
| `knn_score_avg` | averaging scores of the k nearest-by-content warm items | [MA] |
| `logreg_calib` | LogReg calibration of k neighbors' scores + model score | [MA] |
| `embedding_avg` | averaging neighbors' embeddings | [EMB] |

## Strong candidates (goal — beat Grouped MP) — IMPLEMENTED

Empirical result (ML-1M + Goodbooks, ALS donor; target — `grouped_most_popular_pers`):

| Method | Idea | Access | Result |
|---|---|---|---|
| `linmap` | Ridge `content → vector of donor scores across all users` (multi-output); for cold we apply the same regression. = Individual predictor SimCSR in score space | [MA] | **🏆 best.** ML-1M: beats the target on ALL metrics (ndcg@10 0.274 vs 0.213, AUC 0.723 vs 0.720). Goodbooks: beats on AUC (0.773 vs 0.606), loses on top-k |
| `stacking` | meta-logreg: target = interaction fact on val-cold, features = `[genre affinity, genre popularity, knn donor score]` (baseline as a feature) | [MA] | strong. Beats the target on ranking on ML-1M (ndcg@10 0.224), marginally loses on per-user AUC |
| `attn_knn` | softmax-attention over k content neighbors (SimCSR-lite): weight = `softmax(sim/τ)`, value = neighbor score | [MA] | on par with naive KNN: loses to GroupedMP_pers (neighbors' popularity bias is preserved) |
| `debiased_knn` | KNN with subtraction of neighbors' popularity component | [MA] | weaker than KNN on AUC — debiasing removes useful signal |
| `scale_shift` | content KNN over scores + standardization of each cold item across users and fitting to warm statistics (scale `σ*`, shift `μ*`) — MWUF-on-scores idea | [MA] | ⚠️ **weak**: ML-1M AUC 0.652, KION 0.628 (below the target). Per-user normalization kills the inter-item signal. An honest negative result |

**Key finding:** in score space, **direct linear mapping of content to donor scores
(LinMap) turned out to be stronger** than attention/stacking on the knn signal. It transfers the
latent structure learned by the donor (including personalization) through content without the
popularity averaging that sinks naive KNN.

### `stacking_plus` — hybrid (most robust)

`stacking_plus` [MA] — meta-logreg over **[linmap_score, genre_affinity, genre_popularity]**,
trained on val-cold. Uses the strong LinMap signal AND personalized popularity (strong in top-k)
as features. Result: **on ML-1M×ALS it beats all methods, including LinMap, on
every metric**; robust across datasets and donors; recommended by default when a
val-cold fold is available. See the tables in `results/full_matrix.md`.

## [EMB] methods (donor embeddings) — IMPLEMENTED

These methods use the donor's latent factors (ALS/BPR/Two-Tower provide them; CatBoost and EASE do
not — the method is skipped). Empirical result (focused subset: 3 domains × ALS):

| Method | Idea | Access | Result |
|---|---|---|---|
| `embedding_avg` | uniform mean of the embeddings of k content neighbors | [EMB] | weakest [EMB]: on par with naive knn (~0.69 AUC ml-1m) |
| `attention_emb` | content-softmax-weighted mean of neighbors' embeddings (SimCSR at the IT level) | [EMB] | better than embedding_avg thanks to sharp weights |
| `linmap_emb` | Ridge `content → donor latent factors` (Gantner, ICDM 2010); cold score = `user · cold_emb` | [EMB] | **≡ `linmap`** on ALS/BPR (see below): ML-1M AUC 0.723, KION 0.739 — identical to linmap |
| `magnitude_scaling` | `linmap_emb` + popularity debiasing: the cold embedding's norm is pulled toward the mean warm norm `μ_w` (MS, RecSys 2025) | [EMB] | slightly **worse** than linmap_emb (ML-1M 0.715, KION 0.725): here the base does not overestimate popular cold items, and norm alignment removes signal |
| `dropoutnet` | MLP [donor latent (dropout) ⊕ content]→latent; cold: latent=0 (Volkovs 2017) | [EMB], torch | **best ranking on ML-1M** (ndcg@10 0.280, p@1 0.396); on KION weaker than the linear methods |

DropoutNet is the only neural method (extra `deep`, torch, CPU). On ML-1M it gives the
strongest ranking, but it is not universal: on the sparse implicit KION it loses to LinMap.

### 🔑 Finding: `linmap_emb` (Gantner) ≡ `linmap` for bilinear donors

Empirically, `linmap_emb` produced **character-for-character the same metrics** as `linmap`, on ML-1M and KION with
ALS. This is not a bug but a **provable identity**: for a donor with a dot-product score
(`score(u,i) = U_u · V_i`, as in ALS/BPR) the matrix of warm scores equals `S = V_warm · Uᵀ`. Ridge
is linear in the target, and the `αI` regularization lives in feature space (the same for both):

    W_linmap = (XᵀX + αI)⁻¹ Xᵀ S = (XᵀX + αI)⁻¹ Xᵀ (V_warm Uᵀ) = W_emb · Uᵀ,

so `X_cold · W_linmap = (X_cold · W_emb) · Uᵀ` — exactly the prediction of `linmap_emb`. That is,
our `linmap` in score space **is** the attribute-to-latent Gantner mapping for bilinear donors.
Gantner only begins to differ when the donor's score is **not bilinear** (CatBoost, +bias terms) —
then content→factors and content→score diverge. Practical takeaway: on ALS/BPR a separate
`linmap_emb` is redundant; the value of the Gantner mapping is for nonlinear donors.

### `magnitude_scaling` (Magnitude Scaling, RecSys 2025)

Post-hoc debiasing: changes **only the length** of the cold embedding (not the direction), pulling the norm toward the
mean warm norm `μ_w`. Magnitude is a popularity proxy (large norm → systematically
high score). It helps where the cold generator **overestimates** popular items (the failure
mode from the paper on Amazon/Microlens). On our data (ML-1M/KION) the base does not suffer from this, so
MS slightly hurts — an honest, expected result: the method is targeted, not universal. Useful as a
diagnostic tool and a tail regularizer.

## Backlog (not implemented)

- full SimCSR over interaction features [EMB] — partially covered by `attention_emb`/`attention_knn`.
- `deeplinmap` — MLP `content → score` [MA, extra `deep`].
- `mwuf` — meta scaling & shifting [TRAIN].

## On calibration (important correction)

Platt / isotonic calibration is a **monotone** transformation of scores. Therefore it **changes
neither the ranking metrics (Recall/NDCG/MAP/MRR) nor AUC** — all of them are rank-based and invariant to
monotone transformations. Calibration improves only **logloss/Brier** (probability quality).

⚠️ Consequence: LinMap's weak AUC with the CatBoost donor is a **ranking** problem, not a
calibration one; Platt/isotonic will not fix it. The cure is a hybrid (`stacking_plus`) that adds a
donor-independent, well-ranking signal (genre affinity). We keep calibration as an
optional post-processing step for tasks where the probability (logloss) matters, not the order.

## References

- SimCSR — Han & Chun, "Addressing the Item Cold-Start Using Similar Warm Items", 2021
- DropoutNet — Volkovs et al., NeurIPS 2017
- MWUF — Zhu et al., SIGIR 2021, arXiv:2105.04790
- Gantner et al., "Learning Attribute-to-Feature Mappings", ICDM 2010
- On Inherited Popularity Bias in Cold-Start, RecSys 2025, arXiv:2510.11402
- ColdRec toolkit — github.com/YuanchenBei/ColdRec
