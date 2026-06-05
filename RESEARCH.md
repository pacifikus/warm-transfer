# Research Notes — Warm-Transfer Library

> Living document. Updated as the project progresses. Use this as both lab
> notebook and rolling draft of paper sections. Cite, date, and justify decisions.

---

## 1. Project Framing

**Goal.** Build a **post-hoc, model-agnostic library** for item cold-start
recommendation. Given any already-trained recommendation model (BPR, ALS,
LightGCN, Two-Tower, CatBoost, …) and content features of items, the library
must predict scores for new items that the model never saw — **without
retraining or modifying the model**.

**Differentiation from existing work.** Most cold-start methods in the
literature (DropoutNet, MWUF, Heater, contrastive methods) require integrated
training with the base recommender. The ColdRec toolkit (2024) bundles 26 such
methods but assumes you use *its* training pipeline. **No library currently
focuses on truly post-hoc, model-agnostic cold-start handling**, where the
base model is treated as a black box. That is our niche.

**Contributions intended.**
1. **Library artefact.** Python package with a unified wrapper API:
   `WarmTransfer(model, item_features).predict(user_id, cold_item_id)`.
2. **Knowledge artefact.** Systematic comparison of post-hoc cold-start methods
   across multiple datasets and base-model families — when does each method
   work, what determines transfer quality, what are honest baselines.

---

## 2. Literature Review

### Taxonomy of item cold-start methods

| Family | Idea | Key works | Post-hoc? |
|---|---|---|---|
| **Content KNN aggregation** | Pick warm neighbours by content, aggregate model scores | Schein 2002; our baseline | ✅ |
| **Content → latent mapping** | Regress content features onto learned item factors | LinMap (Gantner 2010), DeepMusic (van den Oord 2013) | ✅ |
| **Hybrid co-training** | Train the model itself with content dropout | DropoutNet (NeurIPS 2017), Heater (SIGIR 2020) | ❌ |
| **Meta-learning** | MAML-style adaptation for cold items | MeLU (KDD 2019), MWUF (SIGIR 2021), MetaEmbedding | ❌ |
| **Graph-based** | GNN over item-attribute graphs | Content-based Graph Reconstruction (SIGIR 2024) | ❌ |
| **Generative** | VAE/GAN generate cold embeddings | GoRec, GAR, MoCA-VAE (SIGIR 2022) | ⚠ usually requires retraining |
| **Contrastive** | Self-supervised content/embedding alignment | CLCRec, CCFRec, XSimGCL (2022-2024) | ❌ |
| **Multi-modal** | Combine text + image + structured | M3CSR (RecSys 2024) | ⚠ |
| **LLM-based** | LLM generates features / embeddings / scores | LLMs as Data Augmenters (WWW 2024), Prompt Tuning for Cold-Start (RecSys 2024), MI4Rec (CIKM 2025) | ✅ (some variants) |

**SOTA in 2024-2025** by citation: LLM-based for text-rich domains, contrastive
learning for general-purpose, multi-modal where images/video available, graph
reconstruction for structural signals.

### Existing libraries in this space

- **[ColdRec](https://github.com/YuanchenBei/ColdRec)** (2024-2025). 26 models,
  PyTorch, Optuna tuning, MIT license. **Research toolbox, not productionised.**
  Almost all methods require their training pipeline — not post-hoc.
- **[LightFM](https://github.com/lyst/lightfm)**. Hybrid CF with content
  features, but only its own FM model.
- **[Cornac](https://github.com/PreferredAI/cornac)**. General-purpose,
  includes some cold-start models but not focused.
- **[RecBole](https://github.com/RUCAIBox/RecBole)**. Comprehensive recommender
  benchmark, cold-start as side concern.
- **[Awesome-Cold-Start-Recommendation](https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation)**:
  curated paper list (~200 works).

**Gap our library fills:** post-hoc wrapper. None of the above wraps an
existing trained model and adds cold-start handling without retraining.

### Decision: which approaches to implement in our library

Only post-hoc methods (the ❌ rows above are excluded; users should look at
ColdRec if they want those).

- **KNN-Aggregation (Score Avg)** — simplest baseline ✓ done
- **KNN-Aggregation (Embedding Avg)** — averages embeddings instead of scores ✓ done
- **Calibrated aggregation (Score LogReg)** — logistic regression on aggregated features ✓ done
- **LinMap** — regress content → latent factor matrix ⏳ to-do
- **LLM/SBERT embedding for cold item + alignment** — modern variant ⏳ to-do (optional)

### Honest baselines (must beat these for any claim of usefulness)

- **Random** — sanity floor
- **Most Popular** — non-personalised floor
- **Grouped Most Popular** — category-conditioned popularity; *strong baseline*,
  beats us on tested datasets so far
- **Content-only KNN (no model)** — use content similarity alone, no model — to-do

---

## 3. Method (working draft)

### Setup

- Trained black-box model `M` with embedding access:
  `score = M.predict(user_id, item_id)`, optionally
  `M.user_embedding(user_id)`, `M.item_embedding(item_id)`.
- Item features `F: item_id → R^d` available for *all* items including cold.
- A split-off of warm items used as **pseudo-cold** to train the calibrator.

### Algorithms (post-hoc methods)

1. **Score Aggregation**
   For cold item `c`, find `k` warm neighbours `n_1..n_k` by cosine similarity
   on `F`. Score `(u, c)` = mean / weighted-mean of `M.predict(u, n_i)`.

2. **Embedding Aggregation**
   `emb(c) ≈ Σ sim(c, n_i) · emb(n_i) / Σ sim`. Then
   `score(u, c) = emb(u) · emb(c)`. Requires `M` to expose embeddings.

3. **Calibrated Aggregation (stacking)**
   A meta-learner (logistic regression) is trained on *pseudo-cold* items to map
   a per-`(u, c)` feature vector onto engagement. This is **stacking with
   meta-features** (Wolpert 1992; Sill et al., *Feature-Weighted Linear
   Stacking*, 2009 — the precise lineage; Bao et al., *STREAM*, RecSys 2009) —
   NOT a novel method of ours. These are Netflix-Prize-era, foundational and
   deliberately so: within the post-hoc family this is the recognised
   interpretable representative of "learned aggregation". Newer cold-start SOTA
   (meta-learning / contrastive / GNN, 2023-2024) requires (re)training and is
   therefore out of post-hoc scope (§2). We implement stacking as a faithful
   representative of its family, not as a SOTA claim.

   **Why the previous version was degenerate.** It fed the meta-learner only
   summary statistics of one scalar source (the `k` neighbour model scores:
   mean/max/min/std/sim-weighted-mean/...). All inputs were monotone in the same
   signal, so the learner had nothing distinct to combine and collapsed to ≈ the
   plain mean (empirically Score-LogReg 0.544 vs Score-Avg 0.550 on ML-1M —
   tied/worse). Stacking only adds value when its inputs are of *different
   natures* (Bao 2009). The fix is a principled, heterogeneous feature basis —
   not more statistics of the same scalar.

   **PRE-REGISTERED FEATURE SET (frozen before looking at any result).**
   Per `(user u, cold item c)` with its `k` content-nearest warm neighbours
   `n_1..n_k` (sorted by descending similarity `s_1 >= ... >= s_k`):

   *Group A — model-transfer signal (from the black-box model M):*
   - `a_mean` : plain mean of the neighbour scores `M.predict(u, n_i)`.
   - `a_w`    : similarity-weighted mean `Σ s_i M(u,n_i) / Σ s_i`.

   Group A is deliberately kept to two **k-invariant aggregates** rather than the
   raw rank-ordered top-k vector `a1..ak`. A per-rank vector would make the
   feature dimensionality depend on `k` and let the learner over-fit individual
   neighbour ranks; the two aggregates carry the same transfer signal while
   staying comparable across `k` and across datasets.

   *Group B — neighbourhood-quality meta-features (trust in the transfer):*
   - `b_meansim` : mean similarity `mean(s_i)`.
   - `b_gap`     : top-1 minus top-k similarity `s_1 - s_k` (peakedness).
   - `b_n`       : number of valid neighbours found (coverage).

   *Group C — user-side prior (meta-feature, model-independent):*
   - `c_support` : `log1p(count of u's warm interactions)` (how much we know u).
   - `c_rate`    : u's mean engagement over its warm interactions (u's propensity).

   *Group D — item-side popularity prior (model-independent, raw engagement):*
   - `d_pop`     : mean *raw warm engagement rate* of the `k` neighbours
     (a popularity signal computed from interactions, bypassing M entirely).

   These four groups are genuinely different sources (model transfer / neighbour
   trust / user propensity / content-popularity), which is the whole point of
   stacking. All are computed from the warm interactions + the shared content
   features that *every* method/baseline may use.

   **Honest caveat (read together with §4).** Groups C and D overlap in spirit
   with what Grouped-Most-Popular exploits, so this calibrator may close the gap
   to that baseline. That is acceptable *only because* this feature set is fixed
   from the stacking literature **before** seeing results, is identical across
   all datasets, and is never iterated while watching our own metric. We are
   implementing a known method correctly, not reverse-engineering a baseline. If
   it still loses, that is reported as-is.

   Pipeline unchanged otherwise: `StandardScaler` → `LogisticRegression`
   (logistic) / `Ridge` (ridge), trained on pseudo-cold interactions.

4. **LinMap** *(planned)*
   Train `g: F → R^k` to predict `M.item_embedding(item)` from `F(item)` on
   warm items. For cold `c`, set `emb(c) = g(F(c))`. Then standard scoring.

---

## 4. Experimental Setup

### Benchmark methodology principles (read before touching any feature/preprocessing)

Recommender benchmarks are notoriously sensitive to preprocessing and feature
choices — to the point where uncontrolled preprocessing produces non-comparable
results and fake progress (Dacrema et al., *Are We Really Making Much Progress?*,
RecSys 2019; Rendle et al. on baseline tuning). To stay honest, this project
commits to the following rules. They are binding, not aspirational.

1. **Features are part of the benchmark SPEC, not part of our method.** Every
   method (ours and all baselines/competitors) consumes the *same* fixed feature
   set per dataset. Feature quality is then a controlled constant that cancels
   out of the comparison. We never let our method use "better" features than the
   baselines it is compared against.

2. **Fix preprocessing ONCE, before looking at results.** We do not iterate
   features/preprocessing while watching our own metric climb. That is
   experimenter-side overfitting to the test set — the cardinal sin. Pick a
   reasonable, documented feature set per dataset and freeze it.

3. **Prefer standard/established preprocessing** (standard MovieLens splits,
   5-core Amazon, feature sets used in prior work) so results are comparable to
   the literature rather than floating in a vacuum.

4. **Preprocessing sensitivity is to be MEASURED and REPORTED, not optimised
   away.** If a construction choice (e.g. how similarity weights category vs
   text) affects results, that is an *ablation axis we report*, not a knob we
   turn until our method wins.

5. **A simple baseline beating us is a legitimate finding, not a bug.** When
   Grouped Most Popular beats embedding transfer, the scientific question is
   *why* (analysis), not *how do we engineer around it* (gaming). A nuanced /
   negative result ("category popularity is hard to beat for item cold-start;
   transfer helps only when condition Z holds") is a real contribution.

**Consequence for positioning.** Because the project is *post-hoc and
model-agnostic*, feature engineering is explicitly **not** our contribution. The
benchmark takes whatever features the dataset/user provides as given; our
contribution is the transfer mechanism on top of fixed features and a fixed,
black-box model. This deliberately removes the "did we just pick good features?"
burden from any of our claims.

### Datasets

| Dataset | Status | Domain | Signal | Pos rate | Items | Interactions |
|---|---|---|---|---|---|---|
| MovieLens-1M | ✅ ready | Movies | Rating ≥ 4 | ~57% | 3,883 | 1.0M |
| Amazon Toys & Games | ✅ ready | E-commerce | Rating ≥ 4 | ~86% | 78,698 | 1.8M |
| LastFM-360K | ⏳ planned | Music | Implicit play count | — | — | — |
| Goodbooks-10k | ⏳ planned | Books | Rating | — | — | — |
| Steam | ⏳ planned | Games | Implicit playtime | — | — | — |

#### Standard processed-CSV contract (what every dataset must produce)

Adding a dataset = writing a `prepare_*.py` that emits these two files; loading is
uniform (`ProcessedDataset` + `validate_schema`). The benchmark never special-cases
a dataset, so the contract is the only thing prep scripts must honour.

`interactions.csv`
- Required columns: `user_id`, `item_id`, `engagement`.
- `engagement` is **binary** {0, 1}. The positive→1 rule lives in the prep script
  (it is the one genuine per-dataset difference): explicit ratings → `rating >= 4`;
  implicit signals (play count, playtime) → threshold stated in that script. Optional
  `timestamp` is kept when available but not required.

`item_features.csv`
- Required column `item_id` (unique, no duplicate rows) + **≥1 numeric feature** column.
- All feature columns must be numeric (multi-hot / counts / normalised scalars). Text,
  categories, prices are encoded in the prep step, not at load time.

Cold/warm policy (orchestration, not dataset-specific)
- An item is **warm** iff its interaction count `>= N` (`n_threshold`), else **cold**.
  `N` is a benchmark knob set by the runner, identical across methods on a given run.
- Features for cold items must come *only* from `item_features.csv` (content), never from
  interactions — that is the whole point of the cold-start setting.

`validate_schema` enforces all of the above and fails loudly so a malformed prep output
errors at load time instead of silently corrupting downstream results.

### Base models

| Model | Family | Status |
|---|---|---|
| BPR | Pairwise MF | ✅ done |
| ALS | Pointwise weighted MF | ⏳ planned |
| LightGCN | Graph CF | ⏳ planned |
| CatBoost on features | Gradient boosting | ⏳ planned (control: natively cold-start friendly) |

### Metrics

- **AUC** (global, sensitive to class imbalance)
- **NDCG@k, Recall@k, MAP@k** (per cold item, averaged)
- **MRR** (per cold item, averaged)
- **RelaImpr** relative to Random

### Evaluation protocol

- Warm/cold split by item interaction count threshold `N`.
- BPR validation: 90/10 row-level hold-out on warm to measure model
  generalisation (val AUC). Decoupled from cold-start evaluation.
- All cold items used unless explicitly capped (state `MAX_COLD_EVAL`).
- Random seed fixed for reproducibility.

---

## 5. Experiments Log

### 2026-05-25 — Initial pipeline, Amazon + BPR (single point)

Hyper-tuning attempts on BPR for Amazon Toys converged to a ceiling. Three
distinct configurations all gave the same generalisation:

| BPR config | Train AUC | Val AUC |
|---|---|---|
| 200 iter, lr=0.05, reg=0.001, factors=64 | 99.94% | **0.6412** |
| 200 iter, lr=0.05, reg=0.05, factors=64 | 99.35% | **0.6412** |
| 200 iter, lr=0.05, reg=0.01, factors=32 | 99.43% | **0.6387** |
| 50 iter, lr=0.01, reg=0.01, factors=64 | 57.13% | **0.5214** (underfit) |

Conclusion: ~0.64 is the genuine ceiling of BPR on this dataset; the gap
between train AUC (~99%) and val AUC (~0.64) is intrinsic memorisation due to
parameter / data ratio (~10x more params than positive interactions).

### 2026-05-25 — Cold-start results, full 70K cold items

**Amazon Toys & Games** (BPR with val AUC 0.64):

| Method | AUC | NDCG@1 | NDCG@5 |
|---|---|---|---|
| Random | 0.5104 | 0.7709 | 0.8118 |
| Most Popular | 0.5001 | 0.7788 | 0.8119 |
| Grouped Most Popular | **0.5647** | **0.8266** | **0.8428** |
| Score Avg (k=10) | 0.5123 | 0.7996 | 0.8241 |
| Score LogReg (k=10) | 0.5110 | 0.7914 | 0.8238 |
| Embedding Avg (k=10) | 0.5113 | 0.7929 | 0.8127 |

**MovieLens-1M** (BPR with val AUC 0.66):

| Method | AUC | NDCG@1 | NDCG@5 |
|---|---|---|---|
| Random | 0.5017 | 0.3977 | 0.4538 |
| Most Popular | 0.4929 | 0.4208 | 0.4474 |
| Grouped Most Popular | **0.7090** | **0.6913** | **0.6877** |
| Score Avg (k=10) | 0.5498 | 0.4786 | 0.4942 |
| Score LogReg (k=10) | 0.5444 | 0.4520 | 0.4840 |
| Embedding Avg (k=10) | 0.5493 | 0.4786 | 0.4939 |

**Observations:**
- Grouped Most Popular dominates our current methods on both datasets.
- Our methods do beat Random and Most Popular on ML-1M (significantly),
  on Amazon by smaller margins.
- The gap to Grouped MP is larger on ML-1M (broader categories, denser
  per-category history) than on Amazon — counter-intuitive at first but
  explained by ML-1M's balanced positive rate making AUC a more discriminating
  metric.

---

## 6. Results Summary (rolling)

### 6.1 Calibrator feature-group ablation (2026-06-06)

Goal: decompose the stacking calibrator's performance into its pre-registered
groups (§3.3) to see whether the **model-transfer signal (group A)** contributes,
or whether the result is "laundered popularity" from the model-independent priors
(groups C user-propensity, D neighbour-popularity). All variants share ONE trained
BPR / similarity / split (`benchmarks/ablation.py`); they differ only in active
groups, so the AUC column isolates each group's marginal contribution.

Base model AUC (warm hold-out): ML-1M 0.66, Amazon 0.63 — both generalise weakly.

AUC on the cold set (the honest discriminative metric):

| variant            | ML-1M | Amazon |
|--------------------|-------|--------|
| Score Avg (ref)    | 0.551 | 0.512  |
| Score[A]   transfer only       | 0.551 | 0.512 |
| Score[A+B] transfer+trust      | 0.551 | 0.513 |
| Score[C+D] priors only, no model | 0.704 | 0.616 |
| Score[B+C+D] no transfer       | 0.704 | 0.616 |
| **Score[full] A+B+C+D**        | **0.707** | **0.621** |
| Grouped Most Popular (ref)     | 0.709 | 0.565 |

**Findings (consistent across both datasets):**

1. **Group A (model-transfer score) adds ≈0 to AUC.** Score[A] ≈ Score Avg ≈
   the plain neighbour-score mean. The calibrator fed only the model signal
   collapses to the mean — confirming the transfer carries no discriminative
   information *beyond* averaging, on both datasets.
2. **Groups C+D carry essentially the entire AUC gain** (0.55→0.70 ML-1M,
   0.51→0.62 Amazon). These are model-independent popularity/propensity priors —
   the same signal Grouped MP exploits. The headline calibrator AUC is mostly
   this, not score transfer.
3. **Group A's only measurable effect is a small lift at the very top of the
   ranking** (full vs no-transfer: ndcg@1 +0.02, mrr +0.01 on both datasets),
   not on aggregate AUC. Real but marginal.
4. **Where they differ:** on dense ML-1M the full calibrator ties Grouped MP
   (0.707 vs 0.709); on sparse Amazon it clearly beats it (0.621 vs 0.565) —
   stacking several priors > a single category-popularity signal when data is
   sparse. Still not a transfer win.

**Interpretation / caveat.** Both base BPRs generalise weakly on *warm* items
(val AUC 0.63–0.66; Amazon train AUC was 99.96% — memorised, not learned).
Post-hoc transfer is upper-bounded by base-model quality: you cannot transfer a
signal the model does not itself have. So the honest claim is **conditional**:
*in the weak-base-model regime tested, score transfer adds nothing over a
popularity prior.* Whether group A ever helps is an open question gated on a
base model that genuinely generalises on warm items (see §8). NB on Amazon the
positive rate is ~87%, which inflates ranking metrics (ndcg/mrr ~0.9); trust AUC
there, not the ranking columns.

---

## 7. Decisions Log

- **2026-05-25.** Adopted Variant A: post-hoc model-agnostic library
  positioning. Excluded integrated-training methods (DropoutNet, MWUF, etc.)
  from library scope — out-of-scope, refer users to ColdRec for those.
- **2026-05-25.** TF-IDF features included for Amazon Toys preprocessing. To be
  re-examined: hypothesis that they may dilute category signal in cosine
  similarity. Need ablation.
- **2026-06-05.** Adopted explicit benchmark methodology principles (§4):
  fix-once preprocessing, features = benchmark spec (identical for all methods),
  preprocessing sensitivity is reported as ablation not tuned away, a simple
  baseline beating us is a finding not a bug. Reframes the similarity
  category-vs-text question: it is an **ablation axis**, NOT a knob to turn until
  our method beats Grouped MP. Motivated by the recsys reproducibility critique
  (Dacrema 2019, Rendle).
- **2026-06-06.** Calibrated Aggregation reworked. The original feature set
  (only summary stats of the neighbour model scores) was degenerate: all inputs
  monotone in one scalar, so the meta-learner collapsed to ≈ the plain mean
  (Score-LogReg ≈ Score-Avg). Keeping a method that provably duplicates a simpler
  one under a fancier name was rejected. Reframed the calibrator honestly as
  **stacking with meta-features** (Bao 2009) and pre-registered a heterogeneous
  feature set (§3.3: model-transfer / neighbourhood-trust / user-prior /
  popularity-prior) **before** running, frozen across datasets. This is
  "implement the known method correctly", explicitly NOT "tune to beat Grouped
  MP". Groups C/D may narrow the gap to Grouped MP; permitted only under the
  pre-registration discipline of §4.
- **2026-06-06.** Ran the calibrator feature-group ablation on both datasets
  (§6.1). Result: the model-transfer signal (group A) contributes ≈0 to AUC on
  both ML-1M and Amazon — the full calibrator's gain is essentially all from the
  model-independent priors (groups C/D). Group A gives only a marginal top-of-list
  lift. This is the project's first concrete cross-dataset finding and reframes
  the contribution as a **conditional map** ("transfer is gated on base-model
  generalisation"), not a champion method. Recorded as-is, no tuning. Next probe:
  does group A revive under a base model that generalises well on warm items?
- **2026-05-25.** BPR hyperparameter tuning stopped. Ceiling at val AUC 0.64
  for Amazon Toys deemed dataset-intrinsic. No further per-dataset tuning
  until proper grid search protocol is set up.

---

## 8. Open Questions

- Does our method beat Grouped MP on *any* (dataset, model) combination?
  Partial answer (§6.1): the full calibrator already beats Grouped MP on sparse
  Amazon (0.621 vs 0.565) — but via priors, not transfer. Open on dense data.
- **Does the model-transfer signal (group A) ever contribute to AUC?** §6.1 shows
  ≈0 with weak base models (val AUC 0.63–0.66). Hypothesis: transfer is gated on
  base-model warm generalisation. Probe with a base model that genuinely
  generalises on warm items and re-run the ablation — does group A revive?
- Does TF-IDF in similarity features help or hurt? Run an ablation on Amazon.
- Are post-hoc methods (LinMap especially) categorically different from KNN
  aggregation, or do they converge to similar performance?
- Should we expose a hybrid mode (combine model transfer with content baseline)
  as an explicit library feature, or stay pure?
- How does runtime scale with cold-item count? Currently 7-15 min on Amazon
  (full 70K). For 1M cold items would need different similarity index.

---

## 9. References

- ColdRec (2024-2025). YuanchenBei. https://github.com/YuanchenBei/ColdRec
- Awesome-Cold-Start-Recommendation. https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation
- Volkovs et al. (2017). *DropoutNet: Addressing Cold Start in Recommender Systems.* NeurIPS.
- Zhu et al. (2020). *Recommendation for New Users and New Items via Randomized Training and Mixture-of-Experts Transformation* (Heater). SIGIR.
- Zhu et al. (2021). *Learning to Warm Up Cold Item Embeddings with Meta Scaling and Shifting Networks* (MWUF). SIGIR.
- Gantner et al. (2010). *Learning Attribute-to-Feature Mappings for Cold-Start Recommendations.*
- Schein et al. (2002). *Methods and Metrics for Cold-Start Recommendations.* SIGIR.
- Wolpert (1992). *Stacked Generalization.* Neural Networks.
- Sill, Takács, Mackey & Lin (2009). *Feature-Weighted Linear Stacking.* arXiv:0911.0460 (Netflix Prize). — most precise reference for the calibrator.
- Bao, Bergman & Thompson (2009). *Stacking Recommendation Engines with Additional Meta-Features (STREAM).* RecSys.
- Lee et al. (2019). *MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation.* KDD.
- Liu et al. (2026). *A Review of Deep Learning and Large Language Models for Cold Start Problem in Recommender Systems.* WIREs DMKD. https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70068
- Hu, Koren, Volinsky (2008). *Collaborative Filtering for Implicit Feedback Datasets* (ALS).
- He et al. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* SIGIR.
- Ferrari Dacrema, Cremonesi, Jannach (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys.
- Rendle, Krichene, Zhang, Anderson (2020). *Neural Collaborative Filtering vs. Matrix Factorization Revisited.* RecSys. (baseline-tuning / evaluation rigour)
