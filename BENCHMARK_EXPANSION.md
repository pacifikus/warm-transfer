# Benchmark Expansion Plan — Donors × Datasets

**Status:** proposal (for review with Kristina / Nikita)
**Author:** Egor · **Date:** 2026-06-06

## Motivation

The product is **not** a champion cold-start method — it is a *library + an honest,
broad benchmark* that reveals **when** content-based score transfer helps and when it
does not (see [`README.md`](README.md), [`RESEARCH.md`](RESEARCH.md)).

A cross-check against the parallel `coldscore`/`coldbench` benchmark (3 datasets ×
3 donors × 17 methods) makes two things clear:

1. **"Beats the baseline" is dataset-dependent and partly an artifact of a weak
   baseline.** On `goodbooks`, `grouped_most_popular` scores AUC **0.4618 — below
   random** and `grouped_most_popular_pers` only 0.606; so `linmap` "winning" there
   (0.77) is as much *weak popularity* as *strong transfer*. On `ml-1m`, where
   popularity is strong (`grouped_pers` 0.72), the same `linmap` barely edges it on
   ALS and **loses** on BPR/CatBoost. Nothing wins universally.
2. **The central scientific claim is "donor quality gates transfer"** — but the
   donor axis is the thin one: of 3 donors, **two (ALS, BPR) are the same family**
   (matrix factorization); only CatBoost (GBDT) is different. You cannot substantiate
   a claim *about donor quality* with a donor axis this narrow.

This plan expands the benchmark along the **two axes that actually matter for the
claim** — donors and datasets — with a design that makes the claim *provable* rather
than merely illustrated.

## The headline experiment (design, not just "more runs")

To turn 9 scattered cells into a thesis, the donor axis needs **two sub-axes**:

### Sub-axis A — controlled quality gradient *within* a family (the experiment)

Hold the family fixed, sweep one knob, and read transfer-gain as a function of donor
quality:

- ALS with `factors ∈ {8, 16, 64, 128}` (and/or BPR `iterations ∈ {10, 50, 200}`).
- **Donor quality (X-axis)** = the donor's *own* warm-set ranking quality (per-user
  AUC on held-out warm interactions) — a measured scalar, not a guess.
- **Transfer gain (Y-axis)** = AUC(transfer method) − AUC(grouped_most_popular_pers).
- Output: a **curve** "transfer gain vs donor quality" per dataset. This *is* the
  scientific statement; today it does not exist anywhere.

### Sub-axis B — family breadth (generalization)

Show the curve is not an MF artifact by adding genuinely different donor classes.

## Donor matrix (target ≈ 6–7)

| donor | class | status | role / why | impl cost |
|---|---|---|---|---|
| ALS (factors swept) | MF (ALS) | exists | **controlled quality gradient** (sub-axis A) | low |
| BPR | MF (pairwise) | exists | second MF objective | low |
| CatBoost | GBDT | exists | non-latent donor | done |
| popularity-donor | trivial | **add** | **lower anchor** of the quality curve (a deliberately bad donor) | trivial |
| EASE / SLIM | item–item linear | **add** | different geometry (item-based, not factor) | low (closed-form) |
| LightFM (WARP) | MF-hybrid | **add** | hybrid objective; also the project's own base model | medium |
| two-tower / NeuMF | neural CF | **add (stretch)** | deep family; torch already a dependency | medium |
| SASRec | sequential | later (milestone) | sequential donor; only on timestamped datasets (ml-1m, kion) | high |

Adding a donor = one new adapter against the existing adapter contract
(`adapters/base.py` in coldbench) — the architecture already supports it.

## Dataset matrix (target = 5)

Chosen to differ along axes that affect transfer — sparsity, popularity strength,
content modality, feedback type — not just to raise the count.

| dataset | domain | sparsity | popularity strength | content | role |
|---|---|---|---|---|---|
| ml-1m | movies | dense | **strong** (grouped_pers 0.72) | categorical (genres) | "popularity wins" reference |
| goodbooks | books | medium | **weak** (grouped_mp < random) | tags / authors | "transfer wins" reference |
| kion | video | implicit | medium | categorical + text | implicit-feedback case |
| **Amazon Toys & Games** | e-commerce | **sparse, long-tail** | medium | text / metadata | sparsity + exposes ranking-metric inflation (high positive-rate) |
| **MIND (news)** | news | **true item cold-start** | weak | **rich text** | new modality; content methods should win big — or a strong negative result |

Notes:
- **Amazon Toys** has a prep script already (`datasets/prepare_amazon_toys.py`). Its
  high positive-rate is *useful*: it inflates NDCG/MRR and so motivates reporting
  **per-user AUC** as the honest discriminative metric (ties into the eval-rigor
  layer below).
- **MIND** is the canonical cold-start dataset (news items are genuinely cold every
  day) with rich text content. Pre-registered hypothesis: content transfer
  (`linmap_emb`, `dropoutnet`) beats popularity by a wide margin here; if it does
  **not**, that is itself a strong negative finding. Cost: heavy; needs a text →
  feature step (TF-IDF / title embeddings) as a separate task.

## Eval-rigor layer (what makes the numbers a result)

A 105-row results table is raw material, not a finding. The layer this plan adds on
top of the runs:

- **Pre-registration** — fix hypotheses and the metric *before* looking (no iterating
  features while watching the metric).
- **Per-user vs global AUC** — report per-user AUC as the headline; global AUC is
  inflated by cross-user popularity (on ml-1m: 0.69 vs 0.81).
- **Popularity-bucket decomposition** — head vs tail recall (the existing
  `analysis.py` is unused for the headline); shows *where* a method wins.
- **Two known degeneracies to verify/fix** surfaced by the cross-check:
  - `debiased_knn` drops **below random** on several cells (kion/als 0.492,
    kion/catboost 0.499, ml-1m/catboost 0.495) — a real defect.
  - `logreg_calib` posts high AUC with **near-zero recall** (kion/catboost AUC 0.763,
    recall@10 0.007) — degenerate features (monotone in one scalar).

## Phased rollout

1. **Phase 1 (cheap, high yield):** ALS quality gradient + popularity-donor + EASE on
   the 3 existing datasets → first version of the "donor-quality gates transfer"
   curve. First presentable result.
2. **Phase 2:** + Amazon Toys + LightFM.
3. **Phase 3:** + MIND (with text preprocessing) + two-tower.

End state: **≈7 donors × 5 datasets**, with the central axis (donor quality)
*controlled* rather than incidental.
