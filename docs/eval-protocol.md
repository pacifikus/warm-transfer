# Eval protocol (anti-leakage)

The main scientific risk of the project is leakage in cold-start evaluation. The protocol below + the
splitter invariant tests guard against it.

## Pseudo-cold split (how we do it)

1. **Split by items, not by interactions.** We randomly (stratified) pick a subset of warm items,
   declare them pseudo-cold and **completely remove all of their interactions** from:
   - the donor's train,
   - the Grouped MP and popularity computation,
   - neighbor construction (similarity).
2. **Stratification** of pseudo-cold by category and popularity buckets — otherwise the evaluation is
   biased toward the head/tail of the distribution.
3. **Cold content features** are computed only from their static data (no target leak from their own
   interactions). TF-IDF / normalizations / popularity priors — **only over the train corpus**.
4. **Hyperparameter tuning** (k in KNN, λ in Ridge, …) — on a separate **cold-validation** fold, not on
   the test.
5. Optionally — a **temporal split** (items that appeared after a moment T): fairer than random
   item-holdout, closer to production. Fits naturally on KION (timestamps are available).

## Training supervised meta-methods (a protocol nuance)

Meta-methods (`stacking`, `stacking_plus`, `logreg_calib`) train the meta-model on the **val-cold fold**
(items outside the donor's train and outside the test). Their features (linmap signal, affinity,
popularity) are computed leak-free, and the labels come from val-cold (not from the test).

The current runner implementation trains the meta-model on the same set of users as eval (`test_users`,
selected as users with interactions on test-cold items). This is **not a leak of the test outcome**
(val-labels ≠ test-labels; the meta-model does not see test-cold features/labels) — confirmed by an
independent review (codex). In recommendations the users are always shared between training and
evaluation, so this does not inflate the result. Strictly speaking, what leaks into fit is *the user's
membership in the eval set*; for a maximally purist protocol the meta-model should be trained on a set of
users independent from the test — noted as a direction for further work (it would require donor scores for
an additional set of users).

## Splitter invariant (checked by tests)

- `set(cold_items) ∩ set(train.item_id) == ∅` and the same for `val`.
- Interactions of cold items are present **only** in `test`.
- The donor physically does not see cold items (a separate test: "no cold item in any donor training
  input").
- Determinism under a fixed seed.

## Metrics

- **Ranking metrics — the main ones** for the comparison against Grouped MP: Recall@k, Precision@k,
  MAP@k, NDCG@k, MRR, for k ∈ {1, 5, 10}.
- **AUC + RelaImpr** — additionally. RelaImpr = (AUC_model − 0.5) / (AUC_base − 0.5) − 1.
- **Calibration is orthogonal to ranking AND to AUC**: Platt/isotonic are monotone transformations, so
  they change NEITHER ranking NOR AUC (both are rank-based). They improve only logloss/Brier. We do not
  count calibration as a "transfer method".
- We report the **spread over several seeds** (cold evaluation is noisy) and **over popularity buckets**
  of cold items.

## Metric conventions (we fix OUR OWN, not relying on third-party libs)

- Tie-breaking in top-k is deterministic and documented.
- Users with no relevant items in the test are excluded from averaging (documented).
- k > number of candidates — graceful degradation (no crash).
- Spot-checking against a reference (sklearn) on simple cases, but the contract is ours.
