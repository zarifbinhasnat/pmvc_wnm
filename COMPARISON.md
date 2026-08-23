# PMVC-WNM (this repo) vs. Rafat-Pantho/ML-Banglish-co-training-prototype

In-depth comparison of the two PMVC-WNM implementations, with an empirical
head-to-head benchmark run on identical data, splits, and label budget.

- **This repo** (`pmvc_wnm`): modular `src/` implementation with the BNPC
  numeric phonetic encoder, true two-pool co-training, and a token-level
  spelling-noise injection model.
- **External repo** ([Rafat-Pantho/ML-Banglish-co-training-prototype](https://github.com/Rafat-Pantho/ML-Banglish-co-training-prototype)):
  single-notebook prototype for CSE 4622 (IUT) with a coarse Soundex-style
  phonetic encoder, shared-pool co-training, and class-level pseudo-label
  reweighting.

Both implement the same proposal (two-view co-training + weakly supervised
noise model for Banglish), but they diverge substantially in the phonetic
encoding, the co-training algorithm, what "noise model" means, and code
structure.

---

## 1. Head-to-head benchmark (empirical)

Protocol — identical for both pipelines (`benchmarks/benchmark.py`):

- Dataset: BnSentMix (`huggingface bensentMix.csv` shipped in the external
  repo), 20,015 rows, 4 sentiment classes (imbalanced: 5.3K/6.2K/6.6K/1.9K).
- Stratified 3,000-row sample, seed 42 (the external repo's prototype scale).
- 15% stratified held-out test set; **300 labeled seed samples**; remaining
  ~2,250 rows as the unlabeled pool.
- External pipeline reproduced verbatim from its notebook (same vectorizers,
  classifiers, thresholds); local pipeline run through `src/` as-is with
  `seed_size=300`.

| Model | Macro-F1 | Accuracy | Train (s) |
|---|---|---|---|
| EXT baseline: SVM char n-gram (seed only) | 0.5548 | 0.6178 | 0.4 |
| EXT baseline: RF phonetic (seed only) | 0.3920 | 0.5022 | 0.5 |
| EXT: standard co-training | 0.5740 | 0.6067 | 14.2 |
| EXT: PMVC-WNM (reliability reweight) | 0.5726 | 0.6178 | 16.6 |
| LOCAL baseline: LinearSVC char n-gram (seed) | 0.5465 | 0.6044 | 0.1 |
| **LOCAL baseline: LR + BNPC phonetic (seed)** | **0.5916** | **0.6244** | 0.6 |
| LOCAL: co-training (no noise) | 0.5774 | 0.6156 | 12.2 |
| LOCAL: PMVC-WNM full (noise injection) | 0.5784 | 0.6156 | 14.4 |

The external README reports 0.552/0.553/0.556 for its own runs — this
reproduction matches within noise, so their published numbers are honest and
reproducible.

### Multi-seed stability (seeds 42, 7, 2026 — mean macro-F1)

| View / model | External | Local |
|---|---|---|
| View A char n-gram, seed only | 0.527 | 0.547 |
| View B phonetic, seed only | 0.398 (RF) / 0.397 (LR) | **0.585 (BNPC + LR)** |
| View-B complementarity (B right where A wrong) | 11.6% | 9.3% |

### Noise robustness (30% token corruption of the test set)

Corruption uses this repo's `SPELLING_VARIANTS` swaps from `src/evaluate.py`
(phonetically plausible substitutions — note this mildly favors the local
design, disclosed accordingly). Script: `benchmarks/noise_robustness.py`.

| Model | Clean F1 | Noisy F1 | Drop |
|---|---|---|---|
| EXT PMVC-WNM (predicts with f_A char view) | 0.570 | 0.564 | −1.0% |
| LOCAL BNPC LR (seed only) | 0.592 | 0.568 | −4.0% |
| LOCAL PMVC-WNM full | 0.560 | 0.568 | **+1.4% (no degradation)** |

### What the numbers say

1. **The phonetic encoding is the single biggest differentiator: +0.19
   macro-F1.** The local BNPC View B (0.585 mean) roughly *matches or beats
   char n-grams as a standalone model*, while the external Soundex-style view
   (0.40) is far below its own char-n-gram baseline. Swapping the external
   view's RF for LogisticRegression changes nothing (0.397), so the gap is
   the *encoding*, not the classifier. The external encoder maps every word
   to a 5-consonant-class skeleton and deletes all non-initial vowels
   ("khacchi" → "GJ"), destroying word identity — exactly the failure mode
   documented in `src/view_b_phonetic.py` (condition C2). BNPC keeps the full
   phoneme sequence in 16 classes plus word-boundary markers and dual TF-IDF
   blocks (phoneme n-grams + canonical word codes), so variants collapse
   without losing identity.

2. **The full pipelines are statistically tied at prototype scale
   (~0.57–0.58)**, and both barely beat the plain char-n-gram baseline. Each
   repo's "noise model" contributes ≈±0.001 F1 on clean text at this scale.

3. **The local repo's best single model beats both full pipelines.** BNPC+LR
   at 0.592 outperforms local full PMVC-WNM (0.578) and external PMVC-WNM
   (0.573). The co-training loop currently *subtracts* value from View B:
   the ensemble rule in `PMVCTrainer.predict` falls back to the weaker View A
   whenever View B's confidence is below 0.75, so View A dominates final
   predictions. This is the clearest improvement target for this repo.

4. **Noise injection does earn its keep under distribution shift**: the local
   full model is the only one that does not degrade on the corrupted test
   set, consistent with training-time injection of learned spelling variants.

5. **Run-to-run variance is ~±0.02 F1** at this sample size; single-run
   differences smaller than that (e.g. ext co-training 0.574 vs local 0.578)
   should not be read as wins.

---

## 2. Algorithmic comparison

| Aspect | This repo | External repo |
|---|---|---|
| **View B encoding** | BNPC: 16 two-digit phoneme classes, full sequence kept, word boundaries (`00`), dual TF-IDF (phoneme 1–3 grams + word-code 1–2 grams), granularity ablation flags (`merge_ch_s`, `merge_a_o`) | 8 letter classes (G/J/T/P/S/M/R/V), consecutive dedupe, **all non-initial vowels dropped** → consonant skeleton; char_wb 2–3 gram TF-IDF |
| **Co-training** | Canonical Blum–Mitchell: separate pools `L_A`/`L_B`, f_A teaches f_B with f_A's predictions and vice versa, threshold 0.75, top-K=100/view | Single shared pool; both views trained on the same growing set; pseudo-label taken from whichever view is more confident, threshold 0.6, 40/view/iter. Closer to two-feature self-training than co-training (view independence is not exploited for cross-teaching) |
| **Noise model** | Token-level transition matrix P(spelling \| phoneme code) learned from inter-view disagreement texts; injects spelling variants into 20% of f_A's training tokens from iteration 3 | Per-class reliability score R(c) = agree/(agree+disagree); down-weights pseudo-labels to [0.5, 1.0] in one final f_A retrain. With 4 classes this is 4 scalars — class-level label weighting, not a spelling-noise model. Their own README lists the transition-matrix approach (what this repo implements) as future work |
| **Final prediction** | Confidence-weighted ensemble of f_A and f_B | **f_A alone** — f_B and the phonetic view are discarded at inference |
| **Classifiers** | `CalibratedClassifierCV(LinearSVC)` + `LogisticRegression(class_weight='balanced')` | `SVC(kernel='linear', probability=True)` + `LogisticRegression` (no class weighting despite 3.5:1 imbalance) |
| **Scalability** | LinearSVC is O(n) in samples — full 20K/80K runs feasible in minutes | Kernel SVC with Platt scaling is superquadratic; their README warns 10–60 min at 20K rows. Will not reach the 80K target practically |
| **Validation tooling** | `variant_merge_report`, `vocab_compression`, `collision_report`, `complementarity` — encoder quality measurable before training | Single 3-word sanity print (`khacchi/khacci/khachi`) |

---

## 3. Structural comparison

### This repo

**Strengths**
- Modular `src/` (preprocess / view A / view B / co-training / noise / eval)
  with single-responsibility files; components importable and unit-testable.
- Design rationale documented where it matters (the `view_b_phonetic.py`
  docstring explains *why* the encoding is shaped the way it is, with
  falsifiable targets: ≥80% variant merge, 1.3–2.5× vocab compression).
- Evaluation harness: 5-model ablation, learning curve, noise-robustness
  test, disagreement-convergence plot.
- No data committed; `data/README.md` documents the expected Kaggle source.

**Weaknesses**
- No root `README.md` — a visitor lands on nothing (CLAUDE.md is agent
  config, not a project readme).
- No committed results; before this benchmark all performance claims were
  unvalidated.
- No tests, no CI.
- **Latent bug in `src/cotraining.py:166-170`**: the pool-removal guards are
  swapped. `new_for_A` is created under the `high_conf_B` branch (and
  `new_for_B` under `high_conf_A`), but the removal expression guards
  `new_for_A` with `len(high_conf_A) > 0`. If exactly one view has no
  high-confidence picks in iteration 1 this raises `NameError`; in later
  iterations it silently drops a stale batch from the unlabeled pool. It
  happens to work when both views always have high-confidence picks (as in
  the benchmark runs).
- `run_ablation`/`run_learning_curve` take `train_df.index[:seed_size]` as
  the seed — not stratified, unlike the trainer's own seeding.

### External repo

**Strengths**
- Excellent README: quick-start (local + Colab), parameter documentation,
  honest prototype-scale results *with caveats*, dataset appendix, technical
  report PDFs, proposal-to-implementation mapping table.
- Fully reproducible out of the box: dataset ships with the repo, one
  notebook, synthetic-data fallback if the CSV is missing.
- Optional BiLSTM deep baseline behind a flag.

**Weaknesses**
- Everything lives in one 26-cell notebook with global state — nothing is
  importable, reusable, or testable; the co-training function reaches
  module-level constants.
- ~74 MB of CSVs committed, ~72 MB of which the notebook never reads
  (including a 66 MB dataset). The README's file table claims these are
  git-ignored; they are in fact tracked — the `.gitignore` was added after
  they were committed.
- The trained f_B and the entire phonetic view are unused at inference.
- No class weighting despite 3.5:1 imbalance; macro-F1 suffers on the small
  class.

---

## 4. Verdict

- **Method & performance**: this repo's design is empirically stronger where
  it matters — the BNPC phonetic view carries real signal (+0.19 F1 over the
  external encoding, and the best single model overall), the noise model is
  a genuine text-level noise model rather than 4 class weights, and the
  stack scales linearly to the full 80K-row target. The external repo's own
  results table quietly shows its method failing to beat its baseline
  (0.552 vs 0.553); this benchmark confirms that and localizes the cause to
  the vowel-dropping consonant-skeleton encoding.
- **Structure & presentation**: the external repo is far ahead on
  communication — README, reproducibility, honest reporting, report PDFs.
  This repo is far ahead on engineering — modularity, testability, design
  documentation, validation tooling.
- **Biggest open gap in this repo**: the co-training ensemble currently
  underperforms its own View B baseline (0.578 vs 0.592) because
  `predict()` lets the weaker View A dominate; plus the swapped-guard bug in
  `cotraining.py` and the missing root README.

Reproduce with:

```bash
git clone https://github.com/Rafat-Pantho/ML-Banglish-co-training-prototype ../ML-Banglish-co-training-prototype
python benchmarks/benchmark.py ../ML-Banglish-co-training-prototype
python benchmarks/noise_robustness.py ../ML-Banglish-co-training-prototype
```
